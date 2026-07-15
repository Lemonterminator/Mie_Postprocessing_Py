"""Unified predictor layer: (checkpoint kind, path) -> canonical points table.

One :func:`build_condition_features` implementation replaces the three
near-duplicate group-by-condition feature loops in the legacy scripts, and one
``Predictor`` interface hides the four different loading conventions:

======================  ======================================================
kind                    loader / predict path
======================  ======================================================
``mlp``                 ``efc.load_run_artifacts`` + batched torch forward
                        (single / family_head / residual_fh / residual_film)
``single_svgp``         ``run_gp_baseline.load_gp_artifacts`` +
                        ``predict_physical``
``residual_svgp``       ``residual_multitask_svgp.load_residual_svgp_artifacts``
                        + ``predict_residual_physical`` (family routed)
``ha`` / ``ns``         analytic correlation baselines with fitted params
                        (``evaluate_ha_ns_fixed_tables.predict_baseline``)
======================  ======================================================

Every predictor returns the same canonical points schema:
meta columns + ``time_ms, pen_true_mm, pen_pred_mm, pen_std_mm, resid_mm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from MLP.eval_pipeline.common import PROJECT_ROOT, resolve_path  # noqa: F401  (sys.path side effect)
from MLP.eval_pipeline.datasets import (
    META_COLS_FOR_POINTS,
    EvalSetSpec,
    feature_group_col,
)

from MLP.MLP_training.engineered_feature_common import (  # noqa: E402
    build_dataset_registry,
    build_feature_matrix_np,
    family_id_from_name,
    load_run_artifacts,
)
from MLP.MLP_training.efc.feature_engineering import infer_feature_family  # noqa: E402
from MLP.MLP_training.efc.objectives import split_mu_logvar  # noqa: E402
from MLP.MLP_training.train_stage3_distillation_plus_raw_series import (  # noqa: E402
    build_teacher_raw_dict,
)

MODEL_KINDS = ("mlp", "single_svgp", "residual_svgp", "ha", "ns")


@dataclass(frozen=True)
class ModelSpec:
    """One evaluatable model: label + kind + checkpoint path (+ bookkeeping)."""

    label: str
    kind: str
    path: Path
    family: str = ""
    seed: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"Unknown model kind {self.kind!r}; expected one of {MODEL_KINDS}")


def choose_device(requested: str = "auto") -> torch.device:
    if requested in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _raw_from_meta_row(row: pd.Series) -> dict[str, Any]:
    raw = build_teacher_raw_dict(row)
    experiment_name = row.get("experiment_name", row.get("dataset_key", None))
    if experiment_name is not None and not pd.isna(experiment_name):
        raw["dataset_key"] = str(experiment_name)
        raw["experiment_name"] = str(experiment_name)
    return raw


def build_condition_features(
    df: pd.DataFrame,
    spec: EvalSetSpec,
    *,
    feature_columns: list[str],
    scaler_state: Mapping[str, Any],
    registry: Mapping[str, Any],
    time_feature: str = "time_norm_0_5ms",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, int]:
    """Group rows by condition, rebuild model features per group.

    Returns ``(features, a_scale, truth, meta_df, family_id, skipped_groups)``
    with all arrays row-aligned. ``family_id`` is derived per group from the
    experiment name (Nozzle0 -> 0, else 1) for residual-SVGP routing; MLP-kind
    models that consume ``family_id``/``nozzle_id`` as feature channels get
    them baked into the matrix by ``build_feature_matrix_np`` instead.
    """
    time_col, truth_col = spec.time_col, spec.truth_col
    sort_cols = [c for c in ("experiment_name", "condition_id", "traj_key", time_col, "frame_pos")
                 if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    group_col = feature_group_col(df, spec)
    group_iter = df.groupby(group_col, sort=False, dropna=False) if group_col else [(None, df)]

    feature_blocks: list[np.ndarray] = []
    a_scale_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    family_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []
    skipped = 0

    for _, group in group_iter:
        group = group.sort_values([c for c in (time_col, "frame_pos") if c in group.columns])
        group = group.reset_index(drop=True)
        time_vals = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float)
        truth_vals = pd.to_numeric(group[truth_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(time_vals) & np.isfinite(truth_vals)
        if not np.any(valid):
            skipped += 1
            continue
        group = group.loc[valid].reset_index(drop=True)
        time_ms = group[time_col].to_numpy(dtype=np.float32)
        truth = group[truth_col].to_numpy(dtype=np.float32)
        raw = _raw_from_meta_row(group.iloc[0])
        try:
            features_np, a_scale_np, _ = build_feature_matrix_np(
                raw, time_ms, scaler_state, feature_columns, registry, time_feature=time_feature,
            )
        except Exception as exc:  # skip unbuildable groups, keep going
            skipped += 1
            print(f"[warn] feature build failed for {spec.name} group={group_col}: {exc}")
            continue

        family_value = family_id_from_name(str(raw.get("experiment_name") or raw.get("dataset_key") or ""))
        meta_cols = [c for c in META_COLS_FOR_POINTS if c in group.columns]
        meta = group.loc[:, meta_cols].copy()
        meta["time_ms"] = time_ms
        feature_blocks.append(features_np)
        a_scale_blocks.append(np.asarray(a_scale_np).reshape(-1))
        truth_blocks.append(truth)
        family_blocks.append(np.full(len(group), int(family_value), dtype=np.int64))
        meta_blocks.append(meta)

    if not feature_blocks:
        raise RuntimeError(f"No usable points found for eval set {spec.name!r}.")

    return (
        np.vstack(feature_blocks).astype(np.float32),
        np.concatenate(a_scale_blocks).astype(np.float32),
        np.concatenate(truth_blocks).astype(np.float32),
        pd.concat(meta_blocks, ignore_index=True),
        np.concatenate(family_blocks),
        skipped,
    )


def _assemble_points(meta_df: pd.DataFrame, truth: np.ndarray,
                     mu: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    points = meta_df.copy()
    points["pen_true_mm"] = truth
    points["pen_pred_mm"] = mu
    points["pen_std_mm"] = std
    points["resid_mm"] = points["pen_pred_mm"] - points["pen_true_mm"]
    return points


class MlpPredictor:
    """Stage-3 MLP family: single / family_head / residual_fh / residual_film."""

    def __init__(self, spec: ModelSpec, *, device: str = "auto") -> None:
        self.spec = spec
        self.artifacts = load_run_artifacts(resolve_path(spec.path), device=choose_device(device))
        self.registry = build_dataset_registry()

    def predict(self, df: pd.DataFrame, eval_spec: EvalSetSpec, *,
                batch_points: int = 262_144) -> tuple[pd.DataFrame, int]:
        cfg = self.artifacts.train_config
        features, a_scale, truth, meta_df, _family, skipped = build_condition_features(
            df, eval_spec,
            feature_columns=list(cfg["feature_columns"]),
            scaler_state=self.artifacts.scaler_state,
            registry=self.registry,
            time_feature=str(cfg.get("time_feature", "time_norm_0_5ms")),
        )
        mu, std = self._forward(features, a_scale, batch_points=batch_points)
        return _assemble_points(meta_df, truth, mu, std), skipped

    def _forward(self, features: np.ndarray, a_scale: np.ndarray, *,
                 batch_points: int) -> tuple[np.ndarray, np.ndarray]:
        # Canonical physical-scale reconstruction (parity with the legacy
        # inference_rmse_on_series._predict_points): mu = A_scale * mu_hat,
        # std = A_scale * exp(0.5 * clamp(logvar, [-20, 20])), floored at
        # the run's std_clamp_min.
        model = self.artifacts.model
        cfg = self.artifacts.train_config
        device = next(model.parameters()).device
        family = infer_feature_family(cfg["feature_columns"])
        std_floor = float(cfg.get("std_clamp_min", 0.0))
        mu_chunks: list[np.ndarray] = []
        std_chunks: list[np.ndarray] = []
        n = len(features)
        step = max(int(batch_points), 1)
        with torch.no_grad():
            for start in range(0, n, step):
                stop = min(start + step, n)
                feat_t = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
                scale_t = torch.as_tensor(a_scale[start:stop, None], dtype=torch.float32, device=device)
                mu_hat, log_var_hat = split_mu_logvar(model(feat_t))
                log_var_hat = torch.clamp(log_var_hat, min=-20.0, max=20.0)
                if family == "engineered_v2":
                    mu = scale_t * mu_hat
                    std = scale_t * torch.exp(0.5 * log_var_hat)
                else:
                    mu = mu_hat
                    std = torch.exp(0.5 * log_var_hat)
                std = torch.clamp(std, min=std_floor)
                mu_chunks.append(mu.detach().cpu().numpy().reshape(-1))
                std_chunks.append(std.detach().cpu().numpy().reshape(-1))
        return np.concatenate(mu_chunks), np.concatenate(std_chunks)


def _resolve_svgp_checkpoint(path: Path, *, seed: int | None) -> Path:
    """Accept a bare model.pt, a run dir, or a run dir with per_seed layout."""
    path = resolve_path(path)
    if path.is_file():
        return path
    candidates = []
    if seed is not None:
        candidates.append(path / "per_seed" / f"seed_{seed}" / "model.pt")
        candidates.append(path / "per_seed" / f"seed_{seed:02d}" / "model.pt")
    candidates.append(path / "model.pt")
    per_seed_root = path / "per_seed"
    if per_seed_root.is_dir():
        candidates.extend(sorted(per_seed_root.glob("seed_*/model.pt")))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No SVGP model.pt found under {path}")


class SingleSvgpPredictor:
    """Single-output SVGP baseline (mean SVGP + optional log-variance SVGP)."""

    def __init__(self, spec: ModelSpec, *, device: str = "auto") -> None:
        from MLP.GP_training.run_gp_baseline import load_gp_artifacts

        self.spec = spec
        checkpoint = _resolve_svgp_checkpoint(Path(spec.path), seed=spec.seed)
        self.artifacts = load_gp_artifacts(checkpoint, device=choose_device(device))
        self.registry = build_dataset_registry()

    def predict(self, df: pd.DataFrame, eval_spec: EvalSetSpec, *,
                batch_points: int = 65_536) -> tuple[pd.DataFrame, int]:
        from MLP.GP_training.run_gp_baseline import predict_physical

        cfg = self.artifacts.config
        features, a_scale, truth, meta_df, _family, skipped = build_condition_features(
            df, eval_spec,
            feature_columns=list(cfg["feature_columns"]),
            scaler_state=self.artifacts.scaler_state,
            registry=self.registry,
            time_feature=str(cfg.get("time_feature", "time_norm_0_5ms")),
        )
        mu, std, _, _ = predict_physical(
            self.artifacts, features, a_scale,
            batch_points=int(batch_points),
            include_mean_posterior_var=bool(cfg.get("include_mean_posterior_var", False)),
        )
        return _assemble_points(meta_df, truth, mu, std), skipped


class ResidualSvgpPredictor:
    """Residual multi-task SVGP: shared GP + per-family delta GPs."""

    def __init__(self, spec: ModelSpec, *, device: str = "auto") -> None:
        from MLP.GP_training.residual_multitask_svgp import load_residual_svgp_artifacts

        self.spec = spec
        checkpoint = _resolve_svgp_checkpoint(Path(spec.path), seed=spec.seed)
        self.artifacts = load_residual_svgp_artifacts(checkpoint, device=choose_device(device))
        self.registry = build_dataset_registry()

    def predict(self, df: pd.DataFrame, eval_spec: EvalSetSpec, *,
                batch_points: int = 65_536) -> tuple[pd.DataFrame, int]:
        from MLP.GP_training.residual_multitask_svgp import predict_residual_physical

        cfg = self.artifacts.config
        features, a_scale, truth, meta_df, family_id, skipped = build_condition_features(
            df, eval_spec,
            feature_columns=list(cfg["feature_columns"]),
            scaler_state=self.artifacts.scaler_state,
            registry=self.registry,
            time_feature=str(cfg.get("time_feature", "time_norm_0_5ms")),
        )
        mu, std, _, _ = predict_residual_physical(
            self.artifacts, features, a_scale, family_id,
            batch_points=int(batch_points),
            include_mean_posterior_var=bool(cfg.get("include_mean_posterior_var", False)),
        )
        points = _assemble_points(meta_df, truth, mu, std)
        points["family_id"] = family_id
        return points, skipped


class CorrelationBaselinePredictor:
    """Hiroyasu–Arai / Naber–Siebers analytic correlations with fitted params."""

    def __init__(self, spec: ModelSpec, *, device: str = "auto") -> None:
        del device  # CPU-only analytic model
        self.spec = spec
        self.run_dir = resolve_path(spec.path)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"{spec.kind} baseline run dir not found: {self.run_dir}")

    def predict(self, df: pd.DataFrame, eval_spec: EvalSetSpec, *,
                batch_points: int = 0) -> tuple[pd.DataFrame, int]:
        del batch_points
        from MLP.baseline.comparison_reports.evaluate_ha_ns_fixed_tables import (
            BaselineSpec,
            predict_baseline,
        )

        result = predict_baseline(
            BaselineSpec(kind=self.spec.kind, group=self.spec.kind,
                         label=self.spec.label, run_dir=self.run_dir),
            df, eval_spec.time_col, eval_spec.truth_col,
        )
        points = result.points.rename(columns={
            "truth_mm": "pen_true_mm",
            "pred_mu_mm": "pen_pred_mm",
            "pred_std_mm": "pen_std_mm",
        })
        if len(points) == len(df):
            source = df.reset_index(drop=True)
            for col in META_COLS_FOR_POINTS:
                if col in source.columns and col not in points.columns:
                    points[col] = source[col].to_numpy()
        points["resid_mm"] = points["pen_pred_mm"] - points["pen_true_mm"]
        return points.reset_index(drop=True), 0


_PREDICTOR_BY_KIND = {
    "mlp": MlpPredictor,
    "single_svgp": SingleSvgpPredictor,
    "residual_svgp": ResidualSvgpPredictor,
    "ha": CorrelationBaselinePredictor,
    "ns": CorrelationBaselinePredictor,
}


def load_predictor(spec: ModelSpec, *, device: str = "auto"):
    """Instantiate the predictor matching ``spec.kind`` (loads the checkpoint)."""
    return _PREDICTOR_BY_KIND[spec.kind](spec, device=device)
