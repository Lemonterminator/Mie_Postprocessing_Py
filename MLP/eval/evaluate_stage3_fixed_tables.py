"""Evaluate Stage-3 MLP/SVGP checkpoints on fixed diagnostic tables.

The standard full-clean series evaluator uses the wide CDF exports.  This script
keeps the checkpoint inference path but swaps the evaluation tables to:

* cdf_right_censoring_points/cdf_points_uncensored.csv
* p50_q1_oracle/p50_q1_observed_fit_points.csv
* p50_q1_oracle/p50_q1_predictions.csv

It is intended for post-training comparisons; it does not train or mutate runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.MLP_training.engineered_feature_common import (
    build_dataset_registry,
    build_feature_matrix_np,
    infer_feature_family,
    load_run_artifacts,
    split_mu_logvar,
)
from MLP.MLP_training.run_gp_baseline import (
    load_gp_artifacts,
    predict_physical as predict_gp_physical,
)


UNCENSORED_CSV = PROJECT_ROOT / "MLP" / "synthetic_data" / "cdf_right_censoring_points" / "cdf_points_uncensored.csv"
P50_Q1_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data" / "p50_q1_oracle"
P50_Q1_FIT_METRICS_CSV = P50_Q1_ROOT / "p50_q1_fit_metrics.csv"
P50_Q1_OBSERVED_CSV = P50_Q1_ROOT / "p50_q1_observed_fit_points.csv"
P50_Q1_GRID_CSV = P50_Q1_ROOT / "p50_q1_predictions.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports"


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    group: str
    label: str
    path: Path


@dataclass
class PredictionResult:
    points: pd.DataFrame
    overall: dict[str, Any]
    per_condition: pd.DataFrame
    per_experiment: pd.DataFrame


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def parse_model_spec(text: str, *, kind: str) -> ModelSpec:
    """Parse GROUP,LABEL,PATH, allowing commas in PATH after the second comma."""
    parts = str(text).split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{kind} spec must be GROUP,LABEL,PATH; got {text!r}"
        )
    group, label, path = parts
    return ModelSpec(kind=kind, group=group.strip(), label=label.strip(), path=Path(path.strip()))


def choose_device(requested: str) -> torch.device:
    if str(requested).lower() in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if str(requested).lower() == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.", flush=True)
    return torch.device("cpu")


def raw_from_row(row: pd.Series) -> dict[str, Any]:
    raw = {
        "umbrella_angle_deg": float(row["umbrella_angle_deg"]),
        "plumes": float(row["plumes"]),
        "diameter_mm": float(row["diameter_mm"]),
        "injection_duration_us": float(row["injection_duration_us"]),
        "injection_pressure_bar": float(row["injection_pressure_bar"]),
        "chamber_pressure_bar": float(row["chamber_pressure_bar"]),
        "control_backpressure_bar": float(row["control_backpressure_bar"]),
    }
    if "experiment_name" in row and pd.notna(row["experiment_name"]):
        raw["dataset_key"] = str(row["experiment_name"])
    return raw


def build_point_features(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    scaler_state: Mapping[str, Any],
    registry: Mapping[str, Any],
    time_col: str,
    truth_col: str,
    time_feature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    feature_blocks: list[np.ndarray] = []
    scale_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []

    group_cols = ["condition_id"] if "condition_id" in df.columns else [df.index.name or "index"]
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols == ["condition_id"]:
        grouped = df.groupby("condition_id", sort=False, dropna=False)
    else:
        grouped = [(None, df)]

    for _, grp in grouped:
        grp = grp.reset_index(drop=True)
        if grp.empty:
            continue
        row0 = grp.iloc[0]
        time = grp[time_col].to_numpy(dtype=np.float32)
        truth = grp[truth_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(time) & np.isfinite(truth)
        if not np.any(valid):
            continue
        valid_grp = grp.loc[valid].copy()
        try:
            features_np, a_scale_np, _ = build_feature_matrix_np(
                raw_from_row(row0),
                time[valid],
                scaler_state,
                list(feature_columns),
                registry,
                time_feature=time_feature,
            )
        except Exception as exc:
            condition = row0.get("condition_id", "unknown")
            print(f"[warn] feature build failed for condition {condition}: {exc}", flush=True)
            continue

        feature_blocks.append(features_np.astype(np.float32))
        scale_blocks.append(a_scale_np.reshape(-1).astype(np.float32))
        truth_blocks.append(truth[valid].astype(np.float32))
        meta_cols = [
            c for c in [
                "experiment_name",
                "condition_id",
                "condition_key",
                "time_ms",
                "time_bin",
                "is_observed_window",
                "n_points_in_p50_bin",
                "n_traces_in_bin",
                "plumes",
                "diameter_mm",
                "umbrella_angle_deg",
                "chamber_pressure_bar",
                "injection_duration_us",
                "injection_pressure_bar",
                "control_backpressure_bar",
            ]
            if c in valid_grp.columns
        ]
        meta = valid_grp.loc[:, meta_cols].copy()
        if "time_ms" not in meta.columns:
            meta["time_ms"] = time[valid]
        meta_blocks.append(meta.reset_index(drop=True))

    if not feature_blocks:
        raise RuntimeError("No evaluable points after feature construction.")

    return (
        np.vstack(feature_blocks).astype(np.float32),
        np.concatenate(scale_blocks).astype(np.float32),
        np.concatenate(truth_blocks).astype(np.float32),
        pd.concat(meta_blocks, ignore_index=True),
    )


def predict_mlp(
    run_dir: Path,
    *,
    df: pd.DataFrame,
    time_col: str,
    truth_col: str,
    device: torch.device,
    batch_points: int,
) -> pd.DataFrame:
    artifacts = load_run_artifacts(run_dir, device=str(device))
    registry = build_dataset_registry()
    feature_columns = list(artifacts.train_config["feature_columns"])
    time_feature = str(artifacts.train_config.get("time_feature", "time_norm_0_5ms"))
    features, a_scale, truth, meta = build_point_features(
        df,
        feature_columns=feature_columns,
        scaler_state=artifacts.scaler_state,
        registry=registry,
        time_col=time_col,
        truth_col=truth_col,
        time_feature=time_feature,
    )

    family = infer_feature_family(feature_columns)
    mu_chunks: list[np.ndarray] = []
    std_chunks: list[np.ndarray] = []
    model_device = next(artifacts.model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(features), int(batch_points)):
            stop = min(start + int(batch_points), len(features))
            feat_t = torch.as_tensor(features[start:stop], dtype=torch.float32, device=model_device)
            scale_t = torch.as_tensor(a_scale[start:stop, None], dtype=torch.float32, device=model_device)
            out = artifacts.model(feat_t)
            mu_hat, log_var_hat = split_mu_logvar(out)
            log_var_hat = torch.clamp(log_var_hat, min=-20.0, max=20.0)
            if family == "engineered_v2":
                mu = scale_t * mu_hat
                std = scale_t * torch.exp(0.5 * log_var_hat)
            else:
                mu = mu_hat
                std = torch.exp(0.5 * log_var_hat)
            std_floor = float(artifacts.train_config.get("std_clamp_min", 0.0))
            std = torch.clamp(std, min=std_floor)
            mu_chunks.append(mu.detach().cpu().numpy().reshape(-1))
            std_chunks.append(std.detach().cpu().numpy().reshape(-1))

    out_df = meta.copy()
    out_df["truth_mm"] = truth
    out_df["pred_mu_mm"] = np.concatenate(mu_chunks)
    out_df["pred_std_mm"] = np.concatenate(std_chunks)
    out_df["resid_mm"] = out_df["pred_mu_mm"] - out_df["truth_mm"]
    return out_df


def predict_gp(
    checkpoint: Path,
    *,
    df: pd.DataFrame,
    time_col: str,
    truth_col: str,
    device: torch.device,
    batch_points: int,
) -> pd.DataFrame:
    artifacts = load_gp_artifacts(checkpoint, device=device)
    registry = build_dataset_registry()
    feature_columns = list(artifacts.config["feature_columns"])
    time_feature = str(artifacts.config.get("time_feature", "time_norm_0_5ms"))
    features, a_scale, truth, meta = build_point_features(
        df,
        feature_columns=feature_columns,
        scaler_state=artifacts.scaler_state,
        registry=registry,
        time_col=time_col,
        truth_col=truth_col,
        time_feature=time_feature,
    )
    pred, std, _, _ = predict_gp_physical(
        artifacts,
        features,
        a_scale,
        batch_points=int(batch_points),
        include_mean_posterior_var=bool(artifacts.config.get("include_mean_posterior_var", False)),
    )
    out_df = meta.copy()
    out_df["truth_mm"] = truth
    out_df["pred_mu_mm"] = pred
    out_df["pred_std_mm"] = std
    out_df["resid_mm"] = out_df["pred_mu_mm"] - out_df["truth_mm"]
    return out_df


def metrics_from_points(points: pd.DataFrame) -> dict[str, Any]:
    truth = points["truth_mm"].to_numpy(dtype=float)
    pred = points["pred_mu_mm"].to_numpy(dtype=float)
    std = points["pred_std_mm"].to_numpy(dtype=float)
    resid = pred - truth
    abs_err = np.abs(resid)
    finite = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(std)
    if not np.all(finite):
        truth = truth[finite]
        pred = pred[finite]
        std = std[finite]
        resid = resid[finite]
        abs_err = abs_err[finite]
    std_safe = np.maximum(std, 1e-12)
    if truth.size == 0:
        return {
            "n_points": 0,
            "rmse_mm": float("nan"),
            "mae_mm": float("nan"),
            "bias_mm": float("nan"),
            "median_abs_err_mm": float("nan"),
            "p90_abs_err_mm": float("nan"),
            "p95_abs_err_mm": float("nan"),
            "coverage_1sigma": float("nan"),
            "coverage_2sigma": float("nan"),
            "mean_pred_std_mm": float("nan"),
        }
    return {
        "n_points": int(truth.size),
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
        "mae_mm": float(np.mean(abs_err)),
        "bias_mm": float(np.mean(resid)),
        "median_abs_err_mm": float(np.median(abs_err)),
        "p90_abs_err_mm": float(np.quantile(abs_err, 0.90)),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)),
        "mean_pred_std_mm": float(np.mean(std)),
    }


def grouped_metrics(points: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in points.groupby(list(group_cols), sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(metrics_from_points(grp))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_prediction(points: pd.DataFrame) -> PredictionResult:
    per_condition_cols = [c for c in ["condition_id", "condition_key", "experiment_name"] if c in points.columns]
    per_experiment_cols = [c for c in ["experiment_name"] if c in points.columns]
    return PredictionResult(
        points=points,
        overall=metrics_from_points(points),
        per_condition=grouped_metrics(points, per_condition_cols) if per_condition_cols else pd.DataFrame(),
        per_experiment=grouped_metrics(points, per_experiment_cols) if per_experiment_cols else pd.DataFrame(),
    )


def evaluate_model_on_table(
    spec: ModelSpec,
    *,
    df: pd.DataFrame,
    time_col: str,
    truth_col: str,
    device: torch.device,
    batch_points: int,
) -> PredictionResult:
    if spec.kind == "mlp":
        points = predict_mlp(spec.path, df=df, time_col=time_col, truth_col=truth_col, device=device, batch_points=batch_points)
    elif spec.kind == "gp":
        points = predict_gp(spec.path, df=df, time_col=time_col, truth_col=truth_col, device=device, batch_points=batch_points)
    else:
        raise ValueError(f"Unknown model kind: {spec.kind}")
    return summarise_prediction(points)


def row_from_metrics(
    *,
    eval_set: str,
    spec: ModelSpec,
    metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "eval_set": eval_set,
        "model_group": spec.group,
        "model_label": spec.label,
        "model_kind": spec.kind,
        "model_path": str(spec.path),
    }
    row.update(metrics)
    if extra:
        row.update(extra)
    return row


def aggregate_group_rows(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "n_points",
        "n_conditions",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "median_abs_err_mm",
        "p90_abs_err_mm",
        "p95_abs_err_mm",
        "coverage_1sigma",
        "coverage_2sigma",
        "mean_pred_std_mm",
        "condition_rmse_mean_mm",
        "condition_rmse_median_mm",
        "condition_mae_mean_mm",
        "condition_bias_mean_mm",
    ]
    rows: list[dict[str, Any]] = []
    for (eval_set, group), sub in per_run.groupby(["eval_set", "model_group"], sort=False):
        out: dict[str, Any] = {
            "eval_set": eval_set,
            "model_group": group,
            "n_runs": int(len(sub)),
        }
        rmse = pd.to_numeric(sub["rmse_mm"], errors="coerce")
        if len(sub):
            best_idx = rmse.idxmin()
            out["best_label_by_rmse"] = sub.loc[best_idx, "model_label"]
            out["best_rmse_mm"] = float(sub.loc[best_idx, "rmse_mm"])
        for col in metric_cols:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            out[f"{col}_mean"] = float(vals.mean())
            out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def add_condition_summary(metrics: dict[str, Any], per_condition: pd.DataFrame) -> dict[str, Any]:
    out = dict(metrics)
    if not per_condition.empty:
        out["n_conditions"] = int(len(per_condition))
        out["condition_rmse_mean_mm"] = float(per_condition["rmse_mm"].mean())
        out["condition_rmse_median_mm"] = float(per_condition["rmse_mm"].median())
        out["condition_mae_mean_mm"] = float(per_condition["mae_mm"].mean())
        out["condition_bias_mean_mm"] = float(per_condition["bias_mm"].mean())
    return out


def q1_oracle_reference_rows(fit_metrics: pd.DataFrame, observed: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    obs_points = observed.rename(columns={"penetration_mm": "truth_mm", "q1_fit_mm": "pred_mu_mm"}).copy()
    obs_points["pred_std_mm"] = 0.0
    obs_points["resid_mm"] = obs_points["pred_mu_mm"] - obs_points["truth_mm"]
    obs_overall = metrics_from_points(obs_points)
    rows.append(
        {
            "eval_set": "p50_observed",
            "model_group": "q1_oracle",
            "model_label": "quarter_only_v1_p50_oracle",
            "model_kind": "oracle",
            "model_path": str(P50_Q1_FIT_METRICS_CSV),
            **obs_overall,
            "n_conditions": int(len(fit_metrics)),
            "condition_rmse_mean_mm": float(fit_metrics["rmse"].mean()),
            "condition_rmse_median_mm": float(fit_metrics["rmse"].median()),
            "condition_mae_mean_mm": float(fit_metrics["mae"].mean()),
            "condition_bias_mean_mm": float(fit_metrics["bias"].mean()),
        }
    )
    return rows


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return ""
    return f"{v:.{digits}f}"


def write_markdown_tables(out_dir: Path, per_run: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Stage-3 Fixed-Table Evaluation\n")
    lines.append("Evaluation tables:")
    lines.append(f"- CDF uncensored: `{UNCENSORED_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- P50 observed/Q1 fit: `{P50_Q1_OBSERVED_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Q1 0-5 ms grid: `{P50_Q1_GRID_CSV.relative_to(PROJECT_ROOT)}`\n")

    display_sets = [
        ("cdf_uncensored", "CDF Uncensored Point-Level"),
        ("p50_observed", "P50 Observed Points"),
        ("q1_grid_all", "Q1 Oracle Grid, 0-5 ms"),
        ("q1_grid_extrapolated", "Q1 Oracle Grid, Extrapolated Region"),
    ]
    for eval_set, title in display_sets:
        sub = group_summary.loc[group_summary["eval_set"] == eval_set].copy()
        if sub.empty:
            continue
        lines.append(f"## {title}\n")
        lines.append("| model_group | n_runs | rmse | mae | bias | p95 | cov1 | cov2 | cond_rmse_mean | best |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, row in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("model_group", "")),
                        str(int(row.get("n_runs", 0))),
                        format_value(row.get("rmse_mm_mean")),
                        format_value(row.get("mae_mm_mean")),
                        format_value(row.get("bias_mm_mean")),
                        format_value(row.get("p95_abs_err_mm_mean")),
                        format_value(row.get("coverage_1sigma_mean")),
                        format_value(row.get("coverage_2sigma_mean")),
                        format_value(row.get("condition_rmse_mean_mm_mean")),
                        f"{row.get('best_label_by_rmse', '')} ({format_value(row.get('best_rmse_mm'))})",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Per-Run Headline\n")
    keep = [
        "eval_set",
        "model_group",
        "model_label",
        "n_points",
        "n_conditions",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "p95_abs_err_mm",
        "coverage_1sigma",
        "coverage_2sigma",
        "condition_rmse_mean_mm",
    ]
    shown = per_run.loc[:, [c for c in keep if c in per_run.columns]].copy()
    try:
        lines.append(shown.to_markdown(index=False))
    except ImportError:
        lines.append("```")
        lines.append(shown.to_string(index=False))
        lines.append("```")
    lines.append("")
    (out_dir / "main_table.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_all(
    *,
    specs: Sequence[ModelSpec],
    out_dir: Path,
    device: torch.device,
    batch_points: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "model_specs.json").write_text(
        json.dumps([_jsonable(spec.__dict__) for spec in specs], indent=2),
        encoding="utf-8",
    )

    uncensored = _read_csv(UNCENSORED_CSV)
    fit_metrics = _read_csv(P50_Q1_FIT_METRICS_CSV)
    p50_observed = _read_csv(P50_Q1_OBSERVED_CSV)
    q1_grid = _read_csv(P50_Q1_GRID_CSV)

    per_run_rows: list[dict[str, Any]] = []
    per_condition_frames: list[pd.DataFrame] = []
    per_experiment_frames: list[pd.DataFrame] = []

    per_run_rows.extend(q1_oracle_reference_rows(fit_metrics, p50_observed))

    eval_tables = [
        ("cdf_uncensored", uncensored, "time_ms", "penetration_mm"),
        ("p50_observed", p50_observed, "time_ms", "penetration_mm"),
        ("q1_grid_all", q1_grid, "time_ms", "q1_fit_mm"),
    ]

    for spec in specs:
        print(f"[model] {spec.group}/{spec.label} ({spec.kind})", flush=True)
        for eval_set, df, time_col, truth_col in eval_tables:
            print(f"  [eval] {eval_set}: n_rows={len(df)}", flush=True)
            result = evaluate_model_on_table(
                spec,
                df=df,
                time_col=time_col,
                truth_col=truth_col,
                device=device,
                batch_points=batch_points,
            )
            metrics = add_condition_summary(result.overall, result.per_condition)
            per_run_rows.append(row_from_metrics(eval_set=eval_set, spec=spec, metrics=metrics))

            if not result.per_condition.empty:
                pc = result.per_condition.copy()
                pc.insert(0, "eval_set", eval_set)
                pc.insert(1, "model_group", spec.group)
                pc.insert(2, "model_label", spec.label)
                pc.insert(3, "model_kind", spec.kind)
                per_condition_frames.append(pc)
            if not result.per_experiment.empty:
                pe = result.per_experiment.copy()
                pe.insert(0, "eval_set", eval_set)
                pe.insert(1, "model_group", spec.group)
                pe.insert(2, "model_label", spec.label)
                pe.insert(3, "model_kind", spec.kind)
                per_experiment_frames.append(pe)

            if eval_set == "q1_grid_all" and "is_observed_window" in result.points.columns:
                for flag, name in [(True, "q1_grid_observed_window"), (False, "q1_grid_extrapolated")]:
                    sub_points = result.points.loc[result.points["is_observed_window"].astype(bool) == flag]
                    sub_result = summarise_prediction(sub_points)
                    sub_metrics = add_condition_summary(sub_result.overall, sub_result.per_condition)
                    per_run_rows.append(row_from_metrics(eval_set=name, spec=spec, metrics=sub_metrics))

    per_run = pd.DataFrame(per_run_rows)
    group_summary = aggregate_group_rows(per_run)
    per_run.to_csv(out_dir / "per_run_metrics.csv", index=False)
    group_summary.to_csv(out_dir / "group_summary.csv", index=False)
    if per_condition_frames:
        pd.concat(per_condition_frames, ignore_index=True).to_csv(out_dir / "per_condition_metrics.csv", index=False)
    if per_experiment_frames:
        pd.concat(per_experiment_frames, ignore_index=True).to_csv(out_dir / "per_experiment_metrics.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "batch_points": int(batch_points),
        "tables": {
            "cdf_uncensored": {"path": str(UNCENSORED_CSV), "n_rows": int(len(uncensored))},
            "p50_q1_fit_metrics": {"path": str(P50_Q1_FIT_METRICS_CSV), "n_rows": int(len(fit_metrics))},
            "p50_observed": {"path": str(P50_Q1_OBSERVED_CSV), "n_rows": int(len(p50_observed))},
            "q1_grid": {"path": str(P50_Q1_GRID_CSV), "n_rows": int(len(q1_grid))},
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    write_markdown_tables(out_dir, per_run, group_summary)
    print(f"[done] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-run", action="append", default=[], help="GROUP,LABEL,PATH to a Stage-3 MLP run dir.")
    parser.add_argument("--gp-checkpoint", action="append", default=[], help="GROUP,LABEL,PATH to a GP model.pt checkpoint.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points", type=int, default=262144)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs: list[ModelSpec] = []
    specs.extend(parse_model_spec(item, kind="mlp") for item in args.mlp_run)
    specs.extend(parse_model_spec(item, kind="gp") for item in args.gp_checkpoint)
    if not specs:
        raise SystemExit("Provide at least one --mlp-run or --gp-checkpoint.")
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = DEFAULT_OUTPUT_ROOT / f"stage3_fixed_table_eval_{datetime.now():%Y%m%d_%H%M%S}"
    evaluate_all(
        specs=specs,
        out_dir=out_dir,
        device=choose_device(args.device),
        batch_points=int(args.batch_points),
    )


if __name__ == "__main__":
    main()
