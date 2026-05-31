"""Evaluate a Stage-3 refinement run on canonical point-table eval sets.

The legacy ``inference_rmse_on_series`` evaluator expands
``cdf/series_wide_clean`` and therefore keeps the late-time right-censored
tail.  This module evaluates the same model on point tables produced by the
fit workflow:

* ``cdf_points_uncensored.csv``: conservative CDF points after density/FOV
  right-censoring removal.
* ``p50_q1_observed_fit_points.csv``: per-condition observed P50 bins.
* ``p50_q1_predictions.csv``: per-condition q1 oracle grid, including
  extrapolated regions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.eval.calibration_diagnostics import compute_pit, crps_gaussian, ece, reliability_curve
from MLP.eval.inference_rmse_on_series import _finite_metrics, _predict_points
from MLP.MLP_training.engineered_feature_common import (
    build_dataset_registry,
    build_feature_matrix_np,
    load_run_artifacts,
)
from MLP.MLP_training.train_stage3_distillation_plus_raw_series import build_teacher_raw_dict


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "eval"
DEFAULT_CDF_UNCENSORED = (
    PROJECT_ROOT
    / "MLP"
    / "synthetic_data"
    / "cdf_right_censoring_points"
    / "cdf_points_uncensored.csv"
)
DEFAULT_P50_OBSERVED = (
    PROJECT_ROOT
    / "MLP"
    / "synthetic_data"
    / "p50_q1_oracle"
    / "p50_q1_observed_fit_points.csv"
)
DEFAULT_Q1_GRID = (
    PROJECT_ROOT
    / "MLP"
    / "synthetic_data"
    / "p50_q1_oracle"
    / "p50_q1_predictions.csv"
)

DEFAULT_POINT_EVAL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "cdf_uncensored",
        "path": DEFAULT_CDF_UNCENSORED,
        "time_col": "time_ms",
        "truth_col": "penetration_mm",
        "feature_group_col": "condition_id",
        "description": "CDF points after density/FOV right-censoring removal",
    },
    {
        "name": "p50_observed",
        "path": DEFAULT_P50_OBSERVED,
        "time_col": "time_ms",
        "truth_col": "penetration_mm",
        "feature_group_col": "condition_id",
        "description": "Observed per-condition P50 points used by the q1 oracle fit",
    },
    {
        "name": "q1_grid_all",
        "path": DEFAULT_Q1_GRID,
        "time_col": "time_ms",
        "truth_col": "q1_fit_mm",
        "feature_group_col": "condition_id",
        "description": "Q1 oracle prediction grid over 0-5 ms",
    },
)
DEFAULT_PRIMARY_EVAL_SET = "cdf_uncensored"
PROBABILISTIC_EVAL_SETS = {"cdf_uncensored", "p50_observed"}
PROBABILITY_LEVELS = np.linspace(0.025, 0.975, 19)
PIT_HIST_BINS = np.linspace(0.0, 1.0, 21)

META_COLS_FOR_POINTS = [
    "condition_id",
    "condition_key",
    "traj_key",
    "experiment_name",
    "test_name",
    "file_path",
    "file_name",
    "file_stem",
    "plume_idx",
    "frame_pos",
    "time_bin",
    "time_bin_start_ms",
    "time_bin_end_ms",
    "is_observed_window",
    "n_points_in_bin",
    "n_points_in_p50_bin",
    "n_traces_in_bin",
]


def point_eval_specs_for_synthetic_root(synthetic_root: Path | str) -> tuple[dict[str, Any], ...]:
    """Build canonical point-table specs under a synthetic-data root."""
    root = _resolve_path(synthetic_root)
    return (
        {
            "name": "cdf_uncensored",
            "path": root / "cdf_right_censoring_points" / "cdf_points_uncensored.csv",
            "time_col": "time_ms",
            "truth_col": "penetration_mm",
            "feature_group_col": "condition_id",
            "description": "CDF points after density/FOV right-censoring removal",
        },
        {
            "name": "p50_observed",
            "path": root / "p50_q1_oracle" / "p50_q1_observed_fit_points.csv",
            "time_col": "time_ms",
            "truth_col": "penetration_mm",
            "feature_group_col": "condition_id",
            "description": "Observed per-condition P50 points used by the q1 oracle fit",
        },
        {
            "name": "q1_grid_all",
            "path": root / "p50_q1_oracle" / "p50_q1_predictions.csv",
            "time_col": "time_ms",
            "truth_col": "q1_fit_mm",
            "feature_group_col": "condition_id",
            "description": "Q1 oracle prediction grid over 0-5 ms",
        },
    )


def _resolve_path(path: Path | str) -> Path:
    """Resolve repo-relative paths plus Windows/WSL cross-path forms."""
    text = str(path)
    win_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if win_match and os.name != "nt":
        drive = win_match.group(1).lower()
        rest = win_match.group(2).replace("\\", "/")
        return Path("/mnt") / drive / rest
    wsl_match = re.match(r"^/mnt/([A-Za-z])/(.*)$", text)
    if wsl_match and os.name == "nt":
        drive = wsl_match.group(1).upper()
        rest = wsl_match.group(2).replace("/", "\\")
        return Path(f"{drive}:\\{rest}")
    if win_match:
        return Path(text)
    if wsl_match:
        return Path(text)
    p = Path(text).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _gaussian_nll(truth: np.ndarray, pred: np.ndarray, std: np.ndarray) -> float:
    if len(truth) == 0:
        return float("nan")
    std_safe = np.maximum(std.astype(float), 1e-6)
    z = (truth.astype(float) - pred.astype(float)) / std_safe
    return float(np.mean(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(std_safe) + z * z)))


def _finite_prediction_arrays(points_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = points_df["pen_true_mm"].to_numpy(dtype=float)
    pred = points_df["pen_pred_mm"].to_numpy(dtype=float)
    std = points_df["pen_std_mm"].to_numpy(dtype=float)
    finite = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(std) & (std > 0.0)
    return truth[finite], pred[finite], std[finite], finite


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if weights is None:
        kept = values[finite]
        return float(np.mean(kept)) if kept.size else float("nan")
    weights = np.asarray(weights, dtype=float)
    finite &= np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(np.average(values[finite], weights=weights[finite]))


def _weighted_reliability_curve(pit: np.ndarray, levels: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return reliability_curve(pit, levels)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(pit) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.full(len(levels), np.nan, dtype=float)
    pit_valid = pit[valid]
    weights_valid = weights[valid]
    total = float(np.sum(weights_valid))
    return np.asarray(
        [float(np.sum(weights_valid[pit_valid <= alpha]) / total) for alpha in levels],
        dtype=float,
    )


def _probabilistic_target_type(eval_set: str) -> str | None:
    if eval_set not in PROBABILISTIC_EVAL_SETS:
        return None
    if eval_set == "cdf_uncensored":
        return "raw_uncensored_observation"
    if eval_set == "p50_observed":
        return "p50_aggregate"
    return None


def _p50_weight_summary(points_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ("n_points_in_p50_bin", "n_traces_in_bin"):
        if col not in points_df.columns:
            continue
        values = pd.to_numeric(points_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        out[f"mean_{col}"] = float(np.mean(values))
        out[f"median_{col}"] = float(np.median(values))
        out[f"min_{col}"] = float(np.min(values))
        out[f"max_{col}"] = float(np.max(values))
    return out


def _ensure_time_bin_column(points_df: pd.DataFrame) -> pd.DataFrame:
    if "time_bin" in points_df.columns or "time_ms" not in points_df.columns:
        return points_df
    out = points_df.copy()
    out["time_bin"] = pd.to_numeric(out["time_ms"], errors="coerce")
    return out


def _probabilistic_summary_row(
    *,
    label: str,
    target_type: str,
    truth: np.ndarray,
    pred: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pit = np.clip(compute_pit(truth, pred, std), 1e-9, 1.0 - 1e-9)
    rel = _weighted_reliability_curve(pit, PROBABILITY_LEVELS, weights)
    crps = crps_gaussian(truth, pred, std)
    abs_err = np.abs(truth - pred)
    std_safe = np.maximum(std, 1e-12)
    row = {
        "label": label,
        "target_type": target_type,
        "n_points": int(len(truth)),
        "ece": ece(pit, PROBABILITY_LEVELS) if weights is None else (
            float(np.nanmean(np.abs(rel - PROBABILITY_LEVELS))) if len(rel) else float("nan")
        ),
        "crps_mean": _weighted_mean(crps, weights),
        "crps_std": float(np.nanstd(crps, ddof=1)) if len(crps) > 1 and weights is None else float("nan"),
        "sharpness_mm": _weighted_mean(std, weights),
        "coverage_1sigma": _weighted_mean((abs_err <= std_safe).astype(float), weights),
        "coverage_2sigma": _weighted_mean((abs_err <= 2.0 * std_safe).astype(float), weights),
        "mean_abs_z": _weighted_mean(abs_err / std_safe, weights),
    }
    if weights is not None:
        valid_weights = np.asarray(weights, dtype=float)
        valid_weights = valid_weights[np.isfinite(valid_weights) & (valid_weights > 0.0)]
        row["weight_sum"] = float(np.sum(valid_weights)) if valid_weights.size else 0.0
        row["weight_mean"] = float(np.mean(valid_weights)) if valid_weights.size else float("nan")
    return row, pit, rel, crps


def _write_probabilistic_diagnostics(points_df: pd.DataFrame, *, eval_set: str, out_dir: Path) -> dict[str, Any] | None:
    target_type = _probabilistic_target_type(eval_set)
    if target_type is None:
        return None

    truth, pred, std, finite = _finite_prediction_arrays(points_df)
    if truth.size == 0:
        return None

    rows: list[dict[str, Any]] = []
    reliability_payload: dict[str, Any] = {
        "probability_level": PROBABILITY_LEVELS,
        "nominal_lower_tail_probability": PROBABILITY_LEVELS,
    }
    hist_payload: dict[str, Any] = {
        "pit_bin_left": PIT_HIST_BINS[:-1],
        "pit_bin_right": PIT_HIST_BINS[1:],
    }

    row, pit, rel, crps = _probabilistic_summary_row(
        label="unweighted",
        target_type=target_type,
        truth=truth,
        pred=pred,
        std=std,
    )
    row.update(_p50_weight_summary(points_df) if eval_set == "p50_observed" else {})
    rows.append(row)
    reliability_payload["empirical_lower_tail_fraction_unweighted"] = rel
    reliability_payload["abs_calibration_error_unweighted"] = np.abs(rel - PROBABILITY_LEVELS)
    counts, _ = np.histogram(pit, bins=PIT_HIST_BINS)
    hist_payload["count_unweighted"] = counts
    hist_payload["fraction_unweighted"] = counts / max(int(np.sum(counts)), 1)

    crps_summary = {
        "target_type": target_type,
        "n_points": int(len(crps)),
        "crps_mean": float(np.mean(crps)),
        "crps_std": float(np.std(crps, ddof=1)) if len(crps) > 1 else 0.0,
        "crps_median": float(np.median(crps)),
        "crps_p90": float(np.quantile(crps, 0.90)),
        "crps_p95": float(np.quantile(crps, 0.95)),
    }

    if eval_set == "p50_observed" and "n_points_in_p50_bin" in points_df.columns:
        weights_all = pd.to_numeric(points_df["n_points_in_p50_bin"], errors="coerce").to_numpy(dtype=float)
        weights = weights_all[finite]
        w_row, _w_pit, w_rel, _w_crps = _probabilistic_summary_row(
            label="weighted_by_n_points_in_p50_bin",
            target_type=target_type,
            truth=truth,
            pred=pred,
            std=std,
            weights=weights,
        )
        w_row.update(_p50_weight_summary(points_df))
        rows.append(w_row)
        reliability_payload["empirical_lower_tail_fraction_weighted_by_n_points_in_p50_bin"] = w_rel
        reliability_payload["abs_calibration_error_weighted_by_n_points_in_p50_bin"] = np.abs(w_rel - PROBABILITY_LEVELS)

        valid_w = np.isfinite(weights) & (weights > 0.0)
        weighted_counts, _ = np.histogram(pit[valid_w], bins=PIT_HIST_BINS, weights=weights[valid_w])
        hist_payload["weighted_count_by_n_points_in_p50_bin"] = weighted_counts
        hist_payload["weighted_fraction_by_n_points_in_p50_bin"] = weighted_counts / max(float(np.sum(weighted_counts)), 1.0)
        crps_summary["crps_mean_weighted_by_n_points_in_p50_bin"] = _weighted_mean(crps, weights)

    summary = {
        "eval_set": eval_set,
        "target_type": target_type,
        "probability_levels": [float(x) for x in PROBABILITY_LEVELS],
        "pit_hist_bins": [float(x) for x in PIT_HIST_BINS],
        "recommended_for_thesis_calibration": bool(eval_set == "cdf_uncensored"),
        "notes": (
            "Primary raw-observation calibration set after right-censoring truncation."
            if eval_set == "cdf_uncensored"
            else "Auxiliary calibration on aggregate P50 targets; interpret as condition/time-bin center-trend calibration."
        ),
        "rows": rows,
        "outputs": {
            "probabilistic_summary": str(out_dir / "probabilistic_summary.json"),
            "probabilistic_summary_csv": str(out_dir / "probabilistic_summary.csv"),
            "reliability_curve": str(out_dir / "reliability_curve.csv"),
            "pit_histogram": str(out_dir / "pit_histogram.csv"),
            "crps_summary": str(out_dir / "crps_summary.csv"),
        },
    }

    pd.DataFrame(rows).to_csv(out_dir / "probabilistic_summary.csv", index=False)
    pd.DataFrame(reliability_payload).to_csv(out_dir / "reliability_curve.csv", index=False)
    pd.DataFrame(hist_payload).to_csv(out_dir / "pit_histogram.csv", index=False)
    pd.DataFrame([crps_summary]).to_csv(out_dir / "crps_summary.csv", index=False)
    (out_dir / "probabilistic_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _probabilistic_headline_fields(probabilistic: dict[str, Any] | None) -> dict[str, Any]:
    if not probabilistic:
        return {}
    out: dict[str, Any] = {
        "probabilistic_target_type": probabilistic.get("target_type"),
        "recommended_for_thesis_calibration": probabilistic.get("recommended_for_thesis_calibration"),
    }
    for row in probabilistic.get("rows", []):
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", ""))
        if label == "unweighted":
            for key in ("ece", "crps_mean", "sharpness_mm", "coverage_1sigma", "coverage_2sigma"):
                if key in row:
                    out[f"prob_{key}"] = row[key]
        elif label == "weighted_by_n_points_in_p50_bin":
            for key in ("ece", "crps_mean", "sharpness_mm", "coverage_1sigma", "coverage_2sigma"):
                if key in row:
                    out[f"prob_{key}_weighted_by_n_points_in_p50_bin"] = row[key]
    return out


def _metrics_from_points(points_df: pd.DataFrame, *, rel_err_floor_mm: float) -> dict[str, float | int]:
    truth = points_df["pen_true_mm"].to_numpy(dtype=float)
    pred = points_df["pen_pred_mm"].to_numpy(dtype=float)
    std = points_df["pen_std_mm"].to_numpy(dtype=float)
    metrics = _finite_metrics(truth, pred, std)
    if len(points_df) == 0:
        return {
            **metrics,
            "n_conditions": 0,
            "n_trajectories": 0,
            "mean_rel_err": float("nan"),
            "median_rel_err": float("nan"),
            "truth_min_mm": float("nan"),
            "truth_max_mm": float("nan"),
            "truth_range_mm": float("nan"),
            "nrmse_range": float("nan"),
            "nll_physical": float("nan"),
        }

    resid = points_df["resid_mm"].to_numpy(dtype=float)
    rel_abs = np.abs(resid) / np.maximum(np.abs(truth), float(rel_err_floor_mm))
    truth_range = float(np.max(truth) - np.min(truth))
    return {
        **metrics,
        "n_conditions": int(points_df["condition_id"].nunique()) if "condition_id" in points_df else 0,
        "n_trajectories": int(points_df["traj_key"].nunique()) if "traj_key" in points_df else 0,
        "mean_rel_err": float(np.mean(rel_abs)),
        "median_rel_err": float(np.median(rel_abs)),
        "truth_min_mm": float(np.min(truth)),
        "truth_max_mm": float(np.max(truth)),
        "truth_range_mm": truth_range,
        "nrmse_range": float(metrics["rmse_mm"] / truth_range) if truth_range > 0 else float("nan"),
        "nll_physical": _gaussian_nll(truth, pred, std),
    }


def _group_metrics(points_df: pd.DataFrame, group_cols: list[str], *, rel_err_floor_mm: float) -> pd.DataFrame:
    if not group_cols or points_df.empty or any(col not in points_df.columns for col in group_cols):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = points_df.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, values)}
        row.update(_metrics_from_points(group, rel_err_floor_mm=rel_err_floor_mm))
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_from_meta_row(row: pd.Series) -> dict[str, Any]:
    raw = build_teacher_raw_dict(row)
    experiment_name = row.get("experiment_name", row.get("dataset_key", None))
    if experiment_name is not None and not pd.isna(experiment_name):
        raw["dataset_key"] = str(experiment_name)
        raw["experiment_name"] = str(experiment_name)
    return raw


def _feature_group_col(df: pd.DataFrame, spec: dict[str, Any]) -> str | None:
    requested = spec.get("feature_group_col")
    if requested and requested in df.columns:
        return str(requested)
    for candidate in ("condition_id", "traj_key", "file_path"):
        if candidate in df.columns:
            return candidate
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y"})


def _slice_summaries(points_df: pd.DataFrame, *, rel_err_floor_mm: float) -> dict[str, dict[str, Any]]:
    if "is_observed_window" not in points_df.columns:
        return {}
    observed = _coerce_bool_series(points_df["is_observed_window"])
    out: dict[str, dict[str, Any]] = {}
    for label, mask in (
        ("q1_grid_observed_window", observed),
        ("q1_grid_extrapolated", ~observed),
    ):
        sub = points_df.loc[mask].copy()
        if not sub.empty:
            out[label] = {
                "filter": "is_observed_window" if label.endswith("observed_window") else "not is_observed_window",
                "overall": _metrics_from_points(sub, rel_err_floor_mm=rel_err_floor_mm),
            }
    return out


def _build_points_predictions(
    *,
    artifacts,
    registry: dict,
    df: pd.DataFrame,
    spec: dict[str, Any],
    batch_points: int,
) -> tuple[pd.DataFrame, int]:
    time_col = str(spec["time_col"])
    truth_col = str(spec["truth_col"])
    feature_columns = list(artifacts.train_config["feature_columns"])
    time_feature = str(artifacts.train_config.get("time_feature", "time_norm_0_5ms"))

    sort_cols = [col for col in ["experiment_name", "condition_id", "traj_key", time_col, "frame_pos"] if col in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    group_col = _feature_group_col(df, spec)
    group_iter: Iterable[tuple[Any, pd.DataFrame]]
    group_iter = df.groupby(group_col, sort=False, dropna=False) if group_col else [(None, df)]

    feature_blocks: list[np.ndarray] = []
    a_scale_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []
    skipped_groups = 0

    for _, group in group_iter:
        group = group.sort_values([col for col in [time_col, "frame_pos"] if col in group.columns]).reset_index(drop=True)
        valid = np.isfinite(pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float))
        valid &= np.isfinite(pd.to_numeric(group[truth_col], errors="coerce").to_numpy(dtype=float))
        if not np.any(valid):
            skipped_groups += 1
            continue
        group = group.loc[valid].reset_index(drop=True)
        time_ms = group[time_col].to_numpy(dtype=np.float32)
        truth = group[truth_col].to_numpy(dtype=np.float32)
        try:
            features_np, a_scale_np, _ = build_feature_matrix_np(
                _raw_from_meta_row(group.iloc[0]),
                time_ms,
                artifacts.scaler_state,
                feature_columns,
                registry,
                time_feature=time_feature,
            )
        except Exception as exc:
            skipped_groups += 1
            print(f"[warn] feature build failed for {spec['name']} group={group_col}: {exc}")
            continue

        meta_cols = [col for col in META_COLS_FOR_POINTS if col in group.columns]
        meta = group.loc[:, meta_cols].copy()
        meta["time_ms"] = time_ms
        feature_blocks.append(features_np)
        a_scale_blocks.append(a_scale_np.reshape(-1))
        truth_blocks.append(truth)
        meta_blocks.append(meta)

    if not feature_blocks:
        raise RuntimeError(f"No usable points found for eval set {spec['name']!r}.")

    features = np.vstack(feature_blocks).astype(np.float32)
    a_scale = np.concatenate(a_scale_blocks).astype(np.float32)
    truth = np.concatenate(truth_blocks).astype(np.float32)
    meta_df = pd.concat(meta_blocks, ignore_index=True)
    pred, std = _predict_points(
        artifacts=artifacts,
        features=features,
        a_scale=a_scale,
        batch_points=batch_points,
    )

    points_df = meta_df.copy()
    points_df["pen_true_mm"] = truth
    points_df["pen_pred_mm"] = pred
    points_df["pen_std_mm"] = std
    points_df["resid_mm"] = pred - truth
    return points_df, skipped_groups


def _load_eval_table(spec: dict[str, Any], *, filter_experiment: str | None, t_min_ms: float, t_max_ms: float) -> pd.DataFrame:
    path = _resolve_path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"Eval table not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    time_col = str(spec["time_col"])
    truth_col = str(spec["truth_col"])
    required = {time_col, truth_col, "umbrella_angle_deg", "plumes", "diameter_mm",
                "injection_duration_us", "injection_pressure_bar", "chamber_pressure_bar",
                "control_backpressure_bar"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    if filter_experiment is not None:
        if "experiment_name" not in df.columns:
            raise KeyError(f"{path} missing experiment_name required by filter_experiment.")
        df = df.loc[df["experiment_name"].astype(str) == str(filter_experiment)].copy()
        if df.empty:
            raise ValueError(f"{spec['name']} matched 0 rows for experiment_name={filter_experiment!r}.")
    time_vals = pd.to_numeric(df[time_col], errors="coerce")
    truth_vals = pd.to_numeric(df[truth_col], errors="coerce")
    mask = (
        np.isfinite(time_vals.to_numpy(dtype=float))
        & np.isfinite(truth_vals.to_numpy(dtype=float))
        & (time_vals.to_numpy(dtype=float) >= float(t_min_ms))
        & (time_vals.to_numpy(dtype=float) <= float(t_max_ms))
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"{spec['name']} has no finite rows in [{t_min_ms}, {t_max_ms}] ms.")
    return df.reset_index(drop=True)


def evaluate_point_table(
    *,
    artifacts,
    registry: dict,
    spec: dict[str, Any],
    out_dir: Path,
    filter_experiment: str | None,
    t_min_ms: float,
    t_max_ms: float,
    rel_err_floor_mm: float,
    batch_points: int,
    save_points: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_eval_table(
        spec,
        filter_experiment=filter_experiment,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
    )
    points_df, skipped_groups = _build_points_predictions(
        artifacts=artifacts,
        registry=registry,
        df=df,
        spec=spec,
        batch_points=batch_points,
    )
    points_df = _ensure_time_bin_column(points_df)

    if save_points:
        points_df.to_csv(out_dir / "points.csv", index=False)
    per_condition = _group_metrics(points_df, ["condition_id"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_condition.empty:
        per_condition.to_csv(out_dir / "per_condition.csv", index=False)
    per_experiment = _group_metrics(points_df, ["experiment_name"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_experiment.empty:
        per_experiment.to_csv(out_dir / "per_experiment.csv", index=False)
    per_trajectory = _group_metrics(points_df, ["traj_key"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_trajectory.empty:
        per_trajectory.to_csv(out_dir / "per_trajectory.csv", index=False)
    per_time_bin = _group_metrics(points_df, ["time_bin"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_time_bin.empty:
        per_time_bin.to_csv(out_dir / "per_time_bin.csv", index=False)
    per_condition_time_bin = _group_metrics(points_df, ["condition_id", "time_bin"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_condition_time_bin.empty:
        per_condition_time_bin.to_csv(out_dir / "per_condition_time_bin.csv", index=False)

    prob_summary = _write_probabilistic_diagnostics(points_df, eval_set=str(spec["name"]), out_dir=out_dir)

    slices = _slice_summaries(points_df, rel_err_floor_mm=rel_err_floor_mm)
    if slices:
        pd.DataFrame(
            [{"eval_set": name, **payload["overall"]} for name, payload in slices.items()]
        ).to_csv(out_dir / "slice_metrics.csv", index=False)

    summary = {
        "name": str(spec["name"]),
        "description": str(spec.get("description", "")),
        "path": str(_resolve_path(spec["path"])),
        "time_col": str(spec["time_col"]),
        "truth_col": str(spec["truth_col"]),
        "filter_experiment": filter_experiment,
        "t_window_ms": [float(t_min_ms), float(t_max_ms)],
        "n_input_rows": int(len(df)),
        "skipped_feature_groups": int(skipped_groups),
        "overall": _metrics_from_points(points_df, rel_err_floor_mm=rel_err_floor_mm),
        "probabilistic": prob_summary,
        "slices": slices,
        "outputs": {
            "points": str(out_dir / "points.csv") if save_points else None,
            "per_condition": str(out_dir / "per_condition.csv") if not per_condition.empty else None,
            "per_experiment": str(out_dir / "per_experiment.csv") if not per_experiment.empty else None,
            "per_trajectory": str(out_dir / "per_trajectory.csv") if not per_trajectory.empty else None,
            "per_time_bin": str(out_dir / "per_time_bin.csv") if not per_time_bin.empty else None,
            "per_condition_time_bin": str(out_dir / "per_condition_time_bin.csv") if not per_condition_time_bin.empty else None,
            "probabilistic_summary": str(out_dir / "probabilistic_summary.json") if prob_summary else None,
            "reliability_curve": str(out_dir / "reliability_curve.csv") if prob_summary else None,
            "pit_histogram": str(out_dir / "pit_histogram.csv") if prob_summary else None,
            "crps_summary": str(out_dir / "crps_summary.csv") if prob_summary else None,
            "slice_metrics": str(out_dir / "slice_metrics.csv") if slices else None,
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_point_table_evaluation(
    *,
    refinement_run: Path | str,
    synthetic_root: Path | str | None = None,
    eval_sets: list[str] | None = None,
    primary_eval_set: str = DEFAULT_PRIMARY_EVAL_SET,
    filter_experiment: str | None = None,
    device: torch.device | str | None = None,
    output_root: Path | str | None = None,
    tag: str | None = None,
    t_min_ms: float = 0.0,
    t_max_ms: float = 5.0,
    rel_err_floor_mm: float = 5.0,
    batch_points: int = 65536,
    save_points: bool = True,
    save_plots: bool = True,
    max_traj_plots: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_path = _resolve_path(refinement_run).resolve()
    artifacts = load_run_artifacts(run_path, device=device)
    registry = build_dataset_registry()

    selected = set(eval_sets or [])
    source_specs = point_eval_specs_for_synthetic_root(synthetic_root) if synthetic_root is not None else DEFAULT_POINT_EVAL_SPECS
    specs = [
        dict(spec)
        for spec in source_specs
        if not selected or str(spec["name"]) in selected
    ]
    if not specs:
        raise ValueError(f"No point eval sets selected from {sorted(selected)}.")

    eval_tag = tag or run_path.name
    out_root = _resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = out_root / f"point_eval_{datetime.now():%Y%m%d_%H%M%S}_{eval_tag}"
    out_dir.mkdir(parents=True, exist_ok=False)

    eval_summaries: dict[str, Any] = {}
    per_run_rows: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["name"])
        summary = evaluate_point_table(
            artifacts=artifacts,
            registry=registry,
            spec=spec,
            out_dir=out_dir / name,
            filter_experiment=filter_experiment,
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
            rel_err_floor_mm=rel_err_floor_mm,
            batch_points=int(batch_points),
            save_points=bool(save_points),
        )
        eval_summaries[name] = summary
        per_run_rows.append({
            "eval_set": name,
            **summary["overall"],
            **_probabilistic_headline_fields(summary.get("probabilistic")),
        })
        for slice_name, slice_payload in summary.get("slices", {}).items():
            per_run_rows.append({"eval_set": slice_name, **slice_payload["overall"]})

    if primary_eval_set not in eval_summaries:
        primary_eval_set = next(iter(eval_summaries))

    pd.DataFrame(per_run_rows).to_csv(out_dir / "per_run_metrics.csv", index=False)
    summary = {
        "refinement_run": str(run_path),
        "synthetic_root": None if synthetic_root is None else str(_resolve_path(synthetic_root)),
        "eval_kind": "point_tables",
        "primary_eval_set": primary_eval_set,
        "filter_experiment": filter_experiment,
        "t_window_ms": [float(t_min_ms), float(t_max_ms)],
        "rel_err_floor_mm": float(rel_err_floor_mm),
        "batch_points": int(batch_points),
        "save_points": bool(save_points),
        "save_plots": bool(save_plots),
        "feature_columns": list(artifacts.train_config["feature_columns"]),
        "overall": eval_summaries[primary_eval_set]["overall"],
        "eval_sets": eval_summaries,
        "outputs": {
            "per_run_metrics": str(out_dir / "per_run_metrics.csv"),
        },
    }

    if save_plots and save_points:
        try:
            from MLP.eval.point_eval_figures import export_point_eval_figures

            figure_summary = export_point_eval_figures(
                out_dir,
                max_traj_plots=max_traj_plots,
            )
            summary["figures"] = figure_summary
            summary["outputs"]["figures_dir"] = figure_summary.get("figures_dir")
            summary["outputs"]["figure_manifest"] = figure_summary.get("manifest")
        except Exception as exc:
            summary["figure_export_error"] = str(exc)
            print(f"[warn] point-eval figure export failed: {exc}")
    elif save_plots and not save_points:
        summary["figure_export_skipped"] = "save_points=False"
        print("[warn] point-eval figure export skipped: requires save_points=True.")

    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out_dir, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-run", required=True, type=Path)
    parser.add_argument("--synthetic-root", type=Path, default=None,
                        help="Synthetic-data root containing cdf_right_censoring_points/ and p50_q1_oracle/.")
    parser.add_argument("--eval-set", action="append", dest="eval_sets", default=None,
                        help="Point eval set to run. Can be repeated. Default: all.")
    parser.add_argument("--primary-eval-set", default=DEFAULT_PRIMARY_EVAL_SET)
    parser.add_argument("--filter-experiment", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--t-min-ms", type=float, default=0.0)
    parser.add_argument("--t-max-ms", type=float, default=5.0)
    parser.add_argument("--rel-err-floor-mm", type=float, default=5.0)
    parser.add_argument("--batch-points", type=int, default=65536)
    parser.add_argument("--no-save-points", action="store_true")
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--max-traj-plots", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir, summary = run_point_table_evaluation(
        refinement_run=args.refinement_run,
        synthetic_root=args.synthetic_root,
        eval_sets=args.eval_sets,
        primary_eval_set=args.primary_eval_set,
        filter_experiment=args.filter_experiment,
        device=None if args.device is None or str(args.device).lower() == "auto" else args.device,
        output_root=args.output_root,
        tag=args.tag,
        t_min_ms=args.t_min_ms,
        t_max_ms=args.t_max_ms,
        rel_err_floor_mm=args.rel_err_floor_mm,
        batch_points=args.batch_points,
        save_points=not args.no_save_points,
        save_plots=not args.no_save_plots,
        max_traj_plots=args.max_traj_plots,
    )
    print(f"Wrote point-table evaluation to {out_dir}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
