from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .fit import ResidualUncertainty, parameter_sigma_for_points, residual_sigma_for_points
    from .models import HAParams, predict
except ImportError:  # pragma: no cover
    from fit import ResidualUncertainty, parameter_sigma_for_points, residual_sigma_for_points
    from models import HAParams, predict


def finite_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    std: np.ndarray,
    rel_floor: float,
) -> dict[str, float | int]:
    resid = pred - truth
    abs_err = np.abs(resid)
    std_safe = np.maximum(std, 1e-12)
    truth_range = float(np.max(truth) - np.min(truth)) if truth.size else float("nan")
    rel_abs = abs_err / np.maximum(np.abs(truth), float(rel_floor))
    return {
        "n_points": int(truth.size),
        "rmse_mm": float(np.sqrt(np.mean(resid**2))) if truth.size else float("nan"),
        "mae_mm": float(np.mean(abs_err)) if truth.size else float("nan"),
        "bias_mm": float(np.mean(resid)) if truth.size else float("nan"),
        "median_abs_err_mm": float(np.median(abs_err)) if truth.size else float("nan"),
        "p90_abs_err_mm": float(np.quantile(abs_err, 0.90)) if truth.size else float("nan"),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)) if truth.size else float("nan"),
        "mean_rel_err": float(np.mean(rel_abs)) if truth.size else float("nan"),
        "median_rel_err": float(np.median(rel_abs)) if truth.size else float("nan"),
        "truth_min_mm": float(np.min(truth)) if truth.size else float("nan"),
        "truth_max_mm": float(np.max(truth)) if truth.size else float("nan"),
        "truth_range_mm": truth_range,
        "nrmse_range": (
            float(np.sqrt(np.mean(resid**2)) / truth_range)
            if truth.size and truth_range > 0 else float("nan")
        ),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)) if truth.size else float("nan"),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)) if truth.size else float("nan"),
        "mean_pred_std_mm": float(np.mean(std)) if truth.size else float("nan"),
    }


def build_predictions(
    points: pd.DataFrame,
    params: HAParams,
    residual_unc: ResidualUncertainty,
    boot_params: list[HAParams],
) -> pd.DataFrame:
    out = points.copy()
    pred = predict(out, params)
    sigma_resid = residual_sigma_for_points(out, residual_unc)
    sigma_param = parameter_sigma_for_points(out, boot_params)
    std = np.sqrt(np.square(sigma_resid) + np.square(sigma_param))
    out["pen_pred_mm"] = pred
    out["sigma_resid_mm"] = sigma_resid
    out["sigma_param_mm"] = sigma_param
    out["pen_std_mm"] = std
    out["resid_mm"] = out["pen_pred_mm"] - out["pen_true_mm"]
    return out


def per_trajectory(points_df: pd.DataFrame, t_max_ms: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for traj_key, g in points_df.groupby("traj_key", dropna=False):
        resid = g["resid_mm"].to_numpy(dtype=float)
        abs_resid = np.abs(resid)
        std_safe = np.maximum(g["pen_std_mm"].to_numpy(dtype=float), 1e-12)
        first = g.iloc[0]
        rows.append({
            "traj_key": traj_key,
            "folder": first.get("experiment_name"),
            "dataset_key": first.get("dataset_key"),
            "test_name": first.get("test_name"),
            "file_name": first.get("file_name"),
            "plume_idx": int(first.get("plume_idx", -1)),
            "sample_split": first.get("sample_split"),
            "split_group_id": first.get("split_group_id"),
            "injection_duration_us": float(first.get("injection_duration_us", np.nan)),
            "n_points": int(len(g)),
            "rmse_mm": float(np.sqrt(np.mean(resid**2))),
            "mae_mm": float(np.mean(abs_resid)),
            "bias_mm": float(np.mean(resid)),
            "mean_pen_mm": float(np.mean(g["pen_true_mm"])),
            "max_pen_mm": float(np.max(g["pen_true_mm"])),
            "coverage_1sigma": float(np.mean(abs_resid <= std_safe)),
            "coverage_2sigma": float(np.mean(abs_resid <= 2.0 * std_safe)),
            "is_censored": bool(float(np.max(g["time_ms"])) < (float(t_max_ms) - 1e-6)),
        })
    return pd.DataFrame(rows)


def aggregate_by(per_traj: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        per_traj.groupby(group_col, dropna=False)
        .agg(
            n_traj=("traj_key", "count"),
            rmse_mean_mm=("rmse_mm", "mean"),
            rmse_median_mm=("rmse_mm", "median"),
            mae_mean_mm=("mae_mm", "mean"),
            bias_mean_mm=("bias_mm", "mean"),
            coverage_1sigma=("coverage_1sigma", "mean"),
            coverage_2sigma=("coverage_2sigma", "mean"),
        )
        .reset_index()
        .sort_values("rmse_mean_mm")
    )


def time_bin_metrics(points_df: pd.DataFrame, bin_ms: float = 0.1) -> pd.DataFrame:
    df = points_df.copy()
    df["time_bin"] = np.floor(
        pd.to_numeric(df["time_ms"], errors="coerce") / float(bin_ms)
    ).astype(int)
    rows: list[dict[str, Any]] = []
    for b, g in df.groupby("time_bin", dropna=False):
        truth = g["pen_true_mm"].to_numpy(dtype=float)
        pred = g["pen_pred_mm"].to_numpy(dtype=float)
        std = g["pen_std_mm"].to_numpy(dtype=float)
        m = finite_metrics(truth, pred, std, rel_floor=5.0)
        rows.append({"time_bin": int(b), "time_bin_center_ms": (int(b) + 0.5) * float(bin_ms), **m})
    return pd.DataFrame(rows)


def split_counts(wide_df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in wide_df["sample_split"].value_counts().sort_index().items()}


def evaluate_split(
    points: pd.DataFrame,
    *,
    split: str,
    params: HAParams,
    residual_unc: ResidualUncertainty,
    boot_params: list[HAParams],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if split == "all":
        split_points = points.reset_index(drop=True)
    else:
        split_points = points.loc[points["sample_split"] == split].reset_index(drop=True)
    if split_points.empty:
        raise ValueError(f"No points found for split={split!r}.")
    pred_points = build_predictions(split_points, params, residual_unc, boot_params)
    per_traj = per_trajectory(pred_points, t_max_ms=float(config.get("time_max_ms", 5.0)))
    per_folder = aggregate_by(per_traj, "folder")
    per_nozzle = aggregate_by(per_traj, "dataset_key")
    time_bins = time_bin_metrics(
        pred_points, bin_ms=float(config.get("uncertainty", {}).get("time_bin_ms", 0.1))
    )
    summary = finite_metrics(
        pred_points["pen_true_mm"].to_numpy(dtype=float),
        pred_points["pen_pred_mm"].to_numpy(dtype=float),
        pred_points["pen_std_mm"].to_numpy(dtype=float),
        rel_floor=float(config.get("rel_err_floor_mm", 5.0)),
    )
    summary["n_trajectories"] = int(len(per_traj))
    summary["split"] = split
    return pred_points, per_traj, per_folder, per_nozzle, {"overall": summary, "time_bins": time_bins}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")
