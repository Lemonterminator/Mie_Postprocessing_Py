"""Fit q1 oracle baselines to uncensored per-condition CDF p50 curves.

This workflow consumes ``cdf_points_uncensored.csv`` from the CDF right-
censoring point-table step. It builds one median CDF penetration curve per
operating condition, fits the q1 quarter-root model, and writes both a
synthetic-data-compatible source (``cdf_p50_q1`` by default) and diagnostics.

The output is an oracle/smoothing baseline: it uses the evaluation condition's
own uncensored p50 curve, so it is useful for Stage-2/3 evaluation and
traditional-model comparison, not as Stage-1 training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.core.q1_model import (
    K_SQRT_SENTINEL,
    LOG_K_SQRT_SENTINEL,
    fit_quarter_only,
    spray_penetration_model_quarter_only,
)


FIT_MODEL_NAME = "quarter_only_v1_p50_oracle"
DEFAULT_SOURCE_KEY = "cdf_p50_q1"
DEFAULT_MIN_BINS = 5
DEFAULT_MIN_TRACES_PER_BIN = 4
DEFAULT_EXTRAPOLATE_T_MAX_MS = 5.0
DEFAULT_GRID_BIN_MS = 0.1
META_COLS = [
    "plumes",
    "diameter_mm",
    "umbrella_angle_deg",
    "fps",
    "chamber_pressure_bar",
    "injection_duration_us",
    "injection_pressure_bar",
    "control_backpressure_bar",
]
SERIES_META_COLS = [
    "file_path",
    "file_name",
    "file_stem",
    "plume_idx",
    "delay_frames_raw",
    "delay_frames_used",
    "delay_source",
    "seq_len",
    *META_COLS,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points-uncensored", type=Path, required=True)
    p.add_argument("--synthetic-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-key", default=DEFAULT_SOURCE_KEY)
    p.add_argument("--min-bins", type=int, default=DEFAULT_MIN_BINS)
    p.add_argument("--min-traces-per-bin", type=int, default=DEFAULT_MIN_TRACES_PER_BIN)
    p.add_argument("--extrapolate-t-max-ms", type=float, default=DEFAULT_EXTRAPOLATE_T_MAX_MS)
    p.add_argument("--grid-bin-ms", type=float, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _condition_file_stem(condition_id: Any) -> str:
    return f"condition_{int(condition_id)}"


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def load_manifest(censoring_dir: Path) -> dict[str, Any]:
    path = censoring_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_p50_curves(points: pd.DataFrame, *, min_traces_per_bin: int) -> pd.DataFrame:
    """Collapse uncensored point samples to one p50 row per condition/time bin."""
    required = {
        "condition_id",
        "condition_key",
        "time_bin",
        "time_bin_start_ms",
        "time_bin_end_ms",
        "penetration_mm",
        "experiment_name",
        "n_traces_in_bin",
    }
    missing = sorted(required - set(points.columns))
    if missing:
        raise KeyError(f"uncensored points table missing required columns: {missing}")

    points = points.copy()
    numeric_cols = [
        "condition_id",
        "time_bin",
        "time_bin_start_ms",
        "time_bin_end_ms",
        "penetration_mm",
        "n_traces_in_bin",
        *[col for col in META_COLS if col in points.columns],
    ]
    for col in numeric_cols:
        points[col] = pd.to_numeric(points[col], errors="coerce")

    group_cols = ["condition_id", "condition_key", "time_bin"]
    first_cols = [
        "experiment_name",
        "time_bin_start_ms",
        "time_bin_end_ms",
        "n_traces_in_bin",
        *[col for col in META_COLS if col in points.columns],
    ]
    agg = {
        "penetration_mm": ["median", "mean", "size"],
        **{col: "first" for col in first_cols},
    }
    curves = points.groupby(group_cols, dropna=False).agg(agg)
    curves.columns = [
        "pen_p50_mm" if col == ("penetration_mm", "median")
        else "pen_mean_mm" if col == ("penetration_mm", "mean")
        else "n_points_in_p50_bin" if col == ("penetration_mm", "size")
        else col[0]
        for col in curves.columns
    ]
    curves = curves.reset_index()
    curves["time_ms"] = 0.5 * (curves["time_bin_start_ms"] + curves["time_bin_end_ms"])
    curves = curves.loc[
        np.isfinite(curves["time_ms"])
        & np.isfinite(curves["pen_p50_mm"])
        & (curves["time_ms"] > 0.0)
        & (curves["pen_p50_mm"] > 0.0)
        & (curves["n_traces_in_bin"] >= int(min_traces_per_bin))
    ].copy()
    curves = curves.sort_values(["experiment_name", "condition_id", "time_bin"]).reset_index(drop=True)
    return curves


def _fit_metrics(t_ms: np.ndarray, y_mm: np.ndarray, log_params: np.ndarray) -> dict[str, float]:
    y_hat = spray_penetration_model_quarter_only(log_params, t_ms * 1e-3)
    resid = y_hat - y_mm
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    ss_res = float(np.sum((y_mm - y_hat) ** 2))
    ss_tot = float(np.sum((y_mm - np.mean(y_mm)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2}


def fit_conditions(
    curves: pd.DataFrame,
    *,
    min_bins: int,
    source_key: str,
    extrapolate_t_max_ms: float,
    grid_bin_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit q1 to each condition p50 curve and build observed/grid series rows."""
    fit_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for condition_id, g in curves.groupby("condition_id", sort=True, dropna=False):
        g = g.sort_values("time_ms")
        condition_id_int = int(condition_id)
        if len(g) < int(min_bins):
            continue

        t_ms = g["time_ms"].to_numpy(dtype=float)
        y_mm = g["pen_p50_mm"].to_numpy(dtype=float)
        fit = fit_quarter_only(t_ms * 1e-3, y_mm)
        if not fit["success"] or not np.all(np.isfinite(fit["log_params"])):
            continue

        log_params = np.asarray(fit["log_params"], dtype=float)
        metrics = _fit_metrics(t_ms, y_mm, log_params)
        meta = {col: g[col].iloc[0] if col in g.columns else np.nan for col in META_COLS}
        experiment = str(g["experiment_name"].iloc[0])
        file_stem = _condition_file_stem(condition_id_int)
        file_name = f"{file_stem}.csv"
        file_path = f"condition://{condition_id_int}"
        t_max_s = float(np.nanmax(t_ms) * 1e-3)

        fit_rows.append(
            {
                "fit_model": FIT_MODEL_NAME,
                "penetration_source": source_key,
                "file_path": file_path,
                "file_name": file_name,
                "file_stem": file_stem,
                "plume_idx": 0,
                "condition_id": condition_id_int,
                "condition_key": str(g["condition_key"].iloc[0]),
                "experiment_name": experiment,
                "delay_frames": 0.0,
                "delay_frames_raw": 0.0,
                "delay_frames_used": 0.0,
                "delay_source": "p50_uncensored",
                "k_sqrt": K_SQRT_SENTINEL,
                "k_quarter": fit["k_quarter"],
                "t0": fit["t0"],
                "s": fit["s"],
                "cost": fit["cost"],
                "success": fit["success"],
                "n": fit["n"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "bias": metrics["bias"],
                "r2": metrics["r2"],
                "log_k_sqrt": LOG_K_SQRT_SENTINEL,
                "log_k_quarter": float(log_params[0]),
                "log_t0": float(log_params[1]),
                "log_s": float(log_params[2]),
                "t_max_s": t_max_s,
                "nfev": fit["nfev"],
                "optimality": fit["optimality"],
                "status": fit["status"],
                "std_log_k_quarter": fit["std_log_k_quarter"],
                "std_log_t0": fit["std_log_t0"],
                "std_log_s": fit["std_log_s"],
                "corr_logk_logt0": fit["corr_logk_logt0"],
                "corr_logk_logs": fit["corr_logk_logs"],
                "corr_logt0_logs": fit["corr_logt0_logs"],
                **meta,
                "cost_per_point": 2.0 * _safe_float(fit["cost"]) / max(int(fit["n"]), 1),
                "penetration_far_mm": float(
                    spray_penetration_model_quarter_only(log_params, np.array([5.0e-3]))[0]
                ),
                "mask_basic": True,
                "mask_penetration_far": True,
                "z_t0": 0.0,
                "z_rmse": 0.0,
                "z_cost": 0.0,
                "mask_outlier": False,
                "flag_bad_fit": False,
                "flag_bad_fit_q1": False,
            }
        )

        observed_fit = spray_penetration_model_quarter_only(log_params, t_ms * 1e-3)
        for idx, (row, pred) in enumerate(zip(g.itertuples(index=False), observed_fit)):
            observed_rows.append(
                {
                    "experiment_name": experiment,
                    "condition_id": condition_id_int,
                    "condition_key": str(getattr(row, "condition_key")),
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_stem": file_stem,
                    "plume_idx": 0,
                    "frame_pos": int(idx),
                    "time_s": float(getattr(row, "time_ms")) * 1e-3,
                    "time_ms": float(getattr(row, "time_ms")),
                    "penetration_mm": float(getattr(row, "pen_p50_mm")),
                    "q1_fit_mm": float(pred),
                    "residual_mm": float(pred - getattr(row, "pen_p50_mm")),
                    "n_points_in_p50_bin": int(getattr(row, "n_points_in_p50_bin")),
                    "n_traces_in_bin": int(getattr(row, "n_traces_in_bin")),
                    "delay_frames_raw": 0.0,
                    "delay_frames_used": 0.0,
                    "delay_source": "p50_uncensored",
                    **meta,
                }
            )

        grid = np.arange(0.0, float(extrapolate_t_max_ms) + 0.5 * float(grid_bin_ms), float(grid_bin_ms))
        pred_grid = spray_penetration_model_quarter_only(log_params, grid * 1e-3)
        obs_min = float(np.nanmin(t_ms))
        obs_max = float(np.nanmax(t_ms))
        for idx, (time_ms, pred) in enumerate(zip(grid, pred_grid)):
            prediction_rows.append(
                {
                    "experiment_name": experiment,
                    "condition_id": condition_id_int,
                    "condition_key": str(g["condition_key"].iloc[0]),
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_stem": file_stem,
                    "plume_idx": 0,
                    "frame_pos": int(idx),
                    "time_ms": float(time_ms),
                    "time_s": float(time_ms) * 1e-3,
                    "q1_fit_mm": float(pred),
                    "is_observed_window": bool(obs_min <= time_ms <= obs_max),
                    **meta,
                }
            )

    return (
        pd.DataFrame(fit_rows).replace([np.inf, -np.inf], np.nan),
        pd.DataFrame(observed_rows).replace([np.inf, -np.inf], np.nan),
        pd.DataFrame(prediction_rows).replace([np.inf, -np.inf], np.nan),
    )


def _wide_from_long(series: pd.DataFrame, value_col: str = "penetration_mm") -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=SERIES_META_COLS)
    key_cols = ["file_path", "file_name", "file_stem", "plume_idx"]
    meta_cols = [
        "experiment_name",
        "condition_id",
        "condition_key",
        "delay_frames_raw",
        "delay_frames_used",
        "delay_source",
        *META_COLS,
    ]
    base = (
        series.sort_values(key_cols + ["frame_pos"])
        .groupby(key_cols, dropna=False)
        .agg(
            experiment_name=("experiment_name", "first"),
            condition_id=("condition_id", "first"),
            condition_key=("condition_key", "first"),
            delay_frames_raw=("delay_frames_raw", "first"),
            delay_frames_used=("delay_frames_used", "first"),
            delay_source=("delay_source", "first"),
            seq_len=("frame_pos", "count"),
            **{col: (col, "first") for col in META_COLS if col in series.columns},
        )
        .reset_index()
    )
    time_wide = (
        series.pivot_table(index=key_cols, columns="frame_pos", values="time_ms", aggfunc="first")
        .sort_index(axis=1)
        .rename(columns=lambda c: f"time_ms_{int(c):03d}")
        .reset_index()
    )
    pen_wide = (
        series.pivot_table(index=key_cols, columns="frame_pos", values=value_col, aggfunc="first")
        .sort_index(axis=1)
        .rename(columns=lambda c: f"penetration_mm_{int(c):03d}")
        .reset_index()
    )
    wide = base.merge(time_wide, on=key_cols, how="left").merge(pen_wide, on=key_cols, how="left")
    ordered = [
        col for col in [
            "experiment_name",
            "condition_id",
            "condition_key",
            "file_path",
            "file_name",
            "file_stem",
            "plume_idx",
            "delay_frames_raw",
            "delay_frames_used",
            "delay_source",
            "seq_len",
            *META_COLS,
        ] if col in wide.columns
    ]
    return wide.loc[:, ordered + [col for col in wide.columns if col not in ordered]]


def write_synthetic_source(
    *,
    synthetic_root: Path,
    source_key: str,
    fit_rows: pd.DataFrame,
    observed_series: pd.DataFrame,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if fit_rows.empty or "experiment_name" not in fit_rows.columns:
        return outputs
    for experiment, fit_part in fit_rows.groupby("experiment_name", dropna=False):
        experiment = str(experiment)
        source_dir = synthetic_root / experiment / source_key
        for subdir in ("all", "clean", "series_all", "series_clean", "series_wide_all", "series_wide_clean"):
            (source_dir / subdir).mkdir(parents=True, exist_ok=True)
        series_part = observed_series.loc[observed_series["experiment_name"].astype(str) == experiment].copy()
        fit_out = fit_part.drop(columns=["experiment_name"], errors="ignore").copy()
        series_out = series_part.drop(columns=["experiment_name"], errors="ignore").copy()
        wide = _wide_from_long(series_part).drop(columns=["experiment_name"], errors="ignore")

        for subdir, df in (
            ("all", fit_out),
            ("clean", fit_out),
            ("series_all", series_out),
            ("series_clean", series_out),
            ("series_wide_all", wide),
            ("series_wide_clean", wide),
        ):
            path = source_dir / subdir / "p50_conditions.csv"
            df.to_csv(path, index=False)
            outputs[f"{experiment}/{subdir}"] = str(path)
    return outputs


def write_plots(out_dir: Path, fit_rows: pd.DataFrame, observed: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, str]:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    if not fit_rows.empty:
        ax.hist(fit_rows["rmse"].dropna(), bins=40, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Observed-window p50-q1 RMSE [mm]")
    ax.set_ylabel("Condition count")
    ax.set_title("p50-q1 oracle fit error by condition")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = plots_dir / "rmse_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    paths["rmse_distribution"] = str(path)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=150)
    if not observed.empty:
        sample_obs = observed.sample(n=min(len(observed), 50000), random_state=2026)
        ax.scatter(sample_obs["time_ms"], sample_obs["penetration_mm"], s=2.0, alpha=0.18, color="#4c78a8", label="p50 bins")
    if not predictions.empty:
        sample_pred = predictions.sample(n=min(len(predictions), 50000), random_state=2027)
        ax.scatter(sample_pred["time_ms"], sample_pred["q1_fit_mm"], s=1.0, alpha=0.10, color="#e45756", label="q1 extrapolation")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("CDF penetration [mm]")
    ax.set_title("p50 bins and q1 oracle extrapolations")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = plots_dir / "p50_q1_time_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    paths["p50_q1_time_scatter"] = str(path)
    return paths


def run_p50_q1_oracle(
    *,
    points_uncensored: Path,
    synthetic_root: Path,
    out_dir: Path,
    source_key: str = DEFAULT_SOURCE_KEY,
    min_bins: int = DEFAULT_MIN_BINS,
    min_traces_per_bin: int = DEFAULT_MIN_TRACES_PER_BIN,
    extrapolate_t_max_ms: float = DEFAULT_EXTRAPOLATE_T_MAX_MS,
    grid_bin_ms: float | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    points_uncensored = points_uncensored.expanduser().resolve()
    synthetic_root = synthetic_root.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(points_uncensored.parent)
    if grid_bin_ms is None:
        grid_bin_ms = float(manifest.get("bin_ms", DEFAULT_GRID_BIN_MS))

    points = pd.read_csv(points_uncensored, low_memory=False)
    curves = build_p50_curves(points, min_traces_per_bin=int(min_traces_per_bin))
    fit_rows, observed, predictions = fit_conditions(
        curves,
        min_bins=int(min_bins),
        source_key=source_key,
        extrapolate_t_max_ms=float(extrapolate_t_max_ms),
        grid_bin_ms=float(grid_bin_ms),
    )

    curves.to_csv(out_dir / "p50_condition_curves.csv", index=False)
    fit_rows.to_csv(out_dir / "p50_q1_fit_metrics.csv", index=False)
    predictions.to_csv(out_dir / "p50_q1_predictions.csv", index=False)
    observed.to_csv(out_dir / "p50_q1_observed_fit_points.csv", index=False)
    synthetic_outputs = write_synthetic_source(
        synthetic_root=synthetic_root,
        source_key=source_key,
        fit_rows=fit_rows,
        observed_series=observed,
    )
    plot_outputs = write_plots(out_dir, fit_rows, observed, predictions) if make_plots else {}

    summary = {
        "script": str(Path(__file__).resolve()),
        "baseline_type": "oracle_p50_smoothing_baseline_not_stage1_training_labels",
        "points_uncensored": str(points_uncensored),
        "synthetic_root": str(synthetic_root),
        "source_key": source_key,
        "fit_model": FIT_MODEL_NAME,
        "min_bins": int(min_bins),
        "min_traces_per_bin": int(min_traces_per_bin),
        "extrapolate_t_max_ms": float(extrapolate_t_max_ms),
        "grid_bin_ms": float(grid_bin_ms),
        "n_input_points": int(len(points)),
        "n_p50_bins": int(len(curves)),
        "n_fit_conditions": int(len(fit_rows)),
        "rmse_median_mm": float(fit_rows["rmse"].median()) if "rmse" in fit_rows.columns and not fit_rows.empty else None,
        "rmse_p90_mm": float(fit_rows["rmse"].quantile(0.90)) if "rmse" in fit_rows.columns and not fit_rows.empty else None,
        "outputs": {
            "p50_condition_curves": str(out_dir / "p50_condition_curves.csv"),
            "p50_q1_fit_metrics": str(out_dir / "p50_q1_fit_metrics.csv"),
            "p50_q1_predictions": str(out_dir / "p50_q1_predictions.csv"),
            "p50_q1_observed_fit_points": str(out_dir / "p50_q1_observed_fit_points.csv"),
            "synthetic_source": synthetic_outputs,
            "plots": plot_outputs,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_p50_q1_oracle(
        points_uncensored=args.points_uncensored,
        synthetic_root=args.synthetic_root,
        out_dir=args.out_dir,
        source_key=args.source_key,
        min_bins=args.min_bins,
        min_traces_per_bin=args.min_traces_per_bin,
        extrapolate_t_max_ms=args.extrapolate_t_max_ms,
        grid_bin_ms=args.grid_bin_ms,
        make_plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
