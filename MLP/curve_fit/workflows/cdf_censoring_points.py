"""Build point-level CDF tables with explicit FOV/right-censoring labels.

The script expands ``cdf/series_wide_clean`` into finite point samples in a
fixed time window, groups them by the same operating-condition columns used by
Stage 3, bins by time, and marks all points at/after the first bin where either
raw sample density drops or penetration reaches the field of view.

The generated point table is the bridge between curve fitting and Stage-3
raw-series supervision: each point carries the original CDF measurement plus
``is_right_censored`` and the trigger reason. Companion condition/bin tables
and plots explain why a condition was marked censored.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.MLP_training import train_stage3_distillation_plus_raw_series as stage3


DEFAULT_SYNTHETIC_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "eval"
DEFAULT_SOURCE = "cdf"
DEFAULT_SPLIT = "clean"
DEFAULT_T_MIN_MS = 0.0
DEFAULT_T_MAX_MS = 5.0
DEFAULT_BIN_MS = stage3.BIN_MS
DEFAULT_FOV_CAP_MM = 85.0
DEFAULT_FOV_CAP_FRACTION = 0.95
DEFAULT_FOV_STAT = "pen_p95_mm"
DEFAULT_DENSITY_RATIO = stage3.UNCERTAIN_RATIO
DEFAULT_DENSITY_CONSECUTIVE_BINS = stage3.CONSECUTIVE_BINS
DEFAULT_DENSITY_SMOOTH_WINDOW = 3
DEFAULT_DENSITY_MIN_COUNT = 4


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    return out or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--split", choices=("clean", "all"), default=DEFAULT_SPLIT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dataset", action="append", dest="datasets",
                   help="Optional experiment_name filter. Can be passed multiple times.")
    p.add_argument("--folder", action="append", dest="folders",
                   help="Optional test folder filter, e.g. T1. Can be passed multiple times.")
    p.add_argument("--max-wide-rows", type=int, default=None,
                   help="Debug limit after loading and filtering the wide CDF table.")

    p.add_argument("--t-min-ms", type=float, default=DEFAULT_T_MIN_MS)
    p.add_argument("--t-max-ms", type=float, default=DEFAULT_T_MAX_MS)
    p.add_argument("--bin-ms", type=float, default=DEFAULT_BIN_MS)

    p.add_argument("--condition-cols", default=",".join(stage3.REGIME_GROUP_COLS),
                   help="Comma-separated condition columns used for time-bin censoring.")
    p.add_argument("--cap-group-cols", default="experiment_name",
                   help="Comma-separated columns for estimated FOV caps.")
    p.add_argument("--estimate-fov-cap", action="store_true",
                   help="Estimate cap from trace maxima per --cap-group-cols instead of using --fov-cap-mm.")
    p.add_argument("--cap-percentile", type=float, default=99.0,
                   help="Percentile of trace maxima used when --estimate-fov-cap is set.")
    p.add_argument("--fov-cap-mm", type=float, default=DEFAULT_FOV_CAP_MM,
                   help="Flat FOV cap used unless --estimate-fov-cap is set.")
    p.add_argument("--fov-cap-fraction", type=float, default=DEFAULT_FOV_CAP_FRACTION,
                   help="FOV trigger threshold = this fraction * cap_mm.")
    p.add_argument("--fov-stat", choices=("pen_p90_mm", "pen_p95_mm", "pen_p99_mm", "pen_max_mm"),
                   default=DEFAULT_FOV_STAT,
                   help="Per-bin penetration statistic used for FOV saturation.")
    p.add_argument("--fov-min-count", type=int, default=DEFAULT_DENSITY_MIN_COUNT,
                   help="Minimum trajectory count in a bin before FOV saturation can trigger censoring.")
    p.add_argument("--fov-consecutive-bins", type=int, default=1)

    p.add_argument("--density-ratio", type=float, default=DEFAULT_DENSITY_RATIO,
                   help="Density-drop trigger: smoothed count / peak smoothed count below this value.")
    p.add_argument("--density-min-count", type=int, default=DEFAULT_DENSITY_MIN_COUNT,
                   help="Density-drop trigger also fires when raw bin count is below this value.")
    p.add_argument("--density-consecutive-bins", type=int, default=DEFAULT_DENSITY_CONSECUTIVE_BINS)
    p.add_argument("--density-smooth-window", type=int, default=DEFAULT_DENSITY_SMOOTH_WINDOW)
    plot_group = p.add_mutually_exclusive_group()
    plot_group.add_argument("--plots", dest="plots", action="store_true", default=True,
                            help="Write diagnostic PNG figures under <out-dir>/plots. This is the default.")
    plot_group.add_argument("--no-plots", dest="plots", action="store_false",
                            help="Skip all diagnostic PNG figures and only write CSV/JSON outputs.")
    condition_plot_group = p.add_mutually_exclusive_group()
    condition_plot_group.add_argument("--condition-plots", dest="condition_plots", action="store_true", default=True,
                                      help="Write one time-bin diagnostic PNG per condition. This is the default.")
    condition_plot_group.add_argument("--no-condition-plots", dest="condition_plots", action="store_false",
                                      help="Skip per-condition PNGs while keeping overview plots.")
    p.add_argument("--plot-sample-points", type=int, default=120000,
                   help="Maximum point sample used for scatter plots.")
    p.add_argument("--max-condition-plots", type=int, default=None,
                   help="Debug limit for per-condition plots.")
    p.add_argument("--condition-plot-progress-every", type=int, default=100,
                   help="Print progress every N per-condition plots; set <=0 to silence.")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.t_max_ms <= args.t_min_ms:
        raise ValueError("--t-max-ms must be larger than --t-min-ms.")
    if args.bin_ms <= 0:
        raise ValueError("--bin-ms must be positive.")
    if args.fov_cap_fraction <= 0:
        raise ValueError("--fov-cap-fraction must be positive.")
    if not 0 < args.cap_percentile <= 100:
        raise ValueError("--cap-percentile must be in (0, 100].")
    if args.density_ratio <= 0:
        raise ValueError("--density-ratio must be positive.")
    if args.max_condition_plots is not None and int(args.max_condition_plots) < 1:
        raise ValueError("--max-condition-plots must be >= 1 when provided.")
    for attr in ("fov_min_count", "fov_consecutive_bins", "density_min_count", "density_consecutive_bins"):
        if int(getattr(args, attr)) < 1:
            raise ValueError(f"--{attr.replace('_', '-')} must be >= 1.")


def stable_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(["all"] * len(df), index=df.index)
    return df.loc[:, cols].astype("string").fillna("<NA>").agg("|".join, axis=1)


def first_consecutive_true(mask: np.ndarray, run_len: int) -> int | None:
    if run_len <= 1:
        found = np.flatnonzero(mask)
        return int(found[0]) if len(found) else None
    if len(mask) < run_len:
        return None
    for start in range(0, len(mask) - run_len + 1):
        if bool(np.all(mask[start : start + run_len])):
            return start
    return None


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(window, 1))
    if window == 1 or len(values) == 0:
        return values.astype(float)
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def frame_ids_for_wide(df: pd.DataFrame) -> list[int]:
    return sorted(
        set(stage3.available_frame_ids(df, "time_ms_")).intersection(
            stage3.available_frame_ids(df, "penetration_mm_")
        )
    )


def folder_from_file_path(value: object) -> str:
    text = str(value)
    if not text or text.lower() == "nan":
        return "unknown"
    return Path(text).parent.name


def load_cdf_wide(args: argparse.Namespace) -> pd.DataFrame:
    stage3.SYNTHETIC_ROOT = args.synthetic_root.expanduser().resolve()
    wide = stage3.load_source_table(args.source, split=args.split)
    if args.datasets:
        selected = set(args.datasets)
        wide = wide.loc[wide["experiment_name"].astype(str).isin(selected)].copy()
    if args.folders:
        selected = set(args.folders)
        if "file_path" not in wide.columns:
            raise KeyError("--folder filtering requires a file_path column.")
        folder = wide["file_path"].map(folder_from_file_path)
        wide = wide.loc[folder.astype(str).isin(selected)].copy()
    if args.max_wide_rows is not None:
        wide = wide.head(int(args.max_wide_rows)).copy()
    if wide.empty:
        raise ValueError("No CDF wide rows remain after filtering.")
    return wide.reset_index(drop=True)


def expand_wide_to_points(
    wide: pd.DataFrame,
    *,
    t_min_ms: float,
    t_max_ms: float,
    bin_ms: float,
    condition_cols: list[str],
) -> pd.DataFrame:
    """Convert wide per-trajectory series into one row per valid CDF point."""
    missing_condition = [col for col in condition_cols if col not in wide.columns]
    if missing_condition:
        raise KeyError(f"CDF wide table missing condition columns: {missing_condition}")

    frame_ids = frame_ids_for_wide(wide)
    if not frame_ids:
        raise ValueError("No matching time_ms_* and penetration_mm_* columns found.")

    time_mat = stage3.extract_prefixed_matrix(wide, "time_ms_", frame_ids)
    pen_mat = stage3.extract_prefixed_matrix(wide, "penetration_mm_", frame_ids)
    n_rows = len(wide)
    n_frames = len(frame_ids)

    base_cols = [
        "experiment_name",
        "file_path",
        "file_name",
        "file_stem",
        "plume_idx",
        *stage3.COMMON_META_COLS,
        "delay_frames_raw",
        "delay_frames_used",
        "delay_source",
        "seq_len",
    ]
    base_cols = [col for col in base_cols if col in wide.columns]
    repeated = wide.loc[:, base_cols].iloc[np.repeat(np.arange(n_rows), n_frames)].reset_index(drop=True)
    repeated["wide_row_idx"] = np.repeat(np.arange(n_rows, dtype=np.int32), n_frames)
    repeated["frame_pos"] = np.tile(np.asarray(frame_ids, dtype=np.int32), n_rows)
    repeated["time_ms"] = time_mat.reshape(-1)
    repeated["penetration_mm"] = pen_mat.reshape(-1)
    repeated["test_name"] = repeated["file_path"].map(folder_from_file_path) if "file_path" in repeated else "unknown"
    repeated["traj_key"] = stable_key(
        repeated,
        [col for col in ["experiment_name", "file_path", "plume_idx"] if col in repeated.columns],
    )

    valid = (
        np.isfinite(repeated["time_ms"].to_numpy(dtype=float))
        & np.isfinite(repeated["penetration_mm"].to_numpy(dtype=float))
        & (repeated["time_ms"].to_numpy(dtype=float) >= float(t_min_ms))
        & (repeated["time_ms"].to_numpy(dtype=float) <= float(t_max_ms))
    )
    out = repeated.loc[valid].copy()
    if out.empty:
        raise ValueError("No finite CDF points found in the requested time window.")

    n_bins = int(math.ceil((float(t_max_ms) - float(t_min_ms)) / float(bin_ms)))
    time_rel = (out["time_ms"].to_numpy(dtype=float) - float(t_min_ms)) / float(bin_ms)
    out["time_bin"] = np.floor(time_rel).astype(np.int32).clip(0, n_bins - 1)
    out["time_bin_start_ms"] = float(t_min_ms) + out["time_bin"].astype(float) * float(bin_ms)
    out["time_bin_end_ms"] = np.minimum(out["time_bin_start_ms"] + float(bin_ms), float(t_max_ms))

    out["condition_key"] = stable_key(out, condition_cols)
    out["condition_id"] = pd.factorize(out["condition_key"], sort=True)[0].astype(np.int32)
    return out.reset_index(drop=True)


def attach_fov_caps(
    points: pd.DataFrame,
    *,
    estimate_fov_cap: bool,
    cap_group_cols: list[str],
    cap_percentile: float,
    fov_cap_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach a flat or data-estimated field-of-view cap to each point."""
    out = points.copy()
    missing = [col for col in cap_group_cols if col not in out.columns]
    if missing:
        raise KeyError(f"Point table missing cap group columns: {missing}")

    trace_summary = (
        out.groupby([*cap_group_cols, "traj_key"], dropna=False)
        .agg(
            trace_n_points=("penetration_mm", "size"),
            trace_t_min_ms=("time_ms", "min"),
            trace_t_max_ms=("time_ms", "max"),
            trace_last_time_ms=("time_ms", "max"),
            trace_max_pen_mm=("penetration_mm", "max"),
            trace_last_pen_mm=("penetration_mm", lambda values: float(values.iloc[-1])),
        )
        .reset_index()
    )

    if estimate_fov_cap:
        cap_lookup = (
            trace_summary.groupby(cap_group_cols, dropna=False)["trace_max_pen_mm"]
            .quantile(float(cap_percentile) / 100.0)
            .reset_index()
            .rename(columns={"trace_max_pen_mm": "cap_mm"})
        )
        cap_lookup["cap_method"] = f"trace_max_p{cap_percentile:g}"
        out = out.merge(cap_lookup, on=cap_group_cols, how="left")
        fallback = float(fov_cap_mm)
        if math.isfinite(fallback) and fallback > 0:
            out["cap_mm"] = out["cap_mm"].fillna(fallback)
    else:
        cap_lookup = out.loc[:, cap_group_cols].drop_duplicates().copy()
        cap_lookup["cap_mm"] = float(fov_cap_mm)
        cap_lookup["cap_method"] = "flat_fov_cap_mm"
        out["cap_mm"] = float(fov_cap_mm)

    trace_summary = trace_summary.merge(cap_lookup, on=cap_group_cols, how="left")
    return out, trace_summary


def condition_lookup(points: pd.DataFrame, condition_cols: list[str]) -> pd.DataFrame:
    cols = ["condition_id", "condition_key", *condition_cols]
    return points.loc[:, cols].drop_duplicates("condition_id").sort_values("condition_id").reset_index(drop=True)


def compute_condition_bin_table(
    points: pd.DataFrame,
    *,
    condition_cols: list[str],
    t_min_ms: float,
    t_max_ms: float,
    bin_ms: float,
    fov_stat: str,
    fov_cap_fraction: float,
    fov_min_count: int,
    fov_consecutive_bins: int,
    density_ratio: float,
    density_min_count: int,
    density_consecutive_bins: int,
    density_smooth_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-condition time-bin statistics and choose censoring start bins."""
    n_bins = int(math.ceil((float(t_max_ms) - float(t_min_ms)) / float(bin_ms)))
    condition_meta = condition_lookup(points, condition_cols).set_index("condition_id", drop=False)

    bin_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for condition_id, g in points.groupby("condition_id", sort=True, dropna=False):
        grouped = g.groupby("time_bin", dropna=False)
        base = grouped.agg(
            n_points=("penetration_mm", "size"),
            n_traces=("traj_key", "nunique"),
            pen_min_mm=("penetration_mm", "min"),
            pen_mean_mm=("penetration_mm", "mean"),
            pen_max_mm=("penetration_mm", "max"),
            cap_mm=("cap_mm", "median"),
        )
        quant = grouped["penetration_mm"].quantile([0.50, 0.90, 0.95, 0.99]).unstack()
        quant = quant.rename(columns={
            0.50: "pen_p50_mm",
            0.90: "pen_p90_mm",
            0.95: "pen_p95_mm",
            0.99: "pen_p99_mm",
        })
        stats = base.join(quant, how="outer").reindex(range(n_bins))
        stats.index.name = "time_bin"
        stats["n_points"] = stats["n_points"].fillna(0).astype(int)
        stats["n_traces"] = stats["n_traces"].fillna(0).astype(int)
        cap_mm = float(np.nanmedian(g["cap_mm"].to_numpy(dtype=float)))
        stats["cap_mm"] = stats["cap_mm"].fillna(cap_mm)

        counts = stats["n_traces"].to_numpy(dtype=float)
        counts_smooth = rolling_mean(counts, int(density_smooth_window))
        b_peak = int(np.nanargmax(counts_smooth)) if len(counts_smooth) else 0
        n_ref = float(max(counts_smooth[b_peak], 1.0)) if len(counts_smooth) else 1.0
        coverage_ratio = counts_smooth / n_ref

        # Density drop catches bins where most trajectories have already ended;
        # FOV saturation catches bins whose high-percentile penetration sits at
        # the optical cap. The earlier trigger starts right-censoring.
        density_mask = (coverage_ratio < float(density_ratio)) | (counts < int(density_min_count))
        rel_density = first_consecutive_true(
            density_mask[b_peak:],
            int(density_consecutive_bins),
        )
        b_density_start = b_peak + rel_density if rel_density is not None else n_bins

        fov_values = stats[fov_stat].to_numpy(dtype=float)
        fov_threshold = float(fov_cap_fraction) * cap_mm
        fov_mask = np.isfinite(fov_values) & (fov_values >= fov_threshold) & (counts >= int(fov_min_count))
        rel_fov = first_consecutive_true(
            fov_mask[b_peak:],
            int(fov_consecutive_bins),
        )
        b_fov_start = b_peak + rel_fov if rel_fov is not None else n_bins

        b_censor_start = min(b_density_start, b_fov_start)
        if b_censor_start >= n_bins:
            reason = "none"
        elif b_density_start == b_fov_start:
            reason = "density_drop+fov_saturation"
        elif b_density_start < b_fov_start:
            reason = "density_drop"
        else:
            reason = "fov_saturation"

        stats = stats.reset_index()
        stats.insert(0, "condition_id", int(condition_id))
        stats["condition_key"] = str(condition_meta.loc[condition_id, "condition_key"])
        for col in condition_cols:
            stats[col] = condition_meta.loc[condition_id, col]
        stats["time_bin_start_ms"] = float(t_min_ms) + stats["time_bin"].astype(float) * float(bin_ms)
        stats["time_bin_end_ms"] = np.minimum(stats["time_bin_start_ms"] + float(bin_ms), float(t_max_ms))
        stats["n_traces_smooth"] = counts_smooth
        stats["density_ratio_to_peak"] = coverage_ratio
        stats["density_drop_trigger"] = density_mask
        stats["fov_threshold_mm"] = fov_threshold
        stats["fov_saturation_trigger"] = fov_mask
        stats["b_peak_density"] = b_peak
        stats["b_density_start"] = b_density_start
        stats["b_fov_start"] = b_fov_start
        stats["b_censor_start"] = b_censor_start
        stats["censor_start_reason"] = reason
        stats["is_right_censored_bin"] = stats["time_bin"] >= b_censor_start
        bin_rows.append(stats)

        row = {
            "condition_id": int(condition_id),
            "condition_key": str(condition_meta.loc[condition_id, "condition_key"]),
            "n_points": int(len(g)),
            "n_traces": int(g["traj_key"].nunique()),
            "cap_mm": cap_mm,
            "fov_threshold_mm": fov_threshold,
            "b_peak_density": b_peak,
            "b_density_start": int(b_density_start),
            "b_fov_start": int(b_fov_start),
            "b_censor_start": int(b_censor_start),
            "censor_start_time_ms": (
                float(t_min_ms) + int(b_censor_start) * float(bin_ms)
                if b_censor_start < n_bins
                else math.nan
            ),
            "censor_start_reason": reason,
        }
        for col in condition_cols:
            row[col] = condition_meta.loc[condition_id, col]
        summary_rows.append(row)

    bins = pd.concat(bin_rows, ignore_index=True, sort=False)
    summary = pd.DataFrame(summary_rows).sort_values("condition_id").reset_index(drop=True)
    return bins, summary


def attach_bin_labels(points: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """Merge condition/bin censoring decisions back onto individual points."""
    label_cols = [
        "condition_id",
        "time_bin",
        "n_points",
        "n_traces",
        "n_traces_smooth",
        "density_ratio_to_peak",
        "cap_mm",
        "fov_threshold_mm",
        "b_peak_density",
        "b_density_start",
        "b_fov_start",
        "b_censor_start",
        "censor_start_reason",
        "is_right_censored_bin",
    ]
    out = points.merge(
        bins.loc[:, label_cols].rename(
            columns={
                "n_points": "n_points_in_bin",
                "n_traces": "n_traces_in_bin",
                "cap_mm": "cap_mm_bin",
            }
        ),
        on=["condition_id", "time_bin"],
        how="left",
    )
    out["is_right_censored"] = out["is_right_censored_bin"].fillna(False).astype(bool)
    if "cap_mm" in out.columns and "cap_mm_bin" in out.columns:
        out["cap_mm"] = out["cap_mm"].fillna(out["cap_mm_bin"])
        out = out.drop(columns=["cap_mm_bin"])
    return out


def ordered_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "condition_id",
        "condition_key",
        "traj_key",
        "experiment_name",
        "test_name",
        "file_path",
        "file_name",
        "file_stem",
        "plume_idx",
        "wide_row_idx",
        "frame_pos",
        "time_ms",
        "time_bin",
        "time_bin_start_ms",
        "time_bin_end_ms",
        "penetration_mm",
        "is_right_censored",
        "censor_start_reason",
        "b_censor_start",
        "b_density_start",
        "b_fov_start",
        "cap_mm",
        "fov_threshold_mm",
        "n_points_in_bin",
        "n_traces_in_bin",
        "density_ratio_to_peak",
    ]
    return [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]


def _short_text(value: object, max_len: int = 42) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _safe_path_part(value: object, max_len: int = 96) -> str:
    text = str(value).strip() or "unknown"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._ ")
    return (text[:max_len].rstrip("._ ") or "unknown")


def _compact_num(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:g}".replace(".", "p").replace("-", "m")


def _condition_title(row: pd.Series) -> str:
    parts = [
        _short_text(row.get("experiment_name", "unknown"), 34),
        f"d={row.get('diameter_mm', np.nan):g} mm" if pd.notna(row.get("diameter_mm", np.nan)) else "d=NA",
        f"Pch={row.get('chamber_pressure_bar', np.nan):g} bar" if pd.notna(row.get("chamber_pressure_bar", np.nan)) else "Pch=NA",
        f"Pinj={row.get('injection_pressure_bar', np.nan):g} bar" if pd.notna(row.get("injection_pressure_bar", np.nan)) else "Pinj=NA",
        f"dur={row.get('injection_duration_us', np.nan):g} us" if pd.notna(row.get("injection_duration_us", np.nan)) else "dur=NA",
    ]
    return " | ".join(parts)


def _condition_title_multiline(row: pd.Series) -> str:
    experiment = str(row.get("experiment_name", "unknown"))
    parts = [
        f"d={row.get('diameter_mm', np.nan):g} mm" if pd.notna(row.get("diameter_mm", np.nan)) else "d=NA",
        f"Pch={row.get('chamber_pressure_bar', np.nan):g} bar" if pd.notna(row.get("chamber_pressure_bar", np.nan)) else "Pch=NA",
        f"Pinj={row.get('injection_pressure_bar', np.nan):g} bar" if pd.notna(row.get("injection_pressure_bar", np.nan)) else "Pinj=NA",
        f"dur={row.get('injection_duration_us', np.nan):g} us" if pd.notna(row.get("injection_duration_us", np.nan)) else "dur=NA",
        f"plumes={row.get('plumes', np.nan):g}" if pd.notna(row.get("plumes", np.nan)) else "plumes=NA",
    ]
    return experiment + "\n" + " | ".join(parts)


def _maybe_vline(ax: Any, x: object, label: str, color: str, style: str = "--", linewidth: float = 1.0) -> None:
    try:
        number = float(x)
    except Exception:
        return
    if math.isfinite(number):
        ax.axvline(number, color=color, linestyle=style, linewidth=linewidth, label=label)


def write_condition_plots(
    *,
    plot_dir: Path,
    points: pd.DataFrame,
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    plt: Any,
    max_condition_plots: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    condition_root = plot_dir / "conditions_by_experiment"
    condition_root.mkdir(parents=True, exist_ok=True)

    bins_by_condition = {
        int(condition_id): group.sort_values("time_bin").copy()
        for condition_id, group in bins.groupby("condition_id", sort=False, dropna=False)
    }
    point_cols = ["condition_id", "time_ms", "penetration_mm", "is_right_censored"]
    points_by_condition = {
        int(condition_id): group.loc[:, point_cols].sort_values("time_ms").copy()
        for condition_id, group in points.groupby("condition_id", sort=False, dropna=False)
    }
    rows = summary.sort_values("condition_id").copy()
    if max_condition_plots is not None:
        rows = rows.head(int(max_condition_plots))

    index_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        condition_id = int(row["condition_id"])
        sub = bins_by_condition.get(condition_id)
        if sub is None or sub.empty:
            continue

        experiment = str(row.get("experiment_name", "unknown"))
        experiment_dir = condition_root / _safe_path_part(experiment, 90)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        out_path = experiment_dir / (
            f"cond_{condition_id:04d}"
            f"_Pch{_compact_num(row.get('chamber_pressure_bar'))}"
            f"_Pinj{_compact_num(row.get('injection_pressure_bar'))}"
            f"_dur{_compact_num(row.get('injection_duration_us'))}"
            f"_d{_compact_num(row.get('diameter_mm'))}.png"
        )

        x0 = sub["time_bin_start_ms"].to_numpy(dtype=float)
        x1 = sub["time_bin_end_ms"].to_numpy(dtype=float)
        x = 0.5 * (x0 + x1)
        t_min = float(np.nanmin(x0)) if len(x0) else 0.0
        t_max = float(np.nanmax(x1)) if len(x1) else 5.0
        bin_ms = float(np.nanmedian(x1 - x0)) if len(x0) else 0.1

        b_density = row.get("b_density_start", np.nan)
        b_fov = row.get("b_fov_start", np.nan)
        b_censor = row.get("b_censor_start", np.nan)
        density_t = t_min + float(b_density) * bin_ms if pd.notna(b_density) and int(b_density) < len(sub) else np.nan
        fov_t = t_min + float(b_fov) * bin_ms if pd.notna(b_fov) and int(b_fov) < len(sub) else np.nan
        censor_t = row.get("censor_start_time_ms", np.nan)
        if pd.isna(censor_t) and pd.notna(b_censor) and int(b_censor) < len(sub):
            censor_t = t_min + float(b_censor) * bin_ms

        fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), dpi=140, sharex=True)
        ax = axes[0]
        ax.plot(x, sub["n_traces"].to_numpy(dtype=float), color="#8ab6df", linewidth=1.1, label="n_traces")
        ax.plot(x, sub["n_traces_smooth"].to_numpy(dtype=float), color="#1f2d3d", linewidth=1.8, label="smoothed")
        if pd.notna(censor_t):
            ax.axvspan(float(censor_t), t_max, color="#e45756", alpha=0.10)
            _maybe_vline(ax, censor_t, "censor start", "#e45756", "-", 1.1)
        _maybe_vline(ax, density_t, "density start", "#f58518", "--", 1.0)
        _maybe_vline(ax, fov_t, "FOV start", "#54a24b", ":", 1.3)
        ax.set_ylabel("Trace count / bin")
        ax.set_title(_condition_title_multiline(row), fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=7, loc="upper right")

        ax = axes[1]
        point_sub = points_by_condition.get(condition_id)
        if point_sub is not None and not point_sub.empty:
            censored_mask = point_sub["is_right_censored"].astype(bool)
            uncensored = point_sub.loc[~censored_mask]
            censored = point_sub.loc[censored_mask]
            if not uncensored.empty:
                ax.scatter(
                    uncensored["time_ms"],
                    uncensored["penetration_mm"],
                    s=4.0,
                    alpha=0.22,
                    linewidths=0,
                    color="#4c78a8",
                    label="uncensored points",
                    zorder=1,
                )
            if not censored.empty:
                ax.scatter(
                    censored["time_ms"],
                    censored["penetration_mm"],
                    s=4.0,
                    alpha=0.22,
                    linewidths=0,
                    color="#e45756",
                    label="right-censored points",
                    zorder=1,
                )
        for col, color, linewidth, label in [
            ("pen_p50_mm", "#4c78a8", 1.2, "p50"),
            ("pen_p90_mm", "#72b7b2", 1.2, "p90"),
            ("pen_p95_mm", "#f58518", 1.4, "p95"),
            ("pen_max_mm", "#e45756", 1.0, "max"),
        ]:
            if col in sub.columns:
                ax.plot(x, sub[col].to_numpy(dtype=float), color=color, linewidth=linewidth, label=label, zorder=3)
        if pd.notna(row.get("fov_threshold_mm", np.nan)):
            ax.axhline(float(row["fov_threshold_mm"]), color="#54a24b", linestyle=":", linewidth=1.1, label="FOV threshold")
        if pd.notna(row.get("cap_mm", np.nan)):
            ax.axhline(float(row["cap_mm"]), color="#333333", linestyle="--", linewidth=0.9, label="cap")
        if pd.notna(censor_t):
            ax.axvspan(float(censor_t), t_max, color="#e45756", alpha=0.10)
            ax.axvline(float(censor_t), color="#e45756", linestyle="-", linewidth=1.1)
        _maybe_vline(ax, density_t, "density start", "#f58518", "--", 1.0)
        _maybe_vline(ax, fov_t, "FOV start", "#54a24b", ":", 1.3)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("CDF penetration [mm]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower right", ncol=2)

        fig.text(
            0.012,
            0.012,
            f"condition_id={condition_id} | reason={row.get('censor_start_reason', 'NA')} | "
            f"points={int(row.get('n_points', 0)):,} | traces={int(row.get('n_traces', 0)):,}",
            fontsize=8,
            color="#333333",
        )
        fig.tight_layout(rect=(0, 0.025, 1, 1))
        fig.savefig(out_path)
        plt.close(fig)

        index_rows.append(
            {
                "condition_id": condition_id,
                "experiment_name": experiment,
                "censor_start_reason": row.get("censor_start_reason", ""),
                "n_points": int(row.get("n_points", 0)),
                "n_traces": int(row.get("n_traces", 0)),
                "plot_path": str(out_path),
            }
        )
        if progress_every > 0 and len(index_rows) % int(progress_every) == 0:
            print(f"Generated {len(index_rows)} condition plots...")

    index = pd.DataFrame(index_rows).sort_values(["experiment_name", "condition_id"])
    index_path = condition_root / "condition_plot_index.csv"
    index.to_csv(index_path, index=False)
    return {
        "root": condition_root,
        "index": index_path,
        "n_condition_plots": int(len(index_rows)),
    }


def write_plots(
    *,
    out_dir: Path,
    points: pd.DataFrame,
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    plot_sample_points: int,
    condition_plots: bool = True,
    max_condition_plots: int | None = None,
    condition_plot_progress_every: int = 100,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}

    reason_counts = summary["censor_start_reason"].value_counts(dropna=False).sort_values()
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=150)
    ax.barh(reason_counts.index.astype(str), reason_counts.to_numpy(), color="#4c78a8")
    ax.set_xlabel("Condition count")
    ax.set_ylabel("")
    ax.set_title("Right-censoring start reason by condition")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    paths["censor_reason_counts"] = plot_dir / "censor_reason_counts.png"
    fig.savefig(paths["censor_reason_counts"])
    plt.close(fig)

    exp = (
        points.groupby("experiment_name", dropna=False)
        .agg(
            n_points=("penetration_mm", "size"),
            n_right_censored=("is_right_censored", "sum"),
        )
        .reset_index()
    )
    exp["right_censored_fraction"] = exp["n_right_censored"] / exp["n_points"].clip(lower=1)
    exp = exp.sort_values("right_censored_fraction")
    fig_h = max(4.0, 0.42 * len(exp) + 1.5)
    fig, ax = plt.subplots(figsize=(9.2, fig_h), dpi=150)
    ax.barh(exp["experiment_name"].map(lambda x: _short_text(x, 54)), exp["right_censored_fraction"], color="#e45756")
    ax.set_xlim(0.0, min(1.0, max(0.05, float(exp["right_censored_fraction"].max()) * 1.15)))
    ax.set_xlabel("Right-censored point fraction")
    ax.set_ylabel("")
    ax.set_title("Point-level right-censoring fraction by experiment")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    paths["right_censored_fraction_by_experiment"] = plot_dir / "right_censored_fraction_by_experiment.png"
    fig.savefig(paths["right_censored_fraction_by_experiment"])
    plt.close(fig)

    n_sample = min(int(plot_sample_points), len(points))
    sample = points.sample(n_sample, random_state=42) if n_sample < len(points) else points
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
    unc = sample.loc[~sample["is_right_censored"]]
    cen = sample.loc[sample["is_right_censored"]]
    ax.scatter(unc["time_ms"], unc["penetration_mm"], s=1.8, alpha=0.26, linewidths=0, color="#4c78a8", label=f"kept (n={len(unc):,})")
    ax.scatter(cen["time_ms"], cen["penetration_mm"], s=1.8, alpha=0.28, linewidths=0, color="#e45756", label=f"right-censored (n={len(cen):,})")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("CDF penetration [mm]")
    ax.set_title(f"CDF points after finite/time-window filtering (sample n={len(sample):,})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", markerscale=5)
    fig.tight_layout()
    paths["penetration_time_scatter"] = plot_dir / "penetration_time_scatter.png"
    fig.savefig(paths["penetration_time_scatter"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    bins_mm = np.linspace(
        0.0,
        float(np.nanquantile(points["penetration_mm"].to_numpy(dtype=float), 0.995)),
        80,
    )
    ax.hist(points.loc[~points["is_right_censored"], "penetration_mm"], bins=bins_mm, histtype="step", linewidth=1.4, color="#4c78a8", label="kept")
    ax.hist(points.loc[points["is_right_censored"], "penetration_mm"], bins=bins_mm, histtype="step", linewidth=1.4, color="#e45756", label="right-censored")
    ax.set_xlabel("CDF penetration [mm]")
    ax.set_ylabel("Point count")
    ax.set_title("Penetration distribution after censoring label")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    paths["penetration_histogram"] = plot_dir / "penetration_histogram.png"
    fig.savefig(paths["penetration_histogram"])
    plt.close(fig)

    examples = summary.sort_values(["n_points", "condition_id"], ascending=[False, True]).head(6)
    if len(examples):
        n = len(examples)
        ncols = 2
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.3 * ncols, 3.5 * nrows), dpi=145, squeeze=False)
        for ax, (_, row) in zip(axes.ravel(), examples.iterrows()):
            sub = bins.loc[bins["condition_id"] == row["condition_id"]].sort_values("time_bin")
            x = sub["time_bin_start_ms"].to_numpy(dtype=float)
            ax.plot(x, sub["n_traces"].to_numpy(dtype=float), color="#4c78a8", linewidth=1.0, alpha=0.6, label="n_traces")
            ax.plot(x, sub["n_traces_smooth"].to_numpy(dtype=float), color="#1f2d3d", linewidth=1.7, label="smoothed")
            cstart = row.get("censor_start_time_ms", np.nan)
            if pd.notna(cstart):
                ax.axvspan(float(cstart), float(sub["time_bin_end_ms"].max()), color="#e45756", alpha=0.10)
                ax.axvline(float(cstart), color="#e45756", linestyle="-", linewidth=1.1, label="censor start")
            dstart = row.get("b_density_start", np.nan)
            fstart = row.get("b_fov_start", np.nan)
            bin_ms = float(sub["time_bin_end_ms"].iloc[0] - sub["time_bin_start_ms"].iloc[0])
            if pd.notna(dstart) and int(dstart) < len(sub):
                ax.axvline(float(sub["time_bin_start_ms"].min()) + int(dstart) * bin_ms, color="#f58518", linestyle="--", linewidth=0.9, label="density")
            if pd.notna(fstart) and int(fstart) < len(sub):
                ax.axvline(float(sub["time_bin_start_ms"].min()) + int(fstart) * bin_ms, color="#54a24b", linestyle=":", linewidth=1.1, label="FOV")
            ax.set_title(_condition_title(row), fontsize=8.5)
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Trace count / bin")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="upper right")
        for ax in axes.ravel()[len(examples):]:
            ax.set_visible(False)
        fig.suptitle("Largest-condition time-bin density and censor-start examples", fontsize=12)
        fig.tight_layout()
        paths["time_bin_density_examples"] = plot_dir / "time_bin_density_examples.png"
        fig.savefig(paths["time_bin_density_examples"])
        plt.close(fig)

    if condition_plots:
        paths["conditions_by_experiment"] = write_condition_plots(
            plot_dir=plot_dir,
            points=points,
            bins=bins,
            summary=summary,
            plt=plt,
            max_condition_plots=max_condition_plots,
            progress_every=int(condition_plot_progress_every),
        )

    return paths


def jsonable_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable_paths(item) for item in value]
    return value


def write_outputs(
    *,
    out_dir: Path,
    points: pd.DataFrame,
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    trace_summary: pd.DataFrame,
    manifest: dict[str, Any],
    make_plots: bool = True,
    plot_sample_points: int = 120000,
    condition_plots: bool = True,
    max_condition_plots: int | None = None,
    condition_plot_progress_every: int = 100,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    points = points.loc[:, ordered_columns(points)]
    points.to_csv(out_dir / "cdf_points_all.csv", index=False)
    points.loc[~points["is_right_censored"]].to_csv(out_dir / "cdf_points_uncensored.csv", index=False)
    bins.to_csv(out_dir / "cdf_time_bin_censoring_by_condition.csv", index=False)
    summary.to_csv(out_dir / "cdf_condition_censoring_summary.csv", index=False)
    trace_summary.to_csv(out_dir / "cdf_trace_summary.csv", index=False)
    if make_plots:
        plot_paths = write_plots(
            out_dir=out_dir,
            points=points,
            bins=bins,
            summary=summary,
            plot_sample_points=int(plot_sample_points),
            condition_plots=bool(condition_plots),
            max_condition_plots=max_condition_plots,
            condition_plot_progress_every=int(condition_plot_progress_every),
        )
        manifest.setdefault("outputs", {})["plots"] = jsonable_paths(plot_paths)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CDF point-level right-censoring workflow from parsed arguments."""
    validate_args(args)
    condition_cols = parse_csv_list(args.condition_cols) or list(stage3.REGIME_GROUP_COLS)
    cap_group_cols = parse_csv_list(args.cap_group_cols) or ["experiment_name"]

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_OUTPUT_ROOT / f"cdf_right_censoring_points_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = out_dir.expanduser().resolve()

    wide = load_cdf_wide(args)
    points = expand_wide_to_points(
        wide,
        t_min_ms=float(args.t_min_ms),
        t_max_ms=float(args.t_max_ms),
        bin_ms=float(args.bin_ms),
        condition_cols=condition_cols,
    )
    points, trace_summary = attach_fov_caps(
        points,
        estimate_fov_cap=bool(args.estimate_fov_cap),
        cap_group_cols=cap_group_cols,
        cap_percentile=float(args.cap_percentile),
        fov_cap_mm=float(args.fov_cap_mm),
    )
    bins, summary = compute_condition_bin_table(
        points,
        condition_cols=condition_cols,
        t_min_ms=float(args.t_min_ms),
        t_max_ms=float(args.t_max_ms),
        bin_ms=float(args.bin_ms),
        fov_stat=str(args.fov_stat),
        fov_cap_fraction=float(args.fov_cap_fraction),
        fov_min_count=int(args.fov_min_count),
        fov_consecutive_bins=int(args.fov_consecutive_bins),
        density_ratio=float(args.density_ratio),
        density_min_count=int(args.density_min_count),
        density_consecutive_bins=int(args.density_consecutive_bins),
        density_smooth_window=int(args.density_smooth_window),
    )
    labeled_points = attach_bin_labels(points, bins)

    n_points = int(len(labeled_points))
    n_censored = int(labeled_points["is_right_censored"].sum())
    manifest = {
        "script": str(Path(__file__).resolve()),
        "synthetic_root": str(args.synthetic_root.expanduser().resolve()),
        "source": args.source,
        "split": args.split,
        "t_window_ms": [float(args.t_min_ms), float(args.t_max_ms)],
        "bin_ms": float(args.bin_ms),
        "condition_cols": condition_cols,
        "cap_group_cols": cap_group_cols,
        "estimate_fov_cap": bool(args.estimate_fov_cap),
        "fov_cap_mm": float(args.fov_cap_mm),
        "cap_percentile": float(args.cap_percentile),
        "fov_cap_fraction": float(args.fov_cap_fraction),
        "fov_stat": args.fov_stat,
        "density_ratio": float(args.density_ratio),
        "density_min_count": int(args.density_min_count),
        "density_consecutive_bins": int(args.density_consecutive_bins),
        "density_smooth_window": int(args.density_smooth_window),
        "n_wide_rows": int(len(wide)),
        "n_points_all": n_points,
        "n_points_right_censored": n_censored,
        "n_points_uncensored": int(n_points - n_censored),
        "right_censored_fraction": float(n_censored / n_points) if n_points else math.nan,
        "n_conditions": int(summary["condition_id"].nunique()) if len(summary) else 0,
        "plots_requested": bool(args.plots),
        "plot_sample_points": int(args.plot_sample_points),
        "condition_plots_requested": bool(args.plots and args.condition_plots),
        "max_condition_plots": None if args.max_condition_plots is None else int(args.max_condition_plots),
        "condition_plot_progress_every": int(args.condition_plot_progress_every),
        "outputs": {
            "points_all": str(out_dir / "cdf_points_all.csv"),
            "points_uncensored": str(out_dir / "cdf_points_uncensored.csv"),
            "bin_table": str(out_dir / "cdf_time_bin_censoring_by_condition.csv"),
            "condition_summary": str(out_dir / "cdf_condition_censoring_summary.csv"),
            "trace_summary": str(out_dir / "cdf_trace_summary.csv"),
        },
    }
    write_outputs(
        out_dir=out_dir,
        points=labeled_points,
        bins=bins,
        summary=summary,
        trace_summary=trace_summary,
        manifest=manifest,
        make_plots=bool(args.plots),
        plot_sample_points=int(args.plot_sample_points),
        condition_plots=bool(args.condition_plots),
        max_condition_plots=None if args.max_condition_plots is None else int(args.max_condition_plots),
        condition_plot_progress_every=int(args.condition_plot_progress_every),
    )

    result = {
        "n_wide_rows": manifest["n_wide_rows"],
        "n_points_all": manifest["n_points_all"],
        "n_points_uncensored": manifest["n_points_uncensored"],
        "n_points_right_censored": manifest["n_points_right_censored"],
        "right_censored_fraction": manifest["right_censored_fraction"],
        "n_conditions": manifest["n_conditions"],
        "out_dir": str(out_dir),
    }
    return result


def run_cdf_censoring_points(
    *,
    synthetic_root: Path,
    out_dir: Path,
    source: str = DEFAULT_SOURCE,
    split: str = DEFAULT_SPLIT,
    t_min_ms: float = DEFAULT_T_MIN_MS,
    t_max_ms: float = DEFAULT_T_MAX_MS,
    bin_ms: float = DEFAULT_BIN_MS,
    make_plots: bool = True,
    condition_plots: bool = True,
) -> dict[str, Any]:
    """Programmatic entry point used by ``fit_raw_data.py``."""
    args = argparse.Namespace(
        synthetic_root=synthetic_root,
        source=source,
        split=split,
        out_dir=out_dir,
        datasets=None,
        folders=None,
        max_wide_rows=None,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
        bin_ms=bin_ms,
        condition_cols=",".join(stage3.REGIME_GROUP_COLS),
        cap_group_cols="experiment_name",
        estimate_fov_cap=False,
        cap_percentile=99.0,
        fov_cap_mm=DEFAULT_FOV_CAP_MM,
        fov_cap_fraction=DEFAULT_FOV_CAP_FRACTION,
        fov_stat=DEFAULT_FOV_STAT,
        fov_min_count=DEFAULT_DENSITY_MIN_COUNT,
        fov_consecutive_bins=1,
        density_ratio=DEFAULT_DENSITY_RATIO,
        density_min_count=DEFAULT_DENSITY_MIN_COUNT,
        density_consecutive_bins=DEFAULT_DENSITY_CONSECUTIVE_BINS,
        density_smooth_window=DEFAULT_DENSITY_SMOOTH_WINDOW,
        plots=make_plots,
        condition_plots=condition_plots,
        plot_sample_points=120000,
        max_condition_plots=None,
        condition_plot_progress_every=100,
    )
    return run_from_args(args)


def main() -> None:
    result = run_from_args(parse_args())
    print(f"Wrote CDF right-censoring point tables to: {result['out_dir']}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
