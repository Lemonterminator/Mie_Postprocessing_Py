from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import fit_raw_data as frd


CDF_SOURCE = next(source for source in frd.PENETRATION_SOURCES if source["key"] == "cdf")
DEFAULT_SYNTHETIC_ROOT = frd.data_out_dir
DEFAULT_OUT_DIR = THIS_DIR / "figures" / "fit_bias_audit_cdf"
TIME_BIN_MS = 0.10
MAX_TIME_MS = 5.0
RELATIVE_EPS = 1e-9
SELECTION_METRICS = ("n", "t_max_ms", "penetration_far_mm", "rmse")
FILTER_STAGE_COLS = (
    "delay_alignment_removed_count",
    "pre_onset_removed_count",
    "lower_cut_removed_count",
    "upper_cut_removed_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit CDF preprocessing and sigmoid-fit bias using raw experiment CSVs, "
            "processed synthetic_data tables, and stored fit parameters."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset directory name under MLP/synthetic_data. Can be passed multiple times.",
    )
    parser.add_argument(
        "--folder",
        action="append",
        dest="folders",
        help="Folder stem such as T6 or T13. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=8,
        help="Maximum number of representative plume panels to save.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for audit CSVs and figures.",
    )
    return parser.parse_args()


def infer_nozzle(dataset_name: str) -> str:
    match = re.search(r"Nozzle(\d+)", dataset_name)
    if match:
        return f"Nozzle {match.group(1)}"
    return "Other"


def shift_left(arr: np.ndarray, shift: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    shift = int(shift)
    if shift <= 0:
        out[:] = arr
        return out
    if shift < arr.size:
        out[:-shift] = arr[shift:]
    return out


def shift_by_delta(arr: np.ndarray, delta: int) -> np.ndarray:
    delta = int(delta)
    out = np.full_like(arr, np.nan, dtype=float)
    if delta > 0:
        if delta < arr.size:
            out[delta:] = arr[:-delta]
        return out
    if delta < 0:
        shift = -delta
        if shift < arr.size:
            out[:-shift] = arr[shift:]
        return out
    out[:] = arr
    return out


def safe_quantile(values: pd.Series, q: float) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(q))


def safe_mean(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return float(s.mean())


def safe_median(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return float(s.median())


def folder_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(value))
    if match:
        return int(match.group(1)), str(value)
    return math.inf, str(value)


def extract_wide_series(row: pd.Series, time_cols: list[str], pen_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    time_values = row.loc[time_cols].to_numpy(dtype=float)
    pen_values = row.loc[pen_cols].to_numpy(dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(pen_values)
    return time_values[valid], pen_values[valid]


def build_inventory(
    synthetic_root: Path,
    *,
    datasets: list[str] | None,
    folders: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_frames: list[pd.DataFrame] = []
    wide_frames: list[pd.DataFrame] = []
    dataset_filter = set(datasets or [])
    folder_filter = set(folders or [])

    for dataset_dir in sorted(synthetic_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name not in dataset_filter:
            continue

        cdf_dir = dataset_dir / "cdf"
        all_dir = cdf_dir / "all"
        wide_dir = cdf_dir / "series_wide_all"
        if not all_dir.exists() or not wide_dir.exists():
            continue

        for fit_path in sorted(all_dir.glob("*.csv")):
            if fit_path.name.endswith("_flagged.csv"):
                continue
            folder_name = fit_path.stem
            if folder_filter and folder_name not in folder_filter:
                continue
            wide_path = wide_dir / f"{folder_name}.csv"
            if not wide_path.exists():
                print(f"Skip {fit_path}: missing wide series file {wide_path.name}")
                continue

            fit_df = pd.read_csv(fit_path)
            fit_df["dataset"] = dataset_dir.name
            fit_df["folder"] = folder_name
            fit_df["nozzle"] = fit_df["dataset"].map(infer_nozzle)
            fit_frames.append(fit_df)

            wide_df = pd.read_csv(wide_path)
            wide_df["dataset"] = dataset_dir.name
            wide_df["folder"] = folder_name
            wide_df["nozzle"] = wide_df["dataset"].map(infer_nozzle)
            wide_frames.append(wide_df)

    if not fit_frames:
        raise FileNotFoundError("No CDF fit tables found for the requested dataset/folder filters.")

    fit_df = pd.concat(fit_frames, ignore_index=True)
    wide_df = pd.concat(wide_frames, ignore_index=True)
    fit_df["file_path"] = fit_df["file_path"].astype(str)
    wide_df["file_path"] = wide_df["file_path"].astype(str)
    fit_df["sample_split"] = np.where(fit_df["flag_bad_fit"].fillna(False), "flagged", "clean")
    fit_df["t_max_ms"] = pd.to_numeric(fit_df["t_max_s"], errors="coerce") * 1e3
    return fit_df, wide_df


def load_file_bundle(file_path: str, dataset: str, cache: dict[str, dict]) -> dict:
    cache_key = str(Path(file_path).resolve())
    if cache_key in cache:
        return cache[cache_key]

    csv_path = Path(file_path)
    df_file = frd._read_csv_with_expanded_static_meta(csv_path)
    settings = frd.get_dataset_settings(dataset)
    mm_per_px_scale = 90.0 / settings["or_mm_per_px_reference"]
    time_s, time_ms, cleaned_series, delays_raw, delays_used, delay_sources = frd.prepare_cleaned_series(
        df_file,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=settings["fps_default"],
        max_hydraulic_delay_frames=settings["max_hydraulic_delay_frames"],
        delay_clip_half_window=settings["delay_clip_half_window"],
        penetration_column_prefix=CDF_SOURCE["column_prefix"],
        replace_negative_with_zero=CDF_SOURCE["replace_negative_with_zero"],
        diff_threshold_lower=frd.DIFF_THRESHOLD_LOWER,
        diff_threshold_upper=frd.DIFF_THRESHOLD_UPPER,
    )
    bundle = {
        "csv_path": csv_path,
        "df_file": df_file,
        "settings": settings,
        "mm_per_px_scale": mm_per_px_scale,
        "time_s": np.asarray(time_s, dtype=float),
        "time_ms": np.asarray(time_ms, dtype=float),
        "cleaned_series": np.asarray(cleaned_series, dtype=float),
        "delays_raw": np.asarray(delays_raw, dtype=float),
        "delays_used": np.asarray(delays_used, dtype=float),
        "delay_sources": np.asarray(delay_sources, dtype=object),
    }
    cache[cache_key] = bundle
    return bundle


def trace_cdf_pipeline(bundle: dict, plume_idx: int) -> dict:
    df_file = bundle["df_file"]
    col = f"{CDF_SOURCE['column_prefix']}{plume_idx}"
    if col in df_file.columns:
        raw_original_px = pd.to_numeric(df_file[col], errors="coerce").to_numpy(dtype=float)
    else:
        raw_original_px = np.full(bundle["time_s"].shape, np.nan, dtype=float)
    umbrella_angle_deg = float(pd.to_numeric(df_file["umbrella_angle_deg"].iloc[0], errors="coerce"))
    if not np.isfinite(umbrella_angle_deg):
        umbrella_angle_deg = 180.0
    tilt_ang = (180.0 - umbrella_angle_deg) / 2.0
    umbrella_angle_correction = 1.0 / np.cos(np.deg2rad(tilt_ang))
    pen_correction = bundle["mm_per_px_scale"] * umbrella_angle_correction

    raw_delay = int(np.round(bundle["delays_raw"][plume_idx])) if np.isfinite(bundle["delays_raw"][plume_idx]) else 0
    used_delay = int(np.round(bundle["delays_used"][plume_idx])) if np.isfinite(bundle["delays_used"][plume_idx]) else 0

    raw_aligned_px = shift_left(raw_original_px, used_delay)

    arr = shift_left(raw_original_px, raw_delay)
    arr_after_pre = arr.copy()
    positive_idx = np.flatnonzero(np.isfinite(arr_after_pre) & (arr_after_pre > 0))
    if positive_idx.size == 0:
        pre_onset_removed_count = int(np.isfinite(arr_after_pre).sum())
        arr_after_pre[:] = np.nan
        lower_cut_removed_count = 0
        upper_cut_removed_count = 0
        first_positive_idx = -1
        arr = arr_after_pre
    else:
        first_positive_idx = int(positive_idx[0])
        pre_onset_removed_count = int(np.isfinite(arr_after_pre[:first_positive_idx]).sum())
        if first_positive_idx > 0:
            arr_after_pre[:first_positive_idx] = np.nan

        arr_after_lower = arr_after_pre.copy()
        lower_cut_removed_count = 0
        if frd.ENABLE_DIFF_THRESHOLD_LOWER:
            arr_diff = np.diff(arr_after_lower[first_positive_idx:])
            lower_cut_idx = np.where(arr_diff < frd.DIFF_THRESHOLD_LOWER)[0]
            if lower_cut_idx.size > 0:
                cut = first_positive_idx + int(lower_cut_idx[0]) + 1
                lower_cut_removed_count = int(np.isfinite(arr_after_lower[cut:]).sum())
                arr_after_lower[cut:] = np.nan

        arr_after_upper = arr_after_lower.copy()
        upper_cut_removed_count = 0
        if frd.ENABLE_DIFF_THRESHOLD_UPPER:
            valid_positive_idx = np.flatnonzero(np.isfinite(arr_after_upper) & (arr_after_upper > 0))
            if valid_positive_idx.size == 0:
                arr_after_upper[:] = np.nan
            else:
                first_valid_idx = int(valid_positive_idx[0])
                arr_diff = np.diff(arr_after_upper[first_valid_idx:])
                upper_cut_idx = np.where(arr_diff > frd.DIFF_THRESHOLD_UPPER)[0]
                if upper_cut_idx.size > 0:
                    cut = first_valid_idx + int(upper_cut_idx[0]) + 1
                    upper_cut_removed_count = int(np.isfinite(arr_after_upper[cut:]).sum())
                    arr_after_upper[cut:] = np.nan
        arr = arr_after_upper

    processed_trace_mm = np.asarray(bundle["cleaned_series"][plume_idx], dtype=float).copy()

    return {
        "raw_original_mm": raw_original_px * pen_correction,
        "raw_aligned_mm": raw_aligned_px * pen_correction,
        "processed_trace_mm": processed_trace_mm,
        "delay_frames_raw": raw_delay,
        "delay_frames_used": used_delay,
        "delay_clip_delta_frames": raw_delay - used_delay,
        "original_finite_count": int(np.isfinite(raw_original_px).sum()),
        "raw_aligned_finite_count": int(np.isfinite(raw_aligned_px).sum()),
        "delay_alignment_removed_count": int(max(np.isfinite(raw_original_px).sum() - np.isfinite(raw_aligned_px).sum(), 0)),
        "pre_onset_removed_count": int(pre_onset_removed_count),
        "lower_cut_removed_count": int(lower_cut_removed_count),
        "upper_cut_removed_count": int(upper_cut_removed_count),
        "processed_finite_count": int(np.isfinite(processed_trace_mm).sum()),
        "first_positive_idx_after_raw_delay": int(first_positive_idx),
    }


def compare_trace_to_wide(
    bundle: dict,
    trace: dict,
    wide_row: pd.Series | None,
    time_cols: list[str],
    pen_cols: list[str],
) -> dict:
    trace_time_ms = bundle["time_ms"]
    trace_processed_mm = trace["processed_trace_mm"]
    trace_valid = np.isfinite(trace_time_ms) & np.isfinite(trace_processed_mm)
    trace_time = trace_time_ms[trace_valid]
    trace_pen = trace_processed_mm[trace_valid]

    if wide_row is None:
        wide_time = np.asarray([], dtype=float)
        wide_pen = np.asarray([], dtype=float)
    else:
        wide_time, wide_pen = extract_wide_series(wide_row, time_cols, pen_cols)
    lengths_match = int(trace_time.size == wide_time.size)
    time_match = bool(lengths_match and np.allclose(trace_time, wide_time, atol=1e-9, rtol=0.0))
    pen_max_abs_diff = np.nan
    if lengths_match and trace_pen.size:
        pen_max_abs_diff = float(np.nanmax(np.abs(trace_pen - wide_pen)))
    processed_match = bool(lengths_match and time_match and (np.isnan(pen_max_abs_diff) or pen_max_abs_diff < 1e-9))
    return {
        "trace_time_ms": trace_time,
        "trace_penetration_mm": trace_pen,
        "wide_time_ms": wide_time,
        "wide_penetration_mm": wide_pen,
        "processed_matches_series_wide": processed_match,
        "series_wide_len": int(wide_time.size),
        "processed_match_max_abs_diff": pen_max_abs_diff,
    }


def build_fit_curve(trace_time_ms: np.ndarray, fit_row: pd.Series) -> np.ndarray:
    if trace_time_ms.size == 0:
        return np.asarray([], dtype=float)
    log_params = [
        float(fit_row["log_k_sqrt"]),
        float(fit_row["log_k_quarter"]),
        float(fit_row["log_t0"]),
        float(fit_row["log_s"]),
    ]
    return frd.spray_penetration_model_sigmoid(log_params, trace_time_ms * 1e-3)


def assemble_audit_tables(
    fit_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_indexed = wide_df.set_index(["file_path", "plume_idx"], drop=False)
    time_cols = sorted([col for col in wide_df.columns if col.startswith("time_ms_")], key=folder_sort_key)
    pen_cols = sorted([col for col in wide_df.columns if col.startswith("penetration_mm_")], key=folder_sort_key)

    audit_rows: list[dict] = []
    processed_vs_fit_rows: list[dict] = []
    file_cache: dict[str, dict] = {}

    for row in fit_df.itertuples(index=False):
        row_dict = row._asdict()
        bundle = load_file_bundle(row.file_path, row.dataset, file_cache)
        trace = trace_cdf_pipeline(bundle, int(row.plume_idx))

        wide_key = (row.file_path, row.plume_idx)
        wide_row = None
        if wide_key in wide_indexed.index:
            wide_row = wide_indexed.loc[wide_key]
            if isinstance(wide_row, pd.DataFrame):
                wide_row = wide_row.iloc[0]
        compare = compare_trace_to_wide(bundle, trace, wide_row, time_cols, pen_cols)

        fit_curve = build_fit_curve(compare["trace_time_ms"], pd.Series(row_dict))
        rmse_recomputed = np.nan
        rmse_abs_diff = np.nan
        relative_residual_mean = np.nan
        max_abs_residual = np.nan
        if compare["trace_penetration_mm"].size:
            residual = fit_curve - compare["trace_penetration_mm"]
            rmse_recomputed = float(np.sqrt(np.mean(np.square(residual))))
            rmse_abs_diff = float(abs(rmse_recomputed - float(row.rmse))) if np.isfinite(row.rmse) else np.nan
            relative = np.where(
                np.abs(compare["trace_penetration_mm"]) > RELATIVE_EPS,
                residual / compare["trace_penetration_mm"],
                np.nan,
            )
            relative_residual_mean = safe_mean(pd.Series(relative))
            max_abs_residual = float(np.nanmax(np.abs(residual))) if np.isfinite(residual).any() else np.nan

            for time_ms_value, processed_mm_value, fit_mm_value in zip(
                compare["trace_time_ms"],
                compare["trace_penetration_mm"],
                fit_curve,
            ):
                residual_mm = float(fit_mm_value - processed_mm_value)
                relative_residual = (
                    residual_mm / processed_mm_value if abs(processed_mm_value) > RELATIVE_EPS else np.nan
                )
                processed_vs_fit_rows.append(
                    {
                        "dataset": row.dataset,
                        "folder": row.folder,
                        "nozzle": row.nozzle,
                        "file_path": row.file_path,
                        "file_name": row.file_name,
                        "plume_idx": int(row.plume_idx),
                        "sample_split": row.sample_split,
                        "delay_source": row.delay_source,
                        "injection_pressure_bar": row.injection_pressure_bar,
                        "chamber_pressure_bar": row.chamber_pressure_bar,
                        "time_ms": float(time_ms_value),
                        "time_bin_start_ms": float(np.floor(time_ms_value / TIME_BIN_MS) * TIME_BIN_MS),
                        "processed_penetration_mm": float(processed_mm_value),
                        "fit_penetration_mm": float(fit_mm_value),
                        "residual_mm": residual_mm,
                        "relative_residual": float(relative_residual) if np.isfinite(relative_residual) else np.nan,
                    }
                )

        fps = float(pd.to_numeric(bundle["df_file"]["fps"].iloc[0], errors="coerce"))
        time_shift_ms = trace["delay_frames_used"] / fps * 1e3 if np.isfinite(fps) and fps > 0 else 0.0
        audit_rows.append(
            {
                **row_dict,
                "delay_frames_raw_trace": trace["delay_frames_raw"],
                "delay_frames_used_trace": trace["delay_frames_used"],
                "delay_clip_delta_frames": trace["delay_clip_delta_frames"],
                "original_finite_count": trace["original_finite_count"],
                "raw_aligned_finite_count": trace["raw_aligned_finite_count"],
                "delay_alignment_removed_count": trace["delay_alignment_removed_count"],
                "pre_onset_removed_count": trace["pre_onset_removed_count"],
                "lower_cut_removed_count": trace["lower_cut_removed_count"],
                "upper_cut_removed_count": trace["upper_cut_removed_count"],
                "processed_finite_count_trace": trace["processed_finite_count"],
                "processed_matches_series_wide": compare["processed_matches_series_wide"],
                "processed_match_max_abs_diff": compare["processed_match_max_abs_diff"],
                "series_wide_len": compare["series_wide_len"],
                "rmse_recomputed": rmse_recomputed,
                "rmse_abs_diff": rmse_abs_diff,
                "relative_residual_mean": relative_residual_mean,
                "max_abs_residual_mm": max_abs_residual,
                "aligned_time_shift_ms": float(time_shift_ms),
            }
        )

    audit_df = pd.DataFrame(audit_rows).replace([np.inf, -np.inf], np.nan)
    processed_vs_fit_df = pd.DataFrame(processed_vs_fit_rows).replace([np.inf, -np.inf], np.nan)
    return audit_df, processed_vs_fit_df


def summarize_selection(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    split_specs = [
        ("all", df),
        ("clean", df.loc[df["sample_split"] == "clean"]),
        ("flagged", df.loc[df["sample_split"] == "flagged"]),
    ]
    for split_name, split_df in split_specs:
        if group_cols:
            iterator = split_df.groupby(group_cols, dropna=False)
        else:
            iterator = [((), split_df)]
        for group_key, group_df in iterator:
            row = {"sample_split": split_name, "n_samples": int(len(group_df))}
            if group_cols:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                row.update({col: value for col, value in zip(group_cols, group_key)})
            for metric in SELECTION_METRICS:
                row[f"{metric}_mean"] = safe_mean(group_df[metric])
                row[f"{metric}_median"] = safe_median(group_df[metric])
                row[f"{metric}_p10"] = safe_quantile(group_df[metric], 0.10)
                row[f"{metric}_p90"] = safe_quantile(group_df[metric], 0.90)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_delay_source(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    split_specs = [
        ("all", df),
        ("clean", df.loc[df["sample_split"] == "clean"]),
        ("flagged", df.loc[df["sample_split"] == "flagged"]),
    ]
    for split_name, split_df in split_specs:
        if split_df.empty:
            continue
        if group_cols:
            iterator = split_df.groupby(group_cols, dropna=False)
        else:
            iterator = [((), split_df)]
        for group_key, group_df in iterator:
            counts = group_df["delay_source"].value_counts(dropna=False)
            total = counts.sum()
            for delay_source, count in counts.items():
                row = {
                    "sample_split": split_name,
                    "delay_source": delay_source,
                    "n_samples": int(count),
                    "fraction": float(count / total) if total else np.nan,
                }
                if group_cols:
                    if not isinstance(group_key, tuple):
                        group_key = (group_key,)
                    row.update({col: value for col, value in zip(group_cols, group_key)})
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_filter_stages(audit_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    split_specs = [
        ("all", audit_df),
        ("clean", audit_df.loc[audit_df["sample_split"] == "clean"]),
        ("flagged", audit_df.loc[audit_df["sample_split"] == "flagged"]),
    ]
    for split_name, split_df in split_specs:
        if group_cols:
            iterator = split_df.groupby(group_cols, dropna=False)
        else:
            iterator = [((), split_df)]
        for group_key, group_df in iterator:
            base = {"sample_split": split_name, "n_samples": int(len(group_df))}
            if group_cols:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                base.update({col: value for col, value in zip(group_cols, group_key)})
            base["original_finite_count_mean"] = safe_mean(group_df["original_finite_count"])
            for stage_col in FILTER_STAGE_COLS:
                base[f"{stage_col}_mean"] = safe_mean(group_df[stage_col])
                base[f"{stage_col}_median"] = safe_median(group_df[stage_col])
                base[f"{stage_col}_affected_frac"] = float((pd.to_numeric(group_df[stage_col], errors="coerce") > 0).mean())
                denom = group_df["original_finite_count"].replace(0, np.nan)
                frac = pd.to_numeric(group_df[stage_col], errors="coerce") / denom
                base[f"{stage_col}_fraction_mean"] = safe_mean(frac)
            rows.append(base)
    return pd.DataFrame(rows)


def summarize_time_bins(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg_group_cols = list(group_cols) + ["time_bin_start_ms"]
    if df.empty:
        return pd.DataFrame(columns=agg_group_cols)

    grouped = df.groupby(agg_group_cols, dropna=False)
    summary = grouped.agg(
        n_points=("time_ms", "size"),
        processed_mean=("processed_penetration_mm", "mean"),
        processed_median=("processed_penetration_mm", "median"),
        processed_std=("processed_penetration_mm", "std"),
        fit_mean=("fit_penetration_mm", "mean"),
        fit_median=("fit_penetration_mm", "median"),
        fit_std=("fit_penetration_mm", "std"),
        residual_mean=("residual_mm", "mean"),
        residual_median=("residual_mm", "median"),
        residual_std=("residual_mm", "std"),
        relative_residual_mean=("relative_residual", "mean"),
        relative_residual_median=("relative_residual", "median"),
    ).reset_index()

    for column, source in (
        ("processed_p10", "processed_penetration_mm"),
        ("processed_p90", "processed_penetration_mm"),
        ("fit_p10", "fit_penetration_mm"),
        ("fit_p90", "fit_penetration_mm"),
        ("residual_p10", "residual_mm"),
        ("residual_p90", "residual_mm"),
        ("relative_residual_p10", "relative_residual"),
        ("relative_residual_p90", "relative_residual"),
    ):
        q = 0.10 if column.endswith("p10") else 0.90
        extra = grouped[source].quantile(q).reset_index(name=column)
        summary = summary.merge(extra, on=agg_group_cols, how="left")
    return summary


def plot_selection_bias(audit_df: pd.DataFrame, delay_summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    metric_specs = [
        ("n", "Sequence length n"),
        ("rmse", "RMSE (mm)"),
        ("penetration_far_mm", "Penetration at 5 ms (mm)"),
        ("t_max_ms", "Max available time (ms)"),
    ]
    for ax, (metric, title) in zip(axes[:4], metric_specs):
        values_by_split = [
            pd.to_numeric(audit_df.loc[audit_df["sample_split"] == split, metric], errors="coerce").dropna().to_numpy()
            for split in ("clean", "flagged")
        ]
        box = ax.boxplot(values_by_split, tick_labels=("clean", "flagged"), patch_artist=True)
        for patch, color in zip(box["boxes"], ("#55a868", "#c44e52")):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    ax_delay = axes[4]
    delay_pivot = (
        delay_summary.loc[delay_summary["sample_split"].isin(("all", "clean", "flagged"))]
        .pivot(index="sample_split", columns="delay_source", values="fraction")
        .reindex(("all", "clean", "flagged"))
        .fillna(0.0)
    )
    bottom = np.zeros(len(delay_pivot), dtype=float)
    for delay_source in delay_pivot.columns:
        values = delay_pivot[delay_source].to_numpy(dtype=float)
        ax_delay.bar(delay_pivot.index, values, bottom=bottom, label=str(delay_source))
        bottom += values
    ax_delay.set_ylim(0.0, 1.0)
    ax_delay.set_title("Delay-source composition")
    ax_delay.set_ylabel("Fraction")
    ax_delay.legend()
    ax_delay.grid(alpha=0.25)

    ax_reason = axes[5]
    reason_df = pd.DataFrame(
        {
            "mask_basic_false": (~audit_df["mask_basic"].fillna(False)).astype(float),
            "mask_penetration_far_false": (~audit_df["mask_penetration_far"].fillna(False)).astype(float),
            "mask_outlier_true": audit_df["mask_outlier"].fillna(False).astype(float),
        }
    )
    reason_by_split = (
        pd.concat([audit_df[["sample_split"]], reason_df], axis=1)
        .groupby("sample_split", dropna=False)
        .mean()
        .reindex(("clean", "flagged"))
        .fillna(0.0)
    )
    x = np.arange(len(reason_by_split.index))
    width = 0.24
    for idx, col in enumerate(reason_by_split.columns):
        ax_reason.bar(x + (idx - 1) * width, reason_by_split[col].to_numpy(), width=width, label=col)
    ax_reason.set_xticks(x)
    ax_reason.set_xticklabels(reason_by_split.index)
    ax_reason.set_ylim(0.0, 1.0)
    ax_reason.set_title("Failure flags by split")
    ax_reason.set_ylabel("Fraction of plumes")
    ax_reason.legend(fontsize=8)
    ax_reason.grid(alpha=0.25)

    fig.suptitle("CDF fit selection bias: all vs clean vs flagged", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_filter_stage_audit(audit_df: pd.DataFrame, out_path: Path) -> None:
    stage_labels = {
        "delay_alignment_removed_count": "Delay shift",
        "pre_onset_removed_count": "Leading pre-onset",
        "lower_cut_removed_count": "Lower diff cut",
        "upper_cut_removed_count": "Upper diff cut",
    }
    summary = summarize_filter_stages(audit_df, [])
    summary = summary.loc[summary["sample_split"].isin(("clean", "flagged"))].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(stage_labels), dtype=float)
    width = 0.35
    for idx, split in enumerate(("clean", "flagged")):
        row = summary.loc[summary["sample_split"] == split]
        if row.empty:
            continue
        fraction_values = [float(row[f"{stage}_fraction_mean"].iloc[0]) for stage in stage_labels]
        affected_values = [float(row[f"{stage}_affected_frac"].iloc[0]) for stage in stage_labels]
        axes[0].bar(x + (idx - 0.5) * width, fraction_values, width=width, label=split)
        axes[1].bar(x + (idx - 0.5) * width, affected_values, width=width, label=split)

    for ax, title, ylabel in (
        (axes[0], "Mean removed fraction per stage", "Removed fraction vs original finite count"),
        (axes[1], "Fraction of plumes affected by stage", "Affected plume fraction"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(list(stage_labels.values()), rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("CDF filter-stage audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_processed_vs_fit(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    time_ms = summary_df["time_bin_start_ms"].to_numpy(dtype=float)

    axes[0].plot(time_ms, summary_df["processed_mean"], label="processed mean", color="#222222", linewidth=2.0)
    axes[0].fill_between(
        time_ms,
        summary_df["processed_p10"].to_numpy(dtype=float),
        summary_df["processed_p90"].to_numpy(dtype=float),
        color="#222222",
        alpha=0.15,
        label="processed 10-90%",
    )
    axes[0].plot(time_ms, summary_df["fit_mean"], label="fit mean", color="#4c72b0", linewidth=2.0)
    axes[0].fill_between(
        time_ms,
        summary_df["fit_p10"].to_numpy(dtype=float),
        summary_df["fit_p90"].to_numpy(dtype=float),
        color="#4c72b0",
        alpha=0.20,
        label="fit 10-90%",
    )
    axes[0].set_title("Processed series vs reconstructed fit")
    axes[0].set_xlabel("Time bin start (ms)")
    axes[0].set_ylabel("Penetration (mm)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(time_ms, summary_df["residual_median"], color="#c44e52", linewidth=2.0, label="Residual median")
    axes[1].fill_between(
        time_ms,
        summary_df["residual_p10"].to_numpy(dtype=float),
        summary_df["residual_p90"].to_numpy(dtype=float),
        color="#c44e52",
        alpha=0.20,
        label="Residual 10-90%",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_title("Residual fit - processed by time bin")
    axes[1].set_xlabel("Time bin start (ms)")
    axes[1].set_ylabel("Residual (mm)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("CDF processed-vs-fit distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def select_representative_cases(audit_df: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    preferred = audit_df.loc[
        np.isclose(pd.to_numeric(audit_df["injection_pressure_bar"], errors="coerce"), 2000.0)
        & np.isclose(pd.to_numeric(audit_df["chamber_pressure_bar"], errors="coerce"), 5.0)
    ].copy()

    def pick(df: pd.DataFrame, picked_keys: set[tuple[str, int]], limit: int) -> list[dict]:
        rows: list[dict] = []
        remaining = (
            df.sort_values(
                by=["flag_bad_fit", "rmse", "n"],
                ascending=[False, False, False],
                kind="stable",
            )
            .drop_duplicates(subset=["dataset", "folder", "file_path", "plume_idx"])
        )
        for _, row in remaining.iterrows():
            key = (str(row["file_path"]), int(row["plume_idx"]))
            if key in picked_keys:
                continue
            rows.append(row.to_dict())
            picked_keys.add(key)
            if len(rows) >= limit:
                break
        return rows

    picked: list[dict] = []
    picked_keys: set[tuple[str, int]] = set()

    if not preferred.empty:
        dataset_first = preferred.sort_values(by=["flag_bad_fit", "rmse"], ascending=[False, False]).groupby("dataset", dropna=False).head(1)
        for _, row in dataset_first.iterrows():
            key = (str(row["file_path"]), int(row["plume_idx"]))
            if key in picked_keys:
                continue
            picked.append(row.to_dict())
            picked_keys.add(key)
            if len(picked) >= max_cases:
                return pd.DataFrame(picked)

    if len(picked) < max_cases and not preferred.empty:
        picked.extend(pick(preferred, picked_keys, max_cases - len(picked)))
    if len(picked) < max_cases:
        picked.extend(pick(audit_df, picked_keys, max_cases - len(picked)))
    return pd.DataFrame(picked)


def plot_representative_cases(
    cases_df: pd.DataFrame,
    out_path: Path,
    file_cache: dict[str, dict],
) -> None:
    if cases_df.empty:
        return

    n_cases = len(cases_df)
    n_cols = 2
    n_rows = int(math.ceil(n_cases / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.6 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (_, row) in zip(axes_flat, cases_df.iterrows()):
        bundle = load_file_bundle(str(row["file_path"]), str(row["dataset"]), file_cache)
        trace = trace_cdf_pipeline(bundle, int(row["plume_idx"]))
        processed_valid = np.isfinite(bundle["time_ms"]) & np.isfinite(trace["processed_trace_mm"])
        processed_time = bundle["time_ms"][processed_valid]
        processed_pen = trace["processed_trace_mm"][processed_valid]
        fit_curve = build_fit_curve(processed_time, row)

        fps = float(pd.to_numeric(bundle["df_file"]["fps"].iloc[0], errors="coerce"))
        aligned_time_raw_original = bundle["time_ms"] - (trace["delay_frames_used"] / fps * 1e3 if fps > 0 else 0.0)
        raw_original_valid = np.isfinite(aligned_time_raw_original) & np.isfinite(trace["raw_original_mm"])
        raw_aligned_valid = np.isfinite(bundle["time_ms"]) & np.isfinite(trace["raw_aligned_mm"])

        ax.plot(
            aligned_time_raw_original[raw_original_valid],
            trace["raw_original_mm"][raw_original_valid],
            color="#bbbbbb",
            linewidth=1.0,
            alpha=0.80,
            label="raw_original",
        )
        ax.plot(
            bundle["time_ms"][raw_aligned_valid],
            trace["raw_aligned_mm"][raw_aligned_valid],
            color="#f28e2b",
            linewidth=1.2,
            alpha=0.85,
            label="raw_aligned",
        )
        ax.scatter(
            processed_time,
            processed_pen,
            color="#222222",
            s=10,
            alpha=0.85,
            label="processed_for_fit",
        )
        if processed_time.size:
            ax.plot(processed_time, fit_curve, color="#4c72b0", linestyle="--", linewidth=1.6, label="fit_curve")
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlim(-0.8, min(MAX_TIME_MS, max(np.nanmax(bundle["time_ms"]), 1.5)))
        ax.set_xlabel("Aligned time (ms)")
        ax.set_ylabel("Penetration (mm)")
        ax.grid(alpha=0.25)
        title = (
            f"{row['nozzle']} | {row['folder']} | file={row['file_name']} | plume={int(row['plume_idx'])}\n"
            f"{row['sample_split']} | rmse={float(row['rmse']):.2f} | delay={row['delay_source']}"
        )
        ax.set_title(title, fontsize=10)

    for ax in axes_flat[n_cases:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Representative CDF plume snapshots: raw -> aligned -> processed -> fit", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_df, wide_df = build_inventory(DEFAULT_SYNTHETIC_ROOT, datasets=args.datasets, folders=args.folders)
    print(f"Loaded {len(fit_df)} fit rows and {len(wide_df)} wide rows.")

    audit_df, processed_vs_fit_df = assemble_audit_tables(fit_df, wide_df)
    print(f"Assembled plume audit rows: {len(audit_df)}")
    print(f"Assembled processed-vs-fit rows: {len(processed_vs_fit_df)}")

    selection_overall = summarize_selection(audit_df, [])
    selection_by_dataset = summarize_selection(audit_df, ["dataset"])
    selection_by_nozzle = summarize_selection(audit_df, ["nozzle"])
    selection_by_pressure = summarize_selection(audit_df, ["injection_pressure_bar", "chamber_pressure_bar"])

    delay_overall = summarize_delay_source(audit_df, [])
    delay_by_dataset = summarize_delay_source(audit_df, ["dataset"])
    delay_by_pressure = summarize_delay_source(audit_df, ["injection_pressure_bar", "chamber_pressure_bar"])

    filter_overall = summarize_filter_stages(audit_df, [])
    filter_by_dataset = summarize_filter_stages(audit_df, ["dataset"])

    timebin_overall = summarize_time_bins(processed_vs_fit_df, [])
    timebin_by_dataset = summarize_time_bins(processed_vs_fit_df, ["dataset"])
    timebin_by_pressure = summarize_time_bins(processed_vs_fit_df, ["injection_pressure_bar", "chamber_pressure_bar"])

    representative_cases = select_representative_cases(audit_df, max_cases=max(1, int(args.max_cases)))

    validation_summary = pd.DataFrame(
        [
            {
                "n_plumes": int(len(audit_df)),
                "processed_series_match_fraction": float(audit_df["processed_matches_series_wide"].mean()),
                "processed_series_match_failures": int((~audit_df["processed_matches_series_wide"]).sum()),
                "rmse_recompute_median_abs_diff": safe_median(audit_df["rmse_abs_diff"]),
                "rmse_recompute_p90_abs_diff": safe_quantile(audit_df["rmse_abs_diff"], 0.90),
                "clean_mean_n": safe_mean(audit_df.loc[audit_df["sample_split"] == "clean", "n"]),
                "flagged_mean_n": safe_mean(audit_df.loc[audit_df["sample_split"] == "flagged", "n"]),
                "clean_mean_rmse": safe_mean(audit_df.loc[audit_df["sample_split"] == "clean", "rmse"]),
                "flagged_mean_rmse": safe_mean(audit_df.loc[audit_df["sample_split"] == "flagged", "rmse"]),
                "clean_area_delay_fraction": float(
                    (audit_df.loc[audit_df["sample_split"] == "clean", "delay_source"] == "area").mean()
                ),
                "flagged_area_delay_fraction": float(
                    (audit_df.loc[audit_df["sample_split"] == "flagged", "delay_source"] == "area").mean()
                ),
            }
        ]
    )

    save_csv(audit_df, out_dir / "cdf_plume_audit.csv")
    save_csv(validation_summary, out_dir / "cdf_validation_summary.csv")
    save_csv(selection_overall, out_dir / "selection_summary_overall.csv")
    save_csv(selection_by_dataset, out_dir / "selection_summary_by_dataset.csv")
    save_csv(selection_by_nozzle, out_dir / "selection_summary_by_nozzle.csv")
    save_csv(selection_by_pressure, out_dir / "selection_summary_by_pressure.csv")
    save_csv(delay_overall, out_dir / "delay_source_overall.csv")
    save_csv(delay_by_dataset, out_dir / "delay_source_by_dataset.csv")
    save_csv(delay_by_pressure, out_dir / "delay_source_by_pressure.csv")
    save_csv(filter_overall, out_dir / "filter_stage_summary_overall.csv")
    save_csv(filter_by_dataset, out_dir / "filter_stage_summary_by_dataset.csv")
    save_csv(timebin_overall, out_dir / "processed_vs_fit_timebins_overall.csv")
    save_csv(timebin_by_dataset, out_dir / "processed_vs_fit_timebins_by_dataset.csv")
    save_csv(timebin_by_pressure, out_dir / "processed_vs_fit_timebins_by_pressure.csv")
    save_csv(representative_cases, out_dir / "representative_cases.csv")

    plot_selection_bias(audit_df, delay_overall, out_dir / "selection_bias_overview.png")
    plot_filter_stage_audit(audit_df, out_dir / "filter_stage_audit.png")
    plot_processed_vs_fit(timebin_overall, out_dir / "processed_vs_fit_overall.png")

    file_cache: dict[str, dict] = {}
    plot_representative_cases(
        representative_cases,
        out_dir / "representative_plume_panels.png",
        file_cache,
    )

    print("Key validation checks:")
    print(validation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
