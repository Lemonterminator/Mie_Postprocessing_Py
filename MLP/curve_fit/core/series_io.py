"""Raw-CSV ingestion, plume-column resolution, delay alignment, and series reshape.

Covers the I/O + reshape stages of the raw-fit pipeline:

- ``_read_csv_with_expanded_static_meta`` / ``_infer_num_plumes_from_columns``
  / ``_resolve_penetration_prefix_and_scale`` — load one raw CSV and pick the
  correct penetration columns + mm scale.
- ``prepare_cleaned_series`` — drive trace cleaning + subframe-delay
  estimation + cross-plume delay-clip alignment for every plume in one file.
- ``collect_series_rows`` / ``filter_series_df`` / ``build_wide_series_df`` —
  flatten the aligned plume arrays into long/wide point tables.
- ``get_dataset_settings`` — per-dataset calibration and delay defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import calculate_subframe_delay, get_area_based_delay, penetration_cleaning
from .config import (
    DIFF_THRESHOLD_LOWER,
    DIFF_THRESHOLD_UPPER,
    ENABLE_DELAY_CLIP,
    META_COLS,
    NUM_POINTS_SOI_LINEAR_REGRESSION,
)


def _is_missing_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _first_non_missing(series):
    s = pd.Series(series)
    valid = s[~s.isna()]
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def _read_csv_with_expanded_static_meta(csv_path):
    """Load a raw CSV and broadcast static metadata from its companion JSON."""
    df = pd.read_csv(csv_path)

    meta_path = csv_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    for col in META_COLS:
        first_val = _first_non_missing(df[col]) if col in df.columns else np.nan
        resolved = meta.get(col, first_val)
        if _is_missing_value(resolved):
            resolved = first_val
        if _is_missing_value(resolved):
            if col not in df.columns:
                df[col] = np.nan
            continue
        df[col] = resolved

    return df


def _infer_num_plumes_from_columns(df_file, column_prefix):
    plume_indices = []
    for col in df_file.columns:
        if not col.startswith(column_prefix):
            continue
        suffix = col[len(column_prefix):]
        if suffix.isdigit():
            plume_indices.append(int(suffix))
    if not plume_indices:
        return 0
    return max(plume_indices) + 1


def _resolve_penetration_prefix_and_scale(df_file, penetration_source, mm_per_px_scale):
    """Choose mm or legacy-pixel penetration columns and return the mm scale."""
    mm_prefix = penetration_source.get("column_prefix_mm")
    if mm_prefix and _infer_num_plumes_from_columns(df_file, mm_prefix) > 0:
        return mm_prefix, 1.0

    umbrella_angle_deg = (
        float(pd.to_numeric(df_file["umbrella_angle_deg"].iloc[0], errors="coerce"))
        if "umbrella_angle_deg" in df_file.columns
        else 180.0
    )
    if not np.isfinite(umbrella_angle_deg):
        umbrella_angle_deg = 180.0

    correction = float(mm_per_px_scale)
    if penetration_source.get("legacy_needs_umbrella_correction", False):
        tilt_ang = (180.0 - umbrella_angle_deg) / 2.0
        correction *= 1.0 / np.cos(np.deg2rad(tilt_ang))
    return penetration_source["column_prefix"], correction


def prepare_cleaned_series(
    df_file,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
    penetration_source,
    replace_negative_with_zero=False,
    diff_threshold_lower=DIFF_THRESHOLD_LOWER,
    diff_threshold_upper=DIFF_THRESHOLD_UPPER,
):
    """Build delay-aligned cleaned series arrays for every plume in one CSV."""
    penetration_column_prefix, pen_correction = _resolve_penetration_prefix_and_scale(
        df_file,
        penetration_source,
        mm_per_px_scale,
    )
    if "plumes" in df_file.columns and np.isfinite(pd.to_numeric(df_file["plumes"].iloc[0], errors="coerce")):
        number_of_plumes = int(pd.to_numeric(df_file["plumes"].iloc[0], errors="coerce"))
    else:
        number_of_plumes = _infer_num_plumes_from_columns(df_file, penetration_column_prefix)
    if number_of_plumes <= 0:
        raise ValueError("Unable to infer plume count from metadata or penetration columns.")

    fps = float(pd.to_numeric(df_file["fps"].iloc[0], errors="coerce")) if "fps" in df_file.columns else np.nan
    if np.isnan(fps):
        fps = fps_default
    frame_idx = np.asarray(df_file["frame_idx"]).astype(int)

    time_s = frame_idx / fps

    max_len = int(frame_idx.max()) + 1
    cleaned_series = np.full((number_of_plumes, max_len), np.nan)
    delays_raw = np.full(number_of_plumes, np.nan, dtype=float)
    delay_sources = np.full(number_of_plumes, "", dtype=object)
    temp_series = [None] * number_of_plumes

    for plume_idx in range(number_of_plumes):
        col = f"{penetration_column_prefix}{plume_idx}"
        if col not in df_file.columns:
            continue

        area_delay, delay_source = get_area_based_delay(df_file, plume_idx)
        arr = np.asarray(df_file[col], dtype=float).copy()
        cleaned_serie, first_pos_idx = penetration_cleaning(
            arr,
            pen_correction,
            diff_threshold_lower=diff_threshold_lower,
            diff_threshold_upper=diff_threshold_upper,
            hd_upper_lim=max_hydraulic_delay_frames,
            forced_delay=area_delay if np.isfinite(area_delay) else None,
            replace_negative_with_zero=replace_negative_with_zero,
        )

        fallback_delay_s = float(first_pos_idx) / fps
        delay_s = calculate_subframe_delay(
            time_s, cleaned_serie, first_pos_idx,
            n_points=NUM_POINTS_SOI_LINEAR_REGRESSION,
            fallback_delay_s=fallback_delay_s,
        )
        delays_raw[plume_idx] = delay_s
        delay_sources[plume_idx] = delay_source if np.isfinite(area_delay) else "penetration_fallback"
        temp_series[plume_idx] = np.asarray(cleaned_serie, dtype=float)

    # Robust file-level delay reference, then clip individual plume delays
    # around it so one bad onset does not shift a plume far away.
    valid_delays = delays_raw[np.isfinite(delays_raw)]
    median_delay_s = np.nanmedian(valid_delays) if valid_delays.size else 0.0
    delays_used = np.full(number_of_plumes, median_delay_s, dtype=float)
    valid_raw_mask = np.isfinite(delays_raw)
    if ENABLE_DELAY_CLIP:
        lower_bound = median_delay_s - (float(delay_clip_half_window) / fps)
        upper_bound = median_delay_s + (float(delay_clip_half_window) / fps)
        if np.any(valid_raw_mask):
            delays_used[valid_raw_mask] = np.clip(
                delays_raw[valid_raw_mask],
                lower_bound,
                upper_bound,
            )
    elif np.any(valid_raw_mask):
        delays_used[valid_raw_mask] = delays_raw[valid_raw_mask]

    time_s_aligned = np.full((number_of_plumes, max_len), np.nan)
    time_ms_aligned = np.full((number_of_plumes, max_len), np.nan)

    for plume_idx in range(number_of_plumes):
        series = temp_series[plume_idx]
        if series is None:
            continue

        plume_delay_s = delays_used[plume_idx]
        delay_frames_int = int(np.round(plume_delay_s * fps))

        aligned = np.full_like(series, np.nan, dtype=float)
        if delay_frames_int > 0 and delay_frames_int < series.size:
            aligned[:-delay_frames_int] = series[delay_frames_int:]
        elif delay_frames_int == 0:
            aligned = series
        elif delay_frames_int < 0 and -delay_frames_int < series.size:
            aligned[-delay_frames_int:] = series[:delay_frames_int]

        n = min(aligned.size, max_len)
        if n > 0:
            cleaned_series[plume_idx, :n] = aligned[:n]
            t_exact = (np.arange(n) + delay_frames_int) / fps - plume_delay_s
            time_s_aligned[plume_idx, :n] = t_exact
            time_ms_aligned[plume_idx, :n] = t_exact * 1000.0

    return time_s_aligned, time_ms_aligned, cleaned_series, delays_raw * fps, delays_used * fps, delay_sources


def collect_series_rows(
    file_path,
    time_s,
    time_ms,
    cleaned_series,
    delays_raw,
    delays_used,
    delay_sources,
):
    """Flatten cleaned plume arrays into a long point table."""
    rows = []
    file_path = Path(file_path)
    for plume_idx in range(cleaned_series.shape[0]):
        series = np.asarray(cleaned_series[plume_idx], dtype=float)
        ts = time_s[plume_idx]
        tms = time_ms[plume_idx]
        valid = np.isfinite(ts) & np.isfinite(tms) & np.isfinite(series)
        if not np.any(valid):
            continue

        for idx in np.flatnonzero(valid):
            rows.append(
                {
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "plume_idx": plume_idx,
                    "frame_pos": int(idx),
                    "time_s": float(ts[idx]),
                    "time_ms": float(tms[idx]),
                    "penetration_mm": float(series[idx]),
                    "delay_frames_raw": delays_raw[plume_idx],
                    "delay_frames_used": delays_used[plume_idx],
                    "delay_source": delay_sources[plume_idx],
                }
            )
    return rows


def filter_series_df(series_df, fit_df):
    """Keep only series points whose plume survived into the supplied fit table."""
    if series_df.empty or fit_df.empty:
        return series_df.iloc[0:0].copy()

    key_cols = ["file_path", "file_name", "file_stem", "plume_idx"]
    valid_keys = fit_df.loc[:, key_cols].drop_duplicates()
    return series_df.merge(valid_keys, on=key_cols, how="inner")


def build_wide_series_df(series_df, fit_df):
    """Pivot long cleaned-series points into time_ms_*/penetration_mm_* columns."""
    if series_df.empty:
        base_cols = [
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
        return pd.DataFrame(columns=base_cols)

    key_cols = ["file_path", "file_name", "file_stem", "plume_idx"]
    meta_cols = key_cols + [
        "delay_frames_raw",
        "delay_frames_used",
        "delay_source",
        *META_COLS,
    ]
    fit_meta = fit_df.loc[:, [col for col in meta_cols if col in fit_df.columns]].drop_duplicates(key_cols)

    base = (
        series_df.loc[:, key_cols + ["frame_pos", "delay_frames_raw", "delay_frames_used", "delay_source"]]
        .sort_values(key_cols + ["frame_pos"])
        .groupby(key_cols, dropna=False)
        .agg(
            delay_frames_raw=("delay_frames_raw", "first"),
            delay_frames_used=("delay_frames_used", "first"),
            delay_source=("delay_source", "first"),
            seq_len=("frame_pos", "count"),
        )
        .reset_index()
    )
    if not fit_meta.empty:
        base = base.merge(fit_meta, on=key_cols + ["delay_frames_raw", "delay_frames_used", "delay_source"], how="left")

    time_wide = (
        series_df.pivot_table(index=key_cols, columns="frame_pos", values="time_ms", aggfunc="first")
        .sort_index(axis=1)
        .rename(columns=lambda c: f"time_ms_{int(c):03d}")
        .reset_index()
    )
    pen_wide = (
        series_df.pivot_table(index=key_cols, columns="frame_pos", values="penetration_mm", aggfunc="first")
        .sort_index(axis=1)
        .rename(columns=lambda c: f"penetration_mm_{int(c):03d}")
        .reset_index()
    )

    wide = base.merge(time_wide, on=key_cols, how="left").merge(pen_wide, on=key_cols, how="left")
    return wide


def get_dataset_settings(name):
    """Return per-dataset calibration and delay defaults."""
    if name == "Nozzle0":
        return {
            "or_mm_per_px_reference": 412.0,
            "fps_default": 34000,
            "max_hydraulic_delay_frames": 30,
            "delay_clip_half_window": 2,
        }
    return {
        "or_mm_per_px_reference": 377.0,  # 90 mm reference in px
        "fps_default": 25000,
        "max_hydraulic_delay_frames": 17,
        "delay_clip_half_window": 3,
    }
