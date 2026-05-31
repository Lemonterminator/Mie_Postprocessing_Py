"""Per-plume trace cleaning: subframe-delay estimation and outlier trimming.

These helpers transform one raw penetration array into the cleaned, delay-
aligned series that the q1 fitter consumes. Switches that gate each step
live in ``core.config``; behaviour is unchanged from the original
``fit_raw_data.py`` implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ENABLE_DIFF_THRESHOLD_LOWER,
    ENABLE_DIFF_THRESHOLD_UPPER,
    ENABLE_HYDRAULIC_DELAY_SCAN,
    ENABLE_REPLACE_NEGATIVE_WITH_ZERO,
    MIN_INITIAL_VELOCITY,
)


def calculate_subframe_delay(time_s, series, first_valid_idx, n_points=3, fallback_delay_s=float("nan")):
    """Estimate spray start time by extrapolating the first positive samples."""
    max_idx = min(len(time_s), len(series))

    t_early = []
    y_early = []
    for i in range(first_valid_idx, max_idx):
        if len(t_early) >= n_points:
            break
        if np.isfinite(series[i]) and series[i] > 0:
            t_early.append(time_s[i])
            y_early.append(series[i])

    if len(t_early) < 2:
        return fallback_delay_s

    t_early_arr = np.array(t_early)
    y_early_arr = np.array(y_early)

    slope, intercept = np.polyfit(t_early_arr, y_early_arr, 1)
    if slope > MIN_INITIAL_VELOCITY:
        t_intercept = -intercept / slope
        if 0 <= t_intercept <= t_early_arr[-1]:
            return float(t_intercept)
    return fallback_delay_s


def penetration_cleaning(
    arr,
    scaling_factor,
    diff_threshold_lower=1.0,
    diff_threshold_upper=np.inf,
    hd_upper_lim=15,
    forced_delay=None,
    replace_negative_with_zero=False,
):
    """Clean one plume penetration trace before curve fitting.

    The steps mirror the notebook prototype: optional hydraulic-delay scan,
    optional negative-value clamp, pre-onset removal, unit conversion, and
    lower/upper frame-difference truncation.
    """
    arr = np.asarray(arr, dtype=float).copy()
    first_positive_idx = 0 if forced_delay is None else int(forced_delay)

    if forced_delay is None and ENABLE_HYDRAULIC_DELAY_SCAN:
        scan_limit = min(hd_upper_lim, arr.size - 1)
        for f in range(scan_limit):
            if arr[f + 1] == 0 or np.isnan(arr[f + 1]):
                first_positive_idx += 1
                arr[f] = np.nan

    if replace_negative_with_zero and ENABLE_REPLACE_NEGATIVE_WITH_ZERO:
        arr[arr < 0] = 0.0

    if first_positive_idx > 0 and first_positive_idx < arr.size:
        arr[:first_positive_idx] = np.nan

    positive_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    if positive_idx.size == 0:
        arr[:] = np.nan
        return arr * scaling_factor, first_positive_idx

    actual_first_positive_idx = int(positive_idx[0])

    arr *= scaling_factor

    if ENABLE_DIFF_THRESHOLD_LOWER:
        arr_diff = np.diff(arr[actual_first_positive_idx:])
        lower_cut_idx = np.where(arr_diff < diff_threshold_lower)[0]
        if lower_cut_idx.size > 0:
            arr = arr[: actual_first_positive_idx + lower_cut_idx[0].item() + 1]

    if ENABLE_DIFF_THRESHOLD_UPPER:
        valid_positive_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
        if valid_positive_idx.size == 0:
            arr[:] = np.nan
            return arr, first_positive_idx
        actual_first_positive_idx = int(valid_positive_idx[0])
        arr_diff = np.diff(arr[actual_first_positive_idx:])
        upper_cut_idx = np.where(arr_diff > diff_threshold_upper)[0]
        if upper_cut_idx.size > 0:
            arr = arr[: actual_first_positive_idx + upper_cut_idx[0].item() + 1]

    return arr, first_positive_idx


def get_area_based_delay(df_file, plume_idx):
    """Prefer area-onset delay when an ``area_plume_*`` signal is available."""
    area_col = f"area_plume_{plume_idx}"
    if area_col not in df_file.columns:
        return np.nan, "penetration_fallback"

    area = pd.to_numeric(df_file[area_col], errors="coerce").to_numpy(dtype=float)
    positive_idx = np.flatnonzero(np.isfinite(area) & (area > 0))
    if positive_idx.size == 0:
        return np.nan, "penetration_fallback"

    first_positive_idx = int(positive_idx[0])
    if first_positive_idx <= 0:
        return np.nan, "penetration_fallback"

    if not np.isfinite(area[first_positive_idx - 1]) or area[first_positive_idx - 1] != 0:
        return np.nan, "penetration_fallback"

    return float(first_positive_idx - 1), "area"
