"""Per-row fit-quality masking and clean/flagged split.

``apply_filter_masking`` computes the late-time predicted penetration
from the q1 model parameters, applies the basic/penetration/outlier
masks, and partitions the fit table into clean and flagged subsets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import (
    ENABLE_MASK_BASIC,
    ENABLE_MASK_OUTLIER,
    ENABLE_MASK_PENETRATION_FAR,
    MASK_FAR_TIME_MS,
    MASK_GROUP_COLS,
    MASK_MIN_N,
    MASK_PENETRATION_LOWER_MM,
    MASK_PENETRATION_UPPER_MM,
    MASK_S_UPPER,
    MASK_T0_UPPER,
    MASK_Z_THRESH,
    MIN_TI,
)


def robust_z(series):
    """Median/MAD robust z-score used for within-file fit-quality outlier checks."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return 0.6745 * (s - med) / (mad + 1e-12)


def apply_filter_masking(df, group_cols=MASK_GROUP_COLS, z_thresh=MASK_Z_THRESH):
    """Attach quality masks and split fit rows into clean and flagged subsets.

    The late-time penetration is evaluated from the q1 model:

        S(t) = expit((t - t0) / s) * k_quarter * t**0.25
    """
    out = df.copy()
    if out.empty:
        out["cost_per_point"] = np.nan
        out["penetration_far_mm"] = np.nan
        out["z_t0"] = np.nan
        out["z_rmse"] = np.nan
        out["z_cost"] = np.nan
        out["mask_basic"] = False
        out["mask_penetration_far"] = False
        out["mask_outlier"] = False
        out["flag_bad_fit"] = False
        return out, out.copy(), out.copy()

    out["cost_per_point"] = 2.0 * pd.to_numeric(out["cost"], errors="coerce") / pd.to_numeric(
        out["n"], errors="coerce"
    ).clip(lower=1)
    t_far_s = MASK_FAR_TIME_MS * 1e-3
    log_params_far = out[["log_k_quarter", "log_t0", "log_s"]].apply(
        pd.to_numeric, errors="coerce"
    )
    out["penetration_far_mm"] = np.nan
    valid_far = (
        out["success"].fillna(False)
        & np.isfinite(log_params_far["log_k_quarter"])
        & np.isfinite(log_params_far["log_t0"])
        & np.isfinite(log_params_far["log_s"])
    )
    if valid_far.any():
        lp_far = log_params_far.loc[
            valid_far, ["log_k_quarter", "log_t0", "log_s"]
        ].to_numpy(dtype=float)
        k_quarter_far = np.exp(lp_far[:, 0])
        t0_far = np.exp(lp_far[:, 1]) + MIN_TI
        s_far = np.exp(lp_far[:, 2])
        w_far = expit((t_far_s - t0_far) / s_far)
        far_vals = w_far * k_quarter_far * np.power(t_far_s, 0.25)
        out.loc[valid_far, "penetration_far_mm"] = np.asarray(far_vals, dtype=float)

    if "t_max_s" in out.columns:
        t_max = pd.to_numeric(out["t_max_s"], errors="coerce")
    else:
        t0_numeric = pd.to_numeric(out["t0"], errors="coerce")
        fallback_tmax = np.nan if t0_numeric.notna().sum() == 0 else np.nanmax(t0_numeric)
        t_max = pd.Series(np.full(len(out), fallback_tmax), index=out.index)

    mask_basic = (
        out["success"].fillna(False)
        & np.isfinite(pd.to_numeric(out["t0"], errors="coerce"))
        & np.isfinite(pd.to_numeric(out["rmse"], errors="coerce"))
        & np.isfinite(pd.to_numeric(out["cost_per_point"], errors="coerce"))
        & (pd.to_numeric(out["n"], errors="coerce") >= MASK_MIN_N)
        & (pd.to_numeric(out["t0"], errors="coerce") > 0)
        & (pd.to_numeric(out["t0"], errors="coerce") < t_max)
        & (pd.to_numeric(out["s"], errors="coerce") > 0)
        & (pd.to_numeric(out["s"], errors="coerce") < MASK_S_UPPER)
        & (pd.to_numeric(out["t0"], errors="coerce") < MASK_T0_UPPER)
    )
    out["mask_basic"] = mask_basic if ENABLE_MASK_BASIC else True

    mask_penetration_far = (
        np.isfinite(pd.to_numeric(out["penetration_far_mm"], errors="coerce"))
        & pd.to_numeric(out["penetration_far_mm"], errors="coerce").between(
            MASK_PENETRATION_LOWER_MM, MASK_PENETRATION_UPPER_MM
        )
    )
    out["mask_penetration_far"] = mask_penetration_far if ENABLE_MASK_PENETRATION_FAR else True

    out["z_t0"] = out.groupby(list(group_cols), dropna=False)["t0"].transform(robust_z)
    out["z_rmse"] = out.groupby(list(group_cols), dropna=False)["rmse"].transform(
        lambda s: robust_z(np.log1p(pd.to_numeric(s, errors="coerce")))
    )
    out["z_cost"] = out.groupby(list(group_cols), dropna=False)["cost_per_point"].transform(
        lambda s: robust_z(np.log1p(pd.to_numeric(s, errors="coerce")))
    )

    mask_outlier = (
        out["z_t0"].abs().gt(z_thresh)
        | out["z_rmse"].abs().gt(z_thresh)
        | out["z_cost"].abs().gt(z_thresh)
    ).fillna(False)
    out["mask_outlier"] = mask_outlier if ENABLE_MASK_OUTLIER else False

    out["flag_bad_fit"] = (~out["mask_basic"]) | (~out["mask_penetration_far"]) | out["mask_outlier"]

    clean_df = out.loc[~out["flag_bad_fit"]].copy()
    flagged_df = out.loc[out["flag_bad_fit"]].copy()
    return out, clean_df, flagged_df
