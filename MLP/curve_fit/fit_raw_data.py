from pathlib import Path
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Penetration-series switches.
FIT_PENETRATION_CDF = True
FIT_PENETRATION_BW_X = True
FIT_PENETRATION_BW_POLAR = True

# Filtering switches.
ENABLE_REPLACE_NEGATIVE_WITH_ZERO = True  # For bw_x only: clamp negative penetration to 0 before cleaning.
ENABLE_HYDRAULIC_DELAY_SCAN = True  # Detect early zero/NaN frames and shift the series left.
ENABLE_DIFF_THRESHOLD_LOWER = True  # Cut the series when frame-to-frame growth becomes too small.
ENABLE_DIFF_THRESHOLD_UPPER = True  # Cut the series when frame-to-frame growth jumps unrealistically high.
ENABLE_DELAY_CLIP = True  # Clip plume delay to the median-centered window before plume alignment.
ENABLE_MASK_BASIC = True  # Apply basic fit-quality checks: success, finite metrics, n, t0, s.
ENABLE_MASK_PENETRATION_FAR = True  # Require far-time penetration to lie in the configured mm range.
ENABLE_MASK_OUTLIER = True  # Remove robust-z outliers on t0, rmse, and cost_per_point.

NUM_POINTS_SOI_LINEAR_REGRESSION = 2
MIN_INITIAL_VELOCITY = 1e-7



# Input/output roots (curve_fit/ is one level below MLP/, two below project root)
_THIS_DIR = Path(__file__).resolve().parent
data_root = _THIS_DIR.parent.parent / "Mie_scattering_top_view_results"
data_out_dir = _THIS_DIR.parent / "synthetic_data"


names = [
    "BC20241003_HZ_Nozzle1",
    "BC20241017_HZ_Nozzle2",
    "BC20241014_HZ_Nozzle3",
    "BC20241007_HZ_Nozzle4",
    "BC20241010_HZ_Nozzle5",
    "BC20241011_HZ_Nozzle6",
    "BC20241015_HZ_Nozzle7",
    "BC20241016_HZ_Nozzle8",
    "Nozzle0",
]


# Filtering/masking settings (from notebook prototype defaults)
MASK_GROUP_COLS = ("file_name",)
MASK_Z_THRESH = 3.0
MASK_MIN_N = 10
MASK_S_UPPER = 1e-3  # < 1 ms
MASK_T0_UPPER = 0.8e-3 
MASK_FAR_TIME_MS = 5.0
MASK_PENETRATION_LOWER_MM = 18.0        # penetration (mm) at MASK_FAR_TIME_MS
MASK_PENETRATION_UPPER_MM = 300.0       # penetration (mm) at MASK_FAR_TIME_MS
RMSE_SUCCESS_THRESHOLD_MM = 3.0         # max RMSE (mm) for a fit to count as successful

PLOT_EXTRAP_FACTOR = 1.6
PLOT_NUM_POINTS = 300
PLOT_YLIM_MM = 200.0
FIT_MODEL_NAME = "quarter_only_v1"
LOG_K_SQRT_SENTINEL = -500.0  # Finite log-space sentinel; exp(-500) is effectively zero.
K_SQRT_SENTINEL = 0.0
ABLATION_QUARTER_ONLY = False  # q1 is now the main exported model, so no duplicate q1 overlay/report is needed.
N_WORKERS = 0  # 0 = use all logical CPUs; 1 = single-process (easy debugging); N > 1 = explicit pool size.




DIFF_THRESHOLD_LOWER = 1.0 # mm
DIFF_THRESHOLD_UPPER = 10.0  # mm
MIN_TI = 0.0
MIN_SERIES_POINTS = 10  # discard series with fewer valid points before fitting

def calculate_subframe_delay(time_s, series, first_valid_idx, n_points=3, fallback_delay_s=float('nan')):
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


PENETRATION_SOURCES = [
    {
        "enabled": FIT_PENETRATION_CDF,
        "key": "cdf",
        "column_prefix_mm": "penetration_cdf(mm)_plume_",
        "column_prefix": "penetration_cdf_plume_",
        "label": "penetration_cdf",
        "replace_negative_with_zero": False,
        "legacy_needs_umbrella_correction": True,
    },
    {
        "enabled": FIT_PENETRATION_BW_X,
        "column_prefix_mm": "penetration_bw_x(mm)_plume_",
        "key": "bw_x",
        "column_prefix": "penetration_bw_x_plume_",
        "label": "penetration_bw_x",
        "replace_negative_with_zero": True,
        "legacy_needs_umbrella_correction": True,
    },
    {
        "enabled": FIT_PENETRATION_BW_POLAR,
        "column_prefix_mm": "penetration_bw_polar(mm)_plume_",
        "key": "bw_polar",
        "column_prefix": "penetration_bw_polar_plume_",
        "label": "penetration_bw_polar",
        "replace_negative_with_zero": False,
        "legacy_needs_umbrella_correction": False,
    },
]


def get_enabled_penetration_sources():
    return [source for source in PENETRATION_SOURCES if source["enabled"]]


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
    prefix = column_prefix
    plume_indices = []
    for col in df_file.columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        if suffix.isdigit():
            plume_indices.append(int(suffix))
    if not plume_indices:
        return 0
    return max(plume_indices) + 1


def _resolve_penetration_prefix_and_scale(df_file, penetration_source, mm_per_px_scale):
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


def robust_z(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return 0.6745 * (s - med) / (mad + 1e-12)


def apply_filter_masking(df, group_cols=MASK_GROUP_COLS, z_thresh=MASK_Z_THRESH):
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
        out["flag_bad_fit_q1"] = False
        return out, out.copy(), out.copy()

    out["cost_per_point"] = 2.0 * pd.to_numeric(out["cost"], errors="coerce") / pd.to_numeric(
        out["n"], errors="coerce"
    ).clip(lower=1)
    t_far_s = MASK_FAR_TIME_MS * 1e-3
    log_params_far = out[["log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]].apply(
        pd.to_numeric, errors="coerce"
    )
    out["penetration_far_mm"] = np.nan
    valid_far = (
        out["success"].fillna(False)
        & np.isfinite(log_params_far["log_k_sqrt"])
        & np.isfinite(log_params_far["log_k_quarter"])
        & np.isfinite(log_params_far["log_t0"])
        & np.isfinite(log_params_far["log_s"])
    )
    if valid_far.any():
        lp_far = log_params_far.loc[
            valid_far, ["log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]
        ].to_numpy(dtype=float)
        k_sqrt_far = np.exp(lp_far[:, 0])
        k_quarter_far = np.exp(lp_far[:, 1])
        t0_far = np.exp(lp_far[:, 2]) + MIN_TI
        s_far = np.exp(lp_far[:, 3])
        w_far = expit((t_far_s - t0_far) / s_far)
        far_vals = (1.0 - w_far) * (k_sqrt_far * np.sqrt(t_far_s)) + w_far * (
            k_quarter_far * np.power(t_far_s, 0.25)
        )
        out.loc[valid_far, "penetration_far_mm"] = np.asarray(far_vals, dtype=float)

    if "t_max_s" in out.columns:
        t_max = pd.to_numeric(out["t_max_s"], errors="coerce")
    else:
        t0_numeric = pd.to_numeric(out["t0"], errors="coerce")
        fallback_tmax = np.nan if t0_numeric.notna().sum() == 0 else np.nanmax(t0_numeric)
        t_max = pd.Series(np.full(len(out), fallback_tmax), index=out.index)

    # hard checks from the notebook prototype
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

    # robust outlier checks (prototype: grouped by file_name)
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

    # --- Legacy q1-alias masking for older ablation outputs ---
    if "success_q1" in out.columns:
        t0_q1 = pd.to_numeric(out["t0_q1"], errors="coerce")
        s_q1  = pd.to_numeric(out["s_q1"],  errors="coerce")
        n_q1  = pd.to_numeric(out["n_q1"],  errors="coerce")
        cost_q1 = pd.to_numeric(out["cost_q1"], errors="coerce")
        cpp_q1 = 2.0 * cost_q1 / n_q1.clip(lower=1)

        mask_basic_q1 = (
            out["success_q1"].fillna(False)
            & np.isfinite(t0_q1)
            & np.isfinite(pd.to_numeric(out["rmse_q1"], errors="coerce"))
            & np.isfinite(cpp_q1)
            & (n_q1 >= MASK_MIN_N)
            & (t0_q1 > 0)
            & (t0_q1 < t_max)
            & (t0_q1 < MASK_T0_UPPER)
            & (s_q1 > 0)
            & (s_q1 < MASK_S_UPPER)
        )

        pen_far_q1 = pd.to_numeric(out["penetration_far_mm_q1"], errors="coerce")
        mask_pen_far_q1 = (
            np.isfinite(pen_far_q1)
            & pen_far_q1.between(MASK_PENETRATION_LOWER_MM, MASK_PENETRATION_UPPER_MM)
        )

        out["z_t0_q1"] = out.groupby(list(group_cols), dropna=False)["t0_q1"].transform(robust_z)
        out["z_rmse_q1"] = out.groupby(list(group_cols), dropna=False)["rmse_q1"].transform(
            lambda s: robust_z(np.log1p(pd.to_numeric(s, errors="coerce")))
        )
        _cpp_q1_col = cpp_q1.rename("_cpp_q1")
        out["z_cost_q1"] = _cpp_q1_col.groupby(
            out.groupby(list(group_cols), dropna=False).ngroup()
        ).transform(lambda s: robust_z(np.log1p(s)))

        mask_outlier_q1 = (
            out["z_t0_q1"].abs().gt(z_thresh)
            | out["z_rmse_q1"].abs().gt(z_thresh)
            | out["z_cost_q1"].abs().gt(z_thresh)
        ).fillna(False)

        out["flag_bad_fit_q1"] = (~mask_basic_q1) | (~mask_pen_far_q1) | mask_outlier_q1
    else:
        out["flag_bad_fit_q1"] = out["flag_bad_fit"]

    clean_df = out.loc[~out["flag_bad_fit"]].copy()
    flagged_df = out.loc[out["flag_bad_fit"]].copy()
    return out, clean_df, flagged_df


def penetration_cleaning(
    arr,
    scaling_factor,
    diff_threshold_lower=1.0,
    diff_threshold_upper=np.inf,
    hd_upper_lim=15,
    forced_delay=None,
    replace_negative_with_zero=False,
):
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


def spray_penetration_model_sigmoid(params, t):
    """
    params (log-space): [log_k_sqrt, log_k_quarter, log_t0, log_s]
    """
    log_k_sqrt, log_k_quarter, log_t0, log_s = params

    k_sqrt = np.exp(log_k_sqrt)
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TI
    s = np.exp(log_s)

    t = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    sqrt_segment = k_sqrt * np.sqrt(t)
    quarter_root_segment = k_quarter * np.power(t, 0.25)
    w = expit((t - t0) / s)

    return (1.0 - w) * sqrt_segment + w * quarter_root_segment


def spray_penetration_model_quarter_only(params, t):
    """3-param ablation model: expit((t-t0)/s) * k_quarter * t^0.25

    The sigmoid acts as an onset ramp (0→1), so penetration is near zero
    before t0 and approaches k_quarter * t^0.25 for large t.
    """
    log_k_quarter, log_t0, log_s = params
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TI
    s = np.exp(log_s)
    t = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    w = expit((t - t0) / s)
    return w * k_quarter * np.power(t, 0.25)


def _param_uncertainty_from_jac(res, n_valid, n_params):
    """Estimate parameter standard errors and correlations from a SciPy
    least_squares result. Returns (std, corr) or None if the Jacobian is
    degenerate. ``cov ~= (J^T J)^-1 * sigma^2`` with sigma^2 from the
    unweighted residual variance (``res.fun`` is the raw residual vector,
    independent of the Huber transformation)."""
    try:
        if res.jac is None or n_valid <= n_params:
            return None
        jac = np.asarray(res.jac, dtype=float)
        if jac.size == 0 or not np.all(np.isfinite(jac)):
            return None
        residuals = np.asarray(res.fun, dtype=float)
        sigma2 = float(np.sum(residuals * residuals) / max(n_valid - n_params, 1))
        jtj = jac.T @ jac
        cov = np.linalg.inv(jtj) * sigma2
        diag = np.diag(cov)
        if not np.all(np.isfinite(diag)) or np.any(diag < 0):
            return None
        std = np.sqrt(diag)
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, np.nan)
        return std, corr
    except (np.linalg.LinAlgError, ValueError):
        return None


def fit_quarter_only(t, y, x0_3):
    """Fit the 3-param quarter-only ablation model."""
    valid = np.isfinite(t) & np.isfinite(y)
    nan_result = {
        "log_params_q1": np.full(3, np.nan),
        "k_quarter_q1": np.nan,
        "t0_q1": np.nan,
        "s_q1": np.nan,
        "cost_q1": np.inf,
        "success_q1": False,
        "n_q1": int(valid.sum()),
        "nfev_q1": 0,
        "optimality_q1": np.nan,
        "status_q1": -10,
        "std_log_k_quarter_q1": np.nan,
        "std_log_t0_q1": np.nan,
        "std_log_s_q1": np.nan,
        "corr_logk_logt0_q1": np.nan,
        "corr_logk_logs_q1": np.nan,
        "corr_logt0_logs_q1": np.nan,
    }
    if valid.sum() < 3:
        return nan_result

    t_fit = t[valid]
    y_fit = y[valid]

    def residuals(params):
        y_hat = spray_penetration_model_quarter_only(params, t_fit)
        r = y_hat - y_fit
        if not np.all(np.isfinite(r)):
            return np.full_like(y_fit, 1e6, dtype=float)
        return r

    res = least_squares(residuals, x0_3, method="trf", loss="huber", f_scale=1.0)
    log_k_quarter, log_t0, log_s = res.x

    unc = _param_uncertainty_from_jac(res, int(valid.sum()), 3)
    if unc is not None:
        std, corr = unc
        std_log_k_quarter = float(std[0])
        std_log_t0 = float(std[1])
        std_log_s = float(std[2])
        corr_logk_logt0 = float(corr[0, 1])
        corr_logk_logs = float(corr[0, 2])
        corr_logt0_logs = float(corr[1, 2])
    else:
        std_log_k_quarter = std_log_t0 = std_log_s = np.nan
        corr_logk_logt0 = corr_logk_logs = corr_logt0_logs = np.nan

    return {
        "log_params_q1": res.x,
        "k_quarter_q1": float(np.exp(log_k_quarter)),
        "t0_q1": float(np.exp(log_t0) + MIN_TI),
        "s_q1": float(np.exp(log_s)),
        "cost_q1": float(res.cost),
        "success_q1": bool(res.success),
        "n_q1": int(valid.sum()),
        "nfev_q1": int(getattr(res, "nfev", 0) or 0),
        "optimality_q1": float(getattr(res, "optimality", np.nan)),
        "status_q1": int(getattr(res, "status", -10)),
        "std_log_k_quarter_q1": std_log_k_quarter,
        "std_log_t0_q1": std_log_t0,
        "std_log_s_q1": std_log_s,
        "corr_logk_logt0_q1": corr_logk_logt0,
        "corr_logk_logs_q1": corr_logk_logs,
        "corr_logt0_logs_q1": corr_logt0_logs,
    }


def fit_sigmoid(t, y, ti, x0):
    """Legacy 4-param sqrt/quarter blend retained for archived comparisons."""
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 4:
        return {
            "log_params": np.full(4, np.nan),
            "k_sqrt": np.nan,
            "k_quarter": np.nan,
            "t0": np.nan,
            "s": np.nan,
            "cost": np.inf,
            "success": False,
            "n": int(valid.sum()),
        }

    t_fit = t[valid]
    y_fit = y[valid]

    def residuals(params):
        y_hat = spray_penetration_model_sigmoid(params, t_fit)
        r = y_hat - y_fit
        if not np.all(np.isfinite(r)):
            return np.full_like(y_fit, 1e6, dtype=float)
        return r

    res = least_squares(residuals, x0, method="trf", loss="huber", f_scale=1.0)
    log_k_sqrt, log_k_quarter, log_t0, log_s = res.x

    return {
        "log_params": res.x,
        "k_sqrt": float(np.exp(log_k_sqrt)),
        "k_quarter": float(np.exp(log_k_quarter)),
        "t0": float(np.exp(log_t0) + MIN_TI),
        "s": float(np.exp(log_s)),
        "cost": float(res.cost),
        "success": bool(res.success),
        "n": int(valid.sum()),
    }


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
    time_ms = time_s * 1e3

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
            fallback_delay_s=fallback_delay_s
        )
        delays_raw[plume_idx] = delay_s
        delay_sources[plume_idx] = delay_source if np.isfinite(area_delay) else "penetration_fallback"
        temp_series[plume_idx] = np.asarray(cleaned_serie, dtype=float)

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
    if series_df.empty or fit_df.empty:
        return series_df.iloc[0:0].copy()

    key_cols = ["file_path", "file_name", "file_stem", "plume_idx"]
    valid_keys = fit_df.loc[:, key_cols].drop_duplicates()
    return series_df.merge(valid_keys, on=key_cols, how="inner")


def build_wide_series_df(series_df, fit_df):
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


def _dur_groups(df):
    """Yield (filename_suffix, sub_df) for each injection-duration group.

    Single duration → one group with empty suffix (existing behaviour).
    Multiple durations (e.g. Nozzle0) → one group per duration so that
    each injection condition gets its own plot file (_T340, _T560, …).
    """
    col = "injection_duration_us"
    if col not in df.columns or df.empty:
        yield "", df
        return
    durs = sorted(pd.to_numeric(df[col], errors="coerce").dropna().unique())
    if len(durs) <= 1:
        yield "", df
        return
    dur_series = pd.to_numeric(df[col], errors="coerce")
    for d in durs:
        yield f"_T{int(d)}", df[dur_series == d]


def save_fit_plot(
    folder,
    plot_df,
    csv_files,
    out_plot_dir,
    plot_kind,
    penetration_source,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    cache = {}  # csv_path -> (time_s, time_ms, cleaned_series, inj_dur_s)

    # map filename and stem to path for fallback resolution
    name_to_paths = {}
    stem_to_paths = {}
    for p in csv_files:
        name_to_paths.setdefault(p.name, []).append(p)
        stem_to_paths.setdefault(p.stem, []).append(p)

    def resolve_csv(row):
        row_file_path = getattr(row, "file_path", "")
        if isinstance(row_file_path, str) and row_file_path != "":
            p = Path(row_file_path)
            if p.exists():
                return p
        row_file_name = str(getattr(row, "file_name", ""))
        cands = name_to_paths.get(row_file_name, [])
        if len(cands) == 1:
            return cands[0]
        if len(cands) == 0:
            row_file_stem = str(getattr(row, "file_stem", ""))
            cands = stem_to_paths.get(row_file_stem, [])
            if len(cands) == 1:
                return cands[0]
        return None

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="gray", linewidth=1.2, linestyle="-",  label="raw trace"),
        Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--", label="main q1 fit (∜t only)"),
    ]
    title_suffix = "solid=raw  --=main"
    if ABLATION_QUARTER_ONLY:
        legend_handles.append(
            Line2D([0], [0], color="gray", linewidth=1.2, linestyle=":", label="q1 fit (∜t only)")
        )
        title_suffix += "  ···=q1"

    saved_paths = []
    for _suffix, _group_df in _dur_groups(plot_df):
        rng = np.random.default_rng()
        plt.figure(figsize=(10, 6))
        has_curve = False

        for row in _group_df.itertuples(index=False):
            csv_path = resolve_csv(row)
            if csv_path is None:
                continue

            cache_key = str(csv_path.resolve())
            if cache_key not in cache:
                df_file = _read_csv_with_expanded_static_meta(csv_path)
                time_s, time_ms, cleaned_series, _, _, _ = prepare_cleaned_series(
                    df_file,
                    mm_per_px_scale=mm_per_px_scale,
                    fps_default=fps_default,
                    max_hydraulic_delay_frames=max_hydraulic_delay_frames,
                    delay_clip_half_window=delay_clip_half_window,
                    penetration_source=penetration_source,
                    replace_negative_with_zero=penetration_source["replace_negative_with_zero"],
                    diff_threshold_lower=DIFF_THRESHOLD_LOWER,
                    diff_threshold_upper=DIFF_THRESHOLD_UPPER,
                )
                inj_dur_s = float(df_file["injection_duration_us"].iloc[0]) * 1e-6
                cache[cache_key] = (time_s, time_ms, cleaned_series, inj_dur_s)

            time_s, time_ms, cleaned_series, inj_dur_s = cache[cache_key]
            plume_idx = int(row.plume_idx)
            if plume_idx < 0 or plume_idx >= cleaned_series.shape[0]:
                continue

            ts = time_s[plume_idx]
            tms = time_ms[plume_idx]
            raw_series = cleaned_series[plume_idx]
            valid_raw = np.isfinite(tms) & np.isfinite(raw_series)
            if not np.any(valid_raw):
                continue

            color = rng.random(3)
            plt.plot(tms[valid_raw], raw_series[valid_raw], alpha=0.65, linewidth=1.0, color=color)
            t_end = float(np.nanmax(ts) * PLOT_EXTRAP_FACTOR)
            t_extrap_s = np.linspace(0.0, t_end, PLOT_NUM_POINTS)
            draw_fit = (
                bool(getattr(row, "success", False))
                and np.isfinite(getattr(row, "log_k_sqrt", np.nan))
                and np.isfinite(getattr(row, "log_k_quarter", np.nan))
                and np.isfinite(getattr(row, "log_t0", np.nan))
                and np.isfinite(getattr(row, "log_s", np.nan))
            )
            if draw_fit:
                log_params = [row.log_k_sqrt, row.log_k_quarter, row.log_t0, row.log_s]
                y_extrap = spray_penetration_model_sigmoid(log_params, t_extrap_s)
                plt.plot(
                    1e3 * t_extrap_s,
                    y_extrap,
                    linestyle="--",
                    alpha=0.45,
                    linewidth=1.0,
                    color=color,
                )
            if ABLATION_QUARTER_ONLY:
                draw_q1 = (
                    bool(getattr(row, "success_q1", False))
                    and np.isfinite(getattr(row, "log_k_quarter_q1", np.nan))
                    and np.isfinite(getattr(row, "log_t0_q1", np.nan))
                    and np.isfinite(getattr(row, "log_s_q1", np.nan))
                )
                if draw_q1:
                    lp_q1 = [row.log_k_quarter_q1, row.log_t0_q1, row.log_s_q1]
                    y_q1 = spray_penetration_model_quarter_only(lp_q1, t_extrap_s)
                    plt.plot(
                        1e3 * t_extrap_s,
                        y_q1,
                        linestyle=":",
                        alpha=0.55,
                        linewidth=1.2,
                        color=color,
                    )
            has_curve = True

        if not has_curve:
            plt.text(0.5, 0.5, f"No {plot_kind} traces for this folder", ha="center", va="center")

        plt.legend(handles=legend_handles, fontsize=7, loc="upper left")
        dur_label = f" | T={_suffix[2:]}µs" if _suffix else ""
        plt.title(
            f"{folder.name}{dur_label}: {penetration_source['label']} {plot_kind}  [{title_suffix}]",
            fontsize=9,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Penetration (mm)")
        plt.grid(alpha=0.25)
        plt.ylim(0, PLOT_YLIM_MM)

        out_plot_path = out_plot_dir / f"{folder.name}{_suffix}.png"
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=140)
        plt.close()
        saved_paths.append(out_plot_path)

    return saved_paths


def save_raw_plot(
    folder,
    plot_df,
    csv_files,
    out_plot_dir,
    penetration_source,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    cache = {}

    name_to_paths = {}
    stem_to_paths = {}
    for p in csv_files:
        name_to_paths.setdefault(p.name, []).append(p)
        stem_to_paths.setdefault(p.stem, []).append(p)

    def resolve_csv(row):
        row_file_path = getattr(row, "file_path", "")
        if isinstance(row_file_path, str) and row_file_path != "":
            p = Path(row_file_path)
            if p.exists():
                return p
        row_file_name = str(getattr(row, "file_name", ""))
        cands = name_to_paths.get(row_file_name, [])
        if len(cands) == 1:
            return cands[0]
        if len(cands) == 0:
            row_file_stem = str(getattr(row, "file_stem", ""))
            cands = stem_to_paths.get(row_file_stem, [])
            if len(cands) == 1:
                return cands[0]
        return None

    saved_paths = []
    for _suffix, _group_df in _dur_groups(plot_df):
        rng = np.random.default_rng()
        plt.figure(figsize=(10, 6))
        has_curve = False

        for row in _group_df.itertuples(index=False):
            csv_path = resolve_csv(row)
            if csv_path is None:
                continue

            cache_key = str(csv_path.resolve())
            if cache_key not in cache:
                df_file = _read_csv_with_expanded_static_meta(csv_path)
                time_s, time_ms, cleaned_series, _, _, _ = prepare_cleaned_series(
                    df_file,
                    mm_per_px_scale=mm_per_px_scale,
                    fps_default=fps_default,
                    max_hydraulic_delay_frames=max_hydraulic_delay_frames,
                    delay_clip_half_window=delay_clip_half_window,
                    penetration_source=penetration_source,
                    replace_negative_with_zero=penetration_source["replace_negative_with_zero"],
                    diff_threshold_lower=DIFF_THRESHOLD_LOWER,
                    diff_threshold_upper=DIFF_THRESHOLD_UPPER,
                )
                cache[cache_key] = (time_s, time_ms, cleaned_series)

            time_s, time_ms, cleaned_series = cache[cache_key]
            plume_idx = int(row.plume_idx)
            if plume_idx < 0 or plume_idx >= cleaned_series.shape[0]:
                continue

            tms = time_ms[plume_idx]
            raw_series = cleaned_series[plume_idx]

            mask_time = tms <= 2.0
            valid_raw = np.isfinite(tms) & np.isfinite(raw_series) & mask_time
            if not np.any(valid_raw):
                continue

            color = rng.random(3)
            plt.plot(tms[valid_raw], raw_series[valid_raw], alpha=0.65, linewidth=1.0, color=color)
            has_curve = True

        if not has_curve:
            plt.text(1.0, 0.5, "No traces for this folder", ha="center", va="center")

        dur_label = f" | T={_suffix[2:]}µs" if _suffix else ""
        plt.title(
            f"{folder.name}{dur_label}: {penetration_source['label']} aligned raw traces (0-2ms)"
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Penetration (mm)")
        plt.grid(alpha=0.25)
        plt.xlim(0, 2.0)
        plt.ylim(0, PLOT_YLIM_MM)

        out_plot_path = out_plot_dir / f"{folder.name}{_suffix}.png"
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=140)
        plt.close()
        saved_paths.append(out_plot_path)

    return saved_paths

def process_folder(
    folder,
    out_all_dir,
    out_clean_dir,
    out_series_all_dir,
    out_series_clean_dir,
    out_series_wide_all_dir,
    out_series_wide_clean_dir,
    out_plots_clean_dir,
    out_plots_flagged_dir,
    out_plots_raw_all_dir,
    penetration_source,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"Skip {folder.name}: no csv files.")
        return

    rows = []
    series_rows = []
    for file_path in csv_files:
        df_file = _read_csv_with_expanded_static_meta(file_path)
        time_s, time_ms, cleaned_series, delays_raw, delays_used, delay_sources = prepare_cleaned_series(
            df_file,
            mm_per_px_scale=mm_per_px_scale,
            fps_default=fps_default,
            max_hydraulic_delay_frames=max_hydraulic_delay_frames,
            delay_clip_half_window=delay_clip_half_window,
            penetration_source=penetration_source,
            replace_negative_with_zero=penetration_source["replace_negative_with_zero"],
            diff_threshold_lower=DIFF_THRESHOLD_LOWER,
            diff_threshold_upper=DIFF_THRESHOLD_UPPER,
        )
        series_rows.extend(
            collect_series_rows(
                file_path=file_path,
                time_s=time_s,
                time_ms=time_ms,
                cleaned_series=cleaned_series,
                delays_raw=delays_raw,
                delays_used=delays_used,
                delay_sources=delay_sources,
            )
        )

        meta = {}
        for col in META_COLS:
            meta[col] = df_file[col].iloc[0] if col in df_file.columns else np.nan

        inj_dur_s = float(meta["injection_duration_us"]) * 1e-6
        x0_q1 = np.log([1.0, max(2.0 * inj_dur_s, 1e-9), 1.0])

        number_of_plumes = cleaned_series.shape[0]
        for plume_idx in range(number_of_plumes):
            series = cleaned_series[plume_idx]
            ts = time_s[plume_idx]
            if int((np.isfinite(ts) & np.isfinite(series)).sum()) < MIN_SERIES_POINTS:
                continue
            valid = np.isfinite(ts) & np.isfinite(series)
            t_max_s = float(np.nanmax(ts)) if np.any(np.isfinite(ts)) else float('nan')
            fit = fit_quarter_only(ts, series, x0_q1)
            log_params = fit["log_params_q1"]
            if fit["success_q1"] and np.all(np.isfinite(log_params)) and np.any(valid):
                y_true = series[valid]
                y_hat = spray_penetration_model_quarter_only(log_params, ts[valid])
                rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
                ss_res = float(np.sum((y_true - y_hat) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                rmse = np.nan
                r2 = np.nan
                log_params = np.full(3, np.nan)

            log_k_quarter, log_t0, log_s = log_params

            row_dict = {
                "fit_model": FIT_MODEL_NAME,
                "penetration_source": penetration_source["key"],
                "file_path": str(file_path.resolve()),
                "file_name": file_path.name,
                "file_stem": file_path.stem,
                "plume_idx": plume_idx,
                "delay_frames": delays_used[plume_idx],
                "delay_frames_raw": delays_raw[plume_idx],
                "delay_frames_used": delays_used[plume_idx],
                "delay_source": delay_sources[plume_idx],
                "k_sqrt": K_SQRT_SENTINEL,
                "k_quarter": fit["k_quarter_q1"],
                "t0": fit["t0_q1"],
                "s": fit["s_q1"],
                "cost": fit["cost_q1"],
                "success": fit["success_q1"],
                "n": fit["n_q1"],
                "rmse": rmse,
                "r2": r2,
                "log_k_sqrt": LOG_K_SQRT_SENTINEL,
                "log_k_quarter": log_k_quarter,
                "log_t0": log_t0,
                "log_s": log_s,
                "t_max_s": t_max_s,
                "nfev": fit["nfev_q1"],
                "optimality": fit["optimality_q1"],
                "status": fit["status_q1"],
                "std_log_k_quarter": fit["std_log_k_quarter_q1"],
                "std_log_t0": fit["std_log_t0_q1"],
                "std_log_s": fit["std_log_s_q1"],
                "corr_logk_logt0": fit["corr_logk_logt0_q1"],
                "corr_logk_logs": fit["corr_logk_logs_q1"],
                "corr_logt0_logs": fit["corr_logt0_logs_q1"],
                **meta,
            }
            rows.append(row_dict)

    results_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    masked_df, clean_df, flagged_df = apply_filter_masking(results_df, group_cols=MASK_GROUP_COLS)
    series_df = pd.DataFrame(series_rows).replace([np.inf, -np.inf], np.nan)
    series_clean_df = filter_series_df(series_df, clean_df)
    series_wide_all_df = build_wide_series_df(series_df, masked_df)
    series_wide_clean_df = build_wide_series_df(series_clean_df, clean_df)

    out_all_path = out_all_dir / f"{folder.name}.csv"
    out_clean_path = out_clean_dir / f"{folder.name}.csv"
    out_flagged_path = out_all_dir / f"{folder.name}_flagged.csv"
    out_series_all_path = out_series_all_dir / f"{folder.name}.csv"
    out_series_clean_path = out_series_clean_dir / f"{folder.name}.csv"
    out_series_wide_all_path = out_series_wide_all_dir / f"{folder.name}.csv"
    out_series_wide_clean_path = out_series_wide_clean_dir / f"{folder.name}.csv"
    clean_plot_paths = save_fit_plot(
        folder,
        clean_df,
        csv_files,
        out_plots_clean_dir,
        "clean",
        penetration_source,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )
    flagged_plot_paths = save_fit_plot(
        folder,
        flagged_df,
        csv_files,
        out_plots_flagged_dir,
        "flagged",
        penetration_source,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )
    raw_plot_paths = save_raw_plot(
        folder,
        results_df,
        csv_files,
        out_plots_raw_all_dir,
        penetration_source,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )

    masked_df.to_csv(out_all_path, index=False)
    clean_df.to_csv(out_clean_path, index=False)
    flagged_df.to_csv(out_flagged_path, index=False)
    series_df.to_csv(out_series_all_path, index=False)
    series_clean_df.to_csv(out_series_clean_path, index=False)
    series_wide_all_df.to_csv(out_series_wide_all_path, index=False)
    series_wide_clean_df.to_csv(out_series_wide_clean_path, index=False)

    _fmt_paths = lambda paths: ", ".join(p.name for p in paths)
    print(
        f"[{penetration_source['key']}] Saved {out_all_path.name} ({len(masked_df)} total rows), "
        f"{out_clean_path.name} ({len(clean_df)} clean), "
        f"{out_flagged_path.name} ({len(flagged_df)} flagged), "
        f"{out_series_all_path.name} ({len(series_df)} series rows), "
        f"{out_series_clean_path.name} ({len(series_clean_df)} clean series rows), "
        f"{out_series_wide_all_path.name} ({len(series_wide_all_df)} wide rows), "
        f"{out_series_wide_clean_path.name} ({len(series_wide_clean_df)} clean wide rows), "
        f"{_fmt_paths(clean_plot_paths)} (clean-curve plot(s)), "
        f"{_fmt_paths(flagged_plot_paths)} (flagged-curve plot(s)), "
        f"{_fmt_paths(raw_plot_paths)} (raw-all plot(s)) from {len(csv_files)} files"
    )

    _n = len(masked_df)
    stats = {
        "nozzle": folder.parent.name,
        "folder": folder.name,
        "penetration_source": penetration_source["key"],
        "fit_model": FIT_MODEL_NAME,
        "n_total": _n,
        "n_clean": len(clean_df),
        "n_flagged": len(flagged_df),
        "success_main": np.nan,
        "success_rate_main_pct": np.nan,
        "rmse_clean_main_mm": np.nan,
        "rmse_flagged_main_mm": np.nan,
    }
    if _n > 0:
        _rmse_main = pd.to_numeric(masked_df["rmse"], errors="coerce")
        _ok = int(
            ((~masked_df["flag_bad_fit"].fillna(True)) & (_rmse_main < RMSE_SUCCESS_THRESHOLD_MM)).sum()
        )
        _pct = 100.0 * _ok / _n
        _rmse_c = clean_df["rmse"].median() if len(clean_df) > 0 else float("nan")
        _rmse_f = flagged_df["rmse"].median() if len(flagged_df) > 0 else float("nan")
        stats.update({
            "success_main": _ok,
            "success_rate_main_pct": round(_pct, 1),
            "rmse_clean_main_mm": round(_rmse_c, 3) if np.isfinite(_rmse_c) else np.nan,
            "rmse_flagged_main_mm": round(_rmse_f, 3) if np.isfinite(_rmse_f) else np.nan,
        })
        _line = (
            f"  [fit-report] main/{FIT_MODEL_NAME}  success {_ok}/{_n} ({_pct:.1f}%)  "
            f"RMSE median  clean {_rmse_c:.2f} mm  flagged {_rmse_f:.2f} mm"
        )
        if ABLATION_QUARTER_ONLY and "flag_bad_fit_q1" in masked_df.columns:
            _rmse_q1 = pd.to_numeric(masked_df["rmse_q1"], errors="coerce")
            _ok_q1 = int(
                ((~masked_df["flag_bad_fit_q1"].fillna(True)) & (_rmse_q1 < RMSE_SUCCESS_THRESHOLD_MM)).sum()
            )
            _pct_q1 = 100.0 * _ok_q1 / _n
            _rmse_c_q1 = clean_df["rmse_q1"].median() if "rmse_q1" in clean_df.columns and len(clean_df) > 0 else float("nan")
            _rmse_f_q1 = flagged_df["rmse_q1"].median() if "rmse_q1" in flagged_df.columns and len(flagged_df) > 0 else float("nan")
            stats.update({
                "success_q1": _ok_q1,
                "success_rate_q1_pct": round(_pct_q1, 1),
                "rmse_clean_q1_mm": round(_rmse_c_q1, 3) if np.isfinite(_rmse_c_q1) else np.nan,
                "rmse_flagged_q1_mm": round(_rmse_f_q1, 3) if np.isfinite(_rmse_f_q1) else np.nan,
            })
            _line += (
                f"\n  [fit-report] q1    success {_ok_q1}/{_n} ({_pct_q1:.1f}%)  "
                f"RMSE median  clean {_rmse_c_q1:.2f} mm  flagged {_rmse_f_q1:.2f} mm"
            )
        print(_line)
    return stats


def _process_folder_worker(kwargs):
    """Module-level wrapper so ProcessPoolExecutor can pickle the task."""
    return process_folder(**kwargs)


def get_dataset_settings(name):
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


def main():
    enabled_sources = get_enabled_penetration_sources()
    if not enabled_sources:
        raise ValueError("At least one penetration-series switch must be enabled.")

    # --- Pass 1: collect tasks and pre-create all output directories ---
    tasks = []
    for name in names:
        settings = get_dataset_settings(name)
        mm_per_px_scale = 90.0 / settings["or_mm_per_px_reference"]
        root = data_root / name
        out_dir = data_out_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if not root.exists():
            print(f"Skipping missing dataset root: {root}")
            continue

        subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not subdirs:
            raise FileNotFoundError(f"No subdirs found in {root}")

        print(f"Found {len(subdirs)} subdirs in {root}")
        for folder in subdirs:
            for penetration_source in enabled_sources:
                metric_out_dir = out_dir / penetration_source["key"]
                out_all_dir = metric_out_dir / "all"
                out_clean_dir = metric_out_dir / "clean"
                out_series_all_dir = metric_out_dir / "series_all"
                out_series_clean_dir = metric_out_dir / "series_clean"
                out_series_wide_all_dir = metric_out_dir / "series_wide_all"
                out_series_wide_clean_dir = metric_out_dir / "series_wide_clean"
                out_plots_clean_dir = metric_out_dir / "plots_clean"
                out_plots_flagged_dir = metric_out_dir / "plots_flagged"
                out_plots_raw_all_dir = metric_out_dir / "plots_raw_all"
                for d in (
                    out_all_dir, out_clean_dir,
                    out_series_all_dir, out_series_clean_dir,
                    out_series_wide_all_dir, out_series_wide_clean_dir,
                    out_plots_clean_dir, out_plots_flagged_dir, out_plots_raw_all_dir,
                ):
                    d.mkdir(parents=True, exist_ok=True)

                tasks.append({
                    "folder": folder,
                    "out_all_dir": out_all_dir,
                    "out_clean_dir": out_clean_dir,
                    "out_series_all_dir": out_series_all_dir,
                    "out_series_clean_dir": out_series_clean_dir,
                    "out_series_wide_all_dir": out_series_wide_all_dir,
                    "out_series_wide_clean_dir": out_series_wide_clean_dir,
                    "out_plots_clean_dir": out_plots_clean_dir,
                    "out_plots_flagged_dir": out_plots_flagged_dir,
                    "out_plots_raw_all_dir": out_plots_raw_all_dir,
                    "penetration_source": penetration_source,
                    "mm_per_px_scale": mm_per_px_scale,
                    "fps_default": settings["fps_default"],
                    "max_hydraulic_delay_frames": settings["max_hydraulic_delay_frames"],
                    "delay_clip_half_window": settings["delay_clip_half_window"],
                })

    # --- Pass 2: dispatch ---
    n_workers = N_WORKERS or os.cpu_count()
    if n_workers == 1:
        results = [_process_folder_worker(kwargs) for kwargs in tasks]
    else:
        print(f"Launching {n_workers} workers for {len(tasks)} folder tasks …")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_folder_worker, tasks))

    # --- Pass 3: write fit report ---
    all_stats = [s for s in results if isinstance(s, dict)]
    if all_stats:
        report_df = pd.DataFrame(all_stats)
        report_path = data_out_dir / "fit_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\nFit report saved -> {report_path}  ({len(report_df)} rows)")


if __name__ == "__main__":
    main()
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from summarize_dataset import main as _summarize_main
    _summarize_main()
    try:
        from summarize_filter_survival import main as _survival_main
        _survival_main()
    except Exception as _e:
        print(f"[summarize_filter_survival] skipped: {_e}")
    try:
        from fit_diagnostics import main as _diagnostics_main
        _diagnostics_main()
    except Exception as _e:
        print(f"[fit_diagnostics] skipped: {_e}")
