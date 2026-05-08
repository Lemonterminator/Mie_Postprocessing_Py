from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from .models import (
        CALIBRATED_VARIANTS, LITERATURE_CONSTANTS, VARIANT_TO_CONSTANTS_KEY, ZHOU_VARIANTS,
        HAParams, predict,
    )
except ImportError:  # pragma: no cover
    from models import (
        CALIBRATED_VARIANTS, LITERATURE_CONSTANTS, VARIANT_TO_CONSTANTS_KEY, ZHOU_VARIANTS,
        HAParams, predict,
    )


@dataclass(frozen=True)
class ResidualUncertainty:
    mode: str
    global_sigma_mm: float
    time_bin_ms: float
    sigma_floor_mm: float
    bin_centers_ms: np.ndarray
    bin_sigma_mm: np.ndarray
    bin_count: np.ndarray


def robust_sigma(values: np.ndarray, floor: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(floor)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(arr)) if arr.size > 1 else 0.0
    return float(max(sigma, floor))


def fit_literature(variant: str) -> tuple[HAParams, dict[str, Any]]:
    constants_key = VARIANT_TO_CONSTANTS_KEY[variant]
    consts = LITERATURE_CONSTANTS[constants_key]
    use_zhou = variant in ZHOU_VARIANTS
    params = HAParams(
        kv=float(consts["kv"]),
        kp=float(consts["kp"]),
        kbt=float(consts["kbt"]),
        variant=variant,
        use_zhou=use_zhou,
    )
    fit_info: dict[str, Any] = {
        "success": True,
        "n_train_points": 0,
        "train_rmse_mm": float("nan"),
        "train_mae_mm": float("nan"),
        "train_bias_mm": float("nan"),
        "loss": "none (literature constants)",
        "constants_source": constants_key,
    }
    return params, fit_info


def _residuals_fn(
    theta: np.ndarray,
    df_train: pd.DataFrame,
    variant: str,
    use_zhou: bool,
    y: np.ndarray,
) -> np.ndarray:
    kv = float(np.exp(theta[0]))
    kp = float(np.exp(theta[1]))
    kbt = float(np.exp(theta[2]))
    params = HAParams(kv=kv, kp=kp, kbt=kbt, variant=variant, use_zhou=use_zhou)
    r = predict(df_train, params) - y
    if not np.all(np.isfinite(r)):
        return np.full_like(y, 1e6, dtype=float)
    return r


def fit_ha_calibrated(
    df_train: pd.DataFrame,
    variant: str,
    config: dict[str, Any],
) -> tuple[HAParams, dict[str, Any]]:
    if df_train.empty:
        raise ValueError("Cannot fit H-A calibrated baseline on an empty train split.")
    use_zhou = variant in ZHOU_VARIANTS
    f_scale = float(config.get("huber_f_scale_mm", 1.0))
    y = pd.to_numeric(df_train["pen_true_mm"], errors="coerce").to_numpy(dtype=float)
    init = LITERATURE_CONSTANTS["hiroyasu"]
    x0 = np.array([np.log(init["kv"]), np.log(init["kp"]), np.log(init["kbt"])], dtype=float)
    lb = np.array([np.log(0.05), np.log(0.3), np.log(1.0)], dtype=float)
    ub = np.array([np.log(10.0), np.log(30.0), np.log(500.0)], dtype=float)
    res = least_squares(
        _residuals_fn,
        x0,
        args=(df_train, variant, use_zhou, y),
        method="trf",
        loss="huber",
        f_scale=f_scale,
        bounds=(lb, ub),
    )
    kv = float(np.exp(res.x[0]))
    kp = float(np.exp(res.x[1]))
    kbt = float(np.exp(res.x[2]))
    params = HAParams(kv=kv, kp=kp, kbt=kbt, variant=variant, use_zhou=use_zhou)
    r = predict(df_train, params) - y
    fit_info: dict[str, Any] = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "optimality": float(getattr(res, "optimality", np.nan)),
        "n_train_points": int(len(df_train)),
        "train_rmse_mm": float(np.sqrt(np.mean(r * r))),
        "train_mae_mm": float(np.mean(np.abs(r))),
        "train_bias_mm": float(np.mean(r)),
        "loss": "huber",
        "huber_f_scale_mm": f_scale,
    }
    return params, fit_info


def fit_model(
    df_train: pd.DataFrame,
    variant: str,
    config: dict[str, Any],
) -> tuple[HAParams, dict[str, Any]]:
    if variant in CALIBRATED_VARIANTS:
        return fit_ha_calibrated(df_train, variant, config)
    params, fit_info = fit_literature(variant)
    # evaluate literature metrics on training data
    if not df_train.empty:
        y = pd.to_numeric(df_train["pen_true_mm"], errors="coerce").to_numpy(dtype=float)
        r = predict(df_train, params) - y
        fit_info["n_train_points"] = int(len(df_train))
        fit_info["train_rmse_mm"] = float(np.sqrt(np.mean(r * r)))
        fit_info["train_mae_mm"] = float(np.mean(np.abs(r)))
        fit_info["train_bias_mm"] = float(np.mean(r))
    return params, fit_info


def calibrate_residual_uncertainty(
    df_cal: pd.DataFrame,
    params: HAParams,
    config: dict[str, Any],
) -> ResidualUncertainty:
    unc_cfg = dict(config.get("uncertainty", {}))
    sigma_floor = float(unc_cfg.get("sigma_floor_mm", 0.5))
    mode = str(unc_cfg.get("residual_mode", "time_binned"))
    time_bin_ms = float(unc_cfg.get("time_bin_ms", 0.1))
    min_bin_count = int(unc_cfg.get("min_bin_count", 30))
    pred = predict(df_cal, params)
    residual = pred - pd.to_numeric(df_cal["pen_true_mm"], errors="coerce").to_numpy(dtype=float)
    global_sigma = robust_sigma(residual, floor=sigma_floor)
    if mode != "time_binned":
        return ResidualUncertainty(
            mode="global",
            global_sigma_mm=global_sigma,
            time_bin_ms=time_bin_ms,
            sigma_floor_mm=sigma_floor,
            bin_centers_ms=np.array([], dtype=float),
            bin_sigma_mm=np.array([], dtype=float),
            bin_count=np.array([], dtype=int),
        )
    t = pd.to_numeric(df_cal["time_ms"], errors="coerce").to_numpy(dtype=float)
    max_time = float(np.nanmax(t)) if np.isfinite(t).any() else 5.0
    n_bins = max(int(np.ceil(max_time / time_bin_ms)), 1)
    bin_edges = np.arange(n_bins + 1, dtype=float) * time_bin_ms
    bin_ids = np.clip(np.digitize(t, bin_edges, right=False) - 1, 0, n_bins - 1)
    sigmas = np.full(n_bins, global_sigma, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for idx in range(n_bins):
        vals = residual[bin_ids == idx]
        vals = vals[np.isfinite(vals)]
        counts[idx] = int(vals.size)
        if vals.size >= min_bin_count:
            sigmas[idx] = robust_sigma(vals, floor=sigma_floor)
    centers = bin_edges[:-1] + 0.5 * time_bin_ms
    return ResidualUncertainty(
        mode="time_binned",
        global_sigma_mm=global_sigma,
        time_bin_ms=time_bin_ms,
        sigma_floor_mm=sigma_floor,
        bin_centers_ms=centers,
        bin_sigma_mm=sigmas,
        bin_count=counts,
    )


def residual_sigma_for_points(df: pd.DataFrame, unc: ResidualUncertainty) -> np.ndarray:
    if unc.mode != "time_binned" or unc.bin_sigma_mm.size == 0:
        return np.full(len(df), unc.global_sigma_mm, dtype=float)
    bins = np.floor(pd.to_numeric(df["time_ms"], errors="coerce").to_numpy(dtype=float) / unc.time_bin_ms).astype(int)
    bins = np.clip(bins, 0, len(unc.bin_sigma_mm) - 1)
    return np.maximum(unc.bin_sigma_mm[bins], unc.sigma_floor_mm)


def bootstrap_params(
    df_train: pd.DataFrame,
    variant: str,
    config: dict[str, Any],
) -> tuple[list[HAParams], pd.DataFrame]:
    """Bootstrap only runs for calibrated variants; literature constants have no parameter uncertainty."""
    unc_cfg = dict(config.get("uncertainty", {}))
    n_boot = int(unc_cfg.get("bootstrap_n", 0))
    if n_boot <= 0 or variant not in CALIBRATED_VARIANTS:
        return [], pd.DataFrame()
    group_col = "split_group_id" if str(unc_cfg.get("bootstrap_group", "condition")) == "condition" else "traj_key"
    rng = np.random.default_rng(int(unc_cfg.get("bootstrap_seed", 20260508)))
    groups = np.asarray(pd.Series(df_train[group_col]).drop_duplicates().tolist(), dtype=object)
    grouped_indices = {
        group: idx.to_numpy()
        for group, idx in df_train.groupby(group_col, dropna=False).groups.items()
    }
    params_list: list[HAParams] = []
    rows: list[dict[str, Any]] = []
    for boot_idx in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_idx = np.concatenate([grouped_indices[group] for group in sampled_groups])
        boot_df = df_train.iloc[sampled_idx].reset_index(drop=True)
        try:
            params, info = fit_ha_calibrated(boot_df, variant, config)
        except Exception as exc:
            rows.append({"bootstrap_idx": boot_idx, "success": False, "error": str(exc)})
            continue
        params_list.append(params)
        rows.append({
            "bootstrap_idx": boot_idx,
            "success": bool(info.get("success", True)),
            "kv": float(params.kv),
            "kp": float(params.kp),
            "kbt": float(params.kbt),
            "train_rmse_mm": float(info.get("train_rmse_mm", np.nan)),
        })
    return params_list, pd.DataFrame(rows)


def parameter_sigma_for_points(
    df_eval: pd.DataFrame,
    boot_params: list[HAParams],
) -> np.ndarray:
    if not boot_params:
        return np.zeros(len(df_eval), dtype=float)
    preds = np.empty((len(boot_params), len(df_eval)), dtype=np.float32)
    for idx, params in enumerate(boot_params):
        preds[idx, :] = predict(df_eval, params).astype(np.float32)
    return np.std(preds, axis=0, ddof=1).astype(float) if len(boot_params) > 1 else np.zeros(len(df_eval), dtype=float)


def uncertainty_to_frame(unc: ResidualUncertainty) -> pd.DataFrame:
    if unc.bin_sigma_mm.size == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "time_bin_center_ms": unc.bin_centers_ms,
        "sigma_resid_mm": unc.bin_sigma_mm,
        "n_calibration_points": unc.bin_count,
    })
