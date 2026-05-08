from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:  # Allows running the file directly during quick debugging.
    from .models import NSParams, design_vector, predict
except ImportError:  # pragma: no cover
    from models import NSParams, design_vector, predict


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


def _initial_k(df: pd.DataFrame, *, delay_s: float, use_angle_factor: bool, config: dict[str, Any]) -> float:
    x = design_vector(
        df,
        delay_s=delay_s,
        use_angle_factor=use_angle_factor,
        angle_factor_floor=float(config.get("angle_factor_floor", 0.25)),
        angle_factor_ceiling=float(config.get("angle_factor_ceiling", 4.0)),
    )
    y = pd.to_numeric(df["pen_true_mm"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not np.any(valid):
        return 1.0
    denom = float(np.dot(x[valid], x[valid]))
    if denom <= 0:
        return 1.0
    k = float(np.dot(x[valid], y[valid]) / denom)
    return max(k, 1e-9)


def fit_naber_siebers(df_train: pd.DataFrame, config: dict[str, Any]) -> tuple[NSParams, dict[str, Any]]:
    if df_train.empty:
        raise ValueError("Cannot fit Naber--Siebers baseline on an empty train split.")

    variant = str(config.get("variant", "ns_delay"))
    use_angle_factor = bool(config.get("use_train_angle", False) or config.get("use_oracle_angle", False))
    fit_delay = bool(config.get("fit_delay", False))
    f_scale = float(config.get("huber_f_scale_mm", 1.0))
    delay_bounds_ms = config.get("delay_bounds_ms", [-0.1, 0.8])
    delay_bounds_s = (float(delay_bounds_ms[0]) * 1e-3, float(delay_bounds_ms[1]) * 1e-3)

    k0 = _initial_k(df_train, delay_s=0.0, use_angle_factor=use_angle_factor, config=config)
    y = pd.to_numeric(df_train["pen_true_mm"], errors="coerce").to_numpy(dtype=float)

    def residuals(theta: np.ndarray) -> np.ndarray:
        log_k = float(theta[0])
        delay_s = float(theta[1]) if fit_delay else 0.0
        params = NSParams(k=float(np.exp(log_k)), delay_s=delay_s, variant=variant, use_angle_factor=use_angle_factor)
        r = predict(
            df_train,
            params,
            angle_factor_floor=float(config.get("angle_factor_floor", 0.25)),
            angle_factor_ceiling=float(config.get("angle_factor_ceiling", 4.0)),
        ) - y
        if not np.all(np.isfinite(r)):
            return np.full_like(y, 1e6, dtype=float)
        return r

    if fit_delay:
        x0 = np.array([np.log(k0), 0.0], dtype=float)
        bounds = (np.array([-10.0, delay_bounds_s[0]], dtype=float), np.array([15.0, delay_bounds_s[1]], dtype=float))
    else:
        x0 = np.array([np.log(k0)], dtype=float)
        bounds = (np.array([-10.0], dtype=float), np.array([15.0], dtype=float))

    res = least_squares(residuals, x0, method="trf", loss="huber", f_scale=f_scale, bounds=bounds)
    delay_s = float(res.x[1]) if fit_delay else 0.0
    params = NSParams(k=float(np.exp(res.x[0])), delay_s=delay_s, variant=variant, use_angle_factor=use_angle_factor)
    r = residuals(res.x)
    fit_info = {
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
        "loss": str(config.get("fit_loss", "huber")),
        "huber_f_scale_mm": f_scale,
    }
    return params, fit_info


def fit_angle_lookup(df_train: pd.DataFrame, default_angle_deg: float) -> dict[str, Any]:
    finite = df_train.loc[np.isfinite(pd.to_numeric(df_train["observed_cone_angle_deg"], errors="coerce"))].copy()
    if finite.empty:
        return {"global": float(default_angle_deg), "by_dataset_key": {}}
    finite["observed_cone_angle_deg"] = pd.to_numeric(finite["observed_cone_angle_deg"], errors="coerce")
    global_median = float(finite["observed_cone_angle_deg"].median())
    by_dataset = {
        str(k): float(v)
        for k, v in finite.groupby("dataset_key", dropna=False)["observed_cone_angle_deg"].median().items()
    }
    return {"global": global_median, "by_dataset_key": by_dataset}


def apply_angle_policy(df: pd.DataFrame, config: dict[str, Any], angle_lookup: dict[str, Any] | None = None) -> pd.DataFrame:
    out = df.copy()
    default_angle = float(config.get("angle_default_deg", 14.0))
    if bool(config.get("use_oracle_angle", False)):
        out["angle_for_prediction_deg"] = pd.to_numeric(out["observed_cone_angle_deg"], errors="coerce").fillna(default_angle)
        return out

    if bool(config.get("use_train_angle", False)):
        lookup = angle_lookup or {"global": default_angle, "by_dataset_key": {}}
        global_angle = float(lookup.get("global", default_angle))
        by_dataset = lookup.get("by_dataset_key", {})
        out["angle_for_prediction_deg"] = out["dataset_key"].map(lambda key: by_dataset.get(str(key), global_angle))
        return out

    out["angle_for_prediction_deg"] = default_angle
    return out


def calibrate_residual_uncertainty(
    df_cal: pd.DataFrame,
    params: NSParams,
    config: dict[str, Any],
) -> ResidualUncertainty:
    unc_cfg = dict(config.get("uncertainty", {}))
    sigma_floor = float(unc_cfg.get("sigma_floor_mm", 0.5))
    mode = str(unc_cfg.get("residual_mode", "time_binned"))
    time_bin_ms = float(unc_cfg.get("time_bin_ms", 0.1))
    min_bin_count = int(unc_cfg.get("min_bin_count", 30))

    pred = predict(
        df_cal,
        params,
        angle_factor_floor=float(config.get("angle_factor_floor", 0.25)),
        angle_factor_ceiling=float(config.get("angle_factor_ceiling", 4.0)),
    )
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
    config: dict[str, Any],
) -> tuple[list[NSParams], pd.DataFrame]:
    unc_cfg = dict(config.get("uncertainty", {}))
    n_boot = int(unc_cfg.get("bootstrap_n", 0))
    if n_boot <= 0:
        return [], pd.DataFrame()
    group_col = "split_group_id" if str(unc_cfg.get("bootstrap_group", "condition")) == "condition" else "traj_key"
    rng = np.random.default_rng(int(unc_cfg.get("bootstrap_seed", 20260429)))
    groups = np.asarray(pd.Series(df_train[group_col]).drop_duplicates().tolist(), dtype=object)
    grouped_indices = {group: idx.to_numpy() for group, idx in df_train.groupby(group_col, dropna=False).groups.items()}
    params_list: list[NSParams] = []
    rows: list[dict[str, Any]] = []
    for boot_idx in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_idx = np.concatenate([grouped_indices[group] for group in sampled_groups])
        boot_df = df_train.iloc[sampled_idx].reset_index(drop=True)
        try:
            params, info = fit_naber_siebers(boot_df, config)
        except Exception as exc:
            rows.append({"bootstrap_idx": boot_idx, "success": False, "error": str(exc)})
            continue
        params_list.append(params)
        rows.append(
            {
                "bootstrap_idx": boot_idx,
                "success": bool(info.get("success", True)),
                "k": float(params.k),
                "delay_s": float(params.delay_s),
                "delay_ms": float(params.delay_s * 1e3),
                "train_rmse_mm": float(info.get("train_rmse_mm", np.nan)),
            }
        )
    return params_list, pd.DataFrame(rows)


def parameter_sigma_for_points(
    df_eval: pd.DataFrame,
    boot_params: list[NSParams],
    config: dict[str, Any],
) -> np.ndarray:
    if not boot_params:
        return np.zeros(len(df_eval), dtype=float)
    preds = np.empty((len(boot_params), len(df_eval)), dtype=np.float32)
    for idx, params in enumerate(boot_params):
        preds[idx, :] = predict(
            df_eval,
            params,
            angle_factor_floor=float(config.get("angle_factor_floor", 0.25)),
            angle_factor_ceiling=float(config.get("angle_factor_ceiling", 4.0)),
        ).astype(np.float32)
    return np.std(preds, axis=0, ddof=1).astype(float) if len(boot_params) > 1 else np.zeros(len(df_eval), dtype=float)


def uncertainty_to_frame(unc: ResidualUncertainty) -> pd.DataFrame:
    if unc.bin_sigma_mm.size == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "time_bin_center_ms": unc.bin_centers_ms,
            "sigma_resid_mm": unc.bin_sigma_mm,
            "n_calibration_points": unc.bin_count,
        }
    )

