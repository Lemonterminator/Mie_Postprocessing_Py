"""Reusable q1 quarter-root spray-penetration model helpers.

The helpers here mirror the production q1 model used by ``fit_raw_data.py`` so
secondary workflows can fit derived curves without importing the full raw-data
pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


MIN_TI = 0.0
LOG_K_SQRT_SENTINEL = -500.0
K_SQRT_SENTINEL = 0.0


def spray_penetration_model_quarter_only(params: np.ndarray | list[float], t_s: np.ndarray) -> np.ndarray:
    """Evaluate ``expit((t-t0)/s) * k_quarter * t**0.25`` in seconds/mm."""
    log_k_quarter, log_t0, log_s = params
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TI
    s = np.exp(log_s)
    t = np.clip(np.asarray(t_s, dtype=float), 1e-9, None)
    w = expit((t - t0) / s)
    return w * k_quarter * np.power(t, 0.25)


def _param_uncertainty_from_jac(res, n_valid: int, n_params: int):
    """Estimate log-parameter standard errors/correlations from a least-squares fit."""
    try:
        if res.jac is None or n_valid <= n_params:
            return None
        jac = np.asarray(res.jac, dtype=float)
        if jac.size == 0 or not np.all(np.isfinite(jac)):
            return None
        residuals = np.asarray(res.fun, dtype=float)
        sigma2 = float(np.sum(residuals * residuals) / max(n_valid - n_params, 1))
        cov = np.linalg.inv(jac.T @ jac) * sigma2
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


def fit_quarter_only(t_s: np.ndarray, y_mm: np.ndarray, x0: np.ndarray | None = None) -> dict[str, object]:
    """Fit the q1 quarter-only model to finite positive observations."""
    t_s = np.asarray(t_s, dtype=float)
    y_mm = np.asarray(y_mm, dtype=float)
    valid = np.isfinite(t_s) & np.isfinite(y_mm) & (t_s > 0.0) & (y_mm > 0.0)
    nan_result: dict[str, object] = {
        "log_params": np.full(3, np.nan),
        "k_quarter": np.nan,
        "t0": np.nan,
        "s": np.nan,
        "cost": np.inf,
        "success": False,
        "n": int(valid.sum()),
        "nfev": 0,
        "optimality": np.nan,
        "status": -10,
        "std_log_k_quarter": np.nan,
        "std_log_t0": np.nan,
        "std_log_s": np.nan,
        "corr_logk_logt0": np.nan,
        "corr_logk_logs": np.nan,
        "corr_logt0_logs": np.nan,
    }
    if valid.sum() < 3:
        return nan_result

    t_fit = t_s[valid]
    y_fit = y_mm[valid]
    if x0 is None:
        k0 = max(float(np.nanmedian(y_fit) / np.power(np.nanmedian(t_fit), 0.25)), 1e-6)
        t0 = max(float(np.nanpercentile(t_fit, 15)), 1e-9)
        s0 = max(float(np.nanmedian(np.diff(np.unique(np.sort(t_fit)))) if len(np.unique(t_fit)) > 1 else t0), 1e-6)
        x0 = np.log([k0, t0, s0])

    def residuals(params):
        y_hat = spray_penetration_model_quarter_only(params, t_fit)
        r = y_hat - y_fit
        if not np.all(np.isfinite(r)):
            return np.full_like(y_fit, 1e6, dtype=float)
        return r

    res = least_squares(residuals, x0, method="trf", loss="huber", f_scale=1.0)
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
        "log_params": res.x,
        "k_quarter": float(np.exp(log_k_quarter)),
        "t0": float(np.exp(log_t0) + MIN_TI),
        "s": float(np.exp(log_s)),
        "cost": float(res.cost),
        "success": bool(res.success),
        "n": int(valid.sum()),
        "nfev": int(getattr(res, "nfev", 0) or 0),
        "optimality": float(getattr(res, "optimality", np.nan)),
        "status": int(getattr(res, "status", -10)),
        "std_log_k_quarter": std_log_k_quarter,
        "std_log_t0": std_log_t0,
        "std_log_s": std_log_s,
        "corr_logk_logt0": corr_logk_logt0,
        "corr_logk_logs": corr_logk_logs,
        "corr_logt0_logs": corr_logt0_logs,
    }

