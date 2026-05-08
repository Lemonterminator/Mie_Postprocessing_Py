from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RHO_L_KG_M3 = 832.0  # diesel liquid density (kg/m³), treated as known constant

# Hiroyasu & Arai (1990) constant sets. All formulas are in SI (Pa, m, s, kg/m³);
# output is in metres before the caller converts to mm.
LITERATURE_CONSTANTS: dict[str, dict[str, float]] = {
    "hiroyasu": {"kv": 0.39, "kp": 2.95,  "kbt": 28.65},
    "arai1":    {"kv": 0.60, "kp": 3.36,  "kbt": 15.70},
    "arai2":    {"kv": 0.80, "kp": 3.36,  "kbt":  8.84},
}

VARIANT_TO_CONSTANTS_KEY: dict[str, str] = {
    "ha_hiroyasu":   "hiroyasu",
    "ha_arai1":      "arai1",
    "ha_arai2":      "arai2",
    "zhou_hiroyasu": "hiroyasu",
    "zhou_arai1":    "arai1",
    "zhou_arai2":    "arai2",
}

CALIBRATED_VARIANTS: frozenset[str] = frozenset({"ha_calibrated", "zhou_calibrated"})
ZHOU_VARIANTS: frozenset[str] = frozenset({"zhou_hiroyasu", "zhou_arai1", "zhou_arai2", "zhou_calibrated"})
ALL_VARIANTS: tuple[str, ...] = (
    "ha_hiroyasu", "ha_arai1", "ha_arai2", "ha_calibrated",
    "zhou_hiroyasu", "zhou_arai1", "zhou_arai2", "zhou_calibrated",
)


@dataclass(frozen=True)
class HAParams:
    kv: float
    kp: float
    kbt: float
    variant: str
    use_zhou: bool


def breakup_time(
    d_n_m: np.ndarray,
    delta_p_pa: np.ndarray,
    rho_a_kg_m3: np.ndarray,
    kbt: float,
) -> np.ndarray:
    return kbt * RHO_L_KG_M3 * d_n_m / np.sqrt(rho_a_kg_m3 * delta_p_pa)


def predict_ha(
    time_s: np.ndarray,
    d_n_m: np.ndarray,
    delta_p_pa: np.ndarray,
    rho_a_kg_m3: np.ndarray,
    kv: float,
    kp: float,
    kbt: float,
) -> np.ndarray:
    """Return spray-tip penetration in metres using the Hiroyasu-Arai (1990) model."""
    t_b = breakup_time(d_n_m, delta_p_pa, rho_a_kg_m3, kbt)
    t_safe = np.maximum(time_s, 0.0)
    s_early = kv * np.sqrt(2.0 * delta_p_pa / RHO_L_KG_M3) * t_safe
    s_late = kp * np.power(delta_p_pa / rho_a_kg_m3, 0.25) * np.sqrt(d_n_m * t_safe)
    return np.where(time_s < t_b, s_early, s_late)


def predict_zhou(
    time_s: np.ndarray,
    d_n_m: np.ndarray,
    delta_p_pa: np.ndarray,
    rho_a_kg_m3: np.ndarray,
    t_inj_s: np.ndarray,
    kv: float,
    kp: float,
    kbt: float,
) -> np.ndarray:
    """Return spray-tip penetration in metres using the Zhou (2019) H-A extension.

    Transition from H-A Stage-2 to Stage-3 occurs at t = 2·t_i (not t_i).
    The Stage-3 coefficient √2·kp follows from C⁰ continuity at 2·t_i; it is
    not a free parameter.

    Stage-3:  S = √2·kp · (ΔP/ρ_a)^0.25 · √(d_n) · t_i^0.25 · (t − t_i)^0.25
    """
    t_b = breakup_time(d_n_m, delta_p_pa, rho_a_kg_m3, kbt)
    t_safe = np.maximum(time_s, 0.0)
    t_i = np.maximum(t_inj_s, 1e-9)

    s_early = kv * np.sqrt(2.0 * delta_p_pa / RHO_L_KG_M3) * t_safe
    s_mid = kp * np.power(delta_p_pa / rho_a_kg_m3, 0.25) * np.sqrt(d_n_m * t_safe)
    dt_from_inj = np.maximum(time_s - t_i, 0.0)
    s_late = (
        np.sqrt(2.0) * kp
        * np.power(delta_p_pa / rho_a_kg_m3, 0.25)
        * np.sqrt(d_n_m)
        * np.power(t_i, 0.25)
        * np.power(dt_from_inj, 0.25)
    )
    s_out = np.where(time_s < t_b, s_early, s_mid)
    s_out = np.where(time_s > 2.0 * t_i, s_late, s_out)
    return s_out


def predict(df: pd.DataFrame, params: HAParams) -> np.ndarray:
    """Return predictions in mm."""
    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    d_n_m = pd.to_numeric(df["diameter_m"], errors="coerce").to_numpy(dtype=float)
    delta_p_pa = pd.to_numeric(df["delta_pressure_pa"], errors="coerce").to_numpy(dtype=float)
    rho_a = pd.to_numeric(df["ambient_density_kg_m3"], errors="coerce").to_numpy(dtype=float)
    if params.use_zhou:
        t_inj_s = pd.to_numeric(df["t_inj_s"], errors="coerce").to_numpy(dtype=float)
        s_m = predict_zhou(time_s, d_n_m, delta_p_pa, rho_a, t_inj_s, params.kv, params.kp, params.kbt)
    else:
        s_m = predict_ha(time_s, d_n_m, delta_p_pa, rho_a, params.kv, params.kp, params.kbt)
    return s_m * 1000.0  # m → mm


def params_to_dict(params: HAParams) -> dict[str, Any]:
    return {
        "kv": float(params.kv),
        "kp": float(params.kp),
        "kbt": float(params.kbt),
        "variant": str(params.variant),
        "use_zhou": bool(params.use_zhou),
        "rho_l_kg_m3": RHO_L_KG_M3,
    }
