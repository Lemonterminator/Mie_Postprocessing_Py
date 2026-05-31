from __future__ import annotations

"""Feature variant registry for the heteroscedastic spray-penetration MLP.

Maps each variant name to its feature columns, z-score columns, delta-pressure
exponent for A-scale computation, and target scaling mode.

Production variant: ``a_only``  (alias for ``a_dp050``)
    features: [time_norm_0_5ms, tilt_angle_radian_z, plumes_z,
               injection_duration_us_z, control_backpressure_bar_z]
    A_scale  = ΔP^0.5 · ρ_air^{-0.25} · √d   (Bernoulli/orifice-momentum regime)
    target   = penetration_mm / A_scale         (dimensionless)

The ΔP exponent 0.5 was chosen over the legacy Hiroyasu–Arai value 0.25 because
the 0–5 ms injection window is dominated by momentum-driven penetration, not the
asymptotic Rayleigh–Taylor regime.  ``a_dp025*`` variants preserve HA scaling for
ablation comparisons.  ``a_only`` is the alias used in all published ablation runs.
"""

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYNTHETIC_DATA_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data"
DEFAULT_TEST_MATRIX_ROOT = PROJECT_ROOT / "test_matrix_json"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "MLP" / "runs_mlp"

REFERENCE_PRESSURE_BAR = 4.42
MIN_TIME_SHIFT_S = 0.0

TIME_FEATURE = "time_norm_0_5ms"
BASE_STATIC_FEATURE_COLUMNS = [
    "tilt_angle_radian_z",
    "plumes_z",
    "injection_duration_us_z",
    "control_backpressure_bar_z",
]
BASE_STATIC_ZSCORE_COLUMNS = [
    "tilt_angle_radian",
    "plumes",
    "injection_duration_us",
    "control_backpressure_bar",
]
PRESSURE_FEATURE_COLUMNS = [
    "log_injection_pressure_bar_z",
    "log_chamber_pressure_bar_z",
]
PRESSURE_ZSCORE_COLUMNS = [
    "log_injection_pressure_bar",
    "log_chamber_pressure_bar",
]
DIAMETER_FEATURE_COLUMNS = ["diameter_mm_z"]
DIAMETER_ZSCORE_COLUMNS = ["diameter_mm"]
DEFAULT_A_SCALE_DP_EXP = 0.5
DEFAULT_A_SCALE_DENSITY_EXP = -0.25
DEFAULT_A_SCALE_DIAMETER_EXP = 0.5

FEATURE_COLUMNS_BY_VARIANT: dict[str, list[str]] = {}
ZSCORE_BASE_COLUMNS_BY_VARIANT: dict[str, list[str]] = {}
A_SCALE_DP_EXP_BY_VARIANT: dict[str, float] = {}
TARGET_SCALE_MODE_BY_VARIANT: dict[str, str] = {}


def _register_feature_variant(
    name: str,
    feature_columns: Sequence[str],
    zscore_base_columns: Sequence[str],
    *,
    a_scale_dp_exp: float = DEFAULT_A_SCALE_DP_EXP,
    target_scale_mode: str = "a_scale",
) -> None:
    token = str(name).lower()
    FEATURE_COLUMNS_BY_VARIANT[token] = list(feature_columns)
    ZSCORE_BASE_COLUMNS_BY_VARIANT[token] = list(zscore_base_columns)
    A_SCALE_DP_EXP_BY_VARIANT[token] = float(a_scale_dp_exp)
    TARGET_SCALE_MODE_BY_VARIANT[token] = str(target_scale_mode)


def _a_feature_columns(*, include_pressures: bool = False, include_diameter: bool = False, include_log_a: bool = False) -> list[str]:
    columns = [TIME_FEATURE, *BASE_STATIC_FEATURE_COLUMNS]
    if include_pressures:
        columns.extend(PRESSURE_FEATURE_COLUMNS)
    if include_diameter:
        columns.extend(DIAMETER_FEATURE_COLUMNS)
    if include_log_a:
        columns.append("log_A_z")
    return columns


def _a_zscore_columns(*, include_pressures: bool = False, include_diameter: bool = False, include_log_a: bool = False) -> list[str]:
    columns = list(BASE_STATIC_ZSCORE_COLUMNS)
    if include_pressures:
        columns.extend(PRESSURE_ZSCORE_COLUMNS)
    if include_diameter:
        columns.extend(DIAMETER_ZSCORE_COLUMNS)
    if include_log_a:
        columns.append("log_A")
    return columns


LEGACY_9_FEATURE_COLUMNS = [
    TIME_FEATURE,
    "tilt_angle_radian_z",
    "plumes_z",
    "diameter_mm_z",
    "injection_duration_us_z",
    "log_injection_pressure_bar_z",
    "log_chamber_pressure_bar_z",
    "log_delta_pressure_bar_z",
    "control_backpressure_bar_z",
]

_register_feature_variant(
    "legacy_9_no_scale",
    LEGACY_9_FEATURE_COLUMNS,
    [
        "tilt_angle_radian",
        "plumes",
        "diameter_mm",
        "injection_duration_us",
        "log_injection_pressure_bar",
        "log_chamber_pressure_bar",
        "log_delta_pressure_bar",
        "control_backpressure_bar",
    ],
    target_scale_mode="none",
)

for _label, _dp_exp in (("dp025", 0.25), ("dp050", 0.50)):
    _register_feature_variant(
        f"a_{_label}",
        _a_feature_columns(),
        _a_zscore_columns(),
        a_scale_dp_exp=_dp_exp,
    )
    _register_feature_variant(
        f"a_{_label}_plus_pressures",
        _a_feature_columns(include_pressures=True),
        _a_zscore_columns(include_pressures=True),
        a_scale_dp_exp=_dp_exp,
    )
    _register_feature_variant(
        f"a_{_label}_plus_diameter",
        _a_feature_columns(include_diameter=True),
        _a_zscore_columns(include_diameter=True),
        a_scale_dp_exp=_dp_exp,
    )
    _register_feature_variant(
        f"a_{_label}_plus_pressures_diameter",
        _a_feature_columns(include_pressures=True, include_diameter=True),
        _a_zscore_columns(include_pressures=True, include_diameter=True),
        a_scale_dp_exp=_dp_exp,
    )

# Backward-compatible aliases used by existing configs and saved manifests.
_register_feature_variant("a_only", FEATURE_COLUMNS_BY_VARIANT["a_dp050"], ZSCORE_BASE_COLUMNS_BY_VARIANT["a_dp050"], a_scale_dp_exp=0.50)
_register_feature_variant(
    "a_plus_pressures",
    FEATURE_COLUMNS_BY_VARIANT["a_dp050_plus_pressures"],
    ZSCORE_BASE_COLUMNS_BY_VARIANT["a_dp050_plus_pressures"],
    a_scale_dp_exp=0.50,
)
_register_feature_variant(
    "a_plus_log_a",
    _a_feature_columns(include_log_a=True),
    _a_zscore_columns(include_log_a=True),
    a_scale_dp_exp=0.50,
)
_register_feature_variant(
    "a_plus_diameter",
    FEATURE_COLUMNS_BY_VARIANT["a_dp050_plus_diameter"],
    ZSCORE_BASE_COLUMNS_BY_VARIANT["a_dp050_plus_diameter"],
    a_scale_dp_exp=0.50,
)
_register_feature_variant(
    "a_plus_pressures_diameter",
    FEATURE_COLUMNS_BY_VARIANT["a_dp050_plus_pressures_diameter"],
    ZSCORE_BASE_COLUMNS_BY_VARIANT["a_dp050_plus_pressures_diameter"],
    a_scale_dp_exp=0.50,
)
LEGACY_FEATURE_COLUMNS = {
    "diameter_mm_z",
    "log_injection_pressure_bar_z",
    "log_chamber_pressure_bar_z",
    "log_delta_pressure_bar_z",
}
LEGACY_FILL_DEFAULTS = {
    "injection_pressure_bar": 2000.0,
    "control_backpressure_bar": 4.0,
}


def umbrella_to_tilt_radian(umbrella_angle_deg: float) -> float:
    return float(np.deg2rad((180.0 - float(umbrella_angle_deg)) / 2.0))


def resolve_tilt_angle_radian(raw: Mapping[str, Any]) -> float:
    if "tilt_angle_radian" in raw:
        return float(raw["tilt_angle_radian"])
    if "umbrella_angle_deg" in raw:
        return umbrella_to_tilt_radian(float(raw["umbrella_angle_deg"]))
    raise KeyError("raw input must include either 'tilt_angle_radian' or 'umbrella_angle_deg'.")
