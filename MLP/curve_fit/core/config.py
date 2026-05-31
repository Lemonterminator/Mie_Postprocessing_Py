"""Shared configuration for the curve-fit pipeline.

Contains every module-level switch, threshold, dataset registry, and
penetration-source definition. Imported by ``core.cleaning``,
``core.filter_masking``, ``core.series_io``, ``core.plotting`` and
``workflows.raw_fit``.

Environment variables (read at import time):

- ``FIT_OUTPUT_ROOT`` / ``FIT_INPUT_ROOT`` override the default I/O roots.
- ``FIT_NOZZLE_FILTER`` restricts the dataset list to one suffix match.
- ``FIT_N_WORKERS`` sets the ProcessPoolExecutor pool size.

The ``log_k_sqrt`` / ``k_sqrt`` columns in the fit output are kept as
finite sentinels (``LOG_K_SQRT_SENTINEL``, ``K_SQRT_SENTINEL``) so the
downstream MLP-training feature engineering, which still hard-codes the
sqrt+quarter blend schema, can keep loading fit CSVs unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path


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
MIN_TI = 0.0

# Fit/series limits.
DIFF_THRESHOLD_LOWER = 0.5  # mm
DIFF_THRESHOLD_UPPER = 10.0  # mm
MIN_SERIES_POINTS = 10  # discard series with fewer valid points before fitting

# Masking thresholds (from notebook prototype defaults).
MASK_GROUP_COLS = ("file_name",)
MASK_Z_THRESH = 3.0
MASK_MIN_N = 10
MASK_S_UPPER = 1e-3       # < 1 ms
MASK_T0_UPPER = 0.8e-3
MASK_FAR_TIME_MS = 5.0
MASK_PENETRATION_LOWER_MM = 18.0   # penetration (mm) at MASK_FAR_TIME_MS
MASK_PENETRATION_UPPER_MM = 300.0  # penetration (mm) at MASK_FAR_TIME_MS
RMSE_SUCCESS_THRESHOLD_MM = 3.0    # max RMSE (mm) for a fit to count as successful

# Plot defaults.
PLOT_EXTRAP_FACTOR = 1.6
PLOT_NUM_POINTS = 300
PLOT_YLIM_MM = 200.0

# Output schema sentinels: the q1 model has no sqrt term, but downstream
# MLP-training feature engineering still expects ``k_sqrt``/``log_k_sqrt``
# columns to exist. Writing these finite sentinels (exp(-500) ~ 0) lets the
# sigmoid-blend evaluator collapse cleanly to the q1 branch.
FIT_MODEL_NAME = "quarter_only_v1"
LOG_K_SQRT_SENTINEL = -500.0
K_SQRT_SENTINEL = 0.0


# Input/output roots (curve_fit/core/ is two levels below MLP/, three below project root).
_CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CONFIG_DIR.parents[2]
DATA_ROOT = _CONFIG_DIR.parents[1].parent / "Mie_scattering_top_view_results"
DATA_OUT_DIR = _CONFIG_DIR.parents[1] / "synthetic_data"
if _env := os.environ.get("FIT_OUTPUT_ROOT"):
    DATA_OUT_DIR = Path(_env)
if _env := os.environ.get("FIT_INPUT_ROOT"):
    DATA_ROOT = Path(_env)


# Dataset registry.
NOZZLE_NAMES = [
    "BC20241003_HZ_Nozzle1",
    "BC20241017_HZ_Nozzle2",
    "BC20241014_HZ_Nozzle3",
    "BC20241007_HZ_Nozzle4",
    "BC20241010_HZ_Nozzle5",
    "BC20241011_HZ_Nozzle6",
    "BC20241015_HZ_Nozzle7",
    "BC20241016_HZ_Nozzle8",
    "BC20220627_HZ_Nozzle0",
]
if _env := os.environ.get("FIT_NOZZLE_FILTER"):
    NOZZLE_NAMES = [n for n in NOZZLE_NAMES if n == _env or n.endswith(_env)]


# Worker pool. 0 = use all logical CPUs; 1 = single-process; N > 1 = explicit pool size.
N_WORKERS = 0
if _env := os.environ.get("FIT_N_WORKERS"):
    N_WORKERS = int(_env)


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
    """Return the penetration sources enabled by the module-level switches."""
    return [source for source in PENETRATION_SOURCES if source["enabled"]]
