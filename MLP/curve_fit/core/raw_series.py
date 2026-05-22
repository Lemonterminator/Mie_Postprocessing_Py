"""Shared raw-series cleaning and schema helpers exported from fit_raw_data.

The production implementation currently remains in ``fit_raw_data.py`` for
backward compatibility. This module gives workflow/report code a stable import
location while the raw-fit script is gradually slimmed down.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.fit_raw_data import (  # noqa: F401
    META_COLS,
    PENETRATION_SOURCES,
    apply_filter_masking,
    build_wide_series_df,
    calculate_subframe_delay,
    collect_series_rows,
    filter_series_df,
    get_area_based_delay,
    get_dataset_settings,
    get_enabled_penetration_sources,
    penetration_cleaning,
    prepare_cleaned_series,
    robust_z,
)
