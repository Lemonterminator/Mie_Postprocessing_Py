"""Build point-level CDF tables with explicit FOV/right-censoring labels.

The script expands ``cdf/series_wide_clean`` into finite point samples in a
fixed time window, groups them by the same operating-condition columns used by
Stage 3, bins by time, and marks all points at/after the first bin where either
raw sample density drops or penetration reaches the field of view.

The generated point table is the bridge between curve fitting and Stage-3
raw-series supervision: each point carries the original CDF measurement plus
``is_right_censored`` and the trigger reason. Companion condition/bin tables
and plots explain why a condition was marked censored.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.workflows.cdf_censoring_points``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.workflows.cdf_censoring_points import *  # noqa: F401,F403
from MLP.curve_fit.workflows.cdf_censoring_points import main


if __name__ == "__main__":
    main()
