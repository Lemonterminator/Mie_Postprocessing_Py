"""Quick time-binned physics scaling screen on raw CDF penetration points.

The screen reads every ``cdf/series_wide_all/*.csv`` file under the selected
synthetic-data root, expands wide time/penetration columns to raw point samples,
bins by time, and fits a log-log scaling model.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.raw_series_physics_screening``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.raw_series_physics_screening import *  # noqa: F401,F403
from MLP.curve_fit.reports.raw_series_physics_screening import main


if __name__ == "__main__":
    main()
