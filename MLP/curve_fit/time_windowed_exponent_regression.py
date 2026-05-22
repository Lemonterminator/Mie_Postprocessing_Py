"""Time-windowed log-log scaling regression on raw S(t).

Compared with ``raw_series_physics_screening.py``, this version uses the clean
CDF wide table and canonical chamber pressure/density conversion from the
Stage-3 feature code.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.time_windowed_exponent_regression``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.time_windowed_exponent_regression import *  # noqa: F401,F403
from MLP.curve_fit.reports.time_windowed_exponent_regression import main


if __name__ == "__main__":
    main()
