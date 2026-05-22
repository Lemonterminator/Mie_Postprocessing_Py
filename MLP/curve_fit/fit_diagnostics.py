"""Diagnostic reports for the q1 production curve fit.

Reads the per-folder clean CSVs and series_clean CSVs produced by
``fit_raw_data.main()`` and writes diagnostic plots and summary tables under
``MLP/synthetic_data/fit_diagnostics/``.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.fit_diagnostics``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.fit_diagnostics import *  # noqa: F401,F403
from MLP.curve_fit.reports.fit_diagnostics import main


if __name__ == "__main__":
    main()
