"""Summarize fit-filter survival rates from ``MLP/synthetic_data/fit_report.csv``.

The fitter emits one report row per nozzle/folder/source. This script
aggregates those counts by nozzle, folder, and source; writes CSV/JSON/TEX
summaries; and copies the CDF survival plot to the thesis image directory by
default.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.summarize_filter_survival``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.summarize_filter_survival import *  # noqa: F401,F403
from MLP.curve_fit.reports.summarize_filter_survival import main


if __name__ == "__main__":
    main()
