"""Audit where CDF fit bias enters the preprocessing-to-fit pipeline.

This is a forensic companion to ``fit_raw_data.py``. It reloads the raw
experiment CSVs, replays the CDF cleaning/alignment steps, joins the result
back to the saved ``cdf/all`` and ``cdf/series_wide_all`` tables, and checks
whether bias is introduced by sample selection, delay handling, cleaning
filters, or the curve fit itself.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.analyze_cdf_fit_bias``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.analyze_cdf_fit_bias import *  # noqa: F401,F403
from MLP.curve_fit.reports.analyze_cdf_fit_bias import main


if __name__ == "__main__":
    main()
