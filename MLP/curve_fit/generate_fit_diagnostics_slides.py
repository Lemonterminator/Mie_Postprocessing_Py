"""Generate engineer-facing slides for the q1 fit diagnostics.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.generate_fit_diagnostics_slides``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.generate_fit_diagnostics_slides import *  # noqa: F401,F403
from MLP.curve_fit.reports.generate_fit_diagnostics_slides import main


if __name__ == "__main__":
    main()
