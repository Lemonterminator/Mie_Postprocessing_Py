"""Audit FOV right-censoring thresholds and Hampel spike rates on cleaned traces.

This script measures candidate FOV and spike rules and writes evidence, but it
does not mutate the fitted dataset or Stage-3 training labels.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.audit_fov_and_spikes``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.audit_fov_and_spikes import *  # noqa: F401,F403
from MLP.curve_fit.reports.audit_fov_and_spikes import main


if __name__ == "__main__":
    main()
