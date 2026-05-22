"""Build plume-level spatial right-censoring labels for CDF trajectories.

The audit estimates a per-nozzle field-of-view cap from raw positive CDF
penetration samples, then compares three views of each plume: raw Mie CSV
trace, cleaned synthetic ``series_wide_all`` trace, and stored fit row. A plume
is marked spatially right-censored when either raw or cleaned penetration
reaches the estimated cap within the configured tolerance.

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.audit_cdf_spatial_censoring``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.audit_cdf_spatial_censoring import *  # noqa: F401,F403
from MLP.curve_fit.reports.audit_cdf_spatial_censoring import main


if __name__ == "__main__":
    main()
