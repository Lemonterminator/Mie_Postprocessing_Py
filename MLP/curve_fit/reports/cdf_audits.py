"""Combined CDF audit entry point for bias, censoring, and FOV/spike checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports import (
    analyze_cdf_fit_bias,
    audit_cdf_spatial_censoring,
    audit_fov_and_spikes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", choices=("bias", "spatial-censoring", "fov-spikes"))
    args, passthrough = parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *passthrough]
        if args.audit == "bias":
            analyze_cdf_fit_bias.main()
        elif args.audit == "spatial-censoring":
            audit_cdf_spatial_censoring.main()
        else:
            audit_fov_and_spikes.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
