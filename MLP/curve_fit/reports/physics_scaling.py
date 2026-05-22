"""Combined entry point for raw and clean CDF physics-scaling screens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports import raw_series_physics_screening, time_windowed_exponent_regression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", choices=("raw", "time-windowed"), nargs="?", default="raw")
    args, passthrough = parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *passthrough]
        if args.screen == "raw":
            raw_series_physics_screening.main()
        else:
            time_windowed_exponent_regression.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
