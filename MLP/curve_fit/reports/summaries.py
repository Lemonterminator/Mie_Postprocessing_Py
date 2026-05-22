"""Combined summary entry point for curve-fit dataset and filter-survival reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports import summarize_dataset, summarize_filter_survival


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", choices=("dataset", "survival", "all"), nargs="?", default="all")
    args, passthrough = parser.parse_known_args()

    if args.report in {"dataset", "all"}:
        summarize_dataset.main()
    if args.report in {"survival", "all"}:
        # Reuse the original parser for optional --fit-report/--out-dir flags.
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], *passthrough]
            summarize_filter_survival.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
