"""Programmatic wrapper for the raw Mie trajectory fit production workflow."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit import fit_raw_data


def main() -> None:
    fit_raw_data.main()


if __name__ == "__main__":
    main()
