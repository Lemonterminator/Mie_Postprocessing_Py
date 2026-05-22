"""Summarize the Mie top-view campaign for the thesis dataset section.

The script combines two sources of truth: raw processed result directories
under ``Mie_scattering_top_view_results`` and nozzle/test-matrix definitions
under ``test_matrix_json``. It counts recordings and metadata files, extracts
the designed pressure/duration ranges, and checks what conditions actually
appear in ``*.meta.json``.

Writes:
  MLP/dataset_summary.csv  -- per-dataset row
  MLP/dataset_summary.txt  -- human-readable block for the thesis

Compatibility wrapper: the implementation now lives in
``MLP.curve_fit.reports.summarize_dataset``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.curve_fit.reports.summarize_dataset import *  # noqa: F401,F403
from MLP.curve_fit.reports.summarize_dataset import main


if __name__ == "__main__":
    main()
