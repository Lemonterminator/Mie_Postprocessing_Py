"""Stable CLI wrapper for the RESIDUAL_SVGP training entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.GP_training.residual_multitask_svgp import main  # noqa: E402


if __name__ == "__main__":
    main()
