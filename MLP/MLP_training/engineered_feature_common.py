"""Backward-compatibility shim. All symbols now live in the efc subpackage."""
from __future__ import annotations

try:
    from .efc import *  # noqa: F401, F403
except ImportError:
    # Bare import (MLP_training/ on sys.path): efc is a local package
    from efc import *  # noqa: F401, F403
