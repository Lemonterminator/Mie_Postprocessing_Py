"""Compatibility wrapper for cine utilities and video helpers.

This module restores the historical ``OSCC_postprocessing.cine`` namespace
expected by the GUI and example scripts. Functionality is provided by
``cine_utils`` and ``functions_videos`` along with optional Dewesoft helpers
under ``cine.dewe``.
"""

from .cine_utils import CineReader
from .functions_videos import *  # noqa: F401,F403

__all__ = [
    "CineReader",
] + [name for name in globals() if not name.startswith("_")]
