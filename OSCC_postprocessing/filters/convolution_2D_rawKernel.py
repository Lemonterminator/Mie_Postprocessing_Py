"""Compatibility wrapper for the renamed 2D convolution helpers.

The implementation now lives in ``convolution_2d.py``. This module remains to
preserve historical imports while the codebase is updated.
"""

from .convolution_2d import *  # noqa: F401,F403
