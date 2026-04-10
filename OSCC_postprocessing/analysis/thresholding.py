"""Thresholding helpers for analysis pipelines.

This module is the public analysis-layer entry point for triangle thresholding.
Use :func:`triangle_binarize` in new code. ``triangle_binarize_gpu`` remains as
an alias for older notebooks and callers, but it resolves to the same unified
GPU-first, CPU-fallback implementation.
"""

from __future__ import annotations

from OSCC_postprocessing.binary_ops.thresholding import (
    _triangle_threshold_from_hist,
    triangle_binarize,
    triangle_binarize_from_float,
    triangle_binarize_gpu,
    triangle_binarize_u8,
    triangle_binarize_with_threshold,
)
from OSCC_postprocessing.utils.backend import get_array_module

__all__ = [
    "_triangle_threshold_from_hist",
    "get_array_module",
    "triangle_binarize",
    "triangle_binarize_from_float",
    "triangle_binarize_gpu",
    "triangle_binarize_u8",
    "triangle_binarize_with_threshold",
]
