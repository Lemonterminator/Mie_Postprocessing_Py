"""Compatibility wrappers for legacy binarized-metrics consumers."""

from __future__ import annotations

from .feature_extraction import (
    features_to_binarized_metrics_df,
    processing_from_binarized_video,
)

__all__ = [
    "features_to_binarized_metrics_df",
    "processing_from_binarized_video",
]
