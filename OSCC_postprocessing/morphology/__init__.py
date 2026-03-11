"""Compatibility exports for legacy morphology utilities."""

from .morphological import (
    close_video_parallel,
    dilate_video_parallel,
    erode_video_parallel,
    open_video_parallel,
)

__all__ = [
    "close_video_parallel",
    "dilate_video_parallel",
    "erode_video_parallel",
    "open_video_parallel",
]
