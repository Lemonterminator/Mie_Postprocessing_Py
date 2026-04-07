"""Public entry points for rotation workflows.

Preferred usage is split by responsibility:

- use ``generate_CropRect`` / ``generate_plume_mask`` / ``rotate_all_segments_auto``
  for plume-segment extraction
- use ``rotate_video_nozzle_at_0_half_numpy`` or its CuPy counterpart for the
  nozzle-alignment workflow
- use the lower-level ``rotate_with_alignment(_cpu).py`` modules directly when
  you need explicit access to inverse maps or interpolation internals
"""

from .segment_ops import (
    generate_CropRect,
    generate_plume_mask,
    rotate_all_segments_auto,
    rotate_video_auto,
)
from .rotate_with_alignment_cpu import rotate_video_nozzle_at_0_half_numpy
try:
    from .rotate_with_alignment import rotate_video_nozzle_at_0_half_cupy
except Exception:  # pragma: no cover - depends on CuPy availability
    rotate_video_nozzle_at_0_half_cupy = None

__all__ = [
    "generate_CropRect",
    "generate_plume_mask",
    "rotate_all_segments_auto",
    "rotate_video_auto",
    "rotate_video_nozzle_at_0_half_cupy",
    "rotate_video_nozzle_at_0_half_numpy",
]
