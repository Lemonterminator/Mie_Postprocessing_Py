from __future__ import annotations

import numpy as np

from OSCC_postprocessing.filters.video_filters import median_filter_video_auto
from OSCC_postprocessing.utils.backend import cp, to_numpy
from OSCC_postprocessing.utils.scaling import min_max_scale


def prepare_temporal_smoothing(rotated, smooth_frames):
    rotated_cpu = to_numpy(rotated)
    smoothed_np = median_filter_video_auto(np.swapaxes(rotated_cpu, 0, 2), smooth_frames, 1)
    smoothed_np = np.swapaxes(smoothed_np, 0, 2)

    if smoothed_np.size == 0:
        smoothed_np = np.zeros_like(smoothed_np, dtype=np.float32)
    else:
        smoothed_np = min_max_scale(smoothed_np).astype(np.float32, copy=False)

    smoothed_cp = cp.asarray(smoothed_np, dtype=cp.float32)
    return smoothed_cp, smoothed_np


def rotate_align_video_cpu(
    video: np.ndarray,
    nozzle_center: tuple[float, float],
    offset_deg: float,
    *,
    interpolation: str,
    out_shape: tuple[int, int] | None,
    border_mode: str,
    cval: float,
) -> np.ndarray:
    """Rotate and align a video around the nozzle using the CPU backend."""
    from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as rotate_video_nozzle_at_0_half_backend,
    )

    rotated_np, _, _ = rotate_video_nozzle_at_0_half_backend(
        video,
        nozzle_center,
        offset_deg,
        interpolation=interpolation,
        border_mode=border_mode,
        out_shape=out_shape,
        cval=cval,
    )
    return rotated_np.astype(np.float32, copy=False)


__all__ = ["prepare_temporal_smoothing", "rotate_align_video_cpu"]
