"""High-level helpers for segment-oriented rotation workflows.

This module is the release-facing replacement for the deleted
``rotate_crop.py`` helper collection. Its responsibility is intentionally
narrow: expose the small set of geometry and orchestration helpers still used
by the analysis and pipeline layers, while delegating all actual remapping work
to the canonical affine engines in ``rotate_with_alignment_cpu.py`` and
``rotate_with_alignment.py``.

The public surface is limited to four functions:

- ``generate_CropRect`` computes the standard rectangular strip used to sample
  one plume sector from a radial spray.
- ``generate_plume_mask`` reproduces the historical trapezoidal mask used in
  that strip's local coordinate system.
- ``rotate_video_auto`` rotates a full stack around the image centre.
- ``rotate_all_segments_auto`` rotates one stack into multiple cropped segments
  around a user-provided centre and ROI.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import rotate_video_about_center_numpy

__all__ = [
    "generate_CropRect",
    "generate_plume_mask",
    "rotate_video_auto",
    "rotate_all_segments_auto",
]


def _is_cupy_array(arr) -> bool:
    return hasattr(arr, "__cuda_array_interface__")


def _default_interpolation(array) -> str:
    """Match the historical interpolation choice from the removed wrappers."""
    dtype = getattr(array, "dtype", None)
    if dtype is not None and np.issubdtype(np.dtype(dtype), np.bool_):
        return "nearest"
    return "bicubic"


def _apply_output_mask(segment, mask):
    """Apply a mask defined in output-segment coordinates."""
    if mask is None:
        return segment

    mask_bool = mask.astype(bool, copy=False)
    if _is_cupy_array(segment):
        import cupy as cp

        mask_backend = cp.asarray(mask_bool)
        if segment.dtype == cp.bool_:
            return cp.logical_and(segment, mask_backend[None, ...])
        out = cp.array(segment, copy=True)
        out *= mask_backend[None, ...]
        return out

    if segment.dtype == np.bool_:
        return np.logical_and(segment, mask_bool[None, ...])

    out = np.array(segment, copy=True)
    out *= mask_bool[None, ...]
    return out


def _rotate_video_about_center_cupy(*args, **kwargs):
    from OSCC_postprocessing.rotation.rotate_with_alignment import rotate_video_about_center_cupy

    return rotate_video_about_center_cupy(*args, **kwargs)


def generate_CropRect(inner_radius, outer_radius, number_of_plumes, centre_x, centre_y):
    """Build the standard crop rectangle for one radial plume strip.

    Parameters
    ----------
    inner_radius, outer_radius : float
        Radial bounds of the strip measured from the spray centre.
    number_of_plumes : int
        Number of equally spaced plumes. ``1`` is promoted to ``2`` so the
        sector spans 180 degrees, preserving the historical special case.
    centre_x, centre_y : float
        Spray-centre position in full-frame image coordinates.

    Returns
    -------
    tuple[int, int, int, int]
        ``(x, y, w, h)`` rectangle in full-frame coordinates.
    """
    if number_of_plumes == 1:
        number_of_plumes = 2

    section_angle = 360.0 / number_of_plumes
    half_angle_radian = section_angle / 2.0 * np.pi / 180.0
    half_width = round(outer_radius * np.sin(half_angle_radian))

    x = round(centre_x + inner_radius)
    y = max(0, round(centre_y - half_width))
    w = round(outer_radius - inner_radius)
    h = 2 * half_width
    return (x, y, w, h)


def generate_plume_mask(inner_radius, outer_radius, w, h):
    """Generate the historical trapezoidal mask inside one crop rectangle.

    The mask is expressed in the local coordinates of the rectangular plume
    strip. The left edge corresponds to ``inner_radius`` and the right edge
    corresponds to ``outer_radius``.
    """
    import cv2

    y1 = -h / outer_radius / 2 * inner_radius + h / 2
    y2 = h / outer_radius / 2 * inner_radius + h / 2

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[0, round(y2)], [0, round(y1)], [w, 0], [w, h]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def rotate_video_auto(video_array, angle=0, max_workers=4):
    """Rotate a full stack about the image centre.

    Boolean inputs use nearest-neighbour interpolation. Numeric inputs use
    bicubic interpolation. Pixels outside the source frame are filled with
    zeros, matching the historical wrapper behaviour.
    """
    video = video_array
    H, W = video.shape[1:3]
    center = (W // 2, H // 2)
    interpolation = _default_interpolation(video)

    if _is_cupy_array(video):
        rotated, _, _ = _rotate_video_about_center_cupy(
            video,
            center,
            angle,
            interpolation=interpolation,
            border_mode="constant",
            cval=0.0,
            stack=True,
        )
        return rotated

    rotated, _, _ = rotate_video_about_center_numpy(
        video,
        center,
        angle,
        interpolation=interpolation,
        border_mode="constant",
        cval=0.0,
        stack=True,
        max_workers=max_workers,
    )
    return rotated


def _rotate_one_segment_numpy(video, angle, crop, centre, mask):
    interpolation = _default_interpolation(video)
    # Keep the legacy sign convention from ``rotate_crop.py`` so segment
    # orientation and downstream plume ordering do not change.
    rotated, _, _ = rotate_video_about_center_numpy(
        video,
        centre,
        -angle,
        crop_rect=crop,
        interpolation=interpolation,
        border_mode="constant",
        cval=0.0,
        stack=True,
        max_workers=1,
    )
    return _apply_output_mask(rotated, mask)


def _rotate_one_segment_cupy(video, angle, crop, centre, mask):
    interpolation = _default_interpolation(video)
    # Keep the same legacy sign convention as the NumPy path above.
    rotated, _, _ = _rotate_video_about_center_cupy(
        video,
        centre,
        -angle,
        crop_rect=crop,
        interpolation=interpolation,
        border_mode="constant",
        cval=0.0,
        stack=True,
    )
    return _apply_output_mask(rotated, mask)


def rotate_all_segments_auto(video, angles, crop, centre, mask=None):
    """Rotate one input stack into one cropped segment per requested angle.

    ``crop`` is expressed in full-frame coordinates as ``(x, y, w, h)`` and
    ``mask`` is expressed in the output crop coordinate system. The returned
    list preserves the input ``angles`` ordering exactly.
    """
    if _is_cupy_array(video):
        return [_rotate_one_segment_cupy(video, angle, crop, centre, mask) for angle in angles]

    max_workers = min(len(angles), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_rotate_one_segment_numpy, video, angle, crop, centre, mask)
            for angle in angles
        ]
        return [future.result() for future in futures]
