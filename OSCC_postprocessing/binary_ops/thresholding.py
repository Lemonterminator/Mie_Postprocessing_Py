"""Thresholding and morphology helpers for binary plume masks."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening
from skimage.filters import threshold_otsu
from skimage.morphology import disk

from ._backend import cp


def mask_video(video: np.ndarray, chamber_mask: np.ndarray) -> np.ndarray:
    """Broadcast a 2D chamber mask across a video, resizing if needed."""
    chamber_mask_bool = chamber_mask if chamber_mask.dtype == bool else (chamber_mask > 0)
    if video.shape[1:] != chamber_mask_bool.shape:
        chamber_mask_bool = cv2.resize(
            chamber_mask_bool.astype(np.uint8),
            (video.shape[2], video.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        )
    return video * chamber_mask_bool


def binarize_video_global_threshold(video, method="otsu", thresh_val=None):
    """Binarize a full video using one global threshold."""
    if method == "otsu":
        threshold = threshold_otsu(video)
    elif method == "fixed":
        if thresh_val is None:
            raise ValueError("Provide a threshold value for 'fixed' method.")
        threshold = thresh_val
    else:
        raise ValueError("Invalid method. Use 'otsu' or 'fixed'.")
    return (video >= threshold).astype(np.uint8) * 255


def calculate_bw_area(bw: np.ndarray) -> np.ndarray:
    """Count white pixels per frame in a ``0/255`` video."""
    return np.sum(bw == 255, axis=(1, 2)).astype(np.float32, copy=False)


def apply_morph_open(intermediate_frame, disk_size):
    """Apply 2D binary opening with a disk structuring element."""
    return binary_opening(intermediate_frame, disk(disk_size))


def apply_hole_filling(opened_frame):
    """Fill internal holes in one frame and return both mask and area."""
    filled = binary_fill_holes(opened_frame)
    return (filled * 255).astype(np.uint8), np.sum(filled)


def _apply_morph_open_one(args):
    frame, disk_size = args
    return apply_morph_open(frame, disk_size)


def apply_morph_open_video(intermediate_video: np.ndarray, disk_size: int) -> np.ndarray:
    """Apply morphology opening frame-by-frame using a process pool."""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_apply_morph_open_one, ((frame, disk_size) for frame in intermediate_video)))
    return np.asarray(results)


def apply_hole_filling_video(opened_video: np.ndarray):
    """Fill holes for every frame and collect the filled area per frame."""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(apply_hole_filling, opened_video))

    processed_video = np.zeros_like(opened_video, dtype=np.uint8)
    area = np.zeros(opened_video.shape[0], dtype=np.float32)
    for idx, (processed_frame, frame_area) in enumerate(results):
        processed_video[idx] = processed_frame
        area[idx] = frame_area
    return processed_video, area


def _fill_frame(frame_bool):
    return binary_fill_holes(frame_bool)


def fill_video_holes_parallel(bw_video: np.ndarray, n_workers: int | None = None) -> np.ndarray:
    """Fill holes in each frame independently on CPU."""
    bw_bool_video = bw_video > 0
    with ProcessPoolExecutor(max_workers=n_workers or os.cpu_count()) as executor:
        filled_frames = list(executor.map(_fill_frame, bw_bool_video))
    return np.stack(filled_frames, axis=0)


def fill_video_holes_gpu(bw_video: np.ndarray) -> np.ndarray:
    """GPU-friendly wrapper with CPU fallback for the actual hole filling step."""
    from cupy import asnumpy
    import scipy.ndimage

    bw_gpu = cp.asarray(bw_video) > 0
    filled_cpu = scipy.ndimage.binary_fill_holes(asnumpy(bw_gpu))
    filled_gpu = cp.asarray(filled_cpu)
    if bw_video.max() > 1:
        return filled_gpu.astype(bw_video.dtype) * 255
    return filled_gpu.astype(bw_video.dtype)


def _triangle_threshold_from_hist(hist):
    """Compute the triangle-method threshold from a 256-bin histogram."""
    nz = np.flatnonzero(hist)
    if nz.size == 0:
        return 0

    left, right = int(nz[0]), int(nz[-1])
    peak = int(np.argmax(hist))
    h_peak = float(hist[peak])
    end = left if (peak - left) >= (right - peak) else right
    h_end = float(hist[end])
    dx = end - peak
    dy = h_end - h_peak
    if dx == 0:
        return peak

    idx = np.arange(peak, end + 1) if end > peak else np.arange(end, peak + 1)
    xi = idx - peak
    yi = hist[idx].astype(float) - h_peak
    numerator = np.abs(dy * xi - dx * yi)
    return int(idx[int(np.argmax(numerator))])


def triangle_binarize_u8(u8, blur=True, ignore_zero=False, threshold_on_unblurred=True):
    """Triangle-threshold an 8-bit image, optionally ignoring the zero bin."""
    if u8.dtype != np.uint8:
        u8 = u8.astype(np.uint8, copy=False)

    u8_for_threshold = (
        u8
        if (ignore_zero and threshold_on_unblurred)
        else (cv2.GaussianBlur(u8, (5, 5), 0) if blur else u8)
    )

    if ignore_zero:
        hist = cv2.calcHist([u8_for_threshold], [0], None, [256], [0, 256]).flatten()
        hist[0] = 0
        if hist.sum() == 0:
            return np.zeros_like(u8), 0
        threshold = _triangle_threshold_from_hist(hist)
        u8_apply = cv2.GaussianBlur(u8, (5, 5), 0) if blur else u8
        _, binarized = cv2.threshold(u8_apply, threshold, 255, cv2.THRESH_BINARY)
        return binarized, int(threshold)

    u8_apply = cv2.GaussianBlur(u8, (5, 5), 0) if blur else u8
    threshold, binarized = cv2.threshold(
        u8_apply, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    return binarized, int(threshold)


def triangle_binarize_from_float(
    img_f32,
    blur=True,
    ignore_zero=False,
    threshold_on_unblurred=True,
):
    """Normalize float input to ``uint8`` then apply triangle thresholding."""
    u8 = cv2.normalize(img_f32, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return triangle_binarize_u8(
        u8,
        blur=blur,
        ignore_zero=ignore_zero,
        threshold_on_unblurred=threshold_on_unblurred,
    )
