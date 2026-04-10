"""Thresholding and morphology helpers for binary plume masks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter
from skimage.filters import threshold_otsu

from ._backend import CUPY_AVAILABLE, cp


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
    from skimage.morphology import disk

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
    hist = np.asarray(hist).ravel()
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


def _is_cupy_array(arr) -> bool:
    return CUPY_AVAILABLE and hasattr(arr, "__cuda_array_interface__")


def _resolve_triangle_backend(arr, prefer_gpu: bool | str):
    if prefer_gpu in (False, "cpu", "numpy"):
        return "cpu"
    if prefer_gpu in (True, "gpu", "cupy"):
        return "gpu" if CUPY_AVAILABLE else "cpu"
    if prefer_gpu == "auto":
        return "gpu" if _is_cupy_array(arr) else "cpu"
    raise ValueError("prefer_gpu must be one of auto/cpu/gpu or a bool")


def _normalize_to_u8(arr, *, xp_backend, ignore_zero: bool):
    if xp_backend is np:
        arr_xp = np.asarray(cp.asnumpy(arr) if _is_cupy_array(arr) else arr)
    else:
        arr_xp = xp_backend.asarray(arr)
    if arr_xp.size == 0:
        return xp_backend.zeros_like(arr_xp, dtype=xp_backend.uint8)

    if arr_xp.dtype == xp_backend.uint8:
        u8 = arr_xp.astype(xp_backend.uint8, copy=False)
        if ignore_zero:
            u8 = u8.copy()
        return u8

    work = arr_xp.astype(xp_backend.float16, copy=False)
    valid = work > 0 if ignore_zero else xp_backend.ones_like(work, dtype=bool)
    nz = work[valid]
    if nz.size == 0:
        return xp_backend.zeros(arr_xp.shape, dtype=xp_backend.uint8)

    vmin = nz.min().astype(xp_backend.float32, copy=False)
    vmax = nz.max().astype(xp_backend.float32, copy=False)
    if float(vmax - vmin) <= 1e-6:
        fill = 255 if float(vmax) > 0 else 0
        return xp_backend.full(arr_xp.shape, fill, dtype=xp_backend.uint8)
    span = xp_backend.maximum(vmax - vmin, xp_backend.asarray(1e-6, dtype=xp_backend.float32))
    scaled = (work.astype(xp_backend.float32, copy=False) - vmin) * (255.0 / span)
    u8 = xp_backend.clip(scaled, 0, 255).astype(xp_backend.uint8)
    if ignore_zero:
        u8 = u8.copy()
        u8[~valid] = 0
    return u8


def _maybe_blur_u8(u8, *, use_gpu: bool, blur: bool):
    if not blur:
        return u8
    if use_gpu:
        try:
            from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise RuntimeError("GPU blur unavailable") from exc

        blurred = cupy_gaussian_filter(u8.astype(cp.float32, copy=False), sigma=1.0, truncate=2.0)
        return cp.clip(cp.rint(blurred), 0, 255).astype(cp.uint8)

    blurred = gaussian_filter(np.asarray(u8, dtype=np.float32), sigma=1.0, truncate=2.0)
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def triangle_binarize_with_threshold(
    img,
    *,
    ignore_zero: bool = False,
    blur: bool = True,
    threshold_on_unblurred: bool = True,
    prefer_gpu: bool | str = "auto",
):
    """Triangle-threshold an image and return both the mask and threshold.

    The input is first normalized to an 8-bit working image. Floating-point
    inputs are processed through a float16 working copy before min-max scaling,
    which keeps the common GPU path compact and matches the repository's
    preferred low-precision workflow.

    Parameters
    ----------
    img:
        2-D or 3-D image/volume slice.
    ignore_zero:
        If ``True``, zero-valued pixels are excluded from the histogram and are
        always forced back to background.
    blur:
        Apply a small Gaussian blur before thresholding. This is useful for
        noisy histograms and is available on both CPU and GPU paths.
    threshold_on_unblurred:
        When ``True`` and ``ignore_zero`` is enabled, compute the triangle
        threshold from the unblurred 8-bit image while still applying it to the
        blurred image.
    prefer_gpu:
        ``"auto"`` uses GPU only when the input is already CuPy-backed.
        ``True``/``"gpu"`` force GPU when CuPy is available.
    """
    backend = _resolve_triangle_backend(img, prefer_gpu)

    def _run_cpu():
        u8 = _normalize_to_u8(img, xp_backend=np, ignore_zero=ignore_zero)
        u8_for_threshold = u8 if (ignore_zero and threshold_on_unblurred) else _maybe_blur_u8(u8, use_gpu=False, blur=blur)
        if int(np.max(u8_for_threshold)) == int(np.min(u8_for_threshold)):
            u8_apply = _maybe_blur_u8(u8, use_gpu=False, blur=blur)
            return (u8_apply > 0), 0
        hist = np.histogram(u8_for_threshold, bins=256, range=(0, 256))[0]
        if ignore_zero:
            hist = hist.copy()
            hist[0] = 0
        if hist.sum() == 0:
            mask = np.zeros_like(u8, dtype=bool)
            return mask, 0
        threshold = _triangle_threshold_from_hist(hist)
        u8_apply = _maybe_blur_u8(u8, use_gpu=False, blur=blur)
        return (u8_apply > threshold), int(threshold)

    def _run_gpu():
        u8 = _normalize_to_u8(img, xp_backend=cp, ignore_zero=ignore_zero)
        u8_for_threshold = u8 if (ignore_zero and threshold_on_unblurred) else _maybe_blur_u8(u8, use_gpu=True, blur=blur)
        if float(cp.max(u8_for_threshold)) == float(cp.min(u8_for_threshold)):
            u8_apply = _maybe_blur_u8(u8, use_gpu=True, blur=blur)
            return (u8_apply > 0), 0
        hist = cp.histogram(u8_for_threshold, bins=256, range=(0, 256))[0]
        if ignore_zero:
            hist = hist.copy()
            hist[0] = 0
        if int(cp.asnumpy(hist).sum()) == 0:
            mask = cp.zeros_like(u8, dtype=cp.bool_)
            return mask, 0
        threshold = _triangle_threshold_from_hist(cp.asnumpy(hist))
        u8_apply = _maybe_blur_u8(u8, use_gpu=True, blur=blur)
        return (u8_apply > threshold), int(threshold)

    if backend == "gpu":
        try:
            return _run_gpu()
        except Exception:
            mask, threshold = _run_cpu()
            if CUPY_AVAILABLE:
                return cp.asarray(mask), threshold
            return mask, threshold

    return _run_cpu()


def triangle_binarize(
    img,
    *,
    ignore_zero: bool = False,
    blur: bool = True,
    threshold_on_unblurred: bool = True,
    prefer_gpu: bool | str = "auto",
):
    """Return only the triangle-threshold mask."""
    mask, _ = triangle_binarize_with_threshold(
        img,
        ignore_zero=ignore_zero,
        blur=blur,
        threshold_on_unblurred=threshold_on_unblurred,
        prefer_gpu=prefer_gpu,
    )
    return mask


def triangle_binarize_u8(
    u8,
    blur: bool = True,
    ignore_zero: bool = False,
    threshold_on_unblurred: bool = True,
    *,
    prefer_gpu: bool | str = "auto",
):
    """Triangle-threshold an 8-bit image and return ``(mask, threshold)``."""
    return triangle_binarize_with_threshold(
        u8,
        ignore_zero=ignore_zero,
        blur=blur,
        threshold_on_unblurred=threshold_on_unblurred,
        prefer_gpu=prefer_gpu,
    )


def triangle_binarize_from_float(
    img_f32,
    blur: bool = True,
    ignore_zero: bool = False,
    threshold_on_unblurred: bool = True,
    *,
    prefer_gpu: bool | str = "auto",
):
    """Normalize float input to ``uint8`` then apply triangle thresholding."""
    return triangle_binarize_with_threshold(
        img_f32,
        ignore_zero=ignore_zero,
        blur=blur,
        threshold_on_unblurred=threshold_on_unblurred,
        prefer_gpu=prefer_gpu,
    )


triangle_binarize_gpu = triangle_binarize


__all__ = [
    "_triangle_threshold_from_hist",
    "apply_hole_filling",
    "apply_hole_filling_video",
    "apply_morph_open",
    "apply_morph_open_video",
    "binarize_video_global_threshold",
    "calculate_bw_area",
    "fill_video_holes_gpu",
    "fill_video_holes_parallel",
    "mask_video",
    "triangle_binarize",
    "triangle_binarize_from_float",
    "triangle_binarize_gpu",
    "triangle_binarize_u8",
    "triangle_binarize_with_threshold",
]
