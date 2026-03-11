"""Morphology operators for binary or grayscale videos/volumes.

The original implementation lived in ``OSCC_postprocessing.morphology`` and
only supported frame-wise 2D processing. The implementation now lives in
``binary_ops`` because it is part of the same mask-processing layer.

Two execution modes are supported:

- ``mode="2D"``: apply the operator independently to each frame in a video
- ``mode="3D"``: apply the operator to the full ``(T, H, W)`` volume
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import diamond, octagon

from ._backend import CUPY_AVAILABLE, cndi, cp, return_like_input

__all__ = [
    "dilate_video_parallel",
    "erode_video_parallel",
    "open_video_parallel",
    "close_video_parallel",
]


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).upper()
    if normalized not in {"2D", "3D"}:
        raise ValueError(f"mode must be '2D' or '3D', got {mode!r}")
    return normalized


def _normalize_kernel_size(kernel_size, mode: str) -> tuple[int, ...]:
    """Normalize and validate kernel size for the requested mode."""
    mode = _normalize_mode(mode)
    if kernel_size is None:
        return (3, 3, 3) if mode == "3D" else (3, 3)

    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size, kernel_size) if mode == "3D" else (kernel_size, kernel_size)

    normalized = tuple(int(v) for v in kernel_size)
    expected_dims = 3 if mode == "3D" else 2
    if len(normalized) != expected_dims:
        raise ValueError(
            f"kernel_size must have {expected_dims} values in mode={mode}, got {normalized}"
        )
    if any(v <= 0 for v in normalized):
        raise ValueError(f"kernel_size must be positive, got {normalized}")
    return normalized


def _build_cross_kernel_nd(kernel_size: tuple[int, ...]) -> np.ndarray:
    """Build an axis-aligned cross kernel in 2D or 3D."""
    kernel = np.zeros(kernel_size, dtype=np.uint8)
    center = tuple(size // 2 for size in kernel_size)
    for axis in range(len(kernel_size)):
        slices = [c for c in center]
        slices[axis] = slice(0, kernel_size[axis])
        kernel[tuple(slices)] = 1
    return kernel


def _build_ellipse_kernel_3d(kernel_size: tuple[int, int, int]) -> np.ndarray:
    """Approximate an ellipsoid footprint for 3D morphology."""
    radii = np.maximum(np.asarray(kernel_size, dtype=np.float32) / 2.0, 0.5)
    zz, yy, xx = np.indices(kernel_size, dtype=np.float32)
    center = (np.asarray(kernel_size, dtype=np.float32) - 1.0) / 2.0
    dist = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    return (dist <= 1.0).astype(np.uint8)


def _build_diamond_kernel_3d(kernel_size: tuple[int, int, int]) -> np.ndarray:
    """Approximate a 3D diamond using an L1 ball."""
    radii = np.maximum((np.asarray(kernel_size, dtype=np.float32) - 1.0) / 2.0, 0.5)
    zz, yy, xx = np.indices(kernel_size, dtype=np.float32)
    center = (np.asarray(kernel_size, dtype=np.float32) - 1.0) / 2.0
    dist = (
        np.abs((zz - center[0]) / radii[0])
        + np.abs((yy - center[1]) / radii[1])
        + np.abs((xx - center[2]) / radii[2])
    )
    return (dist <= 1.0).astype(np.uint8)


def _build_kernel(kernel_shape="Ellipse", kernel_size=(3, 3), mode="2D"):
    """Build a structuring element for 2D or 3D morphology."""
    mode = _normalize_mode(mode)
    kernel_shape = str(kernel_shape).capitalize()
    kernel_size = _normalize_kernel_size(kernel_size, mode)

    if mode == "2D":
        if kernel_shape == "Ellipse":
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        if kernel_shape == "Rectangle":
            return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        if kernel_shape == "Cross":
            return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        if kernel_shape == "Diamond":
            return diamond(kernel_size[0]).astype(np.uint8)
        if kernel_shape == "Octagon":
            return octagon(kernel_size[0], kernel_size[0]).astype(np.uint8)
        raise ValueError(
            "Unsupported kernel_shape. Choose from Ellipse, Rectangle, Cross, Diamond, Octagon."
        )

    if kernel_shape == "Ellipse":
        return _build_ellipse_kernel_3d(kernel_size)
    if kernel_shape == "Rectangle":
        return np.ones(kernel_size, dtype=np.uint8)
    if kernel_shape == "Cross":
        return _build_cross_kernel_nd(kernel_size)
    if kernel_shape == "Diamond":
        return _build_diamond_kernel_3d(kernel_size)
    if kernel_shape == "Octagon":
        raise ValueError("kernel_shape='Octagon' is only implemented for mode='2D'.")
    raise ValueError(
        "Unsupported kernel_shape. Choose from Ellipse, Rectangle, Cross, Diamond."
    )


def _morph_framewise(video, op, kernel, max_workers):
    """Apply a 2D OpenCV morphology op to each frame independently."""
    if video.ndim not in (3, 4):
        raise ValueError("2D mode expects video with shape (T,H,W) or (T,H,W,C).")

    out = np.empty_like(video)

    def _one(idx: int):
        frame = np.ascontiguousarray(video[idx])
        return idx, op(frame, kernel)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, idx) for idx in range(video.shape[0])]
        for future in as_completed(futures):
            idx, result = future.result()
            out[idx] = result
    return out


def _morph_volume(video, op_name: str, kernel, iterations: int):
    """Apply a grayscale morphology operation to the full 3D volume."""
    if video.ndim != 3:
        raise ValueError("3D mode expects video with shape (T,H,W).")

    func_map = {
        "dilate": ndimage.grey_dilation,
        "erode": ndimage.grey_erosion,
        "open": ndimage.grey_opening,
        "close": ndimage.grey_closing,
    }
    if op_name not in func_map:
        raise ValueError(f"Unsupported op_name {op_name!r}")

    result = video
    footprint = kernel.astype(bool, copy=False)
    func = func_map[op_name]
    for _ in range(max(1, int(iterations))):
        result = func(result, footprint=footprint)
    return result


def _gpu_footprint(kernel: np.ndarray, *, mode: str, ndim: int):
    """Expand a spatial kernel into an nD footprint for cndi morphology."""
    footprint = kernel.astype(bool, copy=False)
    if mode == "3D":
        if ndim != 3:
            raise ValueError("3D GPU morphology expects a volume with shape (T,H,W).")
        return cp.asarray(footprint)

    if ndim == 3:
        return cp.asarray(footprint[None, :, :])
    if ndim == 4:
        return cp.asarray(footprint[None, :, :, None])
    raise ValueError("2D GPU morphology expects video with shape (T,H,W) or (T,H,W,C).")


def _morph_gpu(video, *, op_name: str, kernel, iterations: int, mode: str, like):
    """Apply morphology with cupyx.scipy.ndimage and preserve the input backend."""
    func_map = {
        "dilate": cndi.grey_dilation,
        "erode": cndi.grey_erosion,
        "open": cndi.grey_opening,
        "close": cndi.grey_closing,
    }
    if op_name not in func_map:
        raise ValueError(f"Unsupported op_name {op_name!r}")

    video_gpu = video if isinstance(video, cp.ndarray) else cp.asarray(video)
    footprint = _gpu_footprint(kernel, mode=mode, ndim=video_gpu.ndim)
    result = video_gpu
    func = func_map[op_name]
    for _ in range(max(1, int(iterations))):
        result = func(result, footprint=footprint)
    return return_like_input(result, like)


def _morph_video(
    video,
    *,
    op_name: str,
    op_2d,
    kernel_shape="Ellipse",
    kernel_size=None,
    iterations=1,
    max_workers=None,
    mode="2D",
):
    """Shared morphology runner for both frame-wise and volume-wise modes.

    Routing policy:
    - prefer ``cupyx.scipy.ndimage`` when CUDA is available and the input shape
      is supported
    - otherwise fall back to the CPU implementation
    """
    mode = _normalize_mode(mode)
    kernel_size = _normalize_kernel_size(kernel_size, mode)

    if hasattr(video, "__cuda_array_interface__"):
        original_dtype = video.dtype
        working = video.astype(cp.float32, copy=False) if video.dtype != cp.float32 else video
    else:
        video_np = np.asarray(video)
        original_dtype = video_np.dtype
        working = video_np.astype(np.float32, copy=False) if video_np.dtype != np.float32 else video_np
    kernel = _build_kernel(kernel_shape=kernel_shape, kernel_size=kernel_size, mode=mode)

    gpu_supported = CUPY_AVAILABLE and getattr(working, "ndim", 0) in (3, 4)
    if mode == "3D":
        gpu_supported = gpu_supported and getattr(working, "ndim", 0) == 3

    if gpu_supported:
        try:
            return _morph_gpu(
                working,
                op_name=op_name,
                kernel=kernel,
                iterations=iterations,
                mode=mode,
                like=video,
            ).astype(original_dtype, copy=False)
        except Exception:
            # Fall back to CPU for unsupported dtypes / kernels / local runtime issues.
            pass

    working_cpu = np.asarray(cp.asnumpy(working) if hasattr(working, "__cuda_array_interface__") else working)
    if mode == "2D":
        workers = max_workers if max_workers is not None else (os.cpu_count() or 4)
        result = _morph_framewise(working_cpu, op_2d, kernel, workers)
    else:
        result = _morph_volume(working_cpu, op_name=op_name, kernel=kernel, iterations=iterations)

    return result.astype(original_dtype, copy=False)


def dilate_video_parallel(
    video,
    kernel_shape="Ellipse",
    kernel_size=None,
    iterations=1,
    max_workers=None,
    mode="2D",
):
    """Dilate a video frame-wise or as a full 3D volume.

    ``kernel_size`` defaults to ``(3, 3)`` in ``mode="2D"`` and ``(3, 3, 3)``
    in ``mode="3D"``.
    """
    return _morph_video(
        video,
        op_name="dilate",
        op_2d=lambda frame, kernel: cv2.dilate(frame, kernel, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        iterations=iterations,
        max_workers=max_workers,
        mode=mode,
    )


def erode_video_parallel(
    video,
    kernel_shape="Ellipse",
    kernel_size=None,
    iterations=1,
    max_workers=None,
    mode="2D",
):
    """Erode a video frame-wise or as a full 3D volume."""
    return _morph_video(
        video,
        op_name="erode",
        op_2d=lambda frame, kernel: cv2.erode(frame, kernel, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        iterations=iterations,
        max_workers=max_workers,
        mode=mode,
    )


def open_video_parallel(
    video,
    kernel_shape="Ellipse",
    kernel_size=None,
    iterations=1,
    max_workers=None,
    mode="2D",
):
    """Open a video frame-wise or as a full 3D volume."""
    return _morph_video(
        video,
        op_name="open",
        op_2d=lambda frame, kernel: cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        iterations=iterations,
        max_workers=max_workers,
        mode=mode,
    )


def close_video_parallel(
    video,
    kernel_shape="Ellipse",
    kernel_size=None,
    iterations=1,
    max_workers=None,
    mode="2D",
):
    """Close a video frame-wise or as a full 3D volume."""
    return _morph_video(
        video,
        op_name="close",
        op_2d=lambda frame, kernel: cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        iterations=iterations,
        max_workers=max_workers,
        mode=mode,
    )
