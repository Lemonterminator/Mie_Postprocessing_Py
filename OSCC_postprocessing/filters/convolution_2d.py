"""2D spatial convolution helpers for frame-wise video filtering.

Despite the historical name, this module does not use a custom RawKernel.
GPU execution routes through ``cupyx.scipy.ndimage`` when CUDA is available.
CPU fallback uses threaded OpenCV or SciPy frame-by-frame filtering.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from scipy import ndimage as ndi

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cnd

    cp.cuda.runtime.getDeviceCount()
    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    cnd = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


__all__ = [
    "CUPY_AVAILABLE",
    "build_2d_kernel",
    "convolve_video_2d",
    "convolution_2D_auto",
    "convolution_2D_cupy",
    "make_kernel",
]


def _mode_map_ndimage(mode: str) -> str:
    mode = (mode or "").lower()
    if mode in {"edge", "nearest"}:
        return "nearest"
    if mode in {"reflect", "mirror"}:
        return "reflect"
    if mode == "wrap":
        return "wrap"
    if mode in {"constant", "const", "zeros", "zero"}:
        return "constant"
    return "nearest"


def _mode_map_cv2(mode: str) -> int | None:
    mode = (mode or "").lower()
    if mode in {"edge", "nearest"}:
        return cv2.BORDER_REPLICATE
    if mode in {"reflect", "mirror"}:
        return cv2.BORDER_REFLECT_101
    if mode == "wrap":
        return cv2.BORDER_WRAP
    if mode in {"constant", "const", "zeros", "zero"}:
        return cv2.BORDER_CONSTANT
    return cv2.BORDER_REPLICATE


def _video_input_is_gpu(video) -> bool:
    return CUPY_AVAILABLE and hasattr(video, "__cuda_array_interface__")


def _return_result(result, *, backend: str, input_was_gpu: bool):
    if backend == "cupy" and CUPY_AVAILABLE:
        return result
    if backend == "numpy":
        return cp.asnumpy(result) if CUPY_AVAILABLE and hasattr(result, "__cuda_array_interface__") else np.asarray(result)
    if input_was_gpu:
        return result
    if CUPY_AVAILABLE and hasattr(result, "__cuda_array_interface__"):
        return cp.asnumpy(result)
    return np.asarray(result)


def _separable_factors_numpy(kernel: np.ndarray, tol: float = 1e-10):
    u, s, vt = np.linalg.svd(kernel, full_matrices=False)
    rank = int(np.sum(s > tol))
    if rank != 1:
        return None
    s0 = s[0]
    ky = u[:, 0] * np.sqrt(s0)
    kx = vt[0, :] * np.sqrt(s0)
    return ky.astype(np.float32), kx.astype(np.float32)


def _separable_factors_cupy(kernel_gpu, tol: float = 1e-10):
    u, s, vt = cp.linalg.svd(kernel_gpu, full_matrices=False)
    rank = int(cp.sum(s > tol).get())
    if rank != 1:
        return None
    s0 = s[0]
    ky = u[:, 0] * cp.sqrt(s0)
    kx = vt[0, :] * cp.sqrt(s0)
    return ky.astype(cp.float32), kx.astype(cp.float32)


def _convolve_frame_cpu(frame, kernel, *, mode: str, cval: float, tol: float):
    kernel_np = np.asarray(kernel, dtype=np.float32)
    frame_np = np.asarray(frame, dtype=np.float32)
    border_type = _mode_map_cv2(mode)
    separable = _separable_factors_numpy(kernel_np, tol=tol)

    # ``filter2D``/``sepFilter2D`` are preferred when their border semantics match.
    can_use_cv2 = border_type is not None and ((mode or "").lower() != "constant" or float(cval) == 0.0)

    if can_use_cv2:
        if separable is not None:
            ky, kx = separable
            return cv2.sepFilter2D(frame_np, ddepth=-1, kernelX=kx, kernelY=ky, borderType=border_type)
        return cv2.filter2D(frame_np, ddepth=-1, kernel=kernel_np, borderType=border_type)

    nd_mode = _mode_map_ndimage(mode)
    if separable is not None:
        ky, kx = separable
        tmp = ndi.convolve1d(frame_np, kx, axis=1, mode=nd_mode, cval=cval)
        return ndi.convolve1d(tmp, ky, axis=0, mode=nd_mode, cval=cval)
    return ndi.convolve(frame_np, kernel_np, mode=nd_mode, cval=cval)


def _convolve_video_cpu(video, kernel, *, mode: str, cval: float, tol: float, max_workers: int | None):
    video_np = np.asarray(video, dtype=np.float32)
    if video_np.ndim != 3:
        raise ValueError(f"video must be (F,H,W), got shape={video_np.shape}")

    frame_count = video_np.shape[0]
    if frame_count == 0:
        return np.empty_like(video_np)

    if max_workers is None:
        max_workers = min(frame_count, max(1, (os.cpu_count() or 1) - 1))

    out = np.empty_like(video_np)

    def work(frame_idx: int):
        return frame_idx, _convolve_frame_cpu(video_np[frame_idx], kernel, mode=mode, cval=cval, tol=tol)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(work, frame_idx) for frame_idx in range(frame_count)]
        for future in as_completed(futures):
            frame_idx, filtered = future.result()
            out[frame_idx] = filtered

    return out


def _convolve_video_gpu(video, kernel, *, mode: str, cval: float, tol: float):
    video_gpu = cp.asarray(video, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel, dtype=cp.float32)

    if video_gpu.ndim != 3:
        raise ValueError(f"video must be (F,H,W), got shape={video_gpu.shape}")
    if kernel_gpu.ndim != 2:
        raise ValueError(f"kernel must be (kH,kW), got shape={kernel_gpu.shape}")

    k_h, k_w = kernel_gpu.shape
    if (k_h % 2) != 1 or (k_w % 2) != 1:
        raise ValueError(f"kernel size should be odd, got {kernel_gpu.shape}")

    nd_mode = _mode_map_ndimage(mode)
    separable = _separable_factors_cupy(kernel_gpu, tol=tol)
    out = cp.empty_like(video_gpu, dtype=cp.float32)

    if separable is not None:
        ky, kx = separable
        for frame_idx in range(video_gpu.shape[0]):
            tmp = cnd.convolve1d(video_gpu[frame_idx], kx, axis=1, mode=nd_mode, cval=cval)
            out[frame_idx] = cnd.convolve1d(tmp, ky, axis=0, mode=nd_mode, cval=cval)
    else:
        for frame_idx in range(video_gpu.shape[0]):
            out[frame_idx] = cnd.convolve(video_gpu[frame_idx], kernel_gpu, mode=nd_mode, cval=cval)

    return out


def convolve_video_2d(
    video,
    kernel,
    mode: str = "edge",
    tol: float = 1e-10,
    cval: float = 0.0,
    *,
    backend: str = "auto",
    max_workers: int | None = None,
):
    """Convolve each frame of a ``(frame, height, width)`` video with a 2D kernel.

    Parameters
    ----------
    video : array-like
        Input video with shape ``(F, H, W)``.
    kernel : array-like
        Spatial convolution kernel with odd size.
    mode : str
        Border mode. Supports ``edge``, ``reflect``, ``wrap`` and ``constant``.
    tol : float
        Rank-1 tolerance used to route separable kernels through 1D convolution.
    cval : float
        Constant border value for ``mode="constant"``.
    backend : {"auto", "cupy", "numpy"}
        Execution backend. ``auto`` prefers GPU when available.
    max_workers : int, optional
        CPU worker count when running the threaded fallback.
    """
    input_was_gpu = _video_input_is_gpu(video)

    if backend not in {"auto", "cupy", "numpy"}:
        raise ValueError("backend must be 'auto', 'cupy', or 'numpy'.")

    use_gpu = backend in {"auto", "cupy"} and CUPY_AVAILABLE

    if use_gpu:
        result = _convolve_video_gpu(video, kernel, mode=mode, cval=cval, tol=tol)
        return _return_result(result, backend=backend, input_was_gpu=input_was_gpu)

    result = _convolve_video_cpu(
        video,
        kernel,
        mode=mode,
        cval=cval,
        tol=tol,
        max_workers=max_workers,
    )
    return _return_result(result, backend="numpy", input_was_gpu=input_was_gpu)


def convolution_2D_auto(video, kernel, mode="edge", tol=1e-10, cval=0.0, max_workers=None):
    """Compatibility wrapper for automatic backend selection."""
    return convolve_video_2d(
        video,
        kernel,
        mode=mode,
        tol=tol,
        cval=cval,
        backend="auto",
        max_workers=max_workers,
    )


def convolution_2D_cupy(video, kernel, mode="edge", tol=1e-10, cval=0.0, max_workers=None):
    """Historical entry point that now routes to GPU when available, else CPU."""
    return convolve_video_2d(
        video,
        kernel,
        mode=mode,
        tol=tol,
        cval=cval,
        backend="cupy" if CUPY_AVAILABLE else "numpy",
        max_workers=max_workers,
    )


def build_2d_kernel(
    name: str,
    wsize: int,
    sigma: float | None = None,
    direction: str | None = None,
    normalize: bool = True,
):
    """Build standard 2D spatial kernels used by the legacy pipeline."""
    if wsize % 2 != 1:
        raise ValueError("wsize must be odd")

    k = wsize // 2
    ax = np.arange(-k, k + 1, dtype=np.float32)
    kernel_name = name.lower()

    if kernel_name == "gaussian":
        if sigma is None:
            raise ValueError("sigma required for gaussian")
        g = np.exp(-(ax**2) / (2 * sigma**2))
        g /= g.sum()
        kernel = np.outer(g, g)
    elif kernel_name == "sobel":
        if sigma is None:
            raise ValueError("sigma required for sobel")
        if direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'")
        g = np.exp(-(ax**2) / (2 * sigma**2))
        g /= g.sum()
        dg = -ax * np.exp(-(ax**2) / (2 * sigma**2))
        dg /= np.sum(np.abs(dg))
        kernel = np.outer(g, dg) if direction == "x" else np.outer(dg, g)
    elif kernel_name == "prewitt":
        if direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'")
        smooth = np.ones_like(ax) / wsize
        deriv = ax / np.sum(np.abs(ax))
        kernel = np.outer(smooth, deriv) if direction == "x" else np.outer(deriv, smooth)
    elif kernel_name == "laplacian":
        if wsize != 3:
            raise ValueError("laplacian usually defined for wsize=3")
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    elif kernel_name == "log":
        if sigma is None:
            raise ValueError("sigma required for log")
        xx, yy = np.meshgrid(ax, ax, indexing="ij")
        r2 = xx**2 + yy**2
        s2 = sigma**2
        kernel = (r2 - 2 * s2) / (s2**2) * np.exp(-r2 / (2 * s2))
        kernel -= kernel.mean()
    else:
        raise ValueError(f"Unknown kernel name: {name}")

    if normalize and kernel_name == "gaussian":
        kernel /= kernel.sum()

    return kernel.astype(np.float32)


def make_kernel(name: str, wsize: int, sigma: float = None, direction: str = None, normalize: bool = True):
    """Historical alias for ``build_2d_kernel``."""
    return build_2d_kernel(name, wsize, sigma=sigma, direction=direction, normalize=normalize)
