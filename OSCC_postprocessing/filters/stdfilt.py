"""Standard-deviation filtering for images and videos.

This module wraps one common primitive used repeatedly in the spray-analysis
pipeline: local standard deviation over a square neighbourhood. In practice the
operation is used as a local-contrast measure. Regions with large texture,
sharp edges, or plume structure produce a larger local standard deviation than
uniform background.

Implementation overview
-----------------------
- CPU path: use :func:`scipy.ndimage.uniform_filter` twice to compute
  ``E[x]`` and ``E[x^2]`` and recover the local variance as
  ``Var[x] = E[x^2] - E[x]^2``.
- GPU path: for the common ``mode="nearest"`` case, use a hand-written CuPy
  RawKernel that applies nearest-neighbour clamping explicitly.
- GPU fallback path: for other border modes, fall back to
  :mod:`cupyx.scipy.ndimage`.

The public API accepts either a single frame ``(H, W)`` or a video
``(F, H, W)``. For videos the filter is purely spatial: each frame is processed
independently and there is no temporal mixing.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter as scipy_uniform_filter

try:
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter as cupy_uniform_filter

    cp.cuda.runtime.getDeviceCount()
    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    cupy_uniform_filter = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


if CUPY_AVAILABLE:
    _stdfilt_kernel = cp.RawKernel(
        r'''
extern "C" __global__
void stdfilt_kernel(
    const float* x,
    float* y,
    const int B,
    const int H,
    const int W,
    const int R
){
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= B || h >= H || w >= W) return;

    float sum = 0.0f;
    float sum2 = 0.0f;
    int count = 0;

    for (int dy = -R; dy <= R; ++dy) {
        int yy = h + dy;
        if (yy < 0) yy = 0;
        if (yy >= H) yy = H - 1;

        for (int dx = -R; dx <= R; ++dx) {
            int xx = w + dx;
            if (xx < 0) xx = 0;
            if (xx >= W) xx = W - 1;

            float v = x[(b * H + yy) * W + xx];
            sum += v;
            sum2 += v * v;
            count += 1;
        }
    }

    float mean = sum / count;
    float var = sum2 / count - mean * mean;
    if (var < 0.0f) var = 0.0f;
    y[(b * H + h) * W + w] = sqrtf(var);
}
''',
        "stdfilt_kernel",
    )
    RAWKERNEL_AVAILABLE = True
else:
    _stdfilt_kernel = None
    RAWKERNEL_AVAILABLE = False


__all__ = [
    "CUPY_AVAILABLE",
    "RAWKERNEL_AVAILABLE",
    "stdfilt",
    "stdfilt_cupy",
    "stdfilt_video_auto",
    "stdfilt_video_cpu",
    "stdfilt_video_cupy",
    "stdfilt_video_per_frame_cpu",
]


def _validate_ksize(ksize: int) -> None:
    """Validate that the filter support is a positive odd integer.

    A symmetric square neighbourhood needs an odd width so that the current
    pixel remains the geometric centre of the window.
    """
    if not isinstance(ksize, int) or ksize <= 0 or ksize % 2 != 1:
        raise ValueError("ksize must be a positive odd integer.")


def _cpu_array(video) -> np.ndarray:
    """Return a host NumPy array, copying from GPU only when required."""
    if CUPY_AVAILABLE and hasattr(video, "__cuda_array_interface__"):
        return cp.asnumpy(video)  # type: ignore[union-attr]
    return np.asarray(video)


def _filter_size(array_ndim: int, ksize: int):
    """Map input dimensionality to a SciPy/CuPy ``size=`` tuple.

    ``(H, W)`` inputs use a plain 2D square window. ``(F, H, W)`` inputs keep
    the temporal axis untouched by using a filter size of ``1`` along frames.
    """
    if array_ndim == 2:
        return (ksize, ksize)
    if array_ndim == 3:
        return (1, ksize, ksize)
    raise ValueError("Input must have shape (H, W) or (frame, height, width).")


def _stdfilt_impl(array, uniform_filter_fn, xp, ksize: int = 3, mode: str = "nearest"):
    """Backend-agnostic stdfilt implementation based on local moments.

    The filter computes the standard deviation inside each local window via the
    identity ``std = sqrt(max(E[x^2] - E[x]^2, 0))``.

    The clamp is important because finite-precision arithmetic can produce a
    tiny negative variance even when the theoretical value is non-negative.
    """
    _validate_ksize(ksize)
    arr = xp.asarray(array, dtype=xp.float32)
    size = _filter_size(arr.ndim, ksize)

    mean_val = uniform_filter_fn(arr, size=size, mode=mode)
    mean_sq_val = uniform_filter_fn(arr * arr, size=size, mode=mode)
    var_val = xp.maximum(mean_sq_val - mean_val * mean_val, xp.float32(0.0))
    return xp.sqrt(var_val).astype(xp.float32, copy=False)


def _stdfilt_cupy_rawkernel(video, ksize: int = 3):
    """Run the custom CuPy RawKernel implementation.

    This kernel is intentionally specialized:

    - input is promoted to contiguous ``float32``
    - border handling is fixed to nearest-neighbour clamping
    - the computation is frame-wise for ``(F, H, W)`` data

    The specialization keeps the GPU path simple and fast for the main workload
    in this repository.
    """
    _validate_ksize(ksize)
    if not RAWKERNEL_AVAILABLE:
        raise RuntimeError("CuPy RawKernel stdfilt backend is not available.")

    arr = cp.asarray(video, dtype=cp.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze_output = True
    elif arr.ndim == 3:
        squeeze_output = False
    else:
        raise ValueError("Input must have shape (H, W) or (frame, height, width).")

    arr = cp.ascontiguousarray(arr)
    batch, height, width = map(int, arr.shape)
    out = cp.empty_like(arr)
    radius = np.int32(ksize // 2)

    block = (16, 16, 1)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
        batch,
    )

    _stdfilt_kernel(
        grid,
        block,
        (
            arr,
            out,
            np.int32(batch),
            np.int32(height),
            np.int32(width),
            radius,
        ),
    )

    if squeeze_output:
        return out[0]
    return out


def stdfilt_video_cpu(video, ksize: int = 3, mode: str = "nearest") -> np.ndarray:
    """Apply frame-wise spatial standard-deviation filtering on CPU.

    Parameters
    ----------
    video:
        Input frame ``(H, W)`` or video ``(F, H, W)``.
    ksize:
        Odd side length of the square neighbourhood.
    mode:
        Border-extension mode forwarded to SciPy.
    """
    return _stdfilt_impl(video, scipy_uniform_filter, np, ksize=ksize, mode=mode)


def stdfilt_video_cupy(video, ksize: int = 3, mode: str = "nearest"):
    """Apply a 2D standard-deviation filter per frame on GPU.

    ``mode="nearest"`` prefers the RawKernel backend because the kernel
    implements nearest-edge clamping directly and avoids the overhead of the
    more general ``cupyx.scipy.ndimage`` path. Other border modes fall back to
    ``cupyx.scipy.ndimage.uniform_filter``.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available on this machine.")
    if mode == "nearest" and RAWKERNEL_AVAILABLE:
        try:
            return _stdfilt_cupy_rawkernel(video, ksize=ksize)
        except Exception as exc:  # pragma: no cover - runtime hardware dependent
            print(f"RawKernel stdfilt failed ({exc}), falling back to cupyx.")
    return _stdfilt_impl(video, cupy_uniform_filter, cp, ksize=ksize, mode=mode)


def stdfilt_video_auto(video, ksize: int = 3, mode: str = "nearest", backend: str = "auto"):
    """Apply standard-deviation filtering with explicit or automatic routing.

    ``backend="auto"`` prefers GPU when CuPy is available and falls back to
    CPU if GPU setup or execution fails.
    """
    if backend == "cupy":
        return stdfilt_video_cupy(video, ksize=ksize, mode=mode)
    if backend == "scipy":
        return stdfilt_video_cpu(_cpu_array(video), ksize=ksize, mode=mode)
    if backend != "auto":
        raise ValueError("backend must be 'auto', 'scipy', or 'cupy'.")

    if CUPY_AVAILABLE:
        try:
            return stdfilt_video_cupy(video, ksize=ksize, mode=mode)
        except Exception as exc:  # pragma: no cover - runtime hardware dependent
            print(f"GPU stdfilt failed ({exc}), falling back to CPU.")

    return stdfilt_video_cpu(_cpu_array(video), ksize=ksize, mode=mode)


def stdfilt(video, ksize: int = 3, mode: str = "nearest", backend: str = "auto"):
    """Public convenience wrapper around :func:`stdfilt_video_auto`.

    This name matches the historic MATLAB-style helper used elsewhere in the
    project, so higher-level code can call ``stdfilt(...)`` regardless of the
    selected execution backend.
    """
    return stdfilt_video_auto(video, ksize=ksize, mode=mode, backend=backend)


def stdfilt_video_per_frame_cpu(video, nhood: int = 3) -> np.ndarray:
    """Backward-compatible alias for the original CPU helper name."""
    return stdfilt_video_cpu(video, ksize=nhood)


def stdfilt_cupy(x, ksize: int = 3):
    """Backward-compatible alias for the original GPU helper name."""
    return stdfilt_video_cupy(x, ksize=ksize)
