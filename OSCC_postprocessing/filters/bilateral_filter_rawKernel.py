"""Canonical bilateral filtering backends for 2D video and 3D volumes.

GPU execution uses a CuPy RawKernel for mature 2D frame-wise filtering.
When CUDA is unavailable, the same public API falls back to a threaded CPU
implementation based on OpenCV's bilateral filter.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


__all__ = [
    "CUPY_AVAILABLE",
    "bilateral_filter_img",
    "bilateral_filter_img_cupy",
    "bilateral_filter_video_cpu",
    "bilateral_filter_video_cupy_fast",
    "bilateral_filter_video_cupy",
    "bilateral_filter_video_volumetric_cpu",
    "bilateral_filter_video_volumetric_chunked_halo",
    "estimate_chunk_size",
]


if CUPY_AVAILABLE:
    _bilateral_kernel = cp.RawKernel(
        r'''
extern "C" __global__
void bilateral2d(
    const float* __restrict__ pad,
    float* __restrict__ out,
    const float* __restrict__ spatial,
    int H, int W, int Wp,
    int k, int wsize,
    float inv_2sr2,
    float eps
){
    int x = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    int y = (int)(blockDim.y * blockIdx.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int cx = x + k;
    int cy = y + k;
    float c = pad[cy * Wp + cx];

    float sum_w = 0.0f;
    float sum_wp = 0.0f;
    int base_y = cy - k;
    int base_x = cx - k;
    int sidx = 0;

    for (int dy = 0; dy < wsize; ++dy){
        int py = base_y + dy;
        int row = py * Wp;
        for (int dx = 0; dx < wsize; ++dx, ++sidx){
            float p = pad[row + (base_x + dx)];
            float d = p - c;
            float range = __expf(-(d * d) * inv_2sr2);
            float w = range * spatial[sidx];
            sum_w += w;
            sum_wp += w * p;
        }
    }
    out[y * W + x] = sum_wp / (sum_w + eps);
}
''',
        "bilateral2d",
    )
    _bilateral_kernel_3d = cp.RawKernel(
        r'''
extern "C" __global__
void bilateral3d(
    const float* __restrict__ pad,
    float* __restrict__ out,
    const float* __restrict__ spatial,
    int F, int H, int W,
    int Hp, int Wp,
    int region_start,
    int kf, int kh, int kw,
    int wsize_f, int wsize_h, int wsize_w,
    float inv_2sr2,
    float eps
){
    int x = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    int y = (int)(blockDim.y * blockIdx.y + threadIdx.y);
    int z = (int)(blockDim.z * blockIdx.z + threadIdx.z);
    if (x >= W || y >= H || z >= F) return;

    int cpz = z + region_start + kf;
    int cpy = y + kh;
    int cpx = x + kw;

    int pad_slice = Hp * Wp;
    float c = pad[cpz * pad_slice + cpy * Wp + cpx];

    float sum_w = 0.0f;
    float sum_wp = 0.0f;
    int sidx = 0;

    for (int dz = 0; dz < wsize_f; ++dz){
        int pz = cpz - kf + dz;
        int slice = pz * pad_slice;
        for (int dy = 0; dy < wsize_h; ++dy){
            int py = cpy - kh + dy;
            int row = slice + py * Wp;
            for (int dx = 0; dx < wsize_w; ++dx, ++sidx){
                float p = pad[row + (cpx - kw + dx)];
                float d = p - c;
                float range = __expf(-(d * d) * inv_2sr2);
                float w = range * spatial[sidx];
                sum_w += w;
                sum_wp += w * p;
            }
        }
    }

    out[z * H * W + y * W + x] = sum_wp / (sum_w + eps);
}
''',
        "bilateral3d",
    )
else:
    _bilateral_kernel = None
    _bilateral_kernel_3d = None


def _validate_wsize(wsize: int) -> None:
    if not isinstance(wsize, int) or wsize <= 0 or wsize % 2 != 1:
        raise ValueError("wsize must be a positive odd integer.")


def _normalize_volumetric_wsize(wsize):
    """Normalize volumetric window size to a validated 3-tuple."""
    if isinstance(wsize, int):
        _validate_wsize(wsize)
        return wsize, wsize, wsize

    try:
        wsize_f, wsize_h, wsize_w = wsize
    except Exception as exc:
        raise ValueError("wsize must be an odd int or a 3-tuple (F, H, W).") from exc
    if not all(isinstance(v, int) for v in (wsize_f, wsize_h, wsize_w)):
        raise ValueError("wsize tuple must contain ints (F, H, W).")
    if not all(v > 0 and v % 2 == 1 for v in (wsize_f, wsize_h, wsize_w)):
        raise ValueError("wsize tuple values must be positive odd integers.")
    return wsize_f, wsize_h, wsize_w


def _cpu_array(video) -> np.ndarray:
    """Return a CPU ndarray, copying from CuPy only when necessary."""
    if CUPY_AVAILABLE and hasattr(video, "__cuda_array_interface__"):
        return cp.asnumpy(video)  # type: ignore[union-attr]
    return np.asarray(video)


def bilateral_filter_img(img, wsize, sigma_d, sigma_r):
    """CPU bilateral filter for a single 2D frame using OpenCV."""
    _validate_wsize(wsize)
    frame = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
    return cv2.bilateralFilter(
        frame,
        d=wsize,
        sigmaColor=float(sigma_r),
        sigmaSpace=float(sigma_d),
    )


def bilateral_filter_img_cupy(img, wsize, sigma_d, sigma_r, mode="edge"):
    """Single-frame GPU bilateral filter with CPU fallback."""
    filtered = bilateral_filter_video_cupy_fast(
        np.asarray(img)[None, :, :] if not (CUPY_AVAILABLE and hasattr(img, "__cuda_array_interface__")) else img[None, :, :],
        wsize=wsize,
        sigma_d=sigma_d,
        sigma_r=sigma_r,
        mode=mode,
    )
    return filtered[0]


def bilateral_filter_video_cpu(video, wsize, sigma_d, sigma_r, max_workers=None):
    """Threaded CPU fallback using OpenCV's bilateral filter frame by frame."""
    _validate_wsize(wsize)
    video_np = np.ascontiguousarray(_cpu_array(video), dtype=np.float32)
    if video_np.ndim != 3:
        raise ValueError("video must have shape (frame, height, width).")

    frame_count = video_np.shape[0]
    if frame_count == 0:
        return np.empty_like(video_np)

    if max_workers is None:
        max_workers = min(frame_count, max(1, (os.cpu_count() or 1) - 1))

    result = np.empty_like(video_np)

    def work(frame_idx: int):
        filtered = cv2.bilateralFilter(
            video_np[frame_idx],
            d=wsize,
            sigmaColor=float(sigma_r),
            sigmaSpace=float(sigma_d),
        )
        return frame_idx, filtered

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(work, frame_idx) for frame_idx in range(frame_count)]
        for future in as_completed(futures):
            frame_idx, filtered = future.result()
            result[frame_idx] = filtered

    return result


def bilateral_filter_video_cupy_fast(
    video,
    wsize,
    sigma_d,
    sigma_r,
    mode="edge",
    block=(16, 16),
    eps=1e-8,
    max_workers=None,
):
    """RawKernel bilateral filter with threaded CPU fallback when CUDA is unavailable."""
    _validate_wsize(wsize)

    if not CUPY_AVAILABLE:
        return bilateral_filter_video_cpu(
            video,
            wsize=wsize,
            sigma_d=sigma_d,
            sigma_r=sigma_r,
            max_workers=max_workers,
        )

    video_gpu = cp.asarray(video, dtype=cp.float32)
    if video_gpu.ndim != 3:
        raise ValueError("video must have shape (frame, height, width).")

    frame_count, height, width = video_gpu.shape
    k = wsize // 2

    pad = cp.pad(video_gpu, ((0, 0), (k, k), (k, k)), mode=mode)
    _, _, padded_width = pad.shape

    ax = cp.arange(-k, k + 1, dtype=cp.float32)
    xx, yy = cp.meshgrid(ax, ax, indexing="ij")
    inv_2sd2 = cp.float32(1.0) / (2.0 * cp.float32(sigma_d) * cp.float32(sigma_d))
    spatial = cp.exp(-(xx * xx + yy * yy) * inv_2sd2).astype(cp.float32).ravel()

    inv_2sr2 = cp.float32(1.0) / (2.0 * cp.float32(sigma_r) * cp.float32(sigma_r))
    eps_gpu = cp.float32(eps)
    out = cp.empty((frame_count, height, width), dtype=cp.float32)

    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
    )

    for frame_idx in range(frame_count):
        _bilateral_kernel(
            (grid[0], grid[1]),
            block,
            (
                pad[frame_idx],
                out[frame_idx],
                spatial,
                height,
                width,
                padded_width,
                k,
                wsize,
                inv_2sr2,
                eps_gpu,
            ),
        )

    return out


def bilateral_filter_video_cupy(video, wsize, sigma_d, sigma_r, mode="edge", **kwargs):
    """Public 2D video bilateral filter entry point with auto CPU/GPU routing."""
    return bilateral_filter_video_cupy_fast(
        video,
        wsize=wsize,
        sigma_d=sigma_d,
        sigma_r=sigma_r,
        mode=mode,
        **kwargs,
    )


def bilateral_filter_video_volumetric_cpu(video, wsize, sigma_d, sigma_r, mode="edge"):
    """Reference CPU implementation of full 3D bilateral filtering."""
    if not isinstance(wsize, int):
        raise ValueError("wsize must be an odd integer for volumetric CPU filtering.")
    _validate_wsize(wsize)

    video_np = np.asarray(video, dtype=np.float32)
    if video_np.ndim != 3:
        raise ValueError("video must have shape (frame, height, width).")

    frame_count, height, width = video_np.shape
    k = wsize // 2
    pad_video = np.pad(video_np, pad_width=k, mode=mode)
    patches = np.lib.stride_tricks.sliding_window_view(pad_video, (wsize, wsize, wsize))

    ax = np.arange(-k, k + 1, dtype=video_np.dtype)
    ff, hh, ww = np.meshgrid(ax, ax, ax, indexing="ij")
    spatial_kernel = np.exp(-(ff**2 + hh**2 + ww**2) / (2.0 * sigma_d * sigma_d))

    center = video_np.reshape(frame_count, height, width, 1, 1, 1)
    diff = patches - center
    r = np.exp(-(diff**2) / (2.0 * sigma_r * sigma_r))
    w = r * spatial_kernel

    w_sum = w.sum(axis=(-1, -2, -3))
    wp = (w * patches).sum(axis=(-1, -2, -3))
    return wp / (w_sum + 1e-8)


def estimate_chunk_size(F, H, W, wsize, dtype=np.float32, safety_factor=0.5):
    """Estimate a safe 3D chunk size from available host RAM."""
    try:
        import psutil

        available_mem = psutil.virtual_memory().available
    except Exception:
        available_mem = int(2 * 1024**3)

    bytes_per_element = np.dtype(dtype).itemsize
    mem_per_frame = H * W * (wsize**3) * bytes_per_element
    max_mem_for_chunk = available_mem * safety_factor
    max_frames = max(1, int(max_mem_for_chunk // mem_per_frame))
    return max_frames, available_mem / (1024**3)


def bilateral_filter_video_volumetric_chunked_halo(
    video,
    wsize,
    sigma_d,
    sigma_r,
    mode="edge",
    dtype=np.float32,
    backend="auto",
    safety_factor=0.5,
    overhead_factor=3.5,
    verbose=True,
):
    """3D bilateral filter with temporal halo chunking and NumPy/CuPy routing."""
    wsize_f, wsize_h, wsize_w = _normalize_volumetric_wsize(wsize)

    ndim = getattr(video, "ndim", None)
    if ndim is None:
        ndim = np.asarray(video).ndim
    if ndim != 3:
        raise ValueError("video must have shape (frame, height, width).")

    frame_count, height, width = video.shape
    kf = wsize_f // 2
    kh = wsize_h // 2
    kw = wsize_w // 2

    if backend == "cupy":
        if not CUPY_AVAILABLE:
            raise RuntimeError("backend='cupy' requested but CuPy is not available.")
        on_gpu = True
    elif backend == "numpy":
        on_gpu = False
    elif backend == "auto":
        on_gpu = CUPY_AVAILABLE
    else:
        raise ValueError("backend must be 'auto', 'numpy', or 'cupy'.")

    xp = cp if on_gpu else np

    def map_dtype(xp_mod, candidate):
        if isinstance(candidate, str):
            return getattr(xp_mod, candidate)
        try:
            return xp_mod.dtype(candidate)
        except Exception:
            return xp_mod.float32

    dtype_xp = map_dtype(xp, dtype)

    if on_gpu:
        video_xp = cp.asarray(video)
        if video_xp.dtype != dtype_xp:
            video_xp = video_xp.astype(dtype_xp, copy=False)
    else:
        video_xp = _cpu_array(video).astype(dtype, copy=False)

    if on_gpu:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        avail_bytes = int(free_bytes)
    else:
        try:
            import psutil

            avail_bytes = int(psutil.virtual_memory().available)
        except Exception:
            avail_bytes = int(2 * 1024**3)

    avail_gb = avail_bytes / (1024**3)
    bytes_per_element = xp.dtype(dtype_xp).itemsize
    if on_gpu:
        padded_frame_elements = (height + 2 * kh) * (width + 2 * kw)
        mem_per_frame = (
            padded_frame_elements
            + height * width
            + height * width
        ) * bytes_per_element
        mem_per_frame *= max(1.5, overhead_factor * 0.5)
    else:
        mem_per_frame = height * width * (wsize_f * wsize_h * wsize_w) * bytes_per_element
        mem_per_frame *= overhead_factor

    max_mem_for_chunk = max(1, int(avail_bytes * safety_factor))
    chunk_size = max(1, int(max_mem_for_chunk // mem_per_frame))
    chunk_size = min(chunk_size, frame_count)
    chunk_size = min(chunk_size, 32 if on_gpu else 128)

    if verbose:
        backend_name = "CuPy (GPU)" if on_gpu else "NumPy (CPU)"
        print(
            f"[bilateral 3D] Backend: {backend_name} | "
            f"Free mem: {avail_gb:.2f} GB | Chunk size: {chunk_size} frames | "
            f"Itemsize: {bytes_per_element} B | wsize^3: {wsize_f * wsize_h * wsize_w}"
        )

    ax_f = xp.arange(-kf, kf + 1, dtype=dtype_xp)
    ax_h = xp.arange(-kh, kh + 1, dtype=dtype_xp)
    ax_w = xp.arange(-kw, kw + 1, dtype=dtype_xp)
    ff, hh, ww = xp.meshgrid(ax_f, ax_h, ax_w, indexing="ij")
    spatial_kernel = xp.exp(
        -(ff**2 + hh**2 + ww**2)
        / (xp.array(2.0, dtype=dtype_xp) * (dtype_xp.type(sigma_d) ** 2))
    ).astype(dtype_xp, copy=False)

    out = xp.empty_like(video_xp, dtype=dtype_xp)
    if on_gpu:
        inv_2sr2 = cp.float32(1.0 / (2.0 * float(sigma_r) * float(sigma_r)))
        eps = cp.float32(1e-8)
    else:
        inv_2sr2 = np.float32(1.0 / (2.0 * float(sigma_r) * float(sigma_r)))
        eps = np.float32(1e-8)

    for start in range(0, frame_count, chunk_size):
        end = min(start + chunk_size, frame_count)
        t0 = max(0, start - kf)
        t1 = min(frame_count, end + kf)

        chunk_halo = video_xp[t0:t1]
        pad_width = ((kf, kf), (kh, kh), (kw, kw))
        pad_chunk = xp.pad(chunk_halo.astype(dtype_xp, copy=False), pad_width=pad_width, mode=mode)

        if on_gpu:
            chunk_frames = end - start
            _, padded_height, padded_width = pad_chunk.shape
            region_start = start - t0
            block = (8, 8, 4)
            grid = (
                (width + block[0] - 1) // block[0],
                (height + block[1] - 1) // block[1],
                (chunk_frames + block[2] - 1) // block[2],
            )
            spatial_flat = spatial_kernel.ravel()
            chunk_out = out[start:end]
            _bilateral_kernel_3d(
                grid,
                block,
                (
                    pad_chunk,
                    chunk_out,
                    spatial_flat,
                    chunk_frames,
                    height,
                    width,
                    padded_height,
                    padded_width,
                    region_start,
                    kf,
                    kh,
                    kw,
                    wsize_f,
                    wsize_h,
                    wsize_w,
                    inv_2sr2,
                    eps,
                ),
            )
        else:
            region_start = start - t0
            region_end = region_start + (end - start)
            patches = xp.lib.stride_tricks.sliding_window_view(pad_chunk, (wsize_f, wsize_h, wsize_w))
            centers = video_xp[start:end].reshape(end - start, height, width, 1, 1, 1)
            diff = patches[region_start:region_end] - centers
            r = xp.exp(-(diff**2) / (xp.array(2.0, dtype=dtype_xp) * (dtype_xp.type(sigma_r) ** 2)))
            w = r * spatial_kernel
            w_sum = w.sum(axis=(-1, -2, -3))
            wp = (w * patches[region_start:region_end]).sum(axis=(-1, -2, -3))
            out[start:end] = (wp / (w_sum + eps)).astype(dtype_xp, copy=False)

        if on_gpu:
            del chunk_halo, pad_chunk, chunk_out
            xp.get_default_memory_pool().free_all_blocks()
        else:
            del chunk_halo, pad_chunk, patches, centers, diff, r, w, w_sum, wp

    return out
