"""Utilities for computing Structural Similarity (SSIM) on videos."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity as ssim_cpu


def _is_gpu_available() -> bool:
    """Return True if a CUDA device is accessible via CuPy."""

    try:
        import cupy as cp  # type: ignore

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _ssim_gpu(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM using GPU acceleration via CuPy."""

    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import gaussian_filter

    arr1 = cp.asarray(img1, dtype=cp.float32)
    arr2 = cp.asarray(img2, dtype=cp.float32)

    # Dynamic range of the pixel values (assume both images share it)
    L = float(cp.maximum(cp.max(arr1), cp.max(arr2)))
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu1 = gaussian_filter(arr1, 1.5)
    mu2 = gaussian_filter(arr2, 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(arr1 * arr1, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(arr2 * arr2, 1.5) - mu2_sq
    sigma12 = gaussian_filter(arr1 * arr2, 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(ssim_map.mean())


def _ssim_cpu(img1: np.ndarray, img2: np.ndarray) -> float:
    data_range = float(np.max(img1) - np.min(img1))
    if data_range == 0:
        data_range = 1.0
    min_dim = min(img1.shape[-2:])
    if min_dim >= 7:
        win = 7
    else:
        win = min_dim if min_dim % 2 == 1 else max(3, min_dim - 1)
    return float(ssim_cpu(img1, img2, data_range=data_range, win_size=win))


def compute_ssim_segments(
    segments: np.ndarray,
    average_segment: np.ndarray,
    *,
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """Compute SSIM for each segment frame against ``average_segment``.

    Parameters
    ----------
    segments:
        Array of shape ``(segment, frame, x, y)``.
    average_segment:
        Array of shape ``(frame, x, y)``.
    use_gpu:
        Force GPU or CPU usage. ``None`` chooses automatically.

    Returns
    -------
    np.ndarray
        SSIM scores with shape ``(segment, frame)``.
    """

    if use_gpu is None:
        use_gpu = _is_gpu_available()

    n_seg, n_frame = segments.shape[0], segments.shape[1]
    out = np.empty((n_seg, n_frame), dtype=float)

    if use_gpu:
        try:
            for i in range(n_seg):
                for j in range(n_frame):
                    out[i, j] = _ssim_gpu(segments[i, j], average_segment[j])
        except Exception:
            # Fallback to CPU if GPU computation fails
            use_gpu = False
            print("GPU SSIM failed, falling back to CPU")

    if not use_gpu:
        def job(idx_pair):
            i, j = idx_pair
            return i, j, _ssim_cpu(segments[i, j], average_segment[j])

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
            futures = [exe.submit(job, (i, j)) for i in range(n_seg) for j in range(n_frame)]
            for fut in as_completed(futures):
                i, j, val = fut.result()
                out[i, j] = val

    return out
