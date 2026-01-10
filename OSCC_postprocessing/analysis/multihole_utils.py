"""Utilities for the multi-hole Mie processing pipelines.

This module centralizes the heavy helper logic from ``mie_multihole_pipeline``
so that it can be re-used across scripts and stays decoupled from plotting or
CLI glue.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from typing import Any, Tuple

import numpy as np

from OSCC_postprocessing.filters.video_filters import median_filter_video_auto
from OSCC_postprocessing.binary_ops.functions_bw import (
    triangle_binarize_from_float,
    keep_largest_component,
    keep_largest_component_cuda,
    penetration_bw_to_index,
)
from OSCC_postprocessing.rotation.rotate_crop import rotate_all_segments_auto


def _get_cupy():
    """Return the CuPy module if available, otherwise ``None``."""
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:
        return None


def has_cupy_gpu() -> Tuple[bool, str]:
    """Return ``(available, info)`` for the current machine."""
    cp = _get_cupy()
    if cp is None:
        return False, "CuPy import failed"
    try:
        ndev = cp.cuda.runtime.getDeviceCount()
        if ndev <= 0:
            return False, "No CUDA/HIP devices found"
        with cp.cuda.Device(0):
            _ = cp.asarray([0]).sum()
    except Exception as exc:  # pragma: no cover - hardware dependent
        return False, f"CuPy GPU unavailable: {exc}"
    return True, f"CuPy ready on {ndev} device(s)"


def resolve_backend(
    use_gpu: str | bool = "auto",
    triangle_backend: str = "auto",
) -> Tuple[bool, str, Any]:
    """
    Decide the preferred backend and triangle threshold implementation.

    Returns ``(use_gpu, triangle_backend, xp)`` where ``xp`` is ``np`` or
    ``cupy`` depending on availability.
    """
    cp = _get_cupy()
    gpu_ok = cp is not None
    if use_gpu == "auto":
        resolved_gpu = gpu_ok
    else:
        resolved_gpu = bool(use_gpu and gpu_ok)

    if triangle_backend == "auto":
        resolved_triangle = "gpu" if resolved_gpu else "cpu"
    elif triangle_backend == "gpu" and not resolved_gpu:
        resolved_triangle = "cpu"
    else:
        resolved_triangle = triangle_backend

    xp = np
    if resolved_gpu and cp is not None:
        # Ensure the GPU median filter is available
        try:
            from cupyx.scipy.ndimage import median_filter as _check  # noqa: F401
        except Exception:
            resolved_gpu = False
            resolved_triangle = "cpu"
        else:
            xp = cp

    return resolved_gpu, resolved_triangle, xp


def triangle_binarize_gpu(px_range_cp, ignore_zeros: bool = False):
    """Histogram-based triangle threshold implemented fully on the GPU."""
    cp = _get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is required for triangle_binarize_gpu")
    x = px_range_cp
    if ignore_zeros:
        posmask = x > 0
        nz = x[posmask]
    else:
        nz = x.ravel()

    if nz.size == 0:
        return cp.zeros_like(x, dtype=cp.bool_)

    vmin = nz.min()
    vmax = nz.max()
    scale = 255.0 / (float(vmax - vmin) + 1e-12)
    u8 = cp.floor((nz - vmin) * scale).astype(cp.uint8, copy=False)

    hist, _ = cp.histogram(u8, bins=256, range=(0, 255))
    nzbins = cp.nonzero(hist)[0]
    i0, i1 = int(nzbins[0]), int(nzbins[-1])
    imax = int(hist.argmax())

    iend = i0 if (imax - i0) > (i1 - imax) else i1
    lo, hi = (iend, imax) if iend < imax else (imax, iend)
    xs = cp.arange(lo, hi + 1)
    ys = hist[xs].astype(cp.float32)
    x0, y0 = float(imax), float(hist[imax])
    x1, y1 = float(iend), float(hist[iend])
    denom = np.hypot(y1 - y0, x1 - x0) + 1e-12
    d = cp.abs((y0 - y1) * xs + (x1 - x0) * ys + (x0 * y1 - x1 * y0)) / denom
    t_idx = int(xs[int(d.argmax())])

    if ignore_zeros:
        mask_full = cp.zeros_like(x, dtype=cp.bool_)
        mask_full[posmask] = u8 > t_idx
        return mask_full
    return (u8 > t_idx).reshape(x.shape)


def preprocess_multihole(
    video,
    hydraulic_delay_estimate: int,
    *,
    gamma: float = 1.0,
    M: int = 3,
    N: int = 3,
    range_mask: bool = True,
    timing: bool = False,
    use_gpu: bool = False,
    triangle_backend: str = "gpu",
    return_numpy: bool = True,
):
    """
    Shared preprocessing routine for multi-hole Mie videos.

    Returns ``(foreground, mask)`` either as NumPy arrays or CuPy arrays.
    """
    xp = np
    cp = None
    if use_gpu:
        cp = _get_cupy()
        if cp is None:
            use_gpu = False
        else:
            from cupyx.scipy.ndimage import median_filter as _cp_median_filter

    start = time.time() if timing else None
    arr = xp.asarray(video) if not use_gpu else cp.asarray(video)  # type: ignore

    if gamma != 1:
        arr = arr.astype((xp if not use_gpu else cp).float32, copy=False)  # type: ignore
        arr = (xp if not use_gpu else cp).power(arr, gamma, dtype=arr.dtype)  # type: ignore

    xp_backend = xp if not use_gpu else cp  # type: ignore
    bkg = xp_backend.median(arr[:hydraulic_delay_estimate], axis=0, keepdims=True).astype(arr.dtype, copy=False)

    sub_bkg = xp_backend.empty_like(arr)
    xp_backend.subtract(arr, bkg, out=sub_bkg)

    if use_gpu and cp is not None:
        sub_bkg_med = _cp_median_filter(sub_bkg, size=(1, M, N))
        if sub_bkg_med is None:
            sub_bkg_med = cp.zeros_like(sub_bkg)
        else:
            cp.maximum(sub_bkg_med, 0, out=sub_bkg_med)
    else:
        sub_bkg_med = median_filter_video_auto(sub_bkg, M, N)
        sub_bkg_med[sub_bkg_med < 0] = 0

    if range_mask:
        q_hi = xp_backend.percentile(sub_bkg_med, 95, axis=0)
        q_lo = xp_backend.percentile(sub_bkg_med, 5, axis=0)
        px_range = q_hi - q_lo

        if use_gpu and triangle_backend == "gpu":
            mask = triangle_binarize_gpu(px_range)  # type: ignore[arg-type]
        else:
            import cv2

            px_cpu = cp.asnumpy(px_range) if use_gpu and cp is not None else px_range
            u8 = cv2.normalize(px_cpu, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, mask_u8 = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            mask = mask_u8.astype(bool)
            if use_gpu and cp is not None:
                mask = cp.asarray(mask)
        foreground = sub_bkg_med * mask[None, ...]
    else:
        foreground = sub_bkg_med

    scale = xp_backend.max(foreground)
    if (scale > 0) if not use_gpu else bool(scale.get() > 0):
        foreground = foreground / scale

    if timing and start is not None:
        print(f"Preprocessing completed in {time.time() - start:.3f}s (use_gpu={use_gpu})")

    if not range_mask:
        mask = xp_backend.ones_like(foreground[0], dtype=bool)

    if return_numpy and use_gpu and cp is not None:
        return cp.asnumpy(foreground), cp.asnumpy(mask)
    if return_numpy:
        return foreground, mask
    return foreground, mask


def rotate_segments_with_masks(
    foreground,
    px_range_mask,
    angles,
    crop,
    centre,
    *,
    region_mask,
    xp,
):
    """Rotate the video and range masks for every plume angle."""
    segments = rotate_all_segments_auto(
        foreground, angles, crop, centre, mask=region_mask
    )
    segments = xp.stack(segments, axis=0)
    segments[segments < 1e-3] = 0

    range_masks = rotate_all_segments_auto(
        px_range_mask[None, :, :], angles, crop, centre, mask=region_mask
    )
    range_masks = xp.stack(range_masks, axis=0).squeeze()
    return segments, range_masks


def compute_td_intensity_maps(segments, range_masks, use_gpu: bool):
    """Return TD maps, per-plume mask counts, and per-frame energies."""
    xp = _get_cupy() if use_gpu else np
    if use_gpu and xp is None:
        xp = np
        use_gpu = False

    td_maps = xp.sum(segments, axis=2)
    counts = range_masks.astype(np.uint8) if not use_gpu else range_masks.astype(xp.uint8)  # type: ignore[attr-defined]
    counts = counts.sum(axis=1)
    counts[counts == 0] = 1
    # if use_gpu:
        # td_maps = td_maps.get()

    counts = xp.asarray(counts)
    td_maps = td_maps / counts[:, None, :]
    energies = xp.sum(td_maps, axis=2)
    return td_maps, counts, energies


def estimate_peak_brightness_frames(energies, use_gpu: bool):
    """Find the peak brightness frame per plume plus host copies."""
    xp = _get_cupy() if use_gpu else np
    if use_gpu and xp is None:
        xp = np
        use_gpu = False

    peak_frames = xp.argmax(energies, axis=1)
    if use_gpu:
        peak_host = np.asarray(peak_frames.get())
    else:
        peak_host = np.asarray(peak_frames)
    avg_peak = int(np.mean(peak_host))
    return peak_frames, avg_peak, peak_host


def estimate_hydraulic_delay_segments(segments, avg_peak: int, use_gpu: bool, width=1.0/7, height = 0.1):
    """
    Hydraulic delay from a near-nozzle ROI derivative.
    Nozzle is assumed to be at position (H//2, 0)
    """
    if avg_peak  == 0:
        return np.zeros((segments.shape[0],), dtype=int)
    
    xp = _get_cupy() if use_gpu else np
    if use_gpu and xp is None:
        xp = np
        use_gpu = False

    rows = segments.shape[2]
    cols = segments.shape[3]
    
    H_low = round(rows * (1/width)//2 *width)
    H_high = round(rows * ((1/width)//2 +1 ) *width)
    W_right = round(cols *height)
    near_nozzle = xp.sum(
        xp.sum(segments[:, :avg_peak, H_low:H_high, :W_right], axis=3), axis=2
    )
    dE = xp.diff(near_nozzle[:, 0:avg_peak], axis=1)
    hydraulic_delay = (dE > 1).argmax(axis=1)
    return np.asarray(hydraulic_delay.get() if use_gpu else hydraulic_delay)


def precompute_td_otsu_masks(td_intensity_maps, use_gpu: bool):
    """GPU-only helper that precomputes Otsu masks for TD maps."""
    if not use_gpu:
        return None
    cp = _get_cupy()
    if cp is None:
        return None

    arrs = cp.transpose(td_intensity_maps, (0, 2, 1))
    if arrs.size == 0:
        return None
    P, X, F = arrs.shape
    a_min = cp.min(arrs, axis=(1, 2))
    a_max = cp.max(arrs, axis=(1, 2))
    denom = cp.maximum(a_max - a_min, 1e-6)
    scale = (255.0 / denom)[:, None, None]
    u8 = cp.clip((arrs - a_min[:, None, None]) * scale, 0, 255).astype(cp.uint8)

    bw_otsu_all = cp.zeros((P, X, F), dtype=bool)
    bins = cp.arange(256, dtype=cp.float32)
    for pp in range(P):
        hist = cp.histogram(u8[pp], bins=256, range=(0, 256))[0].astype(cp.float32)
        w1 = cp.cumsum(hist)
        w2 = w1[-1] - w1
        m1 = cp.cumsum(hist * bins)
        m2_total = m1[-1]
        m2 = m2_total - m1
        mu1 = cp.where(w1 > 0, m1 / w1, 0)
        mu2 = cp.where(w2 > 0, m2 / w2, 0)
        sigma_b2 = w1 * w2 * (mu1 - mu2) ** 2
        t = int(cp.argmax(sigma_b2))
        bw_otsu_all[pp] = u8[pp] >= t
    return bw_otsu_all


def compute_penetration_profiles(
    td_intensity_maps,
    energies,
    hydraulic_delay,
    peak_brightness_frames_host,
    *,
    use_gpu: bool,
    lower: int = 0,
    upper: int = 366,
):
    """
    Column-wise penetration edge detection for every plume.

    Returns a ``(P, F)`` NumPy array.
    """
    P, F, _ = td_intensity_maps.shape
    penetration = np.full((P, F), np.nan, dtype=np.float32)
    bw_otsu_all = precompute_td_otsu_masks(td_intensity_maps, use_gpu)
    xp = _get_cupy() if use_gpu else np
    if use_gpu and xp is None:
        xp = np
        use_gpu = False

    def _process_one_plume(p: int):
        pb = int(peak_brightness_frames_host[p])
        decay_curve = energies[p, pb:]
        if decay_curve.size == 0:
            return p, np.full(F, np.nan, dtype=np.float32)
        if use_gpu:
            decay_curve = decay_curve / xp.max(decay_curve)
            td_intensity_maps[p, pb:, :] = td_intensity_maps[p, pb:, :] / decay_curve[:, None]
            arr = td_intensity_maps[p, :, :].T
            bw = triangle_binarize_gpu(arr)
            bw = keep_largest_component_cuda(bw, connectivity=2)
            edge_tri_cp = xp.argmax(bw[::-1, :], axis=0)
            edge_tri_cp = bw.shape[0] - edge_tri_cp
            edge_tri = xp.asnumpy(edge_tri_cp)
        else:
            decay_curve = decay_curve / np.max(decay_curve)
            td_intensity_maps[p, pb:, :] = td_intensity_maps[p, pb:, :] / decay_curve[:, None]
            arr = td_intensity_maps[p, :, :].T
            arr_np = np.asarray(arr)
            bw_u8, _ = triangle_binarize_from_float(arr_np)
            bw = keep_largest_component(bw_u8 > 0, connectivity=2)
            edge_tri = np.argmax(bw[::-1, :], axis=0)
            edge_tri = bw.shape[0] - edge_tri

        hd = int(hydraulic_delay[p])
        leading = max(0, hd + 5)
        edge_tri[:leading][edge_tri[:leading] == bw.shape[0]] = 0
        row = np.array(edge_tri, dtype=np.float32)

        if use_gpu and bw_otsu_all is not None:
            bw_otsu = bw_otsu_all[p]
            masked = arr[:, hd + 1 : pb] * bw_otsu[:, hd + 1 : pb]
            differential = xp.diff(masked, axis=1)
            differential = xp.maximum(differential, 0)
            edge_diff_cp = xp.argmax(differential[::-1, :], axis=0)
            edge_diff_cp = differential.shape[0] - edge_diff_cp
            edge_diff = xp.asnumpy(edge_diff_cp)
        else:
            import cv2

            arr_np = arr[:, hd + 1 : pb]
            arr_u8 = cv2.normalize(
                arr_np.get() if use_gpu else np.asarray(arr_np),
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            _, bw_otsu = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # type: ignore
            differential = np.diff(arr_np.get() if use_gpu else np.asarray(arr_np), axis=1)
            differential = differential * (bw_otsu[:, 1:] > 0)
            differential[differential < 0] = 0
            edge_diff = np.argmax(differential[::-1, :], axis=0)
            edge_diff = differential.shape[0] - edge_diff

        edge_diff[edge_diff > upper - 10] = 0
        edge_diff[edge_diff < lower + 10] = 0
        start = int(hydraulic_delay[p] + 1)
        end = min(int(pb), row.shape[0])
        decision = row[start : end - 1]
        if edge_diff.size > 0:
            decision = np.maximum(decision, edge_diff[: len(decision)])
        row[start : end - 1] = decision
        return p, row

    n_workers = 1 if use_gpu else min(os.cpu_count() or 1, P, 32)
    if n_workers == 1:
        for p in range(P):
            idx, row = _process_one_plume(p)
            penetration[idx] = row
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_one_plume, p) for p in range(P)]
            for fut in as_completed(futures):
                idx, row = fut.result()
                penetration[idx] = row

    return penetration


def clean_penetration_profiles(penetration, hydraulic_delay, upper: int, lower: int = 0):
    """Apply the standard monotonic/threshold cleanup steps."""
    np.maximum.accumulate(penetration, axis=1, out=penetration)
    penetration[penetration == 0] = np.nan
    for p in range(penetration.shape[0]):
        hd = int(hydraulic_delay[p])
        if 0 <= hd < penetration.shape[1]:
            penetration[p, hd] = 0.0
    penetration[penetration > upper - 2] = np.nan
    half = penetration.shape[1] // 2
    bad = penetration[:, :half] > upper - 10
    penetration[:, :half][bad] = np.nan
    return penetration


def binarize_plume_videos(segments, hydraulic_delay):
    """Wrapper around the per-frame binarization and boundary extraction."""
    cp = _get_cupy()
    is_cupy = cp is not None and hasattr(segments, "__cuda_array_interface__")
    xp = cp if is_cupy else np

    P, F, H, W = segments.shape
    bw_vids = xp.zeros((P, F, H, W), dtype=xp.uint8)
    hd_host = np.asarray(cp.asnumpy(hydraulic_delay) if is_cupy else hydraulic_delay).astype(int)

    def _process_one(i, j):
        frame_np = cp.asnumpy(segments[i, j]) if is_cupy else segments[i, j]
        bw_np, _ = triangle_binarize_from_float(frame_np)
        if is_cupy:
            bw_cp = cp.asarray(bw_np)
            largest_cp = keep_largest_component_cuda(bw_cp)
            return i, j, largest_cp
        largest_np = keep_largest_component(bw_np)
        return i, j, largest_np

    max_workers = min(32, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i in range(P):
            if len(hd_host.shape) > 0:
                j0 = max(int(hd_host[i]), 0)
            else:
                j0 = max(int(hd_host), 0)
            for j in range(j0, F):
                futs.append(ex.submit(_process_one, i, j))
        for fut in as_completed(futs):
            i, j, bw = fut.result()
            bw_vids[i, j] = bw

    col_sum_bw_host = np.sum(cp.asnumpy(bw_vids) if is_cupy else bw_vids, axis=2) >= 1
    penetration_old_host = np.zeros((P, F), dtype=int)
    n_workers = min(os.cpu_count() or 1, P, 32)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(penetration_bw_to_index, col_sum_bw_host[p]): p for p in range(P)
        }
        for fut in as_completed(futs):
            penetration_old_host[futs[fut]] = fut.result()

    if is_cupy:
        return bw_vids, cp.asarray(penetration_old_host)
    return bw_vids, penetration_old_host


def compute_cone_angle_from_angular_density(signal, offset, number_of_plumes, *, bins=3600, use_gpu=False):
    """Shared GPU/CPU cone-angle computation."""
    shift_bins = int(offset / 360 * bins)
    try:
        if use_gpu:
            cp = _get_cupy()
            if cp is None:
                raise RuntimeError
            from cupyx.scipy.ndimage import binary_closing as cp_binary_closing  # type: ignore

            sig_cp = cp.asarray(signal, dtype=cp.float32)
            bw_cp = triangle_binarize_gpu(sig_cp)
            bw_cp = cp.roll(bw_cp, -shift_bins, axis=1)
            struct_cp = cp.ones((1, 3), dtype=cp.bool_)
            bw_closed = cp_binary_closing(bw_cp, structure=struct_cp)

            cone_angle = np.zeros((number_of_plumes, bw_closed.shape[0]), dtype=np.float32)
            deg_per_bin = 360.0 / bins
            for p in range(number_of_plumes):
                start = int(round(p * bins / number_of_plumes))
                end = int(round((p + 1) * bins / number_of_plumes))
                s = bw_closed[:, start:end].sum(axis=1) * deg_per_bin
                cone_angle[p] = cp.asnumpy(s)
            return cone_angle
        raise RuntimeError
    except Exception:
        from scipy.ndimage import binary_closing

        bw_u8, _ = triangle_binarize_from_float(signal, blur=True)
        bw_shifted = np.roll(bw_u8 > 0, -shift_bins, axis=1)
        struct = np.ones((1, 3), dtype=bool)
        bw_closed = binary_closing(bw_shifted, structure=struct)

        cone_angle = np.zeros((number_of_plumes, bw_closed.shape[0]), dtype=np.float32)
        deg_per_bin = 360.0 / bins
        for p in range(number_of_plumes):
            start = int(round(p * bins / number_of_plumes))
            end = int(round((p + 1) * bins / number_of_plumes))
            cone_angle[p] = bw_closed[:, start:end].sum(axis=1) * deg_per_bin
        return cone_angle


def estimate_offset_from_fft(signal, number_of_plumes: int):
    """Return the spray axis offset estimated from the FFT of the angular signal."""
    summed_signal = signal.sum(axis=0)
    fft_vals = np.fft.rfft(summed_signal)
    if number_of_plumes >= len(fft_vals):
        return 0.0
    phase = np.angle(fft_vals[number_of_plumes])
    offset = (-phase / number_of_plumes) * 180.0 / np.pi
    offset %= 360.0
    offset = min(offset, offset - 360, key=abs)
    return offset

