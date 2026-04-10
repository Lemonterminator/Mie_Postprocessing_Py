"""Pipeline helpers for multi-hole spray preprocessing and penetration analysis.

This module sits at the boundary between low-level image processing and
domain-facing spray metrics. Most functions here are orchestration-heavy:
they combine filtering, masking, plume-wise geometric normalization, and
time-distance-map analysis into one sequence that older notebooks previously
implemented inline.

The long-term direction should be to split this file into smaller modules such
as ``preprocess`` and ``penetration``. For now, the functions below remain the
practical shared API for multi-hole experiments.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

import numpy as np

from OSCC_postprocessing.analysis.thresholding import (
    get_array_module,
    triangle_binarize,
)
from OSCC_postprocessing.binary_ops.functions_bw import (
    keep_largest_component,
    keep_largest_component_cuda,
    penetration_bw_to_index,
)
from OSCC_postprocessing.filters.video_filters import median_filter_video_auto
from OSCC_postprocessing.rotation.segment_ops import rotate_all_segments_auto
from OSCC_postprocessing.utils.backend import get_cupy


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
    """Preprocess a multi-hole spray video into normalized plume foreground.

    This function is the common front door for multi-plume intensity analysis.
    It performs the same sequence of steps that many older notebooks duplicated
    by hand:

    1. convert to the active array backend
    2. optional gamma remapping
    3. estimate a static background from pre-injection frames
    4. subtract that background from the full video
    5. apply a spatial median filter to suppress salt-and-pepper noise
    6. optionally derive a range mask from the per-pixel temporal percentile
       spread and keep only the active spray region
    7. normalize the final foreground to ``[0, 1]`` by its global maximum

    Parameters
    ----------
    video:
        Input video with shape ``(F, H, W)``.
    hydraulic_delay_estimate:
        Number of early frames assumed to contain only background. Those frames
        are collapsed with a temporal median to estimate the static background.
    gamma:
        Optional power-law remapping applied before background subtraction.
    M, N:
        Spatial median-filter window height and width.
    range_mask:
        When ``True``, derive a binary support mask from the 95th-5th temporal
        percentile range image.
    timing:
        Print elapsed runtime for quick notebook diagnostics.
    use_gpu:
        Prefer CuPy/CUDA execution where possible.
    triangle_backend:
        Backend hint for triangle thresholding when building the range mask.
    return_numpy:
        When ``True``, convert GPU results back to NumPy before returning.

    Returns
    -------
    foreground, mask:
        Background-subtracted and normalized intensity video plus the static
        binary support mask used to suppress inactive pixels.
    """
    cp = get_cupy() if use_gpu else None
    if use_gpu and cp is None:
        use_gpu = False
    if use_gpu and cp is not None:
        from cupyx.scipy.ndimage import median_filter as _cp_median_filter

    xp_backend = cp if use_gpu and cp is not None else np
    start = time.time() if timing else None
    arr = xp_backend.asarray(video)

    if gamma != 1:
        arr = arr.astype(xp_backend.float32, copy=False)
        arr = xp_backend.power(arr, gamma, dtype=arr.dtype)

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

        prefer_gpu = True if (use_gpu and triangle_backend != "cpu") else False
        mask = triangle_binarize(px_range, prefer_gpu=prefer_gpu)
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
    """Rotate the full video and the support mask into plume-aligned segments.

    Each injector hole is analysed in its own local coordinate system. This
    helper rotates the shared foreground video to each requested plume angle and
    applies the same transformation to the static range mask, producing one
    aligned sub-video per plume.
    """
    segments = rotate_all_segments_auto(foreground, angles, crop, centre, mask=region_mask)
    segments = xp.stack(segments, axis=0)
    segments[segments < 1e-3] = 0

    range_masks = rotate_all_segments_auto(
        px_range_mask[None, :, :], angles, crop, centre, mask=region_mask
    )
    range_masks = xp.stack(range_masks, axis=0).squeeze()
    return segments, range_masks


def compute_td_intensity_maps(segments, range_masks, use_gpu: bool):
    """Collapse rotated plume videos into time-distance intensity maps.

    The input ``segments`` is expected to have shape ``(P, F, H, W)`` after
    plume-wise rotation and cropping. Summing over the transverse image axis
    yields a time-distance map for each plume:

    ``td_maps[p, f, x] = sum_y segments[p, f, y, x]``.

    The maps are then normalized by the number of valid masked pixels in each
    axial column so plumes with slightly different visible widths remain
    comparable.
    """
    cp = get_cupy() if use_gpu else None
    if use_gpu and cp is None:
        use_gpu = False
    xp_backend = cp if use_gpu and cp is not None else np

    td_maps = xp_backend.sum(segments, axis=2)
    counts = range_masks.astype(xp_backend.uint8)
    counts = counts.sum(axis=1)
    counts[counts == 0] = 1

    counts = xp_backend.asarray(counts)
    td_maps = td_maps / counts[:, None, :]
    energies = xp_backend.sum(td_maps, axis=2)
    return td_maps, counts, energies


def estimate_peak_brightness_frames(energies, use_gpu: bool):
    """Locate the brightest frame of each plume from its TD-map energy trace.

    ``energies`` is typically the axial sum of the time-distance map. The
    resulting peak frames are used later to separate the growth regime from the
    post-peak decay regime when normalizing penetration traces.
    """
    cp = get_cupy() if use_gpu else None
    if use_gpu and cp is None:
        use_gpu = False
    xp_backend = cp if use_gpu and cp is not None else np

    peak_frames = xp_backend.argmax(energies, axis=1)
    if use_gpu:
        peak_host = cp.asnumpy(peak_frames)  # type: ignore[union-attr]
    else:
        peak_host = np.asarray(peak_frames)
    avg_peak = int(np.mean(peak_host))
    return peak_frames, avg_peak, peak_host


def estimate_hydraulic_delay_segments(segments, avg_peak: int, use_gpu: bool, width=1.0 / 7, height=0.1):
    """Estimate per-plume hydraulic delay from a near-nozzle intensity rise.

    The method integrates intensity in a small rectangular ROI close to the
    nozzle exit, computes a temporal derivative over the pre-peak region, and
    picks the first frame where that derivative exceeds a fixed threshold.

    Geometric convention
    --------------------
    The rotated segments are assumed to place the nozzle near the left edge,
    roughly at ``(row = H/2, col = 0)``. The ROI therefore spans:

    - a narrow band in the vertical direction around the image centre
    - an early axial interval near the left edge
    """
    if avg_peak == 0:
        xp_backend = get_array_module(segments)
        return xp_backend.zeros((segments.shape[0],), dtype=int)

    cp = get_cupy() if use_gpu else None
    if use_gpu and cp is None:
        use_gpu = False
    xp_backend = cp if use_gpu and cp is not None else np

    rows = segments.shape[2]
    cols = segments.shape[3]

    H_low = round(rows * (1 / width) // 2 * width)
    H_high = round(rows * ((1 / width) // 2 + 1) * width)
    W_right = round(cols * height)
    near_nozzle = xp_backend.sum(
        xp_backend.sum(segments[:, :avg_peak, H_low:H_high, :W_right], axis=3), axis=2
    )
    dE = xp_backend.diff(near_nozzle[:, 0:avg_peak], axis=1)
    hydraulic_delay = (dE > 1).argmax(axis=1)
    if use_gpu and cp is not None:
        return cp.asarray(hydraulic_delay)
    return np.asarray(hydraulic_delay)


def _precompute_td_otsu_masks(td_intensity_maps, use_gpu: bool):
    """Precompute plume-wise Otsu masks for GPU penetration estimation.

    The helper converts each TD map to ``uint8`` independently so the Otsu
    threshold is computed on a stable intensity range per plume. The output is
    cached because later stages may need those masks repeatedly while staying on
    the GPU.
    """
    if not use_gpu:
        return None
    cp = get_cupy()
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
    """Estimate one penetration curve per plume from TD intensity maps.

    The returned penetration value at each frame is a column-wise edge location
    measured in the plume-aligned TD map. Conceptually the function mixes two
    complementary edge detectors:

    - a triangle-threshold mask followed by largest-component cleanup, which is
      robust when the plume body is already bright and connected
    - a positive temporal-differential cue between hydraulic delay and peak
      brightness, which helps recover the advancing front during early growth

    Processing outline
    ------------------
    1. Normalize each plume's post-peak decay segment so late-time intensity
       drop does not bias the thresholding stage.
    2. Build a binary plume support mask using triangle thresholding.
    3. Convert that support mask to an axial edge position by scanning from the
       downstream side back toward the nozzle.
    4. In the early-growth window, fuse the threshold-based edge with the edge
       inferred from positive temporal derivatives.
    5. Return a ``(P, F)`` array of penetration positions.

    Notes
    -----
    The function updates ``td_intensity_maps`` in place during post-peak
    normalization. Callers that need the original values should pass a copy.
    """
    P, F, _ = td_intensity_maps.shape
    cp = get_cupy() if use_gpu else None
    if use_gpu and cp is None:
        use_gpu = False
    xp_backend = cp if use_gpu and cp is not None else np

    penetration = xp_backend.full((P, F), xp_backend.nan, dtype=xp_backend.float32)
    bw_otsu_all = _precompute_td_otsu_masks(td_intensity_maps, use_gpu)

    def _process_one_plume(p: int):
        """Process one plume independently for optional thread-level parallelism."""
        pb = int(peak_brightness_frames_host[p])
        decay_curve = energies[p, pb:]
        if decay_curve.size == 0:
            return p, xp_backend.full(F, xp_backend.nan, dtype=xp_backend.float32)
        if use_gpu:
            decay_curve = decay_curve / xp_backend.max(decay_curve)
            td_intensity_maps[p, pb:, :] = td_intensity_maps[p, pb:, :] / decay_curve[:, None]
            arr = td_intensity_maps[p, :, :].T
            bw = triangle_binarize(arr, prefer_gpu=True)
            bw = keep_largest_component_cuda(bw, connectivity=2)
            edge_tri = xp_backend.argmax(bw[::-1, :], axis=0)
            edge_tri = bw.shape[0] - edge_tri
        else:
            decay_curve = decay_curve / xp_backend.max(decay_curve)
            td_intensity_maps[p, pb:, :] = td_intensity_maps[p, pb:, :] / decay_curve[:, None]
            arr = td_intensity_maps[p, :, :].T
            bw = triangle_binarize(arr, prefer_gpu=False)
            bw = keep_largest_component(bw > 0, connectivity=2)
            edge_tri = xp_backend.argmax(bw[::-1, :], axis=0)
            edge_tri = bw.shape[0] - edge_tri

        hd = int(hydraulic_delay[p])
        leading = max(0, hd + 5)
        edge_tri[:leading][edge_tri[:leading] == bw.shape[0]] = 0
        row = xp_backend.array(edge_tri, dtype=xp_backend.float32)

        if use_gpu and bw_otsu_all is not None:
            bw_otsu = bw_otsu_all[p]
            masked = arr[:, hd + 1 : pb] * bw_otsu[:, hd + 1 : pb]
            differential = xp_backend.diff(masked, axis=1)
            differential = xp_backend.maximum(differential, 0)
            edge_diff = xp_backend.argmax(differential[::-1, :], axis=0)
            edge_diff = differential.shape[0] - edge_diff
        else:
            import cv2

            arr_xp = arr[:, hd + 1 : pb]
            arr_host = cp.asnumpy(arr_xp) if use_gpu and cp is not None else np.asarray(arr_xp)
            arr_u8 = cv2.normalize(
                arr_host,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            _, bw_otsu = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # type: ignore
            differential = np.diff(arr_host, axis=1)
            differential = differential * (bw_otsu[:, 1:] > 0)
            differential[differential < 0] = 0
            edge_diff = np.argmax(differential[::-1, :], axis=0)
            edge_diff = differential.shape[0] - edge_diff
            edge_diff = xp_backend.asarray(edge_diff)

        edge_diff[edge_diff > upper - 10] = 0
        edge_diff[edge_diff < lower + 10] = 0
        start = int(hydraulic_delay[p] + 1)
        end = min(int(pb), row.shape[0])
        decision = row[start : end - 1]
        if edge_diff.size > 0:
            decision = xp_backend.maximum(decision, edge_diff[: len(decision)])
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
    """Apply repository-standard cleanup rules to penetration curves.

    The cleanup enforces a few domain-specific assumptions:

    - penetration should be monotonically non-decreasing once the spray starts
    - zeros outside the hydraulic-delay anchor are treated as missing values
    - values close to the far downstream crop boundary are likely artifacts
    - extremely early large values are also treated as invalid
    """
    xp_backend = get_array_module(penetration)
    xp_backend.maximum.accumulate(penetration, axis=1, out=penetration)
    penetration[penetration == 0] = xp_backend.nan
    for p in range(penetration.shape[0]):
        hd = int(hydraulic_delay[p])
        if 0 <= hd < penetration.shape[1]:
            penetration[p, hd] = 0.0
    penetration[penetration > upper - 2] = xp_backend.nan
    half = penetration.shape[1] // 2
    bad = penetration[:, :half] > upper - 10
    penetration[:, :half][bad] = xp_backend.nan
    return penetration


def binarize_plume_videos(segments, hydraulic_delay):
    """Binarize each plume video frame and recover legacy penetration indices.

    This helper exists mainly for compatibility with older workflows that were
    written around binary plume videos rather than the newer TD-map-based
    penetration estimator. For each plume/frame pair after hydraulic delay it:

    1. applies triangle thresholding
    2. keeps the largest connected component
    3. stores the binary frame in ``bw_vids``
    4. derives a penetration index from the downstream-most occupied column
    """
    cp = get_cupy()
    is_cupy = cp is not None and hasattr(segments, "__cuda_array_interface__")
    xp_backend = cp if is_cupy else np

    P, F, H, W = segments.shape
    bw_vids = xp_backend.zeros((P, F, H, W), dtype=xp_backend.uint8)
    hd_host = xp_backend.asarray(cp.asnumpy(hydraulic_delay) if is_cupy else hydraulic_delay).astype(int)

    def _process_one(i, j):
        frame_xp = segments[i, j]
        bw_xp = triangle_binarize(frame_xp, prefer_gpu=is_cupy)
        if is_cupy:
            largest_cp = keep_largest_component_cuda(bw_xp)
            return i, j, largest_cp
        largest_xp = keep_largest_component(bw_xp)
        return i, j, largest_xp

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

    col_sum_bw_host = xp_backend.sum(cp.asnumpy(bw_vids) if is_cupy else bw_vids, axis=2) >= 1
    penetration_old_host = xp_backend.zeros((P, F), dtype=int)
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


__all__ = [
    "binarize_plume_videos",
    "clean_penetration_profiles",
    "compute_penetration_profiles",
    "compute_td_intensity_maps",
    "estimate_hydraulic_delay_segments",
    "estimate_peak_brightness_frames",
    "preprocess_multihole",
    "rotate_segments_with_masks",
]
