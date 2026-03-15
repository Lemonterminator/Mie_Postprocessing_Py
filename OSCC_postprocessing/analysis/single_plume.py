from OSCC_postprocessing.analysis.multihole_utils import *
from OSCC_postprocessing.binary_ops.functions_bw import *
from OSCC_postprocessing.filters.video_filters import sobel_5x5_kernels, filter_video_fft
from OSCC_postprocessing.filters.svd_background_removal import godec_like
from OSCC_postprocessing.analysis.cone_angle import angle_signal_density_auto
from OSCC_postprocessing.filters.bilateral_filter_rawKernel import (
    bilateral_filter_video_cpu,
    bilateral_filter_video_cupy,
    bilateral_filter_video_volumetric_chunked_halo,
)
from OSCC_postprocessing.io.async_avi_saver import AsyncAVISaver
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import binary_fill_holes
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from OSCC_postprocessing.binary_ops.functions_bw import _triangle_threshold_from_hist, _boundary_points_one_frame
from OSCC_postprocessing.analysis.multihole_utils import triangle_binarize_gpu as _triangle_binarize_gpu
from OSCC_postprocessing.analysis.nozzle import estimate_nozzle_opening_duration
from OSCC_postprocessing.analysis.regression import (
    linear_regression_fixed_intercept,
    ransac_fixed_intercept,
    ransac_line_1d,
)
from OSCC_postprocessing.analysis.video_utils import (
    prepare_temporal_smoothing as _prepare_temporal_smoothing,
    rotate_align_video_cpu as _rotate_align_video_cpu,
)
from OSCC_postprocessing.binary_ops.gpu_helpers import (
    binary_fill_holes_cpu as _binary_fill_holes_cpu,
    binary_fill_holes_gpu as _binary_fill_holes_gpu,
    fallback_fill_holes as _fallback_fill_holes,
    penetration_gpu as _penetration_gpu,
    regionprops_gpu,
)
from OSCC_postprocessing.utils.backend import USING_CUPY, cp, get_cupy as _get_cupy, to_numpy
from OSCC_postprocessing.utils.scaling import min_max_scale as _min_max_scale, robust_scale

def binarize_single_plume_video(
    video,
    hydraulic_delay,
    lighting_unchanged_duration=100,
    prefer_gpu=True,
    return_gpu_mask=False,
    hole_fill_mode="3D",
):
    """
    Binarize a single plume video (F, H, W) and compute penetration.

    Pipeline
    --------
    - Triangle-threshold each frame (GPU if available and `prefer_gpu`; otherwise CPU + threads).
    - Stabilize threshold using the first `lighting_unchanged_duration` frames (excluding `hydraulic_delay` frames).
    - Re-binarize that stable window with the fitted threshold.
    - Hand off to CPU for connected components: keep largest blob, fill holes, compute penetration along x.

    Parameters
    ----------
    video : np.ndarray or cupy.ndarray
        Input video (F, H, W). If NumPy and `prefer_gpu=True`, frames are copied to GPU for thresholding.
    hydraulic_delay : int or scalar-like
        First frame index to start processing (frames < hydraulic_delay are skipped).
    lighting_unchanged_duration : int, optional
        Number of initial frames assumed to have stable lighting for threshold fitting.
    prefer_gpu : bool, optional
        If True and CuPy is available, run the thresholding stage on GPU; otherwise stay on CPU.
    return_gpu_mask : bool, optional
        If True and GPU is used, convert outputs back to CuPy; otherwise return NumPy arrays.
    hole_fill_mode : {"2D","3D"}, optional
        "3D" fills holes across the full (F, H, W) volume; "2D" fills each frame independently.

    Returns
    -------
    largest_blob_mask : np.ndarray or cupy.ndarray
        Binary mask of the largest component after hole filling (CPU stage).
    penetration : np.ndarray or cupy.ndarray
        Penetration indices per frame.
    """
    cp = _get_cupy()
    gpu_enabled = bool(prefer_gpu and cp is not None)

    def _normalize_int(val):
        arr = np.asarray(
            cp.asnumpy(val) if gpu_enabled and hasattr(val, "__cuda_array_interface__") else val
        ).astype(int)
        return int(arr.ravel()[0]) if arr.ndim > 0 else int(arr)

    # Normalize hydraulic_delay to a plain Python int
    hd_host = _normalize_int(hydraulic_delay)

    if gpu_enabled:
        video_gpu = video if hasattr(video, "__cuda_array_interface__") else cp.asarray(video)
        F, H, W = video_gpu.shape
        j0 = max(hd_host, 0)
        stable_end = min(lighting_unchanged_duration, F)

        bw_gpu = cp.zeros((F, H, W), dtype=cp.uint8)
        thres_gpu = cp.zeros(F, dtype=cp.float32)

        def _triangular_binarize_gpu(frame_cp):
            frame_f = cp.asarray(frame_cp, dtype=cp.float32)
            f_min = frame_f.min()
            f_max = frame_f.max()
            if float(f_max - f_min) <= 0:
                return cp.zeros_like(frame_f, dtype=cp.uint8), 0.0

            norm = (frame_f - f_min) / (f_max - f_min)
            u8 = cp.clip(norm * 255.0, 0, 255).astype(cp.uint8)

            hist = cp.histogram(u8, bins=256, range=(0, 255))[0]
            t = _triangle_threshold_from_hist(cp.asnumpy(hist))
            binarized = (u8 >= t).astype(cp.uint8) * 255
            return binarized, float(t)

        # First-pass triangle thresholding on GPU
        for j in range(j0, F):
            bw_frame, th = _triangular_binarize_gpu(video_gpu[j])
            bw_gpu[j] = bw_frame
            thres_gpu[j] = th

        # Stabilize threshold over the lighting-stable window
        if stable_end > 0 and stable_end > hd_host:
            to_stabilize = thres_gpu[:stable_end]
            if hd_host > 0:
                to_stabilize[: min(hd_host, stable_end)] = cp.nan
            fitted_threshold = float(cp.nanmedian(to_stabilize) / 255.0)
            rebinarized = (video_gpu[hd_host:stable_end] >= fitted_threshold).astype(cp.uint8) * 255
            bw_gpu[hd_host:stable_end] = rebinarized

        bw_final_gpu = bw_gpu

    else:
        # CPU path (also used when CuPy is unavailable)
        video_cpu = (
            cp.asnumpy(video) if cp is not None and hasattr(video, "__cuda_array_interface__") else np.asarray(video)
        )
        F, H, W = video_cpu.shape
        j0 = max(hd_host, 0)
        stable_end = min(lighting_unchanged_duration, F)

        bw_cpu = np.zeros((F, H, W), dtype=np.uint8)
        thres_array = np.zeros(F, dtype=float)

        def _tri_bw_one_frame(j):
            bw_np, thres = triangle_binarize_from_float(video_cpu[j])
            return j, bw_np, thres

        max_workers = min(32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_tri_bw_one_frame, j) for j in range(j0, F)]
            for fut in as_completed(futs):
                j, bw_np, thres = fut.result()
                bw_cpu[j] = bw_np
                thres_array[j] = thres

        if stable_end > 0 and stable_end > hd_host:
            to_stabilize = thres_array[:stable_end]
            if hd_host > 0:
                to_stabilize[: min(hd_host, stable_end)] = np.nan
            fitted_threshold = float(np.nanmedian(to_stabilize) / 255.0)
            rebinarized = (video_cpu[hd_host:stable_end] >= fitted_threshold).astype(np.uint8) * 255
            bw_cpu[hd_host:stable_end] = rebinarized

    # ---- Connected components + penetration ----
    if gpu_enabled:
        # GPU largest-component without cuCIM (uses custom propagation)
        largest_blob_mask_gpu = keep_largest_component_nd_cuda(
            bw_final_gpu, connectivity=2
        ).astype(bool, copy=False)
        try:
            largest_blob_mask_filled_gpu = _binary_fill_holes_gpu(largest_blob_mask_gpu, mode=hole_fill_mode)
        except Exception:
            largest_blob_mask_filled_gpu = _fallback_fill_holes(largest_blob_mask_gpu, mode=hole_fill_mode)

        penetration_gpu = _penetration_gpu(largest_blob_mask_filled_gpu)

        if return_gpu_mask:
            return largest_blob_mask_filled_gpu.astype(bool), penetration_gpu

        return (
            cp.asnumpy(largest_blob_mask_filled_gpu).astype(bool),
            cp.asnumpy(penetration_gpu),
        )

    # CPU path
    labels, num = ndi.label(bw_cpu)
    if int(num) == 0:
        empty_mask = np.zeros_like(bw_cpu, dtype=bool)
        penetration = np.full(F, -1, dtype=int)
        return empty_mask, penetration

    volumes = ndi.sum(bw_cpu, labels, range(1, num + 1))
    largest_label = int(np.argmax(volumes)) + 1

    largest_blob_mask = labels == largest_label
    largest_blob_mask_filled = _binary_fill_holes_cpu(largest_blob_mask, mode=hole_fill_mode)

    col_sum_bw = np.sum(largest_blob_mask_filled, axis=1) >= 1
    penetration_host = penetration_bw_to_index(col_sum_bw)

    return largest_blob_mask_filled, penetration_host


def binarize_single_plume_video_gpu(video, hydraulic_delay, lighting_unchanged_duration=100, return_gpu_mask=False):
    """
    Convenience wrapper to keep backward compatibility.

    Runs the GPU-first thresholding path and hands off to CPU for labeling.
    """
    return binarize_single_plume_video(
        video,
        hydraulic_delay,
        lighting_unchanged_duration=lighting_unchanged_duration,
        prefer_gpu=True,
        return_gpu_mask=return_gpu_mask,
        hole_fill_mode="3D",
    )


def bw_boundaries_all_points_single_plume(bw_vid, connectivity=2, parallel=True, max_workers=None, umbrella_angle=180.0):
    """
    Compute all boundary points for a single BW video (F, H, W).

    - Accepts NumPy or CuPy input; GPU inputs are copied to CPU for boundary extraction.
    - Returns a list length F of tuples (coords_top_all, coords_bottom_all) with int32 coords.
    """
    cp = _get_cupy()
    if cp is not None and hasattr(bw_vid, "__cuda_array_interface__"):
        bw_cpu = cp.asnumpy(bw_vid)
    else:
        bw_cpu = np.asarray(bw_vid)

    F, H, W = bw_cpu.shape
    result = [None] * F
    if umbrella_angle == 180.0:
        x_scale=1.0
    else:
        tilt_angle = (180.0-umbrella_angle)/2.0
        tilt_angle_rad = tilt_angle / 180.0 * np.pi
        x_scale = 1.0/np.cos(tilt_angle_rad)


    def work(j):
        bw = np.asarray(bw_cpu[j], dtype=bool)
        return j, _boundary_points_one_frame(bw, connectivity, x_scale=x_scale)

    if parallel:
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(work, j) for j in range(F)]
            for fut in as_completed(futs):
                j, tup = fut.result()
                result[j] = tup
    else:
        for j in range(F):
            _, tup = work(j)
            result[j] = tup

    return result


def bw_boundaries_xband_filter_single_plume(boundary_results, penetration_old, lo=0.1, hi=0.6):
    """
    Filter boundary points (single plume) by an x-band defined by lo/hi fractions of penetration.

    Parameters
    ----------
    boundary_results : list of (coords_top_all, coords_bottom_all)
        Output of bw_boundaries_all_points_single_plume; length F.
    penetration_old : array-like, shape (F,)
        Penetration in pixels along x for each frame.
    lo, hi : floats
        Inclusive fractional range of penetration to keep.

    Returns
    -------
    filtered : list length F of (coords_top, coords_bottom) filtered to x in [xlo, xhi].
    """
    penetration_arr = np.asarray(penetration_old).ravel()
    F = len(boundary_results)
    if penetration_arr.size != F:
        raise ValueError(f"penetration_old must have length {F}, got {penetration_arr.size}")

    def _filter_coords(coords, xlo, xhi):
        coords = np.asarray(coords)
        if coords.size == 0:
            return coords
        x = coords[:, 1]
        xlo_i = int(np.floor(max(0, xlo)))
        xhi_i = int(np.ceil(max(xlo_i, xhi)))
        keep = (x >= xlo_i) & (x <= xhi_i)
        return coords[keep]

    filtered = [None] * F
    for j in range(F):
        if boundary_results[j] is None:
            continue
        coords_top_all, coords_bot_all = boundary_results[j]
        xlo = lo * float(penetration_arr[j])
        xhi = hi * float(penetration_arr[j])

        coords_top = _filter_coords(coords_top_all, xlo, xhi)
        coords_bot = _filter_coords(coords_bot_all, xlo, xhi)

        filtered[j] = (
            coords_top.astype(np.int32, copy=False),
            coords_bot.astype(np.int32, copy=False),
        )

    return filtered


def filter_schlieren(video, shock_wave_duration):
    smooth_frames = 3
    temporal_smoothing, temporal_smoothing_np = _prepare_temporal_smoothing(video, smooth_frames)

    inverted = 1.0 - temporal_smoothing_np

    foreground_godec = godec_like(inverted, 3)
    godec_pos = np.maximum(foreground_godec, 0.0)
    godec_pos = _min_max_scale(cp.asarray(godec_pos, dtype=cp.float32))

    if USING_CUPY:
        godec_pos = godec_pos.get()

    sobel_x, sobel_y = sobel_5x5_kernels()
    gx_video = filter_video_fft(godec_pos[:shock_wave_duration], sobel_x, mode="same")
    gy_video = filter_video_fft(godec_pos[:shock_wave_duration], sobel_y, mode="same")
    grad_mag = np.sqrt(gx_video**2 + gy_video**2)

    gamma = 1.1
    grad_mag_norm_inv = 1 - _min_max_scale(grad_mag) ** gamma

    thres = 0.9
    grad_mag_norm_inv[grad_mag_norm_inv < thres] = 0
    grad_mag_norm_inv[grad_mag_norm_inv > thres] = 1

    shock_wave_filtered = np.zeros_like(godec_pos)
    shock_wave_filtered[:shock_wave_duration] = _min_max_scale(
        grad_mag_norm_inv * godec_pos[:shock_wave_duration]
    )
    shock_wave_filtered[shock_wave_duration:] = godec_pos[shock_wave_duration:]

    return shock_wave_filtered


def mask_angle(video, angle_allowed_deg):
    F, H, W = video.shape

    half_angle_rad = cp.deg2rad(angle_allowed_deg / 2.0)
    tan_half = cp.tan(half_angle_rad)

    y = cp.arange(H)[:, None] - H / 2.0
    x = cp.arange(W)[None, :] + 1e-9

    mask_2d = (x >= 0) & (cp.abs(y / x) <= tan_half)
    width = cp.sum(mask_2d, axis=0)

    return mask_video(video, mask_2d), mask_2d, width


def pre_processing_mie(video, division=True):
    '''
    bilateral_filtered = bilateral_filter_video_volumetric_chunked_halo(
        video, (3, 5, 5), 3, 3
    )
    '''

    if USING_CUPY:
        bilateral_filtered = bilateral_filter_video_cupy(video, 7, 3, 3)
    else:
        bilateral_filtered = bilateral_filter_video_cpu(np.asarray(video), 7, 3, 3)

    

    bkg = bilateral_filtered[0]
    bkg[bkg == 0] = 1e-9
    bkg[bkg == cp.nan] = 1e-9
    if division:
        # divide by first frame and subracted backgrounde
        div_bkg = _min_max_scale(bilateral_filtered * ((1.0 / bkg)[None, :, :]))
        foreground = div_bkg - div_bkg[0][None, :, :]
        px_range_map = cp.max(foreground, axis=0) - cp.min(foreground, axis=0)
    else:
        # subtrated background if not local lighting correction
        foreground = bilateral_filtered - bilateral_filtered[0][None, :, :]
        px_range_map = cp.max(foreground, axis=0) - cp.min(foreground, axis=0)


    mask, _ = triangle_binarize_from_float(to_numpy(px_range_map))
    mask = keep_largest_component(mask)
    mask = binary_fill_holes(mask)
    mask = cp.asarray(mask)

    return (mask)[None, :, :] * foreground


def save_boundary_csv(boundary, out_csv, origin=(0, 0), executor=None):
    items = []
    total = 0
    for frame_idx, frame_pts in enumerate(boundary):
        if frame_pts is None:
            continue

        if isinstance(frame_pts, (tuple, list)) and len(frame_pts) == 2:
            lower = np.asarray(frame_pts[0])
            upper = np.asarray(frame_pts[1])
            if lower.size == 0:
                pts = upper
            elif upper.size == 0:
                pts = lower
            else:
                pts = np.concatenate((lower, upper), axis=0)
        else:
            pts = np.asarray(frame_pts)

        if pts.size == 0:
            continue

        n = int(pts.shape[0])
        items.append((frame_idx, pts))
        total += n

    if total == 0:
        df = pd.DataFrame(columns=["frame", "point_idx", "y", "x"])
        if executor is None:
            df.to_csv(out_csv, index=False)
            return
        return executor.submit(df.to_csv, out_csv, index=False)

    frames = np.empty(total, dtype=np.int32)
    point_idxs = np.empty(total, dtype=np.int32)
    ys = np.empty(total, dtype=np.float32)
    xs = np.empty(total, dtype=np.float32)

    origin_y = np.float32(origin[0])
    origin_x = np.float32(origin[1])
    use_origin = origin_y != 0 or origin_x != 0

    offset = 0
    for frame_idx, pts in items:
        n = int(pts.shape[0])
        end = offset + n
        frames[offset:end] = frame_idx
        point_idxs[offset:end] = np.arange(n, dtype=np.int32)

        y = np.asarray(pts[:, 0], dtype=np.float32)
        x = np.asarray(pts[:, 1], dtype=np.float32)
        if use_origin:
            y = y - origin_y
            x = x - origin_x
        ys[offset:end] = y
        xs[offset:end] = x
        offset = end

    df = pd.DataFrame(
        {"frame": frames, "point_idx": point_idxs, "y": ys, "x": xs},
        copy=False,
    )
    if executor is None:
        df.to_csv(out_csv, index=False)
        return
    return executor.submit(df.to_csv, out_csv, index=False)

