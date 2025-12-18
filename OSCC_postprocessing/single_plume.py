from OSCC_postprocessing.multihole_utils import *
from OSCC_postprocessing.functions_bw import *
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import binary_fill_holes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from OSCC_postprocessing.functions_bw import _triangle_threshold_from_hist, _boundary_points_one_frame

# Prefer GPU-accelerated cuCIM if available; fall back to scikit-image on CPU.
try:
    import cucim.skimage.measure as measure
except ImportError:  # Windows wheels are not published for cuCIM
    from skimage import measure

def _get_cupy():
    """Return the CuPy module if available, otherwise ``None``."""
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:
        return None
    

def _binary_fill_holes_cpu(mask, mode="3D"):
    """CPU hole filling in 3D or per-frame 2D using an anisotropic structure (no per-frame loop)."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mode.upper() == "3D" or mask_bool.ndim < 3:
        return binary_fill_holes(mask_bool)

    # 2D-per-frame fill via 3D call with no time connectivity
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, :, :] = True  # connect only within each frame
    return binary_fill_holes(mask_bool, structure=struct)


def _binary_fill_holes_gpu(mask_cp, mode="3D"):
    """GPU hole filling; falls back to CPU if cupyx implementation is missing."""
    import cupy as cp
    import cupyx.scipy.ndimage as cndi

    try:
        if mode.upper() == "3D" or mask_cp.ndim < 3:
            return cndi.binary_fill_holes(mask_cp)
        struct = cp.zeros((3, 3, 3), dtype=bool)
        struct[1, :, :] = True
        return cndi.binary_fill_holes(mask_cp, structure=struct)
    except Exception:
        # cupyx may not implement binary_fill_holes; fall back once to CPU
        return _fallback_fill_holes(mask_cp, mode=mode)


def _fallback_fill_holes(mask_cp, mode="3D"):
    import cupy as cp

    filled = _binary_fill_holes_cpu(cp.asnumpy(mask_cp), mode=mode)
    return cp.asarray(filled)


def _penetration_gpu(bw_cp):
    import cupy as cp

    arr = bw_cp.astype(bool)
    any_true = arr.any(axis=1)
    # reverse each row
    rev_idx = arr.shape[1] - 1 - arr[:, ::-1].argmax(axis=1)
    rev_idx = rev_idx.astype(int)
    rev_idx = cp.where(any_true, rev_idx, -1)
    return rev_idx


def regionprops_gpu(label_img, intensity_img=None):
    """
    Minimal regionprops-like utility for CuPy.

    Supports: area, centroid, bbox (2D or 3D).
    Falls back with a RuntimeError if CuPy is unavailable.
    """
    cp = _get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available; cannot run GPU regionprops.")

    import cupyx.scipy.ndimage as cndi

    label_img = cp.asarray(label_img)
    if label_img.size == 0:
        return []

    label_ids = cp.unique(label_img)
    label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        return []

    # area / volume
    ones = cp.ones_like(label_img, dtype=cp.float32)
    areas = cndi.sum(ones, label_img, label_ids)

    # Skip labels that disappeared (area == 0), otherwise bboxes lookup will fail.
    valid_mask = areas > 0
    label_ids = label_ids[valid_mask]
    areas = areas[valid_mask]
    if label_ids.size == 0:
        return []

    # centroid
    centroids = cndi.center_of_mass(ones, label_img, label_ids)

    # bbox: use min / max of indices per label; host-side grouping keeps code simple.
    coords = cp.argwhere(label_img > 0)
    if coords.size == 0:
        return []
    labels_flat = label_img[label_img > 0]

    coords_cpu = coords.get()
    labels_cpu = labels_flat.get()

    from collections import defaultdict

    bboxes = defaultdict(lambda: [None, None])  # label -> [min_vec, max_vec]
    for idx, lab in zip(coords_cpu, labels_cpu):
        lab = int(lab)
        if bboxes[lab][0] is None:
            bboxes[lab][0] = list(idx)
            bboxes[lab][1] = list(idx)
        else:
            bmin, bmax = bboxes[lab]
            for i, v in enumerate(idx):
                if v < bmin[i]:
                    bmin[i] = v
                if v > bmax[i]:
                    bmax[i] = v

    label_ids_cpu = label_ids.get()
    areas_cpu = areas.get()
    centroids_cpu = [tuple(float(c) for c in cp.asnumpy(cen)) for cen in centroids]

    props = []
    for i, lab in enumerate(label_ids_cpu):
        bbox_min, bbox_max = bboxes[int(lab)]
        bbox = tuple(bbox_min + [x + 1 for x in bbox_max])  # (min..., max+1...)

        props.append(
            {
                "label": int(lab),
                "area": float(areas_cpu[i]),
                "centroid": centroids_cpu[i],
                "bbox": bbox,
            }
        )
    return props

def ransac_line_1d(
    y,
    x=None,
    max_trials=1000,
    residual_threshold=None,
    min_inliers=2,
    random_state=None,
):
    """
    RANSAC line fit for an indexed 1D array y[i], robust to NaNs.

    Parameters
    ----------
    y : array-like, shape (N,)
        Observations (can contain NaNs).
    x : array-like, shape (N,), optional
        X-coordinates. If None, x = np.arange(N).
        Can also contain NaNs (those points are ignored).
    max_trials : int
        Number of random RANSAC iterations.
    residual_threshold : float or None
        Max |residual| to count a point as inlier.
        If None, uses 1.0 * median absolute deviation of y (rough rule of thumb).
    min_inliers : int
        Minimum number of inliers to accept a model.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    best_a : float
        Slope of best line.
    best_b : float
        Intercept of best line.
    inlier_mask : np.ndarray, shape (N,), dtype=bool
        True where a point is considered an inlier.
    """
    y = np.asarray(y, dtype=float)
    N = y.size

    if x is None:
        x = np.arange(N, dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    # Mask out NaNs in x or y
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if x_valid.size < 2:
        raise ValueError("Not enough valid (non-NaN) points for line fitting.")

    rng = np.random.default_rng(random_state)

    # If threshold not given, use a robust scale based on y's MAD
    if residual_threshold is None:
        med = np.median(y_valid)
        mad = np.median(np.abs(y_valid - med)) + 1e-9
        residual_threshold = 2.5 * mad  # heuristic

    best_inlier_count = 0
    best_a = None
    best_b = None
    best_inlier_mask_valid = None

    for _ in range(max_trials):
        # Randomly pick 2 distinct points
        idx = rng.choice(x_valid.size, size=2, replace=False)
        x_s = x_valid[idx]
        y_s = y_valid[idx]

        if x_s[1] == x_s[0]:
            # Degenerate sample, skip
            continue

        # Fit line y = a x + b to the two points
        a = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        b = y_s[0] - a * x_s[0]

        # Compute residuals on all valid points
        y_pred = a * x_valid + b
        residuals = np.abs(y_valid - y_pred)

        # Inliers under threshold
        inliers = residuals <= residual_threshold
        inlier_count = inliers.sum()

        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_a, best_b = a, b
            best_inlier_mask_valid = inliers

    if best_inlier_mask_valid is None:
        # Fallback: use simple least squares on all valid points
        # (or you can raise an error instead)
        A = np.vstack([x_valid, np.ones_like(x_valid)]).T
        best_a, best_b = np.linalg.lstsq(A, y_valid, rcond=None)[0]
        best_inlier_mask_valid = np.ones_like(y_valid, dtype=bool)

    # Optional: refit using all inliers for a better estimate
    x_in = x_valid[best_inlier_mask_valid]
    y_in = y_valid[best_inlier_mask_valid]
    A = np.vstack([x_in, np.ones_like(x_in)]).T
    best_a, best_b = np.linalg.lstsq(A, y_in, rcond=None)[0]

    # Build a full-length mask (including NaNs as False)
    inlier_mask = np.zeros(N, dtype=bool)
    inlier_mask[valid_mask] = best_inlier_mask_valid

    return best_a, best_b, inlier_mask

def ransac_fixed_intercept(x, y, b0, 
                           max_iter=2000,
                           residual_thresh=1.0,
                           min_inliers=10):
    x = np.asarray(x)
    y = np.asarray(y)

    # mask valid points
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        raise ValueError("No valid points")

    best_a = None
    best_inliers = []
    
    for _ in range(max_iter):
        # 1-point minimal sample
        idx = np.random.randint(0, n)
        xi, yi = x[idx], y[idx]

        # avoid division by zero
        if xi == 0:
            continue

        a_candidate = (yi - b0) / xi

        # compute residuals
        residuals = np.abs((a_candidate * x + b0) - y)

        inliers = np.where(residuals < residual_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_a = a_candidate

        # early stop if enough inliers
        if len(best_inliers) >= min_inliers:
            break

    if best_a is None:
        raise RuntimeError("RANSAC failed.")

    # Optional: refine slope using least squares on inliers
    xi = x[best_inliers]
    yi = y[best_inliers]
    refined_a = np.sum(xi * (yi - b0)) / np.sum(xi**2)

    return refined_a, best_inliers

def linear_regression_fixed_intercept(x, y, b0):
    x = np.asarray(x)
    y = np.asarray(y)

    # Mask out NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    y_centered = y - b0

    numerator = np.sum(x * y_centered)
    denominator = np.sum(x**2)

    if denominator == 0:
        raise ValueError("Cannot estimate slope: x has no variation or all x=0.")

    a = numerator / denominator
    return a

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


def bw_boundaries_all_points_single_plume(bw_vid, connectivity=2, parallel=True, max_workers=None):
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

    def work(j):
        bw = np.asarray(bw_cpu[j], dtype=bool)
        return j, _boundary_points_one_frame(bw, connectivity)

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
