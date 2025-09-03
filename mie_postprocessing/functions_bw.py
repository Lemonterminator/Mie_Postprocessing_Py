from scipy.ndimage import generic_filter, binary_opening, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import concurrent.futures
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import measure
from scipy.ndimage import binary_erosion, generate_binary_structure

# Optional GPU acceleration via CuPy.
# Fall back to NumPy/SciPy on machines without CUDA (e.g. laptops).
try:  # pragma: no cover - runtime hardware dependent
    import cupy as cp  # type: ignore
    # from cupyx.scipy.ndimage import median_filter  # type: ignore
    CUPY_AVAILABLE = True
except Exception:  # ImportError, CUDA failure, etc.
    cp = np  # type: ignore
    # from scipy.ndimage import median_filter  # type: ignore
    cp.asnumpy = lambda x: x  # type: ignore[attr-defined]
    CUPY_AVAILABLE = False

# Optional GPU connected-components via cuCIM (fast GPU labeling)
try:  # pragma: no cover - runtime/hardware dependent
    from cucim.skimage import measure as cucim_measure  # type: ignore
    CUCIM_AVAILABLE = True
except Exception:
    CUCIM_AVAILABLE = False

# -----------------------------
# Masking and Binarization Pipeline
# -----------------------------
def mask_video(video: np.ndarray, chamber_mask: np.ndarray) -> np.ndarray:
    # Ensure chamber_mask is boolean.
    chamber_mask_bool = chamber_mask if chamber_mask.dtype == bool else (chamber_mask > 0)
    # Use broadcasting: multiplies each frame elementwise with the mask.
    if video.shape[1] != chamber_mask.shape[0] or video.shape[2] != chamber_mask.shape[1]:
        chamber_mask_bool = cv2.resize(chamber_mask_bool.astype(np.uint8), (video.shape[2], video.shape[1]), interpolation=cv2.INTER_NEAREST)
        # raise ValueError("Video dimensions and mask dimensions do not match.")
    return video * chamber_mask_bool

# -----------------------------
# Global Threshold Binarization
# -----------------------------
def binarize_video_global_threshold(video, method='otsu', thresh_val=None):
    if method == 'otsu':
        # Compute threshold over the whole video (flattened)
        threshold = threshold_otsu(video)
    elif method == 'fixed':
        if thresh_val is None:
            raise ValueError("Provide a threshold value for 'fixed' method.")
        threshold = thresh_val
    else:
        raise ValueError("Invalid method. Use 'otsu' or 'fixed'.")
    
    # Broadcasting applies the comparison element-wise across the entire video array.
    binary_video = (video >= threshold).astype(np.uint8) * 255
    return binary_video


def calculate_bw_area(BW: np.ndarray):
    num_frames, height, width = BW.shape
    area = np.zeros(num_frames, dtype=np.float32)
    for n in range(num_frames):
        area[n] = np.sum(BW[n] == 255)
    return area

def apply_morph_open(intermediate_frame, disk_size):
    selem = disk(disk_size)
    opened = binary_opening(intermediate_frame, selem)
    return opened

def apply_hole_filling(opened_frame):
    filled = binary_fill_holes(opened_frame)
    processed_frame = (filled * 255).astype(np.uint8)
    frame_area = np.sum(filled)
    return processed_frame, frame_area

def apply_morph_open_video(intermediate_video: np.ndarray, disk_size: int) -> np.ndarray:
    """
    Apply morphological opening to each frame in the video in parallel.
    
    Parameters:
        intermediate_video (np.ndarray): Video after thresholding (frames, height, width).
        disk_size (int): Radius for the disk-shaped structuring element.
    
    Returns:
        np.ndarray: Video after morphological opening.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(
            lambda frame: apply_morph_open(frame, disk_size),
            intermediate_video
        ))
    return np.array(results)

def apply_hole_filling_video(opened_video: np.ndarray):
    """
    Apply hole filling and compute the white pixel area for each frame in the video in parallel.
    
    Parameters:
        opened_video (np.ndarray): Video after morphological opening (frames, height, width).
    
    Returns:
        processed_video (np.ndarray): Video after hole filling, with values 0 or 255.
        area (np.ndarray): Array of white pixel counts per frame.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(
            lambda frame: apply_hole_filling(frame),
            opened_video
        ))
    num_frames = opened_video.shape[0]
    processed_video = np.zeros_like(opened_video, dtype=np.uint8)
    area = np.zeros(num_frames, dtype=np.float32)
    for i, (processed_frame, frame_area) in enumerate(results):
        processed_video[i] = processed_frame
        area[i] = frame_area
    return processed_video, area

    from scipy.ndimage import binary_fill_holes


def _fill_frame(frame_bool):
    filled = binary_fill_holes(frame_bool)
    return filled

def fill_video_holes_parallel(bw_video: np.ndarray,
                               n_workers: int = None) -> np.ndarray:
    """
    Fill holes in each frame of a binary video in parallel.
    
    Parameters
    ----------
    bw_video : np.ndarray
        Binary video data of shape (n_frames, height, width), values 0/1 or 0/255.
    n_workers : int, optional
        Number of worker processes to use. Defaults to os.cpu_count() if None.
    
    Returns
    -------
    np.ndarray
        Hole‑filled binary video, same shape and dtype as input.
    """
    # Ensure we have boolean frames
    # Any nonzero becomes True; zero remains False.
    bw_bool_video = (bw_video > 0)
    
    
    # Launch parallel filling
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        # executor.map returns in order; we collect into a list
        filled_frames = list(exe.map(_fill_frame, bw_bool_video))
    
    # Stack back into a (n_frames, H, W) array
    return np.stack(filled_frames, axis=0)

def fill_video_holes_gpu(bw_video: np.ndarray) -> np.ndarray:
    from cupy import asnumpy
    """
    Fill holes in each frame of a binary video on GPU via CuPy.
    """
    # 1. Upload to GPU and binarize
    bw_gpu = cp.asarray(bw_video) > 0

    # 2. Run hole‐filling on GPU
    # cupyx.scipy.ndimage.binary_fill_holes is not implemented and returns None.
    # Use CPU fallback for hole filling if needed.
    # Alternatively, use a custom GPU implementation or fallback to scipy.ndimage on CPU.
    import scipy.ndimage
    filled_cpu = scipy.ndimage.binary_fill_holes(asnumpy(bw_gpu))
    filled_gpu = cp.asarray(filled_cpu)

    # 3. Download back to host, preserving dtype
    return (filled_gpu.astype(bw_video.dtype) * 255
            if bw_video.max() > 1
            else filled_gpu.astype(bw_video.dtype))
'''
def triangle_binarize(ang_float32, blur=True):
    # 1. Clean / normalize to [0,255] uint8
    img = np.nan_to_num(ang_float32, nan=0.0)  # avoid NaNs
    lo, hi = img.min(), img.max()
    if hi <= lo:
        # constant image, nothing to threshold
        return np.zeros_like(img, dtype=np.uint8), 0.0

    norm = (img - lo) / (hi - lo)           # [0,1]
    u8 = (norm * 255).astype(np.uint8)      # to 0..255

    # 2. Optional smoothing to reduce noise which can destabilize histogram peak
    if blur:
        u8 = cv2.GaussianBlur(u8, (5, 5), 0)

    # 3. Triangle thresholding (global)
    thresh_val, binarized = cv2.threshold(
        u8,
        0,                  # ignored when using OTSU / TRIANGLE flags
        255,                # max value for binary
        cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )

    return binarized, thresh_val

def triangle_binarize_u8(u8, blur=True):
    if blur:
        u8 = cv2.GaussianBlur(u8, (5, 5), 0)
    t, binarized = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binarized, t

def triangle_binarize_from_float(img_f32, blur=True):
    # Fast normalize to 0..255 uint8 in one call (releases GIL)
    u8 = cv2.normalize(img_f32, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return triangle_binarize_u8(u8, blur=blur)
'''


def keep_largest_component(bw, connectivity=2):
    
    """
    Keep only the largest connected component of 1s in a 2D binary array.

    Parameters
    ----------
    bw : array-like of {0,1}, shape (H, W)
    connectivity : int, 1 or 2
        1 -> 4-connectivity; 2 -> 8-connectivity (MATLAB default is 8).

    Returns
    -------
    largest : ndarray of same shape, dtype same as input
        Binary mask with only the largest component preserved.
    """
    binary_mask = np.asarray(bw, dtype=bool)
    if connectivity == 1:
        # 4-connectivity structure
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]], dtype=bool)
    else:
        # 8-connectivity structure
        structure = np.ones((3, 3), dtype=bool)
    
    labeled, num_features = (ndimage.label(binary_mask, structure=structure))

    # If blank then return blank
    if num_features==0: return np.zeros_like(binary_mask, dtype=bw.dtype)

    # Count pixels in each label, 0 is background
    counts = np.bincount(labeled.ravel())
    counts[0] = 0 # ignore background
    largest_label = counts.argmax()
    largest = (labeled == largest_label)
    return largest.astype(bw.dtype)

def keep_largest_component_nd(bw, connectivity=None):
    """
    Keep only the largest connected component in an nD binary array.

    Parameters
    ----------
    bw : array-like of {0,1}, shape (N1, N2, ..., Nk)
    connectivity : int or None
        For 2D:
            1 -> 4-connectivity, 2 -> 8-connectivity
        For 3D:
            1 -> 6-connectivity, 2 -> 18-connectivity, 3 -> 26-connectivity
        For nD:
            in [1, nd], where nd = bw.ndim.
        If None: use full connectivity (= bw.ndim).

    Returns
    -------
    largest : ndarray of same shape, dtype same as input
    """
    binary_mask = np.asarray(bw, dtype=bool)
    nd = binary_mask.ndim
    if connectivity is None:
        connectivity = nd  # full connectivity

    if not (1 <= connectivity <= nd):
        raise ValueError(f"connectivity must be in [1, {nd}], got {connectivity}")

    structure = ndimage.generate_binary_structure(nd, connectivity)
    labeled, num_features = ndimage.label(binary_mask, structure=structure)

    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=bw.dtype)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    largest = (labeled == largest_label)
    return largest.astype(bw.dtype)


# -----------------------------
# CUDA-accelerated Connected Components
# -----------------------------
def _to_cupy(x):
    """Internal: convert numpy array to CuPy if available; pass through CuPy arrays."""
    if CUPY_AVAILABLE and hasattr(x, "__cuda_array_interface__"):
        return x
    return cp.asarray(x) if CUPY_AVAILABLE else x


def _return_like_input(mask_gpu, like):
    """Return array with same library and dtype as input 'like'."""
    if CUPY_AVAILABLE and hasattr(like, "__cuda_array_interface__"):
        # Caller passed a CuPy array: return CuPy
        return mask_gpu.astype(like.dtype, copy=False)
    # Caller passed NumPy array: return NumPy
    return cp.asnumpy(mask_gpu).astype(like.dtype, copy=False) if CUPY_AVAILABLE else mask_gpu.astype(like.dtype, copy=False)


def _generate_neighbor_offsets(nd, connectivity):
    """Generate neighbor offsets for given ndim and connectivity.
    Includes all offsets in {-1,0,1}^nd \ {0} with L1 distance <= connectivity.
    """
    from itertools import product
    offsets = []
    for off in product((-1, 0, 1), repeat=nd):
        if all(o == 0 for o in off):
            continue
        if sum(abs(int(o)) for o in off) <= int(connectivity):
            offsets.append(tuple(int(o) for o in off))
    return offsets


def _slices_for_offset(offset, shape):
    """Return (src_slices, dst_slices) for shifting by 'offset'."""
    src = []
    dst = []
    for k, (dk, nk) in enumerate(zip(offset, shape)):
        if dk == 0:
            src.append(slice(0, nk))
            dst.append(slice(0, nk))
        elif dk > 0:
            src.append(slice(0, nk - dk))
            dst.append(slice(dk, nk))
        else:  # dk < 0
            dk = -dk
            src.append(slice(dk, nk))
            dst.append(slice(0, nk - dk))
    return tuple(src), tuple(dst)


def _gpu_label_propagation(binary_mask, connectivity):
    """Label connected components on GPU via iterative label propagation.

    Returns labels with 0 as background, positive ints per component.
    """
    # Initialize labels: unique id per foreground pixel; 0 for background
    shape = binary_mask.shape
    labels = cp.zeros(shape, dtype=cp.int32)
    if binary_mask.any():
        labels[binary_mask] = cp.arange(1, int(binary_mask.sum()) + 1, dtype=cp.int32)

    # Map labels to image positions to preserve spatial uniqueness
    # Create a base grid of unique ids for all pixels, then mask
    # This guarantees unique labels even for large sparse masks
    if labels.max() == 0:
        # Try alternative initialization using linear indices to avoid counting first
        lin_ids = cp.arange(labels.size, dtype=cp.int32).reshape(shape) + 1
        labels = cp.where(binary_mask, lin_ids, 0)

    offsets = _generate_neighbor_offsets(binary_mask.ndim, connectivity)

    changed = True
    max_iter = 10000  # safety bound
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        for off in offsets:
            src_sl, dst_sl = _slices_for_offset(off, shape)
            neigh = labels[src_sl]
            curr = labels[dst_sl]
            # Compute candidate min label among neighbors where both are foreground
            both_fg = (neigh > 0) & (curr > 0)
            if not both_fg.any():
                continue
            min_lab = cp.where(both_fg, cp.minimum(curr, neigh), curr)
            if (min_lab != curr).any():
                changed = True
                labels[dst_sl] = min_lab
        # Optional early exit if no changes
    return labels


def keep_largest_component_cuda(bw, connectivity=2):
    """
    CUDA version of keep_largest_component for 2D binary arrays.

    - Uses cuCIM on GPU if available; falls back to CPU implementation otherwise.
    - If input is a CuPy array, returns CuPy; if NumPy, returns NumPy.
    """
    # Fallback to CPU if no GPU
    if not CUPY_AVAILABLE:
        return keep_largest_component(bw, connectivity=connectivity)

    bw_gpu = _to_cupy(bw)
    binary_mask = (bw_gpu != 0)

    if CUCIM_AVAILABLE:
        labeled, num_features = cucim_measure.label(binary_mask, connectivity=connectivity, return_num=True)
        if int(num_features) == 0:
            zeros = cp.zeros_like(binary_mask, dtype=bool)
            return _return_like_input(zeros, bw)
    else:
        # Prefer fast CPU fallback over slow GPU propagation for moderate 2D slices
        bw_np = cp.asnumpy(binary_mask)
        largest_np = keep_largest_component(bw_np, connectivity=connectivity)
        largest_cp = cp.asarray(largest_np)
        return _return_like_input(largest_cp, bw)

    # Find largest component label using unique counts on foreground
    labels_fg = labeled[binary_mask]
    uniq, counts = cp.unique(labels_fg, return_counts=True)
    # uniq does not include 0 by construction
    largest_label = uniq[cp.argmax(counts)]
    largest = (labeled == largest_label)
    return _return_like_input(largest, bw)


def keep_largest_component_nd_cuda(bw, connectivity=None):
    """
    CUDA version of keep_largest_component_nd for nD binary arrays.

    - Uses cuCIM on GPU if available; falls back to CPU implementation otherwise.
    - connectivity: int in [1, ndim] or None (defaults to ndim for full connectivity).
    - Preserves input library (NumPy/CuPy) and dtype.
    """
    if not CUPY_AVAILABLE:
        return keep_largest_component_nd(bw, connectivity=connectivity)

    bw_gpu = _to_cupy(bw)
    binary_mask = (bw_gpu != 0)
    nd = binary_mask.ndim
    if connectivity is None:
        connectivity = nd
    if not (1 <= int(connectivity) <= nd):
        raise ValueError(f"connectivity must be in [1, {nd}], got {connectivity}")

    if CUCIM_AVAILABLE:
        labeled, num_features = cucim_measure.label(binary_mask, connectivity=int(connectivity), return_num=True)
        if int(num_features) == 0:
            zeros = cp.zeros_like(binary_mask, dtype=bool)
            return _return_like_input(zeros, bw)
    else:
        labeled = _gpu_label_propagation(binary_mask, connectivity=int(connectivity))
        num_features = int(cp.unique(labeled[binary_mask]).size)
        if num_features == 0:
            zeros = cp.zeros_like(binary_mask, dtype=bool)
            return _return_like_input(zeros, bw)

    labels_fg = labeled[binary_mask]
    uniq, counts = cp.unique(labels_fg, return_counts=True)
    largest_label = uniq[cp.argmax(counts)]
    largest = (labeled == largest_label)
    return _return_like_input(largest, bw)

def penetration_bw_to_index(bw):
    arr = bw.astype(bool)
    # Find where True elements exist
    any_true = arr.any(axis=1)  # shape (N,)
    # Reverse each row to find last occurrence efficiently
    rev_idx = arr.shape[1] - 1 - arr[:, ::-1].argmax(axis=1)
    # Mask rows with no True values
    rev_idx[~any_true] = -1  # or use `np.nan` if float output is acceptable
    return rev_idx    



def _triangle_threshold_from_hist(hist):
    """
    Compute the Triangle threshold index from a 256-bin histogram.
    Returns an integer t in [0, 255].
    """
    # active range
    nz = np.flatnonzero(hist)
    if nz.size == 0:
        return 0
    left, right = int(nz[0]), int(nz[-1])

    # peak
    peak = int(np.argmax(hist))
    h_peak = float(hist[peak])

    # choose the farther endpoint from peak to form the baseline
    if (peak - left) >= (right - peak):
        end = left
    else:
        end = right

    # line through (peak, h_peak) and (end, h_end)
    h_end = float(hist[end])
    dx = end - peak
    dy = h_end - h_peak

    # avoid division by zero when dx==0 (degenerate but handle anyway)
    if dx == 0:
        return peak

    # for each i between peak and end, compute perpendicular distance to the line
    if end > peak:
        idx = np.arange(peak, end + 1)
    else:
        idx = np.arange(end, peak + 1)

    # vector from peak to (i, hist[i])
    xi = idx - peak
    yi = hist[idx].astype(float) - h_peak

    # distance = |dy*xi - dx*yi| / sqrt(dx^2 + dy^2); denominator common -> argmax numerator
    num = np.abs(dy * xi - dx * yi)
    i_rel = int(np.argmax(num))
    t = int(idx[i_rel])
    return t


def triangle_binarize_u8(u8, blur=True, ignore_zero=False, threshold_on_unblurred=True):
    """
    Triangle-threshold an 8-bit image with optional zero-ignoring.

    Parameters
    ----------
    u8 : np.uint8, shape (H, W)
    blur : bool
        Apply Gaussian blur before binarization (for smoother masks).
    ignore_zero : bool
        If True, compute the threshold from the histogram of non-zero pixels only.
        (Zero bin is removed from the histogram to avoid bias from large masked areas.)
    threshold_on_unblurred : bool
        If True and ignore_zero=True, derive the threshold from the *unblurred* image
        to avoid blur leaking non-zero into zero regions; then apply that threshold
        to the (optionally) blurred image.

    Returns
    -------
    binarized : np.uint8 in {0,255}
    t : int (threshold in 0..255 used on the final image)
    """
    if u8.dtype != np.uint8:
        u8 = u8.astype(np.uint8, copy=False)

    # choose image for threshold computation
    u8_for_t = u8 if (ignore_zero and threshold_on_unblurred) else (cv2.GaussianBlur(u8, (5,5), 0) if blur else u8)

    if ignore_zero:
        # histogram on non-zero pixels only (set hist[0]=0)
        hist = cv2.calcHist([u8_for_t], [0], None, [256], [0,256]).flatten()
        hist[0] = 0
        # if all non-zero vanish, return all zeros
        if hist.sum() == 0:
            return np.zeros_like(u8), 0
        t = _triangle_threshold_from_hist(hist)
        # now apply threshold to (optionally) blurred image
        u8_apply = cv2.GaussianBlur(u8, (5,5), 0) if blur else u8
        _, binarized = cv2.threshold(u8_apply, t, 255, cv2.THRESH_BINARY)
        return binarized, int(t)
    else:
        # fast OpenCV path (zeros included)
        u8_apply = cv2.GaussianBlur(u8, (5,5), 0) if blur else u8
        t, binarized = cv2.threshold(u8_apply, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return binarized, int(t)


def triangle_binarize_from_float(img_f32, blur=True, ignore_zero=False, threshold_on_unblurred=True):
    """
    Normalize float image to 0..255 u8 and apply Triangle binarization with optional zero-ignoring.
    """
    # Fast normalize to 0..255 uint8 (releases GIL internally)
    u8 = cv2.normalize(img_f32, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return triangle_binarize_u8(
        u8, blur=blur, ignore_zero=ignore_zero, threshold_on_unblurred=threshold_on_unblurred
    )


# --- Split: compute all boundary points (no x-band filter) ---
def _boundary_points_one_frame(bw, connectivity=2):
    """
    Compute all boundary points of a single binary frame without x-band filtering.

    Returns: (coords_top_all, coords_bottom_all) as (N,2) int32 arrays (y, x).
    """
    H, W = bw.shape
    struct = generate_binary_structure(2, 2 if connectivity == 2 else 1)
    boundary = bw & ~binary_erosion(bw, structure=struct, border_value=0)
    if not boundary.any():
        return (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32))

    ys, xs = np.nonzero(boundary)
    if ys.size == 0:
        return (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32))

    mid = (H - 1) / 2.0
    top_mask = ys <= mid
    bot_mask = ~top_mask

    coords_top = np.column_stack((ys[top_mask], xs[top_mask])).astype(np.int32)
    coords_bot = np.column_stack((ys[bot_mask], xs[bot_mask])).astype(np.int32)
    return coords_top, coords_bot


def bw_boundaries_all_points(
    bw_vids, connectivity=2, parallel=False, max_workers=None
):
    """
    Compute all boundary points for every frame (no x-band filtering).

    Parameters
    ----------
    bw_vids : array, shape (R, F, H, W), binary
    connectivity : 1 (4-neigh) or 2 (8-neigh)
    parallel : bool
    max_workers : int or None

    Returns
    -------
    result : list length R; each item is list length F of tuples (coords_top_all, coords_bottom_all)
    """
    R, F, H, W = bw_vids.shape
    result = [[None] * F for _ in range(R)]

    def work(i, j):
        bw = np.asarray(bw_vids[i, j], dtype=bool)
        return i, j, _boundary_points_one_frame(bw, connectivity)

    if parallel:
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(work, i, j) for i in range(R) for j in range(F)]
            for fut in as_completed(futs):
                i, j, tup = fut.result()
                result[i][j] = tup
    else:
        for i in range(R):
            for j in range(F):
                _, _, tup = work(i, j)
                result[i][j] = tup

    return result


# --- Split: filter previously computed boundary points by x-band ---
def bw_boundaries_xband_filter(boundary_results, penetration_old, lo=0.1, hi=0.6):
    """
    Filter precomputed boundary points by an x-band defined by lo/hi fractions of penetration.

    Parameters
    ----------
    boundary_results : list of lists as returned by bw_boundaries_all_points
        shape [R][F] of (coords_top_all, coords_bottom_all)
    penetration_old : array, shape (R, F)
        Penetration in pixels along x for each (R, F)
    lo, hi : floats
        Inclusive fractional range of penetration to keep

    Returns
    -------
    filtered : same nested structure with coords filtered to x in [xlo, xhi]
    """
    R = len(boundary_results)
    F = len(boundary_results[0]) if R > 0 else 0
    assert penetration_old.shape == (R, F)

    def _filter_coords(coords, xlo, xhi):
        if coords.size == 0:
            return coords
        x = coords[:, 1]
        xlo_i = int(np.floor(max(0, xlo)))
        xhi_i = int(np.ceil(max(xlo_i, xhi)))
        keep = (x >= xlo_i) & (x <= xhi_i)
        return coords[keep]

    filtered = [[None] * F for _ in range(R)]
    for i in range(R):
        for j in range(F):
            coords_top_all, coords_bot_all = boundary_results[i][j]
            xlo = lo * float(penetration_old[i, j])
            xhi = hi * float(penetration_old[i, j])
            coords_top = _filter_coords(coords_top_all, xlo, xhi)
            coords_bot = _filter_coords(coords_bot_all, xlo, xhi)
            filtered[i][j] = (coords_top.astype(np.int32, copy=False), # type: ignore
                              coords_bot.astype(np.int32, copy=False))

    return filtered
