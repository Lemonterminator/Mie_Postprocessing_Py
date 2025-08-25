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

def penetration_bw_to_index(bw):
    arr = bw.astype(bool)
    # Find where True elements exist
    any_true = arr.any(axis=1)  # shape (N,)
    # Reverse each row to find last occurrence efficiently
    rev_idx = arr.shape[1] - 1 - arr[:, ::-1].argmax(axis=1)
    # Mask rows with no True values
    rev_idx[~any_true] = -1  # or use `np.nan` if float output is acceptable
    return rev_idx    

def bw_boundaries_all_segments(
    bw_vids, penetration_old, lo=0.0, hi=1.0, connectivity=2,
    parallel=False, max_workers=None
):
    """
    Parameters
    ----------
    bw_vids : array, shape (R, F, H, W), binary
    penetration_old : array, shape (R, F), in pixels along x
    lo, hi : floats for fraction of penetration to keep (inclusive range)
    connectivity : 1 (4-neigh) or 2 (8-neigh)
    parallel : bool, use threads across frames for speed
    max_workers : int or None

    Returns
    -------
    result : list length R; each item is list length F of tuples (coords_top, coords_bottom),
             where coords_* are (N,2) int arrays of (y,x).
    """
    R, F, H, W = bw_vids.shape
    assert penetration_old.shape == (R, F)
    
    result = [[None] * F for _ in range(R)]


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
