from scipy.ndimage import generic_filter, binary_opening, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import concurrent.futures

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
from concurrent.futures import ProcessPoolExecutor

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

def segment2bw(segment, j0=0, blur=True, max_workers=None):
    T, H, W = segment.shape
    bw_vid = np.empty((T, H, W), dtype=np.uint8)
    thres_array = np.zeros(T, dtype=np.float32)
    seg = (segment*255).astype(np.uint8)

    def work(j):
        binj, t = triangle_binarize_from_float(segment[j], blur=blur)