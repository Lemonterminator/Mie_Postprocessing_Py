import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.morphology import diamond, octagon


# ---------- Helper: build structuring element ----------
def _build_kernel(kernel_shape="Ellipse", kernel_size=(3, 3)):
    """
    Returns a structuring element suitable for OpenCV morphology functions.
    For Diamond/Octagon, uses skimage and converts to uint8.
    """
    if kernel_shape == "Ellipse":
        shape = cv2.MORPH_ELLIPSE
        kernel = cv2.getStructuringElement(shape, kernel_size)
    elif kernel_shape == "Rectangle":
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, kernel_size)
    elif kernel_shape == "Cross":
        shape = cv2.MORPH_CROSS
        kernel = cv2.getStructuringElement(shape, kernel_size)
    elif kernel_shape == "Diamond":
        k = diamond(kernel_size[0])       # boolean
        kernel = k.astype(np.uint8)
    elif kernel_shape == "Octagon":
        # Use same radius for both minor/major for simplicity
        k = octagon(kernel_size[0], kernel_size[0])  # boolean
        kernel = k.astype(np.uint8)
    else:
        raise ValueError("Unsupported kernel_shape. Choose from Ellipse, Rectangle, Cross, Diamond, Octagon.")
    return kernel


# ---------- Generic parallel morphology runner ----------
def _morph_video_parallel(video, op, kernel_shape="Ellipse", kernel_size=(3, 3), max_workers=None, iterations=1):
    """
    Apply a cv2 morphology operation `op` to all frames in `video` in parallel.

    Parameters
    ----------
    video : np.ndarray
        (T, H, W) or (T, H, W, C).
    op : callable
        A function(frame, kernel) -> output_frame. E.g., lambda f,k: cv2.dilate(f, k, iterations=iterations).
        For morphologyEx-based ops, can be lambda f,k: cv2.morphologyEx(f, cv2.MORPH_OPEN, k, iterations=iterations).
    kernel_shape : str
        Ellipse | Rectangle | Cross | Diamond | Octagon.
    kernel_size : tuple(int,int)
        For skimage-based kernels, kernel_size[0] is used as radius/extent.
    max_workers : int or None
        Defaults to os.cpu_count().
    iterations : int
        Iterations for the morphology op.

    Returns
    -------
    out : np.ndarray
        Same shape/dtype as the possibly-converted input (keeps float32 as per caller).
    """

    if video.dtype != np.float32:
        video = video.astype(np.float32)

    if video.ndim not in (3, 4):
        raise ValueError("`video` must have shape (T,H,W) or (T,H,W,C).")

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    kernel = _build_kernel(kernel_shape, kernel_size)
    out = np.empty_like(video)

    def _one(idx: int):
        frame = np.ascontiguousarray(video[idx])
        res = op(frame, kernel)
        return idx, res

    T = video.shape[0]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, i) for i in range(T)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            out[idx] = res

    return out


# ---------- Public API: Dilation, Opening, Closing ----------
def dilate_video_parallel(video, kernel_shape="Ellipse", kernel_size=(3, 3), iterations=1, max_workers=None):
    """
    Parallel dilation with cv2.dilate.
    """
    return _morph_video_parallel(
        video,
        op=lambda f, k: cv2.dilate(f, k, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        max_workers=max_workers,
        iterations=iterations
    )

def erode_video_parallel(video, kernel_shape="Ellipse", kernel_size=(3, 3), iterations=1, max_workers=None):
    """
    Parallel dilation with cv2.dilate.
    """
    return _morph_video_parallel(
        video,
        op=lambda f, k: cv2.erode(f, k, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        max_workers=max_workers,
        iterations=iterations
    )


def open_video_parallel(video, kernel_shape="Ellipse", kernel_size=(3, 3), iterations=1, max_workers=None):
    """
    Parallel opening (erode -> dilate) with cv2.morphologyEx(..., MORPH_OPEN, ...).
    """
    return _morph_video_parallel(
        video,
        op=lambda f, k: cv2.morphologyEx(f, cv2.MORPH_OPEN, k, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        max_workers=max_workers,
        iterations=iterations
    )


def close_video_parallel(video, kernel_shape="Ellipse", kernel_size=(3, 3), iterations=1, max_workers=None):
    """
    Parallel closing (dilate -> erode) with cv2.morphologyEx(..., MORPH_CLOSE, ...).
    """
    return _morph_video_parallel(
        video,
        op=lambda f, k: cv2.morphologyEx(f, cv2.MORPH_CLOSE, k, iterations=iterations),
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        max_workers=max_workers,
        iterations=iterations
    )
