import numpy as np
from scipy.signal import fftconvolve
from concurrent.futures import as_completed, ProcessPoolExecutor
from scipy.ndimage import median_filter as ndi_median_filter
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Optional GPU acceleration via CuPy.
# Fall back to NumPy/SciPy on machines without CUDA (e.g. laptops).
try:  # pragma: no cover - runtime hardware dependent
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA path could not be detected.*",
            category=UserWarning,
        )
        import cupy as cp  # type: ignore
        from cupyx.scipy.ndimage import median_filter  # type: ignore
    CUPY_AVAILABLE = True
except Exception:  # ImportError, CUDA failure, etc.
    cp = np  # type: ignore
    from scipy.ndimage import median_filter  # type: ignore
    cp.asnumpy = lambda x: x  # type: ignore[attr-defined]
    CUPY_AVAILABLE = False

def sobel_5x5_kernels() -> tuple[np.ndarray, np.ndarray]:
    # 1D derivative and smoothing
    d = np.array([1, 2, 0, -2, -1], dtype=np.float32)   # horizontal diff
    s = np.array([1, 4, 6, 4, 1], dtype=np.float32)     # vertical smoothing

    # X: smooth vertically, diff horizontally
    sobel_x = s[:, None] * d[None, :]    # (5, 5)

    # Y: diff vertically, smooth horizontally
    sobel_y = d[:, None] * s[None, :]    # (5, 5)

    return sobel_x, sobel_y

def filter_video_fft(video: np.ndarray, kernel: np.ndarray, mode='same') -> np.ndarray:
    """
    Convolve each frame in `video` with `kernel` via FFT convolution,
    in one call without an explicit Python loop.

    Parameters
    ----------
    video : np.ndarray
        Input video as a 3D array, shape (n_frames, H, W).
    kernel : np.ndarray
        2D convolution kernel, shape (kH, kW).
    mode : {'full', 'valid', 'same'}
        Convolution mode passed to fftconvolve.

    Returns
    -------
    np.ndarray
        Filtered video, same shape as input if mode='same'.
    """
    # fftconvolve will broadcast the first dimension (frames)
    # but we must explicitly tell it which axes are spatial:
    return fftconvolve(video, kernel[np.newaxis, :, :], mode=mode, axes=(1, 2))

def median_filter_video_cuda(video_array: np.ndarray, M: int, N: int, T:int=1) -> np.ndarray:
    """
    GPU-accelerated spatial median filter per frame: applies MxN median
    on each (H,W) frame of video shaped (T, H, W) in one shot.
    """
    from cupy import asnumpy 
    # Move entire video once to GPU
    video_gpu = cp.asarray(video_array)  # shape (T, H, W)
    # Apply median filter with no temporal smoothing: size=(1, M, N)
    filtered_gpu = median_filter(
        video_gpu,
        size=(T, M, N),
        mode='constant',
        cval=0.0
    )
    # Bring result back once
    return asnumpy(filtered_gpu)

def medfilt(image, M, N):
    return ndi_median_filter(image, size=(M, N), mode='constant', cval=0)

def median_filter_video(video_array, M, N, max_workers=None):
    num_frames = video_array.shape[0]
    filtered_frames = [None] * num_frames
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(medfilt, video_array[i], M, N): i
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                filtered_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during median filtering: {exc}")
    return np.array(filtered_frames)

def median_filter_video_cv2(video_array, ksize=3, max_workers=None):
    # video_array: (T, H, W) or (T, H, W, C), uint8/uint16/float32 -> OpenCV prefers uint8/uint16
    T = video_array.shape[0]
    out = np.empty_like(video_array)

    def work(i):
        frame = video_array[i]
        if frame.dtype == np.float32:  # OpenCV supports this too; uint8/uint16 are fastest
            return cv2.medianBlur(frame, ksize)
        # Ensure C-contiguous for OpenCV
        return cv2.medianBlur(np.ascontiguousarray(frame), ksize)

    # OpenCV’s C code releases the GIL -> threads scale well; avoid process pickling overhead
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work, i): i for i in range(T)}
        for f in as_completed(futures):
            i = futures[f]
            out[i] = f.result()
    return out

def median_filter_video_auto(video_array, M, N, T=1, max_workers=None):
    """Apply median filtering using GPU when available.

    Falls back to a CPU implementation when CuPy or a GPU is not
    present. ``T`` controls the temporal window for the GPU
    implementation.
    """
    if CUPY_AVAILABLE:
        try:
            return median_filter_video_cuda(video_array, M, N, T=T)
        except Exception as exc:  # pragma: no cover - runtime hardware dependent
            print(f"GPU median filtering failed ({exc}), falling back to CPU.")
    # CPU fallback
    # If M and N are equal (square window), prefer fast OpenCV medianBlur which
    # accepts a single odd kernel size `ksize`. Otherwise, fall back to SciPy's
    # rectangular median filter implementation.
    if M == N:
        ksize = int(M)
        # OpenCV requires ksize to be an odd integer >= 3
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        return median_filter_video_cv2(video_array, ksize=ksize, max_workers=max_workers)
    else:
        return median_filter_video(video_array, M, N, max_workers=max_workers)

def gaussian_low_pass_filter(img, cutoff):
    rows, cols = img.shape
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u - cols // 2
    v = v - rows // 2
    H = np.exp(-(u**2 + v**2) / (2 * cutoff**2))
    img_fft = np.fft.fft2(img.astype(np.float64))
    img_fft_shifted = np.fft.fftshift(img_fft)
    filtered_fft_shifted = img_fft_shifted * H
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
    filtered_img = np.fft.ifft2(filtered_fft)
    return np.abs(filtered_img)

def Gaussian_LP_video(video_array, cutoff, max_workers=None):
    num_frames = video_array.shape[0]
    filtered_frames = [None] * num_frames
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(gaussian_low_pass_filter, video_array[i], cutoff): i 
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                filtered_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during Gaussian LP filtering: {exc}")
    return np.array(filtered_frames)


def gaussian_video_cpu(video_f32: np.ndarray, ksize=(5,5), sigma=0, max_workers=None):
    """
    video_f32: np.ndarray, shape (T,H,W), dtype float32
    """
    assert video_f32.ndim == 3 and video_f32.dtype == np.float32
    T, H, W = video_f32.shape
    out = np.empty_like(video_f32)

    # Option A: let OpenCV use internal threads, run single-threaded at Python level:
    # cv2.setNumThreads(0)  # 0 = use all cores (default)

    # Option B: parallelize over frames with Python threads; then cap OpenCV threads to avoid oversubscription:
    if max_workers is not None:
        cv2.setNumThreads(1)  # important when using many Python threads
        def work(j):
            out[j] = cv2.GaussianBlur(video_f32[j], ksize, sigma, borderType=cv2.BORDER_REPLICATE)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(work, range(T)))
    else:
        # single Python loop; OpenCV can multithread internally
        for j in range(T):
            out[j] = cv2.GaussianBlur(video_f32[j], ksize, sigma, borderType=cv2.BORDER_REPLICATE)

    return out

