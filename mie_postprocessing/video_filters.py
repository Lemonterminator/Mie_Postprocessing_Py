import numpy as np
from scipy.signal import fftconvolve
from concurrent.futures import as_completed, ProcessPoolExecutor
from scipy.ndimage import median_filter as ndi_median_filter

# Optional GPU acceleration via CuPy.
# Fall back to NumPy/SciPy on machines without CUDA (e.g. laptops).
try:  # pragma: no cover - runtime hardware dependent
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import median_filter  # type: ignore
    CUPY_AVAILABLE = True
except Exception:  # ImportError, CUDA failure, etc.
    cp = np  # type: ignore
    from scipy.ndimage import median_filter  # type: ignore
    cp.asnumpy = lambda x: x  # type: ignore[attr-defined]
    CUPY_AVAILABLE = False



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
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(medfilt, video_array[i], M, N): i
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                filtered_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during median filtering: {exc}")
    return np.array(filtered_frames)

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

