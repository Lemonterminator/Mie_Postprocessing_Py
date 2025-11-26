import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def bilateral_filter_img(img, wsize, sigma_d, sigma_r):
    nrows, ncols = img.shape
    output = np.zeros([nrows, ncols])
    
    k = wsize//2
    half = k
    
    ax = np.arange(-k, k+1)
    xx, yy = np.meshgrid(ax,ax,indexing="ij")
    spatial_kernel = np.exp(-(xx**2+yy**2)/(2.0*sigma_d**2))
    
                   
    for i in range(nrows):
        # Calculate local region limits
        iMin = int(max(i - k, 0))
        iMax = int(min(i + k, nrows - 1))
        # Offsets to crop the precomputed spatial kernel at borders
        siMin = iMin - (i - half)
        siMax = siMin + (iMax - iMin + 1)
        
        for j in range(ncols):
            jMin = int(max(j - k, 0))
            jMax = int(min(j + k, ncols - 1))
            sjMin = jMin - (j-half)
            sjMax = sjMin + (jMax - jMin + 1)
            

            # Use the region limits to extract a local patch from the image,
            # calculate the median value and store it at the correct 
            # index in the output.
            window = img[iMin:iMax+1, jMin:jMax+1]
            s = spatial_kernel[siMin:siMax, sjMin:sjMax]
            
            center = img[i,j]
            diff = window -center
            r = np.exp(-(diff**2)/(2.0*sigma_r**2))
                       
            # Bilateral weights 
            w = s*r
            w_sum = w.sum()
            if w_sum >0:
                output[i,j] = np.sum(window*w)/w_sum
            else:
                output[i,j] = center
    return output

def bilateral_filter_img_cupy(img, wsize, sigma_d, sigma_r, mode="edge"):
    import cupy as cp
    """
    img: cp.ndarray, 2D (H, W), float32 / float64
    """
    assert img.ndim == 2
    H, W = img.shape
    k = wsize // 2

    # Padding
    pad_img = cp.pad(img, pad_width=k, mode=mode) # (H+2k, W+2k)

    # All local windows: shape = (H, W, wsize, wsize)
    patches = cp.lib.stride_tricks.sliding_window_view(
        pad_img, # Source image
        (wsize, wsize) # window size
    ) # patches.shape = (H, W, wsize, wsize)

    # pre-computing spatial kernel

    ax = cp.arange(-k, k+1, dtype=img.dtype)
    xx, yy = cp.meshgrid(ax, ax, indexing="ij")

    spatial_kernel = cp.exp(-(xx**2+yy**2)/(2.0*sigma_d*sigma_d)) # (wsize, wsize)

    # range kernel that depends on the center pixel

    # center: (H, W) -> (H, W, 1, 1)
    center = img[:, :].reshape(H, W, 1, 1)

    # diff, r, w:  (H, W, wsize, wsize)
    diff = patches - center
    r = cp.exp(-(diff**2) / (2.0 * sigma_r * sigma_r))
    
    # broadcasting：spatial_kernel: (wsize, wsize) -> (1,1,wsize,wsize)
    w = r * spatial_kernel  # (H, W, wsize, wsize)

    # Normalizing 
    w_sum = w.sum(axis=(-1, -2))      # (H, W)
    wp = (w * patches).sum(axis=(-1, -2))  # (H, W)

    # Avoid division by zero
    eps = 1e-8
    out = wp / (w_sum + eps)

    return out

def bilateral_filter_video_cpu(video, wsize, sigma_d, sigma_r,max_workers=None):
    
    F, H, W = video.shape
    result = np.empty_like(video)

    def work(f_idx):
        img = video[f_idx]
        filtered = cv2.bilateralFilter(
            img,
            d=wsize,
            sigmaColor=sigma_r,
            sigmaSpace=sigma_d
        )
        return f_idx, filtered
    
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(work, f) for f in range(F)]
        for future in as_completed(futures):
            f_idx, filtered = future.result()
            result[f_idx] = filtered
    return result  
            
def bilateral_filter_video_cupy(video, wsize, sigma_d, sigma_r):
    import cupy as cp
    """
    video: np.ndarray or cp.ndarray, shape = (F, H, W)
    return: cp.ndarray, same shape
    """
    # 1) 统一搬到 GPU
    video_gpu = cp.asarray(video, dtype=cp.float32)
    F, H, W = video_gpu.shape

    result = cp.empty_like(video_gpu)

    for f in range(F):
        result[f] = bilateral_filter_img_cupy(
            video_gpu[f], wsize, sigma_d, sigma_r
        )

    return result
            

    