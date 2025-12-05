import numpy as np
import psutil
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
            


def bilateral_filter_video_volumetric_cpu(video, wsize, sigma_d, sigma_r, mode="edge"):
    """
    video: np.ndarray, 3D (F, H, W), float32 / float64
    """
    assert video.ndim == 3
    F, H, W = video.shape
    k = wsize // 2

    # Padding
    pad_video = np.pad(video, pad_width=k, mode=mode) # (F+2k, H+2k, W+2k)

    # All local windows: shape = (F, H, W, wsize, wsize)
    patches = np.lib.stride_tricks.sliding_window_view(
        pad_video, # Source image
        (wsize, wsize, wsize) # window size
    ) # patches.shape = (F, H , W, wsize, wsize, wsize)

    ax = np.arange(-k, k+1, dtype=video.dtype)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")

    spatial_kernel = np.exp(-(xx**2+yy**2+zz**2)/(2.0*sigma_d*sigma_d)) # (wsize, wsize, wsize)

    # center: (F, H, W) -> (F, H, W, 1, 1, 1)
    center = video[:, :, :].reshape(F, H, W, 1, 1, 1)

    # diff, r, w:  (F, H, W, wsize, wsize)
    diff = patches - center
    r = np.exp(-(diff**2) / (2.0 * sigma_r * sigma_r))

    # broadcasting：spatial_kernel: (wsize, wsize) -> (1,1,1, wsize, wsize,wsize)
    w = r * spatial_kernel  # (F, H, W, wsize, wsize, wsize)

    # Normalizing 
    w_sum = w.sum(axis=(-1, -2, -3))      # (F, H, W)
    wp = (w * patches).sum(axis=(-1, -2, -3))  # (F, H, W)

    # Avoid division by zero
    eps = 1e-8
    out = wp / (w_sum + eps)

    return out 


def estimate_chunk_size(F, H, W, wsize, dtype=np.float32, safety_factor=0.5):
    """
    Estimate chunk size (frames per chunk) based on available RAM.
    Assumes memory dominated by patches: H * W * wsize^3 * frames.
    """
    available_mem = psutil.virtual_memory().available  # bytes
    bpe = np.dtype(dtype).itemsize
    mem_per_frame = H * W * (wsize ** 3) * bpe
    max_mem_for_chunk = available_mem * safety_factor
    max_frames = max(1, int(max_mem_for_chunk // mem_per_frame))
    return max_frames, available_mem / (1024**3)




def bilateral_filter_video_volumetric_chunked_halo(
    video,
    wsize,
    sigma_d,
    sigma_r,
    mode="edge",
    dtype=np.float32,
    backend="auto",            # "auto" | "numpy" | "cupy"
    safety_factor=0.5,         # fraction of free memory to use for chunks
    overhead_factor=3.5,       # multiplier to account for patches, weights, etc.
    verbose=True,
):
    """
    3D bilateral filter over (F, H, W) with temporal halo chunking and NumPy/CuPy routing.

    Parameters
    ----------
    video : np.ndarray or cp.ndarray
        Input video volume (F, H, W).
    wsize : int
        Odd window size (e.g., 3, 5, 7). Uses a cubic neighborhood of shape (wsize, wsize, wsize).
    sigma_d : float
        Spatial standard deviation (applies equally on F, H, W axes).
    sigma_r : float
        Range (intensity) standard deviation.
    mode : str
        Padding mode for edges ('edge', 'reflect', 'symmetric', 'constant', etc.).
        Must be supported by the chosen backend's pad implementation.
    dtype : np.dtype or cp.dtype
        Computation dtype (e.g., np.float32, np.float16). Will be mapped to backend dtype.
    backend : {"auto", "numpy", "cupy"}
        - "auto": choose CuPy if available (and video is CuPy or user allows); else NumPy.
        - "numpy": force CPU (NumPy).
        - "cupy": force GPU (CuPy). Input will be moved to GPU if needed.
    safety_factor : float
        Fraction of available memory to allocate for chunk processing (0 < sf <= 1).
    overhead_factor : float
        Multiplier to approximate additional temporary buffers (patches, weights, etc.).
    verbose : bool
        Print info about memory and chunk sizes.

    Returns
    -------
    out : np.ndarray or cp.ndarray
        Filtered video in the chosen backend's array type.
    """
    assert isinstance(wsize, int) and wsize % 2 == 1, "wsize must be an odd integer"
    assert video.ndim == 3, "video must have shape (F, H, W)"
    F, H, W = video.shape
    k = wsize // 2

    # ----------------------------
    # Backend router (NumPy / CuPy)
    # ----------------------------
    has_cupy = False
    cp = None
    if backend != "numpy":
        try:
            import cupy as cp  # lazy import
            has_cupy = True
        except Exception:
            has_cupy = False
            cp = None

    # Decide backend
    on_gpu = False
    if backend == "cupy":
        if not has_cupy:
            raise RuntimeError("backend='cupy' requested but CuPy is not available.")
        on_gpu = True
    elif backend == "numpy":
        on_gpu = False
    else:  # auto
        # Prefer GPU if CuPy available and either video is already on GPU or we can move it
        on_gpu = has_cupy and (hasattr(video, "device") or True)

    # Array module alias
    xp = cp if on_gpu else np

    # ----------------------------
    # Ensure array is on chosen backend
    # ----------------------------
    # Map dtype to chosen backend
    def map_dtype(xp_mod, d):
        # accept np.float32, np.float16, strings, etc.
        if isinstance(d, str):
            return getattr(xp_mod, d)
        try:
            # If d is already a dtype from xp_mod, return it
            return xp_mod.dtype(d)
        except Exception:
            # Fallback to float32
            return xp_mod.float32

    dtype_xp = map_dtype(xp, dtype)

    # Move/cast the input video
    if on_gpu:
        video_xp = video if hasattr(video, "device") else cp.asarray(video)
        if video_xp.dtype != dtype_xp:
            video_xp = video_xp.astype(dtype_xp, copy=False)
    else:
        # Ensure NumPy array
        if hasattr(video, "__cuda_array_interface__"):  # CuPy array passed
            if cp is None:
                try:
                    import cupy as cp  # lazy import for conversion
                except Exception as exc:
                    raise RuntimeError("CuPy array provided but CuPy is not available to move data to CPU.") from exc
            video_xp = cp.asnumpy(video).astype(dtype, copy=False)
        else:
            video_xp = np.asarray(video, dtype=dtype)

    # ----------------------------
    # Memory-aware chunk size estimate
    # ----------------------------
    # Get free memory (CPU RAM vs GPU mem)
    if on_gpu:
        # Free GPU memory (bytes)
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        avail_bytes = int(free_bytes)
        avail_gb = avail_bytes / (1024**3)
    else:
        # Free system RAM (bytes)
        try:
            import psutil
            avail_bytes = int(psutil.virtual_memory().available)
        except Exception:
            # Fallback: assume 2 GB free
            avail_bytes = int(2 * 1024**3)
        avail_gb = avail_bytes / (1024**3)

    # Approx memory per frame for the big patches tensor + working buffers
    bytes_per_element = xp.dtype(dtype_xp).itemsize
    mem_per_frame = H * W * (wsize ** 3) * bytes_per_element
    # Account for additional arrays (patches, diff, weights, accumulation)
    mem_per_frame *= overhead_factor

    max_mem_for_chunk = max(1, int(avail_bytes * safety_factor))
    # At least 1 frame
    chunk_size = max(1, int(max_mem_for_chunk // mem_per_frame))

    # Avoid overly large chunks (keep time halos reasonable)
    chunk_size = min(chunk_size, F)  # cannot exceed F
    # Optionally cap for stability on GPU/CPU
    # (tune these caps for your hardware; smaller chunks improve peak mem usage)
    if on_gpu:
        chunk_size = min(chunk_size, 32)
    else:
        chunk_size = min(chunk_size, 128)

    if verbose:
        print(
            f"[bilateral 3D] Backend: {'CuPy (GPU)' if on_gpu else 'NumPy (CPU)'} | "
            f"Free mem: {avail_gb:.2f} GB | Chunk size: {chunk_size} frames | "
            f"Itemsize: {bytes_per_element} B | wsize^3: {wsize**3}"
        )

    # ----------------------------
    # Precompute spatial Gaussian kernel (3D)
    # ----------------------------
    ax = xp.arange(-k, k + 1, dtype=dtype_xp)
    xx, yy, zz = xp.meshgrid(ax, ax, ax, indexing="ij")
    spatial_kernel = xp.exp(-(xx**2 + yy**2 + zz**2) / (xp.array(2.0, dtype=dtype_xp) * (dtype_xp.type(sigma_d)**2)))
    spatial_kernel = spatial_kernel.astype(dtype_xp, copy=False)

    # Output allocation
    out = xp.empty_like(video_xp, dtype=dtype_xp)

    # ----------------------------
    # Chunked processing with temporal halo
    # ----------------------------
    for start in range(0, F, chunk_size):
        end = min(start + chunk_size, F)

        # Temporal halo bounds (clamped)
        t0 = max(0, start - k)
        t1 = min(F, end + k)

        # Halo subvolume
        chunk_halo = video_xp[t0:t1]  # shape: (t1 - t0, H, W)
        # Index range (within chunk_halo) for the frames we actually need to output
        region_start = start - t0
        region_end = region_start + (end - start)

        # Pad temporal axis with the full window radius (k) so every frame in chunk_halo
        # has a complete neighborhood; spatial axes also padded by k.
        pad_width = (
            (k, k),  # temporal
            (k, k),  # H
            (k, k),  # W
        )
        pad_chunk = xp.pad(chunk_halo, pad_width=pad_width, mode=mode)

        # Sliding window view on (F,H,W) with cubic window (wsize,wsize,wsize)
        # NumPy: np.lib.stride_tricks.sliding_window_view
        # CuPy:  cp.lib.stride_tricks.sliding_window_view
        swv = xp.lib.stride_tricks.sliding_window_view
        patches = swv(pad_chunk, (wsize, wsize, wsize))  # shape: (frames_halo, H, W, wsize, wsize, wsize)

        # Center intensities for current output region
        centers = video_xp[start:end].reshape(end - start, H, W, 1, 1, 1)

        # Range (intensity) weights
        diff = patches[region_start:region_end] - centers
        r = xp.exp(-(diff**2) / (xp.array(2.0, dtype=dtype_xp) * (dtype_xp.type(sigma_r)**2)))

        # Combine with spatial kernel (broadcast to last 3 dims)
        w = r * spatial_kernel

        # Normalize
        w_sum = w.sum(axis=(-1, -2, -3))
        wp = (w * patches[region_start:region_end]).sum(axis=(-1, -2, -3))

        eps = xp.array(1e-8, dtype=dtype_xp)
        out[start:end] = (wp / (w_sum + eps)).astype(dtype_xp, copy=False)

        # Encourage GPU memory release between chunks
        if on_gpu:
            # Free intermediate refs explicitly (CuPy uses ref counting)
            del chunk_halo, pad_chunk, patches, centers, diff, r, w, w_sum, wp
            xp.get_default_memory_pool().free_all_blocks()

    # ----------------------------
    # Return in selected backend's type
    # ----------------------------
    return out




# Example quick test
if __name__ == "__main__":
    video = np.random.rand(20, 64, 64).astype(np.float32)
    filtered = bilateral_filter_video_volumetric_chunked_halo(
        video, wsize=5, sigma_d=2.0, sigma_r=0.1, mode="edge", dtype=np.float32
    )
    print("Filtered shape:", filtered.shape)
