import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

def max_pooling(image, pool_size):
    """
    Perform max pooling on a 2D image using non-overlapping windows of size pool_size.
    If the image dimensions are not divisible by pool_size, pad the image with edge values.
    
    Parameters:
        image (np.ndarray): 2D input image.
        pool_size (int): Pooling factor.
    
    Returns:
        np.ndarray: Pooled image.
    """
    H, W = image.shape
    # Compute new dimensions using ceiling division.
    new_H = int(np.ceil(H / pool_size))
    new_W = int(np.ceil(W / pool_size))
    # Determine the required padding so that the image dimensions become divisible by pool_size.
    pad_H = new_H * pool_size - H
    pad_W = new_W * pool_size - W
    # Pad the image with edge values.
    padded = np.pad(image, ((0, pad_H), (0, pad_W)), mode='edge')
    pooled = padded.reshape(new_H, pool_size, new_W, pool_size).max(axis=(1, 3))
    return pooled

def upsample(image, pool_size):
    """
    Upsample a 2D image by repeating each element pool_size times in both dimensions.
    
    Parameters:
        image (np.ndarray): 2D input image.
        pool_size (int): Upsampling factor.
    
    Returns:
        np.ndarray: Upsampled image.
    """
    return np.repeat(np.repeat(image, pool_size, axis=0), pool_size, axis=1)

def local_std_integral(frame, std_size):
    """
    Compute the local standard deviation over a (2*std_size+1)x(2*std_size+1) window
    using an integral image approach. No padding is applied during computation,
    and the result is computed for the valid region only. Afterwards, the valid region
    is padded to match the original frame size.
    
    Parameters:
        frame (np.ndarray): 2D image in float32.
        std_size (int): Half-window size.
    
    Returns:
        np.ndarray: Standard deviation image of shape (H, W), same as the original frame.
    """
    k = 2 * std_size + 1  # full window size
    # Precompute squared image.
    I = frame
    I2 = frame * frame
    
    # Compute integral images with an extra row/column of zeros.
    S = np.pad(np.cumsum(np.cumsum(I, axis=0), axis=1), ((1, 0), (1, 0)), mode='constant', constant_values=0)
    S2 = np.pad(np.cumsum(np.cumsum(I2, axis=0), axis=1), ((1, 0), (1, 0)), mode='constant', constant_values=0)
    
    H, W = frame.shape
    # Valid region dimensions.
    out_H = H - k + 1
    out_W = W - k + 1

    # Compute the sum and squared sum over each k x k window.
    sum_window = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
    sum2_window = S2[k:, k:] - S2[:-k, k:] - S2[k:, :-k] + S2[:-k, :-k]
    
    area = k * k
    mean = sum_window / area
    var = sum2_window / area - mean * mean
    var = np.clip(var, 0, None)
    std = np.sqrt(var)
    
    # Clean up intermediate variables.
    del S, S2, sum_window, sum2_window, I2
    gc.collect()
    
    # Pad the valid-region result to match the original frame dimensions.
    std_padded = np.pad(std, ((std_size, std_size), (std_size, std_size)), mode='edge')
    return std_padded

def process_frame_std_optimized(frame, std_size, pool_size):
    """
    Process one frame:
      1. Convert to float32.
      2. Compute the local standard deviation via an integral image approach.
      3. Apply max pooling (to reduce resolution and memory overhead).
      4. Immediately upsample the pooled result.
    
    The final result is in float32 and is cropped to match the original frame dimensions.
    
    Parameters:
        frame (np.ndarray): Input 2D frame (uint16 expected).
        std_size (int): Half-window size for local std computation.
        pool_size (int): Pooling factor.
    
    Returns:
        np.ndarray: Processed frame (float32) with the same dimensions as the original frame.
    """
    # Step 1: Convert to float32.
    frame_f = frame.astype(np.float32)
    
    # Step 2: Compute local standard deviation (returns an image of the same size as frame).
    std_image = local_std_integral(frame_f, std_size)
    
    # Step 3: Pipeline max pooling on the std image.
    pooled = max_pooling(std_image, pool_size)
    
    # Step 4: Upsample the pooled result back.
    upsampled = upsample(pooled, pool_size)
    # Crop the upsampled image to exactly match std_image dimensions.
    H_valid, W_valid = std_image.shape
    upsampled = upsampled[:H_valid, :W_valid]
    
    del frame_f, std_image, pooled
    gc.collect()
    return upsampled

def stdfilt_video_parallel_optimized(video, std_size, pool_size, max_workers=None):
    """
    Process a video (3D numpy array) in parallel. Each frame is processed using
    an optimized pipeline that computes the local standard deviation using an
    integral image approach (without padding during filtering), followed by max pooling and upsampling.
    
    The output video will have the same frame dimensions as the input video.
    
    Parameters:
        video (np.ndarray): Input video with shape (num_frames, height, width) in uint16.
        std_size (int): Half-window size for the standard deviation filter.
        pool_size (int): Pooling factor.
        max_workers (int, optional): Number of parallel worker processes.
    
    Returns:
        np.ndarray: Processed video with shape (num_frames, H, W) in float32.
    """
    num_frames, H, W = video.shape
    video_out = np.empty((num_frames, H, W), dtype=np.float32)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_frame_std_optimized, video[i], std_size, pool_size): i
            for i in range(num_frames)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                video_out[idx] = future.result()
            except Exception as exc:
                print(f'Frame {idx} processing generated an exception: {exc}')
    
    return video_out
