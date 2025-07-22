import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

import pycine.file as cine  # Ensure the pycine package is installed
from scipy.ndimage import median_filter as ndi_median_filter
from scipy.ndimage import generic_filter, binary_opening, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk
import gc
from concurrent.futures import as_completed, ThreadPoolExecutor
import sklearn.cluster

# -----------------------------
# Cine video reading and playback
# -----------------------------
def read_frame(cine_file_path, frame_offset, width, height):
    with open(cine_file_path, "rb") as f:
        f.seek(frame_offset)
        frame_data = np.fromfile(f, dtype=np.uint16, count=width * height).reshape(height, width)
    return frame_data

def load_cine_video(cine_file_path):
    # Read the header
    header = cine.read_header(cine_file_path)
    # Extract width, height, and total frame count
    width = header['bitmapinfoheader'].biWidth
    height = header['bitmapinfoheader'].biHeight
    frame_offsets = header['pImage']  # List of frame offsets
    frame_count = len(frame_offsets)
    print(f"Video Info - Width: {width}, Height: {height}, Frames: {frame_count}")

    # Initialize an empty 3D NumPy array to store all frames
    video_data = np.zeros((frame_count, height, width), dtype=np.uint16)
    # Use ThreadPoolExecutor to read frames in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_index = {
            executor.submit(read_frame, cine_file_path, frame_offsets[i], width, height): i
            for i in range(frame_count)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                video_data[index] = future.result()
            except Exception as e:
                print(f"Error reading frame {index}: {e}")
    return video_data

def get_subfolder_names(parent_folder):
    parent_folder = Path(parent_folder)
    subfolder_names = [item.name for item in parent_folder.iterdir() if item.is_dir()]
    return subfolder_names

def play_video_cv2(video, gain=1, binarize=False, thresh=0.5, intv=17):
    """
    Play a list/array of video frames with OpenCV, with optional binarization.

    Parameters
    ----------
    video : sequence of np.ndarray. Int, float or bool
        视频帧列表，每帧可以是整数、浮点数，也可以是布尔数组。
    gain : float, optional. 
        灰度增益，对原始数值做线性放缩（默认 1）。
    binarize : bool, optional
        是否先将帧转换为布尔再显示（默认 False）。
    thresh : float, optional
        当 binarize=True 且输入不是布尔类型时，使用该阈值做二值化（浮点[0,1]或任意范围均可）。
    """
    total_frames = len(video)
    if total_frames == 0:
        return

    # 先检测第 1 帧的数据类型
    first_dtype = video[0].dtype

    for i in range(total_frames):
        frame = video[i]

        # —— 二值化分支 ——
        if binarize:
            # 如果是非布尔类型，先做阈值处理
            if frame.dtype != bool:
                # 假定浮点帧在 [0,1]，或任意数值，都可以用 thresh 来分割
                frame_bool = frame > thresh
            else:
                frame_bool = frame
            # True→255, False→0
            frame_uint8 = (frame_bool.astype(np.uint8)) * 255

        # —— 原有灰度／色阶分支 ——
        else:
            dtype = frame.dtype
            # 整数：假设是 16-bit 量程，缩到 8-bit
            if np.issubdtype(dtype, np.integer):
                frame_uint8 = gain * (frame / 16).astype(np.uint8)
            # 浮点：假设在 [0,1]，放大到 0–255
            elif np.issubdtype(dtype, np.floating):
                frame_uint8 = np.clip(gain * (frame * 255), 0, 255).astype(np.uint8)
            # 其他类型回退到整数缩放
            else:
                frame_uint8 = gain * (frame / 16).astype(np.uint8)

        # 显示
        cv2.imshow('Frame', frame_uint8)
        # ~60fps 播放，按 'q' 退出
        if cv2.waitKey(intv) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def play_videos_side_by_side(videos, gain=1, binarize=False, thresh=0.5, intv=17):
    """Play multiple videos side by side using OpenCV.

    Parameters
    ----------
    videos : sequence of np.ndarray
        Sequence of videos, each shaped ``(frame, x, y)``.
    gain, binarize, thresh, intv : see :func:`play_video_cv2`.
    """
    if not videos:
        return

    total_frames = min(len(v) for v in videos)
    if total_frames == 0:
        return

    for i in range(total_frames):
        frame = np.hstack([v[i] for v in videos])

        if binarize:
            if frame.dtype != bool:
                frame_bool = frame > thresh
            else:
                frame_bool = frame
            frame_uint8 = frame_bool.astype(np.uint8) * 255
        else:
            dtype = frame.dtype
            if np.issubdtype(dtype, np.integer):
                frame_uint8 = gain * (frame / 16).astype(np.uint8)
            elif np.issubdtype(dtype, np.floating):
                frame_uint8 = np.clip(gain * (frame * 255), 0, 255).astype(np.uint8)
            else:
                frame_uint8 = gain * (frame / 16).astype(np.uint8)

        cv2.imshow('Frame', frame_uint8)
        if cv2.waitKey(intv) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# -----------------------------
# Rotation and Filtering functions
# -----------------------------
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if frame.dtype == np.bool_:
        # Convert boolean mask to uint8: True becomes 255, False becomes 0.
        frame_uint8 = (frame.astype(np.uint8)) * 255
        # Use INTER_NEAREST to preserve mask values.
        rotated_uint8 = cv2.warpAffine(frame_uint8, M, (w, h), flags=cv2.INTER_NEAREST)
        # Convert back to boolean mask.
        rotated = rotated_uint8 > 127 
    else:
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return rotated

def rotate_video(video_array, angle=0, max_workers=None):
    num_frames = video_array.shape[0]
    rotated_frames = [None] * num_frames
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(rotate_frame, video_array[i], angle): i 
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                rotated_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during rotation: {exc}")
    return np.array(rotated_frames)

# -----------------------------
# CUDA-accelerated Rotation
# -----------------------------
def is_opencv_cuda_available():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except AttributeError:
        return False
    
def rotate_frame_cuda(frame, angle, stream=None):
    """
    在 GPU 上旋转单帧图像／掩码。
    
    Parameters
    ----------
    frame : np.ndarray (H×W or H×W×C) or bool mask
    angle : float
    stream: cv2.cuda.Stream (optional) — 用于异步操作
    
    Returns
    -------
    rotated : 同 frame 类型
    """
    h, w = frame.shape[:2]
    # 计算仿射矩阵（在 CPU 上）
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
    
    # 上传到 GPU
    if frame.dtype == np.bool_:
        # 布尔先转 uint8（0/255）
        cpu_uint8 = (frame.astype(np.uint8)) * 255
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(cpu_uint8, stream)
        # warpAffine（最近邻保留掩码边界）
        gpu_rot = cv2.cuda.warpAffine(
            gpu_mat, M, (w, h),
            flags=cv2.INTER_NEAREST, stream=stream
        )
        # 下载并阈值回布尔
        out_uint8 = gpu_rot.download(stream)
        rotated = out_uint8 > 127
    else:
        # 对普通灰度或多通道图像
        gpu_mat = cv2.cuda.GpuMat()

        if stream is not None:
            gpu_mat.upload(frame, stream)
            gpu_rot = cv2.cuda.warpAffine(
                gpu_mat, M, (w, h),
                flags=cv2.INTER_CUBIC, stream=stream
            )
            rotated = gpu_rot.download(stream)
        else:
            gpu_mat.upload(frame)
            gpu_rot = cv2.cuda.warpAffine(
                gpu_mat, M, (w, h),
                flags=cv2.INTER_CUBIC
            )
            rotated = gpu_rot.download()
    
    # 等待 GPU 流完成
    if stream is not None:
        stream.waitForCompletion()
    return rotated

def rotate_video_cuda(video_array, angle=0, max_workers=4):
    """
    并行地在 GPU 上旋转整个视频（每帧独立流）。
    
    Parameters
    ----------
    video_array : np.ndarray, shape=(N, H, W) 或 (N, H, W, C) 或 bool
    angle : float — 旋转角度（度）
    max_workers : int — 并行线程数（每线程管理一个 cv2.cuda.Stream）
    
    Returns
    -------
    np.ndarray — 与输入同形状、同 dtype
    """
    num_frames = video_array.shape[0]
    rotated = [None] * num_frames

    # 预创建若干 CUDA 流
    streams = [cv2.cuda.Stream() for _ in range(max_workers)]

    def task(idx, frame):
        # 分配一个流（简单轮询）
        stream = streams[idx % max_workers]
        return idx, rotate_frame_cuda(frame, angle, stream)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(task, i, video_array[i]) for i in range(num_frames)]
        for fut in as_completed(futures):
            idx, out = fut.result()
            rotated[idx] = out

    return np.stack(rotated, axis=0)

def rotate_video_auto(video_array, angle=0, max_workers=4):
    if is_opencv_cuda_available():
        print("Using CUDA for rotation.")
        return rotate_video_cuda(video_array, angle=angle, max_workers=max_workers)
    else:
        print("CUDA not available, falling back to CPU.")
        return rotate_video(video_array, angle=angle, max_workers=max_workers)
    
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

def map_video_to_range(video):
    """
    Maps a video to a 2D image of its pixel intensity ranges.
    """
    # Assuming video is a 3D numpy array (frames, height, width)
    # Calculate the min and max for each pixel across all frames
    min_vals = np.min(video, axis=0)
    max_vals = np.max(video, axis=0)

    # Create a 2D image where each pixel's value is the range
    range_map = abs(max_vals - min_vals)

    # Normalize the range map to [0, 1] for visualization
    # range_map_normalized = (range_map - np.min(range_map)) / (np.max(range_map) - np.min(range_map))

    # return range_map_normalized

    return range_map

def imhist(image, bins=1000, log=False, exclude_zero=False):
    """
    Plot histogram (and implicitly CDF via cumulated counts if desired) of image data.
    
    Parameters
    ----------
    image : array-like
        Input image values expected in [0, 1].
    bins : int
        Number of histogram bins.
    log : bool
        If True, use logarithmic y-axis.
    exclude_zero : bool
        If True, filter out zero-valued pixels before computing histogram.
    """
    # Flatten image
    data = image.ravel()
    if exclude_zero:
        data = data[data != 0]
    
    hist, edges = np.histogram(data, bins=bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots()
    ax.plot(centers, hist, lw=1.2)
    if log:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)  # avoid log(0) issues
    ax.set_xlabel("Range value")
    ax.set_ylabel("Count" + (" (log scale)" if log else ""))
    ax.set_title("Histogram of image" + (" (zeros excluded)" if exclude_zero else ""))
    ax.grid(True, which='both', ls='--', alpha=0.3)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def video_histogram_with_contour(video, bins=100, exclude_zero=False, log=False):
    """
    Compute per-frame histograms for a video and display both
    a heatmap and a contour plot in a shared figure.

    Parameters
    ----------
    video : np.ndarray
        Grayscale video of shape (frames, height, width), values in [0, 1].
    bins : int
        Number of intensity bins.
    exclude_zero : bool
        If True, omit zero-valued pixels.
    log : bool
        If True, take log(1 + counts).
    """
    frames, h, w = video.shape

    # 1) Build sample pairs (frame_idx, intensity)
    frame_idx = np.repeat(np.arange(frames), h * w)
    intensities = video.ravel()
    if exclude_zero:
        mask = (intensities != 0)
        frame_idx = frame_idx[mask]
        intensities = intensities[mask]

    samples = np.stack((frame_idx, intensities), axis=1)

    # 2) Compute the 2D histogram
    hist, edges = np.histogramdd(
        samples,
        bins=(frames, bins),
        range=((0, frames), (0.0, 1.0))
    )
    # edges[0] has length frames+1, edges[1] has length bins+1

    if log:
        hist = np.log1p(hist)

    # 3) Compute true bin centers for both dimensions
    frame_edges, intensity_edges = edges
    frame_centers = (frame_edges[:-1] + frame_edges[1:]) / 2    # length = frames
    bin_centers   = (intensity_edges[:-1] + intensity_edges[1:]) / 2  # length = bins

    # 4) Build meshgrid so Z==hist has shape (frames, bins)
    X, Y = np.meshgrid(bin_centers, frame_centers)

    # 5) Plot heatmap + contour
    fig, (ax_heat, ax_contour) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True
    )

    # Heatmap
    im = ax_heat.imshow(
        hist,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, 1, 0, frames]
    )
    fig.colorbar(im, ax=ax_heat, label="Log Count" if log else "Count")
    ax_heat.set_ylabel("Frame index")
    ax_heat.set_title("Histogram Heatmap" + (" (zeros excluded)" if exclude_zero else ""))

    # Contour
    cont = ax_contour.contourf(
        X, Y, hist,
        levels=15,
        cmap='viridis'
    )
    fig.colorbar(cont, ax=ax_contour, label="Log Count" if log else "Count")
    ax_contour.set_xlabel("Intensity (normalized 0→1)")
    ax_contour.set_ylabel("Frame index")
    ax_contour.set_title("Histogram Contour" + (" (zeros excluded)" if exclude_zero else ""))

    plt.tight_layout()
    plt.show()

def subtract_median_background(video, frame_range=None):
    """
    Subtract a background image from each frame of a video.
    
    Parameters
    ----------
    video : np.ndarray
        Video frames as a 3D array (N, H, W).

    Returns
    -------
    np.ndarray
        Background-subtracted video.

    Example usage:
        slice object recommended in Python, 
        foreground = subtract_median_background(video, frame_range=slice(0, 30))
    """
    if video.ndim != 3:
        raise ValueError("Video must be 3D (N, H, W).")
    if frame_range is None:
        background = np.median(video[:, :, :], axis=0)
    else:
        background = np.median(video[frame_range, :, :], axis=0) 
    return video  - background[None, :, :]


def kmeans_label_video(video: np.ndarray, k: int) -> np.ndarray:
    """Label pixels into ``k`` brightness clusters using k-means.

    Parameters
    ----------
    video:
        Input video with shape ``(frame, x, y)``.
    k:
        Number of clusters.

    Returns
    -------
    np.ndarray
        Video of integer labels with the same shape as ``video``.
    """
    orig_shape = video.shape
    flat = video.reshape(-1, 1).astype(float)

    kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init='auto', random_state=0)
    kmeans.fit(flat)

    centers = kmeans.cluster_centers_.ravel()
    order = np.argsort(centers)

    mapping = np.empty_like(order)
    mapping[order] = np.arange(k)
    labels = mapping[kmeans.labels_]

    return labels.reshape(orig_shape)


def labels_to_playable_video(labels: np.ndarray, k: int) -> np.ndarray:
    """Convert k-means labels to a float video in ``[0, 1]`` for display."""
    if k <= 1:
        return labels.astype(float)
    return labels.astype(float) / float(k - 1)