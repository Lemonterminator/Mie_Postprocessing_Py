import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import concurrent.futures

from concurrent.futures import as_completed, ProcessPoolExecutor
import pycine.file as cine  # Ensure the pycine package is installed
# from scipy.ndimage import median_filter as ndi_median_filter
# from scipy.ndimage import generic_filter, binary_opening, binary_fill_holes
# from skimage.filters import threshold_otsu
# from skimage.morphology import disk

import sklearn.cluster



# -----------------------------
# Cine video reading and playback
# -----------------------------
def read_frame(cine_file_path, frame_offset, width, height):
    with open(cine_file_path, "rb") as f:
        f.seek(frame_offset)
        frame_data = np.fromfile(f, dtype=np.uint16, count=width * height).reshape(height, width)
    return frame_data

def load_cine_video(cine_file_path, frame_limit=None):
    # Read the header
    header = cine.read_header(cine_file_path)
    # Extract width, height, and total frame count
    width = header['bitmapinfoheader'].biWidth
    height = header['bitmapinfoheader'].biHeight
    frame_offsets = header['pImage']  # List of frame offsets
    frame_count = len(frame_offsets)
    frame_count= min(frame_count, frame_limit) if frame_limit else frame_count
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

def find_larger_than_percentile(image, percentile, bins=4096):
    """
    Plot histogram (and implicitly CDF via cumulated counts if desired) of image data.
    
    Parameters
    ----------
    image : array-like
        Input image values expected in [0, 1].
    bins : int
        Number of histogram bins.
    percentile : float
        Percentile threshold (0-100) to filter pixels.
    """
    assert 0 <= percentile <= 100, "Percentile must be between 0 and 100."
    # Flatten image
    data = image.ravel()
    pixels = data.shape[0]

    hist, edges = np.histogram(data, bins=bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2


    acc = 0
    target = round(percentile*pixels/100.0)
    
    for i in range(0, bins):
        acc += hist[i]
        if acc > target:
            # print(centers[i])
            return centers[i]
    
    return 1.0  # If no pixel exceeds the percentile, return max value (1.0) 

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
    return fig, (ax_heat, ax_contour)

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
    return video  - background[None, :, :], background 

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


# -----------------------------
# Time-Distance Map and Area Calculation
# -----------------------------
def calculate_TD_map(horizontal_video: np.ndarray):
    num_frames, height, width = horizontal_video.shape
    time_distance_map = np.zeros((width, num_frames), dtype=np.float32)
    for n in range(num_frames):
        time_distance_map[:, n] = np.sum(horizontal_video[n], axis=0)
    return time_distance_map
