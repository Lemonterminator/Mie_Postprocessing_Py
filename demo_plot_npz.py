from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from mie_postprocessing.functions_videos import *
from mie_postprocessing.cone_angle import plot_angle_signal_density
from scipy import ndimage

def keep_largest_component(binary_mask, connectivity=2):
    """
    Keep only the largest connected component of 1s in a 2D binary array.

    Parameters
    ----------
    binary_mask : array-like of {0,1}, shape (H, W)
    connectivity : int, 1 or 2
        1 -> 4-connectivity; 2 -> 8-connectivity (MATLAB default is 8).

    Returns
    -------
    largest : ndarray of same shape, dtype same as input
        Binary mask with only the largest component preserved.
    """
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if connectivity == 1:
        # 4-connectivity structure
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]], dtype=bool)
    else:
        # 8-connectivity structure
        structure = np.ones((3, 3), dtype=bool)

    labeled, num_features = ndimage.label(binary_mask, structure=structure)
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=binary_mask.dtype)

    # count pixels in each label (0 is background)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background
    largest_label = counts.argmax()
    largest = (labeled == largest_label)
    return largest.astype(binary_mask.dtype)

def penetration_bw_to_index(bw):
    arr = bw.astype(bool)
    # Find where True elements exist
    any_true = arr.any(axis=1)  # shape (N,)
    # Reverse each row to find last occurrence efficiently
    rev_idx = arr.shape[1] - 1 - arr[:, ::-1].argmax(axis=1)
    # Mask rows with no True values
    rev_idx[~any_true] = -1  # or use `np.nan` if float output is acceptable
    return rev_idx    

def plot_npz(path: Path) -> None:
    data = np.load(path)
    print(f"Loaded {path}")
    if 'average_segment' in data:
        # plt.figure()
        # plt.imshow(data['average_segment'], aspect='auto', origin='lower', cmap='gray')
        # plt.title('Average Segment')
        # plt.colorbar()
        play_video_cv2(data['average_segment'], intv=17)
        
    if 'ssim' in data:
        plt.figure()
        plt.imshow(data['ssim'], aspect='auto', origin='lower', cmap='viridis')
        plt.title('SSIM Matrix')
        plt.colorbar()

    for key in data.files:
        if key.endswith('_area'):
            plt.figure()
            plt.plot(data[key])
            plt.title(key)
            plt.xlabel('Frame')
            plt.ylabel('Area (pixels)')
        elif key.endswith('_signal'):
            '''
            plt.figure()
            plt.imshow(data[key], aspect='auto', origin='lower', cmap='viridis')
            plt.title(key)
            plt.colorbar()
            '''
            # size = data[key].shape
            signal_density_bins = np.linspace(0, 1, 180) # Example bins, adjust as needed
            plot_angle_signal_density(signal_density_bins, data[key])
        elif key.endswith('_time_distance_intensity'):
            plt.figure()
            plt.imshow(data[key], aspect='auto', origin='lower', cmap='viridis')
            plt.title(key)
            plt.colorbar()
    plt.show()


def select_folder() -> Path | None:
    """Open a folder selection dialog starting in the script directory."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title='Select Folder Containing .npz Files',
        initialdir=Path(__file__).resolve().parent
    )
    root.destroy()
    return Path(folder) if folder else None


def main() -> None:
    folder = select_folder()
    if folder is None:
        return

    for file in sorted(folder.glob('*.npz')):
        # plot_npz(file)
        data = np.load(file)
        processed_data = {}
        for key in data.files:
            match key:
            
                case _ if key.endswith('average_segment'):
                    continue
                    # play_video_cv2(data[key], intv=17)
                    video = data[key]
                    bw = np.zeros(video.shape)
                    for i in range(0, video.shape[0]):
                        bw[i], _ = triangle_binarize(video[i])
                    play_videos_side_by_side((video, bw))

                case _ if key.endswith('_signal'):
                    continue
                    angle = data[key]
                    frames, bins = angle.shape
                    half = round(bins/2)
                    ang = np.concatenate((angle[:,half:], angle[:,0:half]), axis=1)
                
                    bw, thres = triangle_binarize(ang)
                    cone_angle1d = bw.sum(axis=1)/bins

                    # plt.plot(cone_angle_per_frame)
                    # signal_density_bins = np.linspace(0, 1, 180) # Example bins, adjust as needed
                    # plot_angle_signal_density(signal_density_bins, data[key])


                case _ if key.endswith('_time_distance_intensity'):
                    
                    penetration_2d = data[key]
                    bw, thres = triangle_binarize(penetration_2d)
                    # plt.imshow(bw, origin="lower")
                    result = keep_largest_component(bw)
                    pen1d = penetration_bw_to_index(result)

                case _ if key.endswith('_area'):
                    area1d = data[key]

                case _:
                    continue


if __name__ == '__main__':
    main()