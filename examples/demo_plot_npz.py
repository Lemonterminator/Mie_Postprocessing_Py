from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from OSCC_postprocessing.functions_videos import *
from OSCC_postprocessing.cone_angle import plot_angle_signal_density
from scipy import ndimage
import pandas as pd

plot_on = False

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

def extract_features_to_csv(npz_path: Path, out_csv: Path, video_id=None):
    data = np.load(npz_path)
    rows = []
    # infer number of frames from any per-frame array
    any_seg = next(k for k in data.files if k.endswith('_signal'))
    n_frames = data[any_seg].shape[0]
    # collect all unique segment prefixes, e.g., "segment0", "segment1", ...
    prefixes = sorted({key.split('_', 1)[0] for key in data.files if '_' in key})


    for prefix in prefixes:
        # make sure required keys exist for this prefix
        cone_angle_key = f"{prefix}_signal"
        penetration_key = f"{prefix}_time_distance_intensity"
        area_key = f"{prefix}_area"
        if not (cone_angle_key in data and penetration_key in data and area_key in data):
            continue  # skip incomplete

        #   Cone Angle
        angle = data[cone_angle_key]
        frames, bins = angle.shape
        half = bins // 2
        ang = np.concatenate((angle[:, half:], angle[:, :half]), axis=1)  # wrap
        bw_angle, _ = triangle_binarize(ang)
        cone_angle1d = bw_angle.sum(axis=1) / bins  # normalized

        #   penetration index
        penetration_2d = data[penetration_key]
        bw_pen, _ = triangle_binarize(penetration_2d)
        largest = keep_largest_component(bw_pen)
        pen1d = penetration_bw_to_index(largest).astype(float)
        pen1d[pen1d < 0] = np.nan  # convert -1 (no signal) to NaN for ML

        # area
        area1d = data[area_key]

        # Assemble rows
        seg_id = int(''.join(filter(str.isdigit, prefix)))      # e.g., "segment3" -> 3
        for frame in range(n_frames):
            rows.append({
                "video": video_id if video_id is not None else npz_path.stem,
                "segment": seg_id,
                "frame": frame,
                "cone_angle": cone_angle1d[frame],
                "penetration_index": pen1d[frame],
                "area": area1d[frame],
            })
        
        df = pd.DataFrame(rows)
        # optional: reorder columns
        df = df[["video", "segment", "frame", "cone_angle", "penetration_index", "area"]]
        df.to_csv(out_csv, index=False)
        print(f"Saved features to {out_csv}")

'''
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
                    if plot_on:
                        # continue
                        # play_video_cv2(data[key], intv=17)
                        video = data[key]
                        bw = np.zeros(video.shape)
                        for i in range(0, video.shape[0]):
                            bw[i], _ = triangle_binarize(video[i])
                        play_videos_side_by_side((video, bw))

                case _ if key.endswith('_signal'):
                    # continue
                    angle = data[key]
                    frames, bins = angle.shape
                    half = round(bins/2)
                    ang = np.concatenate((angle[:,half:], angle[:,0:half]), axis=1)
                
                    bw, thres = triangle_binarize(ang)
                    cone_angle1d = bw.sum(axis=1)/bins

                    if plot_on:
                        plt.plot(cone_angle1d)
                        # signal_density_bins = np.linspace(0, 1, 180) # Example bins, adjust as needed
                        # plot_angle_signal_density(signal_density_bins, data[key])


                case _ if key.endswith('_time_distance_intensity'):
                    
                    penetration_2d = data[key]
                    bw, thres = triangle_binarize(penetration_2d)
                    if plot_on:
                        plt.imshow(bw, origin="lower")
                    result = keep_largest_component(bw)
                    pen1d = penetration_bw_to_index(result)
                    if plot_on:
                        plt.plot(pen1d)

                case _ if key.endswith('_area'):
                    area1d = data[key]
                    if plot_on:
                        plt.plot(area1d)

                case _:
                    continue
'''
def main():
    folder = select_folder()
    if folder is None:
        return

    for file in sorted(folder.glob("*.npz")):
        out_csv = file.with_name(file.stem + "_features.csv")
        extract_features_to_csv(file, out_csv)

if __name__ == '__main__':
    main()
