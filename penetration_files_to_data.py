import os

os.environ.setdefault("MPLBACKEND", "Agg")  # use non-interactive backend for async saves
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from mie_postprocessing.functions_videos import get_subfolder_names
from mie_postprocessing.Data_cleaning_1d_series import *
from mie_postprocessing.async_plot_saver import AsyncPlotSaver
from mie_postprocessing.async_npz_saver import AsyncNPZSaver
import re
import math
import numpy as np

REPETITIONS_PER_CONDITION = 5
HYDRAULIC_DELAY_ESTIMATION_MIN = 10
HYDRAULIC_DELAY_ESTIMATION_MAX = 20

# Inner and outer radius (in pixels) for cropping the images
ir_ = 11 # Nozzle 1,2,3,4
# ir_ = 14 # DS300
or_ = 380
thres = or_ - ir_ - 20

# Z-score threshold for outlier removal
z_threshold = 2.5

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

def numeric_then_alpha_key(p: Path):
    """
    Sort numerically by the first integer in the stem if present; otherwise
    fall back to case-insensitive alphabetical by full name.
    """
    m = re.search(r'\d+', p.stem)
    if m:
        return (0, int(m.group(0)))          # group 0 = first number
    else:
        return (1, p.name.lower())           # non-numeric go after (or before if you swap 0/1)


def load_penetration_array(file: Path) -> np.ndarray | None:
    """Return the penetration array stored in the file or None if not available."""
    if file.suffix == '.npz':
        try:
            with np.load(file, allow_pickle=True) as data:
                key = next((k for k in ('penetration', 'arr_0') if k in data.files), None)
                if key is None:
                    print(f"Skipping {file}: missing 'penetration' array")
                    return None
                arr = data[key]
        except Exception as exc:
            print(f"Skipping {file}: failed to load ({exc})")
            return None
    else:
        try:
            arr = np.load(file, allow_pickle=True)
        except Exception as exc:
            print(f"Skipping {file}: failed to load ({exc})")
            return None

    arr = np.asarray(arr)
    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(row) for row in arr])
        except Exception as exc:
            print(f"Skipping {file}: object array could not be converted ({exc})")
            return None

    if arr.ndim != 2:
        print(f"Skipping {file}: expected 2-D penetration data, got shape {arr.shape}")
        return None

    return arr.astype(np.float32, copy=False)


def _plot_mean_with_std(ax, means: np.ndarray, stds: np.ndarray, title: str, *, ylim: tuple[float, float] | None = None):
    import matplotlib.pyplot as plt  # lazy import to speed up dialog opening
    """Plot each time-series mean with a solid colorful line and its std as a translucent band."""
    frames = means.shape[-1]
    x = np.arange(frames)
    # Choose a colormap that can handle many series distinctly
    cmap = plt.get_cmap('tab20')
    num_series = means.shape[0]
    for i in range(num_series):
        color = cmap(i % cmap.N)
        m = means[i]
        s = stds[i]
        ax.plot(x, m, color=color, linewidth=1.5)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.25, linewidth=0)
    ax.set_xlim(left=0, right=frames)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Penetration (in Pixels)")
    ax.grid(True)

def penetration_data_cleaning(condition_data, z_score_threshold, valid_count_thres_percentage=0.75):
    _, frames = condition_data.shape[1:]
    data = condition_data.reshape(-1, frames)
    entries, frames = data.shape
    valid_count_thres = valid_count_thres_percentage * entries

    # Last index where condition is true:
    valid_counts = (~np.isnan(data)).sum(axis=0)

    data_copy = data.copy()

    where_valid = np.where(valid_counts > valid_count_thres)[0]
    idx = where_valid[-1] if where_valid.any() else 0
    '''
    if idx > HYDRAULIC_DELAY_ESTIMATION_MAX:
        
        import matplotlib.pyplot as plt  # lazy import to speed up dialog opening
        
        plt.plot(data[:, :idx].T)
        plt.plot(data[:, idx:].T)
        plt.xlabel("frames")
        plt.ylabel("penetration")
        plt.title(f"Valid (sample>{valid_count_thres}) VS invalid")
        plt.grid()
        
        data_copy[:, idx:] = np.nan

        total_valid_counts_original = valid_counts.sum()
        total_valid_counts_filtered =  (~np.isnan(data_copy)).sum()

        if total_valid_counts_filtered/total_valid_counts_original < 0.6:
            grid = np.zeros((frames, or_-ir_))

            condition_data_alt = data.reshape(-1, frames)
            for i in range(condition_data_alt.shape[0]):
                for frame in range(condition_data_alt.shape[1]):
                    val = condition_data_alt[i, frame]
                    if ~np.isnan(val):
                        pen = round(condition_data_alt[i, frame])
                        grid[frame, pen] = 1
            grid = grid.astype(np.bool)
    '''
    is_right_censored = np.any(data > thres )
    if is_right_censored:
        time_slices = np.sum(data > thres, axis = 0)>0
        first_right_censored_idx = np.argmax(time_slices)
    else:
        first_right_censored_idx = np.nan
    # Removing outliers via z-score
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std_safe = np.where(std == 0, 1, std)
    z_scores = (condition_data - mean) / std_safe

    # Filtering based on z-score
    A = (np.abs(z_scores) < z_score_threshold)
    B = np.where(A, 1, np.nan)
    condition_data_cleaned = condition_data * B
    # Before the minimum hydraulic delay, there should be no value
    mask_pre = condition_data_cleaned[:, :, :HYDRAULIC_DELAY_ESTIMATION_MIN] >= 0
    condition_data_cleaned[:, :, :HYDRAULIC_DELAY_ESTIMATION_MIN][mask_pre] = np.nan
    # After the maximum hydraulic delay, there should be no zero value
    mask_post = condition_data_cleaned[:, :, HYDRAULIC_DELAY_ESTIMATION_MAX:] == 0
    condition_data_cleaned[:, :, HYDRAULIC_DELAY_ESTIMATION_MAX:][mask_post] = np.nan

    # Update mean
    mean =  np.nanmean(condition_data_cleaned.reshape(-1, frames), axis=0)
    return condition_data_cleaned, mean, std, first_right_censored_idx


def handle_testpoint_penetration(test_point_penetration_folder: Path,
                                 output_root: Path,
                                 plot_saver: AsyncPlotSaver,
                                 npz_saver: AsyncNPZSaver,
                                 z_score_threshold: float = z_threshold,
                                 plot_on: bool = True):
    files = [file for file in test_point_penetration_folder.iterdir() if file.is_file()]
    files = sorted(files, key=numeric_then_alpha_key)
    try:
        testpoint_name = str(test_point_penetration_folder.parts[-2])
    except Exception:
        testpoint_name = "testpoint"
    tp_out_dir = output_root / testpoint_name
    tp_out_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        print(f"No penetration files found in {testpoint_name}")
        return
    condition_count = max(1, math.ceil(len(files) / REPETITIONS_PER_CONDITION))
    for condition in range(condition_count):
        start_idx = condition * REPETITIONS_PER_CONDITION
        end_idx = start_idx + REPETITIONS_PER_CONDITION
        condition_files = files[start_idx:end_idx]
        if not condition_files:
            continue
        valid_arrays = []
        invalid_files = []
        for current_file in condition_files:
            arr = load_penetration_array(current_file)
            if arr is None:
                invalid_files.append(current_file.name)
                continue
            valid_arrays.append(arr)
        if not valid_arrays:
            if invalid_files:
                skipped = ', '.join(invalid_files)
                print(f"Skipping condition {condition + 1:02d} in {testpoint_name}: no usable penetration traces (invalid: {skipped})")
            else:
                print(f"Skipping condition {condition + 1:02d} in {testpoint_name}: no usable penetration traces")
            continue
        reference_shape = valid_arrays[0].shape
        if any(arr.shape != reference_shape for arr in valid_arrays):
            shapes = ', '.join(str(arr.shape) for arr in valid_arrays)
            print(f"Skipping condition {condition + 1:02d} in {testpoint_name}: inconsistent shapes among traces ({shapes})")
            continue
        condition_data = np.stack(valid_arrays, axis=0).astype(np.float32, copy=False)
        if invalid_files:
            skipped = ', '.join(invalid_files)
            print(f"Condition {condition + 1:02d} in {testpoint_name}: ignored {len(invalid_files)} corrupt file(s): {skipped}")
        
        if plot_on:
            ylim = (0, or_ - ir_)

            # 1) Original vs Cleaned overview
            import matplotlib.pyplot as plt  # lazy import to speed up dialog opening
            fig1, ax = plt.subplots(2, 2, figsize=(12, 12))
            for i in range(condition_data.shape[0]):
                ax[0, 0].plot(condition_data[i].T, linewidth=0.8)
                
        # Data cleaning
        plumes, frames = condition_data.shape[1:]
        condition_data_cleaned, mean, std, cen_idx = penetration_data_cleaning(condition_data, z_score_threshold)



        # Averaging over repetition axis
        plume_wise_mean = np.nanmean(condition_data_cleaned, axis=0)
        plume_wise_std = np.nanstd(condition_data_cleaned, axis=0)

        # Averaging over plume number axis
        shot_wise_mean = np.nanmean(condition_data_cleaned, axis=1)
        shot_wise_std = np.nanstd(condition_data_cleaned, axis=1)

        # Save cleaned data and aggregated stats efficiently
        npz_path = tp_out_dir / f"condition_{condition + 1:02d}_stats.npz"
        npz_saver.save(
            npz_path,
            condition_data=condition_data,
            condition_data_std=std,
            condition_data_cleaned=condition_data_cleaned,
            condition_data_cleaned_mean=mean,
            plume_wise_mean=plume_wise_mean,
            plume_wise_std=plume_wise_std,
            shot_wise_mean=shot_wise_mean,
            shot_wise_std=shot_wise_std,
            first_right_censored_idx=cen_idx
        )

        if plot_on:
            if cen_idx is not np.nan:
                ax[0, 0].axvline(cen_idx, color='red', linestyle='--', label='First Right-Censored Frame')
                ax[0, 1].axvline(cen_idx, color='red', linestyle='--', label='First Right-Censored Frame')
                ax[1, 0].axvline(cen_idx, color='red', linestyle='--', label='First Right-Censored Frame')
                
            for i in range(condition_data.shape[0]):
                ax[0, 1].plot(condition_data_cleaned[i].T, linewidth=0.8)
            ax[0, 0].set_title("Original")
            ax[0, 1].set_title("Cleaned")
            ax[0, 0].set_xlabel("Frame Number")
            ax[0, 1].set_xlabel("Frame Number")
            ax[0, 0].set_ylabel("Penetration (in Pixels)")
            ax[0, 1].set_ylabel("Penetration (in Pixels)")
            ax[0, 0].set_xlim(left=0, right=frames)
            ax[0, 1].set_xlim(left=0, right=frames)
            ax[0, 0].set_ylim(*ylim)
            ax[0, 1].set_ylim(*ylim)

            ax[1, 0].set_title("Original + mean")
            ax[1, 0].set_xlabel("Frame Number")
            ax[1, 0].set_ylabel("Penetration (in Pixels)")
            ax[1, 0].set_xlim(left=0, right=frames)


            x = np.arange(frames)
            for i in range(condition_data.shape[0]):
                ax[1, 0].plot(condition_data[i].T, linewidth=0.3)
                ax[1, 0].plot(mean, linewidth=5)
                
                m = mean
                s = std
                ax[1, 0].fill_between(x, m - s, m + s, color="red", alpha=0.25, linewidth=0)

            ax[1, 1].set_title("Censored vs. Uncensored Mean")
            ax[1, 1].set_xlabel("Frame Number")
            ax[1, 1].set_ylabel("Penetration (in Pixels)")
            ax[1, 1].set_xlim(left=0, right=frames)

            if cen_idx is not np.nan:
                m = mean[0:cen_idx]
                ax[1,1].plot(m, linewidth=1, label="Uncensored Mean")
                s = std[0:cen_idx]
                ax[1,1].fill_between(x[0:cen_idx], m - s, m + s, color="red", alpha=0.25, linewidth=0)
                # ax[1,1].plot(np.concatenate(np.nan(int(cen_idx)), mean[cen_idx:]) , linewidth=1, label="Mean after Censoring")
                empty = np.empty(frames)
                empty[:] = np.nan
                empty[cen_idx:] = mean[cen_idx:]
                ax[1,1].plot(empty, linewidth=1, label="Censored Mean")
            else:
                ax[1,1].plot(mean, linewidth=1, label="Mean")

            for a in ax.ravel():
                a.grid(True)
                a.legend()
            plt.suptitle(f"Condition {condition + 1:d} in Testpoint {testpoint_name}")
            plot_saver.submit(fig1, tp_out_dir / f"condition_{condition + 1:02d}_overview.png")

            '''
            # 2) Plume-wise mean +/- std
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            _plot_mean_with_std(ax2, plume_wise_mean, plume_wise_std, "Plume-wise Mean +/- Std", ylim=ylim)
            plot_saver.submit(fig2, tp_out_dir / f"condition_{condition + 1:02d}_plume_mean_std.png")

            # 3) Shot-wise mean +/- std
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            _plot_mean_with_std(ax3, shot_wise_mean, shot_wise_std, "Shot-wise Mean +/- Std", ylim=ylim)
            plot_saver.submit(fig3, tp_out_dir / f"condition_{condition + 1:02d}_shot_mean_std.png")
            '''

def main():

    folder = select_folder()
    if folder is None:
        return
    subfolders = get_subfolder_names(folder)
    # sort numerically by the number after 'T'
    try:
        subfolders = sorted(subfolders, key=lambda x: int(x[1:]))
    except Exception:
        if 'penetration_results' in subfolders:
            subfolders.remove('penetration_results')
        subfolders = sorted(subfolders, key=lambda x: int(x[1:]))

    # Output root under selected folder
    output_root = folder / "penetration_results"
    output_root.mkdir(parents=True, exist_ok=True)
    # Async savers
    plot_saver = AsyncPlotSaver(max_workers=4)
    npz_saver = AsyncNPZSaver(max_workers=2)

    batch_size = 10
    batches = [subfolders[i:i+batch_size] for i in range(0, len(subfolders), batch_size)]
    for batch in batches: 
        for subfolder in batch:
            print("Handling subfolder", subfolder)
            directory_path = folder / subfolder
            all_folders = get_subfolder_names(directory_path)

            for experiment_results in all_folders:
                if experiment_results == 'penetration':
                    TP_P_folder = directory_path / 'penetration'
                    handle_testpoint_penetration(
                        TP_P_folder,
                        output_root=output_root,
                        plot_saver=plot_saver,
                        npz_saver=npz_saver,
                        plot_on=True              # Turn off plotting to save time
                    )
    # Ensure all saves complete before exit
    npz_saver.shutdown(wait=True)
    plot_saver.shutdown(wait=True)

if __name__ == '__main__':
    main()
