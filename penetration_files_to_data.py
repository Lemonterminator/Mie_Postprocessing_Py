from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from mie_postprocessing.functions_videos import get_subfolder_names
from mie_postprocessing.Data_cleaning_1d_series import * 
from mie_postprocessing.async_plot_saver import AsyncPlotSaver
from mie_postprocessing.async_npz_saver import AsyncNPZSaver
import re
import numpy as np

REPETITIONS_PER_CONDITION = 5
HYDRAULIC_DELAY_ESTIMATION_MIN = 17
HYDRAULIC_DELAY_ESTIMATION_MAX = 20

# Inner and outer radius (in pixels) for cropping the images
ir_ = 14
or_ = 380


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


def handle_testpoint_penetration(test_point_penetration_folder: Path,
                                 output_root: Path,
                                 plot_saver: AsyncPlotSaver,
                                 npz_saver: AsyncNPZSaver,
                                 z_score_threshold: float = 2.5,
                                 plot_on: bool = True):
    files = [file for file in test_point_penetration_folder.iterdir() if file.is_file()]
    files = sorted(files, key=numeric_then_alpha_key)
    # print(files)
    conditions = len(files) // REPETITIONS_PER_CONDITION 

    # Determine testpoint name for subfolder structure
    try:
        testpoint_name = str(test_point_penetration_folder.parts[-2])
    except Exception:
        testpoint_name = "testpoint"
    tp_out_dir = output_root / testpoint_name
    tp_out_dir.mkdir(parents=True, exist_ok=True)

    for condition in range(conditions):
        first_file = files[condition*REPETITIONS_PER_CONDITION]
        first_data = np.load(first_file)
        if first_file.suffix == '.npz':
            first_data = first_data['penetration']
        plumes, frames = first_data.shape
        # Condition Data has shape: Repetition, plume number, frames
        condition_data = np.zeros((REPETITIONS_PER_CONDITION, plumes, frames))
        condition_data[0] = first_data
        for repetition in range(1, REPETITIONS_PER_CONDITION):
            current_file = files[condition*REPETITIONS_PER_CONDITION + repetition]
            if current_file.suffix == '.npz':
                current_data = np.load(current_file)['penetration']
            else:
                current_data = np.load(current_file)
            condition_data[repetition] = current_data
        
        data = condition_data.reshape(-1, frames)
        
        # Removing outliers via z-score
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        z_scores = (condition_data - mean) / std
        
        # Filtering based on z-score
        condition_data_cleaned = condition_data * (np.abs(z_scores) < z_score_threshold)
        # Before the minimum hydraulic delay, there should be no value
        condition_data_cleaned[:, :, :HYDRAULIC_DELAY_ESTIMATION_MIN][condition_data_cleaned[:, :, :HYDRAULIC_DELAY_ESTIMATION_MIN]>=0]=np.nan
        # After the maximum hydraulic delay, there should be no zero value
        condition_data_cleaned[:, :, HYDRAULIC_DELAY_ESTIMATION_MAX:][condition_data_cleaned[:, :, HYDRAULIC_DELAY_ESTIMATION_MAX:]==0]=np.nan
        
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
            condition_data_cleaned=condition_data_cleaned,
            plume_wise_mean=plume_wise_mean,
            plume_wise_std=plume_wise_std,
            shot_wise_mean=shot_wise_mean,
            shot_wise_std=shot_wise_std,
        )

        if plot_on:
            ylim = (0, or_ - ir_)

            # 1) Original vs Cleaned overview
            import matplotlib.pyplot as plt  # lazy import to speed up dialog opening
            fig1, ax = plt.subplots(1, 2, figsize=(12, 5))
            for i in range(condition_data.shape[0]):
                ax[0].plot(condition_data[i].T, linewidth=0.8)
            for i in range(condition_data.shape[0]):
                ax[1].plot(condition_data_cleaned[i].T, linewidth=0.8)
            ax[0].set_title("Original")
            ax[1].set_title("Cleaned")
            ax[0].set_xlabel("Frame Number")
            ax[1].set_xlabel("Frame Number")
            ax[0].set_ylabel("Penetration (in Pixels)")
            ax[1].set_ylabel("Penetration (in Pixels)")
            ax[0].set_xlim(left=0, right=frames)
            ax[1].set_xlim(left=0, right=frames)
            ax[0].set_ylim(*ylim)
            ax[1].set_ylim(*ylim)
            for a in ax:
                a.grid(True)
            plt.suptitle(f"Condition {condition + 1:d} in Testpoint {testpoint_name}")
            plot_saver.submit(fig1, tp_out_dir / f"condition_{condition + 1:02d}_overview.png")

            # 2) Plume-wise mean ± std
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            _plot_mean_with_std(ax2, plume_wise_mean, plume_wise_std, "Plume-wise Mean ± Std", ylim=ylim)
            plot_saver.submit(fig2, tp_out_dir / f"condition_{condition + 1:02d}_plume_mean_std.png")

            # 3) Shot-wise mean ± std
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            _plot_mean_with_std(ax3, shot_wise_mean, shot_wise_std, "Shot-wise Mean ± Std", ylim=ylim)
            plot_saver.submit(fig3, tp_out_dir / f"condition_{condition + 1:02d}_shot_mean_std.png")
        






def main():
    folder = select_folder()
    if folder is None:
        return
    subfolders = get_subfolder_names(folder)
    # sort numerically by the number after 'T'
    try:
        subfolders = sorted(subfolders, key=lambda x: int(x[1:]))
    except Exception:
        subfolders.remove('penetration_results')
        subfolders = sorted(subfolders, key=lambda x: int(x[1:]))



    # Output root under selected folder
    output_root = folder / "penetration_results"
    output_root.mkdir(parents=True, exist_ok=True)
    # Async savers
    plot_saver = AsyncPlotSaver(max_workers=2)
    npz_saver = AsyncNPZSaver(max_workers=2)

    for subfolder in subfolders:
        print("Handling subfolder", subfolder)
        # Specify the directory path    
        # If folder is already a Path
        directory_path = folder / subfolder
        all_folders = get_subfolder_names(directory_path)

        # Check the name of the results
        for experiment_results in all_folders:
            if experiment_results == 'penetration':
                TP_P_folder = directory_path / 'penetration'
                handle_testpoint_penetration(
                    TP_P_folder,
                    output_root=output_root,
                    plot_saver=plot_saver,
                    npz_saver=npz_saver,
                )

    # Ensure all saves complete before exit
    npz_saver.shutdown(wait=True)
    plot_saver.shutdown(wait=True)

if __name__ == '__main__':
    main()
