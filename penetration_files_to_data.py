from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from mie_postprocessing.functions_videos import get_subfolder_names
from mie_postprocessing.Data_cleaning_1d_series import * 
import re
import numpy as np
import matplotlib.pyplot as plt

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
    



def handle_testpoint_penetration(test_point_penetration_folder, z_score_threshold = 2.5, plot_on=True):
    files = [file for file in test_point_penetration_folder.iterdir() if file.is_file()]
    files = sorted(files, key=numeric_then_alpha_key)
    # print(files)
    conditions = len(files) // REPETITIONS_PER_CONDITION 

    for condition in range(conditions):
        first_file = files[condition*REPETITIONS_PER_CONDITION]
        first_data = np.load(first_file)
        plumes, frames = first_data.shape
        # Condition Data has shape: Repetition, plume number, frames
        condition_data = np.zeros((REPETITIONS_PER_CONDITION, plumes, frames))
        condition_data[0] = first_data
        for repetition in range(1, REPETITIONS_PER_CONDITION):
            current_file = files[condition*REPETITIONS_PER_CONDITION + repetition]
            condition_data[repetition] = np.load(current_file)
        
        
        data = condition_data.reshape(-1, frames)
        
        # Removing outliers via z-score
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        z_scores = (condition_data - mean) / std
        
        # Cleaned 
        mask = np.abs(z_scores) < z_score_threshold
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

        if plot_on: 
            # Plotting all penetratino of plumes in all repetitions
            fig, ax = plt.subplots(1, 2)

            for i in range(condition_data.shape[0]):
                ax[0].plot(condition_data[i].T)
            for i in range(condition_data.shape[0]):
                ax[1].plot(condition_data_cleaned[i].T)
            ax[0].set_title("Original")
            ax[1].set_title("Cleaned")
            ax[0].set_xlabel("Frame Number")
            ax[1].set_xlabel("Frame Number")
            ax[0].set_ylabel("Penetration (in Pixels)")
            ax[1].set_ylabel("Penetration (in Pixels)")
            ax[0].set_xlim(left=0, right=frames)
            ax[1].set_xlim(left=0, right=frames)
            ax[1].set_ylim(bottom=0, top=or_ - ir_)
            ax[0].set_ylim(bottom=0, top=or_ - ir_)
            
            plt.suptitle(f"Condition {condition + 1:d} in Testpoint " + str(test_point_penetration_folder.parts[-2]))
            ax[0].grid()
            ax[1].grid()
            plt.show()
        






def main():
    folder = select_folder()
    if folder is None:
        return
    subfolders = get_subfolder_names(folder)
    # sort numerically by the number after 'T'
    subfolders = sorted(subfolders, key=lambda x: int(x[1:]))
    
    for subfolder in subfolders:
        # Specify the directory path    
        # If folder is already a Path
        directory_path = folder / subfolder
        all_folders = get_subfolder_names(directory_path)

        # Check the name of the results
        for experiment_results in all_folders:
            if experiment_results == 'penetration':
                TP_P_folder = directory_path / 'penetration'
                handle_testpoint_penetration(TP_P_folder)

    
if __name__ == '__main__':
    main()