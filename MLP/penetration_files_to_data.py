import os

# os.environ.setdefault("MPLBACKEND", "Agg")  # use non-interactive backend for async saves
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from tkinter import filedialog
import importlib.util
from OSCC_postprocessing.functions_videos import get_subfolder_names
from OSCC_postprocessing.Data_cleaning_1d_series import *
from OSCC_postprocessing.async_plot_saver import AsyncPlotSaver
from OSCC_postprocessing.async_npz_saver import AsyncNPZSaver
import re
import math
import numpy as np
import pandas as pd

REPETITIONS_PER_CONDITION = 5
HYDRAULIC_DELAY_ESTIMATION_MIN = 10
HYDRAULIC_DELAY_ESTIMATION_MAX = 20
DEFAULT_INJECTION_PRESSURE = 2000

# Inner and outer radius (in pixels) for cropping the images
ir_ = 11 # Nozzle 1,2,3,4
# ir_ = 14 # DS300
or_ = 380
thres = or_ - ir_ - 5 

# Z-score threshold for outlier removal
z_threshold = 3

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


def _load_module_from_path(path: Path):
    """Import a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_test_matrix(initial_file: Path | None = None):
    """
    GUI prompt to choose a test_matrix file (e.g., Nozzle1.py) and return the loaded module.
    """
    default = initial_file if initial_file else Path(__file__).resolve().parent / "test_matrix" / "Nozzle1.py"
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select test_matrix file (e.g., Nozzle1.py)",
        initialdir=default.parent,
        initialfile=default.name,
        filetypes=[("Python files", "*.py")],
    )
    root.destroy()
    if not file_path:
        return None
    return _load_module_from_path(Path(file_path))

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

def penetration_data_cleaning(condition_data, z_score_threshold):
    """
    Clean penetration traces:
    1) z-score filter outliers,
    2) force zero penetration before the minimum hydraulic delay,
    3) enforce non-decreasing penetration until the first threshold crossing,
    4) right-censor (NaN) all samples from the first threshold crossing onward.
    """
    rep, plume, frames = condition_data.shape
    data = condition_data.reshape(-1, frames)

    # Removing outliers via z-score
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std_safe = np.where(std == 0, 1, std)
    z_scores = (condition_data - mean) / std_safe

    # Filtering based on z-score
    A = (np.abs(z_scores) < z_score_threshold)
    B = np.where(A, 1, np.nan)
    condition_data_cleaned = condition_data * B

    # Enforce zero penetration before hydraulic delay
    condition_data_cleaned[:, :, :HYDRAULIC_DELAY_ESTIMATION_MIN] = 0

    # Flatten for per-trace processing
    flat_cleaned = condition_data_cleaned.reshape(-1, frames)
    first_crossings: list[int] = []

    for row in flat_cleaned:
        # Forward-fill NaNs with the last seen value to avoid drops
        last_val = 0.0
        for i in range(frames):
            if np.isnan(row[i]):
                row[i] = last_val
            else:
                last_val = row[i]

        # Enforce monotonic non-decreasing penetration
        row[:] = np.maximum.accumulate(row)

        # Right-censor once the trace reaches the threshold
        over = row >= thres
        if np.any(over):
            first_idx = int(np.argmax(over))
            first_crossings.append(first_idx)
            row[first_idx:] = np.nan

    condition_data_cleaned = flat_cleaned.reshape(rep, plume, frames)

    # Updated stats after cleaning and censoring
    mean = np.nanmean(flat_cleaned, axis=0)
    std = np.nanstd(flat_cleaned, axis=0)

    first_right_censored_idx = int(min(first_crossings)) if first_crossings else np.nan

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
        return None, None
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
        # Training dataset: cleaned, monotonic traces per plume/rep with NaN tail after threshold
        df_raw = pd.DataFrame()
        rep, plume, frames = condition_data_cleaned.shape
        for r in range(rep):
            for p in range(plume):
                arr = condition_data_cleaned[r, p, :]
                df_raw[f"penetration_cleaned_plume_{p+1:02d}_rep_{r+1:02d}"] = arr
        df_raw["frame_number"] = np.arange(frames)

        # Testing dataset: mean/median/std per frame, NaN after first censoring index
        median = np.nanmedian(condition_data_cleaned, axis=(0, 1))

        def _monotonic_with_nan_tail(arr: np.ndarray) -> np.ndarray:
            # ensure non-decreasing then blank out after censoring index
            filled = np.maximum.accumulate(np.nan_to_num(arr, nan=0.0))
            if not np.isnan(cen_idx):
                filled[int(cen_idx):] = np.nan
            return filled

        mean_for_df = _monotonic_with_nan_tail(mean)
        median_for_df = _monotonic_with_nan_tail(median)
        std_for_df = std.copy()
        if not np.isnan(cen_idx):
            std_for_df[int(cen_idx):] = np.nan

        df_test = pd.DataFrame({
            "penetration_average": mean_for_df,
            "penetration_median": median_for_df,
            "penetration_std": std_for_df,
            "frame_number": np.arange(frames),
        })

        return df_raw, df_test

    # If no valid condition was processed, return empty placeholders
    return None, None


def main():

    test_matrix_module = select_test_matrix()
    if test_matrix_module is None:
        print("No test_matrix file selected; exiting.")
        return

    # Access to all variables defined in the selected test_matrix file
    test_matrix_vars = {k: v for k, v in vars(test_matrix_module).items() if not k.startswith("__")}
    print(f"Loaded test_matrix: {test_matrix_module.__file__}")
    print(f"Available variables: {', '.join(sorted(test_matrix_vars))}")

    folder = select_folder()
    if folder is None:
        return
    
    # Getting the nozzle name
    nozzle_name = folder.name.split("_")[-1]

    # Subfolders within the folder 
    subfolders = get_subfolder_names(folder)

    # sort numerically by the number after 'T'
    try:
        subfolders = sorted(subfolders, key=lambda x: int(x[1:]))
    except Exception:
        if 'penetration_results' in subfolders:
            subfolders.remove('penetration_results')
        subfolders = sorted(subfolders, key=lambda x: int(x[1:]))

    # Output root under selected folder
    # output_root = folder / "penetration_results"
    # Saving data within the 
    cwd = Path.cwd()
    output_root = cwd / nozzle_name 
    output_root.mkdir(parents=True, exist_ok=True)

    NPZ_root = output_root / "penetration_results_NPZ"
    NPZ_root.mkdir(parents=True, exist_ok=True)

    # Additional data processing with Pandas Dataframe

    DF_root = output_root / "Dataframes"
    DF_root.mkdir(parents=True, exist_ok=True)

    test_data = DF_root / "test_data"
    test_data.mkdir(parents=True, exist_ok=True)

    train_data = DF_root / "train_data"
    train_data.mkdir(parents=True, exist_ok=True)


    # Async savers
    plot_saver = AsyncPlotSaver(max_workers=4)
    npz_saver = AsyncNPZSaver(max_workers=2)

    batch_size = 10
    batches = [subfolders[i:i+batch_size] for i in range(0, len(subfolders), batch_size)]

    df_raw_collection = pd.DataFrame()
    df_test_collection = pd.DataFrame()

    for batch in batches: 
        for subfolder in batch:
            print("Handling subfolder", subfolder)
            directory_path = folder / subfolder
            all_folders = get_subfolder_names(directory_path)

            # Getting the testpoint number and the working conditions from the test matrix
            TP_number = int(subfolder.split("T")[-1])
            working_condition = test_matrix_vars["T_GROUP_TO_COND"][TP_number]

            chamber_pressure = working_condition["chamber_pressure"]
            injection_duration = working_condition["injection_duration"]
            try:
                injection_pressure = working_condition["injection_pressure"] 
            except KeyError:
                injection_pressure = DEFAULT_INJECTION_PRESSURE
            
            umbrella_angle = test_matrix_vars["UMBRELLA_ANGLE"]
            tilt_angle = (180 - umbrella_angle)/2 / 180 * np.pi

            plumes = test_matrix_vars["PLUMES"]
            diameter = test_matrix_vars["DIAMETER"]

            FPS = test_matrix_vars["FPS"]

            try:
                px2mm_scale = test_matrix_vars["px2mm_scale"]
            except KeyError:
                px2mm_scale = 1.0




            for experiment_results in all_folders:
                if experiment_results == 'penetration':
                    TP_P_folder = directory_path / 'penetration'
                    
                    df_raw, df_test = handle_testpoint_penetration(
                        TP_P_folder,
                        output_root=NPZ_root,
                        plot_saver=plot_saver,
                        npz_saver=npz_saver,
                        plot_on=False              # Turn off plotting to save time 
                        )                           # type: ignore
                    
                    if df_raw is None or df_test is None:
                        continue
                    else:
                        df_raw["chamber_pressure"] = chamber_pressure
                        df_raw["injection_pressure"] = injection_pressure
                        df_raw["injection_duration"] = injection_duration
                        df_raw["tilt_angle"] = tilt_angle
                        df_raw["px2mm_scale"] = px2mm_scale
                        df_raw["time_us"]  = df_raw.index / FPS * 1e6
                        df_raw["plumes"] = plumes
                        df_raw["diamter"] = diameter
                        

                        df_test["time_us"] = df_test.index / FPS * 1e6
                        df_test["chamber_pressure"] = chamber_pressure  
                        df_test["injection_pressure"] = injection_pressure
                        df_test["injection_duration"] = injection_duration
                        df_test["tilt_angle"] = tilt_angle
                        df_test["px2mm_scale"] = px2mm_scale
                        df_test["plumes"] = plumes
                        df_test["diamter"] = diameter

                        # Save CSVs under train/test data folders using the subfolder name
                        train_csv = train_data / f"{subfolder}.csv"
                        test_csv = test_data / f"{subfolder}.csv"
                        train_csv.parent.mkdir(parents=True, exist_ok=True)
                        test_csv.parent.mkdir(parents=True, exist_ok=True)
                        df_raw.to_csv(train_csv, index=False)
                        df_test.to_csv(test_csv, index=False)

                        df_raw_collection = pd.concat([df_raw_collection, df_raw], ignore_index=True)
                        df_test_collection = pd.concat([df_test_collection, df_test], ignore_index=True)

    
    
        # Ensure all saves complete before exit
    npz_saver.shutdown(wait=True)
    plot_saver.shutdown(wait=True)

    train_csv = output_root / f"{nozzle_name}_train_data.csv"
    test_csv = output_root / f"{nozzle_name}_test_data.csv"
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    df_raw_collection.to_csv(train_csv)
    df_test_collection.to_csv(test_csv)

if __name__ == '__main__':
    main()
