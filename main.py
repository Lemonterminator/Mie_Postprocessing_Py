from OSCC_postprocessing.cine.functions_videos import *
from OSCC_postprocessing.rotation.rotate_crop import *
from OSCC_postprocessing.analysis.cone_angle import *
from OSCC_postprocessing.metrics.ssim import *
from OSCC_postprocessing.filters.video_filters import *
from OSCC_postprocessing.binary_ops.functions_bw import *
from OSCC_postprocessing.playback.video_playback import *
# from mie_multihole_pipeline import *
import matplotlib.pyplot as plt
from main_utils_temp import *

from examples.Schlieren_singlehole_pipeline import *
import subprocess
from scipy.signal import convolve2d
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import gc
import json
from pathlib import Path
from OSCC_postprocessing.io.async_npz_saver import AsyncNPZSaver
import pandas as pd
from main_utils_temp import mie_multihole_video_strip_processing, sobel_magnitude_video_strips, compensate_foreground_gain, traingular_binarize_video_strips, calculate_penetration_bw_num_pixelsthreshold


global parent_folder
global plumes
global offset
global centre
global hydraulic_delay
global gain 
global gamma
global testpoint_name
global video_name


# Define the parent folder and other global variables
parent_folder = r"G:\OSCC\LubeOil\BC20241003_HZ_Nozzle1\cine"
res_dir = r"G:\OSCC\LubeOil\BC20241003_HZ_Nozzle1\results"
rotated_vid_dir = r"G:\OSCC\LubeOil\BC20241003_HZ_Nozzle1\rotated"
experiment_config = r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle1.json"


# =============================================================================
# Experiment Config Loading and Results Management
# =============================================================================

def _as_numpy(arr):
    if USING_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def load_experiment_config(json_path: str | Path) -> dict:
    """
    Load experiment configuration from a JSON file.
    
    Parameters
    ----------
    json_path : str or Path
        Path to the JSON configuration file.
        
    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_nozzle_properties(config: dict) -> dict:
    """
    Extract nozzle properties from the configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from JSON.
        
    Returns
    -------
    dict
        Dictionary containing plumes, diameter_mm, umbrella_angle_deg, fps.
    """
    return config.get("nozzle_properties", {})


def expand_test_matrix(config: dict) -> list[dict]:
    """
    Expand the test matrix into a list of individual test conditions.
    
    Handles both simple cartesian expansion and grouped configurations.
    Each returned dict includes an 'id' key for group identification.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from JSON.
        
    Returns
    -------
    list of dict
        List of test conditions, each with keys like:
        - id: int (1-indexed group ID)
        - chamber_pressure_bar: float
        - injection_duration_us: float
        - injection_pressure_bar: float (if applicable)
    """
    from itertools import product
    
    matrix = config.get("test_matrix", {})
    results = []
    group_id = 1
    
    # Case 1: Simple cartesian expansion
    if matrix.get("expansion") == "cartesian":
        pressures = matrix.get("chamber_pressures_bar", [])
        durations = matrix.get("injection_durations_us", [])
        
        for p, d in product(pressures, durations):
            results.append({
                "id": group_id,
                "chamber_pressure_bar": p,
                "injection_duration_us": d,
            })
            group_id += 1
    
    # Case 2: Grouped configuration (e.g., Nozzle2, DS300)
    elif "groups" in matrix:
        for group in matrix["groups"]:
            # If group has explicit id, use it directly
            if "id" in group:
                results.append({
                    "id": group["id"],
                    "chamber_pressure_bar": group.get("chamber_pressure_bar"),
                    "injection_pressure_bar": group.get("injection_pressure_bar"),
                    "control_backpressure": group.get("control_backpressure"),
                })
            # If group needs cartesian expansion
            elif group.get("expansion") == "cartesian":
                pressures = group.get("chamber_pressures_bar", [])
                durations = group.get("injection_durations_us", [])
                inj_pressure = group.get("injection_pressure_bar")
                
                for p, d in product(pressures, durations):
                    results.append({
                        "id": group_id,
                        "chamber_pressure_bar": p,
                        "injection_duration_us": d,
                        "injection_pressure_bar": inj_pressure,
                    })
                    group_id += 1
    
    return results


def get_test_condition_by_id(config: dict, group_id: int) -> dict | None:
    """
    Get a specific test condition by its group ID.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from JSON.
    group_id : int
        The group ID to look up.
        
    Returns
    -------
    dict or None
        The test condition dictionary, or None if not found.
    """
    conditions = expand_test_matrix(config)
    for cond in conditions:
        if cond.get("id") == group_id:
            return cond
    return None


def extract_group_id_from_path(path: Path | str) -> int | None:
    """
    Extract the test group ID from a file or folder path.
    
    Assumes naming convention like 'T01', 'T1', 'Group_01', or just numeric folders.
    
    Parameters
    ----------
    path : Path or str
        File or folder path to extract group ID from.
        
    Returns
    -------
    int or None
        Extracted group ID, or None if not found.
    """
    path = Path(path)
    # Try to find pattern like T01, T1, Group01, or just numbers
    patterns = [
        r'[Tt](\d+)',      # T01, T1, t01
        r'[Gg]roup[_]?(\d+)',  # Group01, Group_01
        r'^(\d+)$',        # Pure numeric folder name
    ]
    
    for pattern in patterns:
        match = re.search(pattern, path.stem)
        if match:
            return int(match.group(1))
    return None


def create_results_dataframe(
    config: dict,
    group_id: int,
    file_name: str,
    penetration_data: np.ndarray,
    num_plumes: int,
) -> pd.DataFrame:
    """
    Create a results DataFrame with experiment conditions and penetration data.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from JSON.
    group_id : int
        The test group ID.
    file_name : str
        Name of the processed cine file.
    penetration_data : np.ndarray
        Penetration data with shape (num_plumes, num_frames).
    num_plumes : int
        Number of plumes in the injector.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns for experiment conditions and penetration per plume.
    """
    import pandas as pd
    
    # Get test condition and nozzle properties
    condition = get_test_condition_by_id(config, group_id) or {}
    nozzle_props = get_nozzle_properties(config)
    
    num_frames = penetration_data.shape[1] if penetration_data.ndim > 1 else len(penetration_data)
    
    # Build base data
    data = {
        "file": [file_name] * num_frames,
        "frame_idx": list(range(num_frames)),
        "group_id": [group_id] * num_frames,
        # Nozzle properties
        "nozzle_name": [config.get("name", "")] * num_frames,
        "plumes": [nozzle_props.get("plumes")] * num_frames,
        "diameter_mm": [nozzle_props.get("diameter_mm")] * num_frames,
        "umbrella_angle_deg": [nozzle_props.get("umbrella_angle_deg")] * num_frames,
        "fps": [nozzle_props.get("fps")] * num_frames,
        # Test conditions
        "chamber_pressure_bar": [condition.get("chamber_pressure_bar")] * num_frames,
        "injection_duration_us": [condition.get("injection_duration_us")] * num_frames,
        "injection_pressure_bar": [condition.get("injection_pressure_bar")] * num_frames,
    }
    
    # Add penetration data per plume
    for plume_idx in range(num_plumes):
        if penetration_data.ndim > 1:
            data[f"penetration_plume_{plume_idx}"] = penetration_data[plume_idx]
        else:
            data[f"penetration_plume_{plume_idx}"] = penetration_data
    
    return pd.DataFrame(data)


def append_to_master_dataframe(
    master_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append new results to the master DataFrame.
    
    Parameters
    ----------
    master_df : pd.DataFrame or None
        Existing master DataFrame, or None to create new.
    new_df : pd.DataFrame
        New results to append.
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame.
    """
    import pandas as pd
    
    if master_df is None or master_df.empty:
        return new_df.copy()
    return pd.concat([master_df, new_df], ignore_index=True)


def save_master_dataframe(df: pd.DataFrame, output_path: str | Path, format: str = "csv"):
    """
    Save the master DataFrame to disk.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str or Path
        Output file path (without extension).
    format : str
        Output format: 'csv', 'parquet', or 'both'. Default is 'csv'.
    """
    output_path = Path(output_path)
    
    if format in ("csv", "both"):
        df.to_csv(output_path.with_suffix(".csv"), index=False)
    if format in ("parquet", "both"):
        df.to_parquet(output_path.with_suffix(".parquet"), index=False)


frame_limit = 80




# Directory containing mask images and numpy files
# DATA_DIR = Path(__file__).resolve().parent / "data"

# Define a semaphore with a limit on concurrent tasks
SEMAPHORE_LIMIT = 8  # Adjust this based on your CPU capacity
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def play_video_cv2_async(video, gain=1, binarize=False, thresh=0.5, intv=17):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, play_video_cv2, video, gain, binarize, thresh, intv)


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
    


async def main():

    subfolders = get_subfolder_names(parent_folder)  # Ensure get_subfolder_names is defined or imported

    # Initialize background NPZ saver for penetration outputs
    saver = AsyncNPZSaver(max_workers=2)

    # Load experiment configuration
    exp_config = load_experiment_config(experiment_config)
    nozzle_props = get_nozzle_properties(exp_config)

    # General folder for saving results
    save_path = Path(parent_folder).parts[-2]
    try:
        os.mkdir(save_path)
    except FileExistsError:
        print(f"Directory {save_path} already exists. Using existing directory.") 

    try:
        for subfolder in subfolders:
            print(subfolder)
        
            # Specify the directory path
            directory_path = Path(parent_folder + "\\" + subfolder)
            save_path_subfolder = Path(save_path) / subfolder
            try:
                os.mkdir(save_path_subfolder)
            except FileExistsError:
                print(f"Directory {save_path_subfolder} already exists. Using existing directory.")


            # Get a list of all files in the directory
            files = [file for file in directory_path.iterdir() if file.is_file()]
            files = sorted(files, key=numeric_then_alpha_key)  

            for file in files:
                # Find and read config.json
                if file.name == 'config.json':
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # process the data
                        # for item in data:
                            # print(item)
                        number_of_plumes = int(data['plumes'])
                        # offset = float(data['offset']) # Not used in mie_multihole_pipeline
                        centre = (float(data['centre_x']), float(data['centre_y']))

                        ir_ = float(data['inner_radius'])
                        or_ = float(data['outer_radius'])
                        


            # Get test condition from subfolder name (e.g., T1, T01, Group_01)
            group_id = extract_group_id_from_path(subfolder)
            test_condition = get_test_condition_by_id(exp_config, group_id) or {}

            # print(files)
            for file in files:
                if file.suffix == '.cine':



                    
                    testpoint_name = directory_path.stem
                    video_name = file.stem
        
                    print("Procssing:", file.parts[-3], "/", file.parts[-2], "/", file.parts[-1])
                    # start_time = time.time()


                    
                    video = load_cine_video(file, frame_limit=80).astype(np.float32)/4096
                    video = xp.asarray(video)  # Ensure load_cine_video is defined or imported
                    frames, height, width = video.shape

                    
                    centre_x = float(centre[0]) 
                    centre_y = float(centre[1])



                    # Executions part


                    segments, segment_masks, occupied_angles, average_occupied_angle = mie_multihole_video_strip_processing(video,
                                                                                                                            centre,  
                                                                                                                            ir_, 
                                                                                                                            or_, 
                                                                                                                            number_of_plumes, 
                                                                                                                            init_frames=10,)
                    # High pass filtering to find the edge responses                                                                           
                    segments_high_pass = sobel_magnitude_video_strips(segments)

                    # Compensate the gain of the high pass filtered video
                    segments_highpass_compensated = compensate_foreground_gain(segments_high_pass)  

                    # Morphological operations
                    struct1 = cp.zeros((3, 3, 3), dtype=bool)
                    struct1[1, :, :] = True

                    struct2 = cp.ones((3,3,3), dtype=bool)

                    # Binarize and refine by filling the holes and closing the edges
                    segments_highpass_bw = traingular_binarize_video_strips(segments_highpass_compensated, segment_masks=segment_masks, struct_filling=struct1, struct_closing=struct2)

                    # Calculate the penetration from the time-distance intensity heatmap
                    penetration_highpass = calculate_penetration_bw_num_pixelsthreshold(segments_highpass_bw, ir_, or_, thres_num_bw=1)

                    # Calculate the first derivative of the penetration
                    penetration_diff = xp.diff(penetration_highpass, axis=1)

                    # Remove the negative penetration difference
                    diff_threshold = -3
                    x_loc, y_loc = xp.where(penetration_diff < diff_threshold)

                    for plume_idx, frame_idx in zip(x_loc, y_loc):
                        penetration_highpass[plume_idx, frame_idx-1:] = xp.nan

                    TF = ~ xp.isnan(penetration_highpass)

                    P, F = TF.shape
                    for p in range(P):
                        
                        TF[p] = remove_short_true_runs(TF[p], min_len=5)

                    penetration_highpass_cleaned = penetration_highpass * TF
                    
                    # Build DataFrame with experiment config and penetration data
                    num_frames = penetration_highpass_cleaned.shape[1]
                    df = pd.DataFrame({
                        # Frame index
                        'frame_idx': list(range(num_frames)),
                        # Nozzle properties (constant per row)
                        'plumes': [nozzle_props.get('plumes')] * num_frames,
                        'diameter_mm': [nozzle_props.get('diameter_mm')] * num_frames,
                        'umbrella_angle_deg': [nozzle_props.get('umbrella_angle_deg')] * num_frames,
                        'fps': [nozzle_props.get('fps')] * num_frames,
                        # Test conditions from subfolder
                        'chamber_pressure_bar': [test_condition.get('chamber_pressure_bar')] * num_frames,
                        'injection_duration_us': [test_condition.get('injection_duration_us')] * num_frames,
                    })
                    
                    # Add penetration data per plume
                    for plume_idx in range(P):
                        df[f'penetration_highpass_bw_plume_{plume_idx}'] = _as_numpy(penetration_highpass_cleaned[plume_idx])

                    df.to_csv(save_path_subfolder / f'{video_name}.csv', index=False)
    except Exception as e:
        print(f"Error processing: {e}")
        raise

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    import argparse
    import os
    import asyncio, time

    parser = argparse.ArgumentParser(description="Run Mie post-processing pipeline")
    parser.add_argument(
        "-p", "--parent-folder",
        default=os.getenv("MIE_PARENT_FOLDER", parent_folder),
        help="Root folder containing subfolders with .cine files (overrides hardcoded path)"
    )
    args = parser.parse_args()

    # Allow overriding the global parent_folder from CLI/env
    parent_candidate = Path(args.parent_folder)
    if not parent_candidate.exists():
        print(f"Configured parent folder not found: {parent_candidate}")
        print("Set a valid path via --parent-folder or MIE_PARENT_FOLDER, then rerun.")
        raise SystemExit(0)

    parent_folder = str(parent_candidate)

    start = time.time()
    asyncio.run(main())
    print(f"Total elapsed: {time.time() - start:.2f}s")
    

