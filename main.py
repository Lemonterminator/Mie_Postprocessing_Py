from OSCC_postprocessing.cine.functions_videos import *
from OSCC_postprocessing.rotation.rotate_crop import *
from OSCC_postprocessing.analysis.cone_angle import *
from OSCC_postprocessing.metrics.ssim import *
from OSCC_postprocessing.filters.video_filters import *
from OSCC_postprocessing.binary_ops.functions_bw import *
from OSCC_postprocessing.playback.video_playback import *
from OSCC_postprocessing.utils.scaling import *
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import gc
import json
from pathlib import Path
import pandas as pd
from mie_multihole_pipeline import * 

# =============================================================================
# Experiment Config Loading and Results Management
# =============================================================================

# Define the parent folder and other global variables
parent_folder = r"F:\LubeOil\BC20241016_HZ_Nozzle8\cine"

experiment_config = r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle8.json"

# =============================================================================
# Image processing config
# =============================================================================

frame_limit = 80
noise_floor_multiplier=2
nozzle_opening_detection_height = 20
nozzle_opening_detection_width = 30
thres_penetration_num_pix = 5 # minimum width of the binarizaed spary for x-axis penetration detection
save_boundary_points_csv = False

# =============================================================================
# Default nozzle properties, safe fall back if not defined in test matrix or in cine.
# =============================================================================
FPS_default = 34000 # 25000
injection_pressure_bar_default = 2000
control_backpressure_bar_default = 4
umbrella_angle_deg_default = 180


def _is_missing_value(value) -> bool:
    return value is None or bool(pd.isna(value))


def _to_json_scalar(value):
    if _is_missing_value(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_metrics_and_metadata_csv(
    df: pd.DataFrame,
    csv_path: Path,
    metadata: dict,
    metadata_path: Path,
):
    df.to_csv(csv_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def resolve_metadata_with_fallbacks(
    *,
    nozzle_props: dict,
    test_condition: dict,
    injection_duration_us,
    cine_fps=None,
    num_rows: int,
    context: str,
) -> dict:
    """
    Resolve metadata fields for CSV output and apply hard-coded fallbacks
    when values are missing.
    """
    raw_values = {
        "plumes": nozzle_props.get("plumes"),
        "diameter_mm": nozzle_props.get("diameter_mm"),
        "umbrella_angle_deg": nozzle_props.get("umbrella_angle_deg"),
        "fps": cine_fps if not _is_missing_value(cine_fps) else nozzle_props.get("fps"),
        "chamber_pressure_bar": test_condition.get("chamber_pressure_bar"),
        "injection_duration_us": injection_duration_us,
        "injection_pressure_bar": test_condition.get("injection_pressure_bar"),
        "control_backpressure_bar": test_condition.get("control_backpressure"),
    }

    fallback_defaults = {
        "umbrella_angle_deg": umbrella_angle_deg_default,
        "fps": FPS_default,
        "injection_pressure_bar": injection_pressure_bar_default,
        "control_backpressure_bar": control_backpressure_bar_default,
    }

    resolved = {}
    for field, value in raw_values.items():
        if _is_missing_value(value):
            fallback_value = fallback_defaults.get(field)
            if fallback_value is not None and not _is_missing_value(fallback_value):
                print(
                    f"[WARN] {context}: '{field}' missing for {num_rows} row(s). "
                    f"Using fallback default={fallback_value}."
                )
                resolved[field] = fallback_value
            else:
                print(
                    f"[WARN] {context}: '{field}' missing for {num_rows} row(s). "
                    "No fallback default available; writing NaN."
                )
                resolved[field] = value
        else:
            resolved[field] = value

    return resolved

# =============================================================================
# Utility Functions
# =============================================================================


def _as_numpy(arr):
    if USING_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _empty_spray_metrics(num_frames: int) -> dict:
    nan_series = np.full(num_frames, np.nan)
    return {
        "area": nan_series.copy(),
        "penetration_bw_x": nan_series.copy(),
        "boundary": None,
        "estimated_volume": nan_series.copy(),
        "estimated_volume_max": nan_series.copy(),
        "estimated_volume_min": nan_series.copy(),
        "penetration_bw_polar": nan_series.copy(),
        "cone_angle_average": nan_series.copy(),
        "avg_up": nan_series.copy(),
        "avg_low": nan_series.copy(),
        "cone_angle_linear_regression": nan_series.copy(),
        "lg_up": nan_series.copy(),
        "lg_low": nan_series.copy(),
        "nozzle_opening": np.nan,
        "nozzle_closing": np.nan,
    }


def _compute_spray_metrics_for_segment(args):
    idx, segment_bw, opening_h, opening_w, umbrella_angle, penetration_pix = args
    num_frames = int(segment_bw.shape[0]) if hasattr(segment_bw, "shape") and len(segment_bw.shape) > 0 else 0
    try:
        metrics = spary_features_from_bw_video(
            segment_bw,
            opening_h,
            opening_w,
            umbrella_angle=umbrella_angle,
            thres_penetration_num_pix=penetration_pix,
        )
    except Exception as e:
        print(f"[WARN] metric extraction failed for plume {idx}: {e}")
        metrics = _empty_spray_metrics(num_frames)
    return idx, metrics

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

def extract_cine_number(file_path: Path | str) -> int | None:
    """
    Extract cine number from filename stem (e.g., '121.cine' -> 121).
    """
    path = Path(file_path)
    match = re.search(r"\d+", path.stem)
    if not match:
        return None
    return int(match.group(0))

def compute_injection_duration_us(
    config: dict,
    cine_number: int | None,
    fallback: float | int | None = None,
) -> float | int | None:
    """
    Compute injection duration from config lookup formula, if available.
    Falls back to the provided constant value when lookup is unavailable.
    """
    if cine_number is None:
        return fallback

    lookup = config.get("injection_duration_lookup", {})
    formula = lookup.get("formula", {})
    block_expr = formula.get("block")
    rules = formula.get("rules", [])

    if not block_expr or not rules:
        return fallback

    safe_globals = {"__builtins__": {}}
    local_vars = {"cine_number": cine_number}
    try:
        block = eval(str(block_expr), safe_globals, local_vars)
        local_vars["block"] = block
        for rule in rules:
            condition_expr = rule.get("condition")
            result_expr = rule.get("result")
            if not condition_expr or result_expr is None:
                continue
            if bool(eval(str(condition_expr), safe_globals, local_vars)):
                return eval(str(result_expr), safe_globals, local_vars)
    except Exception:
        return fallback

    return fallback

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


def _ensure_directory(path: Path):
    """Create a directory if missing and keep reruns quiet when it already exists."""
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"Directory {path} already exists. Using existing directory.")


def _append_processing_log(log_path: Path, message: str):
    """Append a one-line timestamped progress record to the run log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()
        os.fsync(f.fileno())


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    status: str,
    file_path: Path | None = None,
    outputs: dict | None = None,
    error: str | None = None,
):
    """Persist the latest processing state so reruns can resume safely."""
    payload = {
        "status": status,
        "file": str(file_path) if file_path is not None else None,
        "outputs": outputs or {},
        "error": error,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _load_subfolder_config(files: list[Path]) -> tuple[int, tuple[float, float], float, float]:
    """
    Load the per-testpoint geometry from ``config.json``.

    Each subfolder contains one config file describing the injector layout used
    by all ``.cine`` files inside that folder. We parse it once and reuse the
    values for every video in the folder.
    """
    for file in files:
        if file.name != "config.json":
            continue
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            int(data["plumes"]),
            (float(data["centre_x"]), float(data["centre_y"])),
            float(data["inner_radius"]),
            float(data["outer_radius"]),
        )
    raise FileNotFoundError("config.json not found in subfolder")


def _load_video_to_backend(file: Path):
    """
    Load one video and normalize it into the active array backend.

    The pipeline uses ``xp`` from the imported processing modules, so this
    helper mirrors that choice:
    - GPU path: move to CuPy and normalize in device memory
    - CPU path: keep a NumPy array and normalize on host
    """
    video_cpu = load_cine_video(file, frame_limit=frame_limit)
    if xp is cp:
        video = cp.asarray(video_cpu)
        video = video.astype(cp.float16, copy=False)
        video /= cp.float16(4096)
        cp.cuda.Stream.null.synchronize()
        return video

    video = np.asarray(video_cpu, dtype=np.float16)
    video /= np.float16(4096)
    return video


def _read_cine_fps(file: Path) -> float | int | None:
    """
    Read FPS directly from the Phantom ``.cine`` header when available.

    ``pycine`` exposes the acquisition metadata through ``setup``. We prefer
    the file-native frame rate over the JSON nozzle config because it is tied
    to the actual recording.
    """
    if cine is None:
        return None

    try:
        header = cine.read_header(file)
    except Exception:
        return None

    setup = header.get("setup")
    if setup is None:
        return None

    for attr in ("FrameRate", "FrameRate16", "fPbRate"):
        value = getattr(setup, attr, None)
        if not _is_missing_value(value) and float(value) > 0:
            return value
    return None


def _cleanup_iteration_state(state: dict):
    """
    Release large per-file objects after each ``.cine`` is processed.

    ``state`` stores temporary arrays and DataFrames created during one file's
    processing. Clearing it in one place makes success, early-return, and error
    paths behave the same way.
    """
    if xp is cp:
        # Make sure queued GPU work is finished before releasing memory blocks.
        cp.cuda.Stream.null.synchronize()

    state.clear()
    gc.collect()

    if xp is cp:
        # CuPy keeps freed blocks in its memory pools for reuse. We explicitly
        # return them here because this script runs many files back-to-back and
        # we want predictable long-run behavior, not maximum reuse.
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _collect_metric_columns(hp_segments_bw, num_frames: int, umbrella_angle: float):
    """
    Extract spray metrics for every plume segment.

    ``hp_segments_bw`` is already binarized per plume. We process one plume at
    a time in the current process to avoid Windows ``spawn`` overhead and large
    memory spikes from duplicating plume videos across worker processes.
    """
    metric_columns = {}
    num_segments = len(hp_segments_bw)
    all_boundaries = [None] * num_segments

    for idx, segment_bw in enumerate(hp_segments_bw):
        _, metrics = _compute_spray_metrics_for_segment(
            (
                idx,
                _as_numpy(segment_bw),
                nozzle_opening_detection_height,
                nozzle_opening_detection_width,
                umbrella_angle,
                thres_penetration_num_pix,
            )
        )
        for feature_name, values in metrics.items():
            if feature_name == "boundary":
                if save_boundary_points_csv:
                    all_boundaries[idx] = values
                continue

            col_name = f"{feature_name}_plume_{idx}"
            if np.isscalar(values):
                metric_columns[col_name] = np.full(num_frames, values)
            else:
                metric_columns[col_name] = _as_numpy(values)

    return metric_columns, all_boundaries


def _save_boundary_points(save_path_subfolder: Path, video_name: str, all_boundaries: list):
    """Persist optional boundary-point CSVs for plumes that produced contours."""
    if not save_boundary_points_csv or not any(boundary is not None for boundary in all_boundaries):
        return

    boundary_path_subfolder = save_path_subfolder / "boundary_points"
    boundary_path_subfolder.mkdir(parents=True, exist_ok=True)
    valid_boundaries = [
        (plume_idx, boundary)
        for plume_idx, boundary in enumerate(all_boundaries)
        if boundary is not None
    ]
    max_workers = min(4, len(valid_boundaries))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                save_boundary_csv,
                boundary,
                boundary_path_subfolder / f"{video_name}_plume_{plume_idx}_boundary_points.csv",
            )
            for plume_idx, boundary in valid_boundaries
        ]
        for future in as_completed(futures):
            future.result()


def _process_cine_file(
    *,
    file: Path,
    directory_path: Path,
    save_path_subfolder: Path,
    exp_config: dict,
    nozzle_props: dict,
    test_condition: dict,
    number_of_plumes: int,
    centre: tuple[float, float],
    ir_: float,
    or_: float,
    ring_mask,
    log_path: Path,
    checkpoint_path: Path,
):
    """
    Process one ``.cine`` file end-to-end and return the reusable ring mask.

    The returned ``ring_mask`` lets the caller reuse the same annular mask for
    the rest of the files in the same subfolder, which avoids rebuilding it for
    every video.
    """
    # ``state`` intentionally holds every large temporary object created during
    # this file's processing so ``finally`` can release them uniformly.
    state = {"ring_mask": ring_mask}
    retained_ring_mask = ring_mask

    try:
        video_name = file.stem
        relative_label = f"{file.parts[-3]} / {file.parts[-2]} / {file.parts[-1]}"
        metrics_csv_path = save_path_subfolder / f"{video_name}.csv"
        metadata_path = save_path_subfolder / f"{video_name}.meta.json"

        if metrics_csv_path.exists() and metadata_path.exists():
            skip_message = f"Skipping completed file: {relative_label}"
            print(skip_message)
            _append_processing_log(log_path, skip_message)
            _write_checkpoint(
                checkpoint_path,
                status="skipped_existing",
                file_path=file,
                outputs={"metrics_csv": str(metrics_csv_path), "metadata": str(metadata_path)},
            )
            return retained_ring_mask

        print("Procssing:", relative_label)
        _append_processing_log(log_path, f"Procssing: {relative_label}")
        _write_checkpoint(
            checkpoint_path,
            status="started",
            file_path=file,
            outputs={"metrics_csv": str(metrics_csv_path), "metadata": str(metadata_path)},
        )

        start_time = time.time()
        state["video"] = _load_video_to_backend(file)

        # The raw loader can legally return an empty array for malformed or
        # unreadable inputs. In that case we skip the rest of the pipeline.
        F, W, H = state["video"].shape
        if F == 0:
            return retained_ring_mask

        # The annular mask depends only on geometry, not on the actual video
        # content, so we build it once per subfolder and reuse it.
        if state["ring_mask"] is None:
            state["ring_mask"] = generate_ring_mask(H, W, centre, ir_, or_)
        retained_ring_mask = state["ring_mask"]

        # Preprocessing and postprocessing remain on the active backend
        # (CuPy or NumPy). This keeps the heavy array work in one memory space.
        state["foreground"], state["highpass_filtered"] = mie_multihole_preprocessing(
            state["video"],
            state["ring_mask"],
            wsize=3,
            sigma=1,
            noise_floor_multiplier=noise_floor_multiplier,
        )

        state["hp_segments"] = mie_multihole_postprocessing(
            state["foreground"],
            state["highpass_filtered"],
            centre,
            number_of_plumes,
            ir_,
            or_,
        )
        state["hp_segments_bw"] = triangle_binarize_gpu(robust_scale(state["hp_segments"], 5, 99.9))
        state["penetration_highpass"] = penetration_cdf_all_plumes(
            state["hp_segments"], ir_, quantile=1.0 - 5e-3
        )

        print("GPU work completed in {:.2f}s".format(time.time() - start_time))

        # From here on we mostly do bookkeeping and tabular assembly. The
        # penetration cleaning is kept explicit because these few steps define
        # the final exported time series.
        state["penetration_diff"] = np.diff(state["penetration_highpass"], axis=1)
        diff_threshold = 0
        x_loc, y_loc = np.where(state["penetration_diff"] < diff_threshold)

        for plume_idx, frame_idx in zip(x_loc, y_loc):
            state["penetration_highpass"][plume_idx, frame_idx - 1 :] = np.nan

        state["valid_penetration_mask"] = ~np.isnan(state["penetration_highpass"])

        P, _ = state["valid_penetration_mask"].shape
        for p in range(P):
            state["valid_penetration_mask"][p] = remove_short_true_runs(
                state["valid_penetration_mask"][p], min_len=5
            )

        state["cleaned_penetration_highpass"] = (
            state["penetration_highpass"] * state["valid_penetration_mask"]
        )

        num_frames = state["cleaned_penetration_highpass"].shape[1]
        cine_number = extract_cine_number(file)
        cine_fps = _read_cine_fps(file)
        injection_duration_us = compute_injection_duration_us(
            exp_config,
            cine_number,
            fallback=test_condition.get("injection_duration_us"),
        )
        state["metadata"] = resolve_metadata_with_fallbacks(
            nozzle_props=nozzle_props,
            test_condition=test_condition,
            injection_duration_us=injection_duration_us,
            cine_fps=cine_fps,
            num_rows=num_frames,
            context=f"{save_path_subfolder / f'{video_name}.csv'}",
        )

        # Build the exported table in two phases:
        # 1. penetration traces
        # 2. derived spray metrics from the binarized plume videos
        state["df"] = pd.DataFrame({"frame_idx": list(range(num_frames))})

        for plume_idx in range(P):
            state["df"][f"penetration_cdf_plume_{plume_idx}"] = _as_numpy(
                state["cleaned_penetration_highpass"][plume_idx]
            )

        umbrella_angle = float(state["metadata"].get("umbrella_angle_deg"))
        metric_columns, all_boundaries = _collect_metric_columns(
            state["hp_segments_bw"], num_frames, umbrella_angle
        )
        if metric_columns:
            state["df"] = pd.concat(
                [state["df"], pd.DataFrame(metric_columns, index=state["df"].index)],
                axis=1,
            )

        metadata_payload = {
            key: _to_json_scalar(value)
            for key, value in state["metadata"].items()
        }
        _save_metrics_and_metadata_csv(
            state["df"],
            metrics_csv_path,
            metadata_payload,
            metadata_path,
        )
        _save_boundary_points(save_path_subfolder, video_name, all_boundaries)

        elapsed = time.time() - start_time
        done_message = f"Completed: {relative_label} in {elapsed:.2f}s"
        print(done_message)
        _append_processing_log(log_path, done_message)
        _write_checkpoint(
            checkpoint_path,
            status="completed",
            file_path=file,
            outputs={"metrics_csv": str(metrics_csv_path), "metadata": str(metadata_path)},
        )

    except Exception as exc:
        _append_processing_log(log_path, f"FAILED: {relative_label} :: {type(exc).__name__}: {exc}")
        _write_checkpoint(
            checkpoint_path,
            status="failed",
            file_path=file,
            outputs={"metrics_csv": str(metrics_csv_path), "metadata": str(metadata_path)},
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    finally:
        retained_ring_mask = state.get("ring_mask", retained_ring_mask)
        _cleanup_iteration_state(state)

    return retained_ring_mask
    

async def main():
    """Walk every testpoint folder and process each ``.cine`` file in order."""
    subfolders = get_subfolder_names(parent_folder)  # Ensure get_subfolder_names is defined or imported

    # Load experiment configuration
    exp_config = load_experiment_config(experiment_config)
    nozzle_props = get_nozzle_properties(exp_config)

    # General folder for saving results
    save_path = Path(parent_folder).parts[-2]
    _ensure_directory(Path(save_path))
    log_path = Path(save_path) / "processing.log"
    checkpoint_path = Path(save_path) / "processing_checkpoint.json"
    _append_processing_log(log_path, f"Session started for parent folder: {parent_folder}")

    try:
        for subfolder in subfolders:
            print(subfolder)
        
            # Each subfolder corresponds to one testpoint and has its own
            # geometry/config metadata plus a list of cine files.
            directory_path = Path(parent_folder + "\\" + subfolder)
            save_path_subfolder = Path(save_path) / subfolder
            _ensure_directory(save_path_subfolder)


            # Sort once so repeated runs process files in a deterministic order.
            files = [file for file in directory_path.iterdir() if file.is_file()]
            files = sorted(files, key=numeric_then_alpha_key)  
            number_of_plumes, centre, ir_, or_ = _load_subfolder_config(files)

            # Map folder name (T1, T01, Group_01, ...) back to the matching test
            # condition in the experiment matrix.
            group_id = extract_group_id_from_path(subfolder)
            test_condition = get_test_condition_by_id(exp_config, group_id) or {}
            ring_mask = None  # Lazily initialized by the first valid cine file.

            for file in files:
                if file.suffix != ".cine":
                    continue
                ring_mask = _process_cine_file(
                    file=file,
                    directory_path=directory_path,
                    save_path_subfolder=save_path_subfolder,
                    exp_config=exp_config,
                    nozzle_props=nozzle_props,
                    test_condition=test_condition,
                    number_of_plumes=number_of_plumes,
                    centre=centre,
                    ir_=ir_,
                    or_=or_,
                    ring_mask=ring_mask,
                    log_path=log_path,
                    checkpoint_path=checkpoint_path,
                )



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
    

