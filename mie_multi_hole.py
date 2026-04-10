import time
import warnings
from pathlib import Path

import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import gc
import json
import pandas as pd

# ---- Library imports ----
from OSCC_postprocessing.cine.functions_videos import load_cine_video, get_subfolder_names, cine
from OSCC_postprocessing.binary_ops.functions_bw import spary_features_from_bw_video
from OSCC_postprocessing.binary_ops.masking import generate_ring_mask
from OSCC_postprocessing.playback.video_playback import play_video_cv2
from OSCC_postprocessing.utils.scaling import robust_scale
from OSCC_postprocessing.analysis.hysteresis import remove_short_true_runs
from OSCC_postprocessing.utils.backend import cp, xp, USING_CUPY
from OSCC_postprocessing.analysis.single_plume import save_boundary_csv

# ---- Mie multihole pipeline (library module) ----
from OSCC_postprocessing.analysis.mie_multihole import (
    mie_multihole_preprocessing,
    mie_multihole_postprocessing,
    penetration_cdf_all_plumes,
    triangle_binarize_gpu,
)

# ---- Experiment config utilities (library module) ----
from OSCC_postprocessing.utils.experiment_config import (
    extract_cine_number,
    load_experiment_config,
    get_nozzle_properties,
    get_test_condition_by_id,
    compute_injection_duration_us,
    extract_group_id_from_path,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Experiment Config Loading and Results Management
# =============================================================================
# Previous 2024 nozzle batch configuration kept here for reference:
parent_folders = [
     r"F:\LubeOil\BC20241003_HZ_Nozzle1\cine",
     r"F:\LubeOil\BC20241017_HZ_Nozzle2\BC20241017_HZ_Nozzle2\single",
     r"F:\LubeOil\BC20241014_HZ_Nozzle3\Cine",
     r"F:\LubeOil\BC20241007_HZ_Nozzle4\cine",
     r"F:\LubeOil\BC20241010_HZ_Nozzle5\Cine",
     r"F:\LubeOil\BC20241011_HZ_Nozzle6\Cine",
     r"F:\LubeOil\BC20241015_HZ_Nozzle7\Cine",
     r"F:\LubeOil\BC20241016_HZ_Nozzle8\Cine",
]
#
experiment_configs = [
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle1.json",
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle2.json",
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle3.json",
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle4.json",
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle5.json",
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle6.json",
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle7.json",
     r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle8.json",
]

# parent_folders = [r"F:\LubeOil\BC20220627 - Heinzman DS300 - Mie Top view\Cine"]

# experiment_configs = [r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\DS300.json"]


# Optional single-run fallback used by CLI/env overrides.
parent_folder = parent_folders[-1]
experiment_config = experiment_configs[-1]
results_base_dir = os.getenv("MIE_RESULTS_BASE_DIR")
DEFAULT_RESULTS_BASE_DIR = Path(__file__).resolve().parent / "Mie_scattering_top_view_results"


# =============================================================================
# Image processing config
# =============================================================================

frame_limit = 80
noise_floor_multiplier = 2  # Nozzle 1-8: 3; DS300: 2

# Pre-processing histogram scaling settings
sobel_wsize=3
sobel_sigma=1
threshold=0.02
q_min_foreground=5
q_max_foreground=99.99
q_min_highpass=5
q_max_highpass=99.9999

# Image rotation settings
angular_bins=720
interpolation_mode="nearest"    # "bilinear", "bicubic", "lanczos3"
border_mode = "constant"        # "constant","replicate","reflect"

# Upper quantile settings for Penetration by culmulative distribution function (penetration cdf)
upper_quantile_cdf = 1-5e-3 # Penetration is set at column-wise summed image at x-distance that has 1-5e-3=99.5% of total intensity

# BW feature extraction settings
nozzle_opening_detection_height = 20
nozzle_opening_detection_width = 30
thres_penetration_num_pix = 5  # minimum width of the binarized spray for x-axis penetration detection
segment_bw_q_min = 5
segment_bw_q_max = 99.9
penetration_cleanup_min_len = 5

save_boundary_points_csv = False

# =============================================================================
# Default nozzle properties, safe fall back if not defined in test matrix or in cine.
# =============================================================================
FPS_default = 34000  # 25000
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


def _get_mm_per_px_scale(outer_radius:float) -> float:
    return 90.0 / float(outer_radius)


def _append_penetration_mm_columns(df: pd.DataFrame, mm_per_px_scale: float) -> pd.DataFrame:
    penetration_mm_specs = [
        ("penetration_cdf_plume_", "penetration_cdf(mm)_plume_"),
        ("penetration_bw_x_plume_", "penetration_bw_x(mm)_plume_"),
        ("penetration_bw_polar_plume_", "penetration_bw_polar(mm)_plume_"),
    ]

    for legacy_prefix, mm_prefix in penetration_mm_specs:
        matching_cols = [col for col in df.columns if col.startswith(legacy_prefix)]
        for legacy_col in matching_cols:
            suffix = legacy_col[len(legacy_prefix):]
            mm_col = f"{mm_prefix}{suffix}"
            df[mm_col] = pd.to_numeric(df[legacy_col], errors="coerce") * float(mm_per_px_scale)
    return df


def _build_results_root(
    parent_folder: str | Path,
    results_base_dir_override: str | Path | None = None,
) -> Path:
    parent_path = Path(parent_folder)
    dataset_root = parent_path.parent
    if dataset_root == parent_path:
        raise ValueError(f"Cannot infer dataset root from parent_folder={parent_folder}")
    configured_results_base = (
        Path(results_base_dir_override)
        if results_base_dir_override is not None
        else (Path(results_base_dir) if results_base_dir else None)
    )
    if configured_results_base is not None:
        results_root = configured_results_base
    else:
        results_root = DEFAULT_RESULTS_BASE_DIR
    return results_root / dataset_root.name


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
    idx, segment_bw, opening_h, opening_w, umbrella_angle, penetration_pix, inner_radius = args
    num_frames = int(segment_bw.shape[0]) if hasattr(segment_bw, "shape") and len(segment_bw.shape) > 0 else 0
    try:
        metrics = spary_features_from_bw_video(
            segment_bw,
            opening_h,
            opening_w,
            umbrella_angle=umbrella_angle,
            thres_penetration_num_pix=penetration_pix,
            inner_radius=inner_radius,
        )
    except Exception as e:
        print(f"[WARN] metric extraction failed for plume {idx}: {e}")
        metrics = _empty_spray_metrics(num_frames)
    return idx, metrics


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
        return (0, int(m.group(0)))
    else:
        return (1, p.name.lower())


def _ensure_directory(path: Path):
    """Create a directory if missing and keep reruns quiet when it already exists."""
    path = Path(path)
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    if existed:
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


def _collect_metric_columns(
    hp_segments_bw,
    num_frames: int,
    umbrella_angle: float,
    inner_radius: float,
):
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
                inner_radius,
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


def _has_saved_boundary_points(
    save_path_subfolder: Path,
    video_name: str,
) -> bool:
    boundary_path_subfolder = save_path_subfolder / "boundary_points"
    if not boundary_path_subfolder.exists():
        return False
    matches = list(boundary_path_subfolder.glob(f"{video_name}_plume_*_boundary_points.csv"))
    return len(matches) > 0


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

        boundary_outputs_ready = (
            (not save_boundary_points_csv)
            or _has_saved_boundary_points(save_path_subfolder, video_name)
        )
        if metrics_csv_path.exists() and metadata_path.exists() and boundary_outputs_ready:
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
        elif metrics_csv_path.exists() and metadata_path.exists() and save_boundary_points_csv:
            resume_message = (
                f"Reprocessing file to add missing boundary-point CSVs: {relative_label}"
            )
            print(resume_message)
            _append_processing_log(log_path, resume_message)

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

        # ========================================================================
        # Actual Image processing
        # ========================================================================

        # Preprocessing and postprocessing remain on the active backend
        # (CuPy or NumPy). This keeps the heavy array work in one memory space.
        state["foreground"], state["highpass_filtered"] = mie_multihole_preprocessing(
            state["video"],
            state["ring_mask"],
            wsize=sobel_wsize,
            sigma=sobel_sigma,
            noise_floor_multiplier=noise_floor_multiplier,
            threshold=threshold,
            q_min_foreground=q_min_foreground,
            q_max_foreground=q_max_foreground,
            q_min_highpass=q_min_highpass,
            q_max_highpass=q_max_highpass
        )

        state["postprocess"] = mie_multihole_postprocessing(
            state["foreground"],
            state["highpass_filtered"],
            centre,
            number_of_plumes,
            ir_,
            or_,
            bins=angular_bins,
            INTERPOLATION=interpolation_mode,
            BORDER_MODE=border_mode
        )
        state["hp_segments"] = state["postprocess"]["segments_fg"]
        state["hp_segments_bw"] = triangle_binarize_gpu(
            robust_scale(state["hp_segments"], segment_bw_q_min, segment_bw_q_max)
        )
        umbrella_angle_for_penetration = float(
            nozzle_props.get("umbrella_angle_deg", umbrella_angle_deg_default)
        )
        state["penetration_highpass"] = penetration_cdf_all_plumes(
            state["hp_segments"],
            ir_,
            quantile=upper_quantile_cdf,
            umbrella_angle=umbrella_angle_for_penetration,
        )

        print("GPU work completed in {:.2f}s".format(time.time() - start_time))

        # ========================================================================
        # Extracting results
        # ========================================================================

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
                state["valid_penetration_mask"][p], min_len=penetration_cleanup_min_len
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
        state["df"]["cone_angle_proxy_deg"] = state["postprocess"]["cone_angle_proxy_deg"]
        state["df"]["occupied_angle_total_deg"] = state["postprocess"]["occupied_angle_total_deg"]
        state["df"]["occupied_angle_segment_count"] = state["postprocess"]["occupied_angle_segment_count"]

        for plume_idx in range(P):
            state["df"][f"penetration_cdf_plume_{plume_idx}"] = _as_numpy(
                state["cleaned_penetration_highpass"][plume_idx]
            )

        umbrella_angle = float(state["metadata"].get("umbrella_angle_deg"))
        metric_columns, all_boundaries = _collect_metric_columns(
            state["hp_segments_bw"], num_frames, umbrella_angle, ir_
        )
        if metric_columns:
            state["df"] = pd.concat(
                [state["df"], pd.DataFrame(metric_columns, index=state["df"].index)],
                axis=1,
            )

        dataset_name = save_path_subfolder.parent.name
        mm_per_px_scale = _get_mm_per_px_scale(or_)
        state["df"] = _append_penetration_mm_columns(state["df"], mm_per_px_scale)

        metadata_payload = {
            key: _to_json_scalar(value)
            for key, value in state["metadata"].items()
        }
        metadata_payload["mm_per_px_scale"] = _to_json_scalar(mm_per_px_scale)
        metadata_payload["cone_angle_proxy_deg"] = _to_json_scalar(
            state["postprocess"]["cone_angle_proxy_deg"]
        )
        metadata_payload["occupied_angle_total_deg"] = _to_json_scalar(
            state["postprocess"]["occupied_angle_total_deg"]
        )
        metadata_payload["occupied_angle_segment_count"] = _to_json_scalar(
            state["postprocess"]["occupied_angle_segment_count"]
        )
        metadata_payload["occupied_angle_segment_widths_deg"] = [
            _to_json_scalar(value)
            for value in np.asarray(
                state["postprocess"]["occupied_angle_segment_widths_deg"],
                dtype=float,
            ).tolist()
        ]
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


async def _process_parent_folder(
    parent_folder: str | Path,
    experiment_config: str | Path,
    results_base_dir_override: str | Path | None = None,
):
    """Walk one parent folder and process each ``.cine`` file in order."""
    parent_folder = str(parent_folder)
    experiment_config = str(experiment_config)
    subfolders = get_subfolder_names(parent_folder)

    # Load experiment configuration
    exp_config = load_experiment_config(experiment_config)
    nozzle_props = get_nozzle_properties(exp_config)

    # General folder for saving results
    save_path = _build_results_root(
        parent_folder,
        results_base_dir_override=results_base_dir_override,
    )
    _ensure_directory(save_path)
    print(f"[Paths] source_root={parent_folder}")
    print(f"[Paths] results_root={save_path}")
    log_path = save_path / "processing.log"
    checkpoint_path = save_path / "processing_checkpoint.json"
    _append_processing_log(log_path, f"Session started for parent folder: {parent_folder}")

    try:
        for subfolder in subfolders:
            print(subfolder)

            # Each subfolder corresponds to one testpoint and has its own
            # geometry/config metadata plus a list of cine files.
            directory_path = Path(parent_folder + "\\" + subfolder)
            save_path_subfolder = save_path / subfolder
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


def _resolve_batch_jobs(
    parent_folders_override: list[str] | None = None,
    experiment_configs_override: list[str] | None = None,
) -> list[tuple[str, str]]:
    resolved_parent_folders = (
        list(parent_folders_override)
        if parent_folders_override is not None
        else list(parent_folders)
    )
    resolved_experiment_configs = (
        list(experiment_configs_override)
        if experiment_configs_override is not None
        else list(experiment_configs)
    )

    if len(resolved_parent_folders) != len(resolved_experiment_configs):
        raise ValueError(
            "parent_folders and experiment_configs must have the same length: "
            f"{len(resolved_parent_folders)} != {len(resolved_experiment_configs)}"
        )

    jobs = []
    for idx, (parent_value, config_value) in enumerate(
        zip(resolved_parent_folders, resolved_experiment_configs),
        start=1,
    ):
        parent_candidate = Path(parent_value)
        config_candidate = Path(config_value)
        if not parent_candidate.exists():
            raise FileNotFoundError(
                f"Configured parent folder not found for job {idx}: {parent_candidate}"
            )
        if not config_candidate.exists():
            raise FileNotFoundError(
                f"Configured experiment config not found for job {idx}: {config_candidate}"
            )
        jobs.append((str(parent_candidate), str(config_candidate)))
    return jobs


async def main(
    parent_folders_override: list[str] | None = None,
    experiment_configs_override: list[str] | None = None,
    results_base_dir_override: str | Path | None = None,
):
    """Process every configured parent folder/config pair in sequence."""
    jobs = _resolve_batch_jobs(
        parent_folders_override=parent_folders_override,
        experiment_configs_override=experiment_configs_override,
    )
    for idx, (parent_folder_value, experiment_config_value) in enumerate(jobs, start=1):
        print(
            f"[Batch {idx}/{len(jobs)}] parent_folder={parent_folder_value} | "
            f"experiment_config={experiment_config_value}"
        )
        await _process_parent_folder(
            parent_folder_value,
            experiment_config_value,
            results_base_dir_override=results_base_dir_override,
        )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    import argparse

    parser = argparse.ArgumentParser(description="Run Mie post-processing pipeline")
    parser.add_argument(
        "-p", "--parent-folder",
        default=os.getenv("MIE_PARENT_FOLDER"),
        help="Optional single-run parent folder override"
    )
    parser.add_argument(
        "-c", "--experiment-config",
        default=os.getenv("MIE_EXPERIMENT_CONFIG"),
        help="Optional single-run experiment config override; use together with --parent-folder"
    )
    parser.add_argument(
        "-o", "--results-base-dir",
        default=os.getenv("MIE_RESULTS_BASE_DIR"),
        help=(
            "Optional results root override. Example: "
            r"'G:\Mie_scattering_top_view_results'."
        ),
    )
    args = parser.parse_args()

    parent_folders_override = None
    experiment_configs_override = None
    if args.parent_folder or args.experiment_config:
        if not (args.parent_folder and args.experiment_config):
            print(
                "Single-run override requires both --parent-folder and --experiment-config "
                "(or MIE_PARENT_FOLDER and MIE_EXPERIMENT_CONFIG)."
            )
            raise SystemExit(2)
        parent_folders_override = [args.parent_folder]
        experiment_configs_override = [args.experiment_config]

    start = time.time()
    asyncio.run(
        main(
            parent_folders_override=parent_folders_override,
            experiment_configs_override=experiment_configs_override,
            results_base_dir_override=args.results_base_dir,
        )
    )
    print(f"Total elapsed: {time.time() - start:.2f}s")
