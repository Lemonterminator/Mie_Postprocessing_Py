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
from scipy import ndimage as ndi

# ---- Library imports ----
from OSCC_postprocessing.cine.functions_videos import load_cine_video, get_subfolder_names, cine
from OSCC_postprocessing.binary_ops.functions_bw import (
    cndi,
    keep_largest_component_cuda,
    keep_largest_component_nd_cuda,
    spary_features_from_bw_video,
)
from OSCC_postprocessing.binary_ops.masking import generate_ring_mask
from OSCC_postprocessing.playback.video_playback import (
    save_multiplume_with_boundaries_cv2,
    play_multiplume_with_boundaries_cv2,
)
from OSCC_postprocessing.analysis.hysteresis import remove_short_true_runs
from OSCC_postprocessing.utils.backend import cp, xp, USING_CUPY
from OSCC_postprocessing.analysis.single_plume import save_boundary_npz

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

# experiment_configs = [r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\test_matrix_json\Nozzle0.json"]


# Optional single-run fallback used by CLI/env overrides.
parent_folder = parent_folders[-1]
experiment_config = experiment_configs[-1]
results_base_dir = os.getenv("MIE_RESULTS_BASE_DIR")
DEFAULT_RESULTS_BASE_DIR = Path(__file__).resolve().parent / "Mie_scattering_top_view_results"


# =============================================================================
# Image processing config
# =============================================================================

frame_limit = 80
noise_floor_multiplier = 2  # Nozzle 1-8: 3; Nozzle0: 2

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
repair_bw = False
penetration_cleanup_min_len = 5

save_boundary_points_csv = False

# Visual preview toggles. Both default off so production batch runs are silent.
# - save_preview_avi: dump a tiled multi-plume AVI with boundary overlay per video
#   to the I/O writer pool (truly async, no GUI thread risk).
# - preview_playback: synchronously open an OpenCV window after each video. Blocks
#   the pipeline (no GPU/IO overlap) -- intended for interactive QC, not batch runs.
# Press 'q' during playback to skip the rest of the current video.
save_preview_avi = False
preview_playback = False
preview_fps = 15
preview_tile = None  # (rows, cols), or None for adaptive layout

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


def repair_binary_plume_video(bw_video):
    """
    Refine a plume-wise BW video with the legacy multihole morphology sequence.

    The input is expected to have shape ``(plume, frame, height, width)``.
    Each plume is repaired as one 3-D volume, then each frame is cleaned again
    to keep the dominant connected envelope used by downstream BW metrics.
    """
    P, F, _, _ = bw_video.shape
    is_gpu_array = hasattr(bw_video, "__cuda_array_interface__")
    xp_backend = xp if is_gpu_array else np
    ndi_backend = cndi if is_gpu_array else ndi
    repaired = bw_video.astype(bool, copy=True)

    struct_3d = xp_backend.ones((3, 3, 3), dtype=bool)

    for plume_idx in range(P):
        blob_3d = keep_largest_component_nd_cuda(
            ndi_backend.binary_fill_holes(
                ndi_backend.binary_closing(
                    repaired[plume_idx],
                    structure=struct_3d,
                )
            )
        )
        for frame_idx in range(F):
            repaired[plume_idx, frame_idx] = keep_largest_component_cuda(blob_3d[frame_idx])

    return repaired.astype(bw_video.dtype, copy=False)


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


def _load_cine_to_cpu(file: Path):
    """
    Disk-only stage of video loading.

    Returns a host numpy array. Designed to run in a prefetch worker thread so
    file-N+1's bytes can be read while file-N is still on the GPU. Keeps the
    H2D copy out of the worker on purpose -- CuPy contexts are thread-local and
    issuing GPU work from the same thread that owns the pipeline simplifies
    stream/synchronization reasoning.
    """
    return load_cine_video(file, frame_limit=frame_limit)


def _cpu_to_backend(video_cpu):
    """Normalize and move a host video into the active array backend."""
    if xp is cp:
        video = cp.asarray(video_cpu)
        video = video.astype(cp.float16, copy=False)
        video /= cp.float16(4096)
        cp.cuda.Stream.null.synchronize()
        return video

    video = np.asarray(video_cpu, dtype=np.float16)
    video /= np.float16(4096)
    return video


def _load_video_to_backend(file: Path):
    """Convenience wrapper used by code paths without prefetch."""
    return _cpu_to_backend(_load_cine_to_cpu(file))


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

    Drops Python references and forces a GC pass so CuPy arrays are returned to
    the memory pool. We *intentionally* do not call ``free_all_blocks`` here:
    keeping pooled blocks lets the next file reuse identically-shaped buffers
    instead of paying CUDA driver allocation cost on every video. Pool free is
    deferred to subfolder boundaries (see ``_release_gpu_memory_pool``).
    """
    if xp is cp:
        # Make sure queued GPU work is finished before releasing memory blocks.
        cp.cuda.Stream.null.synchronize()

    state.clear()
    gc.collect()


def _release_gpu_memory_pool():
    """Hard-release CuPy memory pools. Call at subfolder/job boundaries only."""
    if xp is cp:
        cp.cuda.Stream.null.synchronize()
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

    # Stage all H2D copies sequentially (CuPy stream is single-context anyway),
    # then run the CPU-side feature extraction across plumes in parallel threads.
    # spary_features_from_bw_video is mostly numpy/scipy and releases the GIL.
    segments_cpu = [_as_numpy(segment_bw) for segment_bw in hp_segments_bw]

    args_list = [
        (
            idx,
            seg,
            nozzle_opening_detection_height,
            nozzle_opening_detection_width,
            umbrella_angle,
            thres_penetration_num_pix,
            inner_radius,
        )
        for idx, seg in enumerate(segments_cpu)
    ]

    if num_segments <= 1:
        results = [_compute_spray_metrics_for_segment(args_list[0])] if args_list else []
    else:
        max_workers = min(num_segments, max(2, (os.cpu_count() or 4) // 2))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_compute_spray_metrics_for_segment, args_list))

    for idx, metrics in results:
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


_IO_WRITER_EXECUTOR: ThreadPoolExecutor | None = None
_PREFETCH_EXECUTOR: ThreadPoolExecutor | None = None


def _get_io_writer_executor() -> ThreadPoolExecutor:
    """Long-lived pool for all disk-bound writes (metrics CSV, NPZ, AVI)."""
    global _IO_WRITER_EXECUTOR
    if _IO_WRITER_EXECUTOR is None:
        _IO_WRITER_EXECUTOR = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="io-writer"
        )
    return _IO_WRITER_EXECUTOR


def _get_prefetch_executor() -> ThreadPoolExecutor:
    """Single-worker pool for cine disk reads ahead of GPU work."""
    global _PREFETCH_EXECUTOR
    if _PREFETCH_EXECUTOR is None:
        _PREFETCH_EXECUTOR = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cine-prefetch"
        )
    return _PREFETCH_EXECUTOR


def _shutdown_executors():
    global _IO_WRITER_EXECUTOR, _PREFETCH_EXECUTOR
    if _IO_WRITER_EXECUTOR is not None:
        _IO_WRITER_EXECUTOR.shutdown(wait=True)
        _IO_WRITER_EXECUTOR = None
    if _PREFETCH_EXECUTOR is not None:
        _PREFETCH_EXECUTOR.shutdown(wait=True)
        _PREFETCH_EXECUTOR = None


def _write_metrics_csv_and_metadata(df, csv_path, metadata, metadata_path):
    """Worker-side combined metrics CSV + metadata JSON write."""
    df.to_csv(csv_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _boundary_npz_path(save_path_subfolder: Path, video_name: str) -> Path:
    return save_path_subfolder / "boundary_points" / f"{video_name}_boundaries.npz"


def _submit_boundary_save(
    save_path_subfolder: Path,
    video_name: str,
    all_boundaries: list,
    pending_futures: list | None,
):
    """
    Submit a single combined NPZ write for one video to the shared writer pool.

    Returning a future (instead of blocking) lets the next file's GPU work
    overlap with disk I/O. The caller drains the futures list at folder end.
    """
    if not save_boundary_points_csv or not any(b is not None for b in all_boundaries):
        return

    out_path = _boundary_npz_path(save_path_subfolder, video_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    boundaries_per_plume = [
        (plume_idx, boundary)
        for plume_idx, boundary in enumerate(all_boundaries)
        if boundary is not None
    ]
    fut = _get_io_writer_executor().submit(
        save_boundary_npz, boundaries_per_plume, str(out_path)
    )
    if pending_futures is not None:
        pending_futures.append(fut)


def _preview_avi_path(save_path_subfolder: Path, video_name: str) -> Path:
    return save_path_subfolder / "preview" / f"{video_name}_preview.avi"


def _build_preview_host_video(hp_segments):
    """Move (P, F, H, W) plume video to host as a contiguous, swap-axed array.

    Performs the H<->W swap on the GPU (cheap stride-only op, then a single
    contiguous copy) before crossing the PCIe boundary.
    """
    if hasattr(hp_segments, "__cuda_array_interface__"):
        swapped = cp.ascontiguousarray(cp.swapaxes(hp_segments, -2, -1))
        host = cp.asnumpy(swapped)
    else:
        host = np.ascontiguousarray(np.swapaxes(np.asarray(hp_segments), -2, -1))
    return host


def _maybe_save_preview_avi(
    save_path_subfolder: Path,
    video_name: str,
    preview_host_video,
    swapped_boundaries,
    pending_futures: list | None,
):
    """Submit tiled multi-plume AVI write to the writer pool when toggled on."""
    if not save_preview_avi:
        return
    out_path = _preview_avi_path(save_path_subfolder, video_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fut = _get_io_writer_executor().submit(
        save_multiplume_with_boundaries_cv2,
        preview_host_video,
        swapped_boundaries,
        str(out_path),
        fps=preview_fps,
        tile=preview_tile,
        swap_axes=False,  # already swapped on GPU side
    )
    if pending_futures is not None:
        pending_futures.append(fut)


def _maybe_play_preview(preview_host_video, swapped_boundaries):
    """Synchronous tiled multi-plume preview on the main thread (blocking)."""
    if not preview_playback:
        return
    play_multiplume_with_boundaries_cv2(
        preview_host_video,
        swapped_boundaries,
        fps=preview_fps,
        tile=preview_tile,
        swap_axes=False,
    )


def _has_saved_boundary_points(
    save_path_subfolder: Path,
    video_name: str,
) -> bool:
    return _boundary_npz_path(save_path_subfolder, video_name).exists()


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
    pending_io_futures: list | None = None,
    preloaded_video_cpu=None,
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
        if preloaded_video_cpu is not None:
            # Disk read overlapped with the previous file's GPU work; only the
            # H2D / dtype conversion remains and runs on the main thread so
            # CuPy's per-thread context stays consistent.
            state["video"] = _cpu_to_backend(preloaded_video_cpu)
        else:
            state["video"] = _load_video_to_backend(file)
        t_load = time.time()

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
        t_preproc = time.time()

        state["postprocess"] = mie_multihole_postprocessing(
            state["foreground"],
            state["highpass_filtered"],
            centre,
            number_of_plumes,
            ir_,
            or_,
            bins=angular_bins,
            INTERPOLATION=interpolation_mode,
            BORDER_MODE=border_mode,
            segment_bw_q_min=segment_bw_q_min,
            segment_bw_q_max=segment_bw_q_max
        )
        state["hp_segments"] = state["postprocess"]["segments_fg"]
        # Need to ignore zeros in the histogram for large number of zeros made by masking.
        state["hp_segments_bw"] = triangle_binarize_gpu(
            state["hp_segments"],
            ignore_zero=True,
        )
        if repair_bw:
            state["hp_segments_bw"] = repair_binary_plume_video(state["hp_segments_bw"])

        umbrella_angle_for_penetration = float(
            nozzle_props.get("umbrella_angle_deg", umbrella_angle_deg_default)
        )
        state["penetration_highpass"] = penetration_cdf_all_plumes(
            state["hp_segments"],
            ir_,
            quantile=upper_quantile_cdf,
            umbrella_angle=umbrella_angle_for_penetration,
        )
        t_gpu = time.time()

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
        t_metrics = time.time()
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
        # df and metadata_payload are not mutated after this point, so handing
        # them to a writer thread is safe. The future is drained at folder end.
        csv_fut = _get_io_writer_executor().submit(
            _write_metrics_csv_and_metadata,
            state["df"],
            metrics_csv_path,
            metadata_payload,
            metadata_path,
        )
        if pending_io_futures is not None:
            pending_io_futures.append(csv_fut)
        t_save_csv = time.time()

        _submit_boundary_save(
            save_path_subfolder,
            video_name,
            all_boundaries,
            pending_io_futures,
        )
        t_boundary = time.time()

        # Optional visual preview: build host-side tiled video once, then
        # dispatch (async AVI) and/or play (sync imshow). Both are gated by
        # module-level toggles so production batch runs pay no cost.
        preview_host_video = None
        swapped_boundaries = None
        if save_preview_avi or preview_playback:
            preview_host_video = _build_preview_host_video(state["hp_segments"])
            from OSCC_postprocessing.playback.video_playback import (
                _swap_plume_boundary_yx,
            )
            swapped_boundaries = [
                _swap_plume_boundary_yx(b) if b is not None else None
                for b in all_boundaries
            ]
        _maybe_save_preview_avi(
            save_path_subfolder, video_name,
            preview_host_video, swapped_boundaries,
            pending_io_futures,
        )
        _maybe_play_preview(preview_host_video, swapped_boundaries)
        t_preview = time.time()

        elapsed = t_preview - start_time
        done_message = (
            f"Completed: {relative_label} in {elapsed:.2f}s  "
            f"[load={t_load - start_time:.2f}s  "
            f"preproc={t_preproc - t_load:.2f}s  "
            f"postproc+cdf={t_gpu - t_preproc:.2f}s  "
            f"bw_metrics={t_metrics - t_gpu:.2f}s  "
            f"save_csv_submit={t_save_csv - t_metrics:.2f}s"
            + (f"  boundary_submit={t_boundary - t_save_csv:.2f}s" if save_boundary_points_csv else "")
            + (f"  preview={t_preview - t_boundary:.2f}s" if (save_preview_avi or preview_playback) else "")
            + "]"
        )
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

    pending_io_futures: list = []
    loop = asyncio.get_running_loop()
    prefetch_pool = _get_prefetch_executor()

    def _is_completed_skip(file: Path, save_path_subfolder: Path) -> bool:
        """Mirror the early-exit condition inside _process_cine_file.

        Used here to avoid prefetching files that will be skipped anyway.
        """
        video_name = file.stem
        metrics_csv_path = save_path_subfolder / f"{video_name}.csv"
        metadata_path = save_path_subfolder / f"{video_name}.meta.json"
        if not (metrics_csv_path.exists() and metadata_path.exists()):
            return False
        if save_boundary_points_csv and not _has_saved_boundary_points(
            save_path_subfolder, video_name
        ):
            return False
        return True

    try:
        for subfolder in subfolders:
            print(subfolder)

            # Each subfolder corresponds to one testpoint and has its own
            # geometry/config metadata plus a list of cine files.
            directory_path = Path(parent_folder + "\\" + subfolder)
            save_path_subfolder = save_path / subfolder
            _ensure_directory(save_path_subfolder)

            # Sort once so repeated runs process files in a deterministic order.
            all_files = [f for f in directory_path.iterdir() if f.is_file()]
            all_files = sorted(all_files, key=numeric_then_alpha_key)
            number_of_plumes, centre, ir_, or_ = _load_subfolder_config(all_files)

            # Map folder name (T1, T01, Group_01, ...) back to the matching test
            # condition in the experiment matrix.
            group_id = extract_group_id_from_path(subfolder)
            test_condition = get_test_condition_by_id(exp_config, group_id) or {}
            ring_mask = None  # Lazily initialized by the first valid cine file.

            cine_files = [f for f in all_files if f.suffix == ".cine"]

            # Prefetch pipeline (depth=1): while file N is on the GPU, the
            # prefetch worker reads file N+1 from disk. The H2D copy + GPU
            # processing still happen on the main thread for CuPy correctness.
            # Skipped files do not consume a prefetch slot.
            def _next_load_future(start_idx: int):
                for j in range(start_idx, len(cine_files)):
                    nf = cine_files[j]
                    if _is_completed_skip(nf, save_path_subfolder):
                        continue
                    return j, loop.run_in_executor(
                        prefetch_pool, _load_cine_to_cpu, nf
                    )
                return None, None

            next_idx, next_future = _next_load_future(0)

            for i, file in enumerate(cine_files):
                if _is_completed_skip(file, save_path_subfolder):
                    # Still call _process_cine_file so it logs/checkpoints the
                    # skip uniformly. preloaded_video_cpu=None is fine; the
                    # skip-check inside returns before any load happens.
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
                        pending_io_futures=pending_io_futures,
                        preloaded_video_cpu=None,
                    )
                    continue

                # Pull the prefetched buffer for the current file.
                if next_future is not None and next_idx == i:
                    video_cpu = await next_future
                else:
                    # Fallback: prefetch wasn't aligned (shouldn't happen but
                    # keeps the code robust).
                    video_cpu = await loop.run_in_executor(
                        prefetch_pool, _load_cine_to_cpu, file
                    )

                # Kick off the next eligible prefetch BEFORE running GPU work
                # so the disk read overlaps with GPU compute of this file.
                next_idx, next_future = _next_load_future(i + 1)

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
                    pending_io_futures=pending_io_futures,
                    preloaded_video_cpu=video_cpu,
                )

            # Per-subfolder GPU pool free so long batch runs don't accumulate
            # cross-testpoint pool fragmentation. Per-file free has been
            # removed -- we want pool reuse within a subfolder.
            _release_gpu_memory_pool()

    except Exception as e:
        print(f"Error processing: {e}")
        raise
    finally:
        # Drain all backgrounded I/O writes (NPZ boundary, metrics CSV,
        # metadata JSON, preview AVI). Failures are logged but do not mask
        # the original error.
        if pending_io_futures:
            print(f"[I/O] Waiting on {len(pending_io_futures)} pending writes...")
            for fut in as_completed(pending_io_futures):
                try:
                    fut.result()
                except Exception as exc:
                    err_msg = f"I/O write failed: {type(exc).__name__}: {exc}"
                    print(f"[WARN] {err_msg}")
                    _append_processing_log(log_path, err_msg)
            pending_io_futures.clear()


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
