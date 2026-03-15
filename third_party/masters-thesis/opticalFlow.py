"""Optical-flow helpers for the fused Masters-thesis pipeline.

Local changes:
- convert the original per-frame loops into independent frame-pair tasks
- execute Farneback/DeepFlow work with ThreadPoolExecutor
- preserve ordered writes to the output arrays so downstream code remains unchanged
"""

from concurrent.futures import ThreadPoolExecutor
import os

from OSCC_postprocessing.motion import (
    compute_optical_flow_magnitude,
    compute_optical_flows,
    normalize_flow_magnitude,
)


FLOW_NORM_LOWER_PERCENTILE = 5.0
FLOW_NORM_UPPER_PERCENTILE = 99.0
FLOW_MASK_THRESHOLD = 0.6


def _resolve_backend_name(method):
    backend_map = {
        "farneback": "farneback",
        "raft": "raft",
        "deepflow": "deepflow",
        "nvidia_hw": "nvidia_hw",
        "nvidia_hw_of": "nvidia_hw",
    }
    return backend_map.get(str(method).lower(), str(method).lower())


def _scale_float_frame_to_uint8(frame):
    import numpy as np

    arr = np.asarray(frame)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    arr = arr.astype(np.float32, copy=False)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros(arr.shape, dtype=np.uint8)

    finite_vals = arr[finite_mask]
    max_val = float(finite_vals.max())
    if max_val <= 1.0 + 1e-6:
        scaled = arr * 255.0
    elif max_val <= 255.0 + 1e-6:
        scaled = arr
    else:
        scaled = arr * (255.0 / max_val)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _prepare_nvidia_hw_video(video):
    import numpy as np

    arr = np.asarray(video)
    if arr.ndim != 3:
        raise ValueError(f"NVIDIA_HW expects a grayscale video stack shaped (F, H, W), got {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr, 0, 255).astype(np.float32, copy=False)
    else:
        arr = arr.astype(np.float32, copy=False)
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            arr = np.zeros(arr.shape, dtype=np.float32)
        else:
            finite_vals = arr[finite_mask]
            max_val = float(finite_vals.max())
            if max_val > 255.0 + 1e-6:
                arr = arr * (255.0 / max_val)
            arr = np.clip(arr, 0.0, 255.0)
    return arr


def opticalFlowFarnebackCalculation(prev_frame, frame):
    import cv2

    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, # type: ignore
                                        0.5,  # pyramid scale
                                        3,    # levels
                                        15,   # window size
                                        3,    # iterations
                                        5,    # poly_n
                                        1.2,  # poly_sigma
                                        0)    # type: ignore # flags

    return flow


def opticalFlowUnifiedCalculation(prev_frame, frame, method="Farneback", **kwargs):
    """Compatibility wrapper around the unified OSCC optical-flow backend."""
    import numpy as np

    video = np.stack([prev_frame, frame], axis=0)
    backend = _resolve_backend_name(method)
    flow = compute_optical_flows(video, backend=backend, out_hw_last=True, **kwargs)
    return flow[0]


def _default_max_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def _normalize_weighted_magnitude_array(
    mag_array,
    firstFrameNumber,
    *,
    normalize,
    lower_percentile,
    upper_percentile,
):
    import numpy as np

    result = np.asarray(mag_array, dtype=np.float32).copy()
    if not normalize:
        return result

    active = result[firstFrameNumber + 1 :]
    if active.size == 0:
        return result

    result[firstFrameNumber + 1 :] = normalize_flow_magnitude(
        active,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    result[: firstFrameNumber + 1] = 0.0
    return result


def _frame_pairs(video, firstFrameNumber):
    """Yield independent `(frame_idx, prev_frame, frame)` tasks for parallel flow evaluation."""
    first_frame = video[firstFrameNumber]
    for i in range(firstFrameNumber, video.shape[0]):
        prev_frame = first_frame if i == firstFrameNumber else video[i - 1]
        yield i, prev_frame, video[i]


def _process_farneback_weighted_task(task):
    i, prev_frame, frame = task
    import cv2

    flow = opticalFlowFarnebackCalculation(prev_frame, frame)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return i, mag


def _process_farneback_cluster_task(task):
    i, prev_frame, frame = task
    import cv2
    import numpy as np
    from clustering import create_cluster_mask, overlay_cluster_outline

    flow = opticalFlowFarnebackCalculation(prev_frame, frame)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag_n = normalize_flow_magnitude(
        mag,
        lower_percentile=FLOW_NORM_LOWER_PERCENTILE,
        upper_percentile=FLOW_NORM_UPPER_PERCENTILE,
    )
    mask = (mag_n > FLOW_MASK_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)
    clustered_overlay = overlay_cluster_outline(frame, cluster_mask)
    return i, cluster_mask, clustered_overlay, mask


def _process_deepflow_cluster_task(task, deepflow):
    i, prev_frame, frame = task
    import cv2
    import numpy as np
    from clustering import create_cluster_mask, overlay_cluster_outline

    flow = opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag_n = normalize_flow_magnitude(
        mag,
        lower_percentile=FLOW_NORM_LOWER_PERCENTILE,
        upper_percentile=FLOW_NORM_UPPER_PERCENTILE,
    )
    mask = (mag_n > FLOW_MASK_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)
    clustered_overlay = overlay_cluster_outline(frame, cluster_mask)
    return i, cluster_mask, clustered_overlay, mask

def opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow):
    import cv2

    # DeepFlow is the most restrictive path in OpenCV: feed single-channel uint8.
    prev_u8 = _scale_float_frame_to_uint8(prev_frame)
    frame_u8 = _scale_float_frame_to_uint8(frame)

    if len(frame_u8.shape) == 3:
        prev_gray = cv2.cvtColor(prev_u8, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_u8.copy()
        gray = frame_u8.copy()

    flow = deepflow.calc(prev_gray, gray, None) # type: ignore

    return flow

def runOpticalFlowCalculation(firstFrameNumber, video, method, deepflow=None):
    """Compute per-frame binary/clustered flow masks using threaded frame-pair processing."""
    import numpy as np

    nframes = video.shape[0]
    cluster_masks = np.zeros_like(video, dtype=np.uint8)
    clustered_overlays =  np.zeros_like(video, dtype=np.uint8)
    masks =  np.zeros_like(video, dtype=np.uint8)
    tasks = list(_frame_pairs(video, firstFrameNumber))
    max_workers = _default_max_workers()

    if method == 'Farneback':
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, cluster_mask, clustered_overlay, mask in executor.map(_process_farneback_cluster_task, tasks):
                cluster_masks[i] = cluster_mask
                clustered_overlays[i] = clustered_overlay
                masks[i] = mask
                print(f"Processed frame {i+1}/{nframes}")

        return cluster_masks, clustered_overlays, masks

    # Not fully implemented yet, but structure is in place
    elif method == 'DeepFlow':
        if deepflow is None:
            raise ValueError("DeepFlow instance must be provided for DeepFlow method.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            task_iter = ((task, deepflow) for task in tasks)
            for i, cluster_mask, clustered_overlay, mask in executor.map(
                lambda args: _process_deepflow_cluster_task(*args),
                task_iter,
            ):
                cluster_masks[i] = cluster_mask
                clustered_overlays[i] = clustered_overlay
                masks[i] = mask
                print(f"Processed frame {i+1}/{nframes}")

        return cluster_masks, clustered_overlays, masks
    else:
        raise ValueError(f"Unsupported optical flow method: {method}")
    


def runOpticalFlowCalculationWeighted(
    firstFrameNumber,
    video,
    method,
    deepflow=None,
    *,
    normalize=True,
    lower_percentile=FLOW_NORM_LOWER_PERCENTILE,
    upper_percentile=FLOW_NORM_UPPER_PERCENTILE,
):
    """Compute flow magnitudes with solver-specific input preparation.

    Backend input expectations handled here:
    - farneback: (F, H, W) numpy/cupy, integer or float. Backend normalizes internally.
    - raft: (F, H, W) numpy/cupy, integer or float. Backend normalizes and pads internally.
    - nvidia_hw: (F, H, W) cupy float16/float32. This wrapper converts to float32 and clips/scales to <=255.
    - deepflow: per-pair single-channel uint8. Legacy path converts each frame internally.
    - return value: raw pixels/frame if ``normalize=False``; otherwise robustly
      normalized to ``[0, 1]`` across the active video window.
    """
    import numpy as np

    nframes = video.shape[0]
    mag_array = np.zeros_like(video, dtype=np.float32)
    if firstFrameNumber < 0 or firstFrameNumber >= nframes:
        raise ValueError(
            f"firstFrameNumber out of range: {firstFrameNumber} for video with {nframes} frames"
        )

    if firstFrameNumber + 1 >= nframes:
        return mag_array

    if str(method).lower() == "farneback":
        tasks = list(_frame_pairs(video, firstFrameNumber))
        max_workers = _default_max_workers()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, mag in executor.map(_process_farneback_weighted_task, tasks):
                mag_array[i] = mag
                print(f"Processed frame {i+1}/{nframes}")
        return _normalize_weighted_magnitude_array(
            mag_array,
            firstFrameNumber,
            normalize=normalize,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

    # Keep DeepFlow on the legacy per-pair path. The unified backend covers the
    # maintained solvers: Farneback, RAFT, and NVIDIA hardware optical flow.
    if str(method).lower() == "deepflow":
        if deepflow is None:
            raise ValueError("DeepFlow instance must be provided for DeepFlow method.")

        tasks = list(_frame_pairs(video, firstFrameNumber))
        max_workers = _default_max_workers()

        def _process_deepflow_weighted_task(task):
            i, prev_frame, frame = task
            import cv2

            flow = opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            return i, mag

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, mag in executor.map(_process_deepflow_weighted_task, tasks):
                mag_array[i] = mag
                print(f"Processed frame {i+1}/{nframes}")
        return _normalize_weighted_magnitude_array(
            mag_array,
            firstFrameNumber,
            normalize=normalize,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

    backend = _resolve_backend_name(method)
    if backend == "nvidia_hw":
        try:
            import cupy as cp
        except Exception as exc:
            raise RuntimeError("CuPy is required for NVIDIA_HW optical flow") from exc
        prepared = _prepare_nvidia_hw_video(video[firstFrameNumber:])
        video_input = cp.asarray(prepared, dtype=cp.float32)
    else:
        video_input = video[firstFrameNumber:]

    mags = compute_optical_flow_magnitude(
        video_input,
        backend=backend,
        normalize=normalize,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    try:
        import cupy as cp

        if isinstance(mags, cp.ndarray):
            mag_array[firstFrameNumber + 1 :] = cp.asnumpy(mags).astype(np.float32, copy=False)
            return mag_array
    except Exception:
        pass

    mag_array[firstFrameNumber + 1 :] = np.asarray(mags, dtype=np.float32)
    return mag_array
