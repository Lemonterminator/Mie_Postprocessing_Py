"""Optical-flow helpers for the packaged Masters-thesis pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

from OSCC_postprocessing.motion import (
    compute_optical_flow_magnitude,
    normalize_flow_magnitude,
)


FLOW_NORM_LOWER_PERCENTILE = 5.0
FLOW_NORM_UPPER_PERCENTILE = 99.0


def _resolve_backend_name(method: str) -> str:
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

    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def _default_max_workers() -> int:
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


def opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow):
    import cv2

    prev_u8 = _scale_float_frame_to_uint8(prev_frame)
    frame_u8 = _scale_float_frame_to_uint8(frame)

    if len(frame_u8.shape) == 3:
        prev_gray = cv2.cvtColor(prev_u8, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_u8.copy()
        gray = frame_u8.copy()

    return deepflow.calc(prev_gray, gray, None)  # type: ignore[no-any-return]


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
        with ThreadPoolExecutor(max_workers=_default_max_workers()) as executor:
            for i, mag in executor.map(_process_farneback_weighted_task, tasks):
                mag_array[i] = mag
                print(f"Processed frame {i + 1}/{nframes}")
        return _normalize_weighted_magnitude_array(
            mag_array,
            firstFrameNumber,
            normalize=normalize,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

    if str(method).lower() == "deepflow":
        if deepflow is None:
            raise ValueError("DeepFlow instance must be provided for DeepFlow method.")

        tasks = list(_frame_pairs(video, firstFrameNumber))

        def _process_deepflow_weighted_task(task):
            i, prev_frame, frame = task
            import cv2

            flow = opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            return i, mag

        with ThreadPoolExecutor(max_workers=_default_max_workers()) as executor:
            for i, mag in executor.map(_process_deepflow_weighted_task, tasks):
                mag_array[i] = mag
                print(f"Processed frame {i + 1}/{nframes}")
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
