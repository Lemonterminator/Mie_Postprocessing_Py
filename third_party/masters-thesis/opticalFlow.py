"""Optical-flow helpers for the fused Masters-thesis pipeline.

Local changes:
- convert the original per-frame loops into independent frame-pair tasks
- execute Farneback/DeepFlow work with ThreadPoolExecutor
- preserve ordered writes to the output arrays so downstream code remains unchanged
"""

from concurrent.futures import ThreadPoolExecutor
import os


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


def _default_max_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


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
    mask = (mag > 0.4).astype(np.uint8) * 255
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
    mask = (mag > 1.0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)
    clustered_overlay = overlay_cluster_outline(frame, cluster_mask)
    return i, cluster_mask, clustered_overlay, mask

def opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow):
    import cv2
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

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
    


def runOpticalFlowCalculationWeighted(firstFrameNumber, video, method, deepflow=None):
    """Compute only flow magnitudes for weighted fusion, using the same threaded frame-pair model."""
    import numpy as np

    nframes = video.shape[0]
    mag_array = np.zeros_like(video, dtype=np.float32)
    tasks = list(_frame_pairs(video, firstFrameNumber))
    max_workers = _default_max_workers()

    if method == 'Farneback':
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, mag in executor.map(_process_farneback_weighted_task, tasks):
                mag_array[i] = mag
                print(f"Processed frame {i+1}/{nframes}")

        return mag_array # return only magnitude for weighted processing
    else:
        raise ValueError(f"Unsupported optical flow method: {method}")
