from __future__ import annotations

import numpy as np

from OSCC_postprocessing.utils.backend import get_cupy


def estimate_nozzle_opening_duration(
    video,
    use_gpu: bool = True,
    up_thres=0.1,
    down_thres=0.1,
    start_frame=0,
    stop_frame=None,
    width=0.1,
    height=0.1,
):
    """Return the near-nozzle average intensity trace for a fixed ROI."""
    frame_count, h, w = video.shape
    assert video.ndim == 3, "video must be (F, H, W)"

    xp = get_cupy() if use_gpu else np
    if use_gpu and xp is None:
        xp = np
        use_gpu = False

    if stop_frame is None:
        stop_frame = frame_count

    h_low = np.clip(0, round(h // 2 - h * height // 2), h)
    h_high = np.clip(0, round(h // 2 + h * height // 2), h)
    w_right = round(w * width)

    near_nozzle_avg_intensity = xp.sum(
        video[start_frame:stop_frame, h_low:h_high, :w_right],
        axis=(1, 2),
    ) / ((h_high - h_low) * w_right)

    result = xp.zeros(frame_count)
    result[start_frame:stop_frame] = near_nozzle_avg_intensity
    return result


__all__ = ["estimate_nozzle_opening_duration"]
