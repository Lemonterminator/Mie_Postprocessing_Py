"""High-level feature extraction from binarized plume videos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._backend import to_numpy_host
from .connected_components import penetration_bw_to_index


def extract_single_plume_features(
    bw_video: np.ndarray,
    nozzle_opening_detection_height=None,
    nozzle_opening_detection_width=None,
    umbrella_angle=180.0,
    thres_penetration_num_pix=5,
    parallel_boundary=False,
    boundary_max_workers=None,
    bw_video_col_sum=None,
):
    """Compute single-plume geometric features from a binarized video."""
    from OSCC_postprocessing.analysis.regression import linear_regression_fixed_intercept
    from OSCC_postprocessing.analysis.single_plume import (
        bw_boundaries_all_points_single_plume,
        bw_boundaries_xband_filter_single_plume,
    )

    bw_video_host = to_numpy_host(bw_video)
    if bw_video_host.ndim != 3:
        raise ValueError(f"bw_video must be 3D (F, H, W), got shape {bw_video_host.shape}")

    if umbrella_angle == 180.0:
        x_scale = 1.0
    else:
        tilt_angle = (180.0 - umbrella_angle) / 2.0
        x_scale = 1.0 / np.cos(np.deg2rad(tilt_angle))

    frame_count, height, _ = bw_video_host.shape
    if bw_video_col_sum is None:
        bw_video_col_sum = np.sum(bw_video_host, axis=1)
    else:
        bw_video_col_sum = to_numpy_host(bw_video_col_sum)
    area = bw_video_col_sum.sum(axis=-1)
    penetration_bw_x = penetration_bw_to_index(bw_video_col_sum > thres_penetration_num_pix)

    boundary_split_points = bw_boundaries_all_points_single_plume(
        bw_video_host,
        parallel=parallel_boundary,
        max_workers=boundary_max_workers,
        umbrella_angle=umbrella_angle,
    )

    upper_bw_width = bw_video_host[:, : height // 2, :].sum(axis=1)
    lower_bw_width = bw_video_host[:, height // 2 :, :].sum(axis=1)
    estimated_volume = x_scale * np.pi * 0.25 * np.sum((upper_bw_width + lower_bw_width) ** 2, axis=1)
    max_plume_radius = np.maximum(upper_bw_width, lower_bw_width)
    min_plume_radius = np.minimum(upper_bw_width, lower_bw_width)
    estimated_volume_max = np.pi * x_scale * np.sum(max_plume_radius**2, axis=1)
    estimated_volume_min = np.pi * x_scale * np.sum(min_plume_radius**2, axis=1)

    penetration_bw_polar = np.zeros(frame_count)
    for frame_idx in range(frame_count):
        pts = boundary_split_points[frame_idx]
        if pts is None or len(pts) < 2:
            boundary_split_points[frame_idx] = None
            continue
        if len(pts[0]) > 0 and len(pts[1]) > 0:
            uy, ux = pts[1][:, 0], pts[1][:, 1]
            ly, lx = pts[0][:, 0], pts[0][:, 1]
            penetration_bw_polar[frame_idx] = max(
                np.max(np.hypot(uy, ux)),
                np.max(np.hypot(ly, lx)),
            )

    points_all_frames = bw_boundaries_xband_filter_single_plume(
        boundary_split_points,
        to_numpy_host(penetration_bw_x),
    )

    lg_up = np.full(frame_count, np.nan)
    lg_low = np.full(frame_count, np.nan)
    avg_up = np.full(frame_count, np.nan)
    avg_low = np.full(frame_count, np.nan)

    for frame_idx in range(frame_count):
        points = points_all_frames[frame_idx]
        if points is None:
            continue
        if len(points[0]) == 0 or len(points[1]) == 0 or penetration_bw_polar[frame_idx] <= 0:
            continue

        uy, ux = points[1][:, 0], points[1][:, 1]
        ly, lx = points[0][:, 0], points[0][:, 1]
        avg_up[frame_idx] = np.nanmean(np.atan(uy / ux) * 180.0 / np.pi)
        avg_low[frame_idx] = np.nanmean(np.atan(ly / lx) * 180.0 / np.pi)

        try:
            lg_up[frame_idx] = np.atan(linear_regression_fixed_intercept(ux, uy, 0.0)) * 180.0 / np.pi
            lg_low[frame_idx] = np.atan(linear_regression_fixed_intercept(lx, ly, 0.0)) * 180.0 / np.pi
        except ValueError:
            pass

    opening = None
    closing = None
    if nozzle_opening_detection_height is not None and nozzle_opening_detection_width is not None:
        from OSCC_postprocessing.analysis.hysteresis import detect_single_high_interval

        h0 = (height - nozzle_opening_detection_height) // 2
        h1 = (height + nozzle_opening_detection_height) // 2
        w1 = int(nozzle_opening_detection_width)
        near_nozzle_signal = np.sum(bw_video_host[:, h0:h1, :w1], axis=(1, 2))
        (_, _, opening, closing), _, _ = detect_single_high_interval(near_nozzle_signal)

    return {
        "area": area,
        "penetration_bw_x": penetration_bw_x,
        "boundary": boundary_split_points,
        "estimated_volume": estimated_volume,
        "estimated_volume_max": estimated_volume_max,
        "estimated_volume_min": estimated_volume_min,
        "penetration_bw_polar": penetration_bw_polar,
        "cone_angle_average": avg_up - avg_low,
        "avg_up": avg_up,
        "avg_low": avg_low,
        "cone_angle_linear_regression": lg_up - lg_low,
        "lg_up": lg_up,
        "lg_low": lg_low,
        "nozzle_opening": opening,
        "nozzle_closing": closing,
    }


def features_to_binarized_metrics_df(features: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert feature-extraction output into the legacy binarized-metrics table."""
    frame_count = len(features["area"])
    return pd.DataFrame(
        {
            "Frame": np.arange(frame_count),
            "Penetration_from_BW": features["penetration_bw_x"],
            "Penetration_from_BW_Polar": features["penetration_bw_polar"],
            "Cone_Angle_Average": features["cone_angle_average"],
            "Cone_Angle_Linear_Regression": features["cone_angle_linear_regression"],
            "Area": features["area"],
            "Estimated_Volume": features["estimated_volume"],
            "Estimated_Volume_Limit_Upper": features["estimated_volume_max"],
            "Estimated_Volume_Limit_Lower": features["estimated_volume_min"],
            "Cone_Angle_Avg_Upper": features["avg_up"],
            "Cone_Angle_Avg_Lower": features["avg_low"],
            "Cone_Angle_Avg": features["avg_low"],
            "Cone_Angle_Average_Upper": features["avg_up"],
            "Cone_Angle_Average_Lower": features["avg_low"],
            "Cone_Angle_LG_Upper": features["lg_up"],
            "Cone_Angle_LG_Lower": features["lg_low"],
            "Cone_Angle_Linear_Regression_Upper": features["lg_up"],
            "Cone_Angle_Linear_Regression_Lower": features["lg_low"],
            "CA_Avg_Upper": features["avg_up"],
            "CA_Avg_Lower": features["avg_low"],
            "CA_LG_Upper": features["lg_up"],
            "CA_LG_Lower": features["lg_low"],
        }
    )


def processing_from_binarized_video(
    bw_video,
    bw_video_col_sum=None,
    shift_boundary_y=True,
    timing=True,
):
    """Backward-compatible wrapper around ``extract_single_plume_features``."""
    del shift_boundary_y

    if timing:
        import time

        start_time = time.time()

    features = extract_single_plume_features(
        bw_video,
        bw_video_col_sum=bw_video_col_sum,
        thres_penetration_num_pix=0,
        parallel_boundary=True,
    )
    df = features_to_binarized_metrics_df(features)

    if timing:
        elapsed = time.time() - start_time
        print(f"Feature extraction from binarized video completed in: {elapsed:.2f} seconds")

    return df, features["boundary"]


def spary_features_from_bw_video(*args, **kwargs):
    """Backward-compatible alias with the historical misspelling preserved."""
    return extract_single_plume_features(*args, **kwargs)
