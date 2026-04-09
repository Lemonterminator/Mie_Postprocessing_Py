"""High-level feature extraction from binarized plume videos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._backend import to_numpy_host
from .connected_components import penetration_bw_to_index


def _shift_boundary_points_x(boundary_points, x_offset: float):
    """Shift boundary coordinates into an orifice-local x frame."""
    if boundary_points is None or x_offset == 0.0:
        return boundary_points

    shifted = [None] * len(boundary_points)
    for frame_idx, points in enumerate(boundary_points):
        if points is None:
            continue

        top, bottom = points

        def _shift(coords):
            coords_arr = np.asarray(coords)
            if coords_arr.size == 0:
                return coords_arr
            shifted_coords = coords_arr.astype(np.float32, copy=True)
            shifted_coords[:, 1] -= np.float32(x_offset)
            return shifted_coords

        shifted[frame_idx] = (_shift(top), _shift(bottom))

    return shifted


def extract_single_plume_features(
    bw_video: np.ndarray,
    nozzle_opening_detection_height=None,
    nozzle_opening_detection_width=None,
    umbrella_angle=180.0,
    thres_penetration_num_pix=5,
    parallel_boundary=False,
    boundary_max_workers=None,
    bw_video_col_sum=None,
    inner_radius: float = 0.0,
):
    """Compute interpretable 1D plume metrics from a binarized video.

    This routine converts a binary plume movie ``(frame, height, width)`` into
    several frame-wise scalar series that can be exported directly to tables or
    used in downstream statistical analysis.

    Coordinate convention
    ---------------------
    The key design choice is that every plume-resolved geometric quantity is
    expressed in an orifice-local coordinate system:

    - ``x = 0`` is the corresponding orifice exit, not the injector center.
    - ``y = 0`` is the vertical centerline of the extracted strip.
    - positive ``x`` points downstream along the plume strip.

    This follows the plume-local viewpoint commonly used when describing an
    individual jet/plume axis. ECN documentation distinguishes injector-axis
    measurements from jet-axis measurements and notes that plume-resolved
    quantities should be referenced to the jet injection location. For
    multi-hole injectors, tomography work also highlights that using the nozzle
    tip instead of the drill-hole origin introduces a directional bias near the
    injector. In this codebase, ``inner_radius`` is the image-space offset
    between injector center and the corresponding orifice, so we explicitly
    shift the boundary coordinates by that amount before computing plume-local
    metrics.

    Reference links used for this coordinate choice:
    - ECN gasoline jet penetration definition:
      https://ecn.sandia.gov/gasoline-spray-combustion/experimental-diagnostics/gasoline-jet-penetration/
    - ECN nozzle/orifice geometry conventions:
      https://ecn.sandia.gov/diesel-spray-combustion/target-condition/spray-a-nozzle-geometry/
    - Spray G plume-direction / drill-hole-origin discussion:
      https://link.springer.com/article/10.1007/s00348-020-2885-0

    Output philosophy
    -----------------
    Each returned series is intentionally tied to a simple geometric
    construction on the binary plume envelope:

    - ``area``: frame-wise white-pixel count collapsed over the strip.
    - ``penetration_bw_x``: furthest downstream occupied x location.
    - ``penetration_bw_polar``: largest radial distance from the orifice-local
      origin to the detected upper/lower boundary.
    - ``avg_up`` / ``avg_low``: average upper/lower boundary angle in an x-band.
    - ``lg_up`` / ``lg_low``: regression-based upper/lower boundary angle.

    The result is not just fast post-processing; it is a compact, explainable
    map from binary envelope geometry to manuscript-ready 1D observables.
    """
    from OSCC_postprocessing.analysis.regression import linear_regression_fixed_intercept
    from OSCC_postprocessing.analysis.single_plume import (
        bw_boundaries_all_points_single_plume,
        bw_boundaries_xband_filter_single_plume,
    )

    bw_video_host = to_numpy_host(bw_video)
    if bw_video_host.ndim != 3:
        raise ValueError(f"bw_video must be 3D (F, H, W), got shape {bw_video_host.shape}")

    # ``x_scale`` compensates for non-180 deg umbrella geometry.
    # Boundary extraction stores x in a projected strip coordinate. When the
    # camera sees the plume at an oblique umbrella angle, downstream distances
    # in the strip are shorter than the corresponding in-plane distances. We
    # therefore rescale x-based geometric quantities by the standard cosine
    # correction.
    if umbrella_angle == 180.0:
        x_scale = 1.0
    else:
        tilt_angle = (180.0 - umbrella_angle) / 2.0
        x_scale = 1.0 / np.cos(np.deg2rad(tilt_angle))
    inner_radius = float(inner_radius)
    inner_radius_scaled = x_scale * inner_radius

    frame_count, height, _ = bw_video_host.shape
    if bw_video_col_sum is None:
        bw_video_col_sum = np.sum(bw_video_host, axis=1)
    else:
        bw_video_col_sum = to_numpy_host(bw_video_col_sum)

    # ``area`` is the most direct extensive descriptor of the binary envelope:
    # number of foreground pixels per frame after collapse over the transverse
    # direction. It is deliberately left in pixel units because this function
    # operates before any physical calibration.
    area = bw_video_col_sum.sum(axis=-1)

    # ``penetration_bw_x`` uses the last occupied column of the binary plume.
    # The helper returns the furthest foreground x-index in strip coordinates.
    # We then shift the origin from injector-center-based strip coordinates to
    # the corresponding orifice location so that ``x = 0`` means "just outside
    # the hole exit". This keeps BW penetration consistent with the CDF-based
    # penetration logic used elsewhere in the pipeline.
    penetration_bw_x = penetration_bw_to_index(bw_video_col_sum > thres_penetration_num_pix).astype(
        np.float32,
        copy=False,
    )
    if inner_radius:
        penetration_bw_x = np.maximum(0.0, penetration_bw_x - float(inner_radius))
    penetration_bw_x = x_scale * penetration_bw_x

    # Boundary extraction is the bridge from a binary plume mask to interpretable
    # geometry. Each frame is reduced to upper and lower boundary point clouds.
    # The helper already centers y about the strip midline; here we additionally
    # shift x so all subsequent geometry is expressed in the orifice-local frame.
    boundary_split_points = bw_boundaries_all_points_single_plume(
        bw_video_host,
        parallel=parallel_boundary,
        max_workers=boundary_max_workers,
        umbrella_angle=umbrella_angle,
    )
    boundary_split_points = _shift_boundary_points_x(boundary_split_points, inner_radius_scaled)

    # Volume proxies are based on an axisymmetric-envelope interpretation:
    # local upper/lower plume widths are treated as diameter-like quantities,
    # then integrated along x. These are still image-derived proxies rather than
    # absolute physical volumes, but they preserve a clear geometric meaning.
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
            # ``penetration_bw_polar`` is the maximum Euclidean distance from
            # the orifice-local origin to the boundary. This is useful when the
            # plume envelope is not well described by a purely axial metric.
            penetration_bw_polar[frame_idx] = max(
                np.max(np.hypot(uy, ux)),
                np.max(np.hypot(ly, lx)),
            )

    # Cone-angle estimates should not use the entire boundary indiscriminately.
    # Near the tip, pixelation and occasional detached blobs dominate. Far
    # downstream, tip breakup and sparse occupancy can bias the slope. We
    # therefore restrict the analysis to an x-band defined as a fixed fraction
    # of the current penetration, which is a common, interpretable way to focus
    # on the plume flank rather than the root or extreme tip.
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

        # Point-wise average angles use ``atan2(y, x)`` rather than ``atan(y/x)``.
        # This is numerically safer at small x and preserves correct quadrant
        # information for any points that fall slightly upstream after boundary
        # processing or coordinate shifts.
        avg_up[frame_idx] = np.nanmean(np.degrees(np.atan2(uy, ux)))
        avg_low[frame_idx] = np.nanmean(np.degrees(np.atan2(ly, lx)))

        try:
            # Regression angles provide a second, smoother cone-angle estimate.
            # Here we fit boundary slope relative to the fixed orifice origin,
            # which mirrors the common manuscript description of a plume flank as
            # a line emerging from the hole exit.
            lg_up[frame_idx] = np.atan(linear_regression_fixed_intercept(ux, uy, 0.0)) * 180.0 / np.pi
            lg_low[frame_idx] = np.atan(linear_regression_fixed_intercept(lx, ly, 0.0)) * 180.0 / np.pi
        except ValueError:
            pass

    opening = np.nan
    closing = np.nan
    if nozzle_opening_detection_height is not None and nozzle_opening_detection_width is not None:
        from OSCC_postprocessing.analysis.hysteresis import detect_single_high_interval

        h0 = (height - nozzle_opening_detection_height) // 2
        h1 = (height + nozzle_opening_detection_height) // 2
        w1 = int(nozzle_opening_detection_width)

        # Opening/closing detection should monitor the plume root immediately
        # downstream of the orifice, not the injector center. The x-window is
        # therefore anchored at ``inner_radius`` so the binary signal is aligned
        # with the same plume-local origin used by the penetration and angle
        # metrics above.
        x0 = max(0, int(np.floor(inner_radius)))
        x1 = min(bw_video_host.shape[2], x0 + w1)
        near_nozzle_signal = np.sum(bw_video_host[:, h0:h1, x0:x1], axis=(1, 2))
        result, _, _ = detect_single_high_interval(near_nozzle_signal)
        if result is not None:
            _, _, opening, closing = result

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
