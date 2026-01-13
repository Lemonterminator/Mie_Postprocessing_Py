import time
import numpy as np
import pandas as pd

from OSCC_postprocessing.analysis.single_plume import (
    bw_boundaries_all_points_single_plume,
    bw_boundaries_xband_filter_single_plume,
    linear_regression_fixed_intercept,
    ransac_fixed_intercept,
    to_numpy,
)
from OSCC_postprocessing.binary_ops.functions_bw import penetration_bw_to_index


def processing_from_binarized_video(bw_video, bw_video_col_sum=None, shift_boundary_y=True, timing=True):
    assert bw_video.ndim == 3
    F, H, W = bw_video.shape

    start_time = time.time()

    

    area = bw_video.sum(axis=(1, 2))

    if bw_video_col_sum is None:
        bw_video_col_sum = bw_video.sum(axis=1)

    estimated_volume = 0.25 * np.pi * np.sum(bw_video_col_sum**2, axis=1)

    upper_bw_width = bw_video[:, : H // 2, :].sum(axis=1)
    lower_bw_width = bw_video[:, H // 2 :, :].sum(axis=1)

    max_plume_radius = np.maximum(upper_bw_width, lower_bw_width)
    min_plume_radius = np.minimum(upper_bw_width, lower_bw_width)

    estimated_volume_max = np.sum(np.pi * max_plume_radius**2, axis=1)
    estimated_volume_min = np.sum(np.pi * min_plume_radius**2, axis=1)

    penetration_old = penetration_bw_to_index(bw_video_col_sum > 0)
    boundary = bw_boundaries_all_points_single_plume(bw_video, parallel=True)

    penetration_old_polar = np.zeros(F)
    for i in range(F):
        pts = boundary[i]
        if len(pts[0]) > 0 and len(pts[1]) > 0:

            uy, ux = pts[1][:, 0], pts[1][:, 1]
            ly, lx = pts[0][:, 0], pts[0][:, 1]

            if shift_boundary_y:
                uy -= H/2.0
                ly -= H/2.0
            max_r_upper = np.max(np.sqrt(uy**2 + ux**2))
            max_r_lower = np.max(np.sqrt(ly**2 + lx**2))
            penetration_old_polar[i] = max(max_r_upper, max_r_lower)

    points_all_frames = bw_boundaries_xband_filter_single_plume(boundary, penetration_old)

    if timing:
        print(f"Binarization and boundary extraction completed in: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    cone_angle_linear_regression = np.full(F, np.nan)
    cone_angle_ransac = np.full(F, np.nan)
    cone_angle_average = np.full(F, np.nan)

    avg_up = np.full(F, np.nan)
    avg_low = np.full(F, np.nan)
    ransac_up = np.full(F, np.nan)
    ransac_low = np.full(F, np.nan)
    lg_up = np.full(F, np.nan)
    lg_low = np.full(F, np.nan)

    for i in range(F):
        points = points_all_frames[i]
        if len(points[0]) > 0 and len(points[1]) > 0:
            uy, ux = points[1][:, 0], points[1][:, 1]
            ly, lx = points[0][:, 0], points[0][:, 1]

            uy -= H // 2
            ly -= H // 2

            ang_up = np.atan(uy / ux) * 180.0 / np.pi
            ang_low = np.atan(ly / lx) * 180.0 / np.pi

            avg_up[i] = np.nanmean(ang_up)
            avg_low[i] = np.nanmean(ang_low)
            cone_angle_average[i] = avg_up[i] - avg_low[i]

            try:
                ransac_up[i] = np.atan(ransac_fixed_intercept(ux, uy, 0)[0]) * 180.0 / np.pi
                ransac_low[i] = np.atan(ransac_fixed_intercept(lx, ly, 0)[0]) * 180.0 / np.pi
                cone_angle_ransac[i] = ransac_up[i] - ransac_low[i]
            except RuntimeError:
                pass

            try:
                lg_up[i] = np.atan(linear_regression_fixed_intercept(ux, uy, 0.0)) * 180.0 / np.pi
                lg_low[i] = np.atan(linear_regression_fixed_intercept(lx, ly, 0.0)) * 180.0 / np.pi
                cone_angle_linear_regression[i] = lg_up[i] - lg_low[i]
            except ValueError:
                pass

    if timing:
        print(f"Cone angle calculations completed in: {time.time() - start_time:.2f} seconds")

    df = pd.DataFrame(
        {
            "Frame": np.arange(F),
            "Penetration_from_BW": to_numpy(penetration_old),
            "Penetration_from_BW_Polar": to_numpy(penetration_old_polar),
            "Cone_Angle_Average": to_numpy(cone_angle_average),
            "Cone_Angle_RANSAC": to_numpy(cone_angle_ransac),
            "Cone_Angle_Linear_Regression": to_numpy(cone_angle_linear_regression),
            "Area": to_numpy(area),
            "Estimated_Volume": to_numpy(estimated_volume),
            "Estimated_Volume_Limit_Upper": to_numpy(estimated_volume_max),
            "Estimated_Volume_Limit_Lower": to_numpy(estimated_volume_min),
            "Cone_Angle_Avg_Upper": to_numpy(avg_up),
            "Cone_Angle_Avg": to_numpy(avg_low),
            "Cone_Angle_Ransac_Upper": to_numpy(ransac_up),
            "Cone_Angle_Ransac_Lower": to_numpy(ransac_low),
            "Cone_Angle_LG_Upper": to_numpy(lg_up),
            "Cone_Angle_LG_Lower": to_numpy(lg_low),
        }
    )
    return df, boundary
