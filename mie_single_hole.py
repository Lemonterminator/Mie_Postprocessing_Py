# Importing libraries
from OSCC_postprocessing.cine.functions_videos import *
from pathlib import Path
import json
import os
from OSCC_postprocessing.filters.bilateral_filter import *
from OSCC_postprocessing.binary_ops.functions_bw import *
from OSCC_postprocessing.analysis.multihole_utils import * 
import cupy as cp
from OSCC_postprocessing.analysis.single_plume import _binary_fill_holes_gpu
from OSCC_postprocessing.analysis.multihole_utils import (
    preprocess_multihole,
    resolve_backend,
    rotate_segments_with_masks,
    compute_td_intensity_maps,
    estimate_peak_brightness_frames,
    # estimate_hydraulic_delay,
    compute_penetration_profiles,
    clean_penetration_profiles,
    binarize_plume_videos,
    compute_cone_angle_from_angular_density,
    estimate_offset_from_fft,
    triangle_binarize_gpu as _triangle_binarize_gpu,  # Backward compatibility
)

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from OSCC_postprocessing.analysis.cone_angle import angle_signal_density_auto
from OSCC_postprocessing.rotation.rotate_crop import generate_CropRect
from OSCC_postprocessing.analysis.multihole_utils import (
    preprocess_multihole,
    resolve_backend,
    rotate_segments_with_masks,
    compute_td_intensity_maps,
    estimate_peak_brightness_frames,
    # estimate_hydraulic_delay,
    compute_penetration_profiles,
    clean_penetration_profiles,
    binarize_plume_videos,
    compute_cone_angle_from_angular_density,
    estimate_offset_from_fft,
    triangle_binarize_gpu as _triangle_binarize_gpu,  # Backward compatibility
)

from OSCC_postprocessing.analysis.single_plume import (
    pre_processing_mie,
)

from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
    rotate_video_nozzle_at_0_half_numpy,
)
from OSCC_postprocessing.binary_ops.functions_bw import regionprops_3d, reconstruct_blob


import numpy as np
from OSCC_postprocessing.io.async_npz_saver import AsyncNPZSaver
from OSCC_postprocessing.io.async_avi_saver import *
from OSCC_postprocessing.filters.video_filters import *
from OSCC_postprocessing.playback.video_playback import *
from OSCC_postprocessing.analysis.single_plume import (
    USING_CUPY,
    cp,
    _min_max_scale,
    _rotate_align_video_cpu,
    binarize_single_plume_video,
    bw_boundaries_all_points_single_plume,
    bw_boundaries_xband_filter_single_plume,
    filter_schlieren,
    linear_regression_fixed_intercept,
    penetration_bw_to_index,
    pre_processing_mie,
    ransac_fixed_intercept,
    save_boundary_csv,
    to_numpy,
)
from OSCC_postprocessing.analysis.cone_angle import angle_signal_density_auto
from OSCC_postprocessing.binary_ops.binarized_metrics import processing_from_binarized_video
import pandas as pd

# Import rotation utility based on backend availability to avoid hard Cupy dependency
if USING_CUPY:
    from OSCC_postprocessing.rotation.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy as rotate_video_nozzle_at_0_half_backend,
    )
else:
    from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as rotate_video_nozzle_at_0_half_backend,
    )

from OSCC_postprocessing.analysis.hysteresis import *

warnings.filterwarnings("ignore", category=RuntimeWarning)
use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

global timing
timing = True

def _as_numpy(arr):
    if USING_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def plot_metrics_dataframe(df, title="", save_path=None):
    """Plot all numeric columns from a metrics DataFrame. Optionally save to file."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        if save_path is None:
            print("No numeric columns to plot.")
        return
    n_cols = len(numeric_cols)
    n_rows = max(1, (n_cols + 2) // 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 2.5 * n_rows), squeeze=False)
    axes = axes.flatten()
    x = np.arange(len(df))
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.plot(x, df[col], linewidth=1.0)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Frame")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle("Single-hole Mie metrics", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _compute_full_solver_metrics_from_bw(
    bw_video,
    H: int,
    F: int,
    umbrella_angle: float,
):
    bw_video_col_sum = bw_video.sum(axis=1)
    area = bw_video_col_sum.sum(axis=-1)
    penetration_bw_x = penetration_bw_to_index(bw_video_col_sum > 0)
    boundary = bw_boundaries_all_points_single_plume(bw_video, parallel=True, umbrella_angle=180.0)

    if umbrella_angle == 180.0:
        x_scale = 1.0
    else:
        tilt_angle = (180.0 - umbrella_angle) / 2.0
        tilt_angle_rad = tilt_angle / 180.0 * np.pi
        x_scale = 1.0 / np.cos(tilt_angle_rad)

    upper_bw_width = bw_video[:, : H // 2, :].sum(axis=1)
    lower_bw_width = bw_video[:, H // 2 :, :].sum(axis=1)

    estimated_volume = x_scale * np.pi * 0.25 * np.sum((upper_bw_width + lower_bw_width) ** 2, axis=1)

    max_plume_radius = np.maximum(upper_bw_width, lower_bw_width)
    min_plume_radius = np.minimum(upper_bw_width, lower_bw_width)

    estimated_volume_max = np.pi * x_scale * np.sum(max_plume_radius**2, axis=1)
    estimated_volume_min = np.pi * x_scale * np.sum(min_plume_radius**2, axis=1)

    penetration_bw_polar = np.zeros(F)
    for i in range(F):
        pts = boundary[i]
        if len(pts[0]) > 0 and len(pts[1]) > 0:
            uy, ux = pts[1][:, 0], pts[1][:, 1]
            ly, lx = pts[0][:, 0], pts[0][:, 1]

            max_r_upper = np.max(np.sqrt(uy**2 + ux**2))
            max_r_lower = np.max(np.sqrt(ly**2 + lx**2))
            penetration_bw_polar[i] = max(max_r_upper, max_r_lower)

    points_all_frames = bw_boundaries_xband_filter_single_plume(boundary, penetration_bw_x.get())

    lg_up = np.full(F, np.nan)
    lg_low = np.full(F, np.nan)
    avg_up = np.full(F, np.nan)
    avg_low = np.full(F, np.nan)

    for i in range(F):
        points = points_all_frames[i]
        if len(points[0]) > 0 and len(points[1]) > 0:
            uy, ux = points[1][:, 0], points[1][:, 1]
            ly, lx = points[0][:, 0], points[0][:, 1]

            ang_up = np.atan(uy / ux) * 180.0 / np.pi
            ang_low = np.atan(ly / lx) * 180.0 / np.pi

            avg_up[i] = np.nanmean(ang_up)
            avg_low[i] = np.nanmean(ang_low)

            try:
                lg_up[i] = np.atan(linear_regression_fixed_intercept(ux, uy, 0.0)) * 180.0 / np.pi
                lg_low[i] = np.atan(linear_regression_fixed_intercept(lx, ly, 0.0)) * 180.0 / np.pi
            except ValueError:
                pass

    cone_angle_average = avg_up - avg_low
    cone_angle_linear_regression = lg_up - lg_low

    return {
        "area": area,
        "penetration_bw_x": penetration_bw_x,
        "boundary": boundary,
        "estimated_volume": estimated_volume,
        "estimated_volume_max": estimated_volume_max,
        "estimated_volume_min": estimated_volume_min,
        "penetration_bw_polar": penetration_bw_polar,
        "cone_angle_average": cone_angle_average,
        "avg_up": avg_up,
        "avg_low": avg_low,
        "cone_angle_linear_regression": cone_angle_linear_regression,
        "lg_up": lg_up,
        "lg_low": lg_low,
    }



# Rotation + Crop + Filtering
def mie_preprocessing(
    video,
    centre,
    rotation_offset,
    video_strip_relative_height=1.0 / 3,
    INTERPOLATION="nearest",
    BORDER_MODE="constant",
    wsize=7,
    sigma_d=3.0,
    sigma_r=3.0,
    blank_frames=10,
    outer_radius=None,
    preview=True,
):
    
    # 3D grayscale Numpy array
    F0 = video.shape[0]

    if outer_radius is None:
        outer_radius = globals().get("or_")
    if outer_radius is None:
        raise ValueError("outer_radius must be provided or set as global 'or_'.")

    out_h = int(round(float(video_strip_relative_height) * float(outer_radius)))
    out_h = max(1, out_h)
    out_w = int(round(float(outer_radius)))
    if out_w <= 0:
        raise ValueError("outer_radius must be positive.")

    blank_frames = max(1, min(int(blank_frames), F0))

    # Arbitrary rotated image strip shape
    OUT_SHAPE = (out_h, out_w)

    use_gpu = USING_CUPY
    if use_gpu:
        try:
            # Upload to GPU 
            # video_cp = cp.asarray(video)
            segment, _, _ = rotate_video_nozzle_at_0_half_backend(
                video,
                centre,
                rotation_offset,
                interpolation=INTERPOLATION,
                border_mode=BORDER_MODE,
                out_shape=OUT_SHAPE,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"GPU rotation failed ({exc}), falling back to CPU numpy implementation.")
            use_gpu = False
    if not use_gpu:
        segment = _rotate_align_video_cpu(
            to_numpy(video),
            centre,
            rotation_offset,
            interpolation=INTERPOLATION,
            border_mode=BORDER_MODE,
            out_shape=OUT_SHAPE,
            cval=0.0,
        )

    # Bilateral filtering
    if use_gpu:
        bilateral_filtered = bilateral_filter_video_cupy(segment, wsize, sigma_d, sigma_r)
    else:
        bilateral_filtered = bilateral_filter_video_cpu(np.asarray(segment), wsize, sigma_d, sigma_r)
    xp = cp if use_gpu else np

    # Background subtraction
    # Take the filtered first frames as background
    bkg = xp.mean(bilateral_filtered[:blank_frames], axis=0, keepdims=True)

    eps = xp.asarray(1e-9, dtype=bkg.dtype)
    bkg = xp.where(bkg == 0, eps, bkg)
    bkg = xp.where(xp.isnan(bkg), eps, bkg)

    # Foreground is the filtered video - filtered background
    foreground = bilateral_filtered - bkg

    if preview:
        play_videos_side_by_side(
            (
                _as_numpy(xp.swapaxes(segment, 1, 2)),
                _as_numpy(xp.swapaxes(foreground, 1, 2)),
                _as_numpy(xp.swapaxes(10.0 * xp.abs(foreground - segment), 1, 2)),
            ),
            intv=17,
        )

    return segment, foreground, bkg


def mie_single_hole_pipeline(video: xp.ndarray, file_name: str,
                             centre, rotation_offset: float, inner_radius: float, outer_radius: float,
                             video_out_dir: Path, data_out_dir: Path,
                             save_path_plot: Path | None = None,
                             blank_frames=10,  # usually stable in a campaign.
                             umbrella_angle=180.0,  # Single Plume: spray axis orthogonal to line of sight
                             video_strip_relative_height=1.0 / 3,
                             INTERPOLATION="nearest", BORDER_MODE="constant",
                             save_video_strip=True, save_mode="filtered",
                             near_nozzle_relative_height=1.0 / 20, near_nozzle_relative_width=1.0 / 20,
                             solver="full",
                             preview=False,
                             maximum_bw_spray_tolerance_y_axis = 50, #px,
                             nozzle_open_threshold_low=0.1,
                             nozzle_open_threshold_high=0.5,
                             quantize_npz: bool = False,
                             quant_float_upper_bound: float = 1.0,
                             quant_clip_negative: bool = True,
                             quant_store_metadata: bool = True,
                             ): 
    

    

    if preview:
        # Debug inputs
        print("Processing Cine file: ", file_name)
        print("Video has shape: ", video.shape)
        print("Nozzle centred at: ", centre[0], centre[1])
        print("Degrees of rotation:", rotation_offset)
        print("Saving rotated video strip to directory: ", video_out_dir)
        print("Saving data to directory: ", data_out_dir)
    
    if USING_CUPY:
        if isinstance(video, cp.ndarray):
            pass
        if isinstance(video, np.ndarray):
            
            video = cp.asarray(video)
        

    start_time = time.time()

    # Rotation + Filtering + Background Subtraction
    segment, foreground, _ = mie_preprocessing(video, centre, rotation_offset, 
                                                        blank_frames=blank_frames,
                                                        outer_radius=outer_radius, preview=preview)

    print(f"Preprocessing finished in {time.time()-start_time:.2} s")
    F, H, W = foreground.shape


    # Optionally save the rotated video as .npz or .avi for visualization
    if save_video_strip==True:
        avi_saver = AsyncAVISaver()
        npz_saver = AsyncNPZSaver()
        if save_mode == "raw": 
            npz_saver.save(
                video_out_dir / f"{file_name}_rotated_strip.npz",
                quantize_u8=quantize_npz,
                quant_float_upper_bound=quant_float_upper_bound,
                quant_clip_negative=quant_clip_negative,
                quant_store_metadata=quant_store_metadata,
                segment=_as_numpy(segment),
            )
            avi_saver.save(
                video_out_dir / f"{file_name}_rotated_strip.avi",
                _as_numpy(segment),
                fps=20,
                is_color=False,
                auto_normalize=True,
            )
        elif save_mode=="filtered":
            # Save filtered foreground video to video_out_dir (Rotated_Videos folder)
            npz_saver.save(
                video_out_dir / f"{file_name}_foreground.npz",
                quantize_u8=quantize_npz,
                quant_float_upper_bound=quant_float_upper_bound,
                quant_clip_negative=quant_clip_negative,
                quant_store_metadata=quant_store_metadata,
                foreground=_as_numpy(foreground),
            )
            avi_saver.save(
                video_out_dir / f"{file_name}_foreground.avi",
                _as_numpy(foreground),
                fps=10,
                is_color=False,
                auto_normalize=True,
            )



    # Nozzle Closing and Opening time

    # Nozzle Opening and Closing

    # Window Size in the left center of the video

    near_nozzle_video_patch = foreground[:, 
                                         int(H//2 - H*near_nozzle_relative_height//2):int(H//2 + H*near_nozzle_relative_height//2), 
                                         :int(W*near_nozzle_relative_width)
                                         ]
    near_nozzle_intensity_sums = xp.sum(near_nozzle_video_patch, axis=(1, 2))


    Lo_Hi = cp.zeros_like(near_nozzle_intensity_sums, dtype=cp.bool_)


    # y =_min_max_scale( near_nozzle_intensity_sums.T) # cupy -> numpy
    y = robust_scale(near_nozzle_intensity_sums.T, q_min=15, q_max=90)
    
    res, mask, _ = detect_single_high_interval(y, th_lo=nozzle_open_threshold_low, th_hi=nozzle_open_threshold_high)
    
    if res is None:
        hydraulic_delay = np.nan
        nozzle_closing = np.nan
        Lo_Hi[:] = False
    else:    

        (hd, nc, _, _) = res   # 这里 hd/nc 是 x_start/x_end（如果你 x=None）
        hydraulic_delay = hd.get().item()
        
        # Shifting hydraulic delay by 1 frame 
        hd -= 1
        hydraulic_delay -= 1 
        
        nozzle_closing = nc.get().item()

        Lo_Hi = cp.asarray(mask, dtype=cp.bool_)
    
        
    # print("Hydraulic delay:", hd)
    # print("Nozzle closing:", nc)



    # Time-Distance Intensity Map
    td_map = 1.0/H * xp.sum(foreground, axis=1).T

    if preview:
        plt.imshow(_as_numpy(td_map), origin="lower", aspect="auto", cmap="jet")
        plt.title("Time-Distance Intensity Heatmap")
        plt.colorbar()
        plt.xlabel("Frame number")
        plt.ylabel("pixels")
        plt.axvline(hydraulic_delay, label="Hydraulic Delay", color="r")
        plt.axvline(nozzle_closing, label="Nozzle Closing", color="k")
        plt.legend()
        plt.show()

    # Find average pixel intensity in each frame
    average_pixel_intensity = 1.0/ W* xp.sum(td_map, axis=0)

    # Find the largest   
    brightness_peak = xp.argmax(average_pixel_intensity)

    # Find the frame where the average intensity is the largest
    peak_intensity_sum = xp.max(average_pixel_intensity)

    # Ratio of average intensity of each frame against the brightest frame
    ratio = average_pixel_intensity/peak_intensity_sum

    if preview:
        plt.plot(_as_numpy(ratio))
        plt.grid()
        plt.axvline(_as_numpy(brightness_peak).item())
        plt.title(f"Total Frame Intensitites, Peak at Frame: {_as_numpy(brightness_peak)}")
        plt.ylabel("Relative Total Intensity Ratio to the Brightest Frame")
        plt.xlabel("Frame Number")
        plt.axvline(hydraulic_delay, label="Hydraulic Delay", color="r")
        plt.axvline(nozzle_closing, label="Nozzle Closing", color="k")
        plt.legend()
        plt.show() 

    # Compute a gain correction using the 1/ratio, ratio needs to += epsilon to avoid overflow
    eps = 1e-9
    gain_curve = 1.0 / (eps + ratio.astype(xp.float64))

    gain_curve[:int(brightness_peak)] = 1.0

    if preview: 
        plt.plot(_as_numpy(gain_curve))
        plt.axvline(_as_numpy(brightness_peak).item())
        plt.title("Gain compensation curve for frames after the peak")
        plt.grid()
        plt.xlabel("Frame Number")
        plt.ylabel("Gain")
        plt.show()
    
    # Gain correction for frames after the brightest frame, so pixels with lower value won't 
    # Be binarized as 0, therefore improving accuracy

    # Gain correction for the time-distance intensity map
    td_map[:, int(brightness_peak):] *= xp.asarray(gain_curve[None,  int(brightness_peak):])
    
    df = pd.DataFrame()

    if use_gpu:
        # Compute the penetration by binarizing the gain-corrected, time-distance intensity map

        # Min-Max scale, triangular binarize, then keep the largest component (AKA connected area in 2D)
        bw =  keep_largest_component_cuda(
                    triangle_binarize_gpu(
                    _min_max_scale(td_map))
                    , connectivity=2)
        
        # Find the left edge as penetration curve
        penetration_TD = bw.shape[0] - cp.argmax(bw[::-1, :], axis=0)
        penetration_TD = penetration_TD.astype(cp.float32)
        penetration_TD[penetration_TD == bw.shape[0]] = cp.nan

        # Mask
        TF = ~cp.isnan(penetration_TD)

        # Find the longest sub-sequence that is non-nan
        start, end = longest_true_run(_as_numpy(TF))

        # Reset the penetration before SOI to 0.0
        penetration_TD[:start] = 0.0
        # Reset the penetration after saturation to nan
        penetration_TD[end:] = cp.nan
    else:
        # TODO: implement an efficient CPU version
        raise NotImplementedError

    if preview:
        plt.plot(_as_numpy(penetration_TD), color="r")
        plt.imshow(_as_numpy(td_map), origin="lower", aspect="auto", cmap="jet")
        plt.title("Penetration extrated from the time-distance intensity heatmap")
        plt.xlabel("Frame number")
        plt.ylabel("Penetration (pxs)")
        plt.axvline(hydraulic_delay, label="Hydraulic Delay", color="r")
        plt.axvline(nozzle_closing, label="Nozzle Closing", color="k")
        plt.legend()
        plt.colorbar()
        plt.grid()
        plt.show()
    
    # Cone angle

    # We set the origin to the nozzle, and treat upper and lower half as two plumes
    # Then calculate their cone angle by angular density respectively

    accuracy = 0.1 # degrees

    n_bins = int(360.0/accuracy)
    _, signal, _ = angle_signal_density_auto(foreground, 0.0, H//2, N_bins=n_bins)
    AngularDensity = compute_cone_angle_from_angular_density(signal, 0, 2, bins=n_bins, use_gpu=use_gpu)
 
    # Upper & Lower cone angle, sums up to the total cone angle
    cone_angle_AD_up    = AngularDensity[0]
    cone_angle_AD_down  = AngularDensity[1]

    # Cone angle is the sum of the two, since both are defined positive
    cone_angle_AD       = cone_angle_AD_up + cone_angle_AD_down

    if preview:
        plt.plot(_as_numpy(cone_angle_AD),     label="Total Cone Angle")
        plt.plot(_as_numpy(cone_angle_AD_down),        label="Upper Cone Angle") 
        plt.plot(_as_numpy(cone_angle_AD_up),          label="Lower Cone Angle")
        
        plt.grid()
        plt.title("Cone Angle From Angular Intensity Density")
    
        plt.axvline(hydraulic_delay, label="Hydraulic Delay", color="r")
        plt.axvline(nozzle_closing, label="Nozzle Closing", color="k")
        plt.legend()
        plt.xlabel("Frame Number")
        plt.ylabel("Cone Angle (Degrees)")

        plt.show()


    if solver=="full":

        if preview:
            fg_copy = foreground.copy() 

        '''
        # Gain correction for the video
        # foreground[int(brightness_peak):] *= xp.asarray(gain_curve[int(brightness_peak):, None, None])

        
        # Binarize the whole video with a global triangular threshold
        # Then retain the largest 3D blob 
        # (Reasoning: All bw area in each frame is assumed to be connected to the previous and the next, 
        # By sharing at least 1 common pixel. Therefore the spray should be one blob)
        # Finally, fill holes in each frame in 2D mode
        bw_video = _binary_fill_holes_gpu(
                        keep_largest_component_nd_cuda(
                        triangle_binarize_gpu(foreground)
                        , connectivity=2),
                        mode="2D"
                        )
        '''
        
        bw_video_0 = triangle_binarize_gpu(_min_max_scale(foreground))

        # Same connectivity as keep_largest_component_nd_cuda (e.g. 2 for 18-neighbors in 3D)
        
        # Check for regional Properties in 3D
        # Get label image to reconstruct blobs
        props, labels = regionprops_3d(bw_video_0, connectivity=2, return_labels=True, centroid=True)

        # Find the L2 distance for each 3D blob from the y = H//2 axis
        props["y-dist"] = np.abs(props["centroid_1"] - bw_video_0.shape[1]//2)
        
        # Distance based filtering
        filtered_props = props[props["y-dist"] <= maximum_bw_spray_tolerance_y_axis]

        # Retain the largest blob
        filtered_props = filtered_props.sort_values("volume", ascending=False)

        # Get the value from the "label" column of the first row (at position 0)
        label_id = filtered_props.iloc[0]["label"]

        # Reconstruct the video based on filtered labels
        largest_blob = reconstruct_blob(labels, label_id)

        # Per frame keep the largest white 2d blob. Reason: Connected in 3D != Connected in 2D
        for f in range(F):
            largest_blob[f] = keep_largest_component_cuda(largest_blob[f])

        bw_video = _binary_fill_holes_gpu(largest_blob, mode="2D")  # or (labels == label_id)


        full_metrics = _compute_full_solver_metrics_from_bw(
            bw_video=bw_video,
            H=H,
            F=F,
            umbrella_angle=umbrella_angle,
        )
        area = full_metrics["area"]
        penetration_bw_x = full_metrics["penetration_bw_x"]
        boundary = full_metrics["boundary"]
        estimated_volume = full_metrics["estimated_volume"]
        estimated_volume_max = full_metrics["estimated_volume_max"]
        estimated_volume_min = full_metrics["estimated_volume_min"]
        penetration_bw_polar = full_metrics["penetration_bw_polar"]
        cone_angle_average = full_metrics["cone_angle_average"]
        avg_up = full_metrics["avg_up"]
        avg_low = full_metrics["avg_low"]
        cone_angle_linear_regression = full_metrics["cone_angle_linear_regression"]
        lg_up = full_metrics["lg_up"]
        lg_low = full_metrics["lg_low"]


        if preview:
            # Visualize the filtered video, it after gain correction, 
            # and the frame wise difference between these two
            play_videos_side_by_side(
                (
                    _as_numpy(xp.swapaxes(fg_copy, 1, 2)),
                    _as_numpy(xp.swapaxes(foreground, 1, 2)),
                    _as_numpy(xp.swapaxes(xp.abs(foreground - fg_copy), 1, 2)),
                ),
                intv=17,
            )
    
        if preview:
            plt.plot(_as_numpy(penetration_bw_x), label="Penetration from BW: X distance")
            plt.plot(_as_numpy(penetration_TD), label="Penetration from TD heatmap")
            plt.plot(_as_numpy(penetration_bw_polar), label="Penetration from BW: Polar distance")
            plt.title("Penetration Algorithm Comparisons")
            plt.xlabel("xlabel=Frame number")
            plt.ylabel("Penetration (px)")
            plt.grid()
            plt.legend()
        # print(f"Time taken for classical cone angle calculations: {time.time()-start_time}")

    # start_time = time.time()
    # Save penetration from the time-distance intensity map
    df["Penetration_from_TD"]                   = _as_numpy(penetration_TD)
    df["Cone_Angle_Angular_Density"]            = _as_numpy(cone_angle_AD)
    df["Cone_Angle_Angular_Density_Upper"]      = _as_numpy(cone_angle_AD_up)
    df["Cone_Angle_Angular_Density_Lower"]      = _as_numpy(cone_angle_AD_down)
    df["Hydraulic Delay"]                       = _as_numpy(hd)
    df["Nozzle Closing"]                        = _as_numpy(nc)

    executor = None
    if solver == "full":
        # Asynchronously saving the boundary
        executor = ThreadPoolExecutor(max_workers=4)
        save_boundary_csv(boundary, data_out_dir / f"{file_name}_boundary_points.csv", executor=executor)
        
        df["Penetration_from_BW"] = _as_numpy(penetration_bw_x)
        df["Penetration_from_BW_Polar"] = _as_numpy(penetration_bw_polar)
        df["Area"] = _as_numpy(area)
        df["Estimated_Volume"] = _as_numpy(estimated_volume)
        df["Estimated_Volume_Upper_Limit"] = _as_numpy(estimated_volume_max)
        df["Estimated_Volume_Lower_Limit"] = _as_numpy(estimated_volume_min)


        df["Cone_Angle_Average"] = _as_numpy(cone_angle_average)
        df["Cone_Angle_Average_Lower"] = _as_numpy(cone_angle_AD_down)
        df["Cone_Angle_Average_Upper"] = _as_numpy(cone_angle_AD_up)
        df["Cone_Angle_Linear_Regression"] = _as_numpy(cone_angle_linear_regression)
        df["Cone_Angle_Linear_Regression_Lower"] = _as_numpy(lg_low)
        df["Cone_Angle_Linear_Regression_Upper"] = _as_numpy(lg_up)


    # Save main data
    df.to_csv(data_out_dir / f"{file_name}_metrics.csv")

    if save_path_plot is not None:
        plot_path = Path(save_path_plot) / f"{file_name}_metrics.png"
        plot_metrics_dataframe(df, title=file_name + "\n", save_path=plot_path)

    # Shutting down the asynchronous data savers
    if save_video_strip:
        avi_saver.wait()
        avi_saver.shutdown()
        npz_saver.wait()
        npz_saver.shutdown()
    if executor is not None:
        executor.shutdown(wait=True)
    return df


def main():
    file = Path(r"G:\MeOH_test\Mie\T15_Mie Camera_1.cine")
    json_file = Path(r"G:\MeOH_test\Mie\config.json")
    out_dir = Path(r"G:\MeOH_test\Mie\Processed_Results")

    save_name_stem = Path(file).stem

    Path(out_dir/"Rotated_Videos").mkdir(parents= True, exist_ok=True)
    Path(out_dir / "Postprocessed_Data").mkdir(parents=True, exist_ok=True)

    save_path_video = out_dir / "Rotated_Videos"
    save_path_data = out_dir / "Postprocessed_Data"
    save_path_plot = out_dir / "Plots"
    Path(save_path_plot).mkdir(parents=True, exist_ok=True)


    umbrella_angle=180.0
    # Video is in uint12 
    video_bits = 12
    brightness_levels = 2.0**video_bits

    # Load the .cine file into a 3D numpy array (gray scale, shape: (Frame, Height, Width))
    video = load_cine_video(file)


    F, H, W = video.shape

    # Check the hardware used for image processing

    use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

    print("CUDA is used:", use_gpu)

    print("xp represents numpy or cupy? :",xp)

    # Normalize the grayscale video to [0, 1] brightness range
    if use_gpu: 
        video = cp.asarray(video, dtype=cp.float16)     # 直接生成 float16，少一次 astype 临时
        video *= cp.float16(1.0 / brightness_levels)              # 就地，不产生新数组
    else:
        video *= np.float16(1.0 / brightness_levels)


    with open(json_file, 'r', encoding='utf-8') as f:
        # Load metadata
        data = json.load(f)
        number_of_plumes = int(data['plumes'])
        offset = float(data['offset']) # Not used in multi hole (Calculated later by FFT)
        centre = (float(data['centre_x']), float(data['centre_y']))
        ir_ = float(data["inner_radius"])   # inner radius (Injector radius)
        or_ = float(data["outer_radius"])   # outer radius (Quatz window radius)

    print(f"The injector has {number_of_plumes} plumes.")
    print(f"The nozzle is centred at ({centre[0]:.2f}, {centre[1]:.2f}) in image coordinates.")
    
    df = mie_single_hole_pipeline(
        video, file.name, centre, offset, ir_, or_,
        save_path_video, save_path_data, save_path_plot=save_path_plot,
        save_video_strip=False, preview=True,
    )

    print("finsished!")
if __name__ == '__main__':
    main()
