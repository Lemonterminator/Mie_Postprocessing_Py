from OSCC_postprocessing.cine.functions_videos import *
from pathlib import Path
import json
import os
# Importing libraries
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

import matplotlib.pyplot as plt
import numpy as np

from OSCC_postprocessing.analysis.cone_angle import angle_signal_density_auto
from OSCC_postprocessing.binary_ops.functions_bw import bw_boundaries_all_points
from OSCC_postprocessing.rotation.rotate_crop import generate_CropRect, generate_plume_mask
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
warnings.filterwarnings("ignore", category=RuntimeWarning)

global timing
timing = True

if timing:
    import time
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

from OSCC_postprocessing.analysis.single_plume import *
from OSCC_postprocessing.binary_ops.functions_bw import *
from OSCC_postprocessing.analysis.hysteresis import *

def _as_numpy(arr):
    if USING_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def _is_cupy_array(arr):
    return USING_CUPY and hasattr(arr, "__cuda_array_interface__")



# Rotation + Crop + Filtering
def luminesence_preprocessing(
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


def luminescence_pipeline(video: xp.ndarray, file_name: str, 
                             centre, rotation_offset: float, inner_radius: float, outer_radius: float, 
                             video_out_dir: Path, data_out_dir: Path,
                             umbrella_angle = 180.0, # Single Plume has 0 deg of tilt angle (spary axis is orthogonal to line of sight)
                             video_strip_relative_height = 1.0/3, # Relative ratio of the rotated video strip to calibrated outer radius
                             INTERPOLATION = "nearest" ,BORDER_MODE = "constant", # image rotation settings
                             save_video_strip=True, save_mode ="filtered", # filtered rotated strip or raw
                             blank_frames=20,
                             preview=False 
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
    segment, foreground, _ = luminesence_preprocessing(video, centre, rotation_offset, 
                                                        blank_frames=blank_frames,
                                                        outer_radius=outer_radius, preview=preview)
    
    # Save the Foreground video asynchronously
    npz_saver = AsyncNPZSaver(max_workers=2)
    if save_video_strip:
        if save_mode == "filtered":
            avi_saver = AsyncAVISaver(max_workers=2)
            avi_saver.save(video_out_dir / (file_name + ".avi"), _as_numpy(foreground), is_color=False)
            npz_saver.save(video_out_dir / (file_name + ".npz"), foreground=_as_numpy(foreground))
        elif save_mode == "raw":
            avi_saver = AsyncAVISaver(max_workers=2)
            avi_saver.save(video_out_dir / (file_name + ".avi"), _as_numpy(segment), is_color=False)
            npz_saver.save(video_out_dir / (file_name + ".npz"), segment=_as_numpy(segment))


    F, H, W = foreground.shape

    # time-distance intensity map
    xp_backend = cp if _is_cupy_array(foreground) else np
    td_map = xp_backend.sum(foreground, axis=1).T
    
    
    use_gpu_td = _is_cupy_array(td_map)
    if use_gpu_td:
        try:
            bw = keep_largest_component_cuda(
                _triangle_binarize_gpu(_min_max_scale(td_map)),
                connectivity=2,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            warnings.warn(
                f"GPU triangle binarization failed ({exc}); falling back to CPU.",
                RuntimeWarning,
            )
            use_gpu_td = False
    if not use_gpu_td:
        td_map_cpu = _as_numpy(td_map)
        bw_u8, _ = triangle_binarize_from_float(td_map_cpu)
        bw = keep_largest_component(bw_u8 > 0, connectivity=2)

    xp_bw = cp if use_gpu_td else np
    bw_xp = bw if use_gpu_td else np.asarray(bw)

    flame_front = bw_xp.shape[0] - xp_bw.argmax(bw_xp[::-1, :], axis=0)
    flame_front = flame_front.astype(xp_bw.float32)
    flame_front[flame_front == bw_xp.shape[0]] = xp_bw.nan

    flame_lift_off = xp_bw.argmax(bw_xp, axis=0).astype(xp_bw.float32)
    flame_lift_off[flame_lift_off == 0.0] = xp_bw.nan 

    TF = ~xp_bw.isnan(flame_front)

    start, end = longest_true_run(_as_numpy(TF))

    flame_front[:start] = xp_bw.nan
    flame_front[end:] = xp_bw.nan

    if preview:
        plt.plot(_as_numpy(flame_lift_off))
        plt.plot(_as_numpy(flame_front), color="r")
        plt.imshow(_as_numpy(td_map), origin="lower", aspect="auto", cmap="jet")
        plt.title("Penetration extrated from the time-distance intensity heatmap")
        plt.xlabel("Frame number")
        plt.ylabel("Penetration (pxs)")
        plt.colorbar()
        plt.grid()
        plt.show()

    raw_frame_wise_avg_intensity = xp_backend.sum(segment, axis=(1,2))/(1.0*segment.shape[1]*segment.shape[2])
    filtered_frame_wise_avg_intensity = xp_backend.sum(foreground, axis=(1,2))/(1.0*foreground.shape[1]*foreground.shape[2])
    
    if preview:
        plt.plot(_as_numpy(raw_frame_wise_avg_intensity), label="raw")
        plt.plot(_as_numpy(filtered_frame_wise_avg_intensity), label="filtered")
        plt.legend()
        plt.grid()
        plt.title("Average Pixel Value per Frame")
        plt.xlabel("Frame Number")
        plt.ylabel("Value")
        plt.show()



    df = pd.DataFrame()

    # Data handling
    df["Average Pixel Value per Frame (Filtered)"] = np.clip(
        _as_numpy(filtered_frame_wise_avg_intensity), 0.0, 1.0
    )
    df["Average Pixel Value per Frame (Raw)"] = np.clip(
        _as_numpy(raw_frame_wise_avg_intensity), 0.0, 1.0
    )
    df["Flame Front Distance (px)"] = _as_numpy(flame_front)
    df["Flame Lift-Off Distance (px)"] = _as_numpy(flame_lift_off)
    df.to_csv(data_out_dir / (file_name + ".csv"))

    # Save heatmap
    # Save: each row in the image becomes one CSV row
    heatmap_path = data_out_dir / (file_name + "_heatmap.csv")
    np.savetxt(heatmap_path, _as_numpy(td_map), delimiter=",", fmt="%d")
    npz_saver.save(data_out_dir / (file_name + "_heatmap.npz"), heatmap=_as_numpy(td_map))


    if save_video_strip: 
        avi_saver.wait()
        avi_saver.shutdown()
        
    npz_saver.wait()
    npz_saver.shutdown()

    return df
    
def main():


    file = Path(r"G:\MeOH_test\NFL\T56_NFL_Cam_5_experiement.cine")
    json_file = Path(r"G:\MeOH_test\NFL\config.json")
    out_dir = Path(r"G:\MeOH_test\NFL\Processed_Results")

    save_name_stem = Path(file).stem

    Path(out_dir/"Rotated_Videos").mkdir(parents= True, exist_ok=True)
    Path(out_dir / "Postprocessed_Data").mkdir(parents=True, exist_ok=True)

    save_path_video = out_dir / "Rotated_Videos"
    save_path_data = out_dir / "Postprocessed_Data"





    # Video is in uint12 
    video_bits = 12
    brightness_levels = 2.0**video_bits

    # Load the .cine file into a 3D numpy array (gray scale, shape: (Frame, Height, Width))
    video = load_cine_video(file)

    F, H, W = video.shape

    # Normalize the grayscale video to [0, 1] brightness range
    video = video / brightness_levels

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
    df = luminescence_pipeline(video, file.name, centre, offset, ir_, or_, save_path_video, save_path_data, save_video_strip=False, preview=True)

if __name__ == "__main__":
    main()
