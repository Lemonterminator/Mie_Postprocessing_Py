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


from OSCC_postprocessing.analysis.multihole_utils import (
    preprocess_multihole,
    resolve_backend,
    rotate_segments_with_masks,
    compute_td_intensity_maps,
    estimate_peak_brightness_frames,
    estimate_hydraulic_delay,
    compute_penetration_profiles,
    clean_penetration_profiles,
    binarize_plume_videos,
    compute_cone_angle_from_angular_density,
    estimate_offset_from_fft,
    triangle_binarize_gpu as _triangle_binarize_gpu,  # Backward compatibility
)

# Import rotation utility based on backend availability to avoid hard Cupy dependency
if USING_CUPY:
    from OSCC_postprocessing.rotation.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy as rotate_video_nozzle_at_0_half_backend,
    )
else:
    from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as rotate_video_nozzle_at_0_half_backend,
    )

#  Main pipeline function
def singlehole_pipeline(mode, video, offset, centre, file_name, 
                                  rotated_vid_dir, data_dir, 
                                  save_intermediate_results=True,
                                  FPS=20, lighting_unchanged_duration=50, TD_sum_interval=0.5
                                  ):
    F, H, W = video.shape
    shock_wave_duration = 50
    # Rotate and align the video based on nozzle centre and offset
    INTERPOLATION = "nearest"
    BORDER_MODE = "constant"
    OUT_SHAPE = (H // 2, W)
    rotate_start = time.time()
    if USING_CUPY:
        try:
            rotated_gpu, _, _ = rotate_video_nozzle_at_0_half_backend(
                video,
                centre,
                offset,
                interpolation=INTERPOLATION,
                border_mode=BORDER_MODE,
                out_shape=OUT_SHAPE,
            )
            rotated = cp.asarray(rotated_gpu, dtype=cp.float32)
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"GPU rotation failed ({exc}), falling back to CPU numpy implementation.")
            rotated_np = _rotate_align_video_cpu(
                video,
                centre,
                offset,
                interpolation=INTERPOLATION,
                border_mode=BORDER_MODE,
                out_shape=OUT_SHAPE,
                cval=0.0,
            )
            rotated = cp.asarray(rotated_np, dtype=cp.float32)
    else:
        rotated_np = _rotate_align_video_cpu(
            video,
            centre,
            offset,
            interpolation=INTERPOLATION,
            border_mode=BORDER_MODE,
            out_shape=OUT_SHAPE,
            cval=0.0,
        )
        rotated = cp.asarray(rotated_np, dtype=cp.float32)


    # Clip the negative number in the rotated video
    rotated = cp.clip(rotated, 0.0, None)

    # Normalize the rotated video to [0, 1]
    rotated = _min_max_scale(rotated)
    print(f"Rotation + normalization completed in: {time.time() - rotate_start:.2f} seconds")

    avi_saver = AsyncAVISaver(max_workers=4)
    if save_intermediate_results:
        # np.savez(data_SCH / f"{file_name}_rotated.npy", to_numpy(rotated))
        AsyncNPZSaver().save(rotated_vid_dir / f"{file_name}_rotated.npz", rotated=to_numpy(rotated))
        
        f1 = avi_saver.save(
            rotated_vid_dir / f"{file_name}_rotated.avi",
            to_numpy(rotated),
            fps=FPS,
            is_color=False,
            auto_normalize=True,
        )

    if mode == "Schlieren": 
        filtered = filter_schlieren(rotated, shock_wave_duration)

    if mode == "Mie":
        # hydraulic_delay = 15
        
        # filter_mie(rotated)
        # segments, penetration, cone_angle_AngularDensity  = filter_mie(rotated)
        # filtered = segments[0]

        # Enhancing the foreground
        start_time = time.time()
        foreground = pre_processing_mie(rotated)
        print(f"Pre-processing finished in: {time.time() - start_time:.2f} seconds")
        F, H, W = foreground.shape  # dimensions after rotation/cropping
        
        if save_intermediate_results:
            # Save the Foreground video asynchronously
            AsyncNPZSaver().save(data_dir / f"{file_name}_foreground.npz", foreground=to_numpy(foreground))
            
            f2 = avi_saver.save(
                data_dir / f"{file_name}_foreground.avi",
                to_numpy(foreground),
                fps=FPS,
                is_color=False,
                auto_normalize=True,
            )

        # Time-distance intensity based penetration
        td_start = time.time()
        if TD_sum_interval > 0.0 and TD_sum_interval < 1.0:
            half_band = int(H * TD_sum_interval / 2)
            foreground_col_sum = cp.sum(
                foreground[H // 2 - half_band : H // 2 + half_band],
                axis=1,
            )
        else:
            foreground_col_sum = cp.sum(foreground, axis=1)
        foreground_energy = cp.sum(foreground_col_sum, axis=1)

        # Find the frame with the brightest near-nozzle region to estimate hydraulic delay
        _, peak_idx, _ = estimate_peak_brightness_frames(foreground, use_gpu=True)
        hydraulic_delay = estimate_hydraulic_delay(foreground[None, :, :, :], peak_idx, use_gpu=True)[0]

        penetration_td = compute_penetration_profiles(
            foreground_col_sum[None, :, :],
            foreground_energy[None, :],
            cp.asarray([hydraulic_delay]),
            cp.asarray([peak_idx]),
            use_gpu=True,
            lower=0,
            upper=W,
        )
        print(f"TD penetration completed in: {time.time() - td_start:.2f} seconds")


        # Estimating hydarulic delay


        # Cone angle
        # We set the origin to the nozzle, and treat upper and lower half as two plumes
        # Then calculate their cone angle by angular density respectively
        _, signal, _ = angle_signal_density_auto(foreground, 0.0, H//2, N_bins=3600)
        AngularDensity = compute_cone_angle_from_angular_density(signal, 0, 2, bins=3600, use_gpu=True)

        # Upper & Lower cone angle, sums up to the total cone angle
        cone_angle_AD_up = AngularDensity[0]
        cone_angle_AD_low = AngularDensity[1]
        cone_angle_AD = cone_angle_AD_up + cone_angle_AD_low
        ad_up_np = to_numpy(cone_angle_AD_up)
        ad_low_np = to_numpy(cone_angle_AD_low)

        # Shape
        frame_idx = np.arange(F, dtype=np.int32)

        # binarizin the video
        # Note: bw_vid_col_sum is the column wise sum 
        bw_video, _ = binarize_single_plume_video(
            foreground,
            hydraulic_delay,
            lighting_unchanged_duration=lighting_unchanged_duration,
            hole_fill_mode="2D",
        )
        bw_video_col_sum = np.sum(bw_video, axis=1)
        df_bw, boundary = processing_from_binarized_video(bw_video, bw_video_col_sum, timing=True)
        penetration_old = df_bw["Penetration_from_BW"].to_numpy()
        penetration_old_polar = df_bw["Penetration_from_BW_Polar"].to_numpy()
        cone_angle_average = df_bw["Cone_Angle_Average"].to_numpy()
        cone_angle_ransac = df_bw["Cone_Angle_RANSAC"].to_numpy()
        cone_angle_linear_regression = df_bw["Cone_Angle_Linear_Regression"].to_numpy()
        avg_up = df_bw["CA_Avg_Upper"].to_numpy()
        avg_low = df_bw["CA_Avg_Lower"].to_numpy()
        ransac_up = df_bw["CA_Ransac_Upper"].to_numpy()
        ransac_low = df_bw["CA_Ransac_Lower"].to_numpy()
        lg_up = df_bw["CA_LG_Upper"].to_numpy()
        lg_low = df_bw["CA_LG_Lower"].to_numpy()


        plot_start = time.time()
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            from pathlib import Path
            from OSCC_postprocessing.io.async_plot_saver import AsyncPlotSaver
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"Skipping comparison plots (matplotlib unavailable): {exc}")
        else:
            plot_saver = AsyncPlotSaver(max_workers=2)

            fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            angle_series = {
                "Average (upper-lower)": cone_angle_average,
                "Linear regression": cone_angle_linear_regression,
                "RANSAC": cone_angle_ransac,
                "Angular density": ad_up_np + ad_low_np,
            }
            for label, values in angle_series.items():
                axes1[0].plot(frame_idx, values, label=label, linewidth=1.4)
            axes1[0].set_ylabel("Cone angle (deg)")
            axes1[0].grid(True)
            axes1[0].legend()

            axes1[1].plot(frame_idx, avg_up, label="Upper mean angle", linewidth=1.2)
            axes1[1].plot(frame_idx, -avg_low, label="Lower mean angle", linewidth=1.2)
            axes1[1].plot(frame_idx, ad_up_np, label="AD upper", linewidth=1.2)
            axes1[1].plot(frame_idx, ad_low_np, label="AD lower", linewidth=1.2)
            axes1[1].set_xlabel("Frame")
            axes1[1].set_ylabel("Edge angle (deg)")
            axes1[1].grid(True)
            axes1[1].legend()

            output_dir = Path(data_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"{Path(file_name).stem}_cone_angle_comparison.png"
            plot_saver.submit(fig1, plot_path)

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(frame_idx, penetration_old, label="BW penetration (x-axis)", linewidth=1.4)
            ax2.plot(frame_idx, penetration_old_polar, label="BW penetration (polar)", linewidth=1.4)
            pen_td_np = to_numpy(penetration_td[0])
            n_plot = min(len(frame_idx), len(pen_td_np))
            ax2.plot(frame_idx[:n_plot], pen_td_np[:n_plot], label="TD penetration", linewidth=1.4)
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Penetration (pixels)")
            ax2.grid(True)
            ax2.legend()
            plot_path2 = output_dir / f"{Path(file_name).stem}_penetration_comparison.png"
            plot_saver.submit(fig2, plot_path2)


            plot_saver.shutdown(wait=True)
            print(f"Plot generation completed in: {time.time() - plot_start:.2f} seconds")

    def _pad_series(values, length):
        arr = np.asarray(to_numpy(values)).ravel()
        if arr.size == length:
            return arr
        if arr.size > length:
            return arr[:length]
        out = np.full(length, np.nan, dtype=arr.dtype if arr.size else np.float32)
        out[: arr.size] = arr
        return out

    io_start = time.time()
    df = df_bw.copy()
    df["Penetration_from_TD"] = _pad_series(penetration_td[0], F)
    df["Cone_Angle_Angular_Density"] = _pad_series(cone_angle_AD, F)
    df["Hydraulic_Delay"] = _pad_series(hydraulic_delay * np.ones(F), F)
    df["CA_AD_Upper"] = _pad_series(cone_angle_AD_up, F)
    df["CA_AD_Lower"] = _pad_series(cone_angle_AD_low, F)
    df = df[[
        "Frame",
        "Penetration_from_BW",
        "Penetration_from_BW_Polar",
        "Penetration_from_TD",
        "Cone_Angle_Angular_Density",
        "Cone_Angle_Average",
        "Cone_Angle_RANSAC",
        "Cone_Angle_Linear_Regression",
        "Area",
        "Estimated_Volume",
        "Estimated_Volume_Upper_limit",
        "Estimated_Volume_Lower_limit",
        "Hydraulic_Delay",
        "CA_AD_Upper",
        "CA_AD_Lower",
        "CA_Avg_Upper",
        "CA_Avg_Lower",
        "CA_Ransac_Upper",
        "CA_Ransac_Lower",
        "CA_LG_Upper",
        "CA_LG_Lower",
    ]]
    df.to_csv(data_dir / f"{file_name}_metrics.csv")

    # usage
    save_boundary_csv(boundary, data_dir / f"{file_name}_boundary_points.csv", origin=(0,H//2))
    print(f"Metrics + boundary CSV completed in: {time.time() - io_start:.2f} seconds")
    
    
    
    avi_saver.wait()
    avi_saver.shutdown()
    # playing to check 

        

