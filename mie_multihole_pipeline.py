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
    estimate_hydraulic_delay,
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


# Default nozzle radii (pixels)
global lower
global upper
global ir_
global or_
ir_ = 14
or_ = 380
lower = 0
upper = 366

def mie_multihole_pipeline(
    video,
    centre,
    number_of_plumes,
    *,
    gamma=1.0,
    binarize_video=False,
    plot_on=False,
    solver="Fast",
    file_name,
    rotated_vid_dir, 
    data_dir, 
    save_rotated_videos=False,
    FPS=20, 
    lighting_unchanged_duration=50, 
    TD_sum_interval=0.5
):
    """
    Main entry point for the multi-hole Mie processing pipeline.

    Returns rotated plume segments, penetration traces, cone angles, and
    optional binarized videos/boundaries when requested.
    """
    centre_x = float(centre[0])
    centre_y = float(centre[1])
    hydraulic_delay_estimate = 15

    use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")
    if use_gpu:
        from OSCC_postprocessing.rotation.rotate_with_alignment import (
            rotate_video_nozzle_at_0_half_cupy as rotate_video_nozzle_at_0_half_backend,
        )
    else:
        rotate_video_nozzle_at_0_half_backend = rotate_video_nozzle_at_0_half_numpy

    '''
    foreground, px_range_mask = preprocess_multihole(
        video,
        hydraulic_delay_estimate,
        gamma=gamma,
        M=3,
        N=3,
        range_mask=True,
        timing=True,
        use_gpu=use_gpu,
        triangle_backend=triangle_backend,
        return_numpy=not use_gpu,
    )
    '''
    
    video = xp.asarray(video)
    foreground = pre_processing_mie(video, division=False)
    px_range_mask = foreground[0]==0.0

    bins = 3600
    start_time = time.time()
    _, signal, _ = angle_signal_density_auto(foreground, centre_x, centre_y, N_bins=bins)
    offset = estimate_offset_from_fft(signal, number_of_plumes)
    if offset:
        print(f"Estimated offset from FFT: {offset:.3f} degrees")

    angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - offset



    foreground = xp.asarray(foreground, dtype=xp.float32)

    '''
    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)
    
    plume_mask = generate_plume_mask(ir_, or_, crop[2], crop[3])

    # Old rotation with range masks
    px_range_mask = xp.asarray(px_range_mask)
    segments, range_masks = rotate_segments_with_masks(
        foreground,
        px_range_mask,
        angles,
        crop,
        centre,
        region_mask=plume_mask,
        xp=xp,
    )
    '''
    # Rotation 
    F, H, W = video.shape
    segments = []
    INTERPOLATION = "nearest"
    BORDER_MODE = "constant"
    OUT_SHAPE = (H // 4, W//2)

    for idx, angle in enumerate(angles):
        segment, _, _ = rotate_video_nozzle_at_0_half_backend(
                video,
                centre, # (nozzle_x, nozzle_y) # change to centre_x + cos(angle) * r, centre_y + sin(angle) * r
                angle,
                interpolation=INTERPOLATION,
                border_mode=BORDER_MODE,
                out_shape=OUT_SHAPE,
            )
        segments.append(segment)
    # TODO: Check bugs 
    segments = xp.stack(segments, axis=0)  # (P, F, H, W)
    
    plume_mask = generate_plume_mask(ir_, or_, segments[2], segments[3])

    range_masks = plume_mask[None, :, :]

    elapsed = time.time() - start_time
    print(f"Computing all rotated segments finished in {elapsed:.2f} seconds.")

    if solver == "Fast":

        start_time = time.time()
        td_intensity_maps, _, energies = compute_td_intensity_maps(segments, range_masks, use_gpu)
        energy_total = float(np.sum(energies.get()) if use_gpu else np.sum(energies))
        if energy_total < 10:
            return None, None, None, None, None, None

        _, avg_peak, peak_frames_host = estimate_peak_brightness_frames(energies, use_gpu)
        hydraulic_delay = estimate_hydraulic_delay(segments, avg_peak, use_gpu)
        print(f"Vectorized TD-Intensity Heatmaps completed in {time.time() - start_time:.2f}s")


        start_time = time.time()
        penetration = compute_penetration_profiles(
            td_intensity_maps,
            energies,
            hydraulic_delay,
            peak_frames_host,
            use_gpu=use_gpu,
            lower=lower,
            upper=upper,
        )
        penetration = clean_penetration_profiles(penetration, hydraulic_delay, upper, lower)
        print(f"Post processing completed in {time.time() - start_time:.2f}s")

        if plot_on:
            rows = (segments.shape[0] + 2) // 3 + 1
            fig, ax = plt.subplots(rows, 3, figsize=(12, 3 * rows))
            P = segments.shape[0]
            for p in range(P):
                arr = td_intensity_maps[p, :, :].T
                arr_np = arr.get() if use_gpu else arr
                ax[p // 3, p % 3].imshow(arr_np, origin="lower", aspect="auto")
                ax[p // 3, p % 3].plot(penetration[p], color="red")
            if P:
                last = P - 1
                ax[last // 3, (last % 3)].plot(penetration.T)

        bw_vids = None
        boundaries = None
        penetration_old = None

        if binarize_video:
            start_time = time.time()
            bw_vids, penetration_old = binarize_plume_videos(segments, hydraulic_delay, penetration)
            pen_old_np = (
                penetration_old.get() if use_gpu and hasattr(penetration_old, "get") else penetration_old
            )
            pen_old_np = np.asarray(pen_old_np, dtype=np.float32)
            penetration_old = clean_penetration_profiles(pen_old_np, hydraulic_delay, upper, lower)

            if plot_on:
                plt.figure()
                with np.errstate(invalid="ignore", all="ignore"):
                    plt.plot(np.nanmedian(penetration, axis=0), label="Column sum segmentation")
                    plt.plot(np.nanmedian(penetration_old, axis=0), label="Per-frame segmentation")
                plt.xlabel("Frame number")
                plt.ylabel("Penetration [px]")
                plt.title("Penetration Comparison (median)")
                plt.legend()

                plt.figure()
                with np.errstate(invalid="ignore", all="ignore"):
                    plt.plot(np.nanmean(penetration, axis=0), label="Column sum segmentation")
                    plt.plot(np.nanmean(penetration_old, axis=0), label="Per-frame segmentation")
                plt.xlabel("Frame number")
                plt.ylabel("Penetration [px]")
                plt.title("Penetration Comparison (mean)")
                plt.legend()

                plt.figure()
                plt.title("Area of all segments")
                arr = bw_vids.get() if use_gpu else bw_vids
                plt.plot(np.sum(np.sum(arr, axis=3), axis=2).T)

            bw_vids_np = xp.asnumpy(bw_vids) if use_gpu else bw_vids
            boundaries = bw_boundaries_all_points(bw_vids_np)
            print(f"Binarizing video and calculating boundary completed in {time.time() - start_time:.2f}s")

        cone_angle_AngularDensity = compute_cone_angle_from_angular_density(
            signal,
            offset,
            number_of_plumes,
            bins=bins,
            use_gpu=use_gpu,
        )

        if plot_on:
            plt.figure()
            plt.plot(cone_angle_AngularDensity.T)
            plt.show()
            plt.close("all")

        segments_np = xp.asnumpy(segments) if use_gpu else segments
        if use_gpu:
            penetration = np.asarray(penetration)
            if bw_vids is not None:
                bw_vids = bw_vids_np

        return segments_np, penetration, cone_angle_AngularDensity, bw_vids, boundaries, penetration_old

    elif solver == "Slow":
        for idx, foreground in enumerate(segments):
            if save_rotated_videos:
                avi_saver = AsyncAVISaver(max_workers=4)
                # Save the Foreground video asynchronously
                AsyncNPZSaver().save(data_dir / f"{file_name}_plume_number_{idx}.npz", foreground=to_numpy(foreground))
                
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
                plot_path = output_dir / f"{Path(file_name).stem}_plume_{idx}_cone_angle_comparison.png"
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
                plot_path2 = output_dir / f"{Path(file_name).stem}_plume_{idx}_penetration_comparison.png"
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
        df.to_csv(data_dir / f"{file_name}_plume_{idx}_metrics.csv")

        # usage
        save_boundary_csv(boundary, data_dir / f"{file_name}_plume_{idx}_boundary_points.csv", origin=(0,H//2))
        print(f"Metrics + boundary CSV completed in: {time.time() - io_start:.2f} seconds")
        