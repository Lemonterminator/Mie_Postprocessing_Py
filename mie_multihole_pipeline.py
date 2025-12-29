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

    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)
    


    plume_mask = generate_plume_mask(ir_, or_, crop[2], crop[3])

    foreground = xp.asarray(foreground, dtype=xp.float32)
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
    F, H, W = video.shape
    segments = []
    INTERPOLATION = "nearest"
    BORDER_MODE = "constant"
    OUT_SHAPE = (H // 4, W//2)

    for idx, angle in enumerate(angles):
        segment, _, _ = rotate_video_nozzle_at_0_half_backend(
                video,
                centre,
                angle,
                interpolation=INTERPOLATION,
                border_mode=BORDER_MODE,
                out_shape=OUT_SHAPE,
            )
        segments.append(segment)

    elapsed = time.time() - start_time
    print(f"Computing all rotated segments finished in {elapsed:.2f} seconds.")

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

