global timing
timing = True

if timing:
    import time 
import numpy as np
from OSCC_postprocessing.video_filters import gaussian_video_cpu, median_filter_video_auto
from OSCC_postprocessing.async_npz_saver import AsyncNPZSaver
from OSCC_postprocessing.async_avi_saver import *
from OSCC_postprocessing.svd_background_removal import godec_like
from OSCC_postprocessing.video_filters import *
from OSCC_postprocessing.video_playback import *
from OSCC_postprocessing.cone_angle import *
from OSCC_postprocessing.functions_bw import mask_video 
from mie_multihole_pipeline import _triangle_binarize_gpu
from OSCC_postprocessing.bilateral_filter import *
from OSCC_postprocessing.single_plume import *


from OSCC_postprocessing.multihole_utils import (
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

# Prefer GPU if CuPy is available; otherwise use a NumPy-compatible shim.

try:
    import cupy as _cupy  # type: ignore

    _cupy.cuda.runtime.getDeviceCount()
    cp = _cupy
    USING_CUPY = True
except Exception as exc:  # pragma: no cover - hardware dependent
    print(f"CuPy unavailable, falling back to NumPy backend: {exc}")
    USING_CUPY = False

    class _NumpyCompat:
        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, a, dtype=None):
            return np.asarray(a, dtype=dtype)

        def asnumpy(self, a):
            return np.asarray(a)

        def get(self, a):
            return a

    cp = _NumpyCompat()  # type: ignore

def to_numpy(arr):
    return cp.asnumpy(arr) if USING_CUPY else np.asarray(arr)

# Import rotation utility based on backend availability to avoid hard Cupy dependency
if USING_CUPY:
    from OSCC_postprocessing.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy as rotate_video_nozzle_at_0_half_backend,
    )
else:
    from OSCC_postprocessing.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as rotate_video_nozzle_at_0_half_backend,
    )

def _rotate_align_video_cpu(
    video: np.ndarray,
    nozzle_center: tuple[float, float],
    offset_deg: float,
    *,
    interpolation: str,
    out_shape: tuple[int, int] | None,
    border_mode: str,
    cval: float,
) -> np.ndarray:
    """
    Delegate to the NumPy implementation in OSCC_postprocessing.rotate_with_alignment_cpu.
    Returns only the rotated video (np.ndarray).
    """
    rotated_np, _, _ = rotate_video_nozzle_at_0_half_backend(
        video,
        nozzle_center,
        offset_deg,
        interpolation=interpolation,
        border_mode=border_mode,
        out_shape=out_shape,
        cval=cval,
    )
    return rotated_np.astype(np.float32, copy=False)

def _min_max_scale(arr: cp.ndarray) -> cp.ndarray:
    mn = arr.min()
    mx = arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return cp.zeros_like(arr)

def _prepare_temporal_smoothing(rotated: cp.ndarray, smooth_frames: int) -> tuple[cp.ndarray, np.ndarray]:
    rotated_cpu = to_numpy(rotated)
    smoothed_np = median_filter_video_auto(np.swapaxes(rotated_cpu, 0, 2), smooth_frames, 1)
    smoothed_np = np.swapaxes(smoothed_np, 0, 2)

    min_val = smoothed_np.min()
    max_val = smoothed_np.max()
    if max_val > min_val:
        smoothed_np = (smoothed_np - min_val) / (max_val - min_val)
    else:
        smoothed_np = np.zeros_like(smoothed_np, dtype=np.float32)

    smoothed_cp = cp.asarray(smoothed_np, dtype=cp.float32)
    return smoothed_cp, smoothed_np


def filter_schlieren(video, shock_wave_duration):
    smooth_frames = 3
    temporal_smoothing, temporal_smoothing_np = _prepare_temporal_smoothing(video, smooth_frames)
    
    inverted = 1.0 - temporal_smoothing_np

    foreground_godec = godec_like(inverted, 3)
    godec_pos = np.maximum(foreground_godec, 0.0)
    godec_pos = _min_max_scale(cp.asarray(godec_pos, dtype=cp.float32))

    if USING_CUPY:
        godec_pos = godec_pos.get()
    # --- usage ---
    # video: (T, H, W), grayscale
    sobel_x, sobel_y = sobel_5x5_kernels()

    # gradient in x for every frame

    gx_video = filter_video_fft(godec_pos[:shock_wave_duration], sobel_x, mode='same')

    # gradient in y for every frame
    gy_video = filter_video_fft(godec_pos[:shock_wave_duration], sobel_y, mode='same')

    # gradient magnitude (still (T, H, W))
    grad_mag = np.sqrt(gx_video**2 + gy_video**2)

    gamma = 1.1
    grad_mag_norm_inv = 1 - _min_max_scale(grad_mag) ** gamma
    
    thres = 0.9
    grad_mag_norm_inv[grad_mag_norm_inv < thres] = 0
    grad_mag_norm_inv[grad_mag_norm_inv > thres] = 1

    shock_wave_filtered = np.zeros_like(godec_pos)
    shock_wave_filtered[:shock_wave_duration] = _min_max_scale(
        grad_mag_norm_inv * godec_pos[:shock_wave_duration]
        )
    shock_wave_filtered[shock_wave_duration:] = godec_pos[shock_wave_duration:]

    '''
    play_videos_side_by_side((
        cp.swapaxes(godec_pos, 1,2),
        _min_max_scale(cp.swapaxes(shock_wave_filtered, 1, 2))
    ), intv=100)
    '''
    
    # play_video_cv2(godec_pos, intv=6000//FPS)
    '''
    play_videos_side_by_side((
        cp.swapaxes(godec_pos, 1, 2),
        _min_max_scale(cp.swapaxes(np.abs(gx_video), 1, 2)),
        # _min_max_scale(cp.swapaxes(gy_video, 1, 2)),
        # _min_max_scale(cp.swapaxes(grad_mag, 1, 2)),
        _min_max_scale(cp.swapaxes((1-np.sqrt(gx_video**2)), 1, 2)),
        _min_max_scale(cp.swapaxes((1-np.sqrt(gx_video**2))*godec_pos*inverted, 1, 2)),
        cp.swapaxes(inverted, 1, 2)
    ), intv= 600//FPS
    )
    '''
    return shock_wave_filtered


def filter_mie(video, angle=45):
    F, H, W = video.shape

    number_of_plumes = 1
    ########################################################################################################
    # Cone Angle (vectorized + optional GPU acceleration)
    bins = 3600
    shift_bins = int((0.5 * bins))
    # Compute angular signal density on GPU if available, without pulling video to CPU
    signal_density_bins, signal, density = angle_signal_density_auto(
        video, 0, W//2, N_bins=bins
    )
    # import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import binary_closing as cp_binary_closing  # type: ignore
    # Triangle threshold on GPU (ignores zeros) then shift and close
    sig_cp = cp.asarray(signal, dtype=cp.float32)
    bw_cp = _triangle_binarize_gpu(sig_cp)  # boolean mask on device
    bw_cp = cp.roll(bw_cp, -shift_bins, axis=1)
    struct_cp = cp.ones((1, 3), dtype=cp.bool_)
    bw_closed = cp_binary_closing(bw_cp, structure=struct_cp)

    # Compute per-plume widths on GPU, transfer only the result
    cone_angle_AngularDensity = np.zeros((number_of_plumes, bw_closed.shape[0]), dtype=np.float32)
    deg_per_bin = (360.0 / bins)
    for p in range(number_of_plumes):
        start = int(round(p * bins / number_of_plumes))
        end = int(round((p + 1) * bins / number_of_plumes))
        s = bw_closed[:, start:end].sum(axis=1) * deg_per_bin
        cone_angle_AngularDensity[p] = cp.asnumpy(s)
    
    range_masks = np.ones((H, W), dtype=np.float32)
    range_masks = range_masks[None, :, :]

    ########################################################################################################
    # Penetration (vectorized + optional GPU acceleration)
    first_frame =  video[0][None, :,:]
    first_frame[first_frame<1e-3] = 1

    corrected = video/first_frame
    

    corrected[corrected == cp.nan] = 0.0
    masked, mask, _ = mask_angle(corrected, angle)
    
    mask = mask | (corrected[0] < 1e-3)
    width = cp.sum(mask, axis=0)

    masked[masked== cp.nan] = 0.0 
    # heatmap = cp.sum(masked, axis=1)/width[None,:]
    # 1

    

    segments = masked[None, :, :, :]
    use_gpu = True
    # td_intensity_maps, _, energies = compute_td_intensity_maps(segments, range_masks, True)
    td_intensity_maps = cp.sum(masked, axis=1)/width[None,:].squeeze()
    energies = cp.sum(td_intensity_maps, axis=1)

    # Originally made for multiple plumes stacked
    td_intensity_maps= td_intensity_maps[None, :]
    energies = energies[None, : ]

    energy_total = float(np.sum(energies.get()) if use_gpu else np.sum(energies))
    if energy_total < 10:
        return None, None, None, None, None, None

    _, avg_peak, peak_frames_host = estimate_peak_brightness_frames(energies, use_gpu)
    hydraulic_delay = estimate_hydraulic_delay(segments, avg_peak, use_gpu)

    lower = 0
    upper = W
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

    bw_vids, penetration_old_host = binarize_plume_videos(corrected[None,:,:,:], hydraulic_delay)
    
    ## visualization

    joint_video = cp.zeros((F, H*2, W))
    joint_video[:, :, :H] = cp.swapaxes(corrected, 1, 2)
    joint_video[:, :, H:] = cp.swapaxes(corrected*bw_vids[0], 1, 2)


    avi_saver = AsyncAVISaver(max_workers=4)

    f1 = avi_saver.save(
        r"G:\MeOH_test\Mie\Processed_Results\Rotated_Videos\joint.avi ",
        to_numpy(joint_video),
        fps=10,
        is_color=False,
        auto_normalize=True,
    )

    return segments, penetration, cone_angle_AngularDensity
    
def pre_processing_mie(video):
    bilateral_filtered = bilateral_filter_video_volumetric_chunked_halo(video, 3, 3, 3)

    bkg = bilateral_filtered[0]
    bkg[bkg==0] = 1e-9
    bkg[bkg==cp.nan]=1e-9
    div_bkg = _min_max_scale(bilateral_filtered* ((1.0/bkg)[None, :, :]))
    sub_div_bkg = div_bkg - div_bkg[0][None, :,:]

    px_range_map = cp.max(sub_div_bkg, axis=0) - cp.min(sub_div_bkg, axis=0)
    mask, _ = triangle_binarize_from_float(px_range_map.get())
    mask = keep_largest_component(mask)
    mask = binary_fill_holes(mask)
    mask = cp.asarray(mask)

    '''
    bw_vid, penetration_bw = binarize_plume_video(sub_div_bkg, 15)
    
    play_videos_side_by_side((
        cp.swapaxes(video[:-1], 1,2).get(),
        cp.swapaxes(bilateral_filtered[1:], 1,2).get(),
        cp.swapaxes(div_bkg[1:], 1,2).get(), 
        cp.swapaxes(sub_div_bkg[1:], 1,2).get(),
        cp.swapaxes(bw_vid[1:], 1,2).get()
    ), intv=50)
    '''
    return (mask)[None, :, :] * sub_div_bkg



def mask_angle(video, angle_allowed_deg):
    F, H, W = video.shape
    
    # Half-angle in radians and its tangent
    half_angle_rad = cp.deg2rad(angle_allowed_deg / 2.0)
    tan_half = cp.tan(half_angle_rad)

    # Build coordinate grids: x along width, y along height
    # y: shape (H, 1), centered at H/2
    # x: shape (1, W), shifted so apex is at x=0 and we avoid /0
    y = cp.arange(H)[:, None] - H / 2.0      # (H,1)
    x = cp.arange(W)[None, :] + 1e-9         # (1,W), small epsilon to avoid /0

    # Triangle mask: right side only, angles within ±half_angle
    # (H,W) boolean mask
    mask_2d = (x >= 0) & (cp.abs(y / x) <= tan_half)

    # If mask_video expects a (H,W) mask per frame, it can usually broadcast it;
    # otherwise, expand to (F,H,W)
    # mask_3d = np.broadcast_to(mask_2d, video.shape)
    width = cp.sum(mask_2d, axis=0)

    return mask_video(video, mask_2d), mask_2d, width




#  Main pipeline function
def singlehole_pipeline(mode, video, offset, centre, file_name, 
                                  rotated_vid_dir, data_dir, 
                                  save_intermediate_results=True,
                                  FPS=20, lighting_unchanged_duration=50
                                  ):
    F, H, W = video.shape
    shock_wave_duration = 50
    # Rotate and align the video based on nozzle centre and offset
    INTERPOLATION = "nearest"
    BORDER_MODE = "constant"
    OUT_SHAPE = (H // 2, W)
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
            interpolation="bicubic",
            border_mode="constant",
            out_shape=OUT_SHAPE,
            cval=0.0,
        )
        rotated = cp.asarray(rotated_np, dtype=cp.float32)


    # Clip the negative number in the rotated video
    rotated = cp.clip(rotated, 0.0, None)

    # Normalize the rotated video to [0, 1]
    rotated = _min_max_scale(rotated)

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
        hydraulic_delay = 15
        
        # filter_mie(rotated)
        # segments, penetration, cone_angle_AngularDensity  = filter_mie(rotated)
        # filtered = segments[0]

        # Enhancing the foreground
        foreground = pre_processing_mie(rotated)
        F, H, W = foreground.shape  # dimensions after rotation/cropping

        # Estimating hydarulic delay
        _, peak_idx, _ = estimate_peak_brightness_frames(foreground, use_gpu=True)
        hydraulic_delay = estimate_hydraulic_delay(foreground[None, :, :, :], peak_idx, use_gpu=True)[0]

        # Cone angle
        # We set the origin to the nozzle, and treat upper and lower half as two plumes
        # Then calculate their cone angle by angular density respectively
        _, signal, _ = angle_signal_density_auto(foreground, 0.0, H//2, N_bins=3600)
        AngularDensity = compute_cone_angle_from_angular_density(signal, 0, 2, bins=3600, use_gpu=True)

        # Upper & Lower cone angle, sums up to the total cone angle
        cone_angle_AD_up = AngularDensity[0]
        cone_angle_AD_low = AngularDensity[1]
        cone_angle_AD = cone_angle_AD_up + cone_angle_AD_low

        # Shape
        frame_idx = np.arange(F, dtype=np.int32)

        # binarizin the video
        # Note: penetration_bw_sum is the column wise sum 
        bw_vid, penetration_bw_sum = binarize_plume_video(foreground, hydraulic_delay, lighting_unchanged_duration=lighting_unchanged_duration, hole_fill_mode="2D")
        
        # Since penetration_bw_sum is the width (in pixels) of the spary boundary
        # We assume that it is symmetric with regard of the central axis, and take the width as diameter
        # radius = diameter /2, and differential volume per image column is pi * radius^2 
        estimated_volume = 0.25*np.pi*np.sum(penetration_bw_sum**2, axis=1)

        # BW penetration
        penetration_old = penetration_bw_to_index(penetration_bw_sum > 0)

        # BW boundary
        boundary = bw_boundaries_all_points_single_plume(bw_vid, parallel=True)


        points_all_frames = bw_boundaries_xband_filter_single_plume(boundary, penetration_old)

        cone_angle_linear_regression = np.zeros(F)
        cone_angle_ransac = np.zeros(F)
        cone_angle_average = np.zeros(F)
        ad_up_np = to_numpy(cone_angle_AD_up)
        ad_low_np = to_numpy(cone_angle_AD_low)

        avg_up = np.zeros(F)
        avg_low = np.zeros(F)
        ransac_up = np.zeros(F)
        ransac_low = np.zeros(F)
        lg_up  = np.zeros(F)
        lg_low = np.zeros(F)


        for i in range(F):
            points = points_all_frames[i]
            if len(points[0]) > 0 and len(points[1]) > 0:
                uy, ux =  points[1][:,0], points[1][:,1]
                ly, lx = points[0][:,0], points[0][:,1]
                
                # Shift y axis to center of the image
                uy -= H//2
                ly -= H//2

                # 
                ang_up = np.atan(uy/ux)*180.0/np.pi
                ang_low = np.atan(ly/lx)*180.0/np.pi

                avg_up[i] = np.nanmean(ang_up)
                avg_low[i] = np.nanmean(ang_low)
                cone_angle_average[i] = avg_up[i] - avg_low[i]

                ransac_up[i] = np.atan(ransac_fixed_intercept(ux, uy, 0)[0])*180.0/np.pi
                ransac_low[i] = np.atan(ransac_fixed_intercept(lx, ly, 0)[0])*180.0/np.pi
                cone_angle_ransac[i] = ransac_up[i] - ransac_low[i]

                lg_up[i] = np.atan(linear_regression_fixed_intercept(ux, uy, 0.0))*180.0/np.pi
                lg_low[i] = np.atan(linear_regression_fixed_intercept(lx, ly, 0.0))*180.0/np.pi
                cone_angle_linear_regression[i] = lg_up[i] - lg_low[i]





        

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            from pathlib import Path
            from OSCC_postprocessing.async_plot_saver import AsyncPlotSaver
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"Skipping comparison plots (matplotlib unavailable): {exc}")
        else:
            plot_saver = AsyncPlotSaver(max_workers=2)

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            angle_series = {
                "Average (upper-lower)": cone_angle_average,
                "Linear regression": cone_angle_linear_regression,
                "RANSAC": cone_angle_ransac,
                "Angular density": ad_up_np + ad_low_np,
            }
            for label, values in angle_series.items():
                axes[0].plot(frame_idx, values, label=label, linewidth=1.4)
            axes[0].set_ylabel("Cone angle (deg)")
            axes[0].grid(True)
            axes[0].legend()

            axes[1].plot(frame_idx, avg_up, label="Upper mean angle", linewidth=1.2)
            axes[1].plot(frame_idx, avg_low, label="Lower mean angle", linewidth=1.2)
            axes[1].plot(frame_idx, ad_up_np, label="AD upper", linewidth=1.2)
            axes[1].plot(frame_idx, ad_low_np, label="AD lower", linewidth=1.2)
            axes[1].set_xlabel("Frame")
            axes[1].set_ylabel("Edge angle (deg)")
            axes[1].grid(True)
            axes[1].legend()

            output_dir = Path(data_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"{Path(file_name).stem}_cone_angle_comparison.png"
            plot_saver.submit(fig, plot_path)
            plot_saver.shutdown(wait=True)
    # play_video_cv2(filtered)


    
    

    avi_saver.wait()
    avi_saver.shutdown()
    # playing to check 

        
