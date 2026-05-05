"""Mie-imaging preprocessing and postprocessing for multi-hole spray analysis.

Provides the core image-processing building blocks used by the Mie top-view
multihole pipeline:

- ``refined_log_subtraction``: log-space background subtraction with intensity gating
- ``arr_3d_sobel_magnitude_cupy``: Sobel edge-energy computation on 3-D arrays
- ``mie_multihole_preprocessing``: full preprocessing stage (log subtraction + Sobel + masking)
- ``mie_multihole_postprocessing``: angular segmentation into per-plume rotated videos
- ``robust_scale_arr_4d``: percentile-based normalization for 4-D plume arrays
- ``penetration_cdf_all_plumes``: CDF-front penetration estimate for every plume
- ``triangle_binarize``: canonical triangle thresholding entry point
- ``triangle_binarize_gpu``: compatibility alias that maps to the same implementation
"""

from __future__ import annotations

import numpy as np

from OSCC_postprocessing.utils.backend import USING_CUPY, cp, to_numpy, xp
from OSCC_postprocessing.utils.scaling import min_max_scale as _min_max_scale, robust_scale
from OSCC_postprocessing.analysis.thresholding import triangle_binarize
from OSCC_postprocessing.binary_ops.functions_bw import keep_largest_component_cuda
from OSCC_postprocessing.binary_ops._backend import cndi
from OSCC_postprocessing.binary_ops.masking import (
    generate_angular_mask_from_tf,
    periodic_true_segment_lengths,
)
from OSCC_postprocessing.analysis.cone_angle import (
    angle_signal_density_auto,
    estimate_offset_from_fft,
)
from OSCC_postprocessing.analysis.hysteresis import fill_short_false_runs
from OSCC_postprocessing.analysis.penetration_cdf import penetration_cdf_front
from OSCC_postprocessing.filters.convolution_2d import convolution_2D_cupy, make_kernel
from OSCC_postprocessing.rotation.segment_ops import generate_plume_mask

try:
    from cupyx.scipy.ndimage import median_filter as _median_filter
except Exception:
    from scipy.ndimage import median_filter as _median_filter  # type: ignore[assignment]

if USING_CUPY:
    from OSCC_postprocessing.rotation.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy as _rotate_video_nozzle_at_0_half,
    )
else:
    from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as _rotate_video_nozzle_at_0_half,
    )

_as_numpy = to_numpy


triangle_binarize_gpu = triangle_binarize


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def refined_log_subtraction(video, frames_before_SOI, q_min=5, q_max=99.99, noise_floor_multiplier=3.0, threshold=0.05):
    """Log-space background subtraction with intensity gating and soft thresholding.

    Parameters
    ----------
    video:
        Input video array ``(F, H, W)``, linear intensity values.
    frames_before_SOI:
        Number of pre-injection frames used to estimate the static background.
    q_min, q_max:
        Percentile bounds for the final robust scaling step.
    noise_floor_multiplier:
        Pixels below ``background * noise_floor_multiplier`` are gated to zero.

    Returns
    -------
    Scaled ``dF/F`` array with the same shape as ``video``.
    """
    eps = 1e-9

    lg_video = xp.log(video + eps)
    lg_bkg = xp.median(lg_video[:frames_before_SOI], axis=0, keepdims=True)

    log_ratio = lg_video - lg_bkg
    dff = xp.expm1(log_ratio)

    raw_bkg = xp.exp(lg_bkg)
    noise_floor_mask = video > (raw_bkg * noise_floor_multiplier)
    dff *= noise_floor_mask

    baseline = xp.median(dff[:frames_before_SOI], axis=0, keepdims=True)
    dff -= baseline    
    dff_clean = xp.maximum(dff - threshold, 0)

    return robust_scale(dff_clean, q_min=q_min, q_max=q_max)


def arr_3d_sobel_magnitude_cupy(arr_3d, wsize=3, sigma=1.0):
    """Compute frame-wise Sobel gradient magnitude for a 3-D array ``(F, H, W)``.

    Parameters
    ----------
    arr_3d:
        Input array with shape ``(F, H, W)``.
    wsize:
        Kernel window size (pixels).
    sigma:
        Gaussian smoothing sigma applied inside the Sobel kernel.

    Returns
    -------
    Gradient magnitude array with the same shape as ``arr_3d``.
    """
    sobel_x = make_kernel("sobel", wsize, sigma, direction="x")
    sobel_y = make_kernel("sobel", wsize, sigma, direction="y")
    sb_filt_x = convolution_2D_cupy(arr_3d, sobel_x)
    sb_filt_y = convolution_2D_cupy(arr_3d, sobel_y)
    return xp.sqrt(sb_filt_x ** 2 + sb_filt_y ** 2)


def mie_multihole_preprocessing(
    video,
    ring_mask,
    wsize=3,
    sigma=1.0,
    chamber_mask=None,
    frames_before_SOI=10,
    noise_floor_multiplier=3,
    threshold=0.05,
    q_min_foreground=5,
    q_max_foreground=99.99,
    q_min_highpass=5,
    q_max_highpass=99.9999

    ):
    """Full Mie preprocessing stage: log subtraction, Sobel edge energy, masking.

    Parameters
    ----------
    video:
        Raw camera video ``(F, H, W)``.
    ring_mask:
        2-D annular mask ``(H, W)`` that confines the analysis to the active spray region.
    wsize:
        Sobel kernel window size.
    sigma:
        Gaussian sigma for the Sobel kernel.
    chamber_mask:
        Optional additional 2-D mask for chamber boundaries.
    frames_before_SOI:
        Pre-injection frame count used for background estimation.
    noise_floor_multiplier:
        Intensity gating threshold relative to background level.

    Returns
    -------
    lg_foreground:
        Background-subtracted, scaled foreground video.
    lg_foreground_highpass:
        Sobel-filtered edge-energy video (same shape).
    """
    lg_foreground = refined_log_subtraction(
        video, frames_before_SOI, noise_floor_multiplier=noise_floor_multiplier, q_min=q_min_foreground, q_max=q_max_foreground, threshold=threshold
    )

    sb_mag = arr_3d_sobel_magnitude_cupy(lg_foreground, wsize=wsize, sigma=sigma)
    lg_foreground_highpass = robust_scale(sb_mag, q_min=q_min_highpass, q_max=q_max_highpass)

    lg_foreground *= ring_mask[None, :, :]
    lg_foreground_highpass *= ring_mask[None, :, :]

    if chamber_mask is not None:
        chamber_mask = xp.asarray(chamber_mask)
        lg_foreground_highpass *= chamber_mask[None, :, :]

    return lg_foreground, lg_foreground_highpass


def mie_multihole_postprocessing(
    foreground,
    highpass,
    centre,
    number_of_plumes,
    inner_radius,
    outer_radius,
    bins=720,
    INTERPOLATION="nearest",
    BORDER_MODE="constant",
    segment_bw_q_min=5,
    segment_bw_q_max=99.5,
):
    """Segment the full-frame video into per-plume rotated strips.

    Uses FFT-based rotation-offset estimation to align each injector hole with
    the x-axis, then extracts a strip video for each plume.

    Parameters
    ----------
    foreground:
        Background-subtracted video ``(F, H, W)``.
    highpass:
        Sobel-filtered edge-energy video ``(F, H, W)``.
    centre:
        Nozzle centre ``(x, y)`` in pixel coordinates.
    number_of_plumes:
        Number of injector holes.
    inner_radius:
        Inner nozzle radius in pixels (used for plume mask).
    outer_radius:
        Outer radius determining strip crop size in pixels.
    bins:
        Number of angular bins for the signal-density analysis.
    segment_bw_q_min, segment_bw_q_max:
        Percentile bounds used to normalize each plume's 3-D segment volume
        before applying the plume spatial mask.

    Returns
    -------
    dict with keys:
        ``segments_fg`` (P, F, H, W), ``cone_angle_proxy_deg``,
        ``occupied_angle_total_deg``, ``occupied_angle_segment_count``,
        ``occupied_angle_segment_widths_deg``, ``occupied_angle_mask``,
        ``fft_offset_deg``, ``plume_angles_deg``.
    """
    num_frames, frame_height, frame_width = foreground.shape

    # 1. 提取每個角度隨時間變化的信號強度 
    # (Extract signal intensity per angle over time)
    # angular_intensity_series shape: (num_frames, bins)
    _, angular_intensity_series, _ = angle_signal_density_auto(
        foreground, centre[0], centre[1], N_bins=bins
    )

    # 2. 計算噴霧的整體旋轉偏移量和每個噴孔的中心角度 
    # (Calculate global rotation offset and center angles for each plume)
    global_rotation_offset = estimate_offset_from_fft(angular_intensity_series, number_of_plumes)
    plume_center_angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - _as_numpy(global_rotation_offset)

    # 3. 計算時間平均信號並二值化，以識別包含噴霧的活躍角度區域 
    # (Time-average the signal and binarize to find active angle regions containing plumes)
    time_averaged_intensity = xp.sum(angular_intensity_series, axis=0)
    is_angle_active_mask = fill_short_false_runs(
        triangle_binarize_gpu(time_averaged_intensity, ignore_zero=True),
        max_len=3,
    )

    # 4. 基於活躍角度遮罩計算噴霧的幾何特徵 (Calculate geometric properties of plumes based on the active angle mask)
    # 每個活躍片段的長度（以 bin 為單位）
    plume_widths_bins = _as_numpy(
        periodic_true_segment_lengths(is_angle_active_mask)
    ).astype(np.int64, copy=False)
    
    # 將 bin 寬度轉換為角度值
    plume_widths_deg = (
        plume_widths_bins.astype(np.float64) * 360.0 / float(bins)
    )
    total_active_angle_deg = float(_as_numpy(is_angle_active_mask.sum()) * 360.0 / float(bins))
    detected_plume_count = int(plume_widths_deg.size)
    
    # 計算平均噴霧錐角作為參考 
    # (Calculate average cone angle as a proxy)
    average_plume_angle_deg = (
        float(np.mean(plume_widths_deg))
        if detected_plume_count > 0
        else float("nan")
    )

    # 5. 提取的備用空間和角度遮罩變量（目前未被使用，作為參考保留） 
    # (Retain spatial and angular mask variables, for reference)
    # spatial_angular_mask = generate_angular_mask_from_tf(frame_height, frame_width, centre, is_angle_active_mask, bins)  # noqa: F841
    
    # normalized_time_averaged_intensity = _min_max_scale(time_averaged_intensity)
    # smoothed_active_mask = triangle_binarize_gpu(_median_filter(normalized_time_averaged_intensity, 5))
    # reference_average_plume_angle = smoothed_active_mask.sum() / bins * 360.0 / number_of_plumes  # noqa: F841 (kept for reference)

    # 6. 為每個噴霧提取旋轉後的視頻片段 
    # (Extract rotated video strips for each plume)

    # 設定提取區域尺寸 
    # (Height = outer_radius // 2, Width = outer_radius)
    OUT_SHAPE = (int(outer_radius) // 2, int(outer_radius))

    plume_strips_list = []
    for angle in plume_center_angles:
        rotated_strip, _, _ = _rotate_video_nozzle_at_0_half(
            highpass,
            centre,
            angle,
            interpolation=INTERPOLATION,
            border_mode=BORDER_MODE,
            out_shape=OUT_SHAPE,
        )
        rotated_strip = _min_max_scale(rotated_strip)
        plume_strips_list.append(rotated_strip)

    # 將所有噴霧片段堆疊成單個數組 
    # (Stack all plume strips into a single 4D tensor)
    stacked_plume_strips = xp.stack(plume_strips_list, axis=0)  # Shape: (P, F, H, W)
    # Normalize each plume's 3-D volume before applying the spatial plume mask.
    stacked_plume_strips = robust_scale_arr_4d(stacked_plume_strips, q_min=segment_bw_q_min, q_max=segment_bw_q_max)
    _, _, strip_height, strip_width = stacked_plume_strips.shape

    # 7. 應用內部半徑遮罩去除噴嘴本體的干擾 
    # (Apply inner radius mask to remove interference from other plumes and the injector)
    plume_mask = generate_plume_mask(strip_width, strip_height, 360.0 / number_of_plumes, int(inner_radius))
    stacked_plume_strips *= xp.asarray(plume_mask[None, None, :, :])

    return {
        "segments_fg": stacked_plume_strips,
        "cone_angle_proxy_deg": average_plume_angle_deg,
        "occupied_angle_total_deg": total_active_angle_deg,
        "occupied_angle_segment_count": detected_plume_count,
        "occupied_angle_segment_widths_deg": plume_widths_deg,
        "occupied_angle_mask": _as_numpy(is_angle_active_mask).astype(bool, copy=False),
        "fft_offset_deg": float(_as_numpy(global_rotation_offset)),
        "plume_angles_deg": np.asarray(plume_center_angles, dtype=float),
    }


def robust_scale_arr_4d(arr_4d, q_min=5, q_max=99):
    """Apply percentile-based robust scaling to each plume's 3-D sub-array.

    Parameters
    ----------
    arr_4d:
        4-D array with shape ``(P, F, H, W)``.  Modified in place.
    q_min, q_max:
        Percentile clipping bounds passed to :func:`robust_scale`.

    Returns
    -------
    The scaled ``arr_4d`` (same object, modified in place).
    """
    P, F, H, W = arr_4d.shape
    for p, arr_3d in enumerate(arr_4d):
        arr_4d[p] = robust_scale(arr_3d, q_min=q_min, q_max=q_max)
    return arr_4d


def penetration_cdf_all_plumes(
    arr_4d,
    inner_radius,
    quantile=1.0 - 3e-2,
    frames_before_SOI=10,
    umbrella_angle=180.0,
):
    """Estimate CDF-front penetration for every plume in a 4-D segment array.

    Parameters
    ----------
    arr_4d:
        Per-plume video segments with shape ``(P, F, H, W)``.
    inner_radius:
        Inner nozzle radius in pixels; the penetration origin is offset by this.
    quantile:
        CDF quantile level used to define the spray front (e.g. 0.97).
    frames_before_SOI:
        Pre-injection frames used for per-frame baseline removal.
    umbrella_angle:
        Full umbrella angle in degrees; corrects projected x-distance for
        non-180° spray geometries.

    Returns
    -------
    ``(P, F)`` NumPy array of penetration distances in pixels from the nozzle exit.
    """
    heatmaps = xp.sum(arr_4d, axis=2)  # (P, F, W)

    use_gpu = hasattr(heatmaps, "__cuda_array_interface__")
    xhat_all = (cp.empty if use_gpu else np.empty)(
        (heatmaps.shape[0], heatmaps.shape[1]),
        dtype=np.int32,
    )
    opening_structure = cp.ones((7, 3)) if use_gpu else np.ones((7, 3), dtype=bool)

    for idx in range(len(heatmaps)):
        I = heatmaps[idx]
        I -= xp.median(I[:, :frames_before_SOI], axis=1, keepdims=True)
        I = xp.clip(I, 0.0, None)

        mask = triangle_binarize_gpu(_min_max_scale(I))
        mask = 1 - (keep_largest_component_cuda(1 - mask))
        mask = cndi.binary_opening(mask, opening_structure)

        xhat = penetration_cdf_front(I, mask=mask, q=quantile, min_x=10)
        xhat_all[idx] = xhat

    if umbrella_angle == 180.0:
        x_scale = 1.0
    else:
        tilt_angle = (180.0 - float(umbrella_angle)) / 2.0
        x_scale = 1.0 / np.cos(np.deg2rad(tilt_angle))

    penetration_cdf_all = cp.asnumpy(xhat_all) if use_gpu else np.asarray(xhat_all)
    penetration_cdf_all = np.maximum.accumulate(penetration_cdf_all, axis=1)
    return x_scale * np.maximum(0, penetration_cdf_all - inner_radius)


__all__ = [
    "arr_3d_sobel_magnitude_cupy",
    "mie_multihole_postprocessing",
    "mie_multihole_preprocessing",
    "penetration_cdf_all_plumes",
    "refined_log_subtraction",
    "robust_scale_arr_4d",
    "triangle_binarize",
    "triangle_binarize_gpu",
]
