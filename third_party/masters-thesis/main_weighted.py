"""Weighted spray-segmentation pipeline with local library integration.

The processing defaults are kept close to the original masters-thesis script.
Local integration is limited to repo function paths plus config-driven strip
alignment.
"""

from GUI_functions import *
from clustering import *
from OSCC_postprocessing.cine.functions_videos import load_cine_video
from data_capture import *

import opticalFlow as of
import videoProcessingFunctions as vpf
import matplotlib.pyplot as plt
from Legacy.std_functions3 import *
import tkinter as tk
from tkinter import filedialog
import json
import os
import cv2
import numpy as np

from OSCC_postprocessing.rotation.rotate_crop import generate_CropRect

try:
    import cupy as cp
    from OSCC_postprocessing.rotation.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy,
    )
except Exception:
    cp = None
    rotate_video_nozzle_at_0_half_cupy = None


#
# Pipeline tuning parameters
# Edit these values here to adjust the workflow globally.
#

# Preprocessing and visualization
FRAME_LIMIT = 200
FIRST_FRAME_THRESHOLD = 10
BACKGROUND_MASK_THRESHOLD = 20
RUN_TAGS_DEBUG_VIEW = True
VIDEO_FPS = 30

# Optical-flow / intensity fusion
FLOW_SOLVER = "farneback"
USE_INTENSITY_ONLY = False
USE_CUMULATIVE_AS_MASK = True
FLOW_MAG_PERCENTILE_LOW = 5.0
FLOW_MAG_PERCENTILE_HIGH = 99.0
FLOW_MAG_SEED_THRESHOLD = 0.95
FLOW_MAG_START_THRESHOLD = 0.6
WEIGHT_INTENSITY = 0.4 
WEIGHT_MAGNITUDE = 0.8
WEIGHT_FREEHAND = 0.1
WEIGHT_CONE = 0.6
WEIGHT_INTENSITY_CUMULATIVE = 2.0
WEIGHT_CONE_AFTER_START = 1.0
INTENSITY_GAMMA = 3.0

# Geometric priors
CONE_ANGLE_DEG = 20
FALLOFF_ANGLE_DEG = 30
MIN_CONE_LENGTH_PX = 100
PENETRATION_LOOKAHEAD_PX = 50
ROI_RADIUS_PX = 100

# Mask post-processing
LARGEST_BLOB_HORIZONTAL_THRESHOLD = 50
CLUSTER_DISTANCE = 40
CLUSTER_ALPHA = 30


def load_processing_config(video_path):
    """Load the GUI-generated `config.json` sitting next to the selected video."""
    config_path = os.path.join(os.path.dirname(video_path), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json next to video: {video_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_fields = [
        "plumes",
        "offset",
        "centre_x",
        "centre_y",
        "inner_radius",
        "outer_radius",
    ]
    missing = [field for field in required_fields if field not in cfg]
    if missing:
        raise KeyError(f"config.json missing required fields {missing}: {config_path}")
    return cfg, config_path


def build_aligned_strip(video, cfg):
    """Create the nozzle-aligned strip using local config geometry and CuPy rotation when possible."""
    if rotate_video_nozzle_at_0_half_cupy is None or cp is None:
        raise RuntimeError(
            "CuPy rotation backend is unavailable. "
            "Install/configure CuPy before running this pipeline."
        )

    n_plumes = int(cfg["plumes"])
    cx = float(cfg["centre_x"])
    cy = float(cfg["centre_y"])
    offset_deg = float(cfg["offset"]) % 360.0
    inner = int(float(cfg["inner_radius"]))
    outer = int(float(cfg["outer_radius"]))

    crop = generate_CropRect(inner, outer, n_plumes, cx, cy)
    frame_h, frame_w = int(video.shape[1]), int(video.shape[2])
    crop_h = int(min(max(1, crop[3]), frame_h) / 2)
    crop_w = int(min(max(1, outer), frame_w))
    out_shape = (crop_h, crop_w)
    calibration_point = (0.0, float(cy))

    if n_plumes > 1:
        print(
            f"Config specifies {n_plumes} plumes; using the first plume direction from offset={offset_deg:.3f} deg."
        )

    video_gpu = cp.asarray(video)
    aligned_gpu, _, _ = rotate_video_nozzle_at_0_half_cupy(
        video_gpu,
        (cx, cy),
        offset_deg,
        interpolation="nearest",
        border_mode="constant",
        out_shape=out_shape,
        calibration_point=calibration_point,
        cval=0.0,
        stack=True,
    )
    aligned = cp.asnumpy(aligned_gpu).astype(np.uint8, copy=False)
    spray_origin = (0, aligned.shape[1] // 2)
    return aligned, spray_origin


def window_closed(window_name):
    """Return True once an OpenCV window has been closed by the user."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# TODO:
    # optimize performance for larger videos (CUDA?)
    # improve GUI for mask drawing and parameter tuning
    # Maybe standardize naming for videos to use the best settings automatically (TBD)
    # Handle edge cases better, e.g. no motion detected, very short videos, etc?
    # Handle multi file processing better, maybe with a progress bar or batch processing mode?
        # nozzle selection process
        # Combining data
    # Add cone angle display to overlay video
    # Add more metrics to output CSV (spray area, tip velocity, nozzle opening/closing time, etc)


# Hide the main tkinter window
root = tk.Tk()
root.withdraw()


################################
# Load Video Files
###############################
all_files = filedialog.askopenfilenames(title="Select one or more files")
for file in all_files:
    print("Processing:", file)
    video = load_cine_video(file, frame_limit=FRAME_LIMIT)
    config, config_path = load_processing_config(file)
    print(f"Loaded config: {config_path}")

    nframes, height, width = video.shape[:3]
    dtype = video[0].dtype

    for i in range(nframes):
        frame = video[i]
        if np.issubdtype(dtype, np.integer):
            frame_uint8 = (frame / 16).astype(np.uint8)
        elif np.issubdtype(dtype, np.floating):
            frame_uint8 = np.clip((frame * 255), 0, 255).astype(np.uint8)
        elif np.issubdtype(dtype, np.bool_):
            frame_uint8 = np.clip(frame.astype(np.uint8) * 255, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = (frame / 16).astype(np.uint8)

        video[i] = frame_uint8
    video = video.astype(np.uint8)

    ##############################
    # Video Rotation and Stripping
    ##############################
    video_strip, spray_origin = build_aligned_strip(video, config)
    nframes, height, width = video_strip.shape[:3]
    firstFrameNumber = vpf.findFirstFrame(video_strip, threshold=FIRST_FRAME_THRESHOLD)
    first_frame = video_strip[firstFrameNumber]

    # TESTING
    TAGS_segmentation = np.empty_like(video_strip, dtype=np.uint8)
    TAGS_segmentation_diff = np.empty_like(video_strip, dtype=np.float32)

    # Dynamic TAGS background update:
    # 1) Start from the last frame before injection.
    # 2) Segment current frame against current background.
    # 3) Update background pixels only where current frame is classified as background.
    bg_init_idx = max(0, firstFrameNumber - 1)
    tags_background = video_strip[bg_init_idx].copy()

    background_mask_test = vpf.createBackgroundMask(first_frame, threshold=BACKGROUND_MASK_THRESHOLD)

    if RUN_TAGS_DEBUG_VIEW:
        tags_window_names = [
            "TAGS Segmentation",
            "TAGS Segmentation Diff",
            "Current Frame",
        ]
        for i in range(nframes):
            current_frame = video_strip[i].copy()
            current_frame[background_mask_test == 0] = 0

            tags_mask, tags_diff = vpf.tags_segmentation(current_frame, tags_background)

            TAGS_segmentation[i] = tags_mask
            TAGS_segmentation_diff[i] = tags_diff

            background_pixels = tags_mask == 0
            tags_background[background_pixels] = current_frame[background_pixels]

            cv2.imshow("TAGS Segmentation", TAGS_segmentation[i])
            tags_diff_vis = cv2.normalize(tags_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow("TAGS Segmentation Diff", tags_diff_vis)
            cv2.imshow("Current Frame", current_frame)

            if any(window_closed(name) for name in tags_window_names):
                break

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                cv2.waitKey(-1)
        cv2.destroyAllWindows()

    ##############################
    # Freehand Mask Creation
    ##############################

    draw_freehand_mask(video_strip)
    # User can draw multiple separate regions if needed, just make sure to connect them with a line
    # so they are included in the same cluster.
    # Press and hold left mouse button to draw, release to stop drawing, press 'q' to finish and save mask.

    ##############################
    # Filter Visualization
    ###############################

    video_strip = vpf.applyCLAHE(video_strip)

    background_mask = vpf.createBackgroundMask(
        first_frame,
        threshold=BACKGROUND_MASK_THRESHOLD,
    )
    background_mask = edit_mask_overlay(
        first_frame,
        background_mask,
        window_name="Background Mask",
    )

    ##############################
    # Optical Flow Visualization
    ##############################
    use_intensity_only = USE_INTENSITY_ONLY
    use_cumulative_as_mask = USE_CUMULATIVE_AS_MASK

    if use_intensity_only:
        print("Using intensity-only mode (no optical flow contribution).")
        mag_array = np.ones_like(video_strip, dtype=np.float32)
    else:
        print("Using combined intensity and optical flow mode.")
        mag_array = of.runOpticalFlowCalculationWeighted(
            firstFrameNumber,
            video_strip,
            method=FLOW_SOLVER,
            normalize=True,
            lower_percentile=FLOW_MAG_PERCENTILE_LOW,
            upper_percentile=FLOW_MAG_PERCENTILE_HIGH,
        )

    w_intensity = WEIGHT_INTENSITY
    w_magnitude = WEIGHT_MAGNITUDE
    w_freehand = WEIGHT_FREEHAND
    w_cone = WEIGHT_CONE
    intensity_gamma = INTENSITY_GAMMA

    if use_intensity_only:
        w_magnitude = 0
    if use_cumulative_as_mask:
        w_magnitude = 0
        w_intensity = WEIGHT_INTENSITY_CUMULATIVE

    origin_x, origin_y = spray_origin
    cone_angle_deg = CONE_ANGLE_DEG
    falloff_angle = FALLOFF_ANGLE_DEG
    min_cone_length = MIN_CONE_LENGTH_PX
    max_cone_length = max(0, width - origin_x - 1)
    yy_full, xx_full = np.ogrid[:height, :width]
    dx_full = xx_full - origin_x
    dy_full = yy_full - origin_y
    angle_full = np.degrees(np.arctan2(dy_full, dx_full))
    abs_angle_full = np.abs(angle_full)

    full_cone_mask = np.zeros((height, width), dtype=np.float32)
    in_forward = dx_full > 0
    in_main = in_forward & (abs_angle_full <= cone_angle_deg)
    in_falloff = in_forward & (abs_angle_full > cone_angle_deg) & (
        abs_angle_full <= cone_angle_deg + falloff_angle
    )
    full_cone_mask[in_main] = 1.0
    full_cone_mask[in_falloff] = 1.0 - (
        abs_angle_full[in_falloff] - cone_angle_deg
    ) / falloff_angle

    combined_masks = np.zeros_like(video_strip, dtype=np.uint8)
    final_cluster_masks = np.zeros_like(video_strip, dtype=np.uint8)
    intensity_scores = np.zeros_like(video_strip, dtype=np.float32)
    mag_scores = np.zeros_like(video_strip, dtype=np.float32)
    cumulative_masks = np.zeros_like(video_strip, dtype=np.uint8)
    cone_masks = np.zeros_like(video_strip, dtype=np.uint8)

    boundaries = []
    penetration = np.zeros(nframes, dtype=np.float32)
    cone_angle = np.zeros(nframes, dtype=np.float32)
    cone_angle_reg = np.zeros(nframes, dtype=np.float32)
    close_point_distance = np.zeros(nframes, dtype=np.float32)
    angle_d = 0.0
    spray_area = np.zeros(nframes, dtype=np.float32)

    freehand_mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    if freehand_mask is None:
        print("Warning: 'mask.png' not found - proceeding without freehand mask")
        freehand_mask_f = np.zeros((height, width), dtype=np.float32)
    else:
        if freehand_mask.shape != (height, width):
            freehand_mask = cv2.resize(
                freehand_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        freehand_mask_f = (freehand_mask > 0).astype(np.float32)

    total_w = w_intensity + w_magnitude + w_freehand + w_cone
    norm_intensity = w_intensity / total_w
    norm_magnitude = w_magnitude / total_w
    norm_freehand = w_freehand / total_w
    norm_cone = w_cone / total_w

    eps = 1e-6
    cumulative_mask = np.zeros((height, width), dtype=np.uint8)

    write_masks_started = False
    yy, xx = np.ogrid[:height, :width]
    circle_mask = (xx - origin_x) ** 2 + (yy - origin_y) ** 2 <= ROI_RADIUS_PX ** 2

    for idx in range(nframes):
        frame = video_strip[idx]
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame.copy()
        intensity = frame_gray.astype(np.float32)

        if use_cumulative_as_mask:
            masked_pixels = intensity[cumulative_mask > 0]
            if len(masked_pixels) > 0:
                p_low, p_high = np.percentile(masked_pixels, (1.0, 99.0))
            else:
                p_low, p_high = np.percentile(intensity, (1.0, 99.0))
        else:
            p_low, p_high = np.percentile(intensity, (1.0, 99.0))

        if p_high - p_low > 1e-6:
            intensity_n = (intensity - p_low) / (p_high - p_low)
            intensity_n = np.clip(intensity_n, 0.0, 1.0)
            intensity_n = 1.0 - intensity_n
            intensity_n = np.clip(intensity_n ** intensity_gamma, 0.0, 1.0)
        else:
            intensity_n = 1.0 - (np.clip(intensity, 0.0, 255.0) / 255.0)
            intensity_n = np.clip(intensity_n ** intensity_gamma, 0.0, 1.0)

        if use_cumulative_as_mask:
            intensity_n[cumulative_mask == 0] = 0

        intensity_scores[idx] = intensity_n

        mag_n = np.clip(mag_array[idx].astype(np.float32), 0.0, 1.0)
        mag_scores[idx] = mag_n

        new_areas = (mag_n >= FLOW_MAG_SEED_THRESHOLD).astype(np.uint8) * 255
        cumulative_mask = np.maximum(cumulative_mask, new_areas)
        cumulative_masks[idx] = cumulative_mask.copy()

        if idx >= firstFrameNumber:
            motion_near_origin = np.any(mag_n[circle_mask] >= FLOW_MAG_START_THRESHOLD)
            if motion_near_origin:
                write_masks_started = True
                w_cone = WEIGHT_CONE_AFTER_START
        else:
            motion_near_origin = False

        if idx > 0:
            cone_length = max(
                penetration[idx - 1] + PENETRATION_LOOKAHEAD_PX,
                min_cone_length,
            )
        else:
            cone_length = min_cone_length + PENETRATION_LOOKAHEAD_PX
        cone_length = min(cone_length, max_cone_length)

        cone_mask_f = full_cone_mask.copy()
        if cone_length < max_cone_length:
            cutoff_x = int(origin_x + cone_length)
            if cutoff_x + 1 < width:
                cone_mask_f[:, cutoff_x + 1:] = 0.0

        cone_masks[idx] = (cone_mask_f * 255).astype(np.uint8)

        freehand = freehand_mask_f
        cone = cone_mask_f

        if np.count_nonzero(freehand) == 0:
            freehand = np.ones_like(freehand, dtype=np.float32)

        comp_int = (intensity_n + eps) ** norm_intensity
        comp_motion = (mag_n + eps) ** norm_magnitude
        comp_free = (freehand + eps) ** norm_freehand
        comp_cone = (cone + eps) ** norm_cone

        combined_score = comp_int * comp_motion * comp_free * comp_cone
        combined_score = (
            combined_score / np.max(combined_score)
            if np.max(combined_score) > 0
            else combined_score
        )

        if use_intensity_only or use_cumulative_as_mask:
            combined_uint8 = (combined_score * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(
                combined_uint8,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            dynamic_threshold = otsu_thresh / 255.0
            threshold_mask = (combined_score >= dynamic_threshold).astype(np.uint8) * 255
        else:
            peak = combined_score.max()
            threshold_mask = (combined_score >= 0.8 * peak).astype(np.uint8) * 255

        combined_score = cv2.GaussianBlur(combined_score, (5, 5), 0)
        threshold_mask[background_mask == 0] = 0

        nozzle_point_rc = np.array([spray_origin[1], spray_origin[0]], dtype=np.float32)

        if use_intensity_only or use_cumulative_as_mask:
            final_mask = fill_holes_in_mask(threshold_mask)
            final_mask = keep_largest_blob(
                final_mask,
                horizontal_threshold=LARGEST_BLOB_HORIZONTAL_THRESHOLD,
                spray_origin=spray_origin,
            )
            final_cluster_masks[idx] = final_mask
        else:
            final_mask = create_cluster_mask(
                threshold_mask,
                cluster_distance=CLUSTER_DISTANCE,
                alpha=CLUSTER_ALPHA,
            )
            final_cluster_masks[idx] = final_mask

        if write_masks_started:
            combined_masks[idx] = threshold_mask
            frame_boundaries, frame_pen, _, frame_ang, frame_ang_reg, frame_cpd = analyze_boundary(
                final_mask,
                angle_d=angle_d,
                nozzle_point=nozzle_point_rc,
            )
        else:
            combined_masks[idx] = np.zeros_like(threshold_mask)
            frame_boundaries = []
            frame_pen = 0.0
            frame_ang = 0.0
            frame_ang_reg = 0.0
            frame_cpd = 0.0

        boundaries.append(frame_boundaries)
        penetration[idx] = frame_pen
        cone_angle[idx] = frame_ang
        cone_angle_reg[idx] = frame_ang_reg
        close_point_distance[idx] = frame_cpd
        tip_velocity = np.gradient(penetration)
        spray_area[idx] = np.sum(final_mask > 0)
        nozzle_opening_time = 0
        nozzle_closing_time = 0

    print(
        f"Final masks computed with w_intensity={w_intensity}, "
        f"w_magnitude={w_magnitude}, w_freehand={w_freehand}, w_cone={w_cone}, "
        f"intensity_gamma={intensity_gamma}, use_cumulative_as_mask={use_cumulative_as_mask}, "
        "dynamic thresholding (Otsu if cumulative mask or intensity-only, else 80th percentile)"
    )

    results_dir = os.path.join(os.getcwd(), "Results")
    os.makedirs(results_dir, exist_ok=True)
    output_base = os.path.basename(file).replace(".cine", "")
    output_csv = os.path.join(results_dir, f"{output_base}_spray_metrics.csv")
    output_video = os.path.join(results_dir, f"{output_base}_overlay.mp4")

    intensity_values = []
    video_writer = None
    video_fps = VIDEO_FPS
    for i in range(nframes):
        frame = video_strip[i]
        combined = combined_masks[i]
        cluster = final_cluster_masks[i]
        cone = cone_masks[i]

        overlay = overlay_cluster_outline(frame, cluster)
        if overlay.ndim == 2 or (overlay.ndim == 3 and overlay.shape[2] == 1):
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        origin_x, origin_y = spray_origin
        tip_x = int(np.clip(origin_x + penetration[i], 0, width - 1))
        tip_y = int(np.clip(origin_y, 0, height - 1))
        cv2.line(overlay, (int(origin_x), int(origin_y)), (tip_x, tip_y), (0, 255, 255), 5)
        tip_half_len = 40
        y1 = int(np.clip(tip_y - tip_half_len, 0, height - 1))
        y2 = int(np.clip(tip_y + tip_half_len, 0, height - 1))
        cv2.line(overlay, (tip_x, y1), (tip_x, y2), (0, 255, 255), 3)
        cv2.circle(overlay, (int(origin_x), int(origin_y)), 4, (0, 0, 255), -1)

        mean_intensity = cv2.mean(frame, combined)
        intensity_values.append(mean_intensity)

        def resize(img, size=(640, 320)):
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        def ensure_bgr(img):
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        frame_disp = ensure_bgr(resize(frame))
        combined_disp = ensure_bgr(resize(combined))
        cluster_disp = ensure_bgr(resize(cluster))
        intensity_disp = ensure_bgr(resize((intensity_scores[i] * 255).astype(np.uint8)))
        mag_disp = ensure_bgr(resize((mag_scores[i] * 255).astype(np.uint8)))
        cumulative_disp = ensure_bgr(resize(cumulative_masks[i]))
        cone_disp = ensure_bgr(resize(cone))
        freehand_disp = ensure_bgr(resize((freehand_mask_f * 255).astype(np.uint8)))
        overlay_disp = ensure_bgr(resize(overlay))

        text_color = (0, 255, 255)
        cv2.putText(frame_disp, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(combined_disp, "Combined Weighted Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(cluster_disp, "Clustered Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(intensity_disp, "Intensity Score", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(mag_disp, "Optical Flow Magnitude", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(cumulative_disp, "Cumulative Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(cone_disp, "Cone Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(freehand_disp, "Freehand Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(overlay_disp, "Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

        cv2.putText(frame_disp, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(combined_disp, "Combined Weighted Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(cluster_disp, "Clustered Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(intensity_disp, "Intensity Score", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(mag_disp, "Optical Flow Magnitude", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(cumulative_disp, "Cumulative Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(cone_disp, "Cone Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(freehand_disp, "Freehand Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(overlay_disp, "Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        row1 = np.hstack([frame_disp, combined_disp, cluster_disp])
        row2 = np.hstack([intensity_disp, mag_disp, cumulative_disp])
        row3 = np.hstack([cone_disp, freehand_disp, overlay_disp])
        grid = np.vstack([row1, row2, row3])

        cv2.imshow("All Results", grid)

        if video_writer is None:
            h, w = overlay.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            video_writer = cv2.VideoWriter(output_video, fourcc, video_fps, (w, h))
        video_writer.write(overlay)

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            cv2.waitKey(-1)

    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()

    """
    # --- Analyze intensity values ---
    # Needs more work, maybe diffent method to find significant changes

    # calculate derivative of intensity values
    intensity_values = np.array(intensity_values)[:, 0]

    # Apply rolling mean with window size 5
    window_size = 5
    intensity_smoothed = np.convolve(intensity_values, np.ones(window_size) / window_size, mode='valid')

    # Compute derivative on smoothed data
    intensity_derivative = np.diff(intensity_smoothed, prepend=intensity_smoothed[0])

    # --- Create shifted x-axis ---
    frame_numbers = np.arange(firstFrameNumber, firstFrameNumber + len(intensity_derivative))

    start_offset = 10
    if len(intensity_derivative) <= start_offset:
        min_idx = int(np.argmin(intensity_derivative))
    else:
        sliced = intensity_derivative[start_offset:]
        rel_min = int(np.argmin(sliced))
        min_idx = rel_min + start_offset

    min_frame = int(frame_numbers[min_idx])
    min_value = float(intensity_derivative[min_idx])
    print(f"Lowest intensity derivative at frame {min_frame} (index {min_idx}) = {min_value:.6f}")

    plt.plot(frame_numbers, intensity_derivative)
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Intensity Inside Region")
    plt.title("Intensity Over Time (Shifted)")
    plt.show()
    """

    frames = np.arange(nframes)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

    axes[0, 0].plot(frames, penetration)
    axes[0, 0].set_title("Penetration")
    axes[0, 0].set_ylabel("Pixels")

    axes[0, 1].plot(frames, cone_angle)
    axes[0, 1].set_title("Cone Angle")
    axes[0, 1].set_ylabel("Degrees")

    axes[0, 2].plot(frames, cone_angle_reg)
    axes[0, 2].set_title("Regularized Cone Angle")
    axes[0, 2].set_ylabel("Degrees")

    axes[1, 0].plot(frames, close_point_distance)
    axes[1, 0].set_title("Close Point Distance")
    axes[1, 0].set_ylabel("Pixels")
    axes[1, 0].set_xlabel("Frame Number")

    axes[1, 1].plot(frames, spray_area)
    axes[1, 1].set_title("Spray Area")
    axes[1, 1].set_ylabel("Pixels$^2$")
    axes[1, 1].set_xlabel("Frame Number")

    axes[1, 2].axis("off")

    fig.suptitle("Spray Metrics Over Time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.show()

    with open(output_csv, "w") as f:
        f.write(
            "Frame,Penetration (pixels), Cone Angle (degrees), Regularized Cone Angle (degrees), "
            "Close Point Distance (pixels), Spray Area (pixels^2)\n"
        )
        for i in range(nframes):
            f.write(
                f"{i},{penetration[i]},{cone_angle[i]},{cone_angle_reg[i]},"
                f"{close_point_distance[i]},{spray_area[i]}\n"
            )


print("Processing complete.")
