from time import time
import numpy as np
import cv2
from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *
from mie_postprocessing.video_filters import *
from mie_postprocessing.functions_bw import *
from mie_postprocessing.video_playback import *
from mie_postprocessing.functions_rotation import *
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from scipy.ndimage import binary_erosion, generate_binary_structure, binary_fill_holes
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

HYDRAULIC_DELAY = 20

def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary mask."""
    u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_idx)


def _contour_points(mask: np.ndarray) -> np.ndarray:
    """Extract boundary coordinates (y, x) from a binary mask."""
    u8 = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.vstack(contours).reshape(-1, 2)  # (x, y)
    return np.column_stack((pts[:, 1], pts[:, 0])).astype(np.float32)  # (y, x)


def _median_blur_bool(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply median filter to a boolean mask and return bool output."""
    u8 = mask.astype(np.uint8) * 255
    blurred = cv2.medianBlur(u8, ksize)
    return blurred > 0


def _morph_close(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    u8 = mask.astype(np.uint8) * 255
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def schlieren_singlehole_pipeline(video, chamber_mask, centre, offset):

    global_dark_threshold = 0.08

    video_masked = mask_video(video, np.flipud(~chamber_mask))
    ir_ = 0
    or_ = video.shape[1] * np.sqrt(2) - 400

    centre_x = float(centre[0])
    centre_y = float(centre[1])

    F, H, W = video.shape
    
    crop = generate_CropRect(ir_, or_, 6, centre_x, centre_y)
    segments = rotate_all_segments_auto(video_masked, [offset], crop, centre)
    segment = np.stack(segments).squeeze()
    play_video_cv2(np.swapaxes(segment, 1, 2), intv=34)

    video_cropped = video[:, video.shape[2] - int(centre_y): H - 100, int(centre_x): W - 100]

    crop = generate_CropRect(ir_, or_, 4, centre_x, centre_y)
    segments = rotate_all_segments_auto(video_masked, [-offset], crop, centre)
    np.stack(segments).squeeze()

    rotated = rotate_video_auto(video_cropped, angle=offset, max_workers=8)

    rotated = rotated[:, rotated.shape[2] // 5: 4 * rotated.shape[2] // 5]
    # play_video_cv2(rotated)

    # Copying Simon's pipeline
    bkg_frame = 4
    bkg_intensity_scaling = 0.5

    bkg = bkg_intensity_scaling * rotated[bkg_frame]

    sub_bkg = rotated - bkg[None, :, :]

    mask = bkg < 1e-4

    sub_bkg_uint8 = (sub_bkg * 255).astype(np.uint8)
    bw_vid = np.zeros_like(sub_bkg_uint8, dtype=np.bool_)

    for frame in range(F):
        bw_vid[frame] = cv2.threshold(sub_bkg_uint8[frame], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    masked_bw = mask_video(bw_vid, ~mask)
    F, H, W = masked_bw.shape

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    nozzle_y = rotated.shape[1] / 2.0
    nozzle_x = 0.0

    flag = False
    CPD = 0.0

    largest_masks = np.zeros_like(masked_bw, dtype=np.bool_)
    boundary_per_frame = []
    areas = np.zeros(F, dtype=np.int32)
    penetration = np.zeros(F, dtype=np.float32)
    close_point_distance = np.zeros(F, dtype=np.float32)

    for frame in range(HYDRAULIC_DELAY, F):
        diff_med = _median_blur_bool(masked_bw[frame])
        BW_open = _morph_close(diff_med, se)

        darkest_parts = rotated[frame] < global_dark_threshold 
        dark = np.logical_and(darkest_parts, mask)
        dark = _median_blur_bool(dark)
        dark_open = _morph_close(dark, se)

        if (frame > 0 and CPD > 10) or flag:
            bw = np.logical_or(BW_open, dark_open)
            flag = True
        else:
            bw = dark_open

        filled = binary_fill_holes(bw)
        largest = _largest_connected_component(filled)

        if not largest.any():
            boundary_per_frame.append(np.empty((0, 2), dtype=np.float32))
            areas[frame] = 0
            penetration[frame] = 0.0
            close_point_distance[frame] = 0.0
            CPD = 0.0
            continue

        areas[frame] = int(largest.sum())
        largest_masks[frame] = largest

        boundary_pts = _contour_points(largest)
        boundary_per_frame.append(boundary_pts)

        if boundary_pts.size:
            dy = boundary_pts[:, 0] - nozzle_y
            dx = boundary_pts[:, 1] - nozzle_x
            distances = np.hypot(dx, dy)
            penetration[frame] = float(distances.max())
            close_point_distance[frame] = float(distances.min())
            CPD = close_point_distance[frame]
        else:
            penetration[frame] = 0.0
            close_point_distance[frame] = 0.0
            CPD = 0.0
    
    
    foreground = largest_masks*rotated
    play_video_cv2(foreground, intv=34)
    TD_intensity_map = foreground.sum(axis=1)
    
    plt.imshow(np.log(TD_intensity_map+1e-9).T, origin="lower")

    return {
        "masks": largest_masks,
        "boundaries": boundary_per_frame,
        "areas": areas,
        "penetration": penetration,
        "close_point_distance": close_point_distance,
    }
