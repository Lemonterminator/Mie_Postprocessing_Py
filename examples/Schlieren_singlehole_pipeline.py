from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes

from mie_postprocessing.functions_bw import keep_largest_component
from mie_postprocessing.functions_rotation import rotate_video_nozzle_centering
from mie_postprocessing.video_filters import (
    filter_video_fft,
    gaussian_video_cpu,
    median_filter_video_auto,
)

# Default pipeline parameters translated from the original MATLAB routine.
DEFAULT_GAMMA = 2.2
DEFAULT_DARK_THRESHOLD = 1.5
STRUCTURING_ELEMENT_RADIUS = 5
CROP_MARGIN = 8

# 3x3 Laplacian kernel equivalent to fspecial('laplacian', 0).
LAPLACIAN_KERNEL = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                            dtype=np.float32)


@dataclass(frozen=True)
class SchlierenResults:
    penetration: np.ndarray
    cone_angle: np.ndarray
    cone_angle_reg: np.ndarray
    area: np.ndarray
    boundary: List[np.ndarray]
    close_point_distance: np.ndarray
    masks: List[np.ndarray]
    video_writer: Optional[cv2.VideoWriter]


def schlieren_singlehole_pipeline(
    video: Sequence[np.ndarray] | np.ndarray,
    chamber_mask: Optional[np.ndarray],
    centre: Tuple[float, float],
    angle_d: float,
    *,
    plot_on: bool = False,
    video_writer: Optional[cv2.VideoWriter] = None,
) -> SchlierenResults:
    """Thin wrapper retained for backwards compatibility with main.py."""
    x1, y1 = centre
    results = doallframes_schlieren(
        video,
        angle_d=angle_d,
        x1=x1,
        y1=y1,
        plot_on=plot_on,
        mask=chamber_mask,
        video_writer=video_writer,
    )
    return SchlierenResults(*results)


def doallframes_schlieren(
    spice: Sequence[np.ndarray] | np.ndarray,
    *,
    angle_d: float,
    x1: float,
    y1: float,
    plot_on: bool = False,    mask: Optional[np.ndarray] = None,
    video_writer: Optional[cv2.VideoWriter] = None,
    gamma: float = DEFAULT_GAMMA,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    np.ndarray,
    List[np.ndarray],
    Optional[cv2.VideoWriter],
]:
    """Process schlieren video frames and extract flow features."""
    video = _ensure_video_array(spice)
    rotated, processed_mask, nozzle_xy = _align_and_crop_video(
        video,
        mask=mask,
        centre=(x1, y1),
        angle_offset=angle_d,
    )

    pen, ang, ang_reg, area, boundaries, close_dist, outline = _process_frames(
        rotated,
        mask=processed_mask,
        angle_d=angle_d,
        nozzle_xy=nozzle_xy,
        gamma=gamma,
        dark_threshold=dark_threshold,
        plot_on=plot_on,
        video_writer=video_writer,
    )

    return pen, ang, ang_reg, area, boundaries, close_dist, outline, video_writer


def _ensure_video_array(spice: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    array = np.asarray(spice)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("Input video must be 2-D or 3-D (frames, height, width).")

    array = array.astype(np.float32, copy=False)
    if array.size and float(array.max()) > 2.0:
        array = array / 4096.0
    array = np.clip(array, 0.0, 1.0, out=array)
    return array


def _align_and_crop_video(
    video: np.ndarray,
    *,
    mask: Optional[np.ndarray],
    centre: Tuple[float, float],
    angle_offset: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    centre_x, centre_y = centre
    rotation_angle = -float(angle_offset)

    rotated = rotate_video_nozzle_centering(video, centre_x, centre_y, rotation_angle)
    mask_rotated = _rotate_mask(mask, centre_x, centre_y, rotation_angle, video.shape[1:])

    y_start, y_end, x_start, x_end = _auto_crop_bounds(rotated, mask_rotated)

    cropped_video = rotated[:, y_start:y_end, x_start:x_end]
    if mask_rotated is None:
        cropped_mask = np.ones((y_end - y_start, x_end - x_start), dtype=bool)
    else:
        cropped_mask = mask_rotated[y_start:y_end, x_start:x_end]

    nozzle_y_full = (rotated.shape[1] - 1) / 2.0
    nozzle_y = float(nozzle_y_full - y_start)
    nozzle_x = float(-x_start)

    nozzle_x = max(0.0, min(cropped_video.shape[2] - 1.0, nozzle_x))
    nozzle_y = max(0.0, min(cropped_video.shape[1] - 1.0, nozzle_y))

    return cropped_video, cropped_mask.astype(bool, copy=False), (nozzle_x, nozzle_y)


def _rotate_mask(
    mask: Optional[np.ndarray],
    centre_x: float,
    centre_y: float,
    rotation_angle: float,
    original_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    if mask is None:
        return None

    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError("Mask must be 2-D.")

    if mask_arr.shape != original_shape:
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask_bool = mask_arr.astype(bool, copy=False)
    rotated = rotate_video_nozzle_centering(
        mask_bool[None, ...], centre_x, centre_y, rotation_angle, interpolation=cv2.INTER_NEAREST
    )[0]
    return rotated


def _auto_crop_bounds(
    video: np.ndarray,
    mask: Optional[np.ndarray],
    margin: int = CROP_MARGIN,
) -> Tuple[int, int, int, int]:
    _, height, width = video.shape
    if mask is not None and mask.any():
        binary = mask.astype(bool, copy=False)
    else:
        binary = np.any(video > 1e-5, axis=0)
    rows = np.where(np.any(binary, axis=1))[0]
    cols = np.where(np.any(binary, axis=0))[0]

    if rows.size == 0:
        rows = np.arange(height)
    if cols.size == 0:
        cols = np.arange(width)

    y_start = max(0, int(rows[0]) - margin)
    y_end = min(height, int(rows[-1]) + margin + 1)
    x_start = 0
    x_end = min(width, int(cols[-1]) + margin + 1)

    if y_end <= y_start:
        y_start, y_end = 0, height
    if x_end <= x_start:
        x_start, x_end = 0, width

    return y_start, y_end, x_start, x_end


def _process_frames(
    video: np.ndarray,
    *,
    mask: np.ndarray,
    angle_d: float,
    nozzle_xy: Tuple[float, float],
    gamma: float,
    dark_threshold: float,
    plot_on: bool,
    video_writer: Optional[cv2.VideoWriter],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    np.ndarray,
    List[np.ndarray],
]:
    frames = video.shape[0]
    struct_elem = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * STRUCTURING_ELEMENT_RADIUS + 1,) * 2
    )

    highpass = filter_video_fft(video, LAPLACIAN_KERNEL)
    smoothed = gaussian_video_cpu(highpass.astype(np.float32), ksize=(5, 5), sigma=0)
    abs_smoothed = np.abs(smoothed, out=smoothed)
    gamma_adjusted = _apply_gamma(abs_smoothed, gamma)
    diff_med = median_filter_video_auto(gamma_adjusted, 5, 5)
    otsu_masks = _binarize_otsu(diff_med) & mask[None, ...]

    penetration = np.zeros(frames, dtype=np.float32)
    cone_angle = np.zeros(frames, dtype=np.float32)
    cone_angle_reg = np.zeros(frames, dtype=np.float32)
    area = np.zeros(frames, dtype=np.float32)
    close_point_distance = np.zeros(frames, dtype=np.float32)
    boundaries: List[np.ndarray] = []
    outline_frames: List[np.ndarray] = []

    nozzle_x, nozzle_y = nozzle_xy
    nozzle_point = np.array([nozzle_x, nozzle_y], dtype=np.float32)

    for idx in range(frames):
        bw_open = _morph_close(otsu_masks[idx], struct_elem)
        darkest_parts = video[idx] > dark_threshold
        dark_mask = np.logical_and(~darkest_parts, mask)
        dark_open = _morph_close(dark_mask, struct_elem)
        bw = np.logical_or(bw_open, dark_open)

        filled = binary_fill_holes(bw)
        largest = keep_largest_component(filled, connectivity=2)

        outline_frames.append(largest)
        area[idx] = float(np.count_nonzero(largest))

        frame_boundaries, frame_pen, frame_ang, frame_ang_reg, frame_cpd = _analyze_boundary(
            largest,
            angle_d=angle_d,
            nozzle_point=nozzle_point,
        )
        boundaries.append(frame_boundaries)
        penetration[idx] = frame_pen
        cone_angle[idx] = frame_ang
        cone_angle_reg[idx] = frame_ang_reg
        close_point_distance[idx] = frame_cpd

        if plot_on:
            _display_frame(video[idx], frame_boundaries, nozzle_point)
        if video_writer is not None:
            _write_overlay(video[idx], frame_boundaries, nozzle_point, video_writer)

    return (
        penetration,
        cone_angle,
        cone_angle_reg,
        area,
        boundaries,
        close_point_distance,
        outline_frames,
    )


def _apply_gamma(video: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return video
    flat = video.reshape(video.shape[0], -1)
    min_vals = flat.min(axis=1, keepdims=True)
    max_vals = flat.max(axis=1, keepdims=True)
    denom = np.maximum(max_vals - min_vals, 1e-6)
    norm = (flat - min_vals) / denom
    norm **= gamma
    return norm.reshape(video.shape)


def _binarize_otsu(video: np.ndarray) -> np.ndarray:
    frames, height, width = video.shape
    out = np.zeros((frames, height, width), dtype=bool)
    for idx in range(frames):
        frame = video[idx]
        frame_u8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, thresh = cv2.threshold(frame_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out[idx] = thresh > 0
    return out


def _morph_close(mask: np.ndarray, struct_elem: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    mask_u8 = (mask.astype(np.uint8)) * 255
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, struct_elem)
    return closed > 0


def _analyze_boundary(
    mask: np.ndarray,
    *,
    angle_d: float,
    nozzle_point: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, float]:
    if not mask.any():
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, 0.0, 0.0, 0.0, 0.0

    contours, _ = cv2.findContours(
        (mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, 0.0, 0.0, 0.0, 0.0

    contour = max(contours, key=len)
    coords = contour[:, 0, :]
    coords_rc = np.column_stack((coords[:, 1], coords[:, 0])).astype(np.float32)

    relative = coords_rc - nozzle_point[None, :]
    relative_math = np.column_stack((relative[:, 0], -(relative[:, 1])))

    distances = np.linalg.norm(relative_math, axis=1)
    if distances.size == 0:
        return coords_rc, 0.0, 0.0, 0.0, 0.0

    penetration = float(distances.max())
    positive_distances = distances[distances > 0]
    close_point_distance = float(positive_distances.min()) if positive_distances.size else 0.0

    theta = math.radians(angle_d)
    rot = np.array([[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]],
                   dtype=np.float32)
    aligned = relative_math @ rot.T
    x_aligned = aligned[:, 0]
    y_aligned = aligned[:, 1]

    forward_mask = x_aligned > 0
    if not np.any(forward_mask):
        forward_mask = np.ones_like(x_aligned, dtype=bool)

    x_forward = x_aligned[forward_mask]
    y_forward = y_aligned[forward_mask]
    angles = np.degrees(np.arctan2(y_forward, x_forward))
    if angles.size:
        cone_angle = float(angles.max() - angles.min())
    else:
        cone_angle = 0.0

    cone_angle_reg = _regression_cone_angle(x_forward, y_forward)

    return coords_rc, penetration, cone_angle, cone_angle_reg, close_point_distance


def _regression_cone_angle(x_forward: np.ndarray, y_forward: np.ndarray) -> float:
    if x_forward.size < 2:
        return 0.0

    top_mask = y_forward >= 0
    bottom_mask = y_forward <= 0

    angles: List[float] = []
    if np.count_nonzero(top_mask) >= 2:
        m_top, _ = np.polyfit(x_forward[top_mask], y_forward[top_mask], 1)
        angles.append(abs(float(np.degrees(np.arctan(m_top)))))
    if np.count_nonzero(bottom_mask) >= 2:
        m_bottom, _ = np.polyfit(x_forward[bottom_mask], y_forward[bottom_mask], 1)
        angles.append(abs(float(np.degrees(np.arctan(m_bottom)))))

    if len(angles) == 2:
        return angles[0] + angles[1]
    if len(angles) == 1:
        return 2.0 * angles[0]
    return 0.0


def _display_frame(frame: np.ndarray, boundary: np.ndarray, nozzle_point: np.ndarray) -> None:
    plt.clf()
    plt.imshow(frame, cmap="gray", origin="upper")
    if boundary.size:
        plt.plot(boundary[:, 1], boundary[:, 0], "r", linewidth=2)
    plt.plot(nozzle_point[0], nozzle_point[1], "r*", markersize=10)
    plt.pause(0.001)


def _write_overlay(
    frame: np.ndarray,
    boundary: np.ndarray,
    nozzle_point: np.ndarray,
    video_writer: cv2.VideoWriter,
) -> None:
    frame_u8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    if boundary.size:
        contour_xy = boundary[:, ::-1].astype(np.int32)
        cv2.polylines(frame_bgr, [contour_xy], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.circle(
        frame_bgr,
        (int(round(nozzle_point[0])), int(round(nozzle_point[1]))),
        4,
        (0, 0, 255),
        thickness=-1,
    )
    video_writer.write(frame_bgr)
