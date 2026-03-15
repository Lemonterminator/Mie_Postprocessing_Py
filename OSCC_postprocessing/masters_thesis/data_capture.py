"""Boundary metrics extracted from the final binary spray mask."""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np


def analyze_boundary(
    mask: np.ndarray,
    *,
    angle_d: float,
    nozzle_point: np.ndarray,
    threshold_num_px_per_col: int = 10,
) -> Tuple[np.ndarray, float, float, float, float, float]:
    """Analyze the spray boundary to extract cone geometry and penetration metrics."""

    if not mask.any():
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, 0.0, 0.0, 0.0, 0.0, 0.0

    mask_bool = mask.astype(bool)
    mask_y_sum = np.count_nonzero(mask_bool, axis=0)
    col_has_spray = mask_y_sum > threshold_num_px_per_col
    if np.any(col_has_spray):
        last_col = int(np.flatnonzero(col_has_spray)[-1])
        penetration_x = float(max(0, last_col - float(nozzle_point[1])))
    else:
        penetration_x = 0.0

    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, 0.0, 0.0, 0.0, 0.0, 0.0

    all_coords = np.vstack([contour[:, 0, :] for contour in contours])
    coords_rc = np.column_stack((all_coords[:, 1], all_coords[:, 0])).astype(np.float32)

    relative = coords_rc - nozzle_point[None, :]
    relative_math = np.column_stack((relative[:, 1], -(relative[:, 0])))

    distances = np.linalg.norm(relative_math, axis=1)
    if distances.size == 0:
        return coords_rc, 0.0, penetration_x, 0.0, 0.0, 0.0

    penetration = float(distances.max())
    positive_distances = distances[distances > 0]
    close_point_distance = float(positive_distances.min()) if positive_distances.size else 0.0

    theta = math.radians(angle_d)
    rot = np.array(
        [
            [math.cos(-theta), -math.sin(-theta)],
            [math.sin(-theta), math.cos(-theta)],
        ],
        dtype=np.float32,
    )
    aligned = relative_math @ rot.T
    x_aligned = aligned[:, 0]
    y_aligned = aligned[:, 1]

    forward_mask = x_aligned > 0
    if not np.any(forward_mask):
        forward_mask = np.ones_like(x_aligned, dtype=bool)

    x_forward = x_aligned[forward_mask]
    y_forward = y_aligned[forward_mask]

    angles = np.degrees(np.arctan2(y_forward, x_forward))
    cone_angle = float(angles.max() - angles.min()) if angles.size else 0.0
    cone_angle_reg = regression_cone_angle(x_forward, y_forward)

    return (
        coords_rc,
        penetration,
        penetration_x,
        cone_angle,
        cone_angle_reg,
        close_point_distance,
    )


def regression_cone_angle(x_forward: np.ndarray, y_forward: np.ndarray) -> float:
    """Calculate spray cone angle using linear regression on top and bottom spray edges."""

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
