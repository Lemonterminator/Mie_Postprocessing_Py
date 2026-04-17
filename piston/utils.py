from __future__ import annotations

import numpy as np


def quarter_of_a_circle_complement_mask(
    canvas: np.ndarray,
    grid_size: float,
    circle_center: tuple[float, float],
    radius_mm: float,
    loc: str = "top_left",
    negative_mask: bool = True,
) -> np.ndarray:
    """Return the complement of a quadrant circle within its bounding square."""
    height_px, width_px = canvas.shape
    x_mm, y_mm = circle_center
    radius_px = max(1, int(round(radius_mm / grid_size)))

    cx = int(round(x_mm / grid_size))
    cy = int(round(y_mm / grid_size))

    loc = loc.replace("-", "_").lower()
    valid_locs = {"top_left", "top_right", "bottom_left", "bottom_right"}
    if loc not in valid_locs:
        raise ValueError(f"loc must be one of {sorted(valid_locs)}, got {loc!r}")

    yy, xx = np.ogrid[:height_px, :width_px]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2

    if loc == "top_left":
        square_mask = (xx >= cx - radius_px) & (xx <= cx) & (yy >= cy) & (yy <= cy + radius_px)
    elif loc == "top_right":
        square_mask = (xx >= cx) & (xx <= cx + radius_px) & (yy >= cy) & (yy <= cy + radius_px)
    elif loc == "bottom_left":
        square_mask = (xx >= cx - radius_px) & (xx <= cx) & (yy >= cy - radius_px) & (yy <= cy)
    else:
        square_mask = (xx >= cx) & (xx <= cx + radius_px) & (yy >= cy - radius_px) & (yy <= cy)

    complement_mask = square_mask & (~circle_mask)
    return ~complement_mask if negative_mask else complement_mask


def clip_polygon_mm_to_px(points_mm: np.ndarray, grid_size: float, shape: tuple[int, int]) -> np.ndarray:
    """Convert mm polygon vertices to clipped pixel coordinates."""
    points_px = np.rint(points_mm / grid_size).astype(np.int32)
    points_px[:, 0] = np.clip(points_px[:, 0], 0, shape[1] - 1)
    points_px[:, 1] = np.clip(points_px[:, 1], 0, shape[0] - 1)
    return points_px
