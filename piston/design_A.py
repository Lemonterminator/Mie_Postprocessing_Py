from __future__ import annotations

from typing import Any

import cv2
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

    # Select the square that bounds the requested quadrant in grid coordinates.
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


def _validate_design_a_geometry(
    *,
    r_bore: float,
    r_outer_bowl: float,
    r_topland: float,
    r_ring: float,
    r_inner_bowl: float,
    r_lip: float,
    r_floor: float,
    h_vertical: float,
    piston_height: float,
) -> None:
    if not np.isclose(r_outer_bowl + r_topland + r_ring, r_bore):
        raise ValueError(
            f"Bore radius mismatch: r_outer_bowl({r_outer_bowl}) + "
            f"r_topland({r_topland}) + r_ring({r_ring}) = {r_outer_bowl + r_topland + r_ring}, "
            f"but r_bore = {r_bore}"
        )

    if not np.isclose(r_inner_bowl + r_lip + r_floor, r_outer_bowl):
        raise ValueError(
            f"Outer bowl radius mismatch: r_inner_bowl({r_inner_bowl}) + "
            f"r_lip({r_lip}) + r_floor({r_floor}) = {r_inner_bowl + r_lip + r_floor}, "
            f"but r_outer_bowl = {r_outer_bowl}"
        )

    if not np.isclose(h_vertical + r_lip + r_floor, piston_height):
        raise ValueError(
            f"Piston height mismatch: h_vertical({h_vertical}) + "
            f"r_lip({r_lip}) + r_floor({r_floor}) = {h_vertical + r_lip + r_floor}, "
            f"but piston_height = {piston_height}"
        )


def _clip_polygon_mm_to_px(points_mm: np.ndarray, grid_size: float, shape: tuple[int, int]) -> np.ndarray:
    points_px = np.rint(points_mm / grid_size).astype(np.int32)
    points_px[:, 0] = np.clip(points_px[:, 0], 0, shape[1] - 1)
    points_px[:, 1] = np.clip(points_px[:, 1], 0, shape[0] - 1)
    return points_px


def build_piston_design_a_components(
    canvas: np.ndarray,
    grid_size: float,
    *,
    r_bore: float,
    r_outer_bowl: float,
    r_topland: float,
    r_ring: float,
    r_inner_bowl: float,
    r_lip: float,
    r_floor: float,
    r_center_circle: float,
    h_vertical: float,
    piston_height: float,
    cylinder_head_offset: float,
) -> dict[str, Any]:
    """Build Design A piston mask and expose intermediate construction steps."""
    _validate_design_a_geometry(
        r_bore=r_bore,
        r_outer_bowl=r_outer_bowl,
        r_topland=r_topland,
        r_ring=r_ring,
        r_inner_bowl=r_inner_bowl,
        r_lip=r_lip,
        r_floor=r_floor,
        h_vertical=h_vertical,
        piston_height=piston_height,
    )

    # Start from the coarse polygonal envelope before carving the rounded bowl features.
    base_mask = np.zeros_like(canvas, dtype=np.uint8)

    # Left block: center ramp from bore axis up to the inner bowl edge.
    trapezoid_points_mm = np.array(
        [
            [r_center_circle, cylinder_head_offset],
            [r_inner_bowl, cylinder_head_offset + piston_height],
            [0, cylinder_head_offset + piston_height],
            [0, cylinder_head_offset],
        ],
        dtype=np.float32,
    )
    trapezoid_points_px = _clip_polygon_mm_to_px(trapezoid_points_mm, grid_size, base_mask.shape)
    cv2.fillPoly(base_mask, [trapezoid_points_px], color=1)

    # Right block: outer crown / ring land section up to the bore radius.
    rectangle_points_mm = np.array(
        [
            [r_inner_bowl + r_floor, cylinder_head_offset],
            [r_bore, cylinder_head_offset],
            [r_bore, cylinder_head_offset + piston_height],
            [r_inner_bowl + r_floor, cylinder_head_offset + piston_height],
        ],
        dtype=np.float32,
    )
    rectangle_points_px = _clip_polygon_mm_to_px(rectangle_points_mm, grid_size, base_mask.shape)
    cv2.fillPoly(base_mask, [rectangle_points_px], color=1)

    # Add the small top-right floor fillet to round the inner bowl corner.
    qt_mask_1 = quarter_of_a_circle_complement_mask(
        canvas,
        grid_size,
        (r_inner_bowl, cylinder_head_offset + piston_height - r_floor),
        r_floor,
        loc="top_right",
        negative_mask=False,
    )
    after_floor = base_mask.astype(bool) | qt_mask_1

    # Remove the bottom-left lip cutout from the outer bowl transition.
    qt_mask_2 = quarter_of_a_circle_complement_mask(
        canvas,
        grid_size,
        (r_outer_bowl, cylinder_head_offset + r_lip),
        r_lip,
        loc="bottom_left",
        negative_mask=True,
    )
    after_lip = after_floor & qt_mask_2

    # Remove the top-land corner so the crown blends into the bore edge.
    qt_mask_3 = quarter_of_a_circle_complement_mask(
        canvas,
        grid_size,
        (r_bore - r_topland, cylinder_head_offset + r_topland),
        r_topland,
        loc="bottom_right",
        negative_mask=True,
    )
    final_mask = after_lip & qt_mask_3

    return {
        "trapezoid_points_mm": trapezoid_points_mm,
        "trapezoid_points_px": trapezoid_points_px,
        "rectangle_points_mm": rectangle_points_mm,
        "rectangle_points_px": rectangle_points_px,
        "base_mask": base_mask.astype(bool),
        "qt_mask_1": qt_mask_1.astype(bool),
        "qt_mask_2": qt_mask_2.astype(bool),
        "qt_mask_3": qt_mask_3.astype(bool),
        "after_floor": after_floor.astype(bool),
        "after_lip": after_lip.astype(bool),
        "final_mask": final_mask.astype(bool),
    }


def piston_design_A(
    canvas: np.ndarray,
    grid_size: float,
    r_bore: float,
    r_outer_bowl: float,
    r_topland: float,
    r_ring: float,
    r_inner_bowl: float,
    r_lip: float,
    r_floor: float,
    r_center_circle: float,
    h_vertical: float,
    piston_height: float,
    cylinder_head_offset: float,
) -> np.ndarray:
    """Return the final Design A piston mask as a boolean array."""
    components = build_piston_design_a_components(
        canvas,
        grid_size,
        r_bore=r_bore,
        r_outer_bowl=r_outer_bowl,
        r_topland=r_topland,
        r_ring=r_ring,
        r_inner_bowl=r_inner_bowl,
        r_lip=r_lip,
        r_floor=r_floor,
        r_center_circle=r_center_circle,
        h_vertical=h_vertical,
        piston_height=piston_height,
        cylinder_head_offset=cylinder_head_offset,
    )
    return components["final_mask"]
