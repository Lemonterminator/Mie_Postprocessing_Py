from __future__ import annotations

from typing import Any

import numpy as np


def _circle_y_and_slope(
    x_mm: np.ndarray | float,
    *,
    circle_center: tuple[float, float],
    radius_mm: float,
    branch_sign: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    x_arr = np.asarray(x_mm, dtype=float)
    cx, cy = circle_center
    inside = radius_mm**2 - (x_arr - cx) ** 2
    if np.any(inside < -1e-9):
        raise ValueError("x lies outside the circle domain for the requested radius.")

    root = np.sqrt(np.clip(inside, 0.0, None))
    y = cy + branch_sign * root

    slope = np.zeros_like(x_arr, dtype=float)
    nonzero = root > 0
    slope[nonzero] = -(x_arr[nonzero] - cx) / (branch_sign * root[nonzero])

    if np.isscalar(x_mm):
        return float(y), float(slope)
    return y, slope


def _assert_tangent(
    name: str,
    *,
    left_y: float,
    right_y: float,
    left_slope: float,
    right_slope: float,
    depth_tol_mm: float,
    slope_tol: float,
) -> None:
    if not np.isclose(left_y, right_y, atol=depth_tol_mm, rtol=0.0):
        raise AssertionError(
            f"{name} depth mismatch: {left_y:.6f} mm vs {right_y:.6f} mm"
        )
    if not np.isclose(left_slope, right_slope, atol=slope_tol, rtol=0.0):
        raise AssertionError(
            f"{name} slope mismatch: {left_slope:.6f} vs {right_slope:.6f}"
        )


def infer_design_b_arc2_center(
    *,
    flat_depth_mm: float,
    x_flat_end_mm: float,
    x_arc1_arc2_join_mm: float,
    arc1_radius_mm: float,
    arc2_radius_mm: float,
) -> tuple[float, float]:
    """Infer the second-arc center from tangency at the first-arc endpoint."""
    arc1_center = (x_flat_end_mm, flat_depth_mm + arc1_radius_mm)
    join_y, _ = _circle_y_and_slope(
        x_arc1_arc2_join_mm,
        circle_center=arc1_center,
        radius_mm=arc1_radius_mm,
        branch_sign=-1.0,
    )

    tangent_point = np.array([x_arc1_arc2_join_mm, join_y], dtype=float)
    radial_unit = (tangent_point - np.array(arc1_center, dtype=float)) / arc1_radius_mm

    candidates = (
        tangent_point + arc2_radius_mm * radial_unit,
        tangent_point - arc2_radius_mm * radial_unit,
    )
    negative_y_candidates = [tuple(center) for center in candidates if center[1] < 0.0]

    if len(negative_y_candidates) != 1:
        raise ValueError(
            "Expected exactly one tangent-compatible second-arc center with negative y."
        )

    center = negative_y_candidates[0]
    return float(center[0]), float(center[1])


def _design_b_depth_per_column(
    col_x_mm: np.ndarray,
    *,
    flat_depth_mm: float,
    x_flat_end_mm: float,
    x_arc1_arc2_join_mm: float,
    arc1_radius_mm: float,
    x_arc2_lip_join_mm: float,
    arc2_radius_mm: float,
    lip_radius_mm: float,
    x_lip_flat_join_mm: float,
    x_topland_start_mm: float,
    r_topland: float,
    arc2_center: tuple[float, float],
) -> np.ndarray:
    depth_mm = np.zeros_like(col_x_mm, dtype=float)

    arc1_center = (x_flat_end_mm, flat_depth_mm + arc1_radius_mm)
    lip_center = (x_lip_flat_join_mm, lip_radius_mm)
    topland_center = (x_topland_start_mm, r_topland)

    mask_flat_center = col_x_mm <= x_flat_end_mm
    mask_arc1 = (col_x_mm >= x_flat_end_mm) & (col_x_mm <= x_arc1_arc2_join_mm)
    mask_arc2 = (col_x_mm >= x_arc1_arc2_join_mm) & (col_x_mm <= x_arc2_lip_join_mm)
    mask_lip = (col_x_mm >= x_arc2_lip_join_mm) & (col_x_mm <= x_lip_flat_join_mm)
    mask_flat_outer = (col_x_mm >= x_lip_flat_join_mm) & (col_x_mm <= x_topland_start_mm)
    mask_topland = (col_x_mm >= x_topland_start_mm) & (col_x_mm <= x_topland_start_mm + r_topland)

    depth_mm[mask_flat_center] = flat_depth_mm
    depth_mm[mask_arc1], _ = _circle_y_and_slope(
        col_x_mm[mask_arc1],
        circle_center=arc1_center,
        radius_mm=arc1_radius_mm,
        branch_sign=-1.0,
    )
    depth_mm[mask_arc2], _ = _circle_y_and_slope(
        col_x_mm[mask_arc2],
        circle_center=arc2_center,
        radius_mm=arc2_radius_mm,
        branch_sign=1.0,
    )
    depth_mm[mask_lip], _ = _circle_y_and_slope(
        col_x_mm[mask_lip],
        circle_center=lip_center,
        radius_mm=lip_radius_mm,
        branch_sign=-1.0,
    )
    depth_mm[mask_flat_outer] = 0.0
    depth_mm[mask_topland], _ = _circle_y_and_slope(
        col_x_mm[mask_topland],
        circle_center=topland_center,
        radius_mm=r_topland,
        branch_sign=-1.0,
    )

    return depth_mm


def build_piston_design_b(
    canvas: np.ndarray,
    grid_size: float,
    *,
    cylinder_head_offset: float,
    r_bore: float,
    piston_height: float,
    flat_depth_mm: float = 5.0,
    x_flat_end_mm: float = 15.0,
    x_arc1_arc2_join_mm: float = 35.0,
    arc1_radius_mm: float = 63.98,
    x_arc2_lip_join_mm: float = 100.0,
    arc2_radius_mm: float = 78.38,
    lip_radius_mm: float = 6.0,
    x_lip_flat_join_mm: float = 103.1,
    x_topland_start_mm: float = 115.0,
    r_topland: float = 10.0,
    tangent_depth_tol_mm: float = 5e-3,
    tangent_slope_tol: float = 5e-4,
) -> dict[str, Any]:
    """Build a parameterized Design B piston mask from a piecewise crown profile.

    Depth is measured downward from ``cylinder_head_offset`` in mm.
    The designer's topland center ``y = -r_topland`` in a y-up sketch maps to
    ``+r_topland`` in this depth-down convention.
    """
    height_px, width_px = canvas.shape

    if not np.isclose(x_topland_start_mm + r_topland, r_bore):
        raise ValueError(
            f"Bore mismatch: x_topland_start_mm({x_topland_start_mm}) + "
            f"r_topland({r_topland}) must equal r_bore({r_bore})."
        )

    if not (
        0.0 <= x_flat_end_mm <= x_arc1_arc2_join_mm <= x_arc2_lip_join_mm <= x_lip_flat_join_mm <= x_topland_start_mm
    ):
        raise ValueError("Expected nondecreasing Design B segment breakpoints.")

    if not (x_lip_flat_join_mm - lip_radius_mm <= x_arc2_lip_join_mm <= x_lip_flat_join_mm):
        raise ValueError("The lip join must lie inside the lip-circle x-span.")

    arc1_center = (x_flat_end_mm, flat_depth_mm + arc1_radius_mm)
    arc2_center = infer_design_b_arc2_center(
        flat_depth_mm=flat_depth_mm,
        x_flat_end_mm=x_flat_end_mm,
        x_arc1_arc2_join_mm=x_arc1_arc2_join_mm,
        arc1_radius_mm=arc1_radius_mm,
        arc2_radius_mm=arc2_radius_mm,
    )
    lip_center = (x_lip_flat_join_mm, lip_radius_mm)
    topland_center = (x_topland_start_mm, r_topland)

    flat_arc1_y, flat_arc1_slope = _circle_y_and_slope(
        x_flat_end_mm,
        circle_center=arc1_center,
        radius_mm=arc1_radius_mm,
        branch_sign=-1.0,
    )
    _assert_tangent(
        "center flat -> arc1",
        left_y=flat_depth_mm,
        right_y=flat_arc1_y,
        left_slope=0.0,
        right_slope=flat_arc1_slope,
        depth_tol_mm=tangent_depth_tol_mm,
        slope_tol=tangent_slope_tol,
    )

    arc1_join_y, arc1_join_slope = _circle_y_and_slope(
        x_arc1_arc2_join_mm,
        circle_center=arc1_center,
        radius_mm=arc1_radius_mm,
        branch_sign=-1.0,
    )
    arc2_join_y, arc2_join_slope = _circle_y_and_slope(
        x_arc1_arc2_join_mm,
        circle_center=arc2_center,
        radius_mm=arc2_radius_mm,
        branch_sign=1.0,
    )
    _assert_tangent(
        "arc1 -> arc2",
        left_y=arc1_join_y,
        right_y=arc2_join_y,
        left_slope=arc1_join_slope,
        right_slope=arc2_join_slope,
        depth_tol_mm=tangent_depth_tol_mm,
        slope_tol=tangent_slope_tol,
    )

    arc2_lip_y, arc2_lip_slope = _circle_y_and_slope(
        x_arc2_lip_join_mm,
        circle_center=arc2_center,
        radius_mm=arc2_radius_mm,
        branch_sign=1.0,
    )
    lip_join_y, lip_join_slope = _circle_y_and_slope(
        x_arc2_lip_join_mm,
        circle_center=lip_center,
        radius_mm=lip_radius_mm,
        branch_sign=-1.0,
    )
    _assert_tangent(
        "arc2 -> lip",
        left_y=arc2_lip_y,
        right_y=lip_join_y,
        left_slope=arc2_lip_slope,
        right_slope=lip_join_slope,
        depth_tol_mm=tangent_depth_tol_mm,
        slope_tol=tangent_slope_tol,
    )

    lip_flat_y, lip_flat_slope = _circle_y_and_slope(
        x_lip_flat_join_mm,
        circle_center=lip_center,
        radius_mm=lip_radius_mm,
        branch_sign=-1.0,
    )
    _assert_tangent(
        "lip -> outer flat",
        left_y=lip_flat_y,
        right_y=0.0,
        left_slope=lip_flat_slope,
        right_slope=0.0,
        depth_tol_mm=tangent_depth_tol_mm,
        slope_tol=tangent_slope_tol,
    )

    outer_flat_topland_y, outer_flat_topland_slope = _circle_y_and_slope(
        x_topland_start_mm,
        circle_center=topland_center,
        radius_mm=r_topland,
        branch_sign=-1.0,
    )
    _assert_tangent(
        "outer flat -> topland",
        left_y=0.0,
        right_y=outer_flat_topland_y,
        left_slope=0.0,
        right_slope=outer_flat_topland_slope,
        depth_tol_mm=tangent_depth_tol_mm,
        slope_tol=tangent_slope_tol,
    )

    derived_x_lip_flat_join_mm = x_arc2_lip_join_mm + np.sqrt(
        np.clip(lip_radius_mm**2 - (lip_radius_mm - arc2_lip_y) ** 2, 0.0, None)
    )
    if not np.isclose(
        x_lip_flat_join_mm, derived_x_lip_flat_join_mm, atol=tangent_depth_tol_mm, rtol=0.0
    ):
        raise AssertionError(
            f"lip flat join mismatch: {x_lip_flat_join_mm:.6f} mm vs "
            f"{derived_x_lip_flat_join_mm:.6f} mm"
        )

    col_x_mm = np.arange(width_px, dtype=float) * grid_size
    depth_mm = _design_b_depth_per_column(
        col_x_mm,
        flat_depth_mm=flat_depth_mm,
        x_flat_end_mm=x_flat_end_mm,
        x_arc1_arc2_join_mm=x_arc1_arc2_join_mm,
        arc1_radius_mm=arc1_radius_mm,
        x_arc2_lip_join_mm=x_arc2_lip_join_mm,
        arc2_radius_mm=arc2_radius_mm,
        lip_radius_mm=lip_radius_mm,
        x_lip_flat_join_mm=x_lip_flat_join_mm,
        x_topland_start_mm=x_topland_start_mm,
        r_topland=r_topland,
        arc2_center=arc2_center,
    )
    depth_mm = np.clip(depth_mm, 0.0, piston_height)

    crown_row_px = np.clip(
        np.round((cylinder_head_offset + depth_mm) / grid_size).astype(int),
        0,
        height_px,
    )

    final_mask = np.zeros((height_px, width_px), dtype=bool)
    for col, row in enumerate(crown_row_px):
        final_mask[row:, col] = True

    piston_bottom_px = int(round((cylinder_head_offset + piston_height) / grid_size))
    final_mask[piston_bottom_px:, :] = True

    bore_px = int(round(r_bore / grid_size))
    final_mask[:, bore_px:] = False

    return {
        "arc1_center_mm": np.array(arc1_center, dtype=float),
        "arc2_center_mm": np.array(arc2_center, dtype=float),
        "lip_center_mm": np.array(lip_center, dtype=float),
        "topland_center_mm": np.array(topland_center, dtype=float),
        "derived_x_lip_flat_join_mm": float(derived_x_lip_flat_join_mm),
        "arc1_arc2_join_mm": np.array([x_arc1_arc2_join_mm, arc1_join_y], dtype=float),
        "arc2_lip_join_mm": np.array([x_arc2_lip_join_mm, arc2_lip_y], dtype=float),
        "crown_x_mm": col_x_mm,
        "crown_depth_mm": depth_mm,
        "crown_row_px": crown_row_px,
        "final_mask": final_mask,
    }


def piston_design_B(
    canvas: np.ndarray,
    grid_size: float,
    *,
    cylinder_head_offset: float,
    r_bore: float,
    piston_height: float,
    flat_depth_mm: float = 5.0,
    x_flat_end_mm: float = 15.0,
    x_arc1_arc2_join_mm: float = 35.0,
    arc1_radius_mm: float = 63.98,
    x_arc2_lip_join_mm: float = 100.0,
    arc2_radius_mm: float = 78.38,
    lip_radius_mm: float = 6.0,
    x_lip_flat_join_mm: float = 103.1,
    x_topland_start_mm: float = 115.0,
    r_topland: float = 10.0,
    tangent_depth_tol_mm: float = 5e-3,
    tangent_slope_tol: float = 5e-4,
) -> np.ndarray:
    """Return the final Design B piston mask as a boolean array."""
    return build_piston_design_b(
        canvas,
        grid_size,
        cylinder_head_offset=cylinder_head_offset,
        r_bore=r_bore,
        piston_height=piston_height,
        flat_depth_mm=flat_depth_mm,
        x_flat_end_mm=x_flat_end_mm,
        x_arc1_arc2_join_mm=x_arc1_arc2_join_mm,
        arc1_radius_mm=arc1_radius_mm,
        x_arc2_lip_join_mm=x_arc2_lip_join_mm,
        arc2_radius_mm=arc2_radius_mm,
        lip_radius_mm=lip_radius_mm,
        x_lip_flat_join_mm=x_lip_flat_join_mm,
        x_topland_start_mm=x_topland_start_mm,
        r_topland=r_topland,
        tangent_depth_tol_mm=tangent_depth_tol_mm,
        tangent_slope_tol=tangent_slope_tol,
    )["final_mask"]
