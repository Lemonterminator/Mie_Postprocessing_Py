from __future__ import annotations

from typing import Any

import numpy as np

from piston.utils import quarter_of_a_circle_complement_mask


def _quintic_hermite_segment(
    p0: np.ndarray, p1: np.ndarray,
    v0: np.ndarray, v1: np.ndarray,
    a0: np.ndarray, a1: np.ndarray,
    num_points: int = 1000,
) -> np.ndarray:
    t = np.linspace(0, 1, num_points)
    h0 = -6*t**5 + 15*t**4 - 10*t**3 + 1
    h1 = -3*t**5 +  8*t**4 -  6*t**3 + t
    h2 = 0.5 * (-t**5 + 3*t**4 - 3*t**3 + t**2)
    h3 = 0.5 * ( t**5 - 2*t**4 +   t**3)
    h4 = -3*t**5 +  7*t**4 -  4*t**3
    h5 =  6*t**5 - 15*t**4 + 10*t**3
    return (
        np.outer(h0, p0) + np.outer(h1, v0) + np.outer(h2, a0) +
        np.outer(h3, a1) + np.outer(h4, v1) + np.outer(h5, p1)
    )


def _crown_depth_per_column(
    ctrl_pts_mm: np.ndarray,
    ctrl_vels: np.ndarray,
    ctrl_accels: np.ndarray,
    grid_size: float,
    width_px: int,
) -> np.ndarray:
    """Interpolate crown depth (mm, relative to cylinder_head_offset) at every pixel column."""
    x_parts, d_parts = [], []
    for i in range(len(ctrl_pts_mm) - 1):
        seg = _quintic_hermite_segment(
            ctrl_pts_mm[i], ctrl_pts_mm[i + 1],
            ctrl_vels[i],   ctrl_vels[i + 1],
            ctrl_accels[i], ctrl_accels[i + 1],
        )
        x_parts.append(seg[:, 0])
        d_parts.append(seg[:, 1])

    x_all = np.concatenate(x_parts)
    d_all = np.concatenate(d_parts)

    order = np.argsort(x_all)
    x_all, d_all = x_all[order], d_all[order]

    col_x_mm = np.arange(width_px) * grid_size
    return np.interp(col_x_mm, x_all, d_all, left=d_all[0], right=d_all[-1])


def build_piston_design_hermite(
    canvas: np.ndarray,
    grid_size: float,
    *,
    cylinder_head_offset: float,
    r_bore: float,
    r_topland: float,
    piston_height: float,
    ctrl_pts_mm: np.ndarray,
    ctrl_vels: np.ndarray,
    ctrl_accels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a Hermite-crown piston mask.

    ctrl_pts_mm : (N, 2) array — (radial_mm, depth_mm) where depth is
                  measured downward from cylinder_head_offset.
    ctrl_vels   : (N, 2) tangent vectors in the same (radial, depth) space.
    ctrl_accels : (N, 2) second-derivative vectors, defaults to zero.
    """
    height_px, width_px = canvas.shape

    if ctrl_accels is None:
        ctrl_accels = np.zeros_like(ctrl_pts_mm, dtype=float)

    # ── Crown profile ─────────────────────────────────────────────────────────
    depth_mm = _crown_depth_per_column(
        ctrl_pts_mm, ctrl_vels, ctrl_accels, grid_size, width_px
    )
    depth_mm = np.clip(depth_mm, 0.0, piston_height)

    # ── Rasterise: fill solid below crown surface ─────────────────────────────
    crown_row_px = np.clip(
        np.round((cylinder_head_offset + depth_mm) / grid_size).astype(int),
        0, height_px,
    )
    final_mask = np.zeros((height_px, width_px), dtype=bool)
    for col, row in enumerate(crown_row_px):
        final_mask[row:, col] = True

    # ── Fill below piston bottom ──────────────────────────────────────────────
    piston_bottom_px = int(round((cylinder_head_offset + piston_height) / grid_size))
    final_mask[piston_bottom_px:, :] = True

    # ── Topland corner clip (bottom_right quarter-circle complement) ───────────
    qt_topland = quarter_of_a_circle_complement_mask(
        canvas, grid_size,
        (r_bore - r_topland, cylinder_head_offset + r_topland),
        r_topland,
        loc="bottom_right",
        negative_mask=True,
    )
    final_mask = final_mask & qt_topland

    # ── Clip to bore radius ───────────────────────────────────────────────────
    bore_px = int(round(r_bore / grid_size))
    final_mask[:, bore_px:] = False

    return {
        "crown_depth_mm": depth_mm,
        "crown_row_px": crown_row_px,
        "final_mask": final_mask,
    }


def piston_design_hermite(
    canvas: np.ndarray,
    grid_size: float,
    *,
    cylinder_head_offset: float,
    r_bore: float,
    r_topland: float,
    piston_height: float,
    ctrl_pts_mm: np.ndarray,
    ctrl_vels: np.ndarray,
    ctrl_accels: np.ndarray | None = None,
) -> np.ndarray:
    """Return the final Hermite-crown piston mask as a boolean array."""
    return build_piston_design_hermite(
        canvas, grid_size,
        cylinder_head_offset=cylinder_head_offset,
        r_bore=r_bore, r_topland=r_topland, piston_height=piston_height,
        ctrl_pts_mm=ctrl_pts_mm, ctrl_vels=ctrl_vels, ctrl_accels=ctrl_accels,
    )["final_mask"]
