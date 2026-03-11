"""Boundary extraction helpers for single- and multi-plume binary videos."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure


def _boundary_points_one_frame(bw, connectivity=2, x_scale=1.0):
    """Split one frame's boundary points into top and bottom halves."""
    height, _ = bw.shape
    structure = generate_binary_structure(2, 2 if connectivity == 2 else 1)
    boundary = bw & ~binary_erosion(bw, structure=structure, border_value=0)
    if not boundary.any():
        empty = np.empty((0, 2), dtype=np.int32)
        return empty, empty

    ys, xs = np.nonzero(boundary)
    if ys.size == 0:
        empty = np.empty((0, 2), dtype=np.int32)
        return empty, empty

    midline = (height - 1) / 2.0
    top_mask = ys <= midline
    ys_centered = ys.astype(np.float32, copy=False) - midline
    xs_scaled = x_scale * xs.astype(np.float32, copy=False)
    coords_top = np.column_stack((ys_centered[top_mask], xs_scaled[top_mask]))
    coords_bottom = np.column_stack((ys_centered[~top_mask], xs_scaled[~top_mask]))
    return coords_top, coords_bottom


def bw_boundaries_single_plume(bw_vids, connectivity=2, parallel=True, max_workers=None):
    """Compute boundary points for each frame of a single plume video."""
    frame_count = bw_vids.shape[0]
    result = [None] * frame_count

    def work(frame_idx):
        bw = np.asarray(bw_vids[frame_idx], dtype=bool)
        return frame_idx, _boundary_points_one_frame(bw, connectivity)

    if parallel:
        max_workers = max(1, (os.cpu_count() or 1) - 1) if max_workers is None else max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(work, frame_idx) for frame_idx in range(frame_count)]
            for future in as_completed(futures):
                frame_idx, points = future.result()
                result[frame_idx] = points
    else:
        for frame_idx in range(frame_count):
            _, points = work(frame_idx)
            result[frame_idx] = points
    return result


def bw_boundaries_all_plumes(bw_vids, connectivity=2, parallel=True, max_workers=None):
    """Compute boundary points for a batch of plume videos with shape ``(R, F, H, W)``."""
    plume_count, frame_count = bw_vids.shape[:2]
    result = [[None] * frame_count for _ in range(plume_count)]

    def work(plume_idx, frame_idx):
        bw = np.asarray(bw_vids[plume_idx, frame_idx], dtype=bool)
        return plume_idx, frame_idx, _boundary_points_one_frame(bw, connectivity)

    if parallel:
        max_workers = max(1, (os.cpu_count() or 1) - 1) if max_workers is None else max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(work, plume_idx, frame_idx)
                for plume_idx in range(plume_count)
                for frame_idx in range(frame_count)
            ]
            for future in as_completed(futures):
                plume_idx, frame_idx, points = future.result()
                result[plume_idx][frame_idx] = points
    else:
        for plume_idx in range(plume_count):
            for frame_idx in range(frame_count):
                _, _, points = work(plume_idx, frame_idx)
                result[plume_idx][frame_idx] = points
    return result


def _filter_boundary_coords(coords, xlo, xhi):
    coords = np.asarray(coords)
    if coords.size == 0:
        return coords
    x = coords[:, 1]
    xlo_i = int(np.floor(max(0, xlo)))
    xhi_i = int(np.ceil(max(xlo_i, xhi)))
    return coords[(x >= xlo_i) & (x <= xhi_i)]


def bw_boundaries_xband_filter__single_plume(boundary_results, penetration, lo=0.1, hi=0.6):
    """Filter one plume's boundary points to a penetration-relative x-band."""
    frame_count = len(boundary_results)
    if len(penetration) != frame_count:
        raise ValueError(f"penetration must have length {frame_count}, got {len(penetration)}")

    filtered = [None] * frame_count
    for frame_idx in range(frame_count):
        coords_top_all, coords_bottom_all = boundary_results[frame_idx]
        xlo = lo * float(penetration[frame_idx])
        xhi = hi * float(penetration[frame_idx])
        filtered[frame_idx] = (
            _filter_boundary_coords(coords_top_all, xlo, xhi).astype(np.int32, copy=False),
            _filter_boundary_coords(coords_bottom_all, xlo, xhi).astype(np.int32, copy=False),
        )
    return filtered


def bw_boundaries_xband_filter_all_plumes(boundary_results, penetration_old, lo=0.1, hi=0.6):
    """Filter boundary points for a batch of plume videos."""
    plume_count = len(boundary_results)
    frame_count = len(boundary_results[0]) if plume_count > 0 else 0
    if penetration_old.shape != (plume_count, frame_count):
        raise ValueError(
            f"penetration_old must have shape {(plume_count, frame_count)}, got {penetration_old.shape}"
        )

    filtered = [[None] * frame_count for _ in range(plume_count)]
    for plume_idx in range(plume_count):
        for frame_idx in range(frame_count):
            coords_top_all, coords_bottom_all = boundary_results[plume_idx][frame_idx]
            xlo = lo * float(penetration_old[plume_idx, frame_idx])
            xhi = hi * float(penetration_old[plume_idx, frame_idx])
            filtered[plume_idx][frame_idx] = (
                _filter_boundary_coords(coords_top_all, xlo, xhi).astype(np.int32, copy=False),
                _filter_boundary_coords(coords_bottom_all, xlo, xhi).astype(np.int32, copy=False),
            )
    return filtered
