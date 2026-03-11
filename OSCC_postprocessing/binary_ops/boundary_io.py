"""I/O helpers for serialized boundary point clouds."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def load_boundary_file(file_path, return_BW_video=False, F: int = 0, H: int = 0, W: int = 0):
    """Load boundary CSV data and optionally rasterize it back into a BW video."""
    if return_BW_video and not (F > 0 and H > 0 and W > 0):
        raise ValueError("When return_BW_video=True, F/H/W must be positive.")

    boundary_bw = pd.read_csv(file_path)
    required_cols = {"frame", "x", "y"}
    missing_cols = required_cols - set(boundary_bw.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if boundary_bw.empty:
        bw_video = np.zeros((F, H, W), dtype=np.uint8) if return_BW_video else None
        return bw_video, []

    grouped = boundary_bw.groupby("frame", sort=True)
    last_frame = int(boundary_bw["frame"].max())
    boundary_list = [None] * (last_frame + 1)
    bw_video = np.zeros((F, H, W), dtype=np.uint8) if return_BW_video else None

    for frame, group in grouped:
        frame = int(frame)
        pts = group[["x", "y"]].to_numpy(dtype=np.float32)
        boundary_list[frame] = pts

        if not return_BW_video or frame < 0 or frame >= F:
            continue

        x = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, W - 1)
        y = np.clip(np.rint(pts[:, 1] + H // 2).astype(np.int32), 0, H - 1)

        edge = np.zeros((H, W), dtype=np.uint8)
        edge[y, x] = 255
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        edge_pad = np.zeros((H + 2, W + 2), dtype=np.uint8)
        edge_pad[1:-1, 1:-1] = edge
        flooded = edge_pad.copy()
        flood_mask = np.zeros((H + 4, W + 4), dtype=np.uint8)
        cv2.floodFill(flooded, flood_mask, (0, 0), 255)

        interior_pad = cv2.bitwise_and(cv2.bitwise_not(flooded), cv2.bitwise_not(edge_pad))
        filled = interior_pad[1:-1, 1:-1]
        bw_video[frame] = cv2.bitwise_or(edge, filled)

    return bw_video, boundary_list
