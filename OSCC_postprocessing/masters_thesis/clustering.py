"""Cluster and contour utilities for spray-mask cleanup."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from sklearn.cluster import DBSCAN


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def fast_alpha_shape(points: np.ndarray, alpha: float, max_points: int = 2000) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts.astype(int)

    pts = np.unique(pts, axis=0)
    if len(pts) < 4:
        return pts.astype(int)

    if len(pts) > max_points:
        try:
            hull = ConvexHull(pts)
            hull_idx = set(hull.vertices.tolist())
        except QhullError:
            hull_idx = set()

        keep_idx = list(hull_idx)
        remaining = [i for i in range(len(pts)) if i not in hull_idx]
        n_needed = max_points - len(keep_idx)
        if n_needed > 0 and remaining:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False)
            keep_idx += sample_idx.tolist()
        pts = pts[keep_idx]

    try:
        tri = Delaunay(pts)
        simplices = tri.simplices
    except QhullError:
        try:
            hull = ConvexHull(pts)
            return pts[hull.vertices].astype(int)
        except QhullError:
            return pts.astype(int)

    pa, pb, pc = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    a = np.linalg.norm(pb - pc, axis=1)
    b = np.linalg.norm(pc - pa, axis=1)
    c = np.linalg.norm(pa - pb, axis=1)

    s = 0.5 * (a + b + c)
    area = np.maximum(s * (s - a) * (s - b) * (s - c), 1e-12) ** 0.5
    circum_r = (a * b * c) / (4.0 * area)

    good = simplices[circum_r < alpha]
    if good.size == 0:
        hull = ConvexHull(pts)
        return pts[hull.vertices].astype(int)

    edges = np.vstack([good[:, [0, 1]], good[:, [1, 2]], good[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq_edges[counts == 1]

    if boundary.size == 0:
        hull = ConvexHull(pts)
        return pts[hull.vertices].astype(int)

    adj: dict[int, list[int]] = {}
    for u, v in boundary:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))

    loops = []
    visited: set[int] = set()
    max_steps_per_loop = max(1000, len(adj) * 3)
    for start in list(adj.keys()):
        if start in visited:
            continue
        cur = start
        prev = None
        loop = [cur]
        steps = 0
        while True:
            steps += 1
            if steps > max_steps_per_loop:
                break
            neighs = [n for n in adj.get(cur, []) if n != prev]
            if not neighs:
                break
            nxt = neighs[0]
            if nxt == start:
                loop.append(nxt)
                visited.update(loop)
                break
            if nxt in visited:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) > 2:
            if loop[0] == loop[-1]:
                loop = loop[:-1]
            loops.append(pts[loop])

    if not loops:
        hull = ConvexHull(pts)
        return pts[hull.vertices].astype(int)

    best = max(loops, key=polygon_area)
    return np.asarray(best).astype(int)


def create_cluster_mask(mask: np.ndarray, cluster_distance: int = 30, alpha: int = 30) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        for p in cnt:
            points.append(p[0])

    if not points:
        return np.zeros_like(mask)

    pts = np.array(points)
    clustering = DBSCAN(eps=cluster_distance, min_samples=10).fit(pts)
    labels = clustering.labels_

    canvas = np.zeros_like(mask)
    clusters_with_area = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_pts = pts[labels == label]
        if len(cluster_pts) >= 5:
            hull_pts = fast_alpha_shape(cluster_pts, alpha)
            area = cv2.contourArea(hull_pts.astype(np.float32))
            clusters_with_area.append((area, hull_pts))

    clusters_with_area.sort(key=lambda x: x[0], reverse=True)
    for _, hull_pts in clusters_with_area[:1]:
        cv2.fillPoly(canvas, [hull_pts], 255)

    return canvas


def overlay_cluster_outline(frame: np.ndarray, cluster_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(cluster_mask)
    for cnt in contours:
        cv2.polylines(canvas, [cnt], isClosed=True, color=255, thickness=2)
    return cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)


def fill_holes_in_mask(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    h, w = filled.shape[:2]

    for y in range(h):
        if filled[y, 0] == 0:
            cv2.floodFill(filled, None, (0, y), 255)
        if filled[y, w - 1] == 0:
            cv2.floodFill(filled, None, (w - 1, y), 255)

    for x in range(w):
        if filled[0, x] == 0:
            cv2.floodFill(filled, None, (x, 0), 255)
        if filled[h - 1, x] == 0:
            cv2.floodFill(filled, None, (x, h - 1), 255)

    filled = cv2.bitwise_not(filled)
    filled = cv2.bitwise_or(filled, mask)
    return filled


def keep_largest_blob(
    mask: np.ndarray,
    horizontal_threshold: int = 50,
    spray_origin: tuple[int, int] | None = None,
) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask)

    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num_labels <= 1:
        return np.zeros_like(mask)

    largest_label = 0
    largest_size = 0
    for label in range(1, num_labels):
        size = np.count_nonzero(labels == label)
        if size > largest_size:
            largest_size = size
            largest_label = label

    result = np.zeros_like(mask)
    result[labels == largest_label] = 255

    if spray_origin is not None:
        origin_y = int(spray_origin[1])
        largest_mask = (labels == largest_label).astype(np.uint8)
        if origin_y < 0 or origin_y >= mask.shape[0]:
            return result

        largest_row = np.where(largest_mask[origin_y, :] > 0)[0]
        if len(largest_row) == 0:
            return result

        largest_x_min = largest_row.min()
        largest_x_max = largest_row.max()
        for label in range(1, num_labels):
            if label == largest_label:
                continue

            blob_mask = (labels == label).astype(np.uint8)
            blob_row = np.where(blob_mask[origin_y, :] > 0)[0]
            if len(blob_row) == 0:
                continue

            blob_x_min = blob_row.min()
            blob_x_max = blob_row.max()

            if blob_x_max < largest_x_min:
                horizontal_distance = largest_x_min - blob_x_max
            elif blob_x_min > largest_x_max:
                horizontal_distance = blob_x_min - largest_x_max
            else:
                horizontal_distance = 0

            if horizontal_distance <= horizontal_threshold:
                result[labels == label] = 255
        return result

    largest_mask = (labels == largest_label).astype(np.uint8)
    largest_coords = np.column_stack(np.where(largest_mask > 0))
    if len(largest_coords) == 0:
        return result

    largest_x_min = largest_coords[:, 1].min()
    largest_x_max = largest_coords[:, 1].max()
    for label in range(1, num_labels):
        if label == largest_label:
            continue

        blob_mask = (labels == label).astype(np.uint8)
        blob_coords = np.column_stack(np.where(blob_mask > 0))
        if len(blob_coords) == 0:
            continue

        blob_x_min = blob_coords[:, 1].min()
        blob_x_max = blob_coords[:, 1].max()

        if blob_x_max < largest_x_min:
            horizontal_distance = largest_x_min - blob_x_max
        elif blob_x_min > largest_x_max:
            horizontal_distance = blob_x_min - largest_x_max
        else:
            horizontal_distance = 0

        if horizontal_distance <= horizontal_threshold:
            result[labels == label] = 255

    return result
