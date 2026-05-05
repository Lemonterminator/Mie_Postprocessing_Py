"""OpenCV/Matplotlib playback helpers for grayscale spray videos.

This module is intentionally I/O-facing and lightweight. Its role is to turn
already-computed arrays into quickly inspectable visual output for notebooks,
manual QA, and debugging. The functions here do not perform scientific
processing; they only:

- convert frames of varying dtype into displayable ``uint8`` images
- play one or more videos with OpenCV
- overlay precomputed boundary points on frames
- save those overlays back to disk
- preview segmented plume strips with Matplotlib

Historically the boundary-overlay functions contained duplicated coordinate
conversion logic. That logic is now centralized so playback and save paths use
the same interpretation of boundary formats.
"""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "play_video_cv2",
    "play_videos_side_by_side",
    "play_video_with_boundaries_cv2",
    "save_video_with_boundaries_cv2",
    "play_segments_with_boundaries",
    "save_multiplume_with_boundaries_cv2",
    "play_multiplume_with_boundaries_cv2",
    "adaptive_tile_grid",
]


def adaptive_tile_grid(num_plumes: int) -> tuple[int, int]:
    """Pick a near-square (rows, cols) grid for ``num_plumes`` cells.

    Prefers landscape (cols >= rows) and minimizes empty cells, breaking ties
    toward squareness. Tested for 4-15 plumes:
        4->(2,2)  5->(2,3)  6->(2,3)  7->(2,4)  8->(2,4)  9->(3,3)
        10->(2,5) 11->(3,4) 12->(3,4) 13->(2,7) 14->(2,7) 15->(3,5)
    """
    import math

    if num_plumes <= 0:
        return (1, 1)
    best = None
    for rows in range(1, num_plumes + 1):
        cols = math.ceil(num_plumes / rows)
        if cols < rows:
            continue
        empty = rows * cols - num_plumes
        score = (empty, abs(cols - rows))
        if best is None or score < best[0]:
            best = (score, rows, cols)
    return (best[1], best[2])


def _swap_boundary_yx_for_frame(item):
    """Swap (y, x) <-> (x, y) for one frame's boundary entry."""
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and len(item) == 2:
        lower, upper = item

        def _swap(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return arr
            return arr[:, ::-1].copy()
        return (_swap(lower), _swap(upper))
    arr = np.asarray(item)
    if arr.size == 0:
        return arr
    return arr[:, ::-1].copy()


def _swap_plume_boundary_yx(boundary):
    """Apply yx<->xy swap to every frame in one plume's boundary container."""
    if boundary is None:
        return None
    return [_swap_boundary_yx_for_frame(item) for item in boundary]


def _render_one_plume_frame(
    frame_2d, boundary_for_plume, frame_idx, *,
    gain, binarize, thresh, kernel, color_top, color_bottom, alpha,
):
    """Build one plume's BGR frame with boundary overlay."""
    H, W = frame_2d.shape[-2], frame_2d.shape[-1]
    frame_u8 = _frame_to_uint8(frame_2d, gain=gain, binarize=binarize, thresh=thresh)
    frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    if boundary_for_plume is not None:
        mask_top, mask_bot = _boundary_masks(boundary_for_plume, frame_idx, H, W, kernel=kernel)
        frame_bgr = _overlay_boundary_masks(
            frame_bgr, mask_top, mask_bot,
            color_top=color_top, color_bottom=color_bottom, alpha=alpha,
        )
    return frame_bgr


def _tile_plume_frames(plume_bgr_frames, rows: int, cols: int, pad_color=(20, 20, 20)):
    """Tile P plume frames into one (rows*H, cols*W, 3) image, padding empties."""
    P = len(plume_bgr_frames)
    H, W, _ = plume_bgr_frames[0].shape
    pad = np.full((H, W, 3), pad_color, dtype=np.uint8)
    grid_rows = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            i = r * cols + c
            cells.append(plume_bgr_frames[i] if i < P else pad)
        grid_rows.append(cv2.hconcat(cells))
    return cv2.vconcat(grid_rows)


def _multiplume_render_iter(
    plume_video_PFHW, boundaries_per_plume, *,
    swap_axes, tile, gain, binarize, thresh,
    color_top, color_bottom, thickness, alpha,
):
    """Yield (F) tiled BGR frames for a (P, F, H, W) host array.

    If ``swap_axes`` is True, swaps the last two axes of the video
    (H<->W, top-down jet display) and applies the matching (y, x)<->(x, y)
    transform to every plume's boundary container.
    """
    if color_bottom is None:
        color_bottom = color_top

    video = np.asarray(plume_video_PFHW)
    if video.ndim != 4:
        raise ValueError(f"plume_video_PFHW must be 4D (P, F, H, W), got shape {video.shape}")
    P, F = video.shape[0], video.shape[1]
    if F == 0 or P == 0:
        return

    if swap_axes:
        video = np.ascontiguousarray(np.swapaxes(video, -2, -1))
        boundaries = [_swap_plume_boundary_yx(b) for b in boundaries_per_plume]
    else:
        boundaries = list(boundaries_per_plume)

    if tile is None:
        rows, cols = adaptive_tile_grid(P)
    else:
        rows, cols = tile

    kernel = None
    ksz = max(1, 2 * int(thickness) + 1)
    if ksz > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))

    for f in range(F):
        plume_frames = [
            _render_one_plume_frame(
                video[p, f], boundaries[p] if p < len(boundaries) else None, f,
                gain=gain, binarize=binarize, thresh=thresh, kernel=kernel,
                color_top=color_top, color_bottom=color_bottom, alpha=alpha,
            )
            for p in range(P)
        ]
        yield _tile_plume_frames(plume_frames, rows, cols)


def save_multiplume_with_boundaries_cv2(
    plume_video_PFHW,
    boundaries_per_plume,
    save_path,
    *,
    fps=15,
    tile=None,
    swap_axes=True,
    gain=1.0,
    binarize=False,
    thresh=0.5,
    color_top=(0, 0, 255),
    color_bottom=None,
    thickness=1,
    alpha=1.0,
):
    """Tile P plumes per frame with boundary overlay and write one AVI.

    Designed to be submitted to a background ThreadPoolExecutor: it only touches
    host arrays and one VideoWriter instance, so multiple invocations can run
    concurrently as long as their inputs are independent host buffers.
    """
    save_path = os.path.abspath(save_path)
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    frames = _multiplume_render_iter(
        plume_video_PFHW, boundaries_per_plume,
        swap_axes=swap_axes, tile=tile,
        gain=gain, binarize=binarize, thresh=thresh,
        color_top=color_top, color_bottom=color_bottom,
        thickness=thickness, alpha=alpha,
    )

    writer = None
    try:
        for tiled in frames:
            if writer is None:
                Ht, Wt = tiled.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(save_path, fourcc, float(fps), (int(Wt), int(Ht)))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for: {save_path}")
            writer.write(tiled)
    finally:
        if writer is not None:
            writer.release()
    return save_path


def play_multiplume_with_boundaries_cv2(
    plume_video_PFHW,
    boundaries_per_plume,
    *,
    fps=15,
    tile=None,
    swap_axes=True,
    gain=1.0,
    binarize=False,
    thresh=0.5,
    window_name="Multi-plume + Boundary",
    color_top=(0, 0, 255),
    color_bottom=None,
    thickness=1,
    alpha=1.0,
):
    """Synchronously preview tiled multi-plume frames on the main thread.

    Blocking by design: cv2 HighGUI requires the main thread on most platforms,
    so this should not be submitted to a worker pool.
    """
    intv = max(1, int(round(1000.0 / float(fps))))
    frames = _multiplume_render_iter(
        plume_video_PFHW, boundaries_per_plume,
        swap_axes=swap_axes, tile=tile,
        gain=gain, binarize=binarize, thresh=thresh,
        color_top=color_top, color_bottom=color_bottom,
        thickness=thickness, alpha=alpha,
    )
    try:
        for tiled in frames:
            cv2.imshow(window_name, tiled)
            if cv2.waitKey(intv) & 0xFF == ord("q"):
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass


def _frame_to_uint8(frame, gain=1.0, binarize=False, thresh=0.5):
    """Convert one frame to displayable 8-bit grayscale.

    Conversion policy matches the historic viewer behaviour used throughout the
    repo:

    - boolean input becomes ``0`` or ``255``
    - integer input is assumed to be roughly 12-16 bit and scaled down
    - floating input is assumed to live in ``[0, 1]`` and scaled up
    """
    if binarize:
        frame_bool = frame if frame.dtype == bool else np.asarray(frame) > thresh
        return frame_bool.astype(np.uint8) * 255

    dtype = frame.dtype
    if np.issubdtype(dtype, np.integer):
        out = (np.asarray(frame) / 16).astype(np.uint8)
        if gain != 1.0:
            out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return out

    if np.issubdtype(dtype, np.floating):
        return np.clip(gain * (np.asarray(frame) * 255.0), 0, 255).astype(np.uint8)

    out = (np.asarray(frame) / 16).astype(np.uint8)
    if gain != 1.0:
        out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def _empty_coords():
    """Return a canonical empty ``(N, 2)`` integer coordinate array."""
    return np.empty((0, 2), dtype=np.int32)


def _is_centered_y(points):
    """Heuristically detect whether a point set uses centered ``y`` values."""
    if points is None:
        return False
    arr = np.asarray(points)
    if arr.size == 0:
        return False
    y = arr[0:1] if arr.ndim == 1 else arr[:, 0]
    return np.nanmin(y) < 0


def _to_yx(points, H, W, *, assume_xy=False, centered_y=False):
    """Normalize point arrays to clipped integer ``(y, x)`` coordinates.

    Supported conventions
    ---------------------
    - ``(y, x)``: historical internal format used by some boundary functions
    - ``(x, y_centered)``: format produced by ``load_boundary_file`` helpers
    """
    if points is None:
        return _empty_coords()

    arr = np.asarray(points)
    if arr.size == 0:
        return _empty_coords()

    if arr.ndim == 1:
        if arr.shape[0] < 2:
            return _empty_coords()
        arr = arr.reshape(1, -1)

    arr = np.rint(arr[:, :2]).astype(np.int32)
    if assume_xy:
        x = arr[:, 0]
        y = arr[:, 1]
    else:
        y = arr[:, 0]
        x = arr[:, 1]

    if centered_y:
        y = y + H // 2

    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    return np.column_stack([y, x]).astype(np.int32)


def _get_frame_boundaries(boundaries, idx, H, W):
    """Return ``(top, bottom)`` boundary coordinates in canonical ``(y, x)`` form."""
    if boundaries is None or idx >= len(boundaries):
        return _empty_coords(), _empty_coords()

    item = boundaries[idx]
    if item is None:
        return _empty_coords(), _empty_coords()

    if isinstance(item, (list, tuple)) and len(item) == 2:
        centered = _is_centered_y(item[0]) or _is_centered_y(item[1])
        return (
            _to_yx(item[0], H, W, assume_xy=False, centered_y=centered),
            _to_yx(item[1], H, W, assume_xy=False, centered_y=centered),
        )

    return _to_yx(item, H, W, assume_xy=True, centered_y=True), _empty_coords()


def _boundary_masks(boundaries_for_rep, frame_idx, H, W, kernel=None):
    """Rasterize top/bottom boundary point sets into binary masks."""
    mask_top = np.zeros((H, W), dtype=np.uint8)
    mask_bot = np.zeros((H, W), dtype=np.uint8)

    coords_top, coords_bot = _get_frame_boundaries(boundaries_for_rep, frame_idx, H, W)
    if coords_top.size:
        ys, xs = coords_top[:, 0], coords_top[:, 1]
        inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        mask_top[ys[inb], xs[inb]] = 255
    if coords_bot.size:
        ys, xs = coords_bot[:, 0], coords_bot[:, 1]
        inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        mask_bot[ys[inb], xs[inb]] = 255

    if kernel is not None:
        if mask_top.any():
            mask_top = cv2.dilate(mask_top, kernel, iterations=1)
        if mask_bot.any():
            mask_bot = cv2.dilate(mask_bot, kernel, iterations=1)

    return mask_top, mask_bot


def _overlay_boundary_masks(frame_bgr, mask_top, mask_bot, *, color_top, color_bottom, alpha):
    """Overlay colored boundary masks onto a BGR frame."""
    overlay = frame_bgr.copy()
    if mask_top.any():
        overlay[mask_top > 0] = color_top
    if mask_bot.any():
        overlay[mask_bot > 0] = color_bottom

    if alpha >= 1.0:
        mask_any = (mask_top > 0) | (mask_bot > 0)
        frame_bgr[mask_any] = overlay[mask_any]
        return frame_bgr

    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)


def play_video_cv2(video, gain=1, binarize=False, thresh=0.5, intv=17):
    """Play one grayscale video with OpenCV."""
    total_frames = len(video)
    if total_frames == 0:
        return

    for i in range(total_frames):
        frame_uint8 = _frame_to_uint8(video[i], gain=gain, binarize=binarize, thresh=thresh)
        cv2.imshow("Frame", frame_uint8)
        if cv2.waitKey(intv) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def play_videos_side_by_side(videos, gain=1, binarize=False, thresh=0.5, intv=17):
    """Play multiple videos side by side using OpenCV."""
    if not videos:
        return

    total_frames = min(len(v) for v in videos)
    if total_frames == 0:
        return

    for i in range(total_frames):
        frame = np.hstack([v[i] for v in videos])
        frame_uint8 = _frame_to_uint8(frame, gain=gain, binarize=binarize, thresh=thresh)
        cv2.imshow("Frame", frame_uint8)
        if cv2.waitKey(intv) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def play_video_with_boundaries_cv2(
    video,
    boundaries_for_rep,
    gain=1.0,
    binarize=False,
    thresh=0.5,
    intv=17,
    color_top=(0, 0, 255),
    color_bottom=None,
    thickness=1,
    alpha=1.0,
):
    """Overlay precomputed boundary points on each frame and play with OpenCV.

    Supported boundary formats
    --------------------------
    1. ``(top, bottom)`` where each item is ``(N, 2)`` in ``(y, x)``
       or centered-``y`` form.
    2. One ``(N, 2)`` array in ``(x, y_centered)`` format, typically loaded
       from saved boundary files.
    """
    if color_bottom is None:
        color_bottom = color_top

    video = np.asarray(video)
    F = len(video)
    if F == 0:
        return

    kernel = None
    ksz = max(1, 2 * int(thickness) + 1)
    if ksz > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))

    for f in range(F):
        frame = np.asarray(video[f])
        H, W = frame.shape[-2], frame.shape[-1]

        frame_u8 = _frame_to_uint8(frame, gain=gain, binarize=binarize, thresh=thresh)
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)

        mask_top, mask_bot = _boundary_masks(boundaries_for_rep, f, H, W, kernel=kernel)
        frame_bgr = _overlay_boundary_masks(
            frame_bgr,
            mask_top,
            mask_bot,
            color_top=color_top,
            color_bottom=color_bottom,
            alpha=alpha,
        )

        cv2.imshow("Frame + Boundary", frame_bgr)
        if cv2.waitKey(intv) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def save_video_with_boundaries_cv2(
    video,
    boundaries_for_rep,
    gain=1.0,
    binarize=False,
    thresh=0.5,
    fps=10,
    save_path=None,
    color_top=(0, 0, 255),
    color_bottom=None,
    thickness=1,
    alpha=1.0,
):
    """Overlay precomputed boundary points on each frame and save as AVI.

    This function shares the exact same boundary-interpretation code as
    :func:`play_video_with_boundaries_cv2`, so visual inspection and saved video
    export now follow one consistent code path.
    """
    if color_bottom is None:
        color_bottom = color_top

    video = np.asarray(video)
    F = len(video)
    if F == 0:
        raise ValueError("`video` is empty.")

    first = np.asarray(video[0])
    if first.ndim < 2:
        raise ValueError("Each frame must be at least 2D (H, W).")

    H, W = first.shape[-2], first.shape[-1]

    if save_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(os.getcwd(), f"boundary_overlay_{stamp}.avi")
    else:
        save_path = os.path.abspath(save_path)
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(save_path, fourcc, float(fps), (int(W), int(H)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {save_path}")

    kernel = None
    ksz = max(1, 2 * int(thickness) + 1)
    if ksz > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))

    try:
        for f in range(F):
            frame = np.asarray(video[f])
            if frame.shape[-2] != H or frame.shape[-1] != W:
                raise ValueError(
                    f"Frame {f} size mismatch: expected ({H}, {W}), got ({frame.shape[-2]}, {frame.shape[-1]})"
                )

            frame_u8 = _frame_to_uint8(frame, gain=gain, binarize=binarize, thresh=thresh)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)

            mask_top, mask_bot = _boundary_masks(boundaries_for_rep, f, H, W, kernel=kernel)
            frame_bgr = _overlay_boundary_masks(
                frame_bgr,
                mask_top,
                mask_bot,
                color_top=color_top,
                color_bottom=color_bottom,
                alpha=alpha,
            )
            writer.write(frame_bgr)
    finally:
        writer.release()

    return save_path


def play_segments_with_boundaries(
    segments,
    boundaries,
    p=0,
    gain=1.0,
    intv=17,
    cmap="gray",
    origin="upper",
    points_format="rc",
):
    """Preview one plume's segment video with boundary points using Matplotlib."""

    def _ensure_list_of_arrays(boundary_frame):
        if boundary_frame is None:
            return []
        if isinstance(boundary_frame, (list, tuple)):
            return [np.asarray(a) for a in boundary_frame]
        return [np.asarray(boundary_frame)]

    def _to_xy(arr):
        if arr is None or arr.size == 0:
            return np.empty((0, 2))
        a = np.asarray(arr)
        if points_format.lower() in ("rc", "yx"):
            return np.c_[a[:, 1], a[:, 0]]
        return a[:, :2]

    vid = segments[p]
    F = len(vid)

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(
        vid[0] * gain,
        cmap=cmap,
        origin=origin,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_axis_off()

    boundary_groups = _ensure_list_of_arrays(boundaries[p][0])
    scatters = []
    for arr in boundary_groups:
        xy = _to_xy(arr)
        scatters.append(ax.scatter(xy[:, 0], xy[:, 1], s=6))

    try:
        for f in range(F):
            im.set_data(vid[f] * gain)
            boundary_groups = _ensure_list_of_arrays(boundaries[p][f])

            if len(boundary_groups) != len(scatters):
                for scatter in scatters:
                    scatter.remove()
                scatters = []
                for arr in boundary_groups:
                    xy = _to_xy(arr)
                    scatters.append(ax.scatter(xy[:, 0], xy[:, 1], s=6))
            else:
                for scatter, arr in zip(scatters, boundary_groups):
                    scatter.set_offsets(_to_xy(arr))

            fig.canvas.flush_events()
            plt.pause(intv / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()
