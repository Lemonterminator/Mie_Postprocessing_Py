import cv2 
import numpy as np
import matplotlib.pyplot as plt

def play_video_cv2(video, gain=1, binarize=False, thresh=0.5, intv=17):
    """
    Play a list/array of video frames with OpenCV, with optional binarization.

    Parameters
    ----------
    video : sequence of np.ndarray. Int, float or bool
        视频帧列表，每帧可以是整数、浮点数，也可以是布尔数组。
    gain : float, optional. 
        灰度增益，对原始数值做线性放缩（默认 1）。
    binarize : bool, optional
        是否先将帧转换为布尔再显示（默认 False）。
    thresh : float, optional
        当 binarize=True 且输入不是布尔类型时，使用该阈值做二值化（浮点[0,1]或任意范围均可）。
    """
    total_frames = len(video)
    if total_frames == 0:
        return

    # 先检测第 1 帧的数据类型
    # first_dtype = video[0].dtype

    for i in range(total_frames):
        frame = video[i]

        # —— 二值化分支 ——
        if binarize:
            # 如果是非布尔类型，先做阈值处理
            if frame.dtype != bool:
                # 假定浮点帧在 [0,1]，或任意数值，都可以用 thresh 来分割
                frame_bool = frame > thresh
            else:
                frame_bool = frame
            # True→255, False→0
            frame_uint8 = (frame_bool.astype(np.uint8)) * 255

        # —— 原有灰度／色阶分支 ——
        else:
            dtype = frame.dtype
            # 整数：假设是 16-bit 量程，缩到 8-bit
            if np.issubdtype(dtype, np.integer):
                frame_uint8 = gain * (frame / 16).astype(np.uint8)
            # 浮点：假设在 [0,1]，放大到 0–255
            elif np.issubdtype(dtype, np.floating):
                frame_uint8 = np.clip(gain * (frame * 255), 0, 255).astype(np.uint8)
            # 其他类型回退到整数缩放
            else:
                frame_uint8 = gain * (frame / 16).astype(np.uint8)

        # 显示
        cv2.imshow('Frame', frame_uint8)
        # ~60fps 播放，按 'q' 退出
        if cv2.waitKey(intv) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def play_videos_side_by_side(videos, gain=1, binarize=False, thresh=0.5, intv=17):
    """Play multiple videos side by side using OpenCV.

    Parameters
    ----------
    videos : sequence of np.ndarray
        Sequence of videos, each shaped ``(frame, x, y)``.
    gain, binarize, thresh, intv : see :func:`play_video_cv2`.
    """
    if not videos:
        return

    total_frames = min(len(v) for v in videos)
    if total_frames == 0:
        return

    for i in range(total_frames):
        frame = np.hstack([v[i] for v in videos])

        if binarize:
            if frame.dtype != bool:
                frame_bool = frame > thresh
            else:
                frame_bool = frame
            frame_uint8 = frame_bool.astype(np.uint8) * 255
        else:
            dtype = frame.dtype
            if np.issubdtype(dtype, np.integer):
                frame_uint8 = gain * (frame / 16).astype(np.uint8)
            elif np.issubdtype(dtype, np.floating):
                frame_uint8 = np.clip(gain * (frame * 255), 0, 255).astype(np.uint8)
            else:
                frame_uint8 = gain * (frame / 16).astype(np.uint8)

        cv2.imshow('Frame', frame_uint8)
        if cv2.waitKey(intv) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def _frame_to_uint8(frame, gain=1.0, binarize=False, thresh=0.5):
    """Match your play_video_cv2 conversion."""
    if binarize:
        if frame.dtype != bool:
            frame_bool = frame > thresh
        else:
            frame_bool = frame
        return (frame_bool.astype(np.uint8)) * 255

    dtype = frame.dtype
    if np.issubdtype(dtype, np.integer):
        # assume 16-bit-ish scale; shrink to 8-bit then apply gain
        out = (frame / 16).astype(np.uint8)
        if gain != 1.0:
            out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return out
    elif np.issubdtype(dtype, np.floating):
        return np.clip(gain * (frame * 255.0), 0, 255).astype(np.uint8)
    else:
        out = (frame / 16).astype(np.uint8)
        if gain != 1.0:
            out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return out


def play_video_with_boundaries_cv2(
    video,                          # array-like, shape (F, H, W)
    boundaries_for_rep,             # list/seq length F of (coords_top, coords_bottom), each (N,2) (y,x)
    gain=1.0,
    binarize=False,
    thresh=0.5,
    intv=17,
    color_top=(0, 0, 255),          # BGR red
    color_bottom=None,              # defaults to same as color_top
    thickness=1,                    # 0 -> 1px; otherwise dilate radius in pixels
    alpha=1.0                       # 1.0 -> hard paint; <1.0 -> blend
):
    """
    Overlay precomputed boundary points on each frame and play with OpenCV.

    - `boundaries_for_rep[f] = (coords_top, coords_bottom)`
    - Each coords_* is int array of shape (N,2) with (y,x).

    If `color_bottom` is None, uses `color_top`.
    `thickness` uses dilation on the mask: kernel size = 2*thickness + 1.
    """
    if color_bottom is None:
        color_bottom = color_top

    video = np.asarray(video)
    F = len(video)
    if F == 0:
        return

    # Prebuild dilation kernel (optional)
    ksz = max(1, 2 * int(thickness) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz)) if ksz > 1 else None

    for f in range(F):
        frame = video[f]
        H, W = frame.shape[-2], frame.shape[-1]

        # Convert to uint8 grayscale
        frame_u8 = _frame_to_uint8(frame, gain=gain, binarize=binarize, thresh=thresh)

        # Make BGR for color overlay
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)

        # Build masks from coords
        mask_top = np.zeros((H, W), dtype=np.uint8)
        mask_bot = np.zeros((H, W), dtype=np.uint8)

        coords_top, coords_bot = boundaries_for_rep[f]
        # Guard empty
        if coords_top.size:
            ys, xs = coords_top[:, 0], coords_top[:, 1]
            # clip just in case
            inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            mask_top[ys[inb], xs[inb]] = 255
        if coords_bot.size:
            ys, xs = coords_bot[:, 0], coords_bot[:, 1]
            inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            mask_bot[ys[inb], xs[inb]] = 255

        # Thicken if requested
        if kernel is not None:
            if mask_top.any():
                mask_top = cv2.dilate(mask_top, kernel, iterations=1)
            if mask_bot.any():
                mask_bot = cv2.dilate(mask_bot, kernel, iterations=1)

        # Compose a single color overlay
        overlay = frame_bgr.copy()
        if mask_top.any():
            overlay[mask_top > 0] = color_top
        if mask_bot.any():
            overlay[mask_bot > 0] = color_bottom

        if alpha >= 1.0:
            # Hard paint only where masks are set
            m = (mask_top > 0) | (mask_bot > 0)
            frame_bgr[m] = overlay[m]
        else:
            # Blend entire frame; cheaper is to blend once globally
            frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)

        cv2.imshow('Frame + Boundary', frame_bgr)
        if cv2.waitKey(intv) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def play_segments_with_boundaries(segments, boundaries, p=0, gain=1.0, intv=17,
                                  cmap='gray', origin='upper', points_format='rc'):
    """
    points_format: 'xy' if points are (x,y); 'rc' (or 'yx') if points are (row,col).
    """
    def _ensure_list_of_arrays(bf):
        if bf is None:
            return []
        if isinstance(bf, (list, tuple)):
            return [np.asarray(a) for a in bf]
        return [np.asarray(bf)]

    def _to_xy(arr):
        # Map boundary array to (x,y)
        if arr is None or arr.size == 0:
            return np.empty((0, 2))
        a = np.asarray(arr)
        if points_format.lower() in ('rc', 'yx'):
            # (row, col) -> (x=col, y=row)
            return np.c_[a[:, 1], a[:, 0]]
        else:
            # already (x, y)
            return a[:, :2]

    vid = segments[p]
    F = len(vid)

    plt.ion()
    fig, ax = plt.subplots(); vmin, vmax = 0, 1
    im = ax.imshow(vid[0] * gain, cmap=cmap, origin=origin, interpolation='nearest', vmin=vmin, vmax=vmax)
    
    ax.set_axis_off()

    # init scatters from frame 0
    b0 = _ensure_list_of_arrays(boundaries[p][0])
    scatters = []
    for arr in b0:
        xy = _to_xy(arr)
        sct = ax.scatter(xy[:, 0], xy[:, 1], s=6)
        scatters.append(sct)

    try:
        for f in range(F):
            im.set_data(vid[f] * gain)

            bf = _ensure_list_of_arrays(boundaries[p][f])
            # adapt number of groups if needed
            if len(bf) != len(scatters):
                for s in scatters:
                    s.remove()
                scatters = []
                for arr in bf:
                    xy = _to_xy(arr)
                    scatters.append(ax.scatter(xy[:, 0], xy[:, 1], s=6))
            else:
                for sct, arr in zip(scatters, bf):
                    xy = _to_xy(arr)
                    sct.set_offsets(xy)

            fig.canvas.flush_events()
            plt.pause(intv / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()
