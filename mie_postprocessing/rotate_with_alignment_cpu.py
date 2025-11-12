import numpy as np
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt

"""
This module provides functions to rotate video frames efficiently on the GPU using numpy.
First find the nozzle center and the angle offset by the GUI. 
"""

def build_affine_inverse_maps_numpy(
    M,
    out_shape,
    *,
    out_origin=None,
    center_on=None,
):
    """
    Build inverse maps for cv2.remap-style sampling on numpy.

    Parameters
    ----------
    M : array-like shape (2, 3)
        Forward affine (OpenCV convention): `[x_out; y_out] = A [x_in; y_in] + b`.
    out_shape : tuple[int, int]
        `(H_out, W_out)` for the ROI you want to generate.
    out_origin : tuple[float, float], optional
        `(x_off, y_off)` for the top-left pixel of the ROI expressed in the *full*
        output coordinate system. Use this to keep, e.g., the left edge fixed while
        sliding vertically.
    center_on : tuple[float, float], optional
        Convenience alternative to `out_origin`. If provided, the ROI is centered on
        `(u_cal, v_cal)` from the full output coords. Useful when you know the target
        feature you want on the small output's center.

    Returns
    -------
    (mapx, mapy) : tuple[np.ndarray, np.ndarray]
        numpy float32 arrays of shape `(H_out, W_out)` with source sample coordinates.
    """
    H_out, W_out = out_shape

    A = np.asarray(M, dtype=np.float64)
    A2 = A[:, :2]
    b = A[:, 2]
    Ainv = np.linalg.inv(A2)
    binv = -Ainv @ b

    Minv = np.empty((2, 3), dtype=np.float64)
    Minv[:, :2] = Ainv
    Minv[:, 2] = binv
    Minv = np.asarray(Minv, dtype=np.float32)

    if out_origin is not None:
        x_off, y_off = map(float, out_origin)
    elif center_on is not None:
        u_cal, v_cal = map(float, center_on)
        x_off = u_cal - (W_out - 1) / 2.0
        y_off = v_cal - (H_out - 1) / 2.0
    else:
        x_off = 0.0
        y_off = 0.0

    xs = np.arange(W_out, dtype=np.float32) + np.float32(x_off)
    ys = np.arange(H_out, dtype=np.float32) + np.float32(y_off)
    X, Y = np.meshgrid(xs, ys)

    x_src = Minv[0, 0] * X + Minv[0, 1] * Y + Minv[0, 2]
    y_src = Minv[1, 0] * X + Minv[1, 1] * Y + Minv[1, 2]

    return x_src.astype(np.float32), y_src.astype(np.float32)

def build_rotation_affine(center, angle_deg, *, target=None, scale=1.0):
    """
    Construct a 2x3 forward affine that rotates about `center` by `angle_deg`
    and places that center at `target` in the output coordinate system.
    """
    center = np.asarray(center, dtype=np.float64)
    if target is None:
        target = center
    target = np.asarray(target, dtype=np.float64)

    theta = np.deg2rad(angle_deg)
    alpha = scale * np.cos(theta)
    beta = scale * np.sin(theta)
    A = np.array([[alpha, beta], [-beta, alpha]], dtype=np.float64)
    b = target - A @ center

    return np.hstack([A, b[:, None]])

def build_nozzle_rotation_maps(
    frame_shape,
    nozzle_center,
    angle_deg,
    *,
    out_shape=None,
    calibration_point=None,
    keep_left_edge=True,
):
    """
    Pre-compute affine inverse maps for the nozzle alignment workflow.

    Parameters
    ----------
    frame_shape : tuple[int, int]
        `(H_in, W_in)` of the source frames.
    nozzle_center : tuple[float, float]
        Pixel coordinate of the nozzle in the input frame (x, y).
    angle_deg : float
        Rotation angle (positive rotates CCW).
    out_shape : tuple[int, int], optional
        Output ROI shape `(H_out, W_out)`. Defaults to `frame_shape`.
    calibration_point : tuple[float, float], optional
        Desired location of the nozzle in the *full* rotated canvas. Defaults to
        `(0, H_full/2)` to align with the left edge and vertical center.
    keep_left_edge : bool
        When True, forces `x_off=0` so the ROI remains flush with the left edge.
    """
    H_full, W_full = frame_shape
    if out_shape is None:
        out_shape = (H_full, W_full)

    if calibration_point is None:
        calibration_point = (0.0, H_full / 2.0)

    target = calibration_point
    M = build_rotation_affine(nozzle_center, angle_deg, target=target)

    if keep_left_edge:
        x_off = 0.0
    else:
        x_off = target[0] - (out_shape[1] - 1) / 2.0
    y_off = target[1] - (out_shape[0] - 1) / 2.0

    return build_affine_inverse_maps_numpy(
        M,
        out_shape,
        out_origin=(x_off, y_off),
    )

def _reflect_indices(idx, size):
    # OpenCV-style BORDER_REFLECT_101 (no double-border), but here we’ll use simple reflect.
    # For large out-of-range, bring back via modulo and reflection.
    # If you want exact OpenCV behavior, adjust accordingly.
    if size == 1:
        return np.zeros_like(idx)
    mod = np.mod(idx, 2*(size-1))
    return np.where(mod < size, mod, 2*(size-1) - mod)

def _clamp_indices(idx, size):
    return np.clip(idx, 0, size-1)

def _in_bounds(ix, iy, W, H):
    return (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

def _cast_back(out, orig_dtype):
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(orig_dtype)
    else:
        out = out.astype(orig_dtype)
    return out

def remap_frame_nearest_neighbour_numpy(img, mapx, mapy, border_mode="constant", cval=0.0):
    """
    Nearest-neighbour remap on GPU using numpy.
    """
    assert img.ndim in (2, 3)
    H_in, W_in = img.shape[:2]

    orig_dtype = img.dtype
    img_f = img.astype(np.float32, copy=False)

    xi = np.rint(mapx).astype(np.int32)
    yi = np.rint(mapy).astype(np.int32)

    if border_mode == "replicate":
        xi_c = _clamp_indices(xi, W_in)
        yi_c = _clamp_indices(yi, H_in)
        if img_f.ndim == 2:
            sampled = img_f[yi_c, xi_c]
        else:
            sampled = img_f[yi_c, xi_c, :]
    elif border_mode == "reflect":
        xi_r = _reflect_indices(xi, W_in)
        yi_r = _reflect_indices(yi, H_in)
        if img_f.ndim == 2:
            sampled = img_f[yi_r, xi_r]
        else:
            sampled = img_f[yi_r, xi_r, :]
    elif border_mode == "constant":
        mask = (xi >= 0) & (xi < W_in) & (yi >= 0) & (yi < H_in)
        xi_c = _clamp_indices(xi, W_in)
        yi_c = _clamp_indices(yi, H_in)
        fill = np.float32(cval)
        if img_f.ndim == 2:
            gathered = img_f[yi_c, xi_c]
            sampled = np.where(mask, gathered, fill)
        else:
            gathered = img_f[yi_c, xi_c, :]
            sampled = np.where(mask[..., None], gathered, fill)
    else:
        raise ValueError("border_mode must be 'constant', 'replicate', or 'reflect'")

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        sampled = np.clip(np.rint(sampled), info.min, info.max).astype(orig_dtype)
    elif np.issubdtype(orig_dtype, np.floating):
        sampled = sampled.astype(orig_dtype)
    else:
        sampled = sampled.astype(orig_dtype)
    return sampled

def remap_frame_bilinear_numpy(img, mapx, mapy, border_mode="constant", cval=0.0):
    """
    Bilinear remap on GPU using numpy.

    Parameters
    ----------
    img : np.ndarray
        Input image on GPU. Shape (H, W) for grayscale or (H, W, C) for multi-channel.
        Any numeric dtype is accepted. Output preserves dtype.
    mapx, mapy : np.ndarray (float32), shape (H_out, W_out)
        For each output pixel (v,u), (mapx[v,u], mapy[v,u]) gives the *source* (x,y)
        coordinate in the input image (i.e., inverse mapping).
    border_mode : {"constant","replicate","reflect"}
        How to sample when the source coordinate is out of bounds.
        - "constant": use cval wherever a neighbor falls OOB.
        - "replicate": clamp to the closest valid border pixel.
        - "reflect": reflect indices back into range (definition depends on your _reflect_indices).
    cval : float
        Constant fill value for border_mode == "constant".

    Returns
    -------
    np.ndarray
        Remapped image on GPU. Shape is (H_out, W_out[, C]) with same dtype as input.
    """
    assert img.ndim in (2,3)
    H_in, W_in = img.shape[:2]
    H_out, W_out = mapx.shape

    # Keep original dtype for final cast; compute interpolation in float32 for speed/consistency.
    orig_dtype = img.dtype
    img_f = img.astype(np.float32, copy=False)

    # Integer pixel neighbors around (x,y): floor and +1 for right/bottom neighbors.
    x0 = np.floor(mapx).astype(np.int32)
    y0 = np.floor(mapy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts -> bilinear weights along x and y, then the four corner weights.
    wx = (mapx - x0).astype(np.float32)
    wy = (mapy - y0).astype(np.float32)
    w00 = (1 - wx) * (1 - wy)  # top-left
    w10 = (     wx) * (1 - wy)  # top-right
    w01 = (1 - wx) * (     wy)  # bottom-left
    w11 = (     wx) * (     wy)  # bottom-right

    if border_mode == "replicate":
        # Clamp each neighbor index to valid range [0, W_in-1] or [0, H_in-1].
        x0c = _clamp_indices(x0, W_in); x1c = _clamp_indices(x1, W_in)
        y0c = _clamp_indices(y0, H_in); y1c = _clamp_indices(y1, H_in)

        # Gather the four neighbor values directly on GPU.
        # NOTE: avoid .get() here; it would pull arrays to CPU.
        def sample(ix, iy):
            return img_f[iy, ix] if img_f.ndim == 2 else img_f[iy, ix, :]

        v00 = sample(x0c, y0c)
        v10 = sample(x1c, y0c)
        v01 = sample(x0c, y1c)
        v11 = sample(x1c, y1c)

    elif border_mode == "reflect":
        # Reflect indices back into valid range (definition should match your chosen convention).
        x0r = _reflect_indices(x0, W_in); x1r = _reflect_indices(x1, W_in)
        y0r = _reflect_indices(y0, H_in); y1r = _reflect_indices(y1, H_in)

        def sample(ix, iy):
            return img_f[iy, ix] if img_f.ndim == 2 else img_f[iy, ix, :]

        v00 = sample(x0r, y0r)
        v10 = sample(x1r, y0r)
        v01 = sample(x0r, y1r)
        v11 = sample(x1r, y1r)

    elif border_mode == "constant":
        # Build in-bounds masks for each neighbor position.
        def inb(ix, iy):
            return (ix >= 0) & (ix < W_in) & (iy >= 0) & (iy < H_in)

        m00 = inb(x0, y0)
        m10 = inb(x1, y0)
        m01 = inb(x0, y1)
        m11 = inb(x1, y1)

        # Clamp indices for the actual GPU gather; masks will zero/select cval later.
        x0c = _clamp_indices(x0, W_in); x1c = _clamp_indices(x1, W_in)
        y0c = _clamp_indices(y0, H_in); y1c = _clamp_indices(y1, H_in)

        # Masked gather that returns cval where mask==False.
        def sample_masked(img_f, ix, iy, mask, cval):
            img_f = np.asarray(img_f)
            ix    = np.asarray(ix).astype(np.int64, copy=False)
            iy    = np.asarray(iy).astype(np.int64, copy=False)
            mask  = np.asarray(mask, dtype=np.bool_)
            cvald = np.asarray(cval, dtype=np.float32)

            if img_f.ndim == 2:  # (H, W)
                s = img_f[iy, ix].astype(np.float32, copy=False)
                return np.where(mask, s, cvald)
            else:                # (H, W, C)
                s = img_f[iy, ix, :].astype(np.float32, copy=False)
                return np.where(mask[..., None], s, cvald)

        v00 = sample_masked(img_f, x0c, y0c, m00, cval)
        v10 = sample_masked(img_f, x1c, y0c, m10, cval)
        v01 = sample_masked(img_f, x0c, y1c, m01, cval)
        v11 = sample_masked(img_f, x1c, y1c, m11, cval)

    else:
        raise ValueError("border_mode must be 'constant', 'replicate', or 'reflect'")

    # Helper: ensure numpy float32 (no CPU bounce).
    def to_np_f32(x):
        return np.asarray(x, dtype=np.float32)

    v00, v10, v01, v11 = map(to_np_f32, (v00, v10, v01, v11))
    w00, w10, w01, w11 = map(to_np_f32, (w00, w10, w01, w11))
    img_f = np.asarray(img_f)  # ensure resident on device

    # Bilinear blend with correct broadcasting for channels.
    if img_f.ndim == 3:  # (H, W, C)
        out = (w00[..., None] * v00 +
               w10[..., None] * v10 +
               w01[..., None] * v01 +
               w11[..., None] * v11)
    else:  # (H, W)
        out = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11

    # Cast back to the original dtype with appropriate rounding/clipping for integers.
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(orig_dtype)
    elif np.issubdtype(orig_dtype, np.floating):
        out = out.astype(orig_dtype)
    else:
        out = out.astype(orig_dtype)
    return out

def _cubic_keys_weights(t, a=-0.5):
    """
    Compute Keys cubic weights for distances t in [0,1) to the 4 taps:
    at offsets [-1, 0, 1, 2] relative to floor(x).
    Returns weights w of shape t.shape + (4,)
    """
    # distances to taps
    t0 = 1 + t        # x - (x0-1) => 1 + frac
    t1 = t            # x - x0     => frac
    t2 = 1 - t        # (x0+1) - x
    t3 = 2 - t        # (x0+2) - x

    def P(x):
        ax = np.abs(x)
        ax2 = ax * ax
        ax3 = ax2 * ax
        w = np.where(
            ax < 1,
            (a + 2) * ax3 - (a + 3) * ax2 + 1,
            np.where(
                (ax >= 1) & (ax < 2),
                a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a,
                np.zeros_like(ax)
            )
        )
        return w.astype(np.float32)

    w0 = P(t0)
    w1 = P(t1)
    w2 = P(t2)
    w3 = P(t3)
    # stack last axis -> (..., 4)
    return np.stack([w0, w1, w2, w3], axis=-1)

def remap_frame_bicubic_numpy(img, mapx, mapy, border_mode="constant", cval=0.0):
    """
    Bicubic remap (Keys cubic, a = -0.5 like OpenCV INTER_CUBIC).
    img: np.ndarray (H,W) or (H,W,C)
    mapx, mapy: np.float32 (H_out,W_out) source coords
    border_mode: "constant" | "replicate" | "reflect"
    cval: float for constant border
    """
    assert img.ndim in (2, 3)
    H, W = img.shape[:2]
    H_out, W_out = mapx.shape

    orig_dtype = img.dtype
    img_f = img.astype(np.float32, copy=False)

    # integer base indices
    x0 = np.floor(mapx).astype(np.int32)
    y0 = np.floor(mapy).astype(np.int32)

    # fractional distances in [0,1)
    fx = (mapx - x0).astype(np.float32)
    fy = (mapy - y0).astype(np.float32)

    # 4 taps per axis: offsets -1,0,1,2
    x_idx = np.stack([x0 - 1, x0, x0 + 1, x0 + 2], axis=-1).astype(np.int32)
    y_idx = np.stack([y0 - 1, y0, y0 + 1, y0 + 2], axis=-1).astype(np.int32)

    # weights per axis (H_out,W_out,4)
    wx = _cubic_keys_weights(fx, a=-0.5)
    wy = _cubic_keys_weights(fy, a=-0.5)

    # border handling: get valid indices + masks for constant
    if border_mode == "replicate":
        x_is = _clamp_indices(x_idx, W)
        y_is = _clamp_indices(y_idx, H)
        masks = None
    elif border_mode == "reflect":
        x_is = _reflect_indices(x_idx, W)
        y_is = _reflect_indices(y_idx, H)
        masks = None
    elif border_mode == "constant":
        x_is = _clamp_indices(x_idx, W)
        y_is = _clamp_indices(y_idx, H)
        # neighbor masks (H_out,W_out,4,4) per (jy, ix)
        # build lazily in accumulation loop to save memory
        masks = "constant"
        cval_f = np.asarray(cval, dtype=np.float32)
    else:
        raise ValueError("border_mode must be 'constant', 'replicate', or 'reflect'")

    # accumulate
    if img_f.ndim == 2:
        out = np.zeros((H_out, W_out), dtype=np.float32)
    else:
        C = img_f.shape[2]
        out = np.zeros((H_out, W_out, C), dtype=np.float32)

    # 4x4 taps: outer product of wy[:, :, jy] and wx[:, :, ix]
    for jy in range(4):
        iy = y_is[..., jy]
        for ix in range(4):
            ix_arr = x_is[..., ix]
            # sample
            if img_f.ndim == 2:
                samp = img_f[iy, ix_arr]  # (H_out,W_out)
            else:
                samp = img_f[iy, ix_arr, :]  # (H_out,W_out,C)

            if masks == "constant":
                m = _in_bounds(x_idx[..., ix], y_idx[..., jy], W, H)
                if img_f.ndim == 3:
                    m = m[..., None]
                samp = np.where(m, samp, cval_f)

            w = (wy[..., jy] * wx[..., ix]).astype(np.float32)
            if img_f.ndim == 3:
                w = w[..., None]
            out += w * samp

    return _cast_back(out, orig_dtype)

def _lanczos3_weights(t, a=3):
    """
    Lanczos-a (default a=3) weights for distances t in [0,1)
    to taps at offsets [-a+1, ..., a] relative to floor(x).
    For a=3: offsets [-2,-1,0,1,2,3] (6 taps).
    Returns weights w of shape t.shape + (2a,)
    """
    # tap offsets relative to x0
    offs = np.arange(-a + 1, a + 1, dtype=np.int32)  # [-2,-1,0,1,2,3] when a=3
    # shape broadcast to (H_out,W_out,2a)
    t_grid = t[..., None] - offs[None, None, :]
    # sinc in NumPy/numpy is normalized: sinc(x) = sin(pi x)/(pi x)
    # handle t=0 OK.
    # zero outside |x|<a
    w = np.sinc(t_grid) * np.sinc(t_grid / a)
    w = np.where(np.abs(t_grid) < a, w, 0.0)
    # normalize rows slightly to mitigate tiny numeric drift (optional)
    s = np.sum(w, axis=-1, keepdims=True)
    s = np.where(s != 0, s, 1.0)
    w = (w / s).astype(np.float32)
    return w, offs

def remap_frame_lanczos_3_numpy(img, mapx, mapy, border_mode="constant", cval=0.0):
    """
    Lanczos-3 remap on GPU (windowed sinc, support=3).
    img: np.ndarray (H,W) or (H,W,C)
    mapx, mapy: np.float32 (H_out,W_out) source coords
    border_mode: "constant" | "replicate" | "reflect"
    cval: float for constant border
    """
    assert img.ndim in (2, 3)
    H, W = img.shape[:2]
    H_out, W_out = mapx.shape

    orig_dtype = img.dtype
    img_f = img.astype(np.float32, copy=False)

    x0 = np.floor(mapx).astype(np.int32)
    y0 = np.floor(mapy).astype(np.int32)
    fx = (mapx - x0).astype(np.float32)
    fy = (mapy - y0).astype(np.float32)

    wx, xoffs = _lanczos3_weights(fx, a=3)  # (..., 6)
    wy, yoffs = _lanczos3_weights(fy, a=3)

    x_idx = (x0[..., None] + xoffs[None, None, :]).astype(np.int32)  # (H_out,W_out,6)
    y_idx = (y0[..., None] + yoffs[None, None, :]).astype(np.int32)  # (H_out,W_out,6)

    if border_mode == "replicate":
        x_is = _clamp_indices(x_idx, W)
        y_is = _clamp_indices(y_idx, H)
        masks = None
    elif border_mode == "reflect":
        x_is = _reflect_indices(x_idx, W)
        y_is = _reflect_indices(y_idx, H)
        masks = None
    elif border_mode == "constant":
        x_is = _clamp_indices(x_idx, W)
        y_is = _clamp_indices(y_idx, H)
        masks = "constant"
        cval_f = np.asarray(cval, dtype=np.float32)
    else:
        raise ValueError("border_mode must be 'constant', 'replicate', or 'reflect'")

    if img_f.ndim == 2:
        out = np.zeros((H_out, W_out), dtype=np.float32)
    else:
        C = img_f.shape[2]
        out = np.zeros((H_out, W_out, C), dtype=np.float32)

    # 6x6 accumulation
    for jy in range(6):
        iy = y_is[..., jy]
        for ix in range(6):
            ix_arr = x_is[..., ix]
            if img_f.ndim == 2:
                samp = img_f[iy, ix_arr]
            else:
                samp = img_f[iy, ix_arr, :]
            if masks == "constant":
                m = _in_bounds(x_idx[..., ix], y_idx[..., jy], W, H)
                if img_f.ndim == 3:
                    m = m[..., None]
                samp = np.where(m, samp, cval_f)

            w = (wy[..., jy] * wx[..., ix]).astype(np.float32)
            if img_f.ndim == 3:
                w = w[..., None]
            out += w * samp

    return _cast_back(out, orig_dtype)

def remap_video_numpy(
    video_iterable,
    mapx,
    mapy,
    *,
    interpolation="bilinear",
    border_mode="constant",
    cval=0.0,
    stack=False,
    max_workers=None,
):
    """
    Parameters
    ----------
    video_iterable : iterable of np.ndarray
        Frames shaped (H, W) or (H, W, C).
    mapx, mapy : np.ndarray
        Inverse maps from `build_affine_inverse_maps_numpy`.
    interpolation : {"bilinear", "nearest"}
        Selects resampling kernel.
    border_mode : {"constant","replicate","reflect"}
        Border handling strategy passed to the frame remapper.
    cval : float
        Constant padding value for `border_mode == "constant"`.
    stack : bool
        If True, stack the results along axis 0; otherwise return list of frames.
    max_workers : int | None
        Optional thread pool size. Defaults to ThreadPoolExecutor's heuristic when None.
    """
    if interpolation == "bilinear":
        remap_fn = remap_frame_bilinear_numpy
    elif interpolation == "nearest":
        remap_fn = remap_frame_nearest_neighbour_numpy
    elif interpolation == "bicubic":
        remap_fn = remap_frame_bicubic_numpy
    elif interpolation == "lanczos3":
        remap_fn = remap_frame_lanczos_3_numpy
    else:
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")

    def _remap(frame):
        return remap_fn(frame, mapx, mapy, border_mode, cval)

    if max_workers == 1:
        frames_out = [_remap(frame) for frame in video_iterable]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            frames_out = list(executor.map(_remap, video_iterable))
    return np.stack(frames_out, axis=0) if stack else frames_out

def rotate_video_nozzle_at_0_half_numpy(
    video,
    nozzle_center,
    offset_deg,
    *,
    interpolation="bilinear",
    out_shape=None,
    calibration_point=None,
    border_mode="constant",
    cval=0.0,
    stack=True,
    plot_maps=False,
):
    """
    Rotate the input video so the nozzle sits at the left edge and mid-height.

    Parameters
    ----------
    video : array-like
        numpy/Numpy array with frames at axis 0.
    nozzle_center : tuple[float, float]
        `(x, y)` nozzle location in the input frame.
    offset_deg : float
        Calibration offset to remove. Internal rotation is `-offset_deg` so a
        positive offset rotates clockwise, matching previous behaviour.
    out_shape : tuple[int, int], optional
        ROI `(H_out, W_out)` to extract after rotation.
    calibration_point : tuple[float, float], optional
        `(u_cal, v_cal)` in the rotated full canvas representing the nozzle.
    border_mode, cval, stack : forwarded to `remap_video_numpy`.
    interpolation : {"bilinear","nearest", "bicubic", "lanczos3"}
        Sampling kernel passed to `remap_video_numpy`.
    plot_maps : bool
        When True, display the generated inverse maps for inspection.
    """
    video_np = np.asarray(video)
    if video_np.dtype != np.float32:
        video_np = video_np.astype(np.float32)

    frame_shape = (video_np.shape[1], video_np.shape[2])
    angle_deg = -offset_deg

    mapx, mapy = build_nozzle_rotation_maps(
        frame_shape,
        nozzle_center,
        angle_deg,
        out_shape=out_shape,
        calibration_point=calibration_point,
    )

    if plot_maps:
        plot_inverse_maps(mapx, mapy)

    rotated = remap_video_numpy(
        video_np,
        mapx,
        mapy,
        interpolation=interpolation,
        border_mode=border_mode,
        cval=cval,
        stack=stack,
    )
    return rotated, mapx, mapy

def plot_inverse_maps(mapx, mapy):
    # bring back to NumPy if needed
    mapx_np = np.asnumpy(mapx)
    mapy_np = np.asnumpy(mapy)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(mapx_np, cmap='viridis')
    axs[0].set_title("mapx (source X coordinate)")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(mapy_np, cmap='magma')
    axs[1].set_title("mapy (source Y coordinate)")
    plt.colorbar(im1, ax=axs[1])

    plt.show()
    
