import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

"""
This module provides functions to rotate video frames efficiently on the GPU using CuPy.
First find the nozzle center and the angle offset by the GUI. 
"""



def build_affine_inverse_maps_cupy(
    M,
    out_shape,
    *,
    out_origin=None,
    center_on=None,
):
    """
    Build inverse maps for cv2.remap-style sampling on CuPy.

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
    (mapx, mapy) : tuple[cp.ndarray, cp.ndarray]
        CuPy float32 arrays of shape `(H_out, W_out)` with source sample coordinates.
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
    Minv = cp.asarray(Minv, dtype=cp.float32)

    if out_origin is not None:
        x_off, y_off = map(float, out_origin)
    elif center_on is not None:
        u_cal, v_cal = map(float, center_on)
        x_off = u_cal - (W_out - 1) / 2.0
        y_off = v_cal - (H_out - 1) / 2.0
    else:
        x_off = 0.0
        y_off = 0.0

    xs = cp.arange(W_out, dtype=cp.float32) + cp.float32(x_off)
    ys = cp.arange(H_out, dtype=cp.float32) + cp.float32(y_off)
    X, Y = cp.meshgrid(xs, ys)

    x_src = Minv[0, 0] * X + Minv[0, 1] * Y + Minv[0, 2]
    y_src = Minv[1, 0] * X + Minv[1, 1] * Y + Minv[1, 2]

    return x_src.astype(cp.float32), y_src.astype(cp.float32)

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

    return build_affine_inverse_maps_cupy(
        M,
        out_shape,
        out_origin=(x_off, y_off),
    )

def _reflect_indices(idx, size):
    # OpenCV-style BORDER_REFLECT_101 (no double-border), but here we’ll use simple reflect.
    # For large out-of-range, bring back via modulo and reflection.
    # If you want exact OpenCV behavior, adjust accordingly.
    if size == 1:
        return cp.zeros_like(idx)
    mod = cp.mod(idx, 2*(size-1))
    return cp.where(mod < size, mod, 2*(size-1) - mod)

def _clamp_indices(idx, size):
    return cp.clip(idx, 0, size-1)

def remap_frame_nearest_neighbour_cupy(img, mapx, mapy, border_mode="constant", cval=0.0):
    """
    Nearest-neighbour remap on GPU using CuPy.
    """
    assert img.ndim in (2, 3)
    H_in, W_in = img.shape[:2]

    orig_dtype = img.dtype
    img_f = img.astype(cp.float32, copy=False)

    xi = cp.rint(mapx).astype(cp.int32)
    yi = cp.rint(mapy).astype(cp.int32)

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
        fill = cp.float32(cval)
        if img_f.ndim == 2:
            gathered = img_f[yi_c, xi_c]
            sampled = cp.where(mask, gathered, fill)
        else:
            gathered = img_f[yi_c, xi_c, :]
            sampled = cp.where(mask[..., None], gathered, fill)
    else:
        raise ValueError("border_mode must be 'constant', 'replicate', or 'reflect'")

    if cp.issubdtype(orig_dtype, cp.integer):
        info = cp.iinfo(orig_dtype)
        sampled = cp.clip(cp.rint(sampled), info.min, info.max).astype(orig_dtype)
    elif cp.issubdtype(orig_dtype, cp.floating):
        sampled = sampled.astype(orig_dtype)
    else:
        sampled = sampled.astype(orig_dtype)
    return sampled

def remap_frame_bilinear_cupy(img, mapx, mapy, border_mode="constant", cval=0.0):
    """
    Bilinear remap on GPU using CuPy.

    Parameters
    ----------
    img : cp.ndarray
        Input image on GPU. Shape (H, W) for grayscale or (H, W, C) for multi-channel.
        Any numeric dtype is accepted. Output preserves dtype.
    mapx, mapy : cp.ndarray (float32), shape (H_out, W_out)
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
    cp.ndarray
        Remapped image on GPU. Shape is (H_out, W_out[, C]) with same dtype as input.
    """
    assert img.ndim in (2,3)
    H_in, W_in = img.shape[:2]
    H_out, W_out = mapx.shape

    # Keep original dtype for final cast; compute interpolation in float32 for speed/consistency.
    orig_dtype = img.dtype
    img_f = img.astype(cp.float32, copy=False)

    # Integer pixel neighbors around (x,y): floor and +1 for right/bottom neighbors.
    x0 = cp.floor(mapx).astype(cp.int32)
    y0 = cp.floor(mapy).astype(cp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts -> bilinear weights along x and y, then the four corner weights.
    wx = (mapx - x0).astype(cp.float32)
    wy = (mapy - y0).astype(cp.float32)
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
            img_f = cp.asarray(img_f)
            ix    = cp.asarray(ix).astype(cp.int64, copy=False)
            iy    = cp.asarray(iy).astype(cp.int64, copy=False)
            mask  = cp.asarray(mask, dtype=cp.bool_)
            cvald = cp.asarray(cval, dtype=cp.float32)

            if img_f.ndim == 2:  # (H, W)
                s = img_f[iy, ix].astype(cp.float32, copy=False)
                return cp.where(mask, s, cvald)
            else:                # (H, W, C)
                s = img_f[iy, ix, :].astype(cp.float32, copy=False)
                return cp.where(mask[..., None], s, cvald)

        v00 = sample_masked(img_f, x0c, y0c, m00, cval)
        v10 = sample_masked(img_f, x1c, y0c, m10, cval)
        v01 = sample_masked(img_f, x0c, y1c, m01, cval)
        v11 = sample_masked(img_f, x1c, y1c, m11, cval)

    else:
        raise ValueError("border_mode must be 'constant', 'replicate', or 'reflect'")

    # Helper: ensure CuPy float32 (no CPU bounce).
    def to_cp_f32(x):
        return cp.asarray(x, dtype=cp.float32)

    v00, v10, v01, v11 = map(to_cp_f32, (v00, v10, v01, v11))
    w00, w10, w01, w11 = map(to_cp_f32, (w00, w10, w01, w11))
    img_f = cp.asarray(img_f)  # ensure resident on device

    # Bilinear blend with correct broadcasting for channels.
    if img_f.ndim == 3:  # (H, W, C)
        out = (w00[..., None] * v00 +
               w10[..., None] * v10 +
               w01[..., None] * v01 +
               w11[..., None] * v11)
    else:  # (H, W)
        out = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11

    # Cast back to the original dtype with appropriate rounding/clipping for integers.
    if cp.issubdtype(orig_dtype, cp.integer):
        info = cp.iinfo(orig_dtype)
        out = cp.clip(cp.rint(out), info.min, info.max).astype(orig_dtype)
    elif cp.issubdtype(orig_dtype, cp.floating):
        out = out.astype(orig_dtype)
    else:
        out = out.astype(orig_dtype)
    return out


def remap_video_cupy(
    video_iterable,
    mapx,
    mapy,
    *,
    interpolation="bilinear",
    border_mode="constant",
    cval=0.0,
    stack=False,
):
    """
    Parameters
    ----------
    video_iterable : iterable of cp.ndarray
        Frames shaped (H, W) or (H, W, C).
    mapx, mapy : cp.ndarray
        Inverse maps from `build_affine_inverse_maps_cupy`.
    interpolation : {"bilinear", "nearest"}
        Selects resampling kernel.
    border_mode : {"constant","replicate","reflect"}
        Border handling strategy passed to the frame remapper.
    cval : float
        Constant padding value for `border_mode == "constant"`.
    stack : bool
        If True, stack the results along axis 0; otherwise return list of frames.
    """
    if interpolation == "bilinear":
        remap_fn = remap_frame_bilinear_cupy
    elif interpolation == "nearest":
        remap_fn = remap_frame_nearest_neighbour_cupy
    else:
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")

    frames_out = []
    for frame in video_iterable:
        frames_out.append(remap_fn(frame, mapx, mapy, border_mode, cval))
    return cp.stack(frames_out, axis=0) if stack else frames_out



def rotate_video_nozzle_at_0_half_cupy(
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
        CuPy/Numpy array with frames at axis 0.
    nozzle_center : tuple[float, float]
        `(x, y)` nozzle location in the input frame.
    offset_deg : float
        Calibration offset to remove. Internal rotation is `-offset_deg` so a
        positive offset rotates clockwise, matching previous behaviour.
    out_shape : tuple[int, int], optional
        ROI `(H_out, W_out)` to extract after rotation.
    calibration_point : tuple[float, float], optional
        `(u_cal, v_cal)` in the rotated full canvas representing the nozzle.
    border_mode, cval, stack : forwarded to `remap_video_cupy`.
    interpolation : {"bilinear","nearest"}
        Sampling kernel passed to `remap_video_cupy`.
    plot_maps : bool
        When True, display the generated inverse maps for inspection.
    """
    video_cp = cp.asarray(video)
    if video_cp.dtype != cp.float32:
        video_cp = video_cp.astype(cp.float32)

    frame_shape = (video_cp.shape[1], video_cp.shape[2])
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

    rotated = remap_video_cupy(
        video_cp,
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
    mapx_np = cp.asnumpy(mapx)
    mapy_np = cp.asnumpy(mapy)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(mapx_np, cmap='viridis')
    axs[0].set_title("mapx (source X coordinate)")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(mapy_np, cmap='magma')
    axs[1].set_title("mapy (source Y coordinate)")
    plt.colorbar(im1, ax=axs[1])

    plt.show()
    