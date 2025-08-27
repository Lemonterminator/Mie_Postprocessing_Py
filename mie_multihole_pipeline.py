from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *
from mie_postprocessing.video_filters import *
from mie_postprocessing.functions_bw import *
from mie_postprocessing.video_playback import *
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from scipy.ndimage import binary_erosion, generate_binary_structure

global ir_, or_
ir_ = 14
or_ = 380
'''
def preprocessing(video, hydraulic_delay_estimate, gamma=1, M=3, N=3, range_mask=True, time=False):
    if time:
        start_time = time.time()

    if gamma != 1:
        video = video**gamma 
    
    # 1) Compute background once, keepdims avoids [None, :, :]
    bkg = np.median(video[:hydraulic_delay_estimate], axis=0, keepdims=True).astype(video.dtype, copy=False)

    # 2) Allocate the destination explicitly and subtract without an extra temporary
    sub_bkg = np.empty_like(video)
    np.subtract(video, bkg, out=sub_bkg)        # faster + lower peak RAM than video - bkg

    # If you can mutate `video`, this is even leaner:
    # video -= bkg

    # 3) Run your filter
    sub_bkg_med = median_filter_video_auto(sub_bkg, M, N)

    # If `sub_bkg` isn’t needed later, free it explicitly:
    # del sub_bkg

    sub_bkg_med[sub_bkg_med<0] = 0

    # Masking to reduce the effect of induced brightness increase near nozzle
    if range_mask:
        px_range = np.max(sub_bkg_med, axis=0) - np.min(sub_bkg_med, axis=0)
        range_mask, _ = triangle_binarize_from_float(px_range)
        foreground = mask_video(sub_bkg_med, range_mask)
    else:
        foreground = sub_bkg_med
    
    scale = foreground.max()
    # Scaling
    foreground /= scale

    if time:
        print(f"Preprocessing calculation completed in {time.time()-start_time:.3f}s")

    return foreground
'''

def preprocessing(video, hydraulic_delay_estimate, gamma=1.0, M=3, N=3,
                  range_mask=True, timing=False, use_gpu=False,
                  triangle_backend="gpu", return_numpy=True):
    """
    triangle_backend: "gpu" (CuPy histogram-based triangle) or "cpu" (OpenCV)
    """
    import numpy as _np
    import time as _t

    xp = _np
    if use_gpu:
        import cupy as cp
        from cupyx.scipy.ndimage import median_filter as _cp_median_filter
        xp = cp

    t0 = _t.time() if timing else None

    # --- move to GPU if requested
    arr = xp.asarray(video)

    # Careful with gamma: do math in float32 to avoid integer power weirdness
    if gamma != 1:
        arr = arr.astype(xp.float32, copy=False)
        arr = xp.power(arr, gamma, dtype=arr.dtype)

    # Median background over frames [0:hydraulic_delay_estimate]
    bkg = xp.median(arr[:hydraulic_delay_estimate], axis=0, keepdims=True).astype(arr.dtype, copy=False)

    # background subtraction without extra temp
    sub_bkg = xp.empty_like(arr)
    xp.subtract(arr, bkg, out=sub_bkg)

    # --- median filter per frame on GPU/CPU
    # Then clamp negatives in-place
    if use_gpu:
        # size=(1,M,N): filter spatially, not across time
        sub_bkg_med = _cp_median_filter(sub_bkg, size=(1, M, N))
        xp.maximum(sub_bkg_med, 0, out=sub_bkg_med)
    else:
        # your existing CPU routine
        sub_bkg_med = median_filter_video_auto(sub_bkg, M, N)
        sub_bkg_med[sub_bkg_med < 0] = 0


        

    # --- optional range mask near nozzle
    if range_mask:
        px_range = xp.amax(sub_bkg_med, axis=0) - xp.amin(sub_bkg_med, axis=0)

        if use_gpu and triangle_backend == "gpu":
            mask = _triangle_binarize_gpu(px_range)     # boolean (H,W)
        else:
            # tiny transfer; only one HxW plane
            import cv2
            u8 = cv2.normalize(xp.asnumpy(px_range) if use_gpu else px_range,
                               None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
            _, mask_u8 = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            mask = (mask_u8.astype(bool))
            if use_gpu:
                mask = xp.asarray(mask)

        foreground = sub_bkg_med * mask[None, ...]
    else:
        foreground = sub_bkg_med

    # scale safely (avoid divide-by-zero)
    scale = xp.max(foreground)
    if (scale > 0) if not use_gpu else bool(scale.get() > 0):
        foreground = foreground / scale

    if timing:
        print(f"Preprocessing completed in {(_t.time()-t0):.3f}s (use_gpu={use_gpu})")

    return xp.asnumpy(foreground) if (return_numpy and use_gpu) else foreground


# ---- GPU Triangle threshold (CuPy), ignores zeros ---------------------------
def _triangle_binarize_gpu(px_range_cp):
    import cupy as cp
    x = px_range_cp

    # ignore zeros
    nz = x[x > 0]
    if nz.size == 0:
        return cp.zeros_like(x, dtype=cp.bool_)

    # normalize to 0..255 for stable histogram like OpenCV
    vmin = nz.min()
    vmax = nz.max()
    u8 = cp.asarray(cp.floor((nz - vmin) * (255.0 / (vmax - vmin + 1e-12))), dtype=cp.uint8)

    h, _ = cp.histogram(u8, bins=256, range=(0, 255))
    nzbins = cp.nonzero(h)[0]
    i0, i1 = nzbins[0], nzbins[-1]
    imax = int(h.argmax())

    iend = int(i0 if (imax - i0) > (i1 - imax) else i1)
    lo, hi = (iend, imax) if iend < imax else (imax, iend)
    xs = cp.arange(lo, hi + 1)
    ys = h[xs].astype(cp.float32)

    x0, y0 = float(imax), float(h[imax])
    x1, y1 = float(iend), float(h[iend])
    denom = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5 + 1e-12
    # perpendicular distance from line
    d = cp.abs((y0 - y1) * xs + (x1 - x0) * ys + (x0 * y1 - x1 * y0)) / denom
    t_idx = int(xs[int(d.argmax())])

    # apply threshold back on the normalized array
    mask_nz = (u8 >= t_idx)
    mask_full = cp.zeros_like(x, dtype=cp.bool_)
    mask_full[x > 0] = mask_nz
    return mask_full



def has_cupy_gpu():
    """Return (available: bool, info: str)."""
    try:
        import cupy as cp
    except Exception as e:
        return False, f"CuPy import failed: {e}"

    try:
        ndev = cp.cuda.runtime.getDeviceCount()
        if ndev <= 0:
            return False, "No CUDA/HIP devices found"
        # Touch the device once to ensure runtime + context are OK
        with cp.cuda.Device(0):
            _ = cp.asarray([0]).sum()
        return True, f"CuPy ready on {ndev} device(s)"
    except Exception as e:
        return False, f"CuPy GPU unavailable: {e}"

def resolve_backend(use_gpu="auto", triangle_backend="auto"):
    """
    Decide final (use_gpu, triangle_backend, xp) based on availability and hints.
    - use_gpu: True | False | "auto"
    - triangle_backend: "gpu" | "cpu" | "auto"
    """
    ok, _info = has_cupy_gpu()
    if use_gpu == "auto":
        use_gpu = ok
    else:
        use_gpu = bool(use_gpu and ok)

    if triangle_backend == "auto":
        triangle_backend = "gpu" if use_gpu else "cpu"
    elif triangle_backend == "gpu" and not use_gpu:
        triangle_backend = "cpu"  # can’t do GPU triangle without GPU

    xp = np
    if use_gpu:
        import cupy as cp
        xp = cp

        # Optional: verify GPU median filter is available; else fall back to CPU
        try:
            from cupyx.scipy.ndimage import median_filter as _check
        except Exception:
            use_gpu = False
            xp = np
            triangle_backend = "cpu"

    return use_gpu, triangle_backend, xp


def mie_multihole_pipeline(video, centre, number_of_plumes):
    centre_x = float(centre[0])
    centre_y = float(centre[1])

    hydraulic_delay_estimate = 15  # frames to skip for median background
    
    use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

    foreground = preprocessing(
        video,
        hydraulic_delay_estimate,
        gamma=1.0,
        M=3, N=3,
        range_mask=True,
        timing=True,
        use_gpu=use_gpu,
        triangle_backend=triangle_backend,
        return_numpy=not use_gpu,   # keep on device if GPU path is used
    )

        # Cone Angle 

    start_time = time.time()
    # Cone angle
    bins = 3600
    signal_density_bins, signal, density = angle_signal_density_auto(
        foreground, centre_x, centre_y, N_bins=bins
    )

    # Estimate optimal rotation offset using FFT of the summed angular signal
    summed_signal = signal.sum(axis=0)
    fft_vals = np.fft.rfft(summed_signal)
    if number_of_plumes < len(fft_vals):
        phase = np.angle(fft_vals[number_of_plumes])
        offset = (-phase / number_of_plumes) * 180.0 / np.pi
        offset %= 360.0
        offset = min(offset, (offset-360), key=abs)
        print(f"Estimated offset from FFT: {offset:.3f} degrees")
    
    print(f"Cone angle calculation completed in {time.time()-start_time:.3f}s")
    
    
    # Generate the crop rectangle based on the plume parameters
    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)

    angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - offset
    mask = generate_plume_mask(ir_, or_, crop[2], crop[3])

    foreground = foreground.astype(np.float32, copy=False) # type: ignore
    segments=rotate_all_segments_auto(foreground, angles, crop, centre, mask=mask)
    elapsed_time = time.time() - start_time
    print(f"Computing all rotated segments finished in {elapsed_time:.2f} seconds.")
    
    segments = np.stack(segments, axis=0)
    return segments
