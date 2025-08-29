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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

global ir_, or_
ir_ = 14
or_ = 380


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

def binarize_videos(segments, hydraulic_delay, penetration):
    P = segments.shape[0]
    bw_vids = np.zeros(segments.shape)
    for i, segment in enumerate(segments):
        # print(i)
        bw_vid = np.zeros(segment.shape).astype(int)
        
        for j in range(int(hydraulic_delay[i]), len(segment)):
            # bw_vid[j], thres_temp = triangle_binarize_from_float(segment[j], blur=True)
            bw_temp, _ = triangle_binarize_from_float(segment[j].get())
            bw_vid[j] = keep_largest_component(bw_temp)
        bw_vids[i] = bw_vid

    col = penetration[i] 
    # Playing

    # gain = 100
    # play_videos_side_by_side(tuple((gain*(1-bw_vids)*segments.get())[5:10]), intv=70)

    penetration_old = np.zeros(penetration.shape)
    col_sum_bw = np.sum(bw_vids, axis=2) >=1
    n_workers = min(os.cpu_count() + 4, P, 32) # type: ignore
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(penetration_bw_to_index, col_sum_bw[p]): p for p in range(col_sum_bw.shape[0])}
        for fut in as_completed(futs):
            penetration_old[futs[fut]] = fut.result()

    return bw_vids, penetration_old


def mie_multihole_pipeline(video, centre, number_of_plumes, gamma=1.0, binarize_video=False):
    centre_x = float(centre[0])
    centre_y = float(centre[1])

    hydraulic_delay_estimate = 15  # frames to skip for median background
    
    use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

    foreground = preprocessing(
        video,
        hydraulic_delay_estimate,
        gamma=gamma,
        M=3, N=3,
        range_mask=True,
        timing=True,
        use_gpu=use_gpu,
        triangle_backend=triangle_backend,
        return_numpy=not use_gpu,   # keep on device if GPU path is used
    )

    # Cone Angle 
    start_time = time.time()
    # accuracy : 360/bins

    '''
    gamma_foreground = 1.5
    bins = 3600
    signal_density_bins, signal, density = angle_signal_density_auto(
        # foreground**gamma_foreground, centre_x, centre_y, N_bins=bins
        video, centre_x, centre_y, N_bins=bins
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
    '''
    bins = 3600
    signal_density_bins, signal, density = angle_signal_density(
        foreground.get(), centre_x, centre_y, N_bins=bins
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
    
    # Generate the crop rectangle based on the plume parameters
    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)
    # Angles of ideal spary axis
    angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - offset
    # Triangular plume region mask with offsets.
    mask = generate_plume_mask(ir_, or_, crop[2], crop[3])


    # ``foreground`` may still reside on the GPU if ``preprocessing`` ran on
    # CuPy.  OpenCV's CUDA routines expect host ``numpy`` arrays when calling
    # ``cv2.cuda_GpuMat.upload``, so convert to ``numpy`` before rotation.
    foreground = xp.asarray(foreground, dtype=xp.float32, copy=False)
    segments = rotate_all_segments_auto(foreground, angles, crop, centre, mask=mask)
    segments = xp.stack(segments, axis=0)
    segments[segments < 1e-3] = 0
    # zeros_regions = segments[]

    elapsed_time = time.time() - start_time
    

    

    print(f"Computing all rotated segments finished in {elapsed_time:.2f} seconds.")

    # Time-distance-intensity maps
    td_intensity_maps = xp.sum(segments, axis=2)
    P, F, C = td_intensity_maps.shape

    # Number of none-zero in each column in the mask
    count = np.sum(mask.astype(np.uint8), axis=0)
    count = xp.asarray(count) # to GPU if needed

    # Average td_intensity_maps by count
    # Time-distance-intensity maps are now the average column intensity,
    # in each plume, frame, at each distance from the centre 
    td_intensity_maps = td_intensity_maps/count[None, None, :]

    # summing over each image to get the total intensity in each image at each frame
    energies = xp.sum(td_intensity_maps, axis=2)
    # Find the frame with peak brightness
    peak_brightness_frames = xp.argmax(energies, axis=1)
    avg_peak = int(xp.mean(peak_brightness_frames))

    # Hydraulic delay estimation
    # multiplier: manual, x times of the total instensity compared to no plume frames
    multiplier = 100
    # Threshold for the derivative of energy. 
    # Computed as the multiplier *  blank frame enrgy (first frame) / peak energy
    thres_derivative = multiplier*energies[:,0]/energies[:,avg_peak]
    rows = segments.shape[2]; cols = segments.shape[3]

    # Define a custom region near the nozzle to compute the energy
    H_low = round(rows*3/7); H_high = round(rows*4/7)
    W_right = round(cols/10)
    near_nozzle_energies = xp.sum(xp.sum(segments[:, :avg_peak, H_low:H_high, :W_right], axis=3), axis=2)
    # Find the discrete derivative of energy in this region
    dE1dts = np.diff(near_nozzle_energies[:, 0:avg_peak], axis=1)
    # masks = dE1dts > (thres_derivative*np.max(dE1dts, axis=1))[:,None]
    # hydraulic_delay = masks.argmax(axis=1)
    hydraulic_delay =  (dE1dts > 1).argmax(axis=1)

    # Penetration by averaging columns and find the strongest edge
    penetration = np.full((P, F), np.nan, dtype=np.float32)

    # TODO: Multithreading implementation
    lower = 0; upper = 366
    fig, ax = plt.subplots(P//3+1, 3, figsize=(12, 3*P/3))

    # Pre-compute Otsu masks on GPU to avoid CPU transfers
    bw_otsu_all = None
    if use_gpu:
        # arrs shape: (P, X, F) where X is distance/radius (columns after transpose)
        arrs = xp.transpose(td_intensity_maps, (0, 2, 1))
        P_, X_, F_ = arrs.shape

        # Normalize each plume slice to 0..255 uint8 (OpenCV-like)
        a_min = xp.min(arrs, axis=(1, 2))
        a_max = xp.max(arrs, axis=(1, 2))
        denom = xp.maximum(a_max - a_min, 1e-6)
        scale = (255.0 / denom)[:, None, None]
        u8 = xp.clip((arrs - a_min[:, None, None]) * scale, 0, 255).astype(xp.uint8)

        # Compute Otsu threshold per plume on GPU and build masks
        bw_otsu_all = xp.zeros((P_, X_, F_), dtype=bool)
        bins = xp.arange(256, dtype=xp.float32)
        start_time = time.time()
        for pp in range(P_):
            # Histogram on GPU
            hist = xp.histogram(u8[pp], bins=256, range=(0, 256))[0].astype(xp.float32)
            # Cumulative sums for class weights and means
            w1 = xp.cumsum(hist)
            w2 = w1[-1] - w1
            m1 = xp.cumsum(hist * bins)
            m2_total = m1[-1]
            m2 = m2_total - m1
            mu1 = xp.where(w1 > 0, m1 / w1, 0)
            mu2 = xp.where(w2 > 0, m2 / w2, 0)
            sigma_b2 = w1 * w2 * (mu1 - mu2) ** 2
            t = int(xp.argmax(sigma_b2))
            bw_otsu_all[pp] = u8[pp] >= t
        print(f"All Otsu BW computed in {time.time()-start_time :2f}s")
    
    # TODO: Multithreading implementation
    start_time = time.time()
    for p in range(P):
        # Correcting decay after peak brightness frame
        decay_curve = energies[p, peak_brightness_frames[p]:]
        decay_curve = decay_curve/np.max(decay_curve)
        td_intensity_maps[p, peak_brightness_frames[p]:, :] = td_intensity_maps[p, peak_brightness_frames[p]:, :]/decay_curve[:,None]
        
        
        arr = td_intensity_maps[p, :, :].T


        X, F = arr.shape
        bw = _triangle_binarize_gpu(arr)
        
        bw = keep_largest_component(bw.get())
        edge_tri = np.argmax(bw[::-1, :], axis=0)
        edge_tri = bw.shape[0]-edge_tri
        edge_tri[0:int(hydraulic_delay[p]+5)][edge_tri[0:int(hydraulic_delay[p]+5)]==X]=0
        penetration[p] = edge_tri

        plt.figure(1)
        ax[p//3, p%3].imshow(arr.get(), origin="lower", aspect="auto")
        # ax[p//3, p%3].plot(penetration[p], color="red")
        
        # Use pre-computed Otsu masks on GPU to avoid CPU roundtrips
        if use_gpu and bw_otsu_all is not None:
            bw_otsu = bw_otsu_all[p]
        else:
            # CPU fallback using OpenCV (kept for non-GPU runs)
            import cv2
            _, bw_otsu = cv2.threshold(
                cv2.normalize(arr.get(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), # type: ignore
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )  # type: ignore
        
        # bw_otsu = ~ bw_otsu.astype(np.bool)
        
        start = int(hydraulic_delay[p]+1)
        end = int(peak_brightness_frames[p])

        differential = np.diff(arr[:, start:end]* xp.array(bw_otsu[:,start:end]),axis=1).get() #type:ignore
        # differential = (differential * bw_otsu[:,start:end-1]).astype(np.float32)
        differential[differential<0] = 0
        # plt.imshow(differential, origin="lower", aspect="auto")

        edge_diff = np.argmax(differential[::-1, :], axis=0)
        edge_diff = differential.shape[0]-edge_diff
        edge_diff[edge_diff>upper-10] = 0
        edge_diff[edge_diff<lower+10] = 0
       
        decision = np.maximum(penetration[p, start:end-1], edge_diff)
        penetration[p, start:end-1]= decision
        penetration[p, :int(np.max(hydraulic_delay))]=0
        
        ax[p//3, p%3].plot(penetration[p], color="red")

    ax[p//3, p%3+1].plot(penetration.T)

    # Data Cleaning: Enforce monotonicity
    np.maximum.accumulate(penetration, axis=1, out=penetration)
    # Data Cleaning: remove zeros and values too close to the maxium
    penetration[penetration==0] = np.nan
    for p in range(P):
        penetration[p, int(hydraulic_delay[p])] = 0.0
    penetration[penetration>upper-2] = np.nan
    # Data Cleaning: remove too large value before injection
    penetration[:, :int(penetration.shape[1]/2)][penetration[:, :int(penetration.shape[1]/2)] > upper-10] = np.nan
    print(f"Post processing completed in {time.time()-start_time:.2f}s")

    ax[p//3, p%3+2].plot(penetration.T)
        
    # plt.show()

    
    if binarize_video:
        start_time = time.time()

        bw_vids, penetration_old= binarize_videos(segments, hydraulic_delay, penetration)

        # Data Cleaning: Enforce monotonicity
        np.maximum.accumulate(penetration_old, axis=1, out=penetration_old)
        # Data Cleaning: remove zeros and values too close to the maxium
        penetration_old[penetration_old==0] = np.nan
        penetration_old[penetration_old>upper-2] = np.nan
        # Data Cleaning: remove too large value before injection
        penetration_old[:, :int(penetration_old.shape[1]/2)][penetration_old[:, :int(penetration_old.shape[1]/2)] > upper-10] = np.nan
        
        # Comparison
        fig = plt.subplots(1,1)
        with np.errstate(invalid="ignore", all="ignore"):
            plt.plot(np.nanmedian(penetration, axis=0), label="Triangular segmentation for column sum")
            plt.plot(np.nanmedian(penetration_old, axis=0), label="Triagular segmentation for each frame")
        plt.xlabel("Frame number")
        plt.ylabel("Penetration [px]")
        plt.title("Comparision of two penetration calculation methods, median")
        plt.legend()

        fig = plt.subplots(1,1)
        with np.errstate(invalid="ignore", all="ignore"):
            plt.plot(np.nanmean(penetration, axis=0), label="Triangular segmentation for column sum")
            plt.plot(np.nanmean(penetration_old, axis=0), label="Triagular segmentation for each frame")
        plt.xlabel("Frame number")
        plt.ylabel("Penetration [px]")
        plt.title("Comparision of two penetration calculation methods, mean")
        plt.legend()

        fig = plt.subplots(1,1)
        plt.title("Area of all segements")
        # Area demostration
        area_all_segmets = np.sum(np.sum(bw_vids, axis=3), axis=2)
        plt.plot(area_all_segmets.T)

        boundaries = bw_boundaries_all_points(bw_vids)
        i = 4
        print(f"Binarizing video and calculating boundary completed in {time.time()-start_time:2f}s")
        play_video_with_boundaries_cv2(segments[i].get(), boundaries[i], gain=1.0, binarize=False, intv=170)




    # Cone Angle
    bw_ang, _ = triangle_binarize_from_float(signal, blur=True)
    bins = 3600
    shift_bins = int((offset/360*bins))
    bw_shifted = np.roll(bw_ang, -shift_bins, axis=1)
    # Closing
    struct = np.ones((1, 3), dtype=bool)
    plume_widths  = np.zeros((number_of_plumes, bw_ang.shape[0]), dtype= np.float32)
    from scipy.ndimage import binary_closing
    for p in range(number_of_plumes):
        start = int(round(p * bins / number_of_plumes))
        end = int(round((p + 1) * bins / number_of_plumes))
        region = bw_shifted[:, start:end]
        closed = binary_closing(region, structure=struct)
        bw_shifted[:, start:end] = closed
        plume_widths[p] = closed.sum(axis=1) * (360.0 / bins)
    plt.subplots(1,1)
    plt.plot(plume_widths.T)

    plt.show()
    plt.close('all')
        

    if use_gpu:
        segments = xp.asnumpy(segments)
        # bw_vids = xp.asnumpy(bw_vids)


    
    return segments
