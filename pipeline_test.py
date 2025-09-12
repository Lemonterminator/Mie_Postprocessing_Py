from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *
from mie_postprocessing.video_filters import *
from mie_postprocessing.functions_bw import *
from mie_postprocessing.video_playback import *
from mie_multihole_pipeline import *

# from sklearn.preprocessing import normalize
from tkinter import ttk, filedialog, messagebox, colorchooser
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import convolve2d
import asyncio

import os
import gc
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import measure
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.stats import norm

global visualization 
visualization = True

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
from skimage.measure import find_contours  # pip install scikit-image

def upper_boundary_contour(u8, q=0.80, smooth_sigma=1.5, monotonic=True, show=False):
    """
    u8: 2D array (X, F) in your code after transpose; origin='lower' when plotting.
    q:  quantile level for the iso-contour (0..1). Increase if boundary falls too low.
    """
    # 1) Light smoothing to suppress speckle; keep edges sharp-ish
    if smooth_sigma > 0:
        u8f = cv2.GaussianBlur(u8.astype(np.float32), ksize=(0,0), sigmaX=smooth_sigma)
    else:
        u8f = u8.astype(np.float32)

    # 2) Pick an iso-intensity level by quantile (ignore zeros/NaNs)
    valid = np.isfinite(u8f) & (u8f > 0)
    level = np.quantile(u8f[valid], q)

    # 3) Marching-squares contours at this level
    cs = find_contours(u8f, level=level)  # list of (N_i, 2) arrays in (row, col) = (y, x)
    if not cs:
        return None, level

    X, F = u8f.shape

    # 4) Choose the contour that spans the widest x-range (best for an envelope)
    def span_x(c): 
        xs = c[:,1]
        return xs.max() - xs.min()
    C = max(cs, key=span_x)

    # 5) Build an upper envelope: for each integer column, take the MAX y along the contour
    xs = C[:,1]; ys = C[:,0]
    # Bin by integer columns
    col = np.clip(xs.round().astype(int), 0, F-1)
    y_max = np.full(F, np.nan, dtype=np.float32)
    for j in range(F):
        sel = (col == j)
        if np.any(sel):
            y_max[j] = ys[sel].max()  # "upper" in image coords (bigger row index)
    # Fill gaps by linear interpolation
    good = np.isfinite(y_max)
    if good.sum() >= 2:
        y_max[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), y_max[good])

    # 6) Enforce non-decreasing (喷雾贯入应随时间单调增加). Optional.
    if monotonic:
        y_max = np.maximum.accumulate(y_max)

    if show:
        plt.imshow(u8f, origin="lower", aspect="auto")
        plt.plot(y_max, label=f"Contour q={q:.2f}")
        plt.legend(); plt.show()

    # Convert back to your coordinate convention if needed
    return y_max, level



def _process_one_frame(bw, xlo, xhi, connectivity=2):
    """
    bw: (H, W) bool/0-1
    xlo, xhi: integer pixel bounds [inclusive]
    connectivity: 1->4-connectivity, 2->8-connectivity
    Returns: coords_top, coords_bottom as (N,2) arrays of (y,x)
    """
    H, W = bw.shape
    # Boundary extraction: bw AND NOT eroded(bw)
    struct = generate_binary_structure(2, 2 if connectivity == 2 else 1)
    boundary = bw & ~binary_erosion(bw, structure=struct, border_value=0)

    # Clip and build an x-band mask (broadcast along rows)
    xlo_i = max(0, int(np.floor(xlo)))
    xhi_i = min(W - 1, int(np.ceil(xhi)))
    if xlo_i > xhi_i or not boundary.any():
        return (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32))

    xmask = np.zeros((1, W), dtype=bool)
    xmask[:, xlo_i:xhi_i + 1] = True

    sel = boundary & xmask   # (H, W)

    ys, xs = np.nonzero(sel)
    if ys.size == 0:
        return (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32))

    # Split by midline
    mid = (H - 1) / 2.0
    top_mask = ys <= mid
    bot_mask = ~top_mask

    coords_top = np.column_stack((ys[top_mask], xs[top_mask])).astype(np.int32)
    coords_bot = np.column_stack((ys[bot_mask], xs[bot_mask])).astype(np.int32)
    return coords_top, coords_bot


# --- Split: compute all boundary points (no x-band filter) ---
def _boundary_points_one_frame(bw, connectivity=2):
    """
    Compute all boundary points of a single binary frame without x-band filtering.

    Returns: (coords_top_all, coords_bottom_all) as (N,2) int32 arrays (y, x).
    """
    H, W = bw.shape
    struct = generate_binary_structure(2, 2 if connectivity == 2 else 1)
    boundary = bw & ~binary_erosion(bw, structure=struct, border_value=0)
    if not boundary.any():
        return (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32))

    ys, xs = np.nonzero(boundary)
    if ys.size == 0:
        return (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32))

    mid = (H - 1) / 2.0
    top_mask = ys <= mid
    bot_mask = ~top_mask

    coords_top = np.column_stack((ys[top_mask], xs[top_mask])).astype(np.int32)
    coords_bot = np.column_stack((ys[bot_mask], xs[bot_mask])).astype(np.int32)
    return coords_top, coords_bot


def bw_boundaries_all_points(
    bw_vids, connectivity=2, parallel=False, max_workers=None
):
    """
    Compute all boundary points for every frame (no x-band filtering).

    Parameters
    ----------
    bw_vids : array, shape (R, F, H, W), binary
    connectivity : 1 (4-neigh) or 2 (8-neigh)
    parallel : bool
    max_workers : int or None

    Returns
    -------
    result : list length R; each item is list length F of tuples (coords_top_all, coords_bottom_all)
    """
    R, F, H, W = bw_vids.shape
    result = [[None] * F for _ in range(R)]

    def work(i, j):
        bw = np.asarray(bw_vids[i, j], dtype=bool)
        return i, j, _boundary_points_one_frame(bw, connectivity)

    if parallel:
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(work, i, j) for i in range(R) for j in range(F)]
            for fut in as_completed(futs):
                i, j, tup = fut.result()
                result[i][j] = tup
    else:
        for i in range(R):
            for j in range(F):
                _, _, tup = work(i, j)
                result[i][j] = tup

    return result


# --- Split: filter previously computed boundary points by x-band ---
def bw_boundaries_xband_filter(boundary_results, penetration_old, lo=0.1, hi=0.6):
    """
    Filter precomputed boundary points by an x-band defined by lo/hi fractions of penetration.

    Parameters
    ----------
    boundary_results : list of lists as returned by bw_boundaries_all_points
        shape [R][F] of (coords_top_all, coords_bottom_all)
    penetration_old : array, shape (R, F)
        Penetration in pixels along x for each (R, F)
    lo, hi : floats
        Inclusive fractional range of penetration to keep

    Returns
    -------
    filtered : same nested structure with coords filtered to x in [xlo, xhi]
    """
    R = len(boundary_results)
    F = len(boundary_results[0]) if R > 0 else 0
    assert penetration_old.shape == (R, F)

    def _filter_coords(coords, xlo, xhi):
        if coords.size == 0:
            return coords
        x = coords[:, 1]
        xlo_i = int(np.floor(max(0, xlo)))
        xhi_i = int(np.ceil(max(xlo_i, xhi)))
        keep = (x >= xlo_i) & (x <= xhi_i)
        return coords[keep]

    filtered = [[None] * F for _ in range(R)]
    for i in range(R):
        for j in range(F):
            coords_top_all, coords_bot_all = boundary_results[i][j]
            xlo = lo * float(penetration_old[i, j])
            xhi = hi * float(penetration_old[i, j])
            coords_top = _filter_coords(coords_top_all, xlo, xhi)
            coords_bot = _filter_coords(coords_bot_all, xlo, xhi)
            filtered[i][j] = (coords_top.astype(np.int32, copy=False),
                              coords_bot.astype(np.int32, copy=False))

    return filtered


def bw_boundaries_xband_split(
    bw_vids, penetration_old, lo=0.1, hi=0.6, connectivity=2,
    parallel=False, max_workers=None
):
    """
    Parameters
    ----------
    bw_vids : array, shape (R, F, H, W), binary
    penetration_old : array, shape (R, F), in pixels along x
    lo, hi : floats for fraction of penetration to keep (inclusive range)
    connectivity : 1 (4-neigh) or 2 (8-neigh)
    parallel : bool, use threads across frames for speed
    max_workers : int or None

    Returns
    -------
    result : list length R; each item is list length F of tuples (coords_top, coords_bottom),
             where coords_* are (N,2) int arrays of (y,x).
    """
    R, F, H, W = bw_vids.shape
    assert penetration_old.shape == (R, F)

    result = [[None] * F for _ in range(R)]

    def work(i, j):
        bw = np.asarray(bw_vids[i, j], dtype=bool)
        xlo = lo * float(penetration_old[i, j])
        xhi = hi * float(penetration_old[i, j])
        return i, j, _process_one_frame(bw, xlo, xhi, connectivity)

    if parallel:
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(work, i, j) for i in range(R) for j in range(F)]
            for fut in as_completed(futs):
                i, j, tup = fut.result()
                result[i][j] = tup
    else:
        for i in range(R):
            for j in range(F):
                _, _, tup = work(i, j)
                result[i][j] = tup

    return result

def find_penetration(intensity2d:np.ndarray) -> np.ndarray:
     # intensity2d: (F, C), float32
    # pen_2d, _ = triangle_binarize_from_float(intensity2d, blur=False)

    arr = np.asarray(intensity2d, dtype=np.float32)
    assert arr.ndim == 2

    # Valid mask
    mask = np.isfinite(arr) & (arr > 0)

    # normalize only over masked pixels -> uint8 in [0,255]
    u8 = np.zeros_like(arr, dtype=np.uint8)
    if mask.any():
        u8 = cv2.normalize(
            arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,
            mask=mask.astype(np.uint8)  # min/max from masked region only
        ) # type: ignore

    if u8.dtype != np.uint8:
        u8 = u8.astype(np.uint8, copy=False)

    edges = cv2.Canny(u8, 50, 150)
    # visualize
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imshow(u8*edges, origin="lower", cmap="gray")

    plt.show()
 

 
    _, pen_2d = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)        
    

    largest = keep_largest_component(pen_2d, connectivity=2)
    pen1d = penetration_bw_to_index(largest).astype(np.float32)
    pen1d[pen1d < 0] = np.nan
    return pen1d

def segments_computation(segments, mask):

    # segmnets has shape [plume number, frame, rows, cols]

    # Map of intensity and time by summing over rows
    # td_intensity_maps has shape [plume number, frame, cols]
    td_intensity_maps = np.sum(segments, axis=2)

    count = np.sum(mask.astype(np.uint8), axis=0)
    td_intensity_maps = td_intensity_maps/count[None, None, :]
    P, F, C = td_intensity_maps.shape
    penetration = np.full((P, F), np.nan, dtype=np.float32)

    # Plot of energy by summing again over cols (summing over each image)
    energies = np.sum(td_intensity_maps, axis=2)

    # Find the frame with peak brightness
    peak_brightness_frames = np.argmax(energies, axis=1)
    avg_peak = round(np.mean(peak_brightness_frames))

    # Hydraulic delay estimation
    # multiplier: manual, x times of the total instensity compared to no plume frames
    multiplier = 100
    thres_derivative = multiplier*energies[:,0]/energies[:,avg_peak]
    rows = segments.shape[2]; cols = segments.shape[3]
    near_nozzle_energies = np.sum(np.sum(segments[:, :avg_peak, round(rows*3/7):round(rows*4/7), :round(cols/10)], axis=3), axis=2)
    dE1dts = np.diff(near_nozzle_energies[:, 0:avg_peak], axis=1)
    masks = dE1dts > (thres_derivative*np.max(dE1dts, axis=1))[:,None]
    hydraulic_delay = masks.argmax(axis=1) + 1

    # Correcting decay after peak brightness frame
    for p in range(P):
        # decay_correction = td_intensity_maps[p, peak_brightness_frames[p]:, :]
        decay_curve = energies[p, peak_brightness_frames[p]:]
        decay_curve = decay_curve/np.max(decay_curve)
        # energies[p, peak_brightness_frames[p]:] = np.max(decay_curve)
        '''
        # td_intensity_maps[p, peak_brightness_frames[p]:, :] = td_intensity_maps[p, peak_brightness_frames[p]:, :]/decay_curve[:,None]
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(td_intensity_maps[p, peak_brightness_frames[p]:, :], origin="lower", aspect="auto")
        ax[1].imshow(td_intensity_maps[p, peak_brightness_frames[p]:, :]/decay_curve[:,None], origin="lower", aspect="auto")
        ax[0].set_title("Before decay correction")
        ax[1].set_title("After decay correction")

        ax[2].plot(decay_curve)
        ax[2].set_title("Normalized decay curve to let the total intensity constant after peak")
        plt.show()
        '''
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(td_intensity_maps[p], origin="lower", aspect="auto")
        td_intensity_maps[p, peak_brightness_frames[p]:, :] = td_intensity_maps[p, peak_brightness_frames[p]:, :]/decay_curve[:,None]
        # ax[1].imshow(td_intensity_maps[p], origin="lower", aspect="auto")

    # Converting intensity maps to penetration
    start_time = time.time()
    n_workers = min(os.cpu_count() + 4, P, 32) # type: ignore

    for p in range(P):
        thres = 100*np.max(td_intensity_maps[p, :hydraulic_delay[p], :])
        maps_bw = keep_largest_component(td_intensity_maps[p].T > thres)[::-1, :] 
        penetration[p] = C - maps_bw.argmax(axis=0)
        penetration[p, :hydraulic_delay[p]+1][penetration[p, :hydraulic_delay[p]+1] == C]=0.0 
        
        # Demo plotting
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(td_intensity_maps[p].T, origin="lower", aspect="auto", cmap="gray")
        ax[0].plot(penetration[p], color="red", label="Penetration from intensity map")
        ax[0].axvline(hydraulic_delay[p], color="blue")
        ax[0].set_xlabel("Time [frames]")
        ax[2].plot(energies[p])
        ax[2].set_xlabel("Time [frames]")

        ax[0].set_ylabel("Penetration [px]")
        ax[2].set_ylabel("Intensity Sum")

        ax[0].set_title("Time-Distane Intensity \n & \n penetration")
        ax[2].set_title("Intensity sum in all frames")

        ax[0].set_ylim(0, C)

        ax[1].imshow(maps_bw, origin="lower", aspect="auto")


    fig, ax = plt.subplots(P//2, 2, figsize=(12, 3*P/2))

    for p in range(P):
        # Valid mask
        arr = td_intensity_maps[p]
        mask = np.isfinite(arr) & (arr > 0)

        # normalize only over masked pixels -> uint8 in [0,255]
        u8 = np.zeros_like(arr, dtype=np.uint8)
        if mask.any():
            u8 = cv2.normalize(
                arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,
                mask=mask.astype(np.uint8)  # min/max from masked region only
            ) # type: ignore

        if u8.dtype != np.uint8:
            u8 = u8.astype(np.uint8, copy=False)

        # Try transposed
        u8 = cv2.equalizeHist(u8.T)
        X, F = u8.shape
        plt.figure(1)
        ax[p//2, p%2].imshow(u8, origin="lower", aspect="auto")
        
        # _, bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bw, thres = triangle_binarize_u8(u8, blur=False)
        bw= keep_largest_component(bw)
        edge_tri = np.argmax(bw[::-1, :], axis=0)
        edge_tri = bw.shape[0]-edge_tri
        edge_tri[0:round(F/2)][edge_tri[0:round(F/2)]==X]=0
        ax[p//2, p%2].plot(edge_tri, color="blue")
        penetration[p] = edge_tri

        
    '''    
        # Canny
        edges = cv2.Canny(u8, 50, 150)        
        edge_canny = np.argmax(edges[::-1, :], axis=0)
        edge_canny = edges.shape[0]-edge_canny
        edge_canny[0:round(F/2)][edge_canny[0:round(F/2)]==X]=0
        # plt.plot(edge_canny, color="red", label="Canny")
        
        

        Lap = cv2.Laplacian(u8, cv2.CV_64F, ksize=5)
        Lap = np.abs(Lap)
        edge_lap = np.argmax(edges[::-1, :], axis=0)
        edge_lap = edges.shape[0]-edge_lap
        edge_lap[0:round(F/2)][edge_lap[0:round(F/2)]==X]=0

        # plotting 
        plt.plot(edge_canny, color="red", label="Canny")
        # plt.plot(edge_lap, color="green", label="Laplacian")
        plt.legend()

        # u8 is (X, F) after your transpose
        edge_contour, lev = upper_boundary_contour(u8, q=0.80, smooth_sigma=1.5, monotonic=True, show=False)
        if edge_contour is None:
            # fallback: keep old method or lower the level
            edge_contour, _ = upper_boundary_contour(u8, q=0.70, smooth_sigma=1.5, monotonic=True, show=False)
        else: 
            plt.figure()
            plt.imshow(u8, origin="lower", aspect="auto")
            plt.plot(edge_contour, label="Contour upper envelope", linewidth=2)
            plt.legend()
        penetration[p] = edge_contour  # note: this is in "row" units after transpose
        # penetration[p] = edge_canny
    '''
    plt.show()
        
    '''    
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(find_penetration, td_intensity_maps[p]) : p for p in range(P)}
        for fut in as_completed(futs):
            penetration[futs[fut]] = fut.result()
    '''
    if visualization:
        plt.imshow(td_intensity_maps[0].T, origin="lower", aspect="auto")
        plt.plot(penetration[0], color="red")
    print(f"Penetration calculation completed in {time.time()-start_time:.3f}s")
         
    # for plume_num, intensity2d in enumerate(td_intensity_maps):

        






    # Resetting to 0 before hydraulic delay.
    for p in range(P):
        penetration[p, :hydraulic_delay[p]] = 0.0

    ''''''
    # bw_vids has shape [plume number, frame, rows, cols]
    bw_vids = np.zeros(segments.shape)
    
    start_time = time.time()

    # To be parallelized 
    for i, segment in enumerate(segments):
        # print(i)
        bw_vid = np.zeros(segment.shape).astype(int)
        thres_array = np.zeros(segment.shape[0])
        
        # for j in range(hydraulic_delay[i], avg_peak):
        for j in range(hydraulic_delay[i], len(segment)):
            # bw_vid[j], thres_temp = triangle_binarize_from_float(segment[j], blur=True)
            bw_temp, _ = triangle_binarize_from_float(segment[j], blur=True)
            bw_vid[j] = keep_largest_component(bw_temp)
        
        # thres_array[j:] = thres_array[j]
        # bw_vid[0:hydraulic_delay[i]] = (segment[0:hydraulic_delay[i]] > thres_array[hydraulic_delay[i]]).astype(int)
        # bw_vid[avg_peak:] = (segment[avg_peak:] > thres_array[avg_peak]).astype(int)
        # play_video_cv2(bw_vid*255)
        # play_video_cv2((1-bw_vid)*segment)
        
        # bw_vid = keep_largest_component_nd(bw_vid)

        bw_vids[i] = bw_vid
    print(f"Time elapsed in triangular segmetation for all segments: {time.time()-start_time:.3f}s")
    return bw_vids, penetration






def load() -> np.ndarray:
    path = filedialog.askopenfilename(filetypes=[('Cine','*.cine')])
    try:
        video = load_cine_video(path).astype(np.float32)  # Ensure load_cine_video is defined or imported
        return video
    except Exception as e:
        messagebox.showerror('Error', f'Cannot load video:\n{e}')
        return np.empty((0, 0, 0), dtype=np.float32)

def load_json():
    path = filedialog.askopenfilename(filetypes=[('json','*.json')])
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # process the data
        # for item in data:
            # print(item)
        plumes = int(data['plumes'])
        # offset = float(data['offset'])
        centre = (float(data['centre_x']), float(data['centre_y']))
    return centre, plumes

def pipeline_mie():
    ir_=14; or_=380

    centre, number_of_plumes = load_json()

    video = load()
    # video_raw = video
    # cutting some frames
    video = video[0:100, :,:]

    # mie_multihole_pipeline(video, centre, number_of_plumes)

    video= video**2
    # video_raw = video
    frames, height, width = video.shape
    # play_video_cv2(video)

    #####################################################################
    # Pre Processing
    #####################################################################
    start_time = time.time()

    bkg = np.median(video[:15, :, :], axis = 0)[None, :, :]

    sub_bkg = video - bkg

    sub_bkg_med = median_filter_video_auto(sub_bkg, 5, 3)

    sub_bkg_med[sub_bkg_med<0] = 0

    px_range = np.max(sub_bkg_med, axis=0) - np.min(sub_bkg_med, axis=0)
    range_mask, _ = triangle_binarize_from_float(px_range)
    sub_bkg_med = mask_video(sub_bkg_med, range_mask)
    
    if visualization:
        fig, ax = plt.subplots(2,2, figsize=(13,11))
        im1 = ax[0,0].imshow(px_range, origin="lower", cmap="jet")
        fig.colorbar(im1, ax=ax[0,0])
        ax[0,0].set_title("Range of pixel in all frames")

        data = px_range.ravel()
        hist, edges = np.histogram(data, bins=1000, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2
        ax[0,1].plot(centers, hist, lw=1.2, label="Histogram of pixel range in log scale")
        ax[0,1].set_yscale('log')
        ax[0,1].set_ylim(bottom=1)  # avoid log(0) issues
        ax[0,1].set_xlabel("Range value")
        ax[0,1].set_ylabel("Count (log scale)")
        ax[0,1].set_title("Histogram of pixel range in log scale")

        norm = cv2.normalize(px_range, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        thres_otsu, bw_otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ax[0,1].axvline(thres_otsu/255.0, color="green", linestyle="-", label="Otsu thres")


        

        bw, thres = triangle_binarize_from_float(px_range)

        ax[1,0].imshow(bw)

        ax[0,1].axvline(thres/255.0, color="red", linestyle="--", label="Triangular thres")
        ax[0,1].legend()
        ax[1,1].axvline(thres/255.0, color="red", linestyle="--", label="Triangular thres")
        ax[1,1].plot(centers, hist, lw=1.2, label="Histogram of pixel range in linear scale")

        ax[1,1].set_ylim(bottom=1)  # avoid log(0) issues
        ax[1,1].set_xlabel("Range value")
        ax[1,1].set_ylabel("Count (linear scale)")
        ax[1,1].set_title("Histogram of pixel range in linear scale")
        ax[1,1].axvline(thres_otsu/255.0, color="green", linestyle="-", label="Otsu thres")

        ax[0,1].grid()
        ax[1,1].grid()
        ax[1,1].legend()
    
    if visualization:
        fig, ax = plt.subplots(2, 2, figsize=(13,11))
        ax[0,0].imshow(bkg[0]**0.2*10, origin="lower", cmap="gray")
        ax[0,0].set_title("Background\n temporal median of first 17 frames")
        ax[0,1].imshow(video[30]**0.2*10, origin="lower", cmap="gray")
        ax[0,1].set_title("Frame 30")
        sub_bkg[sub_bkg<0]=0.0
        ax[1,0].imshow(sub_bkg[30]**0.2*10, origin="lower", cmap="gray")
        ax[1,0].set_title("Frame 30 - background")
        ax[1,1].imshow(sub_bkg_med[30]**0.2*10, origin="lower", cmap="gray")
        ax[1,1].set_title("After medium filtering")
        fig.suptitle("Preprocessing steps\n gamma=0.2, gain=1")
        fig.tight_layout()
        fig.show()

    if visualization:
        fig, ax = plt.subplots(2,2, figsize=(18,16))
        ax[0,0].imshow(sub_bkg_med[30], origin="lower", cmap="gray")
        ax[0,1].imshow(sub_bkg_med[30]**2, origin="lower", cmap="gray")
        ax[1,0].imshow(sub_bkg_med[30]**3, origin="lower", cmap="gray")
        ax[1,1].imshow(sub_bkg_med[30]*4, origin="lower", cmap="gray")
        ax[0,0].set_title("Gamma=1")
        ax[0,1].set_title("Gamma=2")
        ax[1,0].set_title("Gamma=3")
        ax[1,1].set_title("Gamma=4")
        fig.suptitle("Problem with selection of arbitrary gamma value.")
        fig.tight_layout()
        fig.show()

    scale = sub_bkg_med.max()

    # Scaling
    sub_bkg_med = sub_bkg_med/scale

    # Gamma
    # sub_bkg_med = sub_bkg_med**3

    centre_x = float(centre[0])
    centre_y = float(centre[1])

    print(f"Preprocessing completed in {time.time()-start_time:.3f}s")

    ############################################
    # Cone Angle 

    start_time = time.time()
    # Cone angle
    bins = 3600
    signal_density_bins, signal, density = angle_signal_density(
        sub_bkg_med, centre_x, centre_y, N_bins=bins
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
    '''
    if visualization:
        arr = np.log1p(signal)
        arr2d = arr if arr.ndim == 2 else arr[None]
        frames, n_bins = arr2d.shape

        bins_ext = np.concatenate((np.linspace(0, 360, bins) - 360, np.linspace(0, 360, bins)))
        arr_ext = np.concatenate((arr2d, arr2d), axis=1)
        sum_ext = np.concatenate((summed_signal, summed_signal))

        X, Y = np.meshgrid(bins_ext, np.arange(frames))

        fig, (ax_heat, ax_contour, ax_signal_sum) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

        im = ax_heat.imshow(
            arr_ext,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[bins_ext[0], bins_ext[-1], 0, frames],
        )
        # fig.colorbar(im, ax=ax_heat, label="Log Signal")
        ax_heat.set_ylabel("Frame index")
        ax_heat.set_title("Angular Signal Density Heatmap")
        ax_heat.set_xlim(-180, 180)

        cont = ax_contour.contourf(X, Y, arr_ext, levels=15, cmap="viridis")
        # fig.colorbar(cont, ax=ax_contour, label="Log Signal")
        ax_contour.set_xlabel("Angle (degrees)")
        ax_contour.set_ylabel("Frame index")
        ax_contour.set_title("Angular Signal Density Contour")
        ax_heat.set_xlim(-180, 180)

        summed  = ax_signal_sum.plot(bins_ext, sum_ext)
        ax_signal_sum.set_xlabel("Angle (degrees)")
        ax_signal_sum.set_ylabel("Intesity")
        ax_signal_sum.set_title("Angular Signal Density Summed in all frames")
        ax_signal_sum.set_xlim(-180, 180)

        angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - offset
        plume_angles = -angles

        if plume_angles is not None:
            for ang in plume_angles:
                for shift in (0, 360):
                    ax_contour.axvline(ang + shift, color="cyan", linestyle="--")
                    ax_heat.axvline(ang + shift, color="cyan", linestyle="--")
                    ax_signal_sum.axvline(ang+shift, color="cyan", linestyle="--")

        plt.tight_layout()
        plt.show()


    

    ##################################################
    # Rotation into horizontal segments
    ##################################################
    start_time = time.time()
    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)

    angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - offset
    mask = generate_plume_mask(ir_, or_, crop[2], crop[3])

    segments = rotate_all_segments_auto(sub_bkg_med, angles, crop, centre, mask=mask)

    segments = np.array(segments)

    
    P, F, R, C = segments.shape
    
    print(f"Rotation to segments completed in {time.time()-start_time:.3f}s")
    # Gaussian suppressor
    # x = np.arange(0, C)
    # pmf = norm.pdf(x, loc=0, scale=C/8)
    # pmf/= np.max(pmf)
    # brightness_scale = 1-pmf+np.min(pmf)
    # segments = segments*brightness_scale[None, None, None, :]

    if visualization:
        fig, ax = plt.subplots(2,2, figsize=(18,16))
        ax[0,0].imshow(segments[0][30], origin="lower", cmap="gray")
        ax[0,1].imshow(segments[0][30]**2, origin="lower", cmap="gray")
        ax[1,0].imshow(segments[0][30]**4, origin="lower", cmap="gray")
        ax[0,0].set_title("Gamma=1")
        ax[0,1].set_title("Gamma=2")
        ax[1,0].set_title("Gamma=4")
        fig.suptitle("Problem with selection of arbitrary gamma value.")
        fig.tight_layout()
        fig.show()

    ###################################################
    # Copute BW videos and penetration
    ###################################################

    bw_vids, penetration = segments_computation(segments, mask)

    penetration_old = np.zeros(penetration.shape)
    col_sum_bw = np.sum(bw_vids, axis=2) >=1
    n_workers = min(os.cpu_count() + 4, P, 32) # type: ignore
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(penetration_bw_to_index, col_sum_bw[p]): p for p in range(col_sum_bw.shape[0])}
        for fut in as_completed(futs):
            penetration_old[futs[fut]] = fut.result()
    
    plt.plot(penetration.T)
    plt.show()
    plt.plot(penetration_old.T)
    plt.show()

    # Comparison
    plt.plot(np.mean(penetration, axis=0), label="Triangular segmentation for column sum")
    plt.plot(np.mean(penetration_old, axis=0), label="Triagular segmentation for each frame")
    plt.xlabel("Frame number")
    plt.ylabel("Penetration [px]")
    plt.title("Comparision of two penetration calculation methods")
    plt.legend()
    plt.show()
    
    play_videos_side_by_side(tuple(((1-bw_vids)*segments)[0:5]))
    play_videos_side_by_side(tuple(((1-bw_vids)*segments)[5:]))

    # 100 times gain
    gain = 100
    play_videos_side_by_side(tuple((gain*(1-bw_vids)*segments)[5:10]), intv=70)
    play_videos_side_by_side(tuple((gain*(1-bw_vids)*segments)[5:]), intv=70)

    # 1000_times gain
    gain = 1000
    play_videos_side_by_side(tuple((gain*(1-bw_vids)*segments)[5:10]), intv=70)
    play_videos_side_by_side(tuple((gain*(1-bw_vids)*segments)[5:]), intv=70)

    # Area demostration
    area_all_segmets = np.sum(np.sum(bw_vids, axis=3), axis=2)
    plt.plot(area_all_segmets.T)

    # Cone Angle demostration

    bw_ang, thres = triangle_binarize_from_float(signal, blur=True)
    # plot_angle_signal_density(np.linspace(0, 360, bins), signal*bw_ang, log=True)

    # Show that offset found by FFT calibrates the plume axis
    # plot_angle_signal_density(np.linspace(0, 360, bins) - offset, signal*bw_ang, log=True)
    
    # Cone Angle
    shift_bins = int(round(offset/360*bins))
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
    
    # bw_ang_closed = np.roll(bw_shifted, shift_bins, axis=1)
    # plot_angle_signal_density(np.linspace(0, 360, bins) - offset, signal*bw_ang_closed, log=True)

    print("Plume widths per frame (degrees):")
    plt.plot(plume_widths.T)
    # for p, w in enumerate(plume_widths):
        # print(f"Plume {p}:", w)
    
    # BW boundaries
    boundaries = bw_boundaries_xband_split(bw_vids, penetration_old, lo=0.0, hi=1.0)
    
    # boundaries = bw_boundaries_xband_split(bw_vids, penetration_old, ...)
    # Play repetition i overlaid with red boundary:
    i = 4
    play_video_with_boundaries_cv2(segments[i], boundaries[i], gain=1.0, binarize=False, intv=170)

    # segments: shape [P][F][H][W] (or ndarray with that shape)
    # boundaries: shape [P][F][2][N,2]   # like boundaries[p][f][0], boundaries[p][f][1]
    play_segments_with_boundaries(segments, boundaries, p=0, gain=1.0, intv=170, origin='lower')


def main(visualization=True):
    number_of_plumes = 1
    centre, _ = load_json()
    video = load()

    max = video.max()
    video /=max
    
    centre_x = float(centre[0])
    centre_y = float(centre[1])

    '''
    ############################################
    # Cone Angle 

    start_time = time.time()
    # Cone angle
    bins = 3600
    signal_density_bins, signal, density = angle_signal_density(
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

    if visualization:
        arr = np.log1p(signal)
        arr2d = arr if arr.ndim == 2 else arr[None]
        frames, n_bins = arr2d.shape

        bins_ext = np.concatenate((np.linspace(0, 360, bins) - 360, np.linspace(0, 360, bins)))
        arr_ext = np.concatenate((arr2d, arr2d), axis=1)
        sum_ext = np.concatenate((summed_signal, summed_signal))

        X, Y = np.meshgrid(bins_ext, np.arange(frames))

        fig, (ax_heat, ax_contour, ax_signal_sum) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

        im = ax_heat.imshow(
            arr_ext,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[bins_ext[0], bins_ext[-1], 0, frames],
        )
        # fig.colorbar(im, ax=ax_heat, label="Log Signal")
        ax_heat.set_ylabel("Frame index")
        ax_heat.set_title("Angular Signal Density Heatmap")
        ax_heat.set_xlim(-180, 180)

        cont = ax_contour.contourf(X, Y, arr_ext, levels=15, cmap="viridis")
        # fig.colorbar(cont, ax=ax_contour, label="Log Signal")
        ax_contour.set_xlabel("Angle (degrees)")
        ax_contour.set_ylabel("Frame index")
        ax_contour.set_title("Angular Signal Density Contour")
        ax_heat.set_xlim(-180, 180)

        summed  = ax_signal_sum.plot(bins_ext, sum_ext)
        ax_signal_sum.set_xlabel("Angle (degrees)")
        ax_signal_sum.set_ylabel("Intesity")
        ax_signal_sum.set_title("Angular Signal Density Summed in all frames")
        ax_signal_sum.set_xlim(-180, 180)

        angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - offset
        plume_angles = -angles

        if plume_angles is not None:
            for ang in plume_angles:
                for shift in (0, 360):
                    ax_contour.axvline(ang + shift, color="cyan", linestyle="--")
                    ax_heat.axvline(ang + shift, color="cyan", linestyle="--")
                    ax_signal_sum.axvline(ang+shift, color="cyan", linestyle="--")

        plt.tight_layout()
        plt.show()
    '''
    angles = np.array([-120.0])
    offset = 0
    ir_ =0
    or_ = 300
    
    ##################################################
    # Rotation into horizontal segments
    ##################################################
    start_time = time.time()

    rotated_raw = rotate_video_auto(video, -125)
    rotated = rotated_raw[:, 50:400, :]

    
    F, H, W = rotated.shape
    
    print(f"Rotation to segments completed in {time.time()-start_time:.3f}s")

    play_video_cv2(rotated*100)
    
    td_intensity_map = np.sum(rotated, axis=1).squeeze()
    
    bw, thres = triangle_binarize_from_float(td_intensity_map)

    fig, ax = plt.subplots(2, 2)

    ax[0,0].imshow(td_intensity_map, origin="lower", cmap="jet", label="T-D intensity map" )

    ax[0,1].imshow(np.log(td_intensity_map+1e-9), origin="lower", cmap="jet", label="T-D intensity map in log scale")
    

    data = td_intensity_map.ravel()
    hist, edges = np.histogram(data, bins=1000, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    

    
    ax[1,0].plot(centers, hist, lw=1.2, label="Histogram of T-D intensity map in log scale")
    ax[1,0].axvline(thres/255.0)
    ax[1,0].set_yscale('log')
    ax[1,0].set_ylim(bottom=1)  # avoid log(0) issues
    ax[1,0].set_xlabel("Range value")
    ax[1,0].set_ylabel("Count (log scale)")
    ax[1,0].set_title("Histogram of pixel range in log scale")

    ax[1,1].imshow(np.log(td_intensity_map+1e-9)*bw, origin="lower", cmap="jet", label="Mask * log(map)")

    plt.show()   



if __name__  == '__main__':
    main(visualization=False)
    # pipeline_mie()
