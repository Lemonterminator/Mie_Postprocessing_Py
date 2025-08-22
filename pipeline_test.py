from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *
from mie_postprocessing.video_filters import *
from mie_postprocessing.functions_bw import *
from sklearn.preprocessing import normalize
from tkinter import ttk, filedialog, messagebox, colorchooser
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import convolve2d
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc
import json
from pathlib import Path
import time

from skimage import measure
from scipy.ndimage import binary_erosion, generate_binary_structure

global visualization 
visualization = True

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
    pen_2d, _ = triangle_binarize_from_float(intensity2d, blur=True)
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


    # Converting intensity maps to penetration
    start_time = time.time()
    n_workers = min(os.cpu_count() + 4, P, 32) # type: ignore
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(find_penetration, td_intensity_maps[p]) : p for p in range(P)}
        for fut in as_completed(futs):
            penetration[futs[fut]] = fut.result()
    print(f"Penetration calculation completed in {time.time()-start_time:.3f}s")
         
    # for plume_num, intensity2d in enumerate(td_intensity_maps):

        


    # Plot of energy by summing again over cols (summing each image)
    energies = np.sum(td_intensity_maps, axis=2)

    peak_brightness_frames = np.argmax(energies, axis=1)
    avg_peak = round(np.mean(peak_brightness_frames))

    multiplier = 100
    thres_derivative = multiplier*energies[:,0]/energies[:,avg_peak]
    rows = segments.shape[2]; cols = segments.shape[3]
    near_nozzle_energies = np.sum(np.sum(segments[:, :avg_peak, round(rows*3/7):round(rows*4/7), :round(cols/10)], axis=3), axis=2)
    dE1dts = np.diff(near_nozzle_energies[:, 0:avg_peak], axis=1)
    masks = dE1dts > (thres_derivative*np.max(dE1dts, axis=1))[:,None]
    hydraulic_delay = masks.argmax(axis=1) + 1

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
            bw_vid[j], thres_temp = triangle_binarize(segment[j], blur=True)
            thres_array[j] = thres_temp/255.0
        
        
        # thres_array[j:] = thres_array[j]
        # bw_vid[0:hydraulic_delay[i]] = (segment[0:hydraulic_delay[i]] > thres_array[hydraulic_delay[i]]).astype(int)
        # bw_vid[avg_peak:] = (segment[avg_peak:] > thres_array[avg_peak]).astype(int)
        # play_video_cv2(bw_vid*255)
        # play_video_cv2((1-bw_vid)*segment)
        bw_vid = keep_largest_component_nd(bw_vid)
        bw_vids[i] = bw_vid
    print(f"Time elapsed in triangular segmetation for all segments: {time.time()-start_time:.3f}s")
    return bw_vids, penetration

def load() -> np.ndarray:
    path = filedialog.askopenfilename(filetypes=[('Cine','*.cine')])
    try:
        video = load_cine_video(path).astype(np.float32)/4096  # Ensure load_cine_video is defined or imported
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

def main():
    ir_=14; or_=380

    centre, number_of_plumes = load_json()

    video = load()
    # video_raw = video
    # cutting some frames
    video = video[:60, :,:]
    # video_raw = video
    frames, height, width = video.shape
    # play_video_cv2(video)

    #####################################################################
    # Pre Processing
    #####################################################################
    start_time = time.time()

    bkg = np.median(video[:17, :, :], axis = 0)[None, :, :]

    sub_bkg = video - bkg

    sub_bkg_med = median_filter_video_auto(sub_bkg, 5, 3)

    sub_bkg_med[sub_bkg_med<0] = 0

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
    plt.plot(np.mean(penetration, axis=0))
    plt.plot(np.mean(penetration_old, axis=0))
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
    plot_angle_signal_density(np.linspace(0, 360, bins), signal*bw_ang, log=True)

    # Show that offset found by FFT calibrates the plume axis
    plot_angle_signal_density(np.linspace(0, 360, bins) - offset, signal*bw_ang, log=True)
    
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
    
    bw_ang_closed = np.roll(bw_shifted, shift_bins, axis=1)
    plot_angle_signal_density(np.linspace(0, 360, bins) - offset, signal*bw_ang_closed, log=True)

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

if __name__  == '__main__':
    main()