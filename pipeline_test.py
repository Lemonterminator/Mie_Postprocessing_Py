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

def segments_computation(segments):
    # segmnets has shape [plume number, frame, rows, cols]

    # Map of intensity and time by summing over rows
    td_intensity_maps = np.sum(segments, axis=2)
    # Plot of energy by summing again over cols (summing each image)
    energies = np.sum(td_intensity_maps, axis=2)

    peak_brightness_frames = np.argmax(energies, axis=1)
    avg_peak = round(np.mean(peak_brightness_frames))

    multiplier = 100
    thres_derivative = multiplier*energies[:,0]/energies[:,avg_peak]
    rows = segments.shape[2]; cols = segments.shape[3]
    near_nozzle_energies = np.sum(np.sum(segments[:, :avg_peak, round(rows/3):round(rows*2/3), :round(cols/4)], axis=3), axis=2)
    dE1dts = np.diff(near_nozzle_energies[:, 0:avg_peak], axis=1)
    masks = dE1dts > (thres_derivative*np.max(dE1dts, axis=1))[:,None]
    hydraulic_delay = masks.argmax(axis=1) + 1

    ''''''
    bw_vids = np.zeros(segments.shape)
    start_time = time.time()
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
        bw_vids[i] = bw_vid
    print(f"Time elapsed in triangular segmetation for all segments: {time.time()-start_time:.2f}")
    return bw_vids

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
    video_raw = video
    # cutting some frames
    video = video[:60, :,:]
    video_raw = video
    frames, height, width = video.shape
    # play_video_cv2(video)

    bkg = np.median(video[:9, :, :], axis = 0)

    sub_bkg = video - bkg[None, :, :]

    sub_bkg_med = median_filter_video_auto(sub_bkg, 3, 3)

    sub_bkg_med[sub_bkg_med<0] = 0

    scale = sub_bkg_med.max()

    # Scaling
    sub_bkg_med = sub_bkg_med/scale

    # Gamma
    sub_bkg_med = sub_bkg_med**3

    centre_x = float(centre[0])
    centre_y = float(centre[1])

    # Cone angle
    signal_density_bins, signal, density = angle_signal_density(
        sub_bkg_med, centre_x, centre_y, N_bins=3600
    )

    # Estimate optimal rotation offset using FFT of the summed angular signal
    summed_signal = signal.sum(axis=0)
    fft_vals = np.fft.rfft(summed_signal)
    if number_of_plumes < len(fft_vals):
        phase = np.angle(fft_vals[number_of_plumes])
        offset = (-phase / number_of_plumes) * 180.0 / np.pi
        offset %= 360.0
        offset = min(offset, (offset-360), key=abs)
        print(f"Estimated offset from FFT: {offset:.2f} degrees")

    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)

    angles = np.linspace(0, 360, number_of_plumes, endpoint=False) - offset
    mask = generate_plume_mask(ir_, or_, crop[2], crop[3])

    


    segments = rotate_all_segments_auto(sub_bkg_med, angles, crop, centre, mask=mask)

    segments = np.array(segments)


    bw_vids = segments_computation(segments)
    

    '''

    # play_video_cv2(sub_bkg_med)
    # play_videos_side_by_side((video, sub_bkg, sub_bkg_med), intv=170)
    bw_vid = np.zeros(sub_bkg_med.shape)
    thres_array = np.zeros(sub_bkg_med.shape[0])
    for i in range(sub_bkg.shape[0]):
        bw, thres = triangle_binarize(sub_bkg_med[i])
        bw_vid[i] = bw
        thres_array[i] = thres
    play_video_cv2(bw_vid)
    # plt.plot(thres_array)
    index = np.argmin(thres_array)
    min_val = thres_array[index]
    thres_array[:index] = min_val
    plt.plot(thres_array)
    plt.show()
    for i in range(0, index):
        bw_vid[i] = sub_bkg_med[i]> thres_array[i]/255.0
    '''



    
    '''
    lap5x5 = np.array([
        [-1, -3, -4, -3, -1],
        [-3,  0,  6,  0, -3],
        [-4,  6, 20,  6, -4],
        [-3,  0,  6,  0, -3],
        [-1, -3, -4, -3, -1],
    ], dtype=np.float32)

    # video_lap = filter_video_fft(video, lap5x5)
    # play_videos_side_by_side((video_lap, video))

    video_lap = filter_video_fft(sub_bkg, lap5x5)
    video_lap= np.abs(video_lap)
    video_lap_med = median_filter_video_auto(video_lap, 3, 3, 3)
    play_videos_side_by_side((video_lap_med, video_lap, sub_bkg, video))

    bins, signal, density = angle_signal_density(video_lap_med, centre[0], centre[1], 3600)
    plot_angle_signal_density(bins, signal, log=True, plume_angles=None)
    
    bw, thres = triangle_binarize(signal)
    plt.imshow(bw, origin="lower")

    '''
if __name__ == '__main__':
    main()