from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *
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
        # plumes = int(data['plumes'])
        # offset = float(data['offset'])
        centre = (float(data['centre_x']), float(data['centre_y']))
    return centre

def main():
    centre = load_json()

    video = load()
    frames, height, width = video.shape
    # play_video_cv2(video)

    bkg = np.median(video[:20, :, :], axis = 0)

    sub_bkg = video - bkg[None, :, :]


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
    video_lap_med = median_filter_video_cuda(video_lap, 3, 3, 3)
    play_videos_side_by_side((video_lap_med, video_lap, sub_bkg, video))

    bins, signal, density = angle_signal_density(video_lap_med, centre[0], centre[1], 3600)
    plot_angle_signal_density(bins, signal, log=True, plume_angles=None)
    
    bw, thres = triangle_binarize(signal)
    plt.imshow(bw, origin="lower")


if __name__ == '__main__':
    main()