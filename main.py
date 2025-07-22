from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *

import matplotlib.pyplot as plt
import subprocess
from scipy.signal import convolve2d
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc
import json
from pathlib import Path

global parent_folder
global plumes 
global offset
global centre

parent_folder = r"G:\Master_Thesis\BC20220627 - Heinzman DS300 - Mie Top view\Cine\Interest"

# Directory containing mask images and numpy files
DATA_DIR = Path(__file__).resolve().parent / "data"

# Define a semaphore with a limit on concurrent tasks
SEMAPHORE_LIMIT = 2  # Adjust this based on your CPU capacity
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def play_video_cv2_async(video, gain=1, binarize=False, thresh=0.5, intv=17):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, play_video_cv2, video, gain, binarize, thresh, intv)

def MIE_pipeline(video, number_of_plumes, offset, centre):
    # video[video<0.05]=0
    video_histogram_with_contour(video, bins=100, exclude_zero=True, log=True)

    foreground = subtract_median_background(video, frame_range=slice(0, 30))

    centre_x = float(centre[0])
    centre_y = float(centre[1])

    # Cone angle
    signal_density_bins, signal, density = angle_signal_density(foreground, centre_x, centre_y, N_bins=3600)

    plot_angle_signal_density(signal_density_bins, signal)


    
    # play_video_cv2(foreground, intv=17)
    '''
    gamma = foreground ** 2 # gamma correction
    gamma[gamma < 2e-2 ] = 0  # thresholding
    gain = gamma * 5  # gain correction
    gain[gain > 1] = 1  # limit gain to 1
    # play_video_cv2(gain, intv=17)

    print("Gain correction has range from %f to %f" % (gain.min(), gain.max()))
    '''
    
    
    gamma = foreground**2
    video_histogram_with_contour(gamma, bins=100, exclude_zero=True, log=True)

    gain = gamma * 5
    video_histogram_with_contour(gain, bins=100, exclude_zero=True, log=True) 
    
    gain[gain < 1.5e-2] = 0 # thresholding
    gain[gain > 0] = 1
    # centre = (384.9337805142379, 382.593916979227)
    # crop = (round(centre[0]), round(centre[1]- 768/16), round(768/2), round(768/8))

    ir_ = 14
    or_ = 380

    # Generate the crop rectangle based on the plume parameters
    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)

    # offset = 2
    angles = np.linspace(0, 360, number_of_plumes, endpoint=False) + offset
    mask = generate_plume_mask(ir_, or_, crop[2], crop[3])

    
    start_time = time.time()
    

    
    segments = []

    # Multithreaded rotation and cropping
    with ThreadPoolExecutor(max_workers=min(len(angles), os.cpu_count() or 1)) as exe:
        future_map = {
            exe.submit(
                rotate_and_crop, gain, angle, crop, centre,
                is_video=True, mask=mask
            ): idx for idx, angle in enumerate(angles)
        }
        segments_with_idx = []
        for fut in as_completed(future_map):
            idx = future_map[fut]
            result = fut.result()
            segments_with_idx.append((idx, result))
        # Sort by index to preserve order
        segments_with_idx.sort(key=lambda x: x[0])
        segments = [seg for idx, seg in segments_with_idx]
    
    elapsed_time = time.time() - start_time
    print(f"Computing all rotated segments finished in {elapsed_time:.2f} seconds.")
    
    
    # Free intermediate arrays to reduce peak memory usage
    del foreground, gain, gamma
    gc.collect()

    '''
    # Stacking the segments into a 4D array
    segments = [seg for seg in segments if seg is not None]
    if not segments:
        raise ValueError("No valid segments to stack.")
    '''
    segments = np.stack(segments, axis=0)

    average_segment = np.mean(segments, axis=0) # Average across the segments

    '''
    # SSIM 
    start_time = time.time()
    ssim_matrix = compute_ssim_segments(segments,average_segment)    
    elapsed_time = time.time() - start_time
    print(f"Computing all SSIM finished in {elapsed_time:.2f} seconds.")

    plt.plot(ssim_matrix.transpose())
    # plt.show()
    '''

    # Kmeans 
    '''
    start_time = time.time()
    for segment in segments:
        labels = kmeans_label_video(segment, k=2)
        playable = labels_to_playable_video(labels, k=2)
        # play_videos_side_by_side([segment, playable], intv=34)
    
    elapsed_time = time.time() - start_time
    print(f"Computing all Kmeans labels finished in {elapsed_time:.2f} seconds.")
    '''
    for segment in segments:
        time_distance_intensity = np.sum(segment, axis=1)  # Force computation of the segment to avoid lazy evaluation issues
        plt.imshow(time_distance_intensity,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                )
        plt.show()
            # Cone angle
        signal_density_bins, signal, density = angle_signal_density(segment, 0.0, segment.shape[1]/2.0, N_bins=180)

        plot_angle_signal_density(signal_density_bins, signal)


async def main():

    subfolders = get_subfolder_names(parent_folder)  # Ensure get_subfolder_names is defined or imported

    
    chamber_mask_path = DATA_DIR / "chamber_mask.npy"
    test_mask_path = DATA_DIR / "test_mask.npy"
    region_unblocked_path = DATA_DIR / "region_unblocked.npy"

    if chamber_mask_path.exists():
        chamber_mask = np.load(chamber_mask_path)
    else:
        subprocess.run(["python", "masking.py"], check=True)
        chamber_mask = np.load(chamber_mask_path)

    if test_mask_path.exists():
        test_mask = np.load(test_mask_path) == 0

    if region_unblocked_path.exists():
        region_unblocked = np.load(region_unblocked_path)
    


    for subfolder in subfolders:
        print(subfolder)
    
        # Specify the directory path
        directory_path = Path(parent_folder + "\\" + subfolder)
    
        # Get a list of all files in the directory
        files = [file for file in directory_path.iterdir() if file.is_file()]

        for file in files:
            if file.name == 'config.json':
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # process the data
                    # for item in data:
                        # print(item)
                    plumes = int(data['plumes'])
                    offset = float(data['offset'])
                    centre = (float(data['centre_x']), float(data['centre_y']))

        # print(files)
        for file in files:
            if file.suffix == '.cine':
                print("Procssing:", file)

                video = load_cine_video(file).astype(np.float32)/4096  # Ensure load_cine_video is defined or imported
                frames, height, width = video.shape
                # video = video.astype(float)
                # play_video_cv2(video)
                # video = video**2

                if "Shadow" in file.name:
                    continue

                    # Angle of Rotation
                    rotation = 45

                    # Strip cutting
                    x_start = 1
                    x_end = -1
                    y_start = 150
                    y_end = -250

                    
                    '''
                    start_time = time.time()
                    RT = rotate_video(video, rotation)
                    elapsed_time = time.time() - start_time
                    print(f"Rotating video with CPU finished in {elapsed_time:.2f} seconds.")
                    '''
                    
                    start_time = time.time()
                    RT = rotate_video_cuda(video, rotation)
                    elapsed_time = time.time() - start_time
                    print(f"Rotating video with GPU finished in {elapsed_time:.2f} seconds.")
                    
                    



                    # frame, y, x
                    strip = RT[15:400, y_start:y_end, x_start:x_end]
                    strip = strip.astype(float)
                    
                    
                    masked_strip = strip

                    # play_video_cv2(masked_strip)

                    lap = np.array([[1,1,1],[1,-8,1],[1,1,1]], dtype=float)
                    
                    HP = filter_video_fft(strip, lap, mode='same')

                    avg = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float)

                    HP_avg = filter_video_fft(HP, avg, mode='same')
                    # play_video_cv2(HP_avg)
                    await play_video_cv2_async(HP_avg)

                        
                    # STD filtering
                    # parameters:
                    # Standard deviation filter window size
                    # Downsampling factor
                    # std_video = stdfilt_video(strip, std_size, pool_size)

                    '''                
                    std_video = stdfilt_video_parallel_optimized(masked_strip, std_size=3, pool_size=2)

                    bw_std = binarize_video_global_threshold(std_video, method='fixed', thresh_val=2E2)

                    bw_std_filled = fill_video_holes_parallel(bw_std)

                    '''

                    '''                
                    start_time = time.time()
                    velocity_field = compute_optical_flow(strip)
                    elapsed_time = time.time() - start_time
                    print(f"OFE with CPU finished in {elapsed_time:.2f} seconds.")
                    '''

                    HP_delay = np.zeros(HP_avg.shape, dtype=float)

                    HP_delay[1:-1, :, :] = HP_avg[0:-2, :,:]

                    HP_res = np.abs(HP_delay-HP_avg)
                    # import numpy as np
                    import cupy as cp
                    import matplotlib.pyplot as plt

                    # 2) Upload to GPU
                    vol_gpu = cp.array(HP_res)
                    # 3) Plan and execute 3D complex‐to‐complex FFT in place
                    #    (cuFFT automatically picks the fastest algorithm)
                    vol_fft = cp.fft.fftn(vol_gpu, axes=(0,1,2))

                    # 4) Shift zero‐frequency to center (optional)
                    vol_fft = cp.fft.fftshift(vol_fft, axes=(0,1,2))

                    # 5) Compute magnitude (abs) and bring back to CPU
                    mag_gpu = cp.abs(vol_fft)
                    mag = cp.asnumpy(mag_gpu)
                    nx, ny, nz = HP_res.shape
                    # 6) Visualize three orthogonal slices
                    fig, axes = plt.subplots(1,3, figsize=(12,4))
                    slices = [
                        mag[nx//2, :, :],  # Y–Z at center X
                        mag[:, ny//2, :],  # X–Z at center Y
                        mag[:, :, nz//2],  # X–Y at center Z
                    ]
                    titles = ['Slice X=mid','Slice Y=mid','Slice Z=mid']
                    for ax, slc, title in zip(axes, slices, titles):
                        im = ax.imshow(np.log1p(slc), origin='lower')
                        ax.set_title(title)
                        fig.colorbar(im, ax=ax, fraction=0.046)
                    plt.tight_layout()
                    plt.show()






                    # await play_video_cv2_async(HP_res/1024)

                                    
                    start_time = time.time()
                    velocity_field = compute_optical_flow_cuda(HP_res)
                    elapsed_time = time.time() - start_time
                    print(f"OFE with GPU finished in {elapsed_time:.2f} seconds.")

                    scalar_velocity_field = compute_flow_scalar_video(velocity_field, multiplier=1, y_scale=1)
                    

                    start_time = time.time()
                    scalar_velocity_field_med = median_filter_video_cuda(HP_res, 5, 5)
                    elapsed_time = time.time() - start_time
                    print(f"Medfilt with GPU finished in {elapsed_time:.2f} seconds.")

                    await play_video_cv2_async(scalar_velocity_field_med/1024)


                    bw_flow = mask_video(binarize_video_global_threshold(scalar_velocity_field_med, method='otsu'), chamber_mask)

                    await play_video_cv2_async(bw_flow, gain=255)
        
                    
                                    
                    start_time = time.time()
                    bw_flow_filled = fill_video_holes_parallel(bw_flow)
                    elapsed_time = time.time() - start_time
                    print(f"Hole-filling with CPU finished in {elapsed_time:.2f} seconds.")
                    

                    start_time = time.time()
                    bw_flow_filled = fill_video_holes_gpu(bw_flow)
                    elapsed_time = time.time() - start_time
                    print(f"Hole-filling with GPU finished in {elapsed_time:.2f} seconds.")

                    await play_video_cv2_async(bw_flow_filled, gain=255, binarize=True)


                    # results = compute_boundaries_parallel_all(bw_flow_filled)
                    

                    # results = compute_boundaries_parallel_all(bw_std_filled)


                    # play_video_cv2(strip)
                    # play_video_cv2(masked_strip)
                    # play_video_cv2(bw_std, 4)
                    # play_video_cv2(std_video/1E3)
                    # masked_std_video = mask_video(std_video, chamber_mask)
                    # play_video_cv2(masked_std_video/1E3)
                    # play_video_cv2(scalar_velocity_field, 1)
                    # play_video_cv2(bw_flow_filled, 10)



                    # ... after computing `results = compute_boundaries_parallel_all(bw_flow)` ...
                    
                    plt.ion()
                    fig, ax = plt.subplots()
                    im = ax.imshow(masked_strip[0], cmap='gray')
                    ax.set_title("Frame 0 Boundaries")
                    ax.axis('off')

                    for idx, res in enumerate(results):
                        frame = masked_strip[idx]
                        im.set_data(frame)
                        ax.set_title(f"Frame {idx} Boundaries")
                        
                        # 1) Remove old contour lines
                        #    This clears any Line2D objects currently on the axes.
                        for ln in ax.lines:
                            ln.remove()

                        '''
                        # Plotting all countors                    
                        # 2) Plot every contour for every component
                        for comp in res.components:
                            for contour in comp.boundaries:
                                y, x = contour[:, 0], contour[:, 1]
                                ax.plot(x, y, '-r', linewidth=2)
                        


                        # Countor of the biggest area
                        # 2) If any components found, plot only the largest one
                        if res.components:
                            # Find component with max area
                            largest_comp = max(res.components, key=lambda c: c.area)
                            
                            # Plot its first contour (there may be multiple loops)
                            if largest_comp.boundaries:
                                contour = largest_comp.boundaries[0]
                                y, x = contour[:, 0], contour[:, 1]
                                ax.plot(x, y, '-r', linewidth=2)
                    '''
                        # Longest contour
                        # 2) Find the single longest contour across all components
                        longest_contour = None
                        max_len = 0
                        for comp in res.components:
                            for contour in comp.boundaries:
                                n_pts = contour.shape[0]
                                if n_pts > max_len:
                                    max_len = n_pts
                                    longest_contour = contour
                        if longest_contour is not None:
                            y, x = longest_contour[:, 0], longest_contour[:, 1]
                            ax.plot(x, y, '-r', linewidth=2)

                        # 3) Redraw
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.1)
                    plt.close('all')
                elif "OH" in file.name:

                    continue

                    RT = rotate_video(video, -45)
                    strip = RT[0:150, 250:550, :]
                    LP_filtered = Gaussian_LP_video(strip, 40)
                    med = median_filter_video(LP_filtered, 5, 5)
                    

                    BW = binarize_video_global_threshold(med,"fixed", 800)

                    play_video_cv2(strip*10)
                    play_video_cv2(BW*255.0)

                    TD_map = calculate_TD_map(strip)
                    area = calculate_bw_area(BW)
                    
                    '''
                    plt.figure()
                    plt.imshow(TD_map, cmap='jet', aspect='auto')
                    plt.title("Average Time–Distance Map")
                    plt.xlabel("Time (frames)")
                    plt.ylabel("Distance (pixels)")
                    plt.colorbar(label="Sum Intensity")
                    plt.show()

                    plt.figure(figsize=(10, 4))
                    plt.plot(area, color='blue')
                    plt.xlabel("Frame")
                    plt.ylabel("Area (white pixels)")
                    plt.title("Area Over Time")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()'''
                else:
                    
                    # gamma correcetion of video
                    # mie_video = mask_video(video[15:150,:,:], chamber_mask)
                    mie_video = mask_video(video, ~chamber_mask)

                    MIE_pipeline(mie_video, plumes, offset, centre)
                    

if __name__ == '__main__':
    
    from multiprocessing import freeze_support
    freeze_support()
    import asyncio, time

    start = time.time()
    asyncio.run(main())
    print(f"Total elapsed: {time.time() - start:.2f}s")
    
