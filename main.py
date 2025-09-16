from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *
from mie_postprocessing.video_filters import *
from mie_postprocessing.functions_bw import *
from mie_postprocessing.video_playback import *
from mie_multihole_pipeline import *
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import convolve2d
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import gc
import json
from pathlib import Path
from mie_postprocessing.async_npz_saver import AsyncNPZSaver

global parent_folder
global plumes
global offset
global centre
global hydraulic_delay
global gain 
global gamma
global testpoint_name
global video_name
global ir_ 
global or_


# Define the parent folder and other global variables
parent_folder = r"C:\Users\LJI008\OneDrive - Wärtsilä Corporation\Documents\BC20241010_HZ_Nozzle5\Cine"
hydraulic_delay = 11  # Hydraulic delay in frames, adjust as needed
gain = 3 # Gain factor for video processing
gamma = 1.3  # Gamma correction factor for video processing

# Inner and outer radius (in pixels) for cropping the images
# ir_ = 14 # DS300
ir_ = 11 # Nozzle 1
or_ = 380


# Directory containing mask images and numpy files
DATA_DIR = Path(__file__).resolve().parent / "data"

# Define a semaphore with a limit on concurrent tasks
SEMAPHORE_LIMIT = 8  # Adjust this based on your CPU capacity
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def play_video_cv2_async(video, gain=1, binarize=False, thresh=0.5, intv=17):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, play_video_cv2, video, gain, binarize, thresh, intv)


def numeric_then_alpha_key(p: Path):
    """
    Sort numerically by the first integer in the stem if present; otherwise
    fall back to case-insensitive alphabetical by full name.
    """
    m = re.search(r'\d+', p.stem)
    if m:
        return (0, int(m.group(0)))          # group 0 = first number
    else:
        return (1, p.name.lower())           # non-numeric go after (or before if you swap 0/1)
    

        

        
        


async def main():

    subfolders = get_subfolder_names(parent_folder)  # Ensure get_subfolder_names is defined or imported

    # Initialize background NPZ saver for penetration outputs
    saver = AsyncNPZSaver(max_workers=2)

    
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
    
    # General folder for saving results
    save_path = Path(parent_folder).parts[-2]
    try:
        os.mkdir(save_path)
    except FileExistsError:
        print(f"Directory {save_path} already exists. Using existing directory.") 

    try:
        for subfolder in subfolders:
            print(subfolder)
        
            # Specify the directory path
            directory_path = Path(parent_folder + "\\" + subfolder)
            save_path_subfolder = Path(save_path) / subfolder
            try:
                os.mkdir(save_path_subfolder)
            except FileExistsError:
                print(f"Directory {save_path_subfolder} already exists. Using existing directory.")

            # Get a list of all files in the directory
            files = [file for file in directory_path.iterdir() if file.is_file()]
            files = sorted(files, key=numeric_then_alpha_key)  

            for file in files:
                if file.name == 'config.json':
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # process the data
                        # for item in data:
                            # print(item)
                        number_of_plumes = int(data['plumes'])
                        # offset = float(data['offset'])
                        centre = (float(data['centre_x']), float(data['centre_y']))

            # print(files)
            for file in files:
                if file.suffix == '.cine':
                    testpoint_name = directory_path.stem
                    video_name = file.stem
        
                    print("Procssing:", file.parts[-3], "/", file.parts[-2], "/", file.parts[-1])
                    # start_time = time.time()
                    video = load_cine_video(file, frame_limit=50).astype(np.float32)/4096  # Ensure load_cine_video is defined or imported
                    frames, height, width = video.shape

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
                        RT = rotate_video_auto(video, rotation)
                        elapsed_time = time.time() - start_time
                        print(f"Rotating video finished in {elapsed_time:.2f} seconds.")
                        
                        



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
                        # Attempt 3D FFT visualization with CuPy if available.
                        try:
                            import cupy as cp  # type: ignore
                            import matplotlib.pyplot as plt

                            vol_gpu = cp.asarray(HP_res)
                            vol_fft = cp.fft.fftn(vol_gpu, axes=(0, 1, 2))
                            vol_fft = cp.fft.fftshift(vol_fft, axes=(0, 1, 2))

                            mag_gpu = cp.abs(vol_fft)
                            mag = cp.asnumpy(mag_gpu)
                            nx, ny, nz = HP_res.shape
                            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                            slices = [
                                mag[nx // 2, :, :],
                                mag[:, ny // 2, :],
                                mag[:, :, nz // 2],
                            ]
                            titles = ['Slice X=mid', 'Slice Y=mid', 'Slice Z=mid']
                            for ax, slc, title in zip(axes, slices, titles):
                                im = ax.imshow(np.log1p(slc), origin='lower')
                                ax.set_title(title)
                                fig.colorbar(im, ax=ax, fraction=0.046)
                            plt.tight_layout()
                            plt.show()
                        except Exception as exc:
                            print(f"CuPy 3D FFT visualization skipped: {exc}")






                        # await play_video_cv2_async(HP_res/1024)

                                        
                        start_time = time.time()
                        velocity_field = compute_optical_flow_cuda(HP_res)
                        elapsed_time = time.time() - start_time
                        print(f"OFE with GPU finished in {elapsed_time:.2f} seconds.")

                        scalar_velocity_field = compute_flow_scalar_video(velocity_field, multiplier=1, y_scale=1)
                        

                        start_time = time.time()
                        scalar_velocity_field_med = median_filter_video_auto(HP_res, 5, 5)
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

                        centre_x = float(centre[0])
                        centre_y = float(centre[1])

                        
                        segments, penetration, cone_angle_AngularDensity, bw_vids, boundaries, penetration_old = mie_multihole_pipeline(
                            video, centre, number_of_plumes, 
                            gamma=gamma, binarize_video=False, 
                            plot_on=True
                            )

                        SSIM = False
                        if SSIM:
                            average_segment = np.mean(segments, axis=0) # Average across the segments
                            data = {'average_segment': average_segment}
                            # SSIM 
                            start_time = time.time()
                            ssim_matrix = compute_ssim_segments(segments,average_segment)    
                            elapsed_time = time.time() - start_time
                            print(f"Computing all SSIM finished in {elapsed_time:.2f} seconds.")

                            # plt.plot(ssim_matrix.transpose())
                            # plt.show()
                            data = {"ssim": ssim_matrix}
                        '''
                        np.savez_compressed(
                            file_save_path,
                            **data
                        )
                        '''
                        # file_save_path = (save_path_subfolder / file.with_suffix('.npz').name).resolve()
                        penetration_folder = save_path_subfolder / "penetration"
                        penetration_folder.mkdir(parents=True, exist_ok=True)

                        file_save_path_penetration = (penetration_folder / file.stem).resolve()
                        
                        # Queue async save (compressed NPZ). Access later via key 'penetration'.
                        saver.save(file_save_path_penetration, penetration=penetration)
                        

                        cone_angle_folder = save_path_subfolder / "cone_angle"
                        cone_angle_folder.mkdir(parents=True, exist_ok=True)

                        file_save_path_cone_angle = (cone_angle_folder / file.stem).resolve()

                        saver.save(file_save_path_cone_angle, cone_angle=cone_angle_AngularDensity)
    finally:
        # Ensure all background saves complete before exiting
        saver.shutdown(wait=True)

                    

if __name__ == '__main__':
    
    from multiprocessing import freeze_support
    freeze_support()
    import asyncio, time

    start = time.time()
    asyncio.run(main())
    print(f"Total elapsed: {time.time() - start:.2f}s")
    
