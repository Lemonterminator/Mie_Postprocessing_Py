from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_crop import *
from mie_postprocessing.cone_angle import *
from mie_postprocessing.ssim import *
from mie_postprocessing.video_filters import *
from mie_postprocessing.functions_bw import *
from mie_postprocessing.video_playback import *
from mie_multihole_pipeline import *
import matplotlib.pyplot as plt

from Schlieren_singlehole_pipeline import *
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
parent_folder = r"Z:\BC20220627 - Heinzman DS300 - Mie Top view\Cine"
DATA_DIR = Path(r"G:\Samuel")

hydraulic_delay = 11  # Hydraulic delay in frames, adjust as needed
gain = 1 # Gain factor for video processing
gamma = 1  # Gamma correction factor for video processing

# Inner and outer radius (in pixels) for cropping the images
# ir_ = 14 # DS300
ir_ = 11 # Nozzle 1,2,3,4
or_ = 380


# Directory containing mask images and numpy files
# DATA_DIR = Path(__file__).resolve().parent / "data"


chamber_mask_path = DATA_DIR / "chamber_mask.npy"

if chamber_mask_path.exists():
    chamber_mask = np.load(chamber_mask_path)
else:
    subprocess.run(["python", "masking.py"], check=True)
    chamber_mask = np.load(chamber_mask_path)

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
                        offset = float(data['offset'])
                        centre = (float(data['centre_x']), float(data['centre_y']))

            # print(files)
            for file in files:
                if file.suffix == '.cine':
                    testpoint_name = directory_path.stem
                    video_name = file.stem
        
                    print("Procssing:", file.parts[-3], "/", file.parts[-2], "/", file.parts[-1])
                    # start_time = time.time()


                    if "Schlieren" in file.name:
                        video = load_cine_video(file).astype(np.float32)/4096  # Ensure load_cine_video is defined or imported
                        frames, height, width = video.shape
                        schlieren_singlehole_pipeline(video, chamber_mask, centre, offset)


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
                        video = load_cine_video(file, frame_limit=50).astype(np.float32)/4096  # Ensure load_cine_video is defined or imported
                        frames, height, width = video.shape

                        centre_x = float(centre[0])
                        centre_y = float(centre[1])

                        
                        segments, penetration, cone_angle_AngularDensity, bw_vids, boundaries, penetration_old = mie_multihole_pipeline(
                            video, centre, number_of_plumes, 
                            gamma=gamma, 
                            binarize_video=False, 
                            plot_on=False
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

                        if penetration is not None:
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

    import argparse
    import os
    import asyncio, time

    parser = argparse.ArgumentParser(description="Run Mie post-processing pipeline")
    parser.add_argument(
        "-p", "--parent-folder",
        default=os.getenv("MIE_PARENT_FOLDER", parent_folder),
        help="Root folder containing subfolders with .cine files (overrides hardcoded path)"
    )
    args = parser.parse_args()

    # Allow overriding the global parent_folder from CLI/env
    parent_candidate = Path(args.parent_folder)
    if not parent_candidate.exists():
        print(f"Configured parent folder not found: {parent_candidate}")
        print("Set a valid path via --parent-folder or MIE_PARENT_FOLDER, then rerun.")
        raise SystemExit(0)

    parent_folder = str(parent_candidate)

    start = time.time()
    asyncio.run(main())
    print(f"Total elapsed: {time.time() - start:.2f}s")
    
