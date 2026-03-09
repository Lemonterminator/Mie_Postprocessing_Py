import asyncio
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

async def load_video_async(file):
    loop = asyncio.get_running_loop()
    # Use default executor (or specify one if needed)
    video = await loop.run_in_executor(None, load_cine_video, file)
    return video

async def compute_flow_async(video):
    loop = asyncio.get_running_loop()
    velocity_field = await loop.run_in_executor(None, compute_optical_flow, video)
    # velocity_field = await loop.run_in_executor(None, compute_optical_flow_cuda, video)
    return velocity_field

async def rotate_video_async(video, angle):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, rotate_video, video, angle)

async def gaussian_lp_video_async(video, cutoff):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, Gaussian_LP_video, video, cutoff)

async def median_filter_video_async(video, M, N):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, median_filter_video, video, M, N)

async def binarize_video_async(video, method, thresh_val):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, binarize_video_global_threshold, video, method, thresh_val)



# Define a semaphore with a limit on concurrent tasks
SEMAPHORE_LIMIT = 2  # Adjust this based on your CPU capacity
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


def calculate_TD_map(video):
    """Computes the time–distance map by summing pixel intensities over rows."""
    return np.sum(video, axis=1).T  # shape: [width, frames]

def calculate_bw_area(BW):
    """Counts the white pixels (255) in each binary frame."""
    return np.sum(BW == 255, axis=(1, 2))



async def process_file(file):
    async with semaphore:
        print("Processing:", file)
        video = await load_video_async(file)

        if "Sch" in file.name:
            velocity_field = await compute_flow_async(video)

            for flow in velocity_field:
                flow_x = flow[:, :, 0]
                flow_y = flow[:, :, 1]
                flow_mag = np.sqrt(flow_x**2 + flow_y**2)
                flow_mag_normalized = flow_mag / 1E1
                flow_img = cv2.flip(flow_mag_normalized, 0)
                # cv2.imshow('Image', flow_img.astype(np.uint8))
                # cv2.waitKey(30)

            # cv2.destroyAllWindows()

        elif "OH" in file.name:
            RT = await rotate_video_async(video, -45)
            strip = RT[0:150, 250:550, :]

            LP_filtered = await gaussian_lp_video_async(strip, 40)
            med = await median_filter_video_async(LP_filtered, 5, 5)
            BW = await binarize_video_async(med, method="fixed", thresh_val=300)

            TD_map = calculate_TD_map(strip)
            area = calculate_bw_area(BW)

            '''
            # Show time-distance map
            plt.imshow(TD_map, cmap='jet', aspect='auto')
            plt.title("Average Time–Distance Map")
            plt.xlabel("Time (frames)")
            plt.ylabel("Distance (pixels)")
            plt.colorbar(label="Sum Intensity")
            plt.show()

            # Show area over time
            plt.figure(figsize=(10, 4))
            plt.plot(area, color='blue')
            plt.xlabel("Frame")
            plt.ylabel("Area (white pixels)")
            plt.title("Area Over Time")
            plt.grid(True)
            plt.tight_layout()
            plt.show()'
            '''

async def process_subfolder(subfolder, parent_folder):
    directory_path = Path(parent_folder) / subfolder
    files = [file for file in directory_path.iterdir() if file.is_file()]
    # Schedule processing of all files concurrently
    tasks = [asyncio.create_task(process_file(file)) for file in files]
    await asyncio.gather(*tasks)

async def main_async(parent_folder):
    subfolders = get_subfolder_names(parent_folder)
    # Schedule processing for each subfolder concurrently
    tasks = [asyncio.create_task(process_subfolder(subfolder, parent_folder)) for subfolder in subfolders]
    await asyncio.gather(*tasks)


import time
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional for frozen executables
    parent_folder = r"G:\2024_Internal_Transient_Sprays\BC20240524_ReactiveSpray_HZ4\Cine"
    start_time = time.time()
    asyncio.run(main_async(parent_folder))
    elapsed_time = time.time() - start_time

    print(f"Async main_async() finished in {elapsed_time:.2f} seconds.")

