from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import napari
import numpy as np
from OSCC_postprocessing.functions_videos import *
from video_config import *

DATA_DIR = Path(__file__).resolve().parent / "data"

def main():
        

    # Parameters for the video
    video = load_cine_video(r"G:\Master_Thesis\BC20220627 - Heinzman DS300 - Mie Top view\Cine\T9\79.cine")

    RT = rotate_video(video, rotation)

    # frame, y, x
    # strip_raw = RT[first_frame:last_frame, y_start:y_end, x_start:x_end]


    # strip_raw = video
    # strip = 4096-strip_raw  # Invert the video frames
    # play_video_cv2(strip_raw/4096, intv=17)


    # strip = 4096.0-video
    strip = video/4096.0  # Normalize the video frames to [0, 1] range

    play_video_cv2(strip, intv=17)

    chamber_mask = np.load(DATA_DIR / "chamber_mask.npy")  # Load the mask from a .npy file
    masked_strip = mask_video(strip, chamber_mask)

    viewer = napari.view_image(
        #strip,
        masked_strip,
        name='Rotated Video Strip',
        ndisplay=3,                          # render in 3D, not just 2D :contentReference[oaicite:0]{index=0}
        # title=['Frame','Y','X'],      # label dimensions so sliders read out correctly :contentReference[oaicite:1]{index=1}
        colormap='gray'
    )
    napari.run()
    

    
    
    # masked_strip = mask_video(strip, chamber_mask)

    background = np.median(masked_strip[0:50, :, :], axis=0)

    # Subtract background
    foreground = masked_strip - background[None, :, :]

    # Clip to avoid negative values (assuming uint16 data)
    # foreground = np.clip(foreground, 0, None).astype(np.uint16)
    # foreground = np.abs(foreground).astype(np.uint16)  # Ensure no negative values

    viewer = napari.view_image(
        # masked_strip,
        foreground,
        name='3D Video',
        ndisplay=3,                          # render in 3D, not just 2D :contentReference[oaicite:0]{index=0}
        # title=['Frame','Y','X'],      # label dimensions so sliders read out correctly :contentReference[oaicite:1]{index=1}
        colormap='gray'
    )
    napari.run()

    # Optional: Visualize one frame
    # plt.imshow(foreground[60], cmap='gray')

    
    
    plt.imshow(background, cmap='gray')
    plt.title("Calculated Background as the median of all frames")
    plt.colorbar()
    plt.show()


    # np.savez_compressed("masked_strip.npz", masked_strip=masked_strip)

    # Frame-to-frame difference
    diff = np.abs(np.diff(foreground.astype(np.int32), axis=0))
    diff = np.clip(diff, 0, 65535).astype(np.uint16)

    # Append one frame to keep original length
    diff = np.concatenate([diff, np.zeros_like(diff[:1])], axis=0)

    viewer = napari.view_image(
        # masked_strip,
        diff,
        name='3D Video',
        ndisplay=3,                          # render in 3D, not just 2D :contentReference[oaicite:0]{index=0}
        # title=['Frame','Y','X'],      # label dimensions so sliders read out correctly :contentReference[oaicite:1]{index=1}
        colormap='gray'
    )
    napari.run()

    play_video_cv2(diff**2.2/2**16, intv = 100)

    # Apply a simple temporal moving average filter
    diff_sum = np.zeros(diff.shape, dtype=np.float32)

    for i in range(4):
        diff_sum += np.roll(diff, shift=-i, axis=0)

        
    play_video_cv2(diff_sum**2.2/2**(16+4), intv= 100)



    np.savez_compressed("diff_filtered.npz", masked_strip=masked_strip)
    
    

if __name__ == "__main__":
    main()
