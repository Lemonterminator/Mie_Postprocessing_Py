import os
import numpy as np
from pathlib import Path
import cv2

from OSCC_postprocessing.cine.functions_videos import load_cine_video as _canonical_load_cine_video

# -----------------------------
# Cine video reading and playback
# -----------------------------
def load_cine_video(cine_file_path, frame_limit=None):
    """Compatibility wrapper around the canonical Cine loader."""
    return _canonical_load_cine_video(cine_file_path, frame_limit=frame_limit)

def get_subfolder_names(parent_folder):
    parent_folder = Path(parent_folder)
    subfolder_names = [item.name for item in parent_folder.iterdir() if item.is_dir()]
    return subfolder_names

def play_video_cv2(video, gain=1):
    total_frames = len(video)
    dtype = video[0].dtype
    
    for i in range(total_frames):
        frame = video[i]
        if np.issubdtype(dtype, np.integer):
            # For integer types (e.g., uint16): scale down from 16-bit to 8-bit.
            frame_uint8 = gain * (frame / 16).astype(np.uint8)
        elif np.issubdtype(dtype, np.floating):
            # For float types (e.g., float32): assume values in [0,1] and scale up to 8-bit.
            frame_uint8 = np.clip(gain * (frame * 255), 0, 255).astype(np.uint8)
                    # Boolean case: map False→0, True→255
        elif np.issubdtype(dtype, np.bool_):
            # logger.debug("Frame %d: boolean dtype; converting to uint8", i)
            # Convert bool→uint8 and apply gain, then clip
            frame_uint8 = np.clip(frame.astype(np.uint8) * 255 * gain, 0, 255).astype(np.uint8)

        else:
            # Fallback for any other type
            frame_uint8 = gain * (frame / 16).astype(np.uint8)
        
        cv2.imshow('Frame', frame_uint8)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
