from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mie_postprocessing import *
from mie_postprocessing.functions_videos import *
from mie_postprocessing.video_playback import *
from mie_postprocessing.rotate_with_alignment import * 

cfg = {
  "plumes": 4,
  "offset": -45.0,
  "centre_x": 109.0,
  "centre_y": 74.0,
  "calib_radius": 640.0
}

import cupy as cp
import numpy as np  # only for constructing M; you can use cp as well

video_u16 = cp.load("C:/Users/Jiang/Documents/Mie_Py/Mie_Postprocessing_Py/ex.npy")

patch1 = video_u16[:, :, 0:80]
patch1_total_intensity = cp.sum(np.sum(patch1, axis=2),axis=1)/(patch1.shape[1]*patch1.shape[2])
scaling = patch1_total_intensity / np.max(patch1_total_intensity)


video_float = video_u16.astype(cp.float32)/scaling[:, None, None]

# play_video_cv2(video_float)


# Remap all frames with the same transform
import time
start_time = time.time()


centre_x = cfg["centre_x"]
centre_y = cfg["centre_y"]
offset = cfg["offset"]
plumes = cfg["plumes"]

full_h, full_w = int(video_float.shape[1]), int(video_float.shape[2])
roi_shape = (full_h/3, full_w)  # adjust if you want a tighter crop
calibration_point = (0.0, full_h / 2.0)

out_frames, mapx, mapy = rotate_video_nozzle_at_0_half_cupy(
    video_float,
    (centre_x, centre_y),
    offset,
    out_shape=roi_shape,
    calibration_point=calibration_point,
    interpolation="lanczos3",
    border_mode="constant",
    cval=0.0,
    stack=True,
    plot_maps=False,
)
print("Remapping took %.2f seconds"%(time.time() - start_time))
play_video_cv2(out_frames.get())

from mie_postprocessing import compute_raft_flows, compute_farneback_flows

flows = compute_farneback_flows(video_float, levels=10)

flows_scalar = flows[:, 0, :, :]**2 + flows[:, 1, :, :]**2
flows_scalar = np.sqrt(flows_scalar)
play_video_cv2(flows_scalar)



