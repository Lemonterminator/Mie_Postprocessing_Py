from functions_videos import load_cine_video
import numpy as np
import tkinter as tk 
from tkinter import filedialog
import cv2
from clustering import *
from videoProcessingFunctions import *
from opticalFlow import *


root = tk.Tk()
root.withdraw()

################################
# Main function
###############################
all_files = filedialog.askopenfilenames(title="Select one or more files")
for file in all_files:
    print("Processing:", file)
    video = load_cine_video(file)  # Ensure load_cine_video is defined or imported

    # play_video_cv2(video)

    nframes, height, width = video.shape[:3]
    dtype = video[0].dtype
    
    for i in range(nframes):
        frame = video[i]
        if np.issubdtype(dtype, np.integer):
            # For integer types (e.g., uint16): scale down from 16-bit to 8-bit.
            frame_uint8 = (frame / 16).astype(np.uint8)
        elif np.issubdtype(dtype, np.floating):
            # For float types (e.g., float32): assume values in [0,1] and scale up to 8-bit.
            frame_uint8 = np.clip((frame * 255), 0, 255).astype(np.uint8)
            # Boolean case: map False→0, True→255
        elif np.issubdtype(dtype, np.bool_):
            # logger.debug("Frame %d: boolean dtype; converting to uint8", i)
            # Convert bool→uint8 and apply gain, then clip
            frame_uint8 = np.clip(frame.astype(np.uint8) * 255, 0, 255).astype(np.uint8)
        else:
            # Fallback for any other type
            frame_uint8 = (frame / 16).astype(np.uint8)

        video[i] = frame_uint8
    video = video.astype(np.uint8)

    ##############################
    # Video Rotation and Stripping
    ##############################
    rotated_video = createRotatedVideo(video, 60)
    video_strip = createVideoStrip(rotated_video)

    # Find first real frame in video
    firstFrameNumber = findFirstFrame(video_strip)

   ##############################
    # Optical Flow Visualization
    ##############################
    first_frame = video_strip[firstFrameNumber]
    prev_frame = first_frame

    for i in range(firstFrameNumber, nframes):
        frame = video_strip[i]

        # --- Compute DeepFlow optical flow ---
        flow = opticalFlowFarnebackCalculation(prev_frame, frame) # Farneback 0.3 threshold

        # Compute magnitude (motion strength) and angle (not needed here)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # threshold movement
        mask = (mag > 0.3).astype(np.uint8) * 255

        # clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

        # Use clustering to get signal outlines
        # cluster_distance affects distance threshold for clustering points
        # alpha affects distance for concave hull generation
        cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)
        # USE CLUSTER_MASK TO FOR DATA EXTRACTION LATER

        # Create overlay of filled and clustered mask on original frame
        clustered_overlay = overlay_cluster_outline(frame, cluster_mask)

        # Display results
        # cv2.imshow('filled mask', filled_mask)
        cv2.imshow('Clustered Overlay', clustered_overlay)
        cv2.imshow("mask", mask)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(40) & 0xFF

        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

        prev_frame = frame

        print(f"Processed frame {i+1}/{nframes}")

    cv2.destroyAllWindows()