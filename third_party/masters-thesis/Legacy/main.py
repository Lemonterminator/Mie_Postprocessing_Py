from matplotlib import pyplot as plt
from clustering import *
from extrapolation import SprayConeBackfill, extrapolate_cone
from functions_videos import *
from Legacy.functions_optical_flow import *

import opticalFlow as of
import videoProcessingFunctions as vpf
from Legacy.std_functions3 import *
import tkinter as tk
from tkinter import filedialog

# Pipeline:
    # Compute optical flow between frames
    # Compute magnitude + direction maps
    # Cluster or threshold flow difference
    # Generate a mask
    # Clean mask with morphology

# IDEAS:
    # use confidence mapping to combine optical flow and thresholding
    # instead of combining binary masks use confidence values from otsu thresholding and optical flow to create a weighted mask
    # Have user set nozzle point and cut video automatically after rotation (Take 200 pixels up and down from nozzle point?, set nozzle point as far left point?)

# TODO:
    # combine optical flow with otsu thresholding for better results (done)
    # improve clustering method to avoid holes in masks
        # maybe use morphological operations + fill, to close holes
        # or add some interpolation algorithm to fill gaps in masks over time
    # optimize performance for larger videos (CUDA?)
    # improve GUI for mask drawing and parameter tuning
    # add option to save/load masks (TBD)
    # add some extrapolation algorithm to extend masks before/after detected motion (partially done)

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

################################
# Load Video Files
###############################
all_files = filedialog.askopenfilenames(title="Select one or more files")
for file in all_files:
    print("Processing:", file)
    video = load_cine_video(file)  # Ensure load_cine_video is defined or imported

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
    rotated_video = vpf.createRotatedVideo(video, 60) # Rotate 60 degrees clockwise
    video_strip = rotated_video

    firstFrameNumber = vpf.findFirstFrame(video_strip)
    first_frame = video_strip[firstFrameNumber]

    ######################################
    # Simple background Removal Visualization **UNUSED**
    ######################################
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     foreground = vpf.removeBackground(frame, first_frame)
    #     cv2.imshow('Foreground', foreground)
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(60) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    drawing = False
    points = []

    def draw_mask(event, x, y, flags, param):
        global drawing, points, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            points.append((x, y))
            cv2.line(mask, points[-2], points[-1], 255, thickness=2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

            if len(points) > 2:
                contour = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [contour], 255)

            points = []

    frame = video_strip[nframes // 2]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.namedWindow("Draw Mask")
    cv2.setMouseCallback("Draw Mask", draw_mask)

    while True:
        # Ensure overlay is 3-channel BGR (frame may be grayscale)
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            overlay = frame.copy()

        # Apply red overlay safely (works even if mask has no 255 pixels)
        mask_bool3 = (mask == 255)[:, :, None]
        overlay = np.where(mask_bool3, np.array([0, 0, 255], dtype=overlay.dtype), overlay)

        cv2.imshow("Draw Mask", overlay)

        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # reset mask
            mask[:] = 0

    cv2.destroyAllWindows()
    cv2.imwrite("mask.png", mask)


    ##############################
    # Filter Visualization
    ###############################
    vpf.applyCLAHE(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('CLAHE filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows() 

    # video_strip = vpf.SVDfiltering(video_strip, k=5)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('SVD filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.removeBackgroundSimple(video_strip, first_frame, threshold=10)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Simple Background Removal', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    background_mask = vpf.createBackgroundMask(first_frame, threshold=10) # Chamber walls have an intensity of about 3
    otsu_video = vpf.OtsuThreshold(video_strip, background_mask)
    # for i in range(nframes):
    #     frame = otsu_video[i]
    #     cv2.imshow('Otsu Threshold', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.applyLaplacianFilter(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Laplacian filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows() 

    # vpf.applyDoGfilter(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('DoG filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.adaptiveGaussianThreshold(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Adaptive Gaussian Threshold', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.chanVeseSegmentation(video_strip)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Chan-Vese Segmentation', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # vpf.temporalMedianFilter(video_strip, firstFrameNumber)
    # for i in range(nframes):
    #     frame = video_strip[i]
    #     cv2.imshow('Temporal Median Filter', frame)
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # mask = vpf.adaptive_background_subtraction(video_strip)
    # for i in range(nframes):
    #     frame = mask[i]
    #     cv2.imshow('Temporal Median Filter', frame)
    #     cv2.imshow('Original', video_strip[i])
    #     key = cv2.waitKey(40) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()



    ##############################
    # Optical Flow Visualization
    ##############################
    cluster_masks, clustered_overlays, masks = of.runOpticalFlowCalculation(firstFrameNumber, video_strip, method='Farneback')
    for i in range(firstFrameNumber, nframes):
        
        frame = video_strip[i]
        clustered_overlay = clustered_overlays[i]
        mask = masks[i]

        cv2.imshow('Clustered Overlay', clustered_overlay)
        cv2.imshow("mask", mask)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(40) & 0xFF

        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

    cv2.destroyAllWindows()

    # GOAL: if both otsu and cluster masks agree, keep the region
    # try to remake with non binary masks?

    # --- Combine Otsu masks and cluster masks with adjustable weights ---
    # Change these weights to tune contribution from each method
    w_otsu = 0.2      # weight for Otsu mask (0.0 - 1.0)
    w_cluster = 0.6   # weight for cluster mask (0.0 - 1.0)
    w_freehand = 0.2  # weight for freehand mask drawn by user (0.0 - 1.0)
    threshold = 1   # threshold on weighted sum (0.0 - 1.0) — adjust to control how strict combination is

    # Prepare combined masks array
    combined_masks = np.zeros_like(cluster_masks, dtype=np.uint8)
    otsu_optical = np.zeros_like(cluster_masks, dtype=np.uint8)

    # Load freehand mask created earlier by the user (expects single-channel binary image "mask.png")
    freehand_mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    if freehand_mask is None:
        print("Warning: 'mask.png' not found — proceeding without freehand mask")
        freehand_mask_f = np.zeros((height, width), dtype=np.float32)
    else:
        # Resize to match frames if necessary, keep nearest neighbour to preserve binary nature
        if freehand_mask.shape != (height, width):
            freehand_mask = cv2.resize(freehand_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        freehand_mask_f = (freehand_mask > 0).astype(np.float32)  # 0.0 or 1.0

    for idx in range(nframes):
        otsu_mask = otsu_video[idx].astype(np.float32) / 255.0
        cluster_mask = cluster_masks[idx].astype(np.float32) / 255.0
        freehand = freehand_mask_f  # same for all frames, shape (height, width)

        # If a mask is empty for this frame, treat it as a full-frame mask
        # (counts as if the whole frame is the mask)
        if np.count_nonzero(otsu_mask) == 0:
            otsu_mask = np.ones_like(otsu_mask, dtype=np.float32)
        if np.count_nonzero(freehand) == 0:
            freehand = np.ones_like(freehand, dtype=np.float32)
        # if np.count_nonzero(cluster_mask) == 0:  
        #     cluster_mask = np.ones_like(cluster_mask, dtype=np.float32)

        # Normalize weights (avoid division by zero)
        total_w = w_otsu + w_cluster + w_freehand
        if total_w <= 0:
            norm_otsu = norm_cluster = norm_freehand = 1.0/3.0
        else:
            norm_otsu = w_otsu / total_w
            norm_cluster = w_cluster / total_w
            norm_freehand = w_freehand / total_w

        # Weighted sum and binarize
        weighted = (otsu_mask * norm_otsu) + (cluster_mask * norm_cluster) + (freehand * norm_freehand)
        combined = (weighted >= threshold).astype(np.uint8) * 255

        # Optional small cleanup
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        combined = cv2.dilate(combined, np.ones((5,5), np.uint8), iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # combined_cluster_mask = create_cluster_mask(combined, cluster_distance=50, alpha=40) # Further clustering on combined mask, may need tuning, or skip

        # combined_masks[idx] = combined_cluster_mask
        otsu_optical[idx] = combined 

    print(f"Combined masks computed with w_otsu={w_otsu}, w_cluster={w_cluster}, w_freehand={w_freehand}, threshold={threshold}")

    # Show combined masks (press 'q' to quit, 'p' to pause)
    intensity_values = [] # store average intensities
    for i in range(firstFrameNumber, nframes):
        frame = video_strip[i]
        # combined = combined_masks[i]
        otsu_optical_mask = otsu_optical[i]

        # Compute mean intensity inside the mask
        mean_intensity = cv2.mean(frame, otsu_optical_mask)
        intensity_values.append(mean_intensity)

        clustered_overlay = overlay_cluster_outline(frame, otsu_optical_mask) #may not need clustering

        cv2.imshow('Otsu + Optical flow', otsu_optical_mask)
        cv2.imshow('Clustered Overlay on Combined Mask', clustered_overlay)
        # cv2.imshow('Combined Mask', combined)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(200) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

    cv2.destroyAllWindows()


    # --- Analyze intensity values ---
    # Needs more work, maybe diffent method to find significant changes

    # calculate derivative of intensity values
    intensity_values = np.array(intensity_values)   [:,0]  # Extract first channel if mean returns a tuple
    
    # Apply rolling mean with window size 5
    window_size = 5
    intensity_smoothed = np.convolve(intensity_values, np.ones(window_size)/window_size, mode='valid')
    
    # Compute derivative on smoothed data
    intensity_derivative = np.diff(intensity_smoothed, prepend=intensity_smoothed[0])

    # --- Create shifted x-axis ---
    # Adjust frame_numbers to match the length after rolling mean
    frame_numbers = np.arange(firstFrameNumber, firstFrameNumber + len(intensity_derivative))

    # only consider frames at least 10 after firstFrameNumber
    start_offset = 10
    if len(intensity_derivative) <= start_offset:
        # not enough frames — fall back to full range
        min_idx = int(np.argmin(intensity_derivative))
    else:
        sliced = intensity_derivative[start_offset:]
        rel_min = int(np.argmin(sliced))
        min_idx = rel_min + start_offset

    min_frame = int(frame_numbers[min_idx])
    min_value = float(intensity_derivative[min_idx])
    print(f"Lowest intensity derivative at frame {min_frame} (index {min_idx}) = {min_value:.6f}")

    plt.plot(frame_numbers, intensity_derivative)
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Intensity Inside Region")
    plt.title("Intensity Over Time (Shifted)")
    plt.show()


    ##############################
    # Extrapolation work in progress
    ##############################


    # IDEA: fill the area close to the nozzle, take an area close to the nozzle of the detection area (left most side to middle?)
    #       calculate the angle which it makes up, and fill the resulting cone.
    #       Maybe use a system which detects if the area around the nozzle is hidden, if not then don't run
    # PROBLEMS: detected area needs to be noise free (although probably should be anyway)
    #       Close point calculation will not work... not that it would've worked anyway
    #       Need to add a way to remove noise outside of detection as a pre-process, idea is to use the nozzle as the
    #       point from which 25(less?) degrees above and below and extends to end of frame and keep any detected areas inside that region.
    #       NOTE, keep areas outside of the angle IF they are connected to and area inside the angle. 

    spray_origin = (1, height // 2) # Known spray origin (x, y), TODO: add a way to set this interactively

    for i in range(firstFrameNumber, nframes):

        detected_mask = otsu_optical[i]

        final_mask = extrapolate_cone(detected_mask, spray_origin, min_points=1)

        cv2.imshow("Final Mask", final_mask)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)


    # Known spray origin (x, y)
    # spray_origin = (1, height // 2)
    # backfiller = SprayConeBackfill(spray_origin)

    # for i in range(firstFrameNumber, nframes):

    #     # detection step
    #     detected_mask = otsu_optical[i]
    #     frame = video_strip[i]

    #     # Backfill missing left-side cone
    #     backfill_mask = backfiller.backfill(detected_mask)

    #     # Merge
    #     final_mask = cv2.bitwise_or(detected_mask, backfill_mask)

    #     # Visualization
    #     vis = frame.copy()
    #     # Ensure vis is 3-channel BGR (frame may be grayscale)
    #     if vis.ndim == 2 or (vis.ndim == 3 and vis.shape[2] == 1):
    #         vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    #     # Create red overlay and apply mask
    #     overlay = np.zeros_like(vis)
    #     overlay[:] = (0, 0, 255)
    #     final_mask = final_mask.astype(np.uint8) * 255

    #     cv2.copyTo(overlay, final_mask, vis)

    #     cv2.circle(vis, spray_origin, 4, (0, 255, 0), -1)

    #     cv2.imshow("Spray Tracking", vis)
    #     cv2.imshow("Detected Mask", detected_mask)
    #     cv2.imshow("Final Mask", final_mask)

    #     key = cv2.waitKey(100) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)


        # vis = frame.copy()
        # # Ensure vis is 3-channel BGR
        # if vis.ndim == 2 or (vis.ndim == 3 and vis.shape[2] == 1):
        #     vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        # # Ensure final_mask matches frame size
        # if final_mask.shape != vis.shape[:2]:
        #     final_mask = cv2.resize(final_mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        # mask_bool = final_mask > 0
        # # Only assign if there are masked pixels to avoid NumPy assignment errors when mask is empty
        # if np.any(mask_bool):
        #     vis[mask_bool] = (0, 0, 255)
        # cv2.circle(vis, spray_origin, 4, (0, 255, 0), -1)



print("Processing complete.")


# import time
# if __name__ == '__main__':
#     from multiprocessing import freeze_support
#     freeze_support()  # Optional: Needed if freezing to an executable

#     start_time = time.time()
#     main()
#     elapsed_time = time.time() - start_time

#     print(f"Sequential main() finished in {elapsed_time:.2f} seconds.")
