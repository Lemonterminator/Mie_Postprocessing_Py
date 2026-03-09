def opticalFlowFarnebackCalculation(prev_frame, frame):
    import cv2

    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, # type: ignore
                                        0.5,  # pyramid scale
                                        3,    # levels
                                        15,   # window size
                                        3,    # iterations
                                        5,    # poly_n
                                        1.2,  # poly_sigma
                                        0)    # type: ignore # flags

    return flow

def opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow):
    import cv2
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    flow = deepflow.calc(prev_gray, gray, None) # type: ignore

    return flow

def runOpticalFlowCalculation(firstFrameNumber, video, method, deepflow=None):
    import cv2
    import numpy as np
    from clustering import create_cluster_mask, overlay_cluster_outline 

    first_frame = video[firstFrameNumber]
    prev_frame = first_frame

    nframes = video.shape[0]
    cluster_masks = np.zeros_like(video, dtype=np.uint8)
    clustered_overlays =  np.zeros_like(video, dtype=np.uint8)
    masks =  np.zeros_like(video, dtype=np.uint8)

    if method == 'Farneback':
        for i in range(firstFrameNumber, nframes):
            frame = video[i]

            # --- Compute DeepFlow optical flow ---
            flow = opticalFlowFarnebackCalculation(prev_frame, frame) # Farneback 0.3 threshold

            # Compute magnitude (motion strength) and angle (not needed here)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # threshold movement
            mask = (mag > 0.4).astype(np.uint8) * 255

            # clean up
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

            # Use clustering to get signal outlines
            cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)

            clustered_overlay = overlay_cluster_outline(frame, cluster_mask)

            prev_frame = frame

            cluster_masks[i] = cluster_mask
            clustered_overlays[i] = clustered_overlay # only for visualization
            masks[i] = mask # only for visualization
            print(f"Processed frame {i+1}/{nframes}")

        return cluster_masks, clustered_overlays, masks

    # Not fully implemented yet, but structure is in place
    elif method == 'DeepFlow':
        if deepflow is None:
            raise ValueError("DeepFlow instance must be provided for DeepFlow method.")
        for i in range(firstFrameNumber, nframes):
            frame = video[i]

            # --- Compute DeepFlow optical flow ---
            flow = opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow) # DeepFlow 1.0 threshold

            # Compute magnitude (motion strength) and angle (not needed here)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # threshold movement
            mask = (mag > 1.0).astype(np.uint8) * 255

            # clean up
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

            # Use clustering to get signal outlines
            cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)
            clustered_overlay = overlay_cluster_outline(frame, cluster_mask)

            prev_frame = frame

            cluster_masks[i] = cluster_mask
            clustered_overlays[i] = clustered_overlay # only for visualization
            masks[i] = mask # only for visualization
            print(f"Processed frame {i+1}/{nframes}")

        return cluster_masks, clustered_overlays, masks
    else:
        raise ValueError(f"Unsupported optical flow method: {method}")
    


def runOpticalFlowCalculationWeighted(firstFrameNumber, video, method, deepflow=None):
    import cv2
    import numpy as np

    first_frame = video[firstFrameNumber]
    prev_frame = first_frame

    nframes = video.shape[0]
    mag_array = np.zeros_like(video, dtype=np.float32)

    if method == 'Farneback':
        for i in range(firstFrameNumber, nframes): # limit to half for testing
            frame = video[i]

            flow = opticalFlowFarnebackCalculation(prev_frame, frame) 

            # Compute magnitude (motion strength) and angle (not needed here)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            mag_array[i] = mag

            prev_frame = frame

            print(f"Processed frame {i+1}/{nframes}")

        return mag_array # return only magnitude for weighted processing
    else:
        raise ValueError(f"Unsupported optical flow method: {method}")