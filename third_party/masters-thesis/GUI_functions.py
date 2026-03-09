def set_spray_origin(file, rotated_video, firstFrameNumber, nframes, height):    
    import cv2
    import json    
    import os

    # Load saved spray origins
    origins_file = 'spray_origins.json'
    if os.path.exists(origins_file):
        with open(origins_file, 'r') as f:
            spray_origins = json.load(f)
    else:
        spray_origins = {}

    # Set spray origin
    if file in spray_origins:
        spray_origin = tuple(spray_origins[file])
        print(f"Reusing spray origin for {file}: {spray_origin}")
    else:
        # UI to select
        class PointHolder:
            def __init__(self):
                self.point = None
        
        holder = PointHolder()
        def select_origin(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                holder.point = (x, y)  # type: ignore
                print(f"Selected spray origin: {holder.point}")
        
        cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[firstFrameNumber+100]) # Show a frame after firstFrameNumber for context, may need adjustment
        cv2.setMouseCallback('Set Spray Origin - Click on the nozzle', select_origin)
        
        current_frame = firstFrameNumber + 100
        while holder.point is None:
            key = cv2.waitKeyEx(10)
            if key == ord('q'):
                break
            elif key == 2424832:  # left arrow
                current_frame = max(firstFrameNumber, current_frame - 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[current_frame])
            elif key == 2555904:  # right arrow
                current_frame = min(nframes - 1, current_frame + 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[current_frame])
        cv2.destroyWindow('Set Spray Origin - Click on the nozzle')
        
        if holder.point is None:
            spray_origin = (1, height // 2)  # Default
        else:
            spray_origin = holder.point
        
        # Save
        spray_origins[file] = list(spray_origin)
        with open(origins_file, 'w') as f:
            json.dump(spray_origins, f)

        print(f"Spray origin for {file}: {spray_origin}")

    return spray_origin

def draw_freehand_mask(video_strip):
    import cv2
    import numpy as np

    nframes, height, width = video_strip.shape[:3]
    
    drawing = False
    points = []

    def draw_mask(event, x, y, flags, param):
        nonlocal drawing, points, mask

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