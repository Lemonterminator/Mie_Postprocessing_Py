"""Interactive OpenCV helpers used by the fused Masters-thesis workflow.

Local changes:
- support contour-fill editing with left-drag to add and right-drag to erase
- render masks as color overlays instead of opaque binary debug windows
- allow OpenCV windows to exit via `q`, `Esc`, or direct window close
"""

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
    """Create a binary mask from a contour-fill overlay editor on the middle strip frame."""
    import cv2
    import numpy as np

    nframes, height, width = video_strip.shape[:3]
    frame = video_strip[nframes // 2]
    mask = edit_mask_overlay(frame, np.zeros(frame.shape[:2], dtype=np.uint8), window_name="Draw Mask")
    cv2.destroyAllWindows()
    cv2.imwrite("mask.png", mask)
    return mask


def edit_mask_overlay(frame, initial_mask, window_name="Edit Mask"):
    """Edit a mask with contour add/remove gestures rendered as live red/blue overlays."""
    import cv2
    import numpy as np

    mask = initial_mask.copy().astype(np.uint8)
    drawing_mode = None
    points = []

    def ensure_bgr(img):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    def apply_contour(target_mask, contour_points, value):
        if len(contour_points) < 3:
            return
        contour = np.array(contour_points, dtype=np.int32)
        fill_value = 255 if value else 0
        cv2.fillPoly(target_mask, [contour], fill_value)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing_mode, points, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_mode = "add"
            points = [(x, y)]
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing_mode = "remove"
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing_mode is not None:
            if not points or points[-1] != (x, y):
                points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and drawing_mode == "add":
            if not points or points[-1] != (x, y):
                points.append((x, y))
            apply_contour(mask, points, value=True)
            drawing_mode = None
            points = []
        elif event == cv2.EVENT_RBUTTONUP and drawing_mode == "remove":
            if not points or points[-1] != (x, y):
                points.append((x, y))
            apply_contour(mask, points, value=False)
            drawing_mode = None
            points = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        overlay = ensure_bgr(frame)

        mask_bool = mask > 0
        overlay[mask_bool] = (
            0.55 * overlay[mask_bool] + 0.45 * np.array([0, 0, 255], dtype=np.float32)
        ).astype(np.uint8)

        if len(points) >= 2 and drawing_mode is not None:
            contour = np.array(points, dtype=np.int32)
            line_color = (0, 0, 255) if drawing_mode == "add" else (255, 0, 0)
            cv2.polylines(overlay, [contour], isClosed=False, color=line_color, thickness=2)

        cv2.imshow(window_name, overlay)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(40) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            mask[:] = 0
            drawing_mode = None
            points = []

    cv2.destroyWindow(window_name)
    return mask
