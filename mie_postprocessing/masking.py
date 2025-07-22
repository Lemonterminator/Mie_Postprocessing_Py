import os
import concurrent.futures
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pycine.file as cine  # Ensure you have the 'pycine' package or equivalent installed

def read_frame(cine_file_path, frame_offset, width, height):
    with open(cine_file_path, "rb") as f:
        f.seek(frame_offset)
        frame_data = np.fromfile(f, dtype=np.uint16, count=width * height).reshape(height, width)
    return frame_data

# This function takes the path of a cine file and reads it as an numpy array.
def load_cine_video(cine_file_path):
    # Read the header
    header = cine.read_header(cine_file_path)

    # Extract width, height, and total frame count
    width = header['bitmapinfoheader'].biWidth
    height = header['bitmapinfoheader'].biHeight
    frame_offsets = header['pImage']  # List of frame offsets
    frame_count = len(frame_offsets)
    print(f"Video Info - Width: {width}, Height: {height}, Frames: {frame_count}")

    # Initialize an empty 3D NumPy array to store all frames
    video_data = np.zeros((frame_count, height, width), dtype=np.uint16)

    # Use ThreadPoolExecutor to read frames in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a dictionary to map futures to their frame indices
        future_to_index = {
            executor.submit(read_frame, cine_file_path, frame_offsets[i], width, height): i
            for i in range(frame_count)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                video_data[index] = future.result()
            except Exception as e:
                print(f"Error reading frame {index}: {e}")

    return np.array(video_data)

# Assume masking_gui is defined as in the previous code block
def masking_gui(img, file_name):
    """
    An interactive GUI to create a binary mask over an image.
    Users can add or block areas using circle, rectangle, or triangle shapes.
    
    Parameters:
      img: 2D numpy array representing a grayscale image.
      file_name: Base name to save the mask (mask saved as file_name.png and file_name.npy).
    
    Returns:
      mask: A 2D boolean numpy array where True indicates the selected (active) region.
    """
    from matplotlib.path import Path
    rows, cols = img.shape
    if os.path.exists(file_name + ".npy"):
        mask = np.load(file_name + ".npy")
        print(f"Loaded existing mask from {file_name}.npy")
    else:
        mask = np.ones((rows, cols), dtype=bool)
        print("Initialized new mask.")
    
    while True:
        
        '''        
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.imshow(mask, cmap='jet', alpha=0.3)
        plt.title("Current Mask (white=active)")
        plt.show(block=False)
        '''

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        plt.imshow(mask, cmap='gray', alpha=0.5)
        plt.title("Current Mask (white=active)")
        plt.show(block=False)
        # Expand limits by 50 pixels on all sides
        ax.set_xlim(-50, img.shape[1] + 50)
        ax.set_ylim(img.shape[0] + 50, -50)
        # plt.title("Click on two opposite corners (click outside allowed)")
        # pts = plt.ginput(2)
        # plt.close(fig)

        
        action = input("Type 'add' to add area, 'block' to block area, or 'quit' to finish: ").strip().lower()
        if action == 'quit':
            plt.close('all')
            break

        shape = input("Choose shape ('circle', 'rectangle', or 'triangle'): ").strip().lower()

        if shape == 'circle':
            print("Select two points for the circle (first and opposite point on the diameter).")
            pts = plt.ginput(2)
            if len(pts) < 2:
                print("Not enough points selected. Try again.")
                plt.close('all')
                continue
            (x1, y1), (x2, y2) = pts
            cen_x = (x1 + x2) / 2.0
            cen_y = (y1 + y2) / 2.0
            radius = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2.0
            
            xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
            circle_mask = ((xx - cen_x)**2 + (yy - cen_y)**2) <= radius**2
            if action == 'add':
                mask[circle_mask] = True
            elif action == 'block':
                mask[circle_mask] = False

        elif shape == 'rectangle':
            print("Select two opposite corners of the rectangle (you can click outside the image).")
            pts = plt.ginput(2)
            if len(pts) < 2:
                print("Not enough points selected. Try again.")
                plt.close('all')
                continue
            # Get points from user
            (x1, y1), (x2, y2) = pts
            # Create rectangle parameters (using floating point for accuracy)
            rect_x = min(x1, x2)
            rect_y = min(y1, y2)
            rect_w = abs(x2 - x1)
            rect_h = abs(y2 - y1)
            
            # Snap to left and top boundaries (MATLAB's equivalent: if rect(1)<1 then adjust)
            if rect_x < 0:
                rect_w = rect_w + rect_x  # since rect_x is negative, this subtracts its absolute value from the width
                rect_x = 0
            if rect_y < 0:
                rect_h = rect_h + rect_y
                rect_y = 0

            # Snap to the right and bottom edges
            if rect_x + rect_w > cols:
                rect_w = cols - rect_x
            if rect_y + rect_h > rows:
                rect_h = rows - rect_y

            # Convert to integer pixel indices.
            # Adding 1 to include the full pixel range (MATLAB's imshow includes whole pixel).
            x_min = int(round(rect_x))
            y_min = int(round(rect_y))
            x_max = int(round(rect_x + rect_w))
            y_max = int(round(rect_y + rect_h))
            
            # Create a rectangle mask that covers the selected area
            rect_mask = np.zeros((rows, cols), dtype=bool)
            rect_mask[y_min:y_max+1, x_min:x_max+1] = True
            
            if action == 'add':
                mask[rect_mask] = True
            elif action == 'block':
                mask[rect_mask] = False


        elif shape == 'triangle':
            print("Select three points to define the triangle.")
            pts = plt.ginput(3)
            if len(pts) < 3:
                print("Not enough points selected. Try again.")
                plt.close('all')
                continue
            triangle = np.array(pts)
            xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
            points = np.vstack((xx.flatten(), yy.flatten())).T
            tri_path = Path(triangle)
            tri_mask = tri_path.contains_points(points).reshape(rows, cols)
            if action == 'add':
                mask[tri_mask] = True
            elif action == 'block':
                mask[tri_mask] = False
        else:
            print("Invalid shape selection.")
            plt.close('all')
            continue

        plt.close('all')
        print("Mask updated.\n")
    
    # Save the final mask.
    np.save(file_name + ".npy", mask)
    cv2.imwrite(file_name + ".png", (mask.astype(np.uint8) * 255))
    print(f"Mask saved as '{file_name}.npy' and '{file_name}.png'")
    return mask