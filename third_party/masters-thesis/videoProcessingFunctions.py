################################
# Removes background from a frame using the first frame as reference
################################
def removeBackgroundSimple(video, first_frame, threshold=10):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]

        # Compute absolute difference between the frame and the background
        diff = cv2.absdiff(frame, first_frame)

        # Threshold the grayscale difference to create a binary mask
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Use the mask to extract the foreground from the original frame
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        video[i] = foreground

    return video 


def createBackgroundMask(first_frame, threshold=10):
    import numpy as np

    # Binary mask: 1 where difference is greater than threshold, else 0 (uint8)
    mask = (first_frame > threshold).astype(np.uint8) * 255

    return mask


################################
# Rotates each frame in the video by a specified angle
################################
def createRotatedVideo(video, angle):
    import cv2
    import numpy as np

    nframes, height, width = video.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_video = np.zeros_like(video)
    for i in range(nframes):
        rotated_frame = cv2.warpAffine(video[i], rotation_matrix, (width, height))
        rotated_video[i] = rotated_frame

        # experimental - try different interpolation and border modes
        # same results as above
        # rotated_frame = cv2.warpAffine(
        #     video[i],
        #     rotation_matrix,
        #     (width, height),
        #     flags=cv2.INTER_NEAREST,
        #     borderMode=cv2.BORDER_CONSTANT,
        #     borderValue=0,
        # )
        # rotated_video[i] = rotated_frame

    return rotated_video

################################
# Creates a video strip by extracting a symmetric band around spray_origin
################################
def createVideoStrip(video, spray_origin, strip_half_height=200):
    import numpy as np

    nframes, height, width = video.shape

    # Ensure integer row index
    if isinstance(spray_origin, (tuple, list, np.ndarray)) and len(spray_origin) >= 2:
        origin_row = spray_origin[1]
    else:
        origin_row = spray_origin
    origin_row = int(round(origin_row)) # type: ignore
    origin_row = max(0, min(height - 1, origin_row))

    # Compute maximum symmetric half-height possible
    max_above = min(strip_half_height, origin_row)
    max_below = min(strip_half_height, (height - 1) - origin_row)
    half_height = min(max_above, max_below)

    start_row = origin_row - half_height
    end_row = origin_row + half_height + 1

    strip_height = end_row - start_row
    video_strip = np.zeros((nframes, strip_height, width), dtype=video.dtype)

    for i in range(nframes):
        video_strip[i] = video[i, start_row:end_row, :]

    return video_strip

###############################
# Takes a video and returns the first frame with mean intensity above a threshold
################################
def findFirstFrame(video, threshold=10):
    import numpy as np
    nframes = video.shape[0]

    for i in range(1, nframes):
        frame = video[i]
        # Compute mean brightness
        mean_intensity = frame.mean()

        if mean_intensity > threshold:
            return i
        
    return 0  # Default to first frame if no suitable frame is found

################################
# Applies a binary threshold to each frame in the video, UNUSED
################################
def removeBackgroundThreshold(video, threshold=30):
    #Consider making frame specific
    import cv2
    import numpy as np
    
    nframes = video.shape[0]
    for i in range(nframes):
        frame = video[i]
        _, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        video[i] = frame

    return video

################################
# Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to each frame in the video
################################
def applyCLAHE(video, clipLimit=2.0, tileGridSize=(8,8)):
    import cv2
    import numpy as np

    nframes = video.shape[0]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    for i in range(nframes):
        frame = video[i]
        frame = clahe.apply(frame)
        video[i] = frame

    return video

def applyLaplacianFilter(video):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = cv2.Laplacian(frame, cv2.CV_64F)
        frame = cv2.convertScaleAbs(frame)
        video[i] = frame

    return video

def applyDoGfilter(video, ksize1=5, ksize2=9):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        blur1 = cv2.GaussianBlur(frame, (ksize1, ksize1), 0)
        blur2 = cv2.GaussianBlur(frame, (ksize2, ksize2), 0)
        dog = cv2.subtract(blur1, blur2)
        dog = cv2.convertScaleAbs(dog)
        video[i] = dog

    return video

def adaptiveGaussianThreshold(video, maxValue=255, blockSize=11, C=2):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = cv2.adaptiveThreshold(frame, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blockSize, C)
        video[i] = frame

    return video


def OtsuThreshold(video, background_mask):
    import cv2
    import numpy as np

    out = video.copy()  
    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]

        # Otsu binarization
        _, otsu_frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Alternative fixed thresholding
        # _, otsu_frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

        if background_mask is None:
            # No mask provided: return Otsu result
            foreground = otsu_frame
        else:
            bg = background_mask

            # Normalize mask to uint8 0/255 format for XOR
            if bg.dtype != np.uint8:
                bg = bg.astype(np.uint8)
            if bg.max() == 1:
                bg = bg * 255

            # If a single mask (not per-frame) was passed, allow it to be used for all frames
            # If a per-frame list/array was passed, pick the corresponding frame mask
            if isinstance(bg, (list, tuple)):
                bg_frame = bg[i]
            elif bg.ndim == 3 and bg.shape[0] == nframes:
                bg_frame = bg[i]
            else:
                bg_frame = bg

            # Resize mask if necessary to match frame shape
            if bg_frame.shape != otsu_frame.shape:
                bg_frame = cv2.resize(bg_frame, (otsu_frame.shape[1], otsu_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # XOR Otsu result with background mask
            foreground = cv2.bitwise_xor(otsu_frame, bg_frame)

        out[i] = foreground

    return out



def invertVideo(video):
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = 255 - frame
        video[i] = frame

    return video


# chan-vese segmentation is too slow, consider other methods
def chanVeseSegmentation(video):
    from skimage.segmentation import chan_vese
    import numpy as np
    import cv2
    from skimage import img_as_float # type: ignore

    nframes = video.shape[0]


    for i in range(120, 150): # Limiting to frames 120-150 for speed during testing
        frame = video[i]

        # Convert to float
        img_f = img_as_float(frame)

        # Smooth to suppress schlieren texture
        img_smooth = cv2.GaussianBlur(img_f, (0, 0), 5)

        # Apply Chan-Vese segmentation
        cv_result = chan_vese(
                img_smooth,
                mu=0.3,                # contour smoothness
                lambda1=1.0,           # inside-region weight
                lambda2=1.0,           # outside-region weight
                tol=1e-3,
                max_num_iter=200,
                dt=0.5,
                init_level_set="checkerboard"
        )
        cv_result = (cv_result * 255).astype(np.uint8)  # type: ignore

        video[i] = cv_result

    return video

def temporalMedianFilter(video, firstFrameNumber):
    import cv2
    import numpy as np

    frames = np.stack(video[firstFrameNumber:firstFrameNumber+30], axis=0)
    background = np.median(frames, axis=0)

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        diff = np.abs(frame - background)
        mask = diff

        video[i] = mask

    return video


# Adaptive background subtraction with online update
# only really works well for videos with static background
# which would probably work better with simpler methods
def adaptive_background_subtraction(
    video,
    start=0,
    n_bg_frames=10,
    alpha=0.01,
    blur_sigma=3,
    thresh_percentile=80
):
    import numpy as np
    import cv2

    # ---- INITIAL BACKGROUND (temporal median) ----
    bg_frames = video[start:start + n_bg_frames]
    bg_frames = np.array([
        cv2.GaussianBlur(f, (0,0), blur_sigma) for f in bg_frames
    ])
    background = np.median(bg_frames, axis=0).astype(np.float32)

    plume_masks = []
    eps = 1e-3

    for frame in video:
        # ---- Preprocess ----
        frame_blur = cv2.GaussianBlur(frame, (0,0), blur_sigma).astype(np.float32)

        # ---- Normalized intensity difference ----
        diff_i = cv2.absdiff(frame_blur, background)
        diff_i /= (background + eps)

        # ---- Gradient difference ----
        gx_f = cv2.Sobel(frame_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy_f = cv2.Sobel(frame_blur, cv2.CV_32F, 0, 1, ksize=3)

        gx_b = cv2.Sobel(background, cv2.CV_32F, 1, 0, ksize=3)
        gy_b = cv2.Sobel(background, cv2.CV_32F, 0, 1, ksize=3)

        diff_g = np.sqrt((gx_f - gx_b)**2 + (gy_f - gy_b)**2)

        # ---- Combined foreground score ----
        score = diff_i + 0.7 * diff_g

        # ---- Threshold (percentile-based) ----
        thresh = np.percentile(score, thresh_percentile)
        plume_mask = score > thresh

        plume_masks.append(plume_mask.astype(np.uint8) * 255)

        # ---- ADAPTIVE BACKGROUND UPDATE (masked) ----
        background_mask = ~plume_mask
        background[background_mask] = (
            (1 - alpha) * background[background_mask]
            + alpha * frame_blur[background_mask]
        )

    # background.astype(np.uint8),
    return np.stack(plume_masks)


def SVDfiltering(video, k=10):
    import numpy as np

    nframes, height, width = video.shape

    # Reshape to (nframes, height*width)
    video_reshaped = video.reshape(nframes, -1).astype(np.float32)

    # SVD
    U, s, Vt = np.linalg.svd(video_reshaped, full_matrices=False)

    # Reconstruct background with top k components
    S_k = np.diag(s[:k])
    background = U[:, :k] @ S_k @ Vt[:k, :]

    # Subtract background
    foreground = video_reshaped - background

    # Clip to 0-255 and convert back to uint8
    foreground = np.clip(foreground, 0, 255).astype(np.uint8)

    # Reshape back
    video_filtered = foreground.reshape(nframes, height, width)

    return video_filtered


def tags_segmentation(spray_img, background_img, cell_size=5, n_bins=9, norm_order=1):
    import cv2
    import numpy as np
    """
    Implementation of the TAGS method for spray image segmentation.
    
    Args:
        spray_img: The current spray image (grayscale).
        background_img: The corresponding background image (grayscale).
        cell_size: Dimension of the local cell (e.g., 3x3)[cite: 126].
        n_bins: Number of orientation bins (Dimension N)[cite: 140].
        norm_order: Norm order (p) for difference assessment (1 for L1, 2 for L2)[cite: 149].
    """
    
    def get_gradient_statistics_vectors(img):
        # 1. Gamma Correction (Square root) to enhance dark regions 
        img_gamma = np.sqrt(img.astype(np.float32))
        
        # 2. Gradient Calculation (Gx and Gy) [cite: 134]
        # Using simple subtraction as per Eq 1 and 2 in the study
        gx = cv2.copyMakeBorder(img_gamma, 0, 0, 1, 1, cv2.BORDER_REPLICATE)
        gx = gx[:, 2:] - gx[:, :-2]
        
        gy = cv2.copyMakeBorder(img_gamma, 1, 1, 0, 0, cv2.BORDER_REPLICATE)
        gy = gy[2:, :] - gy[:-2, :]
        
        # 3. Polar Coordinate Conversion [cite: 134, 136]
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # 4. Orientation Statistics (Cell Partition & Histogram) [cite: 139, 142]
        h, w = img.shape
        half_cell = cell_size // 2
        statistics_volume = np.zeros((h, w, n_bins), dtype=np.float32)
        
        # Define bin width (e.g., 40 degrees for 9 bins) [cite: 140]
        bin_width = 360.0 / n_bins
        
        # Calculate histograms for each pixel's cell
        for i in range(n_bins):
            # Mask for current orientation bin [cite: 141]
            lower = i * bin_width
            upper = (i + 1) * bin_width
            bin_mask = (angle >= lower) & (angle < upper)
            bin_magnitude = np.where(bin_mask, magnitude, 0)
            
            # Sum magnitudes in the local cell [cite: 142]
            # boxFilter effectively sums values in a cell_size window
            statistics_volume[:, :, i] = cv2.boxFilter(bin_magnitude, -1, 
                                                       (cell_size, cell_size), 
                                                       normalize=False)

        # 5. Normalization [cite: 143, 144]
        sum_v = np.sum(statistics_volume, axis=2, keepdims=True)
        # Avoid division by zero
        vn = np.divide(statistics_volume, sum_v, 
                       out=np.zeros_like(statistics_volume), 
                       where=sum_v != 0)
        return vn

    # Calculate Normalized Gradient Statistics Vectors (Vn) for both images
    vn_spray = get_gradient_statistics_vectors(spray_img)
    vn_bg = get_gradient_statistics_vectors(background_img)

    # 6. Difference Assessment (p-norm) [cite: 148, 149]
    diff = np.abs(vn_spray - vn_bg)
    if norm_order == 1:
        diff_map = np.sum(diff, axis=2) # L1-norm [cite: 150, 151]
    else:
        diff_map = np.power(np.sum(np.power(diff, norm_order), axis=2), 1.0/norm_order)

    # 7. Threshold Binarization using Otsu's Method [cite: 155, 157]
    # Scale diff_map to 0-255 for OpenCV's Otsu
    diff_map_8u = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(diff_map_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_mask, diff_map
