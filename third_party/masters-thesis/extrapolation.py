import cv2
import numpy as np

class SprayConeBackfill:
    def __init__(self, origin, min_points=50):
        self.origin = np.array(origin, dtype=np.float32)
        self.min_points = min_points

    def backfill(self, detected_mask):
        h, w = detected_mask.shape
        ox, oy = self.origin

        # Only consider right half for detection
        xs, ys = np.where(detected_mask[:, w//2:] > 0)
        if len(xs) < self.min_points:
            return np.zeros_like(detected_mask)

        # Shift xs back to full image coordinates
        xs = xs + w // 2
        points = np.column_stack((xs, ys))

        # Find closest detected x to origin
        first_x = np.min(xs)

        # Distance from origin to first detection
        D = first_x - ox
        if D <= 5:
            return np.zeros_like(detected_mask)

        # Estimate width at first detection
        near_mask = np.abs(xs - first_x) < 3
        width = np.ptp(ys[near_mask])

        if width < 2:
            return np.zeros_like(detected_mask)

        half_width = width / 2.0
        theta = np.arctan2(half_width, D)

        # Create backfill mask
        backfill = np.zeros_like(detected_mask)
        center_y = oy

        for x in range(int(ox), int(first_x)):
            dx = x - ox
            if dx <= 0:
                continue

            curr_half_width = np.tan(theta) * dx
            y1 = int(center_y - curr_half_width)
            y2 = int(center_y + curr_half_width)

            if y2 < 0 or y1 >= h:
                continue

            y1 = max(0, y1)
            y2 = min(h - 1, y2)

            backfill[y1:y2, x] = 255

        return backfill

import cv2
import numpy as np

# def extrapolate_cone(mask, spray_origin, min_points=20):
#     """
#     Extrapolate a cone from the spray origin based on the visible mask region.

#     Parameters
#     ----------
#     mask : np.ndarray
#         Binary mask (0/255 or 0/1).
#     spray_origin : tuple
#         (x, y) coordinate of the spray nozzle.
#     min_points : int
#         Minimum number of mask points near the nozzle required to compute angles.

#     Returns
#     -------
#     final_mask : np.ndarray
#         Mask with extrapolated cone added.
#     """

#     h, w = mask.shape[:2]
#     ox, oy = spray_origin

#     # Ensure mask is binary 0/1
#     m = (mask > 0).astype(np.uint8)

#     # --- 1. Extract points near the nozzle (left side of the mask) ---
#     # Take a vertical slice near the nozzle (e.g., first 10% of width)
#     slice_width = max(5, w // 10)
#     region = m[:, :slice_width]

#     ys, xs = np.where(region > 0)

#     if len(xs) < min_points:
#         # Not enough visible spray near the nozzle → return original mask
#         return mask.copy()

#     # Convert local xs to global xs
#     xs_global = xs
#     ys_global = ys

#     # --- 2. Compute angles of detected spray points relative to nozzle ---
#     angles = np.arctan2(ys_global - oy, xs_global - ox)

#     min_angle = np.min(angles)
#     max_angle = np.max(angles)

#     # --- 3. Create a cone mask by scanning all pixels and checking angle ---
#     yy, xx = np.mgrid[0:h, 0:w]
#     ang = np.arctan2(yy - oy, xx - ox)

#     cone_mask = ((ang >= min_angle) & (ang <= max_angle)).astype(np.uint8)

#     # --- 4. Combine with original mask ---
#     final_mask = np.clip(m + cone_mask, 0, 1).astype(np.uint8) * 255

#     return final_mask

import cv2
import numpy as np

def extrapolate_cone(mask, spray_origin, min_points=20):
    """
    Extrapolate a cone from the spray origin based on the visible mask region.
    The cone:
      - uses angles from the leftmost part of the detected mask
      - does not extend beyond (leftmost_x + 10% of width)
    """

    h, w = mask.shape[:2]
    ox, oy = spray_origin

    # Ensure binary 0/1
    m = (mask > 0).astype(np.uint8)

    # --- 1. Find all detected mask pixels ---
    all_ys, all_xs = np.where(m > 0)
    if len(all_xs) == 0:
        return mask.copy()

    # Leftmost detected x (closest to nozzle along x)
    leftmost_x = np.min(all_xs)

    # --- 2. Define a band near the leftmost side of the mask ---
    slice_width = max(5, w // 10)
    band_x_min = leftmost_x
    band_x_max = min(w - 1, leftmost_x + slice_width)

    yy_band, xx_band = np.where(
        (m > 0) &
        (np.arange(w)[None, :] >= band_x_min) &
        (np.arange(w)[None, :] <= band_x_max)
    )

    if len(xx_band) < min_points:
        # Not enough points to estimate angles reliably
        return mask.copy()

    # --- 3. Compute angular spread from spray origin to band points ---
    angles = np.arctan2(yy_band - oy, xx_band - ox)
    min_angle = np.min(angles)
    max_angle = np.max(angles)

    # --- 4. Compute maximum allowed x for extrapolation ---
    max_extension = int(0.05 * w)
    x_max = min(w - 1, leftmost_x + max_extension)

    # --- 5. Build cone mask over full frame ---
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - oy, xx - ox)

    cone_mask = (
        (ang >= min_angle) &
        (ang <= max_angle) &
        (xx >= ox) &          # only to the right of the nozzle
        (xx <= x_max)         # stop at leftmost_x + 10% width
    ).astype(np.uint8)

    # --- 6. Combine with original mask ---
    final_mask = np.clip(m + cone_mask, 0, 1).astype(np.uint8) * 255

    return final_mask