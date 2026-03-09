import math
from typing import List, Tuple
import cv2
import numpy as np

def analyze_boundary(mask: np.ndarray,*,angle_d: float,nozzle_point: np.ndarray,) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Analyze the spray boundary to extract cone geometry and penetration metrics.
    
    Args:
        mask: Binary mask of the spray region
        angle_d: Rotation angle in degrees (for alignment with spray axis)
        nozzle_point: (row, col) coordinates of the spray nozzle
    
    Returns:
        Tuple of:
        - coords_rc: Boundary contour coordinates (row, col)
        - penetration: Maximum distance from nozzle to boundary
        - cone_angle: Spray cone angle from extreme points
        - cone_angle_reg: Spray cone angle from regression fit
        - close_point_distance: Minimum distance from nozzle to boundary
    """
    # Handle empty mask case
    if not mask.any():
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, 0.0, 0.0, 0.0, 0.0

    # Extract contours from the spray mask
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, 0.0, 0.0, 0.0, 0.0

    # Combine all contours into a single boundary
    all_coords = np.vstack([contour[:, 0, :] for contour in contours])
    # Convert from OpenCV format (x, y) to (row, col) for spatial calculations
    coords_rc = np.column_stack((all_coords[:, 1], all_coords[:, 0])).astype(np.float32)

    # Calculate relative positions from nozzle point
    relative = coords_rc - nozzle_point[None, :]
    # Convert to mathematical coordinate system: x right (col), y up (flip row)
    # relative is (row_diff, col_diff), convert to (x_math, y_math)
    relative_math = np.column_stack((relative[:, 1], -(relative[:, 0])))

    # Calculate distances from nozzle to each boundary point
    distances = np.linalg.norm(relative_math, axis=1)
    if distances.size == 0:
        return coords_rc, 0.0, 0.0, 0.0, 0.0

    # Penetration: maximum distance from nozzle (spray reaches farthest at this distance)
    penetration = float(distances.max())
    # Close point distance: minimum distance from nozzle (excluding zero distance)
    positive_distances = distances[distances > 0]
    close_point_distance = float(positive_distances.min()) if positive_distances.size else 0.0

    # Rotate coordinate system to align with spray axis (forward direction)
    # This compensates for video rotation angle
    theta = math.radians(angle_d)
    rot = np.array([[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]],
                   dtype=np.float32)
    aligned = relative_math @ rot.T
    x_aligned = aligned[:, 0]
    y_aligned = aligned[:, 1]

    # Filter to only forward-pointing boundary points (in the spray direction)
    forward_mask = x_aligned > 0
    if not np.any(forward_mask):
        forward_mask = np.ones_like(x_aligned, dtype=bool)

    x_forward = x_aligned[forward_mask]
    y_forward = y_aligned[forward_mask]
    
    # Calculate cone angle from extreme angles in the spray cross-section
    angles = np.degrees(np.arctan2(y_forward, x_forward))
    if angles.size:
        cone_angle = float(angles.max() - angles.min())
    else:
        cone_angle = 0.0

    # Calculate cone angle using linear regression for more robust estimate
    cone_angle_reg = regression_cone_angle(x_forward, y_forward)

    return coords_rc, penetration, cone_angle, cone_angle_reg, close_point_distance


def regression_cone_angle(x_forward: np.ndarray, y_forward: np.ndarray) -> float:
    """
    Calculate spray cone angle using linear regression on top and bottom spray edges.
    
    Fits separate lines to upper and lower spray boundaries to estimate 
    the cone half-angles, then sums them for total cone angle.
    
    Args:
        x_forward: X-coordinates of forward-pointing boundary points
        y_forward: Y-coordinates of forward-pointing boundary points
    
    Returns:
        Total cone angle in degrees (sum of upper and lower half-angles)
    """
    if x_forward.size < 2:
        return 0.0

    # Split boundary into upper (y >= 0) and lower (y <= 0) portions
    top_mask = y_forward >= 0
    bottom_mask = y_forward <= 0

    angles: List[float] = []
    
    # Fit line to upper spray edge and extract angle
    if np.count_nonzero(top_mask) >= 2:
        m_top, _ = np.polyfit(x_forward[top_mask], y_forward[top_mask], 1)
        angles.append(abs(float(np.degrees(np.arctan(m_top)))))
    
    # Fit line to lower spray edge and extract angle
    if np.count_nonzero(bottom_mask) >= 2:
        m_bottom, _ = np.polyfit(x_forward[bottom_mask], y_forward[bottom_mask], 1)
        angles.append(abs(float(np.degrees(np.arctan(m_bottom)))))

    # Combine upper and lower half-angles
    if len(angles) == 2:
        return angles[0] + angles[1]
    if len(angles) == 1:
        # If only one edge detected, assume symmetric spray
        return 2.0 * angles[0]
    return 0.0
