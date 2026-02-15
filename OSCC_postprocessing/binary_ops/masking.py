from OSCC_postprocessing.analysis.multihole_utils import resolve_backend
import cv2
import numpy as np


use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

def generate_angular_mask_from_tf(H, W, centre, TF, bins):
    """
    Generates a 2D mask where pixels are set to the value of TF 
    corresponding to their angular bin relative to the centre.
    """
    # Create a grid of (y, x) coordinates
    # xp is cupy or numpy depending on your backend configuration
    y, x = xp.indices((H, W))
    
    # Calculate the difference from the centre
    # centre is (x, y)
    dx = x - centre[0]
    dy = y - centre[1]
    
    # Calculate angles in degrees [0, 360)
    # arctan2 returns values in [-pi, pi]
    angles = xp.degrees(xp.arctan2(dy, dx))
    angles = (angles + 360) % 360
    
    # Convert angles to bin indices
    # bins is the total number of angular bins (e.g., 720)
    bin_indices = (angles / 360.0 * bins).astype(xp.int32)
    
    # Handle edge case where angle is exactly 360 (though modulo usually handles this)
    bin_indices = xp.clip(bin_indices, 0, bins - 1)
    
    # Create the mask by indexing the TF array with the bin indices
    # If TF is 1 for plume and 0 for background, this mask will be 1 for plume pixels.
    mask = TF[bin_indices]
    
    return mask


def periodic_true_segment_lengths(mask: xp.ndarray) -> xp.ndarray:
    """
    Return the length (in bins) of each contiguous ``True`` run in a periodic 1D mask.

    Parameters
    ----------
    mask : xp.ndarray
        1D boolean mask (e.g. the TF vector used for angular masking). The array
        is treated as periodic, so runs that wrap from the end back to the start
        are counted as a single segment.

    Returns
    -------
    xp.ndarray
        Array of positive integers giving the bin length of each occupied segment.
        Segments are returned in the order they appear when traversing the mask
        once from index 0 to ``len(mask) - 1``.

    Examples
    --------
    >>> periodic_true_segment_lengths([False, True, True, False, True])
    array([2, 1])
    >>> periodic_true_segment_lengths([True, True, True])
    array([3])
    """
    mask_bool = xp.asarray(mask, dtype=bool).ravel()
    n_bins = mask_bool.size
    if n_bins == 0:
        return xp.empty(0, dtype=xp.int64)

    if mask_bool.all():
        return xp.array([n_bins], dtype=xp.int64)

    if not mask_bool.any():
        return xp.empty(0, dtype=xp.int64)

    mask_ext = xp.concatenate((mask_bool, mask_bool[:1]))
    diffs = mask_ext[1:].astype(xp.int8) - mask_ext[:-1].astype(xp.int8)
    starts = xp.nonzero(diffs == 1)[0] + 1
    ends = xp.nonzero(diffs == -1)[0] + 1
    lengths = (ends - starts) % n_bins
    return lengths.astype(xp.int64)


def periodic_true_segment_angles(mask: xp.ndarray, bins: int | None = None) -> xp.ndarray:
    """
    Convert the periodic True-segment lengths into angular extent in degrees.

    Parameters
    ----------
    mask : xp.ndarray
        1D boolean mask (e.g. TF) describing occupied angular bins.
    bins : int, optional
        Total number of bins used when defining the mask. Defaults to ``len(mask)``.

    Returns
    -------
    xp.ndarray
        Angular widths (in degrees) of each contiguous ``True`` segment.

    Notes
    -----
    The sum of the returned angles is equivalent to ``(TF.sum() / bins) * 360``.
    """
    mask_array = xp.asarray(mask)
    lengths = periodic_true_segment_lengths(mask_array)
    if lengths.size == 0:
        return xp.empty(0, dtype=float)

    total_bins = mask_array.size
    bins = total_bins if bins is None else bins
    if bins <= 0:
        raise ValueError("bins must be a positive integer")

    return lengths.astype(float) * 360.0 / bins


def generate_ring_mask(H, W, centre, ir_, or_):
    """
    Generates a boolean mask for a ring defined by inner and outer radii.
    
    Parameters:
    -----------
    H, W : int
        Height and Width of the image/mask.
    centre : tuple
        (x, y) coordinates of the center.
    ir_ : float
        Inner radius.
    or_ : float
        Outer radius.
    xp : module
        The array module to use (numpy or cupy).
        
    Returns:
    --------
    mask : xp.ndarray
        Boolean mask where True indicates pixels within the ring.
    """
    # Create a grid of (y, x) coordinates
    y, x = xp.indices((H, W))
    
    # Calculate the squared distance from the centre
    # centre is (x, y)
    dx = x - centre[0]
    dy = y - centre[1]
    dist_sq = dx**2 + dy**2
    
    # Create the mask where distance is between ir_ and or_
    # Using squared distances avoids computing square roots for efficiency
    mask = (dist_sq >= ir_**2) & (dist_sq <= or_**2)
    
    return mask

def generate_plume_mask(w, h, angle=None, x0=0, y0=None):

    if y0 is None:
        y0 = h/2

    if angle is None:
        y1 = 0
        y2 = h
    else:
        half_angle_radian = angle / 2.0 * np.pi/180.0
        y1 = -w * np.tan(half_angle_radian) + h/2
        y2 = w * np.tan(half_angle_radian) + h/2

    # Create blank single-channel mask of same height/width
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define polygon vertices as Nx2 integer array  
    pts = np.array([[x0, y0], [w, y1], [w, y2]], dtype=np.int32)
    
    # Fill the polygon on the mask
    cv2.fillPoly(mask, [pts], (255,))

    # cv2.imshow("plume_mask", mask) # Debug

    # Apply mask to extract polygon region
    return mask >0 