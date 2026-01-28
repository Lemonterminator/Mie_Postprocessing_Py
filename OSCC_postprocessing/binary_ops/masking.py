from OSCC_postprocessing.analysis.multihole_utils import resolve_backend

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


def generate_ring_mask(H, W, centre, ir_, or_, xp):
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
