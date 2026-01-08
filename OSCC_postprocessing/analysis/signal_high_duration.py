import numpy as np

def mad(x):
    """Compute the Median Absolute Deviation of an array."""
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12

def hysteresis_threshold(y, th_lo, th_hi):
    """
    Hysteresis thresholding of a 1D signal. > th_hi triggers high state. < th_lo triggers low state.
    Parameters:
        y (np.ndarray): 1D input signal.
        th_lo (float): Low threshold.
        thi_hi (float): High threshold.
    """

    high    = y > th_hi
    low     = y < th_lo

    mask = np.zeros_like(y, dtype=bool)
    state = False
    