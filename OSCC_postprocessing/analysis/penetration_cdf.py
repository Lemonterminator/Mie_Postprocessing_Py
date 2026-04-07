import numpy as np
from OSCC_postprocessing.utils.backend import get_array_module


def monotone_non_decreasing(x):
    return np.maximum.accumulate(x)


def penetration_cdf_front(I, mask=None, q=0.97, min_x=0):
    """
    I: (T, X)
    mask: optional ``(T, X)`` {0,1} array; only accumulate inside the spray region
    q: cumulative quantile, e.g. 0.95~0.995
    """
    xp = get_array_module(I)
    I = xp.asarray(I, xp.float32)
    T, X = I.shape

    # Remove per-frame background using a low quantile. This is more suitable
    # than the median when the image contains a large dark background and a
    # comparatively bright spray region.
    bg = xp.quantile(I, 0.10, axis=1, keepdims=True)
    S = I - bg
    S[S < 0] = 0

    if mask is not None:
        S = S * mask.astype(xp.float32)

    S[:, :min_x] = 0

    cdf = xp.cumsum(S, axis=1)
    tot = cdf[:, -1] + 1e-6
    target = q * tot

    xhat = xp.argmax(cdf >= target[:, None], axis=1).astype(xp.int32)
    # If a frame has almost no intensity, ``tot`` is near zero and ``argmax``
    # returns 0. Callers can post-process that case if needed.
    return xhat
