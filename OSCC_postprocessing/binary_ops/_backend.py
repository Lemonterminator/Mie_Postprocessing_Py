"""Backend helpers shared by binary post-processing utilities."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

try:  # pragma: no cover - runtime hardware dependent
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA path could not be detected.*",
            category=UserWarning,
        )
        import cupy as cp  # type: ignore
        import cupyx.scipy.ndimage as cndi  # type: ignore

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - runtime hardware dependent
    cp = np  # type: ignore
    cndi = ndimage  # type: ignore
    cp.asnumpy = lambda x: x  # type: ignore[attr-defined]
    CUPY_AVAILABLE = False

try:  # pragma: no cover - runtime hardware dependent
    from cucim.skimage import measure as cucim_measure  # type: ignore

    CUCIM_AVAILABLE = True
except Exception:  # pragma: no cover - runtime hardware dependent
    cucim_measure = None
    CUCIM_AVAILABLE = False


def to_numpy_host(arr):
    """Return a NumPy array on host memory regardless of backend."""
    if hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_cupy(arr):
    """Convert NumPy input to CuPy when available; pass CuPy through."""
    if CUPY_AVAILABLE and hasattr(arr, "__cuda_array_interface__"):
        return arr
    return cp.asarray(arr) if CUPY_AVAILABLE else arr


def return_like_input(arr, like):
    """Return output with the same backend and dtype convention as ``like``."""
    if CUPY_AVAILABLE and hasattr(like, "__cuda_array_interface__"):
        return arr.astype(like.dtype, copy=False)
    if CUPY_AVAILABLE:
        return cp.asnumpy(arr).astype(like.dtype, copy=False)
    return arr.astype(like.dtype, copy=False)
