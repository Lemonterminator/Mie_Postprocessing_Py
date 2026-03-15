from __future__ import annotations

import numpy as np


class _NumpyCompat:
    """Small NumPy shim that mimics the CuPy bits used in the codebase."""

    ndarray = np.ndarray

    def __getattr__(self, name):
        return getattr(np, name)

    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype=dtype)

    def asnumpy(self, a):
        return np.asarray(a)

    def get(self, a):
        return a

    def get_array_module(self, arr):
        return np


try:  # pragma: no cover - hardware dependent
    import cupy as _cupy  # type: ignore

    _cupy.cuda.runtime.getDeviceCount()
    cp = _cupy
    USING_CUPY = True
except Exception as exc:  # pragma: no cover - hardware dependent
    print(f"CuPy unavailable, falling back to NumPy backend: {exc}")
    cp = _NumpyCompat()  # type: ignore[assignment]
    USING_CUPY = False


def get_cupy():
    """Return the real CuPy module if import succeeds, otherwise ``None``."""
    try:  # pragma: no cover - hardware dependent
        import cupy as cupy  # type: ignore

        return cupy
    except Exception:
        return None


def is_cupy_array(arr) -> bool:
    return USING_CUPY and hasattr(arr, "__cuda_array_interface__")


def to_numpy(arr):
    return cp.asnumpy(arr) if is_cupy_array(arr) else np.asarray(arr)


def get_array_module(arr):
    if is_cupy_array(arr):
        return cp
    return np


__all__ = [
    "USING_CUPY",
    "cp",
    "get_array_module",
    "get_cupy",
    "is_cupy_array",
    "to_numpy",
]
