"""Unified array-backend selection for the whole :mod:`OSCC_postprocessing` package.

This module is intended to be the single source of truth for deciding whether a
piece of code should run on NumPy/CPU or CuPy/GPU. Historically the package had
multiple local backend detectors, each with slightly different rules for:

- whether CuPy import success is enough to call the GPU available
- whether to use a real ``cupy`` module or a NumPy-compatible fallback object
- how to decide the array namespace (``xp``) for a given array
- how to express preferred-vs-actual backend choices

That fragmentation easily leads to subtle bugs:

- one module binds ``xp = cupy`` at import time, but another later passes a
  NumPy array into it
- one helper treats ``cupy`` import success as GPU-ready, while another also
  checks device visibility or ``cupyx`` availability
- callers mix ``np``, ``cp``, and ``xp`` assumptions in the same pipeline

The goal here is to centralize those rules and expose a small set of functions
that every submodule can reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class _NumpyCompat:
    """NumPy object that mimics the small CuPy surface used by this codebase.

    The package contains many call sites written against a ``cp`` namespace.
    When CUDA is unavailable we still want those sites to keep working without
    branching everywhere, so this shim forwards most attribute access to
    :mod:`numpy` and implements a few CuPy-like helpers such as ``asnumpy`` and
    ``get``.
    """

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


def get_cupy():
    """Return the real CuPy module when it is both importable and runtime-usable.

    This function is stricter than a plain ``import cupy``:

    1. import CuPy
    2. query the CUDA runtime via ``getDeviceCount``

    If either step fails we return ``None`` instead of a partially usable CuPy
    object. That keeps higher-level code from assuming GPU execution is safe
    when the runtime is actually broken or absent.
    """
    try:  # pragma: no cover - hardware dependent
        import cupy as cupy  # type: ignore

        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:
        return None


_REAL_CUPY = get_cupy()
USING_CUPY = _REAL_CUPY is not None
cp = _REAL_CUPY if _REAL_CUPY is not None else _NumpyCompat()  # type: ignore[assignment]
xp = cp if USING_CUPY else np


@dataclass(frozen=True)
class BackendSpec:
    """Resolved runtime backend choice for a computation.

    Attributes
    ----------
    use_gpu:
        Whether the resolved execution path should use GPU arrays.
    triangle_backend:
        Which thresholding implementation should be preferred by callers that
        support both CPU and GPU triangle thresholding.
    xp:
        The array namespace to use for new allocations and math.
    cp:
        The CuPy-compatible namespace. This is either the real :mod:`cupy`
        module or the local :class:`_NumpyCompat` shim.
    np:
        The canonical NumPy module, included for convenience so callers can
        keep related backend objects together in one struct.
    """

    use_gpu: bool
    triangle_backend: str
    xp: Any
    cp: Any
    np: Any = np

    @property
    def name(self) -> str:
        return "cupy" if self.use_gpu else "numpy"


def has_cupy_gpu() -> tuple[bool, str]:
    """Return whether CuPy can see and use a GPU on this machine.

    The returned string is intended for diagnostics and UI messages rather than
    strict program logic.
    """
    cupy = get_cupy()
    if cupy is None:
        return False, "CuPy import failed"
    try:
        ndev = cupy.cuda.runtime.getDeviceCount()
        if ndev <= 0:
            return False, "No CUDA/HIP devices found"
        with cupy.cuda.Device(0):
            _ = cupy.asarray([0]).sum()
    except Exception as exc:  # pragma: no cover - hardware dependent
        return False, f"CuPy GPU unavailable: {exc}"
    return True, f"CuPy ready on {ndev} device(s)"


def is_cupy_array(arr) -> bool:
    """Return ``True`` when ``arr`` is a live CuPy-backed array."""
    return USING_CUPY and hasattr(arr, "__cuda_array_interface__")


def to_numpy(arr):
    """Convert any NumPy/CuPy-like array to a host NumPy array."""
    return cp.asnumpy(arr) if is_cupy_array(arr) else np.asarray(arr)


def get_array_module(arr):
    """Return the active array namespace for ``arr``.

    This is the preferred helper when writing functions that should operate on
    either NumPy or CuPy input without hard-coding a global ``xp`` decided at
    module import time.
    """
    if is_cupy_array(arr):
        return cp
    return np


def resolve_backend(
    use_gpu: str | bool = "auto",
    triangle_backend: str = "auto",
) -> tuple[bool, str, Any]:
    """
    Decide the preferred array backend and triangle-threshold implementation.

    Returns ``(use_gpu, triangle_backend, xp)`` for backward compatibility.

    New code should prefer :func:`get_backend_spec`, which keeps all backend
    information bundled together and is less error-prone than passing several
    loosely-related values around.
    """
    spec = get_backend_spec(use_gpu=use_gpu, triangle_backend=triangle_backend)
    return spec.use_gpu, spec.triangle_backend, spec.xp


def get_backend_spec(
    use_gpu: str | bool = "auto",
    triangle_backend: str = "auto",
) -> BackendSpec:
    """
    Resolve the active runtime backend into one structured object.

    This is the preferred entry point for new code because it keeps ``np``,
    ``cp``, ``xp``, and threshold backend decisions in one place.

    Resolution strategy
    -------------------
    1. Determine whether GPU usage is requested or allowed.
    2. Verify a real CuPy runtime is available.
    3. If GPU is requested, also verify the specific dependencies needed by
       common downstream image-processing code, such as ``cupyx.scipy.ndimage``.
    4. If any check fails, fall back to CPU/NumPy and force triangle backend to
       ``"cpu"``.

    This policy is intentionally conservative: we prefer a predictable CPU
    fallback over partially resolved GPU state.
    """
    cupy = get_cupy()
    gpu_ok = cupy is not None
    if use_gpu == "auto":
        resolved_gpu = gpu_ok
    else:
        resolved_gpu = bool(use_gpu and gpu_ok)

    if triangle_backend == "auto":
        resolved_triangle = "gpu" if resolved_gpu else "cpu"
    elif triangle_backend == "gpu" and not resolved_gpu:
        resolved_triangle = "cpu"
    else:
        resolved_triangle = triangle_backend

    xp_backend = np
    cp_backend = cp
    if resolved_gpu and cupy is not None:
        try:
            from cupyx.scipy.ndimage import median_filter as _check  # noqa: F401
        except Exception:
            resolved_gpu = False
            resolved_triangle = "cpu"
            xp_backend = np
            cp_backend = _NumpyCompat()
        else:
            xp_backend = cupy
            cp_backend = cupy

    return BackendSpec(
        use_gpu=resolved_gpu,
        triangle_backend=resolved_triangle,
        xp=xp_backend,
        cp=cp_backend,
    )


def get_preferred_xp(use_gpu: str | bool = "auto"):
    """Return only the resolved array namespace when a full spec is unnecessary."""
    return get_backend_spec(use_gpu=use_gpu).xp


__all__ = [
    "BackendSpec",
    "USING_CUPY",
    "cp",
    "get_array_module",
    "get_backend_spec",
    "get_cupy",
    "get_preferred_xp",
    "has_cupy_gpu",
    "is_cupy_array",
    "resolve_backend",
    "to_numpy",
    "xp",
]
