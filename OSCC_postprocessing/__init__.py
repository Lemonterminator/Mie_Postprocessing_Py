"""Mie Postprocessing utilities.

Submodules include helpers for masking, cone angle computation, filters,
cine I/O, async saving, and more. Import what you need, e.g.:

    from OSCC_postprocessing.io.async_plot_saver import AsyncPlotSaver

The package is intentionally light on top-level imports to keep
import time down; pull from submodules directly for specific tools.
"""

from __future__ import annotations

import os
from pathlib import Path


_CUDA_DLL_DIR_HANDLES: list[object] = []
_CUDA_PRELOAD_HANDLES: list[object] = []


def _add_dll_dir(path: Path) -> None:
    """Add a DLL directory and keep the handle alive (Windows)."""

    if not path.is_dir():
        return
    try:
        handle = os.add_dll_directory(str(path))
        _CUDA_DLL_DIR_HANDLES.append(handle)
    except Exception:
        return


def _windows_add_cuda_dll_dirs() -> None:
    """Best-effort add CUDA DLL directories on Windows.

    CuPy on Windows needs NVRTC (`nvrtc64_*.dll`) available on the DLL search
    path at runtime. When installed via pip, NVRTC can live under
    `site-packages/nvidia/cuda_nvrtc/bin`, which is not always searched by
    default.
    """

    if os.name != "nt":
        return

    def _preload_nvrtc_from(cuda_bin: Path) -> None:
        try:
            import ctypes

            cuda_bin_x64 = cuda_bin / "x64"
            if not cuda_bin_x64.is_dir():
                return

            nvrtc = sorted(cuda_bin_x64.glob("nvrtc64_*.dll"))
            builtins = sorted(cuda_bin_x64.glob("nvrtc-builtins64_*.dll"))
            if not nvrtc:
                return

            _CUDA_PRELOAD_HANDLES.append(ctypes.CDLL(str(nvrtc[0])))
            if builtins:
                _CUDA_PRELOAD_HANDLES.append(ctypes.CDLL(str(builtins[0])))
        except Exception:
            return

    try:
        cuda_root = None
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            cuda_root = Path(cuda_path)
        else:
            default_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            if default_root.is_dir():
                candidates = [p for p in default_root.iterdir() if p.is_dir() and p.name.lower().startswith("v")]
                if candidates:
                    # Pick the newest version folder (lexicographic works for vMAJOR.MINOR).
                    cuda_root = sorted(candidates, key=lambda p: p.name.lower())[-1]

        if cuda_root is None or not cuda_root.is_dir():
            return

        # Help CuPy locate the CUDA toolkit for NVRTC compilation.
        os.environ.setdefault("CUDA_PATH", str(cuda_root))

        cuda_bin = cuda_root / "bin"
        _add_dll_dir(cuda_bin)
        _add_dll_dir(cuda_bin / "x64")

        # NVRTC loads its builtins DLL via its own loader; on Windows it can fail
        # unless the DLLs are loaded from full paths first (so their directory is
        # considered a DLL load dir). Preloading fixes that reliably.
        _preload_nvrtc_from(cuda_bin)
    except Exception:
        return

    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if spec and spec.submodule_search_locations:
            cuda_nvrtc_root = Path(list(spec.submodule_search_locations)[0])
            cuda_nvrtc_bin = cuda_nvrtc_root / "bin"
            _add_dll_dir(cuda_nvrtc_bin)

        nvidia_spec = importlib.util.find_spec("nvidia")
        if nvidia_spec and nvidia_spec.submodule_search_locations:
            nvidia_root = Path(list(nvidia_spec.submodule_search_locations)[0])
            if nvidia_root.is_dir():
                for child in nvidia_root.iterdir():
                    bin_dir = child / "bin"
                    _add_dll_dir(bin_dir)
    except Exception:
        pass


_windows_add_cuda_dll_dirs()
from .motion.optical_flow import compute_raft_flows, compute_farneback_flows  # Optical flow wrappers

__all__ = [
    'compute_raft_flows',
    'compute_farneback_flows',
]

