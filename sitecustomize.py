"""Windows CUDA DLL helper.

This repository uses CuPy optionally. On Windows, some CUDA runtime DLLs (e.g.
NVRTC) can be provided via pip packages under `site-packages/nvidia/**/bin`,
but those directories are not always on the DLL search path by default.

Python auto-imports `sitecustomize` on startup (when `site` is enabled),
so we use it to add the relevant folders when running from this repo root.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path


def _add_dll_dir(path: Path) -> None:
    if not path.is_dir():
        return
    try:
        os.add_dll_directory(str(path))  # Python 3.8+ (Windows)
    except Exception:
        return


def _maybe_add_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return

    warnings.filterwarnings(
        "ignore",
        message=r"CUDA path could not be detected.*",
        category=UserWarning,
    )

    repo_root = Path(__file__).resolve().parent
    venv_site = repo_root / ".venv" / "Lib" / "site-packages"
    nvidia_root = venv_site / "nvidia"
    if not nvidia_root.is_dir():
        return

    # CuPy needs NVRTC at import time; other CUDA DLLs can be added later if needed.
    _add_dll_dir(nvidia_root / "cuda_nvrtc" / "bin")

    # Best-effort: add any other NVIDIA-provided `bin` folders present in the venv.
    for child in nvidia_root.iterdir():
        _add_dll_dir(child / "bin")


_maybe_add_cuda_dll_dirs()
