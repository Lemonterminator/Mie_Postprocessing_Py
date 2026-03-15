from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Any


try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore


_DLL_HANDLE: ctypes.CDLL | None = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_SOURCE_DIR = _REPO_ROOT / "native" / "nvidia_of_bridge"
_BRIDGE_BUILD_DIR = _REPO_ROOT / "build" / "nvidia_of_bridge"
_BRIDGE_BUILD_DIR_NINJA = _REPO_ROOT / "build" / "nvidia_of_bridge_ninja2"
_DEFAULT_SDK_ROOT = Path(
    os.environ.get(
        "NVIDIA_OPTICAL_FLOW_SDK_DIR",
        r"C:\Users\Jiang\Documents\Mie_Py\optical_flow\nvidia_optical_flow\Optical_Flow_SDK_5.0.7",
    )
)


def _dll_candidates() -> list[Path]:
    env = os.environ.get("NVIDIA_OF_BRIDGE_DLL")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))

    candidates.extend(
        [
            _BRIDGE_BUILD_DIR / "Release" / "nvidia_of_bridge.dll",
            _BRIDGE_BUILD_DIR / "Debug" / "nvidia_of_bridge.dll",
            _BRIDGE_BUILD_DIR / "RelWithDebInfo" / "nvidia_of_bridge.dll",
            _BRIDGE_BUILD_DIR / "nvidia_of_bridge.dll",
            _BRIDGE_BUILD_DIR_NINJA / "nvidia_of_bridge.dll",
        ]
    )
    return candidates


def _find_vcvars64() -> Path:
    candidates = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate vcvars64.bat for Visual Studio 2022")


def _load_bridge() -> ctypes.CDLL:
    global _DLL_HANDLE
    if _DLL_HANDLE is not None:
        return _DLL_HANDLE

    for candidate in _dll_candidates():
        if candidate.is_file():
            if os.name == "nt":
                os.add_dll_directory(str(candidate.parent))
            lib = ctypes.CDLL(str(candidate))
            lib.nvidia_of_get_last_error.restype = ctypes.c_char_p
            lib.nvidia_of_compute_flow.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            lib.nvidia_of_compute_flow.restype = ctypes.c_int
            lib.nvidia_of_get_caps.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.nvidia_of_get_caps.restype = ctypes.c_int
            _DLL_HANDLE = lib
            return lib

    raise FileNotFoundError(
        "nvidia_of_bridge.dll not found. Run build_nvidia_hw_optical_flow_bridge() first."
    )


def build_nvidia_hw_optical_flow_bridge(
    *,
    sdk_root: str | os.PathLike[str] | None = None,
    build_dir: str | os.PathLike[str] | None = None,
    config: str = "Release",
) -> Path:
    """Configure and build the native NVIDIA Optical Flow bridge."""

    sdk_path = Path(sdk_root) if sdk_root is not None else _DEFAULT_SDK_ROOT
    out_dir = Path(build_dir) if build_dir is not None else _BRIDGE_BUILD_DIR_NINJA
    out_dir.mkdir(parents=True, exist_ok=True)

    vcvars = _find_vcvars64()
    nvcc = Path(os.environ.get("CUDACXX", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"))

    configure_cmd = (
        f'call "{vcvars}" && '
        f'cmake -S "{_BRIDGE_SOURCE_DIR}" -B "{out_dir}" -G Ninja '
        f'-DCMAKE_CUDA_COMPILER="{nvcc}" -DNVOF_SDK_ROOT="{sdk_path}"'
    )
    build_cmd = f'call "{vcvars}" && cmake --build "{out_dir}"'

    subprocess.run(["cmd.exe", "/c", configure_cmd], check=True)
    subprocess.run(["cmd.exe", "/c", build_cmd], check=True)

    dll_path = out_dir / "nvidia_of_bridge.dll"
    if not dll_path.is_file():
        raise FileNotFoundError(f"Build completed but DLL was not found: {dll_path}")
    return dll_path


def get_nvidia_hw_optical_flow_caps(*, device_id: int | None = None) -> dict[str, Any]:
    """Return supported grid sizes and ROI support for the current CUDA device."""

    if cp is None:
        raise ImportError("cupy is required for NVIDIA hardware optical flow support")

    lib = _load_bridge()
    device = int(cp.cuda.Device().id if device_id is None else device_id)
    grids = (ctypes.c_uint32 * 8)()
    grid_count = ctypes.c_int()
    roi_supported = ctypes.c_int()

    rc = lib.nvidia_of_get_caps(
        device,
        grids,
        8,
        ctypes.byref(grid_count),
        ctypes.byref(roi_supported),
    )
    if rc != 0:
        raise RuntimeError(lib.nvidia_of_get_last_error().decode("utf-8", "replace"))

    return {
        "device_id": device,
        "supported_grid_sizes": [int(grids[i]) for i in range(grid_count.value)],
        "roi_supported": bool(roi_supported.value),
        "mode": "hardware_optical_flow",
    }


def _infer_input_scale(video: "cp.ndarray", input_max: float | None) -> float:
    if input_max is not None:
        if input_max <= 0:
            raise ValueError("input_max must be positive")
        return 255.0 / float(input_max)

    vmax = float(cp.max(video).item()) if video.size else 1.0
    if vmax <= 0.0:
        return 1.0
    if vmax <= 1.0 + 1e-6:
        return 255.0
    if vmax <= 255.0 + 1e-6:
        return 1.0
    return 255.0 / vmax


def compute_nvidia_hw_flows(
    video: "cp.ndarray",
    *,
    input_max: float | None = None,
    preset: str = "slow",
    grid_size: int = 1,
    enable_temporal_hints: bool = True,
    device_id: int | None = None,
) -> "cp.ndarray":
    """Compute pairwise NVIDIA hardware optical flow on a CuPy video.

    Parameters
    ----------
    video:
        CuPy array of shape ``(F, H, W)`` with dtype ``float16`` or ``float32``.
    input_max:
        Optional explicit max intensity corresponding to 255 after quantization.
    preset:
        One of ``slow``, ``medium``, ``fast``.
    grid_size:
        Requested output grid. ``1`` returns dense ``(H, W, 2)`` flow per pair.
    enable_temporal_hints:
        Keep the hardware temporal state enabled across adjacent frame pairs.
    device_id:
        CUDA device id. Defaults to the current CuPy device.

    Returns
    -------
    cupy.ndarray
        Float32 array of shape ``(F-1, H, W, 2)``.
    """

    if cp is None:
        raise ImportError("cupy is required for NVIDIA hardware optical flow support")
    if not isinstance(video, cp.ndarray):
        raise TypeError("video must be a cupy.ndarray")
    if video.ndim != 3:
        raise ValueError(f"Expected a (F, H, W) array, got {video.shape}")
    if video.shape[0] < 2:
        raise ValueError("Need at least 2 frames")
    if video.dtype not in (cp.float16, cp.float32):
        raise TypeError("video dtype must be float16 or float32")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    preset_map = {
        "slow": 5,
        "medium": 10,
        "fast": 20,
    }
    try:
        preset_value = preset_map[preset.lower()]
    except KeyError as exc:
        raise ValueError("preset must be one of: slow, medium, fast") from exc

    lib = _load_bridge()
    device = int(cp.cuda.Device().id if device_id is None else device_id)

    with cp.cuda.Device(device):
        video_c = cp.ascontiguousarray(video)
        flows = cp.empty((video_c.shape[0] - 1, video_c.shape[1], video_c.shape[2], 2), dtype=cp.float32)
        stream = cp.cuda.get_current_stream()
        stream.synchronize()

        dtype_code = 0 if video_c.dtype == cp.float16 else 1
        input_scale = _infer_input_scale(video_c, input_max)

        rc = lib.nvidia_of_compute_flow(
            ctypes.c_void_p(int(video_c.data.ptr)),
            ctypes.c_int(dtype_code),
            ctypes.c_int(int(video_c.shape[0])),
            ctypes.c_int(int(video_c.shape[1])),
            ctypes.c_int(int(video_c.shape[2])),
            ctypes.c_float(float(input_scale)),
            ctypes.c_int(device),
            ctypes.c_int(preset_value),
            ctypes.c_int(int(grid_size)),
            ctypes.c_int(1 if enable_temporal_hints else 0),
            ctypes.c_uint64(int(stream.ptr)),
            ctypes.c_void_p(int(flows.data.ptr)),
        )
        if rc != 0:
            raise RuntimeError(lib.nvidia_of_get_last_error().decode("utf-8", "replace"))

        stream.synchronize()
        return flows
