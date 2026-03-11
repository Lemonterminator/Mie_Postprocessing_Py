# NVIDIA Hardware Optical Flow Setup on Windows

This guide installs and verifies the NVIDIA Optical Flow SDK backend used by
`OSCC_postprocessing.motion.compute_nvidia_hw_flows`.

## What this backend is

This repository now supports three optical-flow backends through the unified API:

- `farneback`: classic OpenCV CPU optical flow
- `raft`: deep-learning RAFT optical flow
- `nvidia_hw`: NVIDIA hardware optical-flow engine from the Optical Flow SDK

The unified frontend is:

- [optical_flow.py](/C:/Users/Jiang/Documents/Mie_Py/Mie_Postprocessing_Py/OSCC_postprocessing/motion/optical_flow.py)

The NVIDIA backend implementation is:

- [nvidia_hw_optical_flow.py](/C:/Users/Jiang/Documents/Mie_Py/Mie_Postprocessing_Py/OSCC_postprocessing/motion/nvidia_hw_optical_flow.py)

## Requirements

- Windows 10 or 11
- NVIDIA GPU with Optical Flow hardware support
  - Turing or newer is the practical target
- Recent NVIDIA display driver
- Visual Studio 2022 with C++ build tools
- CUDA Toolkit installed
- CuPy installed in the project environment

## Download and install

1. Install the NVIDIA display driver.
   - Check with `nvidia-smi`

2. Install Visual Studio 2022 Community or Build Tools with:
   - Desktop development with C++

3. Install the CUDA Toolkit.
   - Check with `nvcc --version`

4. Download the NVIDIA Optical Flow SDK from the official page:
   - https://developer.nvidia.com/opticalflow-sdk

5. Extract it somewhere stable, for example:
   - `C:\Users\Jiang\Documents\Mie_Py\optical_flow\nvidia_optical_flow\Optical_Flow_SDK_5.0.7`

## Build and verify

From the repo root, run:

```powershell
.\scripts\setup_nvidia_optical_flow_windows.ps1 -BuildBridge
```

If your SDK is in a different directory:

```powershell
.\scripts\setup_nvidia_optical_flow_windows.ps1 `
  -SdkRoot "D:\SDKs\Optical_Flow_SDK_5.0.7" `
  -BuildBridge
```

The script checks:

- GPU driver
- CUDA toolkit
- SDK path
- CuPy import
- native bridge build
- runtime capability query
- smoke test on a small CuPy video

## Unified API usage

### Automatic backend selection

```python
from OSCC_postprocessing.motion import compute_optical_flows

flows = compute_optical_flows(video, backend="auto")
```

Behavior:

- CuPy `float16` or `float32` input prefers `nvidia_hw`
- everything else falls back to `farneback`

### Force the NVIDIA backend

```python
import cupy as cp
from OSCC_postprocessing.motion import compute_optical_flows

video = cp.asarray(video_np, dtype=cp.float32)
flows = compute_optical_flows(
    video,
    backend="nvidia_hw",
    out_hw_last=True,
    preset="medium",
    grid_size=1,
)
```

Output shape:

- `out_hw_last=True`: `(F-1, H, W, 2)`
- `out_hw_last=False`: `(F-1, 2, H, W)`

### Direct NVIDIA API

```python
import cupy as cp
from OSCC_postprocessing.motion import compute_nvidia_hw_flows, get_nvidia_hw_optical_flow_caps

print(get_nvidia_hw_optical_flow_caps())

video = cp.random.random((10, 256, 256), dtype=cp.float32)
flows = compute_nvidia_hw_flows(video, preset="medium", grid_size=1)
```

## Masters-thesis compatibility layer

The existing thesis helper now delegates into the unified backend layer:

- [opticalFlow.py](/C:/Users/Jiang/Documents/Mie_Py/Mie_Postprocessing_Py/third_party/masters-thesis/opticalFlow.py)

Supported method strings there now include:

- `Farneback`
- `NVIDIA_HW`
- `NVIDIA_HW_OF`

Example:

```python
mag = runOpticalFlowCalculationWeighted(
    firstFrameNumber,
    video_strip,
    method="NVIDIA_HW",
)
```

## Notes

- The NVIDIA SDK does not provide Farneback, TV-L1, or Brox.
- It exposes NVIDIA's hardware optical-flow engine and related modes.
- The current bridge is optimized for grayscale CuPy input shaped `(F, H, W)`.
- Input dtype must be `float16` or `float32` for the `nvidia_hw` backend.
