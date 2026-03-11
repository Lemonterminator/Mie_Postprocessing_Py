param(
    [string]$SdkRoot = "C:\Users\Jiang\Documents\Mie_Py\optical_flow\nvidia_optical_flow\Optical_Flow_SDK_5.0.7",
    [string]$PythonExe = ".venv311\Scripts\python.exe",
    [switch]$BuildBridge
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Write-Step "Checking GPU driver"
nvidia-smi

Write-Step "Checking CUDA toolkit"
nvcc --version

Write-Step "Checking SDK path"
if (-not (Test-Path (Join-Path $SdkRoot "NvOFInterface\nvOpticalFlowCommon.h"))) {
    throw "Invalid SDK root: $SdkRoot"
}

Write-Step "Checking Python environment"
& $PythonExe -c "import sys; print(sys.executable)"
& $PythonExe -c "import cupy; import OSCC_postprocessing; print('cupy', cupy.__version__)"

$env:NVIDIA_OPTICAL_FLOW_SDK_DIR = $SdkRoot

if ($BuildBridge) {
    Write-Step "Building native NVIDIA optical flow bridge"
    & $PythonExe -c "from OSCC_postprocessing.motion.nvidia_hw_optical_flow import build_nvidia_hw_optical_flow_bridge; print(build_nvidia_hw_optical_flow_bridge(sdk_root=r'$SdkRoot'))"
}

Write-Step "Querying NVIDIA optical-flow capabilities"
& $PythonExe -c "from OSCC_postprocessing.motion import get_nvidia_hw_optical_flow_caps; print(get_nvidia_hw_optical_flow_caps())"

Write-Step "Running smoke test"
& $PythonExe -c "import cupy as cp; from OSCC_postprocessing.motion import compute_nvidia_hw_flows; x=cp.random.random((3,64,64), dtype=cp.float32); y=compute_nvidia_hw_flows(x); print(y.shape, y.dtype)"

Write-Host ""
Write-Host "Setup verification completed." -ForegroundColor Green
