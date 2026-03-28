param(
    [string]$SdkRoot,
    [string]$PythonExe,
    [switch]$BuildBridge
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-ExternalCommand([string]$Description, [string]$FilePath, [string[]]$Arguments = @()) {
    Write-Step $Description
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Resolve-PythonCommand([string]$RequestedPythonExe) {
    if ($RequestedPythonExe) {
        if (Test-Path $RequestedPythonExe) {
            return @{
                Exe = (Resolve-Path $RequestedPythonExe).Path
                Args = @()
            }
        }
        $command = Get-Command $RequestedPythonExe -ErrorAction SilentlyContinue
        if ($command) {
            return @{
                Exe = $command.Source
                Args = @()
            }
        }
        throw "Python executable not found: $RequestedPythonExe"
    }

    $repoVenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $repoVenvPython) {
        return @{
            Exe = $repoVenvPython
            Args = @()
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            Exe = $python.Source
            Args = @()
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @{
            Exe = $pyLauncher.Source
            Args = @("-3")
        }
    }

    throw "Could not locate Python. Pass -PythonExe or create .venv\Scripts\python.exe."
}

function Resolve-SdkRoot([string]$RequestedSdkRoot) {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($RequestedSdkRoot) {
        $candidates.Add($RequestedSdkRoot)
    }
    if ($env:NVIDIA_OPTICAL_FLOW_SDK_DIR) {
        $candidates.Add($env:NVIDIA_OPTICAL_FLOW_SDK_DIR)
    }

    $packagesDir = Join-Path $RepoRoot "packages"
    if (Test-Path $packagesDir) {
        foreach ($candidate in (Get-ChildItem $packagesDir -Directory -Filter "Optical_Flow_SDK*" | Sort-Object Name -Descending)) {
            $candidates.Add($candidate.FullName)
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "NvOFInterface\nvOpticalFlowCommon.h")) {
            return (Resolve-Path $candidate).Path
        }
    }

    $searched = if ($candidates.Count -gt 0) { $candidates -join ", " } else { "<none>" }
    throw "Could not locate a valid NVIDIA Optical Flow SDK root. Searched: $searched"
}

function Invoke-PythonCode([string]$Description, [string]$Code) {
    Write-Step $Description
    & $script:PythonCommand.Exe @($script:PythonCommand.Args + @("-c", $Code))
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-BridgeDllCandidates() {
    return @(
        (Join-Path $RepoRoot "build\nvidia_of_bridge_ninja2\nvidia_of_bridge.dll"),
        (Join-Path $RepoRoot "build\nvidia_of_bridge\Release\nvidia_of_bridge.dll"),
        (Join-Path $RepoRoot "build\nvidia_of_bridge\RelWithDebInfo\nvidia_of_bridge.dll"),
        (Join-Path $RepoRoot "build\nvidia_of_bridge\Debug\nvidia_of_bridge.dll"),
        (Join-Path $RepoRoot "build\nvidia_of_bridge\nvidia_of_bridge.dll")
    )
}

$script:PythonCommand = Resolve-PythonCommand $PythonExe
$ResolvedSdkRoot = Resolve-SdkRoot $SdkRoot
$env:NVIDIA_OPTICAL_FLOW_SDK_DIR = $ResolvedSdkRoot

Invoke-ExternalCommand "Checking GPU driver" "nvidia-smi"
Invoke-ExternalCommand "Checking CUDA toolkit" "nvcc" @("--version")
Invoke-PythonCode "Checking Python environment" "import sys; import cupy; import OSCC_postprocessing; print(sys.executable); print('cupy', cupy.__version__)"

$ExistingBridgeDll = Get-BridgeDllCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$ShouldBuildBridge = $BuildBridge -or (-not $ExistingBridgeDll)
if ($ShouldBuildBridge) {
    Invoke-PythonCode "Building native NVIDIA optical flow bridge" "import os; from OSCC_postprocessing.motion.nvidia_hw_optical_flow import build_nvidia_hw_optical_flow_bridge; print(build_nvidia_hw_optical_flow_bridge(sdk_root=os.environ['NVIDIA_OPTICAL_FLOW_SDK_DIR']))"
} else {
    Write-Step "Using existing native NVIDIA optical flow bridge"
    Write-Host $ExistingBridgeDll -ForegroundColor DarkGray
}

Invoke-PythonCode "Querying NVIDIA optical-flow capabilities" "from OSCC_postprocessing.motion import get_nvidia_hw_optical_flow_caps; print(get_nvidia_hw_optical_flow_caps())"
Invoke-PythonCode "Running smoke test" "import cupy as cp; from OSCC_postprocessing.motion import compute_nvidia_hw_flows; x=cp.random.random((3,64,64), dtype=cp.float32); y=compute_nvidia_hw_flows(x); print(y.shape, y.dtype)"

Write-Host ""
Write-Host "Setup verification completed." -ForegroundColor Green
