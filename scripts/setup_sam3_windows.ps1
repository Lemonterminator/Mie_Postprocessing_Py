param(
    [string]$VenvPath = ".venv",
    [string]$KernelName = "mie-postprocessing-sam3",
    [string]$KernelDisplayName = "Python (.venv) - SAM3",
    [string]$ModelDir = ".models/sam3_hf",
    [string]$Token = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvFullPath = Join-Path $repoRoot $VenvPath
$pythonExe = Join-Path $venvFullPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "[info] Creating virtual environment at $venvFullPath"
    py -3.12 -m venv $venvFullPath
}

Write-Host "[info] Using Python: $pythonExe"

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -e .
& $pythonExe -m pip install "transformers>=5.4.0" requests ipykernel
& $pythonExe -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName

if ($Token) {
    $env:HF_TOKEN = $Token
}

& $pythonExe (Join-Path $PSScriptRoot "download_sam3.py") --dest $ModelDir
& $pythonExe (Join-Path $PSScriptRoot "run_sam3_smoketest.py")

Write-Host "[ok] SAM3 environment setup finished."
Write-Host "[ok] Open examples/Sam3.ipynb and select kernel '$KernelDisplayName'."
