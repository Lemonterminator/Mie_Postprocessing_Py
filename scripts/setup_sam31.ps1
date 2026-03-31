param(
    [string]$Token = "",
    [string]$PythonBin = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "WSL is not available. Native Windows install is currently blocked because Meta's sam3 import chain requires Triton, and pip does not provide a native Windows triton wheel."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoRootForBash = $repoRoot.Replace("\", "/")
$repoRootWsl = (wsl.exe bash -lc "wslpath -a '$repoRootForBash'").Trim()
$scriptPath = "$repoRootWsl/scripts/setup_sam31_wsl.sh"

$tokenPrefix = ""
if ($Token) {
    $tokenPrefix = "HF_TOKEN='$Token' "
}

$pythonPrefix = ""
if ($PythonBin) {
    $pythonPrefix = "PYTHON_BIN='$PythonBin' "
}

$cmd = "cd '$repoRootWsl' && ${tokenPrefix}${pythonPrefix}bash '$scriptPath'"
Write-Host "[info] Running in WSL: $cmd"
wsl.exe bash -lc $cmd
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
