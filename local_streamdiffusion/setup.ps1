[CmdletBinding()]
param(
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentDirectory = Join-Path $projectDirectory ".venv"
$environmentPython = Join-Path $environmentDirectory "Scripts\python.exe"

Set-Location -LiteralPath $projectDirectory

function Test-Python310 {
    param([string]$Executable, [string[]]$Arguments)
    try {
        $versionText = & $Executable @Arguments -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return $versionText -eq "3.10"
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    $python = Get-Command python -ErrorAction SilentlyContinue

    if ($launcher -and (Test-Python310 -Executable $launcher.Source -Arguments @("-3.10"))) {
        Write-Host "Creating the Python 3.10 environment..." -ForegroundColor Cyan
        & $launcher.Source -3.10 -m venv $environmentDirectory
    }
    elseif ($python -and (Test-Python310 -Executable $python.Source -Arguments @())) {
        Write-Host "Creating the Python 3.10 environment..." -ForegroundColor Cyan
        & $python.Source -m venv $environmentDirectory
    }
    else {
        throw "Python 3.10 (64-bit) is required. Install it from https://www.python.org/downloads/release/python-31011/ with Add python.exe to PATH enabled, then run setup.ps1 again."
    }
}

Write-Host "Updating pip..." -ForegroundColor Cyan
& $environmentPython -m pip install --upgrade "pip==23.3.2" "setuptools==69.0.3" "wheel==0.42.0"

Write-Host "Installing CUDA 12.1 PyTorch..." -ForegroundColor Cyan
& $environmentPython -m pip install `
    "torch==2.1.0" `
    "torchvision==0.16.0" `
    --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing xFormers..." -ForegroundColor Cyan
& $environmentPython -m pip install "xformers==0.0.22.post7" --no-deps

Write-Host "Installing application dependencies..." -ForegroundColor Cyan
& $environmentPython -m pip install -r (Join-Path $projectDirectory "requirements.txt")

Write-Host "Verifying the installation..." -ForegroundColor Cyan
& $environmentPython (Join-Path $projectDirectory "verify_install.py")

if (-not $SkipModels) {
    Write-Host "Downloading models for offline use (several GB)..." -ForegroundColor Cyan
    & $environmentPython (Join-Path $projectDirectory "download_models.py")
}

Write-Host "" 
Write-Host "Setup is complete. Start the application with run.ps1." -ForegroundColor Green
