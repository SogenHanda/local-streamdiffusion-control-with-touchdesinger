[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentDirectory = Join-Path $projectDirectory ".venv-trt"
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
        Write-Host "Creating the isolated TensorRT Python 3.10 environment..." -ForegroundColor Cyan
        & $launcher.Source -3.10 -m venv $environmentDirectory
    }
    elseif ($python -and (Test-Python310 -Executable $python.Source -Arguments @())) {
        Write-Host "Creating the isolated TensorRT Python 3.10 environment..." -ForegroundColor Cyan
        & $python.Source -m venv $environmentDirectory
    }
    else {
        throw "Python 3.10 (64-bit) is required."
    }
}

Write-Host "Installing the pinned StreamDiffusion runtime..." -ForegroundColor Cyan
& $environmentPython -m pip install --upgrade "pip==23.3.2" "setuptools==69.0.3" "wheel==0.42.0"
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
& $environmentPython -m pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
& $environmentPython -m pip install "xformers==0.0.22.post7" --no-deps
if ($LASTEXITCODE -ne 0) { throw "xFormers installation failed." }
& $environmentPython -m pip install -r (Join-Path $projectDirectory "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Application dependency installation failed." }

Write-Host "Installing TensorRT 9 and ONNX build dependencies..." -ForegroundColor Cyan
& $environmentPython -m pip install "protobuf==3.20.2" "cuda-python==12.3.0" "onnx==1.15.0" "onnxruntime==1.16.3" "colored==2.2.4" "pywin32"
if ($LASTEXITCODE -ne 0) { throw "ONNX/TensorRT support dependency installation failed." }
& $environmentPython -m pip install "nvidia-cudnn-cu12==8.9.4.25" "nvidia-cublas-cu12==12.9.2.10" "nvidia-cuda-nvrtc-cu12==12.9.86" --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw "cuDNN installation failed." }
& $environmentPython -m pip install --pre --extra-index-url https://pypi.nvidia.com "tensorrt==9.0.1.post11.dev4" --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw "TensorRT 9 installation failed." }
& $environmentPython -m pip install "polygraphy==0.48.1" "onnx-graphsurgeon==0.5.2"
if ($LASTEXITCODE -ne 0) { throw "Polygraphy/ONNX GraphSurgeon installation failed." }

Write-Host "Verifying TensorRT..." -ForegroundColor Cyan
& $environmentPython (Join-Path $projectDirectory "verify_tensorrt.py")
if ($LASTEXITCODE -ne 0) {
    throw "TensorRT verification failed. See the messages above."
}

Write-Host ""
Write-Host "TensorRT setup is complete." -ForegroundColor Green
Write-Host "1. Run build_tensorrt_engine.cmd once per model/resolution/step/GPU."
Write-Host "2. Start the application with run_tensorrt.cmd."
