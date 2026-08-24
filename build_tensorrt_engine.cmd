@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv-trt\Scripts\python.exe" (
    echo TensorRT environment not found. Run setup_tensorrt.ps1 first.
    pause
    exit /b 1
)

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "DIFFUSERS_OFFLINE=1"
set "HF_HOME=%~dp0.cache\huggingface"
set "PYTHONUTF8=1"

".venv-trt\Scripts\python.exe" "build_tensorrt_engine.py" %*

if errorlevel 1 (
    echo.
    echo TensorRT build failed.
    pause
)

endlocal
