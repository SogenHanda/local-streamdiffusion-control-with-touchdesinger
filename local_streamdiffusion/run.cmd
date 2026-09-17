@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Run setup.ps1 first.
    pause
    exit /b 1
)

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "DIFFUSERS_OFFLINE=1"
set "HF_HOME=%~dp0.cache\huggingface"

".venv\Scripts\python.exe" "app.py" %*

if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)

endlocal
