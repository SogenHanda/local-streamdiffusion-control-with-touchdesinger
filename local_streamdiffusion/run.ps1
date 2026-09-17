$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $environmentPython)) {
    throw "Python environment not found. Run setup.ps1 first."
}

Set-Location -LiteralPath $projectDirectory
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:DIFFUSERS_OFFLINE = "1"
$env:HF_HOME = Join-Path $projectDirectory ".cache\huggingface"

& $environmentPython (Join-Path $projectDirectory "app.py") @args
