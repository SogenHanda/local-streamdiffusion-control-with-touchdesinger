$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $environmentPython)) {
    throw "Python environment not found. Run setup.ps1 first."
}

Set-Location -LiteralPath $projectDirectory
& $environmentPython (Join-Path $projectDirectory "download_models.py") @args
