# Activate the project's virtual environment and start the FastAPI server
# Usage: Open PowerShell, then: .\run_server.ps1

$venvActivate = Join-Path $PSScriptRoot "animal\Scripts\Activate.ps1"
if (-Not (Test-Path $venvActivate)) {
    Write-Error "Virtualenv activate script not found: $venvActivate"
    exit 1
}

Write-Output "Activating virtualenv: $venvActivate"
& $venvActivate

Write-Output "Starting Uvicorn server on http://127.0.0.1:8000"
# Start without reloader to avoid automatic restarts closing the process
uvicorn serve.app:app --host 127.0.0.1 --port 8000
