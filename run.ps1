# One-command run: sets up on first use, then starts the web UI and opens your browser.
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { .\setup.ps1 }
& .\.venv\Scripts\python -m streamclipper serve @args
