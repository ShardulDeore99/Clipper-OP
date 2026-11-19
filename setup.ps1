# One-command setup: creates .venv, installs pinned dependencies, runs the startup check.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv .venv } else { python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ is required. Install it from python.org and re-run." }
& .\.venv\Scripts\python -m pip install --upgrade pip
& .\.venv\Scripts\python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed. See the message above." }
# Caption font: Anton (free, OFL). If this download fails, captions fall back to Impact / Arial Black.
New-Item -ItemType Directory -Force assets\fonts | Out-Null
if (-not (Test-Path assets\fonts\Anton-Regular.ttf)) {
  try { Invoke-WebRequest "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf" -OutFile assets\fonts\Anton-Regular.ttf -ErrorAction Stop }
  catch { Write-Warning "Could not download Anton. Put Anton-Regular.ttf into assets\fonts yourself, or captions will use Impact." }
}
& .\.venv\Scripts\python -m streamclipper doctor
Write-Host "Downloading Whisper models (one-time network use)..."
& .\.venv\Scripts\python -m streamclipper fetch-models
if ($LASTEXITCODE -ne 0) { throw "Model download failed. Check your internet connection and re-run setup.ps1." }
