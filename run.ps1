# One-command run: sets up on first use, then starts the web UI and opens your browser.
Set-Location $PSScriptRoot
# WinGet's portable FFmpeg install is not always added to the user PATH automatically.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  $ffmpegPackage = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe'
  $ffmpegRelease = Get-ChildItem -LiteralPath $ffmpegPackage -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^ffmpeg-.+-full_build$' } |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
  if ($ffmpegRelease) {
    $ffmpegBin = Join-Path $ffmpegRelease.FullName 'bin'
    if ((Test-Path (Join-Path $ffmpegBin 'ffmpeg.exe')) -and (Test-Path (Join-Path $ffmpegBin 'ffprobe.exe'))) {
      $env:Path = "$ffmpegBin;$env:Path"
    }
  }
}
if (-not (Test-Path .\.venv\Scripts\python.exe)) { .\setup.ps1 }
& .\.venv\Scripts\python -m streamclipper serve @args
