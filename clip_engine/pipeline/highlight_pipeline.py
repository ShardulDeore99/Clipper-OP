"""Phase 1 entry point for authorized local video or WAV analysis."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..detectors.audio_detector import analyze_wav


def analyze_media(path: str | Path, ffmpeg: str = "ffmpeg") -> dict:
    """Analyze audio from a local recording without modifying the source file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Media file does not exist: {source}")
    if source.suffix.lower() == ".wav":
        result = analyze_wav(source)
    else:
        with tempfile.TemporaryDirectory(prefix="fragbytez-audio-") as tmp:
            wav = Path(tmp) / "analysis.wav"
            proc = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                                   "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                                  capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0 or not wav.is_file():
                detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no output audio track"
                raise RuntimeError(f"Could not extract audio with FFmpeg: {detail}")
            result = analyze_wav(wav)
    return {"source": str(source), "media_kind": "wav" if source.suffix.lower() == ".wav" else "video",
            "audio_analysis": result}
