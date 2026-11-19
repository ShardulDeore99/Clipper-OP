"""Thin wrappers around ffmpeg/ffprobe. Every failure raises StageError with the real reason."""
import json
import subprocess
from pathlib import Path

from .errors import StageError


def run(cmd: list[str], stage: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except FileNotFoundError:
        raise StageError(stage, f"{cmd[0]} was not found.",
                         "Install FFmpeg and add it to PATH, or set [tools] in config.toml.")
    except subprocess.TimeoutExpired:
        raise StageError(stage, f"{Path(cmd[0]).name} timed out after {timeout:.0f}s.")
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-6:])
        raise StageError(stage, f"{Path(cmd[0]).name} failed (exit {p.returncode}): {tail}")
    return p


def _fps(rate: str) -> float:
    try:
        n, d = rate.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except (ValueError, AttributeError):
        return 0.0


def probe(cfg, path, stage: str) -> dict:
    p = run([cfg["tools"]["ffprobe"], "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(path)], stage)
    data = json.loads(p.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "width": video.get("width", 0),
        "height": video.get("height", 0),
        "fps": _fps(video.get("avg_frame_rate", "0/1")),
    }
