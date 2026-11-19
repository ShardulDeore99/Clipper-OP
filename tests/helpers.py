import subprocess
import tempfile
from pathlib import Path

from streamclipper.config import load_config


def temp_cfg(tmp: Path):
    return load_config(overrides={"general": {"data_dir": str(tmp / "data")}})


def make_sample(path: Path, seconds: int = 6, audio: bool = True) -> Path:
    """Tiny 720p test video generated with ffmpeg (video + optional sine tone)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc=size=1280x720:rate=30:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else []) + [str(path)]
    subprocess.run(cmd, check=True)
    return path


def make_talk_video(path: Path, seconds: int = 60, burst_at=30) -> Path:
    """720p video; quiet noise bed with loud 3 s bursts (the 'funny moments'). burst_at: seconds or a list."""
    bursts = [burst_at] if isinstance(burst_at, int) else list(burst_at)
    inputs, parts = [], []
    for k, b in enumerate(bursts):
        inputs += ["-f", "lavfi", "-i", "sine=frequency=300:duration=3:sample_rate=48000"]
        parts.append(f"[{k + 2}:a]adelay={b * 1000}|{b * 1000},apad=whole_dur={seconds},volume=0.6[s{k}]")
    mix = "[1:a]" + "".join(f"[s{k}]" for k in range(len(bursts))) + f"amix=inputs={len(bursts) + 1}:duration=first:dropout_transition=0,volume={len(bursts) + 1}[a]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=30:duration={seconds}",
                    "-f", "lavfi", "-i", f"anoisesrc=amplitude=0.01:duration={seconds}:sample_rate=48000", *inputs,
                    "-filter_complex", ";".join(parts + [mix]), "-map", "0:v", "-map", "[a]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path
