"""Stage 2: extract 16 kHz mono WAV for Whisper, and record basic video facts."""
from pathlib import Path

from .. import cache, media
from ..errors import StageError

NAME = "audio"
FILE_KEYS = ("wav",)


def cache_inputs(ctx) -> dict:
    video = Path(ctx.out["video"])
    a = ctx.cfg["audio"]
    return {"video": str(video), "size": video.stat().st_size,
            "sample_rate": a["sample_rate"], "channels": a["channels"]}


def run(ctx, progress) -> dict:
    video = Path(ctx.out["video"])
    info = media.probe(ctx.cfg, video, NAME)
    if info["duration"] <= 0:
        raise StageError(NAME, "Could not read a duration from this file.", "Is it a valid video?")
    if not info["has_audio"]:
        raise StageError(NAME, "This video has no audio track.",
                         "There is nothing to transcribe. Pick a source with speech.")
    wav = cache.stage_dir(ctx.cfg, ctx.key, NAME) / "audio.wav"
    a = ctx.cfg["audio"]
    media.run([ctx.cfg["tools"]["ffmpeg"], "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(video), "-vn", "-ac", str(a["channels"]), "-ar", str(a["sample_rate"]),
               "-c:a", "pcm_s16le", str(wav)], NAME)
    if not wav.is_file() or wav.stat().st_size < 1000:
        raise StageError(NAME, "Audio extraction produced an empty file.")
    return {"wav": str(wav), **info}
