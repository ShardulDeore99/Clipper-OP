"""Transcript data helpers: SRT output and a stable JSON shape. No model code here."""
import json
import textwrap
from pathlib import Path


def srt_time(t: float) -> str:
    ms = round(max(t, 0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def make_segment(idx: int, start: float, end: float, words: list[dict]) -> dict:
    return {"id": idx, "start": round(start, 3), "end": round(end, 3),
            "text": " ".join(w["w"] for w in words), "words": words}


def to_srt(segments: list[dict], width: int = 42) -> str:
    cues = []
    for n, seg in enumerate(segments, 1):
        text = "\n".join(textwrap.wrap(seg["text"], width)) or seg["text"]
        cues.append(f"{n}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{text}\n")
    return "\n".join(cues)


def write_transcript(folder: Path, data: dict) -> tuple[Path, Path]:
    js, srt = folder / "transcript.json", folder / "transcript.srt"
    js.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    srt.write_text(to_srt(data["segments"]), encoding="utf-8")
    return js, srt
