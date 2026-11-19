"""M7 automatic checks. A flag always carries the reason shown in the UI."""
import re
import subprocess

from .boundaries import cut_end
from .errors import StageError
from .media import probe


def flag(code: str, message: str, severity: str = "warn") -> dict:
    return {"code": code, "message": message, "severity": severity}


def boundary_flags(words, starts, ends, start: float, end: float) -> list[dict]:
    """Starts/ends mid-sentence (or mid-word)? Judged on the real word boundaries, so it also works after manual trims."""
    if not words:
        return []
    out = []
    i = next((k for k, w in enumerate(words) if w["end"] > start + 0.02), None)
    j = next((k for k in range(len(words) - 1, -1, -1) if words[k]["start"] < end - 0.02), None)
    if i is None or j is None or j < i:
        return [flag("no_speech_in_range", "No words fall inside this range.")]
    sset = {p["i"] for p in starts}
    eset = {p["i"] for p in ends}
    if words[i]["start"] < start - 0.03:
        out.append(flag("start_mid_word", "The clip starts in the middle of a word."))
    elif i not in sset:
        out.append(flag("start_mid_sentence", f"The clip starts mid-sentence, at \"{words[i]['w']}\"."))
    if words[j]["end"] > end + 0.03:
        out.append(flag("end_mid_word", "The clip ends in the middle of a word."))
    elif j not in eset:
        out.append(flag("end_mid_sentence", f"The clip ends mid-sentence, after \"{words[j]['w']}\"."))
    return out


def edge_silence_flags(cfg, video, dur: float) -> list[dict]:
    """More than 1 s of silence at the start or end of the rendered clip."""
    p = subprocess.run([cfg["tools"]["ffmpeg"], "-hide_banner", "-nostats", "-i", str(video), "-af",
                        "silencedetect=noise=-45dB:d=1", "-f", "null", "-"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", p.stderr)]
    ends = [(float(a), float(b)) for a, b in re.findall(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", p.stderr)]
    out = []
    if starts and starts[0] <= 0.05:
        n = ends[0][1] if ends else dur
        out.append(flag("dead_air_start", f"{n:.1f} s of silence at the start."))
    if starts and (len(ends) < len(starts) or dur - (ends[-1][0] if ends else 0) < 0.1) and dur - starts[-1] >= 1.0:
        out.append(flag("dead_air_end", f"{dur - starts[-1]:.1f} s of silence at the end."))
    return out


def format_flags(cfg, video) -> tuple[list[dict], dict]:
    info = probe(cfg, video, "check")
    w, h = (int(x) for x in cfg["render"]["export_size"].split("x"))
    out = []
    if (info["width"], info["height"]) != (w, h):
        out.append(flag("wrong_size", f"Output is {info['width']}x{info['height']}, expected {w}x{h}.", "error"))
    return out, info


def loudness_flags(cfg, measured: dict | None) -> list[dict]:
    if measured is None:
        return [flag("no_audio_level", "Could not measure the audio level (silent clip?).")]
    target = cfg["render"]["target_lufs"]
    if abs(measured["input_i"] - target) > 2.0:
        return [flag("loudness", f"Loudness is {measured['input_i']:.1f} LUFS, target {target}.")]
    return []


def face_flags(framing: dict) -> list[dict]:
    v = framing.get("violations", 0.0)
    if v > 0.08:
        return [flag("face_cut_off", f"A face touches or leaves the crop edge in {v * 100:.0f}% of the sampled frames.")]
    return []
