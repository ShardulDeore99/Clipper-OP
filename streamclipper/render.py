"""M6: one encode per clip, seeking before the input, loudness-normalised, QSV when it really works."""
import json
import logging
import re
import subprocess
import time
from pathlib import Path

from .errors import StageError
from .media import run

log = logging.getLogger("streamclipper")
_QSV: dict = {}


def qsv_works(cfg) -> bool:
    """Listed is not the same as working: do a real 0.3 s encode once per ffmpeg path."""
    ff = cfg["tools"]["ffmpeg"]
    if ff not in _QSV:
        p = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
                            "-t", "0.3", "-c:v", "h264_qsv", "-f", "null", "-"], capture_output=True, timeout=30)
        _QSV[ff] = p.returncode == 0
    return _QSV[ff]


def parse_loudnorm(stderr: str) -> dict | None:
    blocks = re.findall(r"\{[^{}]*\}", stderr, re.S)
    if not blocks:
        return None
    try:
        d = json.loads(blocks[-1])
        if float(d["input_i"]) == float("-inf") or d["input_i"] in ("-inf", "inf"):
            return None
        return {k: float(d[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def measure_loudness(cfg, video, start: float | None = None, dur: float | None = None) -> dict | None:
    t = cfg["render"]
    cmd = [cfg["tools"]["ffmpeg"], "-hide_banner", "-nostats"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]
    cmd += ["-i", str(video), "-vn", "-af", f"loudnorm=I={t['target_lufs']}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    p = run(cmd, "render", timeout=max(60, (dur or 600) * 4))
    return parse_loudnorm(p.stderr)


def loudnorm_filter(cfg, m: dict | None) -> str | None:
    if not m:
        return None
    return (f"loudnorm=I={cfg['render']['target_lufs']}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")


def build_cmd(cfg, video, start, dur, filter_complex, has_subs, out, final, mode, afilter, qsv=False) -> list[str]:
    t = cfg["render"]
    fc = filter_complex + (";[v]subtitles=subs.ass:fontsdir=fonts[vo]" if has_subs else "")
    cmd = [cfg["tools"]["ffmpeg"], "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),          # seek BEFORE -i, cut only this range
           "-filter_complex", fc, "-map", "[vo]" if has_subs else "[v]", "-map", "0:a:0"]
    if final:
        cmd += ["-c:v", "libx264", "-preset", "medium" if mode == "quality" else "fast", "-crf", str(t["export_crf"])]
    elif qsv:
        cmd += ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", str(t["review_crf"] + 1)]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast" if mode == "fast" else "fast", "-crf", str(t["review_crf"])]
    cmd += ["-pix_fmt", "yuv420p", "-r", str(t["export_fps"])]
    if afilter:
        cmd += ["-af", afilter]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(out)]
    return cmd


def render_clip(cfg, video, start, dur, filter_complex, has_subs, workdir: Path, out: Path, final: bool, mode: str,
                loud: dict | None) -> dict:
    """Encode once. Returns {encoder, seconds, loudness_in}. QSV failure falls back to x264 and says so."""
    afilter = loudnorm_filter(cfg, loud)
    want_qsv = (not final) and cfg["render"]["review_encoder"] in ("auto", "h264_qsv") and qsv_works(cfg)
    tmp = out.with_suffix(".tmp.mp4")
    for qsv in ([True, False] if want_qsv else [False]):
        t0 = time.perf_counter()
        cmd = build_cmd(cfg, video, start, dur, filter_complex, has_subs, tmp, final, mode, afilter, qsv)
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=workdir,
                           timeout=max(180, dur * 20))
        if p.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 1000:
            tmp.replace(out)
            return {"encoder": "h264_qsv" if qsv else "libx264", "seconds": round(time.perf_counter() - t0, 2),
                    "loudness_in": loud["input_i"] if loud else None}
        tail = "\n".join(p.stderr.strip().splitlines()[-5:])
        if qsv:
            log.warning("h264_qsv encode failed, falling back to libx264: %s", tail)
            continue
        raise StageError("render", f"ffmpeg failed (exit {p.returncode}): {tail}")
    raise StageError("render", "Render failed.")
