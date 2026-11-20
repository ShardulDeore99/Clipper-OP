"""Startup checks. Required failures stop the app; optional ones are reported, never hidden."""
import importlib.util
import re
import shutil
from dataclasses import dataclass
from datetime import date

from .errors import StageError
from .media import run


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _ffmpeg(cfg, *args, timeout=20):
    return run([cfg["tools"]["ffmpeg"], "-hide_banner", *args], "startup", timeout)


def ytdlp_age_days(version: str, today: date | None = None) -> int | None:
    """yt-dlp versions are dates (2026.08.19). Returns how old that is, or None if unparseable."""
    try:
        y, m, d = (int(x) for x in version.split(".")[:3])
        return ((today or date.today()) - date(y, m, d)).days
    except ValueError:
        return None


def _ytdlp_checks() -> list[Check]:
    if importlib.util.find_spec("yt_dlp") is None:
        return [Check("yt-dlp", False, "missing: links will fail, local files still work", required=False)]
    from yt_dlp.version import __version__ as ver
    age = ytdlp_age_days(ver)
    old = age is not None and age > 45
    out = [Check("yt-dlp", not old,
                 f"{ver}" + (f" is {age} days old: YouTube downloads often fail on old versions. "
                             r"Run: .\.venv\Scripts\python -m streamclipper update-ytdlp" if old else ""),
                 required=False)]
    from .stages.download import js_runtime
    rt = js_runtime()
    out.append(Check("JavaScript runtime", rt is not None,
                     rt or "none found: YouTube may fail. Install one: winget install DenoLand.Deno",
                     required=False))
    return out


def _caption_check(cfg) -> list[Check]:
    """Draw real captions with this FFmpeg + font + relative-path setup. Catches 'captions silently missing'."""
    import tempfile
    from pathlib import Path
    from .captions import make_ass, probe_time, verify_render
    from .fonts import find_font, stage_font
    try:
        font = find_font(cfg)
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            stage_font(font, wd)
            words = [{"w": w, "start": 0.2 + i * 0.4, "end": 0.55 + i * 0.4} for i, w in enumerate("mai kyu aau nahi tu ja".split())]
            (wd / "subs.ass").write_text(make_ass(cfg, font, words)[0], encoding="utf-8")
            px = verify_render(cfg["tools"]["ffmpeg"], wd, probe_time(words))
        note = f"captions render with font '{font['family']}' ({px} px)"
        if font["is_fallback"]:
            note += f"; '{cfg['captions']['font']}' not found: put Anton-Regular.ttf in assets/fonts"
        return [Check("caption rendering", True, note)]
    except StageError as e:
        return [Check("caption rendering", False, e.message + " " + e.hint)]


def _framing_checks() -> list[Check]:
    out = []
    for mod, why in (("numpy", "needed for moment detection"), ("cv2", "needed for face tracking (layout no_face works without it)")):
        ok = importlib.util.find_spec(mod) is not None
        out.append(Check(mod, ok, "installed" if ok else f"missing: {why}", required=(mod == "numpy")))
    if importlib.util.find_spec("cv2"):
        try:
            from .faces import HaarDetector
            HaarDetector()
            out.append(Check("face detector", True, "OpenCV face models found", required=False))
        except StageError as e:
            out.append(Check("face detector", False, e.message, required=False))
    return out


def run_checks(cfg) -> list[Check]:
    checks: list[Check] = []
    try:
        ver = _ffmpeg(cfg, "-version").stdout.splitlines()[0]
        checks.append(Check("ffmpeg", True, ver))
    except StageError as e:
        return [Check("ffmpeg", False, e.message + " " + e.hint)]
    try:
        run([cfg["tools"]["ffprobe"], "-version"], "startup", 20)
        checks.append(Check("ffprobe", True, "found"))
    except StageError as e:
        checks.append(Check("ffprobe", False, e.message + " " + e.hint))

    filters = _ffmpeg(cfg, "-filters").stdout
    has_libass = all(re.search(rf"\b{f}\b", filters) for f in ("subtitles", "ass"))
    checks.append(Check("libass (captions)", has_libass,
                        "subtitle burn-in available" if has_libass else
                        "this FFmpeg build cannot burn in captions; install a full/gpl build"))

    encoders = _ffmpeg(cfg, "-encoders").stdout
    checks.append(Check("libx264", "libx264" in encoders,
                        "software H.264 encoder available" if "libx264" in encoders
                        else "needed as the fallback encoder"))

    qsv = False
    if "h264_qsv" in encoders:  # listed is not the same as working: try a real 0.3 s encode
        try:
            _ffmpeg(cfg, "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "0.3",
                    "-c:v", "h264_qsv", "-f", "null", "-", timeout=20)
            qsv = True
        except StageError:
            pass
    checks.append(Check("h264_qsv", qsv,
                        "Intel Quick Sync works" if qsv else
                        "not usable; reviews will use libx264 veryfast (slower)", required=False))

    checks.extend(_ytdlp_checks())

    checks.extend(_caption_check(cfg))
    checks.extend(_framing_checks())
    has_fw = importlib.util.find_spec("faster_whisper") is not None
    checks.append(Check("faster-whisper", has_fw,
                        "installed" if has_fw else "missing: transcription will fail; run setup.ps1",
                        required=False))

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cfg.data_dir).free / 1e9
    checks.append(Check("disk space", free_gb >= 5, f"{free_gb:.1f} GB free in data folder",
                        required=False))
    return checks


def format_checks(checks: list[Check]) -> str:
    out = []
    for c in checks:
        mark = "OK  " if c.ok else ("FAIL" if c.required else "WARN")
        out.append(f"[{mark}] {c.name}: {c.detail}")
    return "\n".join(out)


def require_ok(checks: list[Check]) -> None:
    bad = [c for c in checks if c.required and not c.ok]
    if bad:
        raise StageError("startup", "Startup check failed: " +
                         "; ".join(f"{c.name} ({c.detail})" for c in bad))
