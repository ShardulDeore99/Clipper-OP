"""Stage 3: transcribe ONCE per source with faster-whisper (CPU, int8).

Languages: "en" (English stays English), "hinglish" (Hindi / Hindi+English written in casual Roman
letters), or "auto": the language is detected for every ~30 s window, so English stretches stay
English and Hinglish stretches come out Roman. If the installed faster-whisper cannot do that,
auto falls back to one language per video (and the transcript says so).
Whisper models are never downloaded here: no hidden network calls. Use `fetch-models` once.
"""
import inspect
import logging
import time
from pathlib import Path

from .. import cache
from ..errors import StageError
from ..romanize import casual_variants, romanize
from ..transcript import make_segment, write_transcript

log = logging.getLogger("streamclipper")
NAME = "transcribe"
FILE_KEYS = ("transcript_json", "srt")
ALLOWED_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en"}  # never medium/large
LANGUAGES = ("auto", "en", "hinglish")
_MODELS: dict = {}


def normalize_language(value: str) -> str:
    value = "hinglish" if value == "hi" else value
    if value not in LANGUAGES:
        raise StageError(NAME, f"Unknown language '{value}'.", f"Use one of {', '.join(LANGUAGES)}.")
    return value


def settings(ctx) -> tuple[str, str]:
    t = ctx.cfg["transcribe"]
    model = t["model_fast"] if ctx.job["mode"] == "fast" else t["model_quality"]
    if model not in ALLOWED_MODELS:
        raise StageError(NAME, f"Model '{model}' is not allowed.",
                         f"Use one of {sorted(ALLOWED_MODELS)}. medium/large are too heavy for this laptop.")
    language = normalize_language(ctx.job.get("language") or t["language"])
    if model.endswith(".en") and language != "en":
        raise StageError(NAME, f"'{model}' is English-only.", "Use a multilingual model for Hinglish/auto.")
    return model, language


def cache_inputs(ctx) -> dict:
    model, language = settings(ctx)
    t, wav = ctx.cfg["transcribe"], Path(ctx.out["wav"])
    return {"wav": str(wav), "size": wav.stat().st_size, "model": model, "language": language,
            "compute_type": t["compute_type"], "beam_size": t["beam_size"], "vad": t["vad_filter"],
            "prompt": t["hinglish_prompt"], "romanize": 2}


def models_dir(cfg) -> Path:
    return cfg.data_dir / "models"


def load_model(cfg, name: str, allow_download: bool = False):
    t = cfg["transcribe"]
    key = (name, t["compute_type"], t["cpu_threads"])
    if key in _MODELS:
        return _MODELS[key]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise StageError(NAME, "faster-whisper is not installed.", "Run setup.ps1 again.")
    try:
        model = WhisperModel(name, device="cpu", compute_type=t["compute_type"],
                             cpu_threads=t["cpu_threads"], download_root=str(models_dir(cfg)),
                             local_files_only=not allow_download)
    except Exception as e:
        if allow_download:
            raise StageError(NAME, f"Could not download model '{name}': {e}")
        raise StageError(NAME, f"Whisper model '{name}' could not be loaded ({type(e).__name__}: {e}).",
                         r"If it is not downloaded yet, run: .\.venv\Scripts\python -m streamclipper fetch-models")
    _MODELS[key] = model
    return model


def fetch_models(cfg) -> list[str]:
    """The one explicit, user-triggered network step for models."""
    t = cfg["transcribe"]
    names = sorted({t["model_fast"], t["model_quality"]})
    for name in names:
        if name not in ALLOWED_MODELS:
            raise StageError(NAME, f"Model '{name}' is not allowed.", f"Use one of {sorted(ALLOWED_MODELS)}.")
        load_model(cfg, name, allow_download=True)
    return names


def _start(model, wav: str, t: dict, language: str):
    """Return (segments, info, language_used, how). Detection never causes a second transcription."""
    common = dict(beam_size=t["beam_size"], vad_filter=t["vad_filter"],
                  vad_parameters={"min_silence_duration_ms": 500}, word_timestamps=True)

    def hinglish():
        # Hindi audio + a Roman-letter prompt steers Whisper to Roman output; the prompt only
        # persists across windows when conditioning on previous text is on.
        segs, info = model.transcribe(wav, language="hi", initial_prompt=t["hinglish_prompt"],
                                      condition_on_previous_text=True, **common)
        return segs, info, "hinglish"

    if language == "hinglish":
        return (*hinglish(), "fixed")
    if language == "en":
        segs, info = model.transcribe(wav, language="en", condition_on_previous_text=False, **common)
        return segs, info, "en", "fixed"
    if _supports_multilingual(model):
        segs, info = model.transcribe(wav, language=None, multilingual=True,
                                      initial_prompt=t["hinglish_prompt"],
                                      condition_on_previous_text=True, **common)
        _require_supported(segs, info)
        return segs, info, "hinglish" if info.language in ("hi", "ur") else "en", "per-window"
    log.warning("faster-whisper has no multilingual mode; auto uses one language per video")
    segs, info = model.transcribe(wav, language=None, condition_on_previous_text=False, **common)
    if info.language in ("hi", "ur"):  # info is ready before any segment is decoded
        if hasattr(segs, "close"):
            segs.close()
        return (*hinglish(), "per-video")
    _require_supported(segs, info)
    return segs, info, "en", "per-video"


def _supports_multilingual(model) -> bool:
    try:
        return "multilingual" in inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):
        return False


def _require_supported(segs, info) -> None:
    if info.language not in ("en", "hi", "ur"):
        if hasattr(segs, "close"):
            segs.close()
        raise StageError(NAME, f"Detected language '{info.language}'. StreamClipper supports English and Hinglish.",
                         "If detection is wrong, pick English or Hinglish when adding the source.")


def _clean_words(seg, casual: bool) -> list[dict]:
    out = []
    for w in seg.words or []:
        text = romanize(w.word.strip())
        if casual:
            text = casual_variants(text)
        if text:
            out.append({"w": text, "start": round(w.start, 3),
                        "end": round(max(w.end, w.start), 3), "p": round(w.probability, 3)})
    return out


def run(ctx, progress) -> dict:
    model_name, language = settings(ctx)
    t, duration = ctx.cfg["transcribe"], float(ctx.out["duration"])
    model = load_model(ctx.cfg, model_name)
    t0 = time.perf_counter()
    segs, info, used, how = _start(model, ctx.out["wav"], t, language)

    segments, dropped = [], 0
    for seg in segs:
        words = _clean_words(seg, casual=(used != "en" or how == "per-window"))
        if not words:
            dropped += 1
            continue
        segments.append(make_segment(len(segments), words[0]["start"], words[-1]["end"], words))
        progress(min(seg.end / duration, 0.99) if duration else 0.5)
    secs = time.perf_counter() - t0
    n_words = sum(len(s["words"]) for s in segments)
    if n_words == 0:
        log.warning("No speech detected in %s", ctx.out["wav"])
    if dropped:
        log.warning("Dropped %d segments without word timestamps", dropped)

    data = {"language": used, "language_mode": how, "detected": getattr(info, "language", used), "model": model_name,
            "duration": duration, "speech": n_words > 0,
            "stats": {"words": n_words, "segments": len(segments), "seconds": round(secs, 2),
                      "x_realtime": round(duration / secs, 2) if secs else None},
            "segments": segments}
    js, srt = write_transcript(cache.stage_dir(ctx.cfg, ctx.key, NAME), data)
    return {"transcript_json": str(js), "srt": str(srt), "language": used,
            "words": n_words, "speech": n_words > 0, "x_realtime": data["stats"]["x_realtime"]}
