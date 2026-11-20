"""Stage 4: sentence + pause boundaries from word timestamps, plus the energy curve and silences."""
import json

from .. import audio_features as af, cache
from ..boundaries import boundary_points, build_sentences, load_words
from clip_engine.detectors.audio_detector import analyze_wav

NAME = "segment"
FILE_KEYS = ("segment_json",)


def cache_inputs(ctx) -> dict:
    m = ctx.cfg["moments"]
    return {"wav": cache.sig(ctx.out["wav"], content=False), "transcript": cache.sig(ctx.out["transcript_json"]),
            "pause": m["pause_seconds"], "pause_boundary": m["pause_boundary"], "v": 2}


def run(ctx, progress) -> dict:
    m = ctx.cfg["moments"]
    words, _ = load_words(ctx.out["transcript_json"])
    db = af.energy_db(ctx.out["wav"])
    progress(0.6)
    thr, floor = af.silence_threshold(db)
    pseudo = len(words) < 5  # no talking (e.g. gameplay): use non-silent runs so boundaries still exist
    if pseudo:
        words = af.pseudo_words(db, thr)
    words = [{"w": w["w"], "start": w["start"], "end": w["end"]} for w in words]
    sents = build_sentences(words, m["pause_seconds"]) if words else []
    starts, ends = boundary_points(words, sents, m["pause_boundary"])
    data = {"duration": float(ctx.out["duration"]), "pseudo": pseudo, "words": words, "sentences": sents,
            "starts": starts, "ends": ends, "silences": af.silences(db, thr), "thr_db": round(thr, 1),
            "floor_db": round(floor, 1),
            "energy": {"hop": af.HOP, "db": [round(float(x), 1) for x in af.pool(db)]},
            "audio_analysis": analyze_wav(ctx.out["wav"])}
    path = cache.stage_dir(ctx.cfg, ctx.key, NAME) / "segment.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"segment_json": str(path), "pseudo_words": pseudo}
