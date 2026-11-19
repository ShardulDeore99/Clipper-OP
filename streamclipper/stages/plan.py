"""Stage 6: boundaries, duration fit, hook/ending, merge/dedupe -> the clip plan (with reasons)."""
import json
from pathlib import Path

from .. import cache
from ..planner import params_from_config, plan_clips
from clip_engine.ai.transcript_matcher import has_target_match, parse_targets

NAME = "plan"
FILE_KEYS = ("clips_json",)


def cache_inputs(ctx) -> dict:
    return {"segment": cache.sig(ctx.out["segment_json"]), "candidates": cache.sig(ctx.out["candidates_json"]),
            "params": json.loads(json.dumps(params_from_config(ctx.cfg))),
            "moment_transcript": ctx.job["meta"].get("moment_transcript", ""), "v": 3}  # tuples -> lists


def run(ctx, progress) -> dict:
    seg = json.loads(Path(ctx.out["segment_json"]).read_text(encoding="utf-8"))
    cands = json.loads(Path(ctx.out["candidates_json"]).read_text(encoding="utf-8"))["candidates"]
    p = params_from_config(ctx.cfg)
    if ctx.job["meta"].get("live_trigger"):
        p["min_score"] = 0.0  # a live trigger already passed the monitor's own threshold
    clips, rejected = plan_clips(cands, seg["words"], seg["sentences"], seg["starts"], seg["ends"], p, seg["duration"])
    transcript = ctx.job["meta"].get("moment_transcript", "").strip()
    if transcript:
        kept = []
        for clip in clips:
            if has_target_match(clip.get("text", ""), transcript):
                clip["provided_transcript_match"] = True
                kept.append(clip)
            else:
                rejected.append({"t": clip["key_moment"], "reason": "planned cut did not contain a supplied transcript line"})
        clips = kept
        if not cands:
            rejected.extend({"reason": "supplied line was not found in the Whisper transcript", "target": line}
                            for line in parse_targets(transcript))
    path = cache.stage_dir(ctx.cfg, ctx.key, NAME) / "clips.json"
    path.write_text(json.dumps({"clips": clips, "rejected": rejected}), encoding="utf-8")
    return {"clips_json": str(path), "n_clips": len(clips)}
