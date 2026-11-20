"""Stage 5: cheap signals first -> a short list of candidate moments (only these get planned)."""
import json
from pathlib import Path

import numpy as np

from .. import cache
from ..signals import find_candidates, moment_curve
from clip_engine.ai.transcript_analyzer import analyze_context, transcript_candidates
from clip_engine.detectors.visual_detector import analyze_candidates as analyze_visual_candidates
from clip_engine.intelligence.highlight_scorer import score_highlight
from clip_engine.ai.transcript_matcher import find_matches, parse_time_ranges

NAME = "score"
FILE_KEYS = ("candidates_json",)


def load_chat(ctx) -> list[float]:
    """Optional chat spikes: a file named <video>.chat.json next to a local video (list of seconds or {t})."""
    if ctx.job.get("kind") != "file":
        return []
    f = Path(ctx.job["source"]).with_suffix(".chat.json")
    if not f.is_file():
        return []
    raw = json.loads(f.read_text(encoding="utf-8"))
    return [float(x["t"] if isinstance(x, dict) else x) for x in raw]


def cache_inputs(ctx) -> dict:
    m = ctx.cfg["moments"]
    chat = Path(ctx.job["source"]).with_suffix(".chat.json") if ctx.job.get("kind") == "file" else None
    return {"segment": cache.sig(ctx.out["segment_json"]), "k": m["candidates_per_minute"],
            "keywords": m["keywords"], "chat": cache.sig(chat) if chat and chat.is_file() else None,
            "intelligence": ctx.cfg["intelligence"],
            "moment_transcript": ctx.job["meta"].get("moment_transcript", ""),
            "live": ctx.job["meta"].get("live_trigger"), "v": 6}


def run(ctx, progress) -> dict:
    seg = json.loads(Path(ctx.out["segment_json"]).read_text(encoding="utf-8"))
    duration, db01 = seg["duration"], np.array(seg["energy"]["db"])
    live = ctx.job["meta"].get("live_trigger")
    parts = None
    chat = []
    provided_transcript = ctx.job["meta"].get("moment_transcript", "").strip()
    if live:  # live monitor: the trigger IS the candidate
        cands = [{"t": live["t"], "strength": 0.8, "signals": {"energy": 0.8}, "delta_db": live.get("delta_db", 10),
                  "burst_end": live.get("burst_end", live["t"])}]
    else:
        chat = load_chat(ctx)
        parts, curve, delta = moment_curve(duration, db01, seg["words"], seg["sentences"],
                                           ctx.cfg["moments"]["keywords"], chat)
        if provided_transcript:
            cands = []
            ranges = parse_time_ranges(provided_transcript)
            if ranges is not None:
                for start, end in ranges:
                    if end > duration:
                        continue  # the planner will report that this requested range is past the source end
                    cands.append({"t": round((start + end) / 2, 2), "strength": 0.9,
                                  "signals": {"speech": 0.9, "humor": 0.9}, "delta_db": 0.0,
                                  "burst_end": end, "source": "provided_time_range",
                                  "provided_range": {"start": start, "end": end},
                                  "reasons": ["user-provided timestamp range"]})
            else:
                for match in find_matches(seg["words"], provided_transcript):
                    cands.append({"t": match["t"], "strength": 0.9, "signals": {"speech": 0.9, "humor": 0.9},
                                  "delta_db": 0.0, "burst_end": match["end"], "source": "provided_transcript",
                                  "provided_match": match, "reasons": ["matched user-provided transcript"]})
        else:
            cands = find_candidates(parts, curve, db01, delta, duration, ctx.cfg["moments"]["candidates_per_minute"])
            for speech_candidate in transcript_candidates(seg["sentences"], seg["words"],
                                                           ctx.cfg["intelligence"]["humor_candidate_threshold"]):
                if all(abs(speech_candidate["t"] - c["t"]) >= 8.0 for c in cands):
                    cands.append(speech_candidate)

    progress(0.65)
    visual = analyze_visual_candidates(ctx.out["video"], cands) if cands else []
    for i, candidate in enumerate(cands):
        context = analyze_context(seg["words"], candidate["t"])
        if live:
            reason = str(live.get("reason", ""))
            audio_score = min(100.0, max(0.0, float(live.get("delta_db", 0)) / 18 * 100)) if "loud" in reason else None
            chat_score = 100.0 if "chat" in reason else None
        else:
            audio_keys = ("energy", "nonspeech", "excite", "pause_beat", "laugh")
            audio_len = min((len(parts[k]) for k in audio_keys), default=0) if parts else 0
            second = min(audio_len - 1, max(0, int(candidate["t"]))) if audio_len else 0
            audio_score = max((float(parts[k][second]) for k in audio_keys), default=0.0) * 100 if audio_len else 0.0
            chat_score = (float(candidate["signals"].get("chat", 0.0)) * 100) if chat else None
        visual_info = visual[i] if i < len(visual) else {"available": False}
        signals = {"audio": audio_score,
                   "chat": chat_score,
                   "speech": context["speech_score"] if context["available"] else None,
                   "visual": visual_info.get("visual_score") if visual_info.get("available") else None}
        cfg_i = ctx.cfg["intelligence"]
        ranking = score_highlight(signals,
            {name: cfg_i[f"{name}_weight"] for name in ("audio", "chat", "speech", "visual")},
            {"high": cfg_i["high_priority_score"], "potential": cfg_i["potential_score"],
             "review": cfg_i["review_score"]})
        if candidate.get("source") == "provided_transcript":
            ranking["label"] = "user-selected transcript match"
        candidate["signals"].update({name: round(value / 100, 3) for name, value in signals.items() if value is not None})
        candidate["strength"] = round(max(ranking["score"] / 100, 0.85)
                                       if candidate.get("source") == "provided_transcript"
                                       else ranking["score"] / 100, 3)
        candidate["speech_analysis"] = context
        candidate["visual_analysis"] = visual_info
        candidate["highlight_analysis"] = ranking
        candidate["reasons"] = sorted(set(candidate.get("reasons", []) + context["labels"] + visual_info.get("evidence", [])))
        progress(0.65 + 0.3 * (i + 1) / max(len(cands), 1))
    path = cache.stage_dir(ctx.cfg, ctx.key, NAME) / "candidates.json"
    path.write_text(json.dumps({"candidates": cands}), encoding="utf-8")
    return {"candidates_json": str(path), "n_candidates": len(cands)}
