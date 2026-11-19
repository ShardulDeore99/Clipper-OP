"""Turn candidate moments into clips: boundaries, duration fit, hook/ending scoring, merge, dedupe."""
import hashlib
import math

from .boundaries import (cut_end, cut_start, sentence_at_time, sentence_index, snap_end, snap_start)
from .signals import LAUGH_RE, norm, why_text

FILLERS = {"um", "uh", "so", "and", "aur", "toh", "like", "basically", "actually"}


def _clip01(x):
    return max(0.0, min(1.0, x))


def hook_score(text: str, hook_words: list[str]) -> float:
    t = " " + text.lower() + " "
    toks = text.split()
    f = 0.35 * ("?" in text) + 0.25 * ("!" in text) + 0.15 * (0 < len(toks) <= 10)
    f += 0.35 * any(f" {h} " in t or f" {h}," in t or f" {h}!" in t for h in hook_words)
    if toks and norm(toks[0]) in FILLERS:
        f -= 0.25
    return round(_clip01(f), 3)


def ending_score(last_text: str, payoff_t: float, end_t: float, burst_end: float, key_t: float) -> float:
    f = 0.4 * ((end_t - payoff_t) <= 2.0)
    f += 0.3 * any(c in last_text for c in "!?")
    f += 0.3 * (any(LAUGH_RE.match(norm(w)) for w in last_text.split()) or burst_end > key_t + 0.5)
    return round(_clip01(f), 3)


def fit_score(dur, p) -> float:
    lo, hi, mn, mx = p["target_min"], p["target_max"], p["min_s"], p["max_s"]
    if lo <= dur <= hi:
        return 1.0
    if dur < lo:
        return round(_clip01(0.5 + 0.5 * (dur - mn) / max(lo - mn, 1e-6)), 3)
    return round(_clip01(1 - 0.5 * (dur - hi) / max(mx - hi, 1e-6)), 3)


def _nearest_prev_start(sents, i0):
    k = sentence_index(sents, i0)
    if i0 != sents[k]["w0"]:
        return sents[k]["w0"]
    return sents[k - 1]["w0"] if k > 0 else None


def _next_end(sents, i1):
    k = sentence_index(sents, i1)
    if i1 != sents[k]["w1"]:
        return sents[k]["w1"]
    return sents[k + 1]["w1"] if k + 1 < len(sents) else None


def fit_duration(i0, i1, key_t, words, sents, p, duration):
    """Extend to the minimum length (setup first, else forward); trim the front if over the maximum."""
    notes = []

    def times(a, b):
        return cut_start(words, a), cut_end(words, b, p["tail"], duration)

    s, e = times(i0, i1)
    for _ in range(25):
        if e - s >= p["min_s"]:
            break
        back, fwd = _nearest_prev_start(sents, i0), _next_end(sents, i1)
        opts = []
        # Do not pad a short moment with dialogue across a long VAD gap. Such
        # gaps often mean Whisper missed speech, and spanning them creates clips
        # with tens of seconds of dead air.
        back_gap = words[i0]["start"] - words[i0 - 1]["end"] if back is not None and i0 > 0 else 0
        fwd_start = sents[sentence_index(sents, fwd)]["w0"] if fwd is not None else None
        fwd_gap = words[fwd_start]["start"] - words[i1]["end"] if fwd_start is not None else 0
        if back is not None and key_t - words[back]["start"] <= 10 and back_gap <= 2.5:
            opts.append(("start", back, i1))
        if fwd is not None and fwd_gap <= 2.5:
            opts.append(("end", i0, fwd))
        if back is not None and not opts and back_gap <= 2.5:
            opts.append(("start", back, i1))
        opts = [o for o in opts if times(o[1], o[2])[1] - times(o[1], o[2])[0] <= p["max_s"]]
        if not opts:
            break
        side, i0, i1 = opts[0]
        notes.append(f"extended the {side} by one sentence to reach the {p['min_s']:g} s minimum")
        s, e = times(i0, i1)
    for _ in range(25):
        if e - s <= p["max_s"]:
            break
        k = sentence_index(sents, i0)
        nxt = sents[k + 1]["w0"] if k + 1 < len(sents) else None
        if nxt is None or key_t - words[nxt]["start"] < 2.0 or nxt >= i1:
            break
        i0 = nxt
        notes.append("trimmed the start to stay under the maximum length")
        s, e = times(i0, i1)
    return i0, i1, s, e, notes


def make_clip(i0, i1, key_t, payoff_t, strength, signals, delta_db, burst_end, start_reason, end_reason,
              notes, words, sents, starts, ends, p, duration):
    s, e = cut_start(words, i0), cut_end(words, i1, p["tail"], duration)
    dur = e - s
    ks, ke = sentence_index(sents, i0), sentence_index(sents, i1)
    first = " ".join(w["w"] for w in words[i0:sents[ks]["w1"] + 1]).strip()
    last = sents[ke]["text"]
    hook = hook_score(first, p["hook_words"])
    best_hook, best_hs = first, hook
    for k in range(ks + 1, ke):  # a stronger line inside the clip can be offered as a cold-open text
        hs = hook_score(sents[k]["text"], p["hook_words"])
        if hs > best_hs + 0.2:
            best_hook, best_hs = sents[k]["text"], hs
    trimmed = " ".join(best_hook.split()[:8]) + ("..." if len(best_hook.split()) > 8 else "")
    end_s = ending_score(last, payoff_t, e, burst_end, key_t)
    gaps = [words[j + 1]["start"] - words[j]["end"] for j in range(i0, i1)]
    dead = sum(max(g - 0.6, 0) for g in gaps if g >= 1.0)
    density = _clip01(1 - dead / max(dur, 1e-6))
    q = {"sentence": 1.0, "pause": 0.6}
    s_kind = "sentence" if "sentence start" in start_reason else "pause" if "pause start" in start_reason else "fallback"
    e_kind = "sentence" if "sentence end" in end_reason else "pause" if "pause end" in end_reason else "fallback"
    quality = (q.get(s_kind, 0.2) + q.get(e_kind, 0.2)) / 2
    fit = fit_score(dur, p)
    score = 0.45 * strength + 0.15 * hook + 0.15 * end_s + 0.10 * density + 0.10 * fit + 0.05 * quality
    flags = []
    if gaps and max(gaps) > 3.0:
        flags.append("long silence inside the clip")
    text = " ".join(w["w"] for w in words[i0:i1 + 1]).strip()
    uid = hashlib.sha1(f"{key_t:.1f}".encode()).hexdigest()[:8]
    return {"cid": f"c{uid}", "start": round(s, 3), "end": round(e, 3), "duration": round(dur, 2),
            "score": round(score, 3), "strength": strength, "key_moment": round(key_t, 2),
            "payoff": round(payoff_t, 2), "i0": i0, "i1": i1, "start_kind": s_kind, "end_kind": e_kind,
            "start_reason": start_reason, "end_reason": end_reason, "notes": notes,
            "hook_text": trimmed, "hook_score": hook, "hook_is_opening": best_hook is first,
            "ending_score": end_s, "density": round(density, 3), "fit": fit,
            "why": why_text(signals, delta_db), "signals": signals, "burst_end": burst_end,
            "flags_plan": flags, "text": text}


def build_clip(c, words, sents, starts, ends, p, duration):
    key_t = c["t"]
    ks = sentence_at_time(sents, key_t)
    payoff_t = max(ks["end"], c.get("burst_end", key_t))
    sp, sreason = snap_start(starts, key_t, p["lead"])
    i0, sreason = (sp["i"], sreason) if sp else (ks["w0"], "fallback: start of the key sentence")
    ep, ereason = snap_end(ends, payoff_t)
    i1, ereason = (ep["i"], ereason) if ep else (ks["w1"], "fallback: end of the key sentence")
    if words[i0]["start"] > key_t:
        i0, sreason = ks["w0"], "fallback: start of the key sentence"
    if i1 < i0:
        return "boundaries crossed"
    i0, i1, s, e, notes = fit_duration(i0, i1, key_t, words, sents, p, duration)
    if e - s < p["min_s"]:
        return f"only {e - s:.1f} s available, under the {p['min_s']:g} s minimum"
    if e - s > p["max_s"]:
        return f"{e - s:.0f} s does not fit under {p['max_s']:g} s without cutting the payoff"
    return make_clip(i0, i1, key_t, payoff_t, c["strength"], c["signals"], c.get("delta_db", 0), c.get("burst_end", key_t),
                     sreason, ereason, notes, words, sents, starts, ends, p, duration)


def _merge(a, b, words, sents, starts, ends, p, duration):
    best = a if a["strength"] >= b["strength"] else b
    m = make_clip(min(a["i0"], b["i0"]), max(a["i1"], b["i1"]), best["key_moment"], best["payoff"], best["strength"],
                  best["signals"], 0, best["burst_end"], (a if a["i0"] <= b["i0"] else b)["start_reason"],
                  (a if a["i1"] >= b["i1"] else b)["end_reason"], a["notes"] + b["notes"] +
                  [f"merged with the moment at {b['key_moment']:.0f} s"], words, sents, starts, ends, p, duration)
    m["why"] = best["why"]
    m["score"] = round(min(1.0, max(a["score"], b["score"]) + 0.03), 3)
    return m


def merge_overlaps(clips, words, sents, starts, ends, p, duration, rejected):
    out = []
    for c in sorted(clips, key=lambda c: c["start"]):
        if out and c["start"] < out[-1]["end"] + 0.5:
            prev = out[-1]
            if max(prev["end"], c["end"]) - prev["start"] <= p["max_s"]:
                out[-1] = _merge(prev, c, words, sents, starts, ends, p, duration)
            else:
                lo, hi = sorted([prev, c], key=lambda x: x["score"])
                rejected.append({"t": lo["key_moment"], "reason": "overlaps a better clip and too long to merge"})
                out[-1] = hi
        else:
            out.append(c)
    return out


def drop_duplicates(clips, rejected):
    kept = []
    for c in sorted(clips, key=lambda c: -c["score"]):
        ws = {norm(w) for w in c["text"].split()}
        if len(ws) >= 6 and any(len(ws & {norm(w) for w in k["text"].split()}) / len(ws | {norm(w) for w in k["text"].split()}) >= 0.7
                                for k in kept if len(k["text"].split()) >= 6):
            rejected.append({"t": c["key_moment"], "reason": "near-duplicate of a higher-scoring clip"})
            continue
        kept.append(c)
    return kept


def plan_clips(cands, words, sents, starts, ends, p, duration):
    clips, rejected = [], []
    if not words or not sents:
        return [], [{"t": c["t"], "reason": "nothing to snap boundaries to"} for c in cands]
    for c in cands:
        r = build_clip(c, words, sents, starts, ends, p, duration)
        if isinstance(r, str):
            rejected.append({"t": c["t"], "reason": r})
        else:
            clips.append(r)
    clips = merge_overlaps(clips, words, sents, starts, ends, p, duration, rejected)
    clips = drop_duplicates(clips, rejected)
    kept = []
    for c in clips:
        if c["score"] < p["min_score"]:
            rejected.append({"t": c["key_moment"], "reason": f"score {c['score']:.2f} is under min_score {p['min_score']:g}"})
        else:
            kept.append(c)
    if p["max_per_source"]:
        kept.sort(key=lambda c: -c["score"])
        for c in kept[p["max_per_source"]:]:
            rejected.append({"t": c["key_moment"], "reason": "over max_per_source"})
        kept = kept[:p["max_per_source"]]
    kept.sort(key=lambda c: c["start"])
    for n, c in enumerate(kept, 1):
        c["n"] = n
    return kept, rejected


def params_from_config(cfg) -> dict:
    c, m = cfg["clips"], cfg["moments"]
    return {"min_s": c["min_seconds"], "max_s": c["max_seconds"], "target_min": c["target_min_seconds"],
            "target_max": c["target_max_seconds"], "lead": tuple(c["lead_in_seconds"]),
            "tail": tuple(c["tail_out_seconds"]), "min_score": m["min_score"],
            "max_per_source": c["max_per_source"], "hook_words": [h.lower() for h in m["hook_words"]]}
