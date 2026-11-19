"""Sentence and pause boundaries from word timestamps, and the snapping rules (pure functions)."""
import json
import re
from pathlib import Path
from bisect import bisect_left

NONE_GAP = 99.0
END_RE = re.compile(r"[.?!\u2026]+[\"')\]]*$")


def load_words(transcript_path) -> tuple[list[dict], dict]:
    data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    return [w for seg in data["segments"] for w in seg["words"]], data


def build_sentences(words: list[dict], pause: float = 0.6) -> list[dict]:
    sents, start = [], 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i]["start"] - words[i - 1]["end"] >= pause or END_RE.search(words[i - 1]["w"]):
            ws = words[start:i]
            sents.append({"i": len(sents), "w0": start, "w1": i - 1, "start": ws[0]["start"], "end": ws[-1]["end"],
                          "text": " ".join(w["w"] for w in ws).strip()})
            start = i
    return sents


def boundary_points(words, sents, min_gap: float = 0.35) -> tuple[list[dict], list[dict]]:
    """Possible clip starts (sentence starts, pause resumes) and ends (sentence ends, before pauses)."""
    s_starts, s_ends = {s["w0"] for s in sents}, {s["w1"] for s in sents}
    starts, ends = [], []
    for i, w in enumerate(words):
        before = w["start"] - words[i - 1]["end"] if i else NONE_GAP
        after = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else NONE_GAP
        if i in s_starts:
            starts.append({"t": w["start"], "kind": "sentence", "gap": round(before, 2), "i": i})
        elif before >= min_gap:
            starts.append({"t": w["start"], "kind": "pause", "gap": round(before, 2), "i": i})
        if i in s_ends:
            ends.append({"t": w["end"], "kind": "sentence", "gap": round(after, 2), "i": i})
        elif after >= min_gap:
            ends.append({"t": w["end"], "kind": "pause", "gap": round(after, 2), "i": i})
    return starts, ends


def sentence_index(sents, word_i: int) -> int:
    lo, hi = 0, len(sents) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sents[mid]["w0"] <= word_i:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sentence_at_time(sents, t: float) -> dict:
    k = bisect_left([s["end"] for s in sents], t)
    k = min(k, len(sents) - 1)
    # Whisper can leave large timestamp gaps (VAD or missed speech). A moment
    # in such a gap belongs to the nearer sentence, not automatically the next
    # one; otherwise a brief event can snap to dialogue many seconds later.
    if k > 0 and t < sents[k]["start"]:
        prev, nxt = sents[k - 1], sents[k]
        if t - prev["end"] <= nxt["start"] - t:
            return prev
    return sents[k]


def snap_start(points, key_t: float, lead=(2.0, 4.0), widen: float = 8.0):
    """Start 2-4 s before the key moment, snapped to a sentence start (preferred) or a pause resume."""
    ideal = key_t - sum(lead) / 2

    def pick(lo, hi):
        cands = [p for p in points if lo <= p["t"] <= hi]
        return min(cands, key=lambda p: abs(p["t"] - ideal) + (0 if p["kind"] == "sentence" else 1.0), default=None)

    p, how = pick(key_t - lead[1], key_t - lead[0]), ""
    if p is None:
        p, how = pick(key_t - widen, key_t - 0.5), f" (nothing {lead[0]:g}-{lead[1]:g} s before; used nearest)"
    if p is None:
        return None, "no word boundary found before the key moment"
    why = f"{p['kind']} start, {key_t - p['t']:.1f} s before the key moment"
    if p["gap"] < NONE_GAP and p["gap"] >= 0.35:
        why += f", after a {p['gap']:.1f} s pause"
    return p, why + how


def snap_end(points, payoff_t: float, reach: float = 4.0):
    """End on the payoff's sentence end (preferred) or the next pause. Never before the payoff."""
    cands = [p for p in points if payoff_t - 0.05 <= p["t"] <= payoff_t + reach]
    p = min(cands, key=lambda p: (p["t"] - payoff_t) + (0 if p["kind"] == "sentence" else 1.5), default=None)
    if p is None:
        return None, "no word boundary found after the payoff"
    return p, f"{p['kind']} end, {p['t'] - payoff_t:.1f} s after the payoff"


def cut_start(words, i: int, preroll: float = 0.15) -> float:
    t = words[i]["start"]
    room = preroll if i == 0 else min(preroll, max(t - words[i - 1]["end"], 0) * 0.5)
    return max(0.0, t - room)


def cut_end(words, i: int, tail=(0.7, 1.0), limit: float | None = None) -> float:
    """0.7-1.0 s of room after the last word, never running into the next word."""
    t = words[i]["end"]
    gap = words[i + 1]["start"] - t if i + 1 < len(words) else NONE_GAP
    room = min(tail[1], max(tail[0], gap * 0.8))
    room = min(room, max(gap - 0.08, 0.05))
    end = t + room
    return min(end, limit) if limit else end


def boundary_kind_at(points, t: float, tol: float = 0.4) -> str | None:
    """Is time t on (or just before/after) a real boundary? Used by the checks after manual trims."""
    near = [p for p in points if abs(p["t"] - t) <= tol]
    if not near:
        return None
    return "sentence" if any(p["kind"] == "sentence" for p in near) else "pause"
