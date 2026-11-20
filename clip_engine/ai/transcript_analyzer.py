"""Context-window speech cues for highlight candidates.

This transparent CPU heuristic does not claim to understand every joke. It
combines transcript context, laughter markers, surprise, incongruity phrases,
and delivery punctuation; an optional learned classifier can replace it later.
"""
from __future__ import annotations

import re

LAUGHTER = re.compile(r"\b(?:ha[\s,.-]*){2,}h?\b|\b(?:he[\s,.-]*){2,}h?\b|\b(?:hi[\s,.-]*){2,}\b|\b(?:lol|lmao|rofl)\b", re.I)
SURPRISE = ("no way", "what just happened", "what the hell", "can't believe", "cannot believe",
            "are you kidding", "oh my god", "what was that", "did that just", "are baap re",
            "oh bhai", "kya hua", "ye kaise", "sach me")
HUMOR_CUES = ("plot twist", "turns out", "what could go wrong", "skill issue", "i thought",
              "of course", "my bad", "i'm so bad", "im so bad", "you serious", "bruh", "ye kya",
              "bhai kya", "kya scene", "kya baat", "mai mar gaya", "main mar gaya", "pagal")
FRUSTRATION = ("what the fuck", "this is bullshit", "i hate", "are you serious", "damn it", "rage",
               "nahi yaar", "bakwas", "kya kar raha")


def _contains(text: str, phrases: tuple[str, ...]) -> bool:
    return any(p in text for p in phrases)


def analyze_context(words: list[dict], t: float, window_seconds: float = 12.0) -> dict:
    """Score a timestamp's surrounding transcript context on 0–100 scales."""
    lo, hi = t - window_seconds / 2, t + window_seconds / 2
    chosen = [w for w in words if w.get("start", -1) <= hi and w.get("end", 0) >= lo]
    text = " ".join(str(w.get("w", "")) for w in chosen).strip()
    low = text.lower()
    tokens = re.findall(r"[\w']+", low)
    laugh = bool(LAUGHTER.search(low))
    surprise = _contains(low, SURPRISE)
    humor_phrase = _contains(low, HUMOR_CUES)
    frustrated = _contains(low, FRUSTRATION)
    punctuation = min(1.0, (text.count("!") + 0.7 * text.count("?") + 0.4 * text.count("…")) / 3)
    repeated = any(tokens[i] == tokens[i - 1] and len(tokens[i]) > 2 for i in range(1, len(tokens)))
    conversational_turn = any(x in low for x in ("but", "then", "actually", "turns out", "i thought"))
    humor = min(100.0, 45 * laugh + 38 * humor_phrase + 12 * conversational_turn +
                10 * repeated + 10 * punctuation + 8 * surprise)
    excitement = min(100.0, 45 * surprise + 35 * punctuation + 20 * laugh)
    frustration = min(100.0, 65 * frustrated + 20 * punctuation + 15 * ("!" in text))
    speech_score = max(humor, excitement, frustration)
    labels = []
    if humor >= 35:
        labels.append("possible humor cues")
    if surprise:
        labels.append("surprise phrase")
    if laugh:
        labels.append("laughter in transcript")
    if frustrated:
        labels.append("frustration phrase")
    if punctuation >= 0.5:
        labels.append("emphatic delivery punctuation")
    return {"available": bool(chosen), "text": text, "start": round(max(0, chosen[0]["start"]), 2) if chosen else None,
            "end": round(chosen[-1]["end"], 2) if chosen else None,
            "humor_score": round(humor, 1), "excitement_score": round(excitement, 1),
            "frustration_score": round(frustration, 1), "speech_score": round(speech_score, 1),
            "labels": labels}


def transcript_candidates(sentences: list[dict], words: list[dict], threshold: float = 32.0) -> list[dict]:
    """Return transcript-led moments with concrete cue explanations."""
    per_minute: dict[int, list[dict]] = {}
    for sentence in sentences:
        t = float(sentence["start"] + sentence["end"]) / 2
        result = analyze_context(words, t)
        if result["humor_score"] >= threshold or result["excitement_score"] >= 65 or result["frustration_score"] >= 65:
            cues = result["labels"] or ["emphatic speech context"]
            candidate = {"t": round(t, 2), "strength": round(result["speech_score"] / 100, 3),
                         "signals": {"speech": result["speech_score"] / 100,
                                     "humor": result["humor_score"] / 100},
                         "speech_analysis": result, "delta_db": 0.0, "burst_end": sentence["end"],
                         "source": "transcript", "reasons": cues}
            bucket = per_minute.setdefault(int(t // 60), [])
            if all(abs(candidate["t"] - old["t"]) >= 8.0 for old in bucket):
                bucket.append(candidate)
    return [c for bucket in per_minute.values() for c in sorted(bucket, key=lambda c: -c["strength"])[:2]]
