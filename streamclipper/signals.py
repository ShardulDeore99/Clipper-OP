"""Cheap per-second signals -> a 'moment strength' curve -> a short list of candidate moments.

Signals: loudness vs the local baseline, loud non-speech (laughing/shouting/game sound), reaction
keywords, excited wording, a pause right before a burst, laughter words, and chat spikes (if a chat
file exists). Speaker change needs a diarization model, which is too heavy for this laptop: not used.
"""
import math
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

WEIGHTS = {"energy": 0.35, "nonspeech": 0.25, "keyword": 0.20, "excite": 0.15,
           "pause_beat": 0.15, "laugh": 0.25, "chat": 0.30}
LAUGH_RE = re.compile(r"^(?:ha){2,}h?$|^(?:he){2,}h?$|^(?:hi){2,}$|^l+o+l+$|^lmao+$|^rofl$|^lmfao$", re.I)
_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
WHY = {"energy": "loud", "nonspeech": "loud non-speech (laughing/shouting/game sound)",
       "keyword": "reaction word", "excite": "excited wording", "pause_beat": "pause before a burst",
       "laugh": "laughter", "chat": "chat spike", "speech": "speech context", "humor": "possible humor cues",
       "visual": "visual activity", "audio": "audio activity"}


def norm(w: str) -> str:
    return _PUNCT.sub("", w.lower())


def _rolling_median(x: np.ndarray, win: int = 31) -> np.ndarray:
    if len(x) == 0:
        return x
    return np.median(sliding_window_view(np.pad(x, win // 2, mode="edge"), win), axis=1)


def moment_curve(duration, db01, words, sents, keywords, chat_times=None):
    n = max(int(math.ceil(duration)), 1)
    pad = np.full(max(n * 10 - len(db01), 0), -100.0)
    sec_db = np.concatenate([db01, pad])[: n * 10].reshape(n, 10).max(axis=1)
    delta = sec_db - _rolling_median(sec_db)
    energy = np.clip(delta / 12, 0, 1)

    def b(t):
        return min(max(int(t), 0), n - 1)

    wps = np.zeros(n)
    for w in words:
        wps[b(w["start"])] += 1
    near = np.convolve(wps, [1, 1, 1], "same")
    nonspeech = np.where((near == 0) & (delta >= 6), energy * 0.8, 0.0)

    norms = [norm(w["w"]) for w in words]
    kw = {k.lower() for k in keywords}
    keyword, laugh = np.zeros(n), np.zeros(n)
    for i in range(len(words)):
        hits = sum(" ".join(norms[i:i + L]) in kw for L in (1, 2, 3) if i + L <= len(words))
        keyword[b(words[i]["start"])] += hits
        if LAUGH_RE.match(norms[i]):
            laugh[b(words[i]["start"])] = 1.0
    keyword = np.clip(keyword / 2, 0, 1)

    excite = np.zeros(n)
    rates = [(len(s["text"].split()) / max(s["end"] - s["start"], 0.3)) for s in sents if len(s["text"].split()) >= 3]
    med_rate = float(np.median(rates)) if rates else 0.0
    for s in sents:
        toks = s["text"].split()
        rep = sum(1 for a, c in zip(toks, toks[1:]) if norm(a) and norm(a) == norm(c))
        val = 0.4 * s["text"].count("!") + 0.15 * s["text"].count("?") + 0.3 * rep
        if med_rate and len(toks) >= 3 and len(toks) / max(s["end"] - s["start"], 0.3) > 1.4 * med_rate:
            val += 0.3
        excite[b(s["end"])] = max(excite[b(s["end"])], min(val, 1.0))

    pause_beat = np.zeros(n)
    for a, c in zip(words, words[1:]):
        if c["start"] - a["end"] >= 0.8:
            i = b(c["start"])
            if energy[i:i + 2].max() >= 0.4:
                pause_beat[i] = 0.6

    chat = np.zeros(n)
    if chat_times:
        counts = np.bincount([b(t) for t in chat_times], minlength=n).astype(float)
        med = float(np.median(counts))
        mad = float(np.median(np.abs(counts - med)))
        chat = np.clip((counts - med) / (1 + 3 * mad) / 3, 0, 1)

    parts = {"energy": energy, "nonspeech": nonspeech, "keyword": keyword, "excite": excite,
             "pause_beat": pause_beat, "laugh": laugh, "chat": chat}
    curve = np.clip(sum(WEIGHTS[k] * v for k, v in parts.items()), 0, 1)
    return parts, np.convolve(curve, [0.25, 0.5, 0.25], "same"), delta


def find_candidates(parts, curve, db01, delta, duration, per_minute, min_sep=8.0, min_strength=0.2):
    picked = []
    # Enforce the configured quota within each minute. A global top-k lets a
    # few loud moments in one part of a long video crowd out quieter moments
    # elsewhere, which looks like the clipper skipped whole sections.
    for minute in range(math.ceil(duration / 60)):
        lo, hi = minute * 60, min((minute + 1) * 60, len(curve))
        count = 0
        for offset in np.argsort(-curve[lo:hi]):
            i = lo + int(offset)
            if curve[i] < min_strength or count >= per_minute:
                break
            if all(abs(i - p) >= min_sep for p in picked):
                picked.append(i)
                count += 1
    out = []
    for i in sorted(picked):
        seg = db01[i * 10:(i + 1) * 10]
        t = i + (int(np.argmax(seg)) * 0.1 if len(seg) else 0.5)
        lo, hi = max(i - 1, 0), i + 2
        signals = {n: round(float(a[lo:hi].max()), 2) for n, a in parts.items() if a[lo:hi].max() > 0.05}
        burst_end = t
        if parts["nonspeech"][lo:hi].max() > 0.3:
            e = i
            while e + 1 < len(curve) and parts["nonspeech"][e + 1] > 0.3:
                e += 1
            burst_end = e + 1.0
        out.append({"t": round(t, 2), "strength": round(float(curve[i]), 3), "signals": signals,
                    "delta_db": round(float(delta[i]), 1), "burst_end": round(burst_end, 2)})
    return out


def why_text(signals: dict, delta_db: float) -> str:
    top = sorted(signals, key=signals.get, reverse=True)[:3]
    parts = [f"loud (+{delta_db:.0f} dB)" if n == "energy" else WHY[n] for n in top]
    return ", ".join(parts) or "weak signals"
