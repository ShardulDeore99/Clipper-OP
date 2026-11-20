"""Optional reaction stickers (text badges). Off by default. Capped per clip length, spaced, never on a face,
and never in the last 1.5 s."""
import random

from .signals import LAUGH_RE, norm

STICKERS = {"omg": "OMG", "wtf": "WTF", "wow": "WOW", "insane": "INSANE", "crazy": "CRAZY", "clutch": "CLUTCH",
            "pagal": "PAGAL", "gazab": "GAZAB", "kamaal": "KAMAAL", "gg": "GG"}
SPOTS = [(300, 420), (780, 420), (300, 640), (780, 640), (540, 300)]  # top half, away from the caption band


def max_stickers(duration: float) -> int:
    return 2 if duration <= 15 else 4 if duration <= 30 else 6


def plan_stickers(words: list[dict], duration: float, faces_fn=None, min_gap=2.0, seed=7) -> list[dict]:
    rng, out, last = random.Random(seed), [], -99.0
    cap = max_stickers(duration)
    for w in words:
        tok = norm(w["w"])
        text = "LOL" if LAUGH_RE.match(tok) else STICKERS.get(tok)
        if not text or w["start"] - last < min_gap or w["start"] > duration - 1.5 - 1.0 or len(out) >= cap:
            continue
        t0, t1 = w["start"], min(w["start"] + 1.0, duration - 1.5)
        rects = faces_fn(t0) if faces_fn else []
        spots = SPOTS[:]
        rng.shuffle(spots)
        for x, y in spots:
            box = (x - 160, y - 90, x + 160, y + 90)
            if not any(not (box[2] < r[0] or box[0] > r[2] or box[3] < r[1] or box[1] > r[3]) for r in rects):
                out.append({"t0": t0, "t1": t1, "text": text, "x": x, "y": y})
                last = t0
                break
    return out


def sticker_events(stickers: list[dict]) -> list[str]:
    from .captions import _ts
    return [f"Dialogue: 1,{_ts(s['t0'])},{_ts(s['t1'])},Sticker,,0,0,0,,{{\\an5\\pos({s['x']},{s['y']})\\frz-8}}{s['text']}"
            for s in stickers]
