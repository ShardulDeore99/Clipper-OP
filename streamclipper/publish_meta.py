"""Title, description, hashtags and cover-frame suggestion, written locally from the transcript. Nothing is posted."""
import re
from collections import Counter

from .signals import norm

STOP = set("""the and for that this with have from your what when were they them then than just like will would could about
into over only also some there their here been being very more most much such even said says dont cant wont aur toh hai
hain nahi kya yeh woh ke ki ka ko se me mein par bhi jo kuch mai tum aap tu ja""".split())
GAMING = {"game", "gameplay", "level", "boss", "stream", "fight", "match", "rank", "noob", "gg", "player", "steam",
          "multiplayer", "quest", "loot", "clutch", "lag", "server"}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def _cut(text: str, n: int) -> str:
    text = text.strip(" .,;:-")
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].strip(" .,;:-")


def make_meta(clip: dict, language: str | None, source_title: str, framing: dict | None = None) -> dict:
    text = clip.get("text", "")
    start = int(clip["start"])
    hook = clip.get("hook_text", "").rstrip(".")
    title = _cut(hook, 60) if hook else f"Highlight at {start // 60}:{start % 60:02d}"
    title = title[:1].upper() + title[1:]
    sents = _sentences(text)
    desc = " ".join(sents[:2])[:220].strip() if sents else f"Highlight from {source_title}."
    toks = [norm(w) for w in text.split()]
    freq = Counter(t for t in toks if len(t) >= 4 and t.isalpha() and t not in STOP)
    tags = [f"#{w}" for w, c in freq.most_common(3) if c >= 2]
    if sum(1 for t in toks if t in GAMING) >= 2:
        tags += ["#gaming", "#gamingclips"]
    if language == "hinglish":
        tags.append("#hinglish")
    tags += ["#shorts", "#reels"]
    tags = list(dict.fromkeys(tags))[:8]
    dur = clip["end"] - clip["start"]
    cover = min(max(clip["key_moment"] - clip["start"], 0.5), max(dur - 0.5, 0.5))
    if framing and framing.get("samples"):  # prefer the sample near the key moment with the biggest face
        near = [s for s in framing["samples"] if abs(s["t"] - cover) <= 1.5 and s["boxes"]]
        if near:
            best = max(near, key=lambda s: max((b[2] - b[0]) * (b[3] - b[1]) for b in s["boxes"]))
            cover = best["t"]
    return {"title": title, "description": desc, "hashtags": tags, "cover_time": round(cover, 2)}
