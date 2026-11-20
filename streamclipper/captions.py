"""M5: caption groups -> .ass (bold uppercase, outline, active-word highlight) + a render self-test.

The self-test draws the finished .ass over black with the same ffmpeg, font folder and relative paths
the real render uses. If no caption pixels appear it raises: captions can never silently go missing.
"""
import re
import subprocess
from pathlib import Path

import numpy as np

from .boundaries import END_RE
from .errors import StageError
from .romanize import casual_variants, has_devanagari, romanize
from .signals import norm

COLORS = {
    "lime": "&H00FF00&", "yellow": "&H00FFFF&", "white": "&HFFFFFF&",
    # ASS colors are written as &HAABBGGRR&.
    "funny": "&H0000E6FF&",    # #FFE600
    "hype": "&H0066FF00&",     # #00FF66
    "rage": "&H002D2DFF&",     # #FF2D2D
    "fail": "&H00008A&",        # #FF8A00
    "scary": "&H00FFE500&",     # #00E5FF
    "wholesome": "&H00C87AFF&", # #FF7AC8
    "neutral": "&H0000E6FF&",   # #FFE600
}
THEME_ACCENTS = {
    "funny": "&H002D2DFF&",     # #FF2D2D red
    "hype": "&H0000E6FF&",      # #FFE600 yellow
    "rage": "&H0000E6FF&",      # #FFE600 yellow
    "fail": "&H00FFE500&",      # #00E5FF cyan
    "scary": "&H002D2DFF&",     # #FF2D2D red
    "wholesome": "&H00FFFFFF&", # #FFFFFF white
    "neutral": "&H00FFFFFF&",   # #FFFFFF white
}
BAND = (1000, 1500)  # captions live here: lower-middle, above the bottom 20% (y > 1536)


def load_glossary(path) -> list[tuple[list[str], list[str]]]:
    rules = []
    p = Path(path)
    if not p.is_file():
        return rules
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=>" in line:
            heard, written = (x.strip() for x in line.split("=>", 1))
            if heard and written:
                rules.append(([norm(t) for t in heard.split()], written.split()))
    return sorted(rules, key=lambda r: -len(r[0]))


def apply_glossary(words: list[dict], rules) -> list[dict]:
    out, i = [], 0
    while i < len(words):
        for heard, written in rules:
            n = len(heard)
            if i + n <= len(words) and [norm(w["w"]) for w in words[i:i + n]] == heard:
                span = words[i:i + n]
                t0, t1 = span[0]["start"], span[-1]["end"]
                step = (t1 - t0) / len(written)
                for k, tok in enumerate(written):
                    out.append({"w": tok, "start": round(t0 + k * step, 3), "end": round(t0 + (k + 1) * step, 3)})
                i += n
                break
        else:
            out.append(dict(words[i]))
            i += 1
    return out


def apply_caption_edit(words: list[dict], text: str) -> list[dict]:
    """The user retyped the captions: keep the timing. Same word count = 1:1, else spread over the speech span."""
    toks = text.split()
    if not toks or not words:
        return []
    if len(toks) == len(words):
        return [{"w": t, "start": w["start"], "end": w["end"]} for t, w in zip(toks, words)]
    t0, t1 = words[0]["start"], words[-1]["end"]
    step = (t1 - t0) / len(toks)
    return [{"w": t, "start": round(t0 + k * step, 3), "end": round(t0 + (k + 1) * step, 3)} for k, t in enumerate(toks)]


def clip_words(all_words: list[dict], start: float, end: float, glossary=(), edited: str | None = None) -> list[dict]:
    """Words inside the clip, shifted to clip time (0 = clip start)."""
    inside = [w for w in all_words if w["w"] and w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
    inside = [{"w": w["w"], "start": round(max(w["start"] - start, 0), 3), "end": round(min(w["end"] - start, end - start), 3)}
              for w in inside]
    inside = apply_glossary(inside, glossary)
    return apply_caption_edit(inside, edited) if edited else inside


def clean_token(tok: str) -> tuple[str, bool]:
    """Roman letters only, uppercase, ASS-safe. Returns (text, had_unsupported_glyphs)."""
    t = casual_variants(romanize(tok)) if has_devanagari(tok) else tok
    t = t.replace("{", "(").replace("}", ")").replace("\\", "")
    t = re.sub(r"[.,;:]+$", "", t)
    kept = "".join(ch for ch in t if ord(ch) < 0x250 or ch in "\u2019\u2018\u201c\u201d\u2013\u2014")
    return kept.upper(), kept != t


def build_groups(words: list[dict], per=(2, 5)) -> list[list[dict]]:
    groups, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            full = len(cur) >= per[1]
            enough = END_RE.search(cur[-1]["w"])
            if full or enough or gap >= 0.45:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        if (len(cur) == 1 and groups and len(groups[-1]) < per[1]
                and not END_RE.search(groups[-1][-1]["w"])
                and cur[0]["start"] - groups[-1][-1]["end"] < 0.45):
            groups[-1] += cur
        else:
            groups.append(cur)
    return groups


def _lines(tokens: list[str], max_lines: int) -> str:
    one = " ".join(tokens)
    if len(tokens) < 2 or len(one) <= 14 or max_lines < 2:
        return one
    best = min(range(1, len(tokens)), key=lambda k: abs(len(" ".join(tokens[:k])) - len(" ".join(tokens[k:]))))
    return " ".join(tokens[:best]) + r"\N" + " ".join(tokens[best:])


def _ts(t: float) -> str:
    cs = round(max(t, 0) * 100)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def _overlap_px(y_center, rect, half_h=110, margin=20) -> float:
    """Vertical overlap (px) between a caption block centred at y_center and a face rectangle."""
    _, y0, _, y1 = rect
    return max(0.0, min(y1, y_center + half_h + margin) - max(y0, y_center - half_h - margin))


def choose_y(cfg, t0, t1, faces_fn) -> int:
    """Default y, else the nearest allowed y with no face under the caption, else the least-covered one."""
    base = cfg["captions"]["y_center"]
    if cfg["captions"].get("position_locked"):
        return base
    if faces_fn is None:
        return base
    rects = [r for k in (t0, (t0 + t1) / 2, t1) for r in faces_fn(k)]
    cands = [y for y in (base, 1120, 1400, 1060, 1440, 1500, 1000) if BAND[0] <= y <= BAND[1]]
    cost = {y: sum(_overlap_px(y, r) for r in rects) for y in cands}
    return min(cands, key=lambda y: (cost[y], cands.index(y)))


def make_ass(cfg, font: dict, words: list[dict], faces_fn=None, extra_events: list[str] | None = None) -> tuple[str, dict]:
    c = cfg["captions"]
    color = COLORS.get(c["highlight_color"], COLORS["lime"])
    accent_color = THEME_ACCENTS.get(c["highlight_color"], "&H00FFFFFF&")
    box = c.get("look") == "box"  # dark rounded-looking box behind the words: readable on bright game HUDs
    cap_outline, cap_border, cap_width, cap_shadow = ("&H10101010", 3, 16, 0) if box else ("&H00000000", 1, 9, 3)
    stripped = 0
    toks = []
    for w in words:
        t, lost = clean_token(w["w"])
        stripped += lost
        if t:
            toks.append({**w, "t": t})
    groups = build_groups([{**w, "w": w["t"]} for w in toks], tuple(c["words_per_caption"]))
    events = []
    for gi, g in enumerate(groups):
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        y = choose_y(cfg, g[0]["start"], g[-1]["end"], faces_fn)
        x = c.get("x_center", 540)
        for j, w in enumerate(g):
            t0 = w["start"]
            t1 = g[j + 1]["start"] if j + 1 < len(g) else w["end"] + 0.12
            if nxt is not None:
                t1 = min(t1, nxt - 0.02)
            t1 = max(t1, t0 + 0.08)
            active_color = accent_color if j == len(g) - 1 else color
            shown = [(f"{{\\c{active_color}}}{x['w']}{{\\c&HFFFFFF&}}" if k == j else x["w"]) for k, x in enumerate(g)]
            events.append(f"Dialogue: 0,{_ts(t0)},{_ts(t1)},Cap,,0,0,0,,{{\\an5\\pos({x},{y})}}{_lines(shown, c['max_lines'])}")
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font['family']},{c['font_size']},&H00FFFFFF,&H00FFFFFF,{cap_outline},&H80000000,{font['bold']},0,0,0,100,100,1,0,{cap_border},{cap_width},{cap_shadow},5,50,50,0,1
Style: Sticker,{font['family']},{int(c['font_size'] * 1.2)},&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,{font['bold']},0,0,0,100,100,1,0,1,9,3,5,50,50,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    allev = events + (extra_events or [])
    stats = {"groups": len(groups), "events": len(events), "words": len(toks), "glyphs_removed": stripped}
    return head + "\n".join(allev) + "\n", stats


def probe_time(words: list[dict]) -> float | None:
    """Choose the middle of a long caption event that contains renderable text."""
    tokens = []
    for word in words:
        text, _ = clean_token(word["w"])
        if text:
            tokens.append({**word, "w": text})
    if not tokens:
        return None
    groups = build_groups(tokens)
    events = []
    for gi, group in enumerate(groups):
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        for j, word in enumerate(group):
            end = group[j + 1]["start"] if j + 1 < len(group) else word["end"] + 0.12
            if next_start is not None:
                end = min(end, next_start - 0.02)
            if end > word["start"]:
                events.append((word["start"], end, len(group)))
    start, end, _ = max(events, key=lambda span: (span[2] > 1, span[1] - span[0]))
    return round(start + (end - start) / 2, 2)


def verify_render(ffmpeg: str, workdir: Path, t: float, min_pixels: int = 300) -> int:
    """Draw subs.ass over black at time t using the render's own relative paths. Returns bright-pixel count."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=black:s=540x960:r=30:d={t + 0.7:.2f}",
           "-vf", "subtitles=subs.ass:fontsdir=fonts,format=gray", "-ss", f"{t:.2f}", "-frames:v", "1", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True, cwd=workdir)
    if p.returncode != 0:
        raise StageError("render", "Captions could not be drawn: " + p.stderr.decode("utf-8", "replace")[-300:],
                         "Check that FFmpeg has libass (run doctor) and that the caption font is readable.")
    count = int((np.frombuffer(p.stdout, dtype=np.uint8) > 200).sum())
    if count < min_pixels:
        raise StageError("render", "Captions exist but did not render (the self-test frame is empty).",
                         "The caption font or FFmpeg's libass is not working. Run doctor.")
    return count
