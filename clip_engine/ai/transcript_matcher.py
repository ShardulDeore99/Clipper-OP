"""Match user-supplied exact speech lines against Whisper word timestamps."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_RANGE_LIKE = re.compile(r"\d+:\d+\s*(?:-|–|—|to)\s*\d+:\d+", re.I)
_RANGE = re.compile(r"^\s*[-*•]?\s*(\d+(?::\d+){1,2}(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?::\d+){1,2}(?:\.\d+)?)\s*$", re.I)


def _time_seconds(value: str) -> float | None:
    parts = value.split(":")
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        if seconds >= 60:
            return None
        return minutes * 60 + seconds
    if len(nums) == 3:
        hours, minutes, seconds = nums
        if minutes >= 60 or seconds >= 60:
            return None
        return hours * 3600 + minutes * 60 + seconds
    return None


def parse_time_ranges(text: str) -> list[tuple[float, float]] | None:
    """Parse one explicit clip range per line; return None when the input is speech text."""
    lines = [line.strip() for line in re.split(r"[\r\n]+", text or "") if line.strip()]
    if not lines or not any(_RANGE_LIKE.search(line) for line in lines):
        return None
    ranges = []
    for line in lines:
        match = _RANGE.fullmatch(line)
        if not match:
            return None
        start, end = (_time_seconds(value) for value in match.groups())
        if start is None or end is None or end <= start:
            return None
        ranges.append((start, end))
    return ranges


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def parse_targets(text: str) -> list[str]:
    """Accept one target line per line; also split long lines at sentence punctuation."""
    out = []
    for line in re.split(r"[\r\n]+", text or ""):
        if _RANGE_LIKE.search(line):
            continue
        for part in re.split(r"(?<=[.!?])\s+", line.strip()):
            if len(_tokens(part)) >= 4:
                out.append(part.strip())
    return out


def find_matches(words: list[dict], transcript: str, threshold: float = 0.72) -> list[dict]:
    """Return the best non-overlapping word spans for each supplied spoken line."""
    targets = parse_targets(transcript)
    source = [_tokens(str(w.get("w", "")))[0] if _tokens(str(w.get("w", ""))) else "" for w in words]
    matches = []
    occupied: list[tuple[int, int]] = []
    for target in targets:
        wanted = _tokens(target)
        wiggle = max(1, round(len(wanted) * 0.2))
        sizes = range(max(3, len(wanted) - wiggle), len(wanted) + wiggle + 1)
        found = []
        for size in sizes:
            if size > len(source):
                continue
            for start in range(len(source) - size + 1):
                end = start + size
                if any(start < b and end > a for a, b in occupied):
                    continue
                ratio = SequenceMatcher(None, wanted, source[start:end], autojunk=False).ratio()
                if ratio >= threshold:
                    found.append((ratio, start, end))
        if not found:
            continue
        ratio, start, end = max(found, key=lambda x: (x[0], -x[1]))
        occupied.append((start, end))
        first, last = words[start], words[end - 1]
        matches.append({"t": round((float(first["start"]) + float(last["end"])) / 2, 2),
                        "start": round(float(first["start"]), 2), "end": round(float(last["end"]), 2),
                        "target": target, "match_score": round(ratio, 3), "word_start": start, "word_end": end - 1})
    return matches


def has_target_match(clip_text: str, transcript: str, threshold: float = 0.72) -> bool:
    """Guard the final planned clip so transcript-only mode cannot leak unrelated clips."""
    clip = _tokens(clip_text)
    if not clip:
        return False
    for target in parse_targets(transcript):
        wanted = _tokens(target)
        wiggle = max(1, round(len(wanted) * 0.2))
        for size in range(max(3, len(wanted) - wiggle), min(len(clip), len(wanted) + wiggle) + 1):
            for start in range(len(clip) - size + 1):
                if SequenceMatcher(None, wanted, clip[start:start + size], autojunk=False).ratio() >= threshold:
                    return True
    return False
