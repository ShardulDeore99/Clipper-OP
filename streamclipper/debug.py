"""--debug output: per clip a contact sheet (~8 frames), the chosen start/end with reasons, layout, .ass/.srt."""
import json
import shutil
from pathlib import Path

from .media import run


def save_debug(cfg, key: str, rec: dict, clip_dir: Path, rejected: list[dict] | None = None) -> Path:
    root = cfg.data_dir / "debug" / key
    d = root / rec["cid"]
    d.mkdir(parents=True, exist_ok=True)
    plan = {k: rec.get(k) for k in ("cid", "start", "end", "duration", "score", "key_moment", "payoff", "start_reason",
                                    "end_reason", "start_kind", "end_kind", "notes", "hook_text", "why", "signals",
                                    "layout_used", "flags", "meta")}
    (d / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in ("subs.ass", "captions.srt", "framing.json"):
        if (clip_dir / name).is_file():
            shutil.copy2(clip_dir / name, d / name)
    review = clip_dir / "review.mp4"
    if review.is_file():
        dur = max(rec["end"] - rec["start"], 1.0)
        run([cfg["tools"]["ffmpeg"], "-y", "-hide_banner", "-loglevel", "error", "-i", str(review), "-vf",
             f"fps={8 / dur:.5f},scale=240:-2,tile=4x2", "-frames:v", "1", str(d / "contact.jpg")], "save")
    if rejected is not None:
        (root / "rejected.json").write_text(json.dumps(rejected, indent=1), encoding="utf-8")
    return d
