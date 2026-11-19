"""M7 editing: what the buttons in "My clips" do. Each runs only the steps its change needs."""
import logging
import shutil

from .clipwork import ClipWork, load_source
from .config import job_cfg
from .framing import LAYOUTS

log = logging.getLogger("streamclipper")
EDITABLE = {"start", "end", "layout", "caption_text"}


def validate(cfg, changes: dict, current: dict, duration: float) -> dict:
    bad = set(changes) - EDITABLE
    if bad:
        raise ValueError(f"Cannot change: {', '.join(sorted(bad))}")
    out = dict(changes)
    start, end = float(out.get("start", current["start"])), float(out.get("end", current["end"]))
    if not 0 <= start < end <= duration + 0.05:
        raise ValueError(f"Trim must be inside the video (0 to {duration:.1f} s) with start before end.")
    if end - start < 3:
        raise ValueError("A clip must be at least 3 seconds long.")
    if end - start > max(cfg["clips"]["max_seconds"], 90):
        raise ValueError("That clip is too long.")
    if "layout" in out and out["layout"] not in LAYOUTS:
        raise ValueError(f"Layout must be one of {', '.join(LAYOUTS)}.")
    out["start"], out["end"] = round(start, 3), round(end, 3)
    return out


def _context(cfg, db, clip_id):
    row = db.get_clip(clip_id)
    job = db.get_job(row["job_id"])
    return row, load_source(cfg, job["source_key"], job["mode"], job.get("title") or ""), job_cfg(cfg, job)


def apply_edit(cfg, db, clip_id: int, changes: dict) -> None:
    """Re-frame / re-caption / re-render one clip after a trim, caption edit or layout change."""
    row, src, cfg = _context(cfg, db, clip_id)
    old = row["data"]
    changes = validate(cfg, changes, old, src.duration)
    clip = {**old, **changes, "duration": round(changes["end"] - changes["start"], 2)}
    if "caption_text" in changes:
        clip["caption_text"] = (changes["caption_text"] or "").strip() or None
    if changes["start"] != old["start"]:
        clip["start_reason"] = f"manual trim (was {old['start']:.2f} s)"
    if changes["end"] != old["end"]:
        clip["end_reason"] = f"manual trim (was {old['end']:.2f} s)"
    clip["text"] = " ".join(w["w"] for w in src.words if w["w"] and clip["start"] - 0.05 <= w["start"] and w["end"] <= clip["end"] + 0.05)
    db.update_clip(clip_id, status="rendering", data={"edited": True, "error": None})
    try:
        work = ClipWork(cfg, src, clip)
        work.frame()
        r = work.render()
        flags = work.check()
        meta = work.describe()
    except Exception as e:
        log.exception("Edit failed for clip %s", clip_id)
        db.update_clip(clip_id, status="failed", data={**clip, "edited": True, "error": str(e)})
        raise
    files = {**old.get("files", {}), "review": r["path"], "cover": meta["cover"]}
    files.pop("final", None)  # the old final export no longer matches
    db.update_clip(clip_id, status="ready", start=clip["start"], end=clip["end"], path=r["path"],
                   data={**clip, "edited": True, "error": None, "flags": flags, "meta": meta, "files": files,
                         "layout_used": r["layout"],
                         "render": {"encoder": r["encoder"], "seconds": r["seconds"], "loudness_in": r["loudness_in"], "caption": r["caption"]}})


def export_final(cfg, db, clip_id: int) -> str:
    """Final export (about crf 18, loudness-normalised, 1080x1920). Made on demand, then cached."""
    row, src, cfg = _context(cfg, db, clip_id)
    db.update_clip(clip_id, data={"exporting": True})
    try:
        out = ClipWork(cfg, src, row["data"]).render(final=True)
    except Exception as e:
        db.update_clip(clip_id, data={"exporting": False, "error": f"Export failed: {e}"})
        raise
    files = {**row["data"].get("files", {}), "final": out["path"]}
    db.update_clip(clip_id, data={"exporting": False, "files": files, "error": None})
    return out["path"]


def delete_clip(cfg, db, clip_id: int) -> None:
    row = db.get_clip(clip_id)
    job = db.get_job(row["job_id"])
    db.delete_clip(clip_id)
    if job.get("source_key"):
        shutil.rmtree(cfg.data_dir / "clips" / job["source_key"] / row["cid"], ignore_errors=True)
