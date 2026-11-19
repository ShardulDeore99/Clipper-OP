"""Shared helpers for the per-clip stages (frame, render, check, save)."""
import json
from pathlib import Path

from ..clipwork import ClipWork, load_source
from ..config import job_cfg


def load_plan(ctx) -> list[dict]:
    return json.loads(Path(ctx.out["clips_json"]).read_text(encoding="utf-8"))["clips"]


def edited(ctx) -> set[str]:
    """Clips the user changed by hand: pipeline re-runs must never overwrite them."""
    return {r["cid"] for r in ctx.db.list_clips(ctx.job["id"]) if r["data"].get("edited")}


def works(ctx):
    src = load_source(ctx.cfg, ctx.key, ctx.job["mode"], ctx.job.get("title") or "")
    skip = edited(ctx)
    cfg = job_cfg(ctx.cfg, ctx.job)
    return [ClipWork(cfg, src, c) for c in load_plan(ctx) if c["cid"] not in skip]
