"""Stage 10: title/description/hashtags/cover for each clip and save them to "My clips". Always runs (cheap);
clips the user edited are left alone. Nothing is posted anywhere."""
import json
from pathlib import Path

from ..clipwork import ClipWork, load_source
from ..config import job_cfg
from ..debug import save_debug
from ._clips import edited, load_plan

NAME = "save"
FILE_KEYS = ()
ALWAYS = True


def cache_inputs(ctx) -> dict:
    return {}


def run(ctx, progress) -> dict:
    plan = json.loads(Path(ctx.out["clips_json"]).read_text(encoding="utf-8"))
    renders = json.loads(Path(ctx.out["renders_json"]).read_text(encoding="utf-8"))
    checks = json.loads(Path(ctx.out["checks_json"]).read_text(encoding="utf-8"))
    src = load_source(ctx.cfg, ctx.key, ctx.job["mode"], ctx.job.get("title") or "")
    skip, saved, failed = edited(ctx), 0, 0
    jcfg = job_cfg(ctx.cfg, ctx.job)
    for n, clip in enumerate(plan["clips"]):
        cid = clip["cid"]
        if cid in skip:
            continue
        work = ClipWork(jcfg, src, clip)
        rec = {**clip, "edited": False, "flags": checks.get(cid, []), "language": src.language, "files": {}}
        if cid in renders["failed"]:
            rec["error"], status, path = renders["failed"][cid], "failed", None
            failed += 1
        else:
            r = renders["results"][cid]
            rec["meta"] = work.describe()
            rec["files"] = {"review": r["path"], "cover": rec["meta"]["cover"]}
            rec["layout_used"] = r["layout"]
            rec["render"] = {**{k: r[k] for k in ("encoder", "seconds", "loudness_in")}, "caption": r["caption"]}
            status, path = "ready", r["path"]
            if ctx.cfg["general"]["debug"]:
                save_debug(ctx.cfg, ctx.key, rec, work.dir, plan["rejected"])
        ctx.db.upsert_clip(ctx.job["id"], cid, clip["start"], clip["end"], path, status, rec)
        saved += 1
        progress((n + 1) / max(len(plan["clips"]), 1))
    ctx.db.delete_stale_clips(ctx.job["id"], {c["cid"] for c in plan["clips"]})
    return {"n_clips": len(plan["clips"]), "n_saved": saved, "n_failed": failed}
