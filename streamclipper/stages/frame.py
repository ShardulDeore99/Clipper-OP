"""Stage 7: faces, layout and smoothed crop path for every planned clip (M4)."""
import json

from .. import cache
from ..config import job_cfg
from ._clips import works

NAME = "frame"
FILE_KEYS = ("frames_json",)


def cache_inputs(ctx) -> dict:
    return {"clips": cache.sig(ctx.out["clips_json"]), "framing": job_cfg(ctx.cfg, ctx.job)["framing"], "v": 1}


def run(ctx, progress) -> dict:
    ws, layouts = works(ctx), {}
    for n, w in enumerate(ws):
        layouts[w.clip["cid"]] = w.frame()["layout"]
        progress((n + 1) / max(len(ws), 1))
    path = cache.stage_dir(ctx.cfg, ctx.key, NAME) / "frames.json"
    path.write_text(json.dumps(layouts), encoding="utf-8")
    return {"frames_json": str(path)}
