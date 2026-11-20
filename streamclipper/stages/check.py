"""Stage 9: automatic checks on every rendered clip (M7). Each flag carries its reason."""
import json
from pathlib import Path

from .. import cache
from ..checks import flag
from ._clips import works

NAME = "check"
FILE_KEYS = ("checks_json",)


def cache_inputs(ctx) -> dict:
    return {"renders": cache.sig(ctx.out["renders_json"]), "render": ctx.cfg["render"], "v": 1}


def run(ctx, progress) -> dict:
    renders = json.loads(Path(ctx.out["renders_json"]).read_text(encoding="utf-8"))
    ws, flags = works(ctx), {}
    for n, w in enumerate(ws):
        cid = w.clip["cid"]
        if cid in renders["failed"]:
            flags[cid] = [flag("render_failed", renders["failed"][cid], "error")]
        else:
            flags[cid] = w.check()
        progress((n + 1) / max(len(ws), 1))
    path = cache.stage_dir(ctx.cfg, ctx.key, NAME) / "checks.json"
    path.write_text(json.dumps(flags), encoding="utf-8")
    return {"checks_json": str(path), "n_flagged": sum(1 for f in flags.values() if any(x["severity"] != "info" for x in f))}
