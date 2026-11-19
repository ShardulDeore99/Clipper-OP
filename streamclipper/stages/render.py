"""Stage 8: captions + one encode per clip (M5 + M6). One bad clip is recorded and shown, never hidden;
if every clip fails the job fails."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from .. import cache
from ..config import job_cfg
from ..errors import StageError
from ._clips import works

log = logging.getLogger("streamclipper")
NAME = "render"
FILE_KEYS = ("renders_json",)


def cache_inputs(ctx) -> dict:
    jc = job_cfg(ctx.cfg, ctx.job)
    return {"clips": cache.sig(ctx.out["clips_json"]), "captions": jc["captions"], "render": jc["render"],
            "framing": jc["framing"], "mode": ctx.job["mode"], "v": 1}


def run(ctx, progress) -> dict:
    ws, done = works(ctx), {"n": 0}
    results, failed = {}, {}

    def one(w):
        try:
            results[w.clip["cid"]] = w.render()
        except Exception as e:  # recorded on the clip and shown in the UI; the job fails if nothing worked
            log.exception("Render failed for clip %s", w.clip["cid"])
            failed[w.clip["cid"]] = str(e)
        done["n"] += 1
        progress(done["n"] / max(len(ws), 1))

    workers = max(1, min(2, int(ctx.cfg["render"]["max_parallel_renders"])))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, ws))
    if ws and not results:
        raise StageError(NAME, f"All {len(ws)} clips failed to render. First error: {next(iter(failed.values()))}")
    path = cache.stage_dir(ctx.cfg, ctx.key, NAME) / "renders.json"
    path.write_text(json.dumps({"results": results, "failed": failed}), encoding="utf-8")
    out = {"renders_json": str(path), "n_failed": len(failed)}
    if failed:
        out["no_cache"] = True  # Retry must try the failed clips again
    return out
