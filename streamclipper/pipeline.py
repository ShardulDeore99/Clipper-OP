"""Staged, cached, resumable pipeline. Later milestones append stages to STAGES."""
import logging
import time
from dataclasses import dataclass, field

from . import cache
from .db import Database
from .errors import StageError
from .stages import audio, check, download, frame, plan, render, save, score, segment, transcribe
from .timing import Timer

log = logging.getLogger("streamclipper")

STAGES = [download, audio, transcribe, segment, score, plan, frame, render, check, save]  # implemented so far
PLANNED = ["download", "audio", "transcribe", "segment", "score", "plan",
           "frame", "render", "check", "save"]  # shown in the UI strip


@dataclass
class Context:
    cfg: object
    db: Database
    job: dict
    key: str
    out: dict = field(default_factory=dict)  # outputs of finished stages, shared downstream


def run_job(cfg, db: Database, job_id: int) -> Timer:
    db.update_job(job_id, status="running", error=None, stage="download", progress=0.0)
    db.clear_stage_runs(job_id)
    timer, current = Timer(), "download"
    try:
        job = db.get_job(job_id)
        kind, source = download.classify(job["source"])  # re-check: the file may have moved
        db.update_job(job_id, kind=kind, source=source)
        with timer.stage("identify"):
            key = download.source_key(kind, source)
        db.update_job(job_id, source_key=key)
        ctx = Context(cfg, db, db.get_job(job_id), key)

        for stage in STAGES:
            current = stage.NAME
            db.update_job(job_id, stage=current, progress=0.0)
            inputs = stage.cache_inputs(ctx)
            hit = None if getattr(stage, "ALWAYS", False) else cache.lookup(cfg, key, current, inputs, stage.FILE_KEYS)
            if hit is not None:
                ctx.out.update(hit)
                timer.add_cached(current)
                db.record_stage(job_id, current, 0.0, True)
                continue
            t0 = time.perf_counter()
            progress = lambda f, c=current: db.update_job(job_id, stage=c, progress=round(f, 3))
            outputs = stage.run(ctx, progress)
            secs = time.perf_counter() - t0
            if not outputs.pop("no_cache", False):  # a partial result is never cached, so Retry redoes the missing part
                cache.store(cfg, key, current, inputs, outputs, secs)  # only after success
            ctx.out.update(outputs)
            timer.rows.append((current, secs, False))
            db.record_stage(job_id, current, secs, False)

        meta = {k: ctx.out[k] for k in ("duration", "width", "height", "fps", "language", "words", "speech", "x_realtime", "n_clips", "n_flagged", "n_failed")
                if k in ctx.out}
        db.update_job(job_id, status="done", progress=1.0, title=ctx.out.get("title"), meta=meta)
        log.info("Job %s finished\n%s", job_id, timer.table())
        return timer
    except Exception as e:
        err = e if isinstance(e, StageError) else StageError(current, f"{type(e).__name__}: {e}")
        log.exception("Job %s failed in stage %s", job_id, err.stage)
        db.update_job(job_id, status="failed", stage=err.stage, error=str(err))
        raise err from e
