"""Local web UI + JSON API. One worker thread runs jobs one at a time (never two heavy jobs at once)."""
import logging
import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from .. import RIGHTS_NOTICE, cache, clipops
from ..config import MODES, STYLE_CHOICES, clean_style
from ..credentials import available as credential_store_available, delete_secret, get_secret, set_secret
from ..db import Database
from ..errors import StreamClipperError
from ..framing import LAYOUTS
from ..live import LiveManager
from ..pipeline import PLANNED, STAGES, run_job
from ..stages.download import classify
from ..stages.transcribe import LANGUAGES, normalize_language
from clip_engine.ai.transcript_matcher import parse_targets

log = logging.getLogger("streamclipper")
STATIC = Path(__file__).parent / "static"


class Worker(threading.Thread):
    def __init__(self, cfg, db):
        super().__init__(daemon=True)
        self.cfg, self.db, self.q = cfg, db, queue.Queue()

    def submit(self, job_id: int) -> None:
        self.q.put(job_id)

    def run(self) -> None:
        while True:  # one task at a time: never transcribe and render together
            item = self.q.get()
            try:
                if isinstance(item, int):
                    run_job(self.cfg, self.db, item)
                elif item[0] == "edit":
                    clipops.apply_edit(self.cfg, self.db, item[1], item[2])
                elif item[0] == "export":
                    clipops.export_final(self.cfg, self.db, item[1])
            except Exception as e:  # already logged and stored on the job/clip; the UI shows it
                log.error("Task %s failed: %s", item, e)


def create_app(cfg):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    db = Database(cfg.db_path)
    worker = Worker(cfg, db)
    live = LiveManager(cfg, db, worker.submit)

    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        for job_id in db.pending_job_ids():  # resume anything interrupted by a restart
            worker.submit(job_id)
        yield

    app = FastAPI(title="StreamClipper", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    class Edit(BaseModel):
        start: float | None = None
        end: float | None = None
        layout: str | None = None
        caption_text: str | None = None

    class NewLive(BaseModel):
        source: str
        chat_file: str | None = None
        mode: str | None = None
        language: str | None = None
        style: dict | None = None

    class NewJob(BaseModel):
        source: str
        mode: str | None = None
        language: str | None = None
        style: dict | None = None
        moment_transcript: str | None = None

    class ProfileUpdate(BaseModel):
        youtube_api_key: str

    def view(job_id: int) -> dict:
        try:
            job = db.get_job(job_id)
        except LookupError:
            raise HTTPException(404, "Job not found")
        job["timings"] = db.stage_runs(job_id)
        return job

    @app.get("/")
    def index():
        return FileResponse(STATIC / "home.html")

    @app.get("/start")
    def start():
        return FileResponse(STATIC / "start.html")

    @app.get("/app")
    def studio():
        return FileResponse(STATIC / "index.html")

    @app.get("/profile")
    def profile_page():
        return FileResponse(STATIC / "profile.html")

    def youtube_credential_status():
        try:
            if get_secret():
                return {"configured": True, "source": "secure profile storage"}
        except Exception:
            pass
        if os.getenv("YOUTUBE_API_KEY"):
            return {"configured": True, "source": "environment variable"}
        return {"configured": False, "source": "not configured"}

    @app.get("/api/profile")
    def get_profile():
        # Never return the credential itself to the browser.
        return {"youtube": youtube_credential_status()}

    @app.put("/api/profile/youtube-key")
    def save_youtube_key(body: ProfileUpdate):
        value = body.youtube_api_key.strip()
        if not value or len(value) > 300:
            raise HTTPException(400, "Enter a valid YouTube API key.")
        try:
            if not credential_store_available():
                raise RuntimeError("Windows Credential Manager is unavailable.")
            set_secret(value)
        except Exception as e:
            log.warning("Could not save YouTube credential in OS key store: %s", type(e).__name__)
            raise HTTPException(503, "Secure credential storage is unavailable. The key was not saved.")
        return {"saved": True, "youtube": youtube_credential_status()}

    @app.delete("/api/profile/youtube-key")
    def remove_youtube_key():
        try:
            if not credential_store_available():
                raise RuntimeError("Windows Credential Manager is unavailable.")
            delete_secret()
        except Exception as e:
            log.warning("Could not remove YouTube credential from OS key store: %s", type(e).__name__)
            raise HTTPException(503, "Could not remove the saved key from secure credential storage.")
        return {"removed": True, "youtube": youtube_credential_status()}

    @app.get("/sources/{job_id}")
    def source_project(job_id: int):
        return FileResponse(STATIC / "index.html")

    @app.get("/clips/{clip_id}")
    def clip_detail(clip_id: int):
        get_clip(clip_id)
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    def meta():
        return {"planned": PLANNED, "implemented": [s.NAME for s in STAGES], "modes": list(MODES), "layouts": list(LAYOUTS), "languages": list(LANGUAGES), "style_choices": {k: list(v) for k, v in STYLE_CHOICES.items()}, "default_language": cfg["transcribe"]["language"],
                "default_mode": cfg["general"]["mode"], "notice": RIGHTS_NOTICE,
                "unlimited_clips": cfg["clips"]["max_per_source"] == 0}

    @app.get("/api/jobs")
    def jobs():
        return [view(j["id"]) for j in db.list_jobs()]

    @app.get("/api/jobs/{job_id}/analysis")
    def job_analysis(job_id: int):
        job = view(job_id)
        if not job.get("source_key"):
            return {"ready": False}
        root = cache.cache_root(cfg, job["source_key"])
        try:
            segment = json.loads((root / "segment" / "segment.json").read_text(encoding="utf-8"))
            candidates = json.loads((root / "score" / "candidates.json").read_text(encoding="utf-8"))["candidates"]
            plan = json.loads((root / "plan" / "clips.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            return {"ready": False}
        energy = segment.get("energy", {}).get("db", [])
        audio = segment.get("audio_analysis", {})
        stride = max(1, (len(energy) + 239) // 240)
        return {"ready": True, "duration": segment.get("duration", 0), "energy": energy[::stride],
                "candidates": [{"t": c["t"], "strength": c["strength"],
                                "humor": c.get("speech_analysis", {}).get("humor_score", 0),
                                "visual": c.get("visual_analysis", {}).get("visual_score")}
                               for c in candidates],
                "audio_score": audio.get("audio_score", 0), "audio_events": audio.get("events", []),
                "clips": [{"start": c["start"], "end": c["end"], "score": c["score"]} for c in plan.get("clips", [])],
                "counts": {"candidates": len(candidates), "clips": len(plan.get("clips", [])),
                           "rejected": len(plan.get("rejected", [])),
                           "visual": sum(bool(c.get("visual_analysis", {}).get("available")) for c in candidates),
                           "humor": sum(c.get("speech_analysis", {}).get("humor_score", 0) >= 35 for c in candidates)}}

    @app.post("/api/jobs")
    def add(body: NewJob):
        mode = body.mode or cfg["general"]["mode"]
        if mode not in MODES:
            raise HTTPException(400, f"Mode must be one of {list(MODES)}")
        try:
            kind, source = classify(body.source)
        except StreamClipperError as e:
            raise HTTPException(400, str(e))
        try:
            language = normalize_language(body.language) if body.language else None
        except StreamClipperError as e:
            raise HTTPException(400, str(e))
        try:
            style = clean_style(body.style)
        except StreamClipperError as e:
            raise HTTPException(400, str(e))
        transcript = (body.moment_transcript or "").strip()
        if transcript and len(transcript) > 12000:
            raise HTTPException(400, "The moment transcript must be under 12,000 characters.")
        if transcript and not parse_targets(transcript):
            raise HTTPException(400, "Add at least one exact spoken line with four or more words.")
        job_id = db.create_job(source, kind, mode, language)
        meta = {"style": style} if style else {}
        if transcript:
            meta.update({"moment_transcript": transcript, "transcript_only": True})
        if meta:
            db.update_job(job_id, meta=meta)
        worker.submit(job_id)
        return view(job_id)

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: int):
        job = view(job_id)
        if job["status"] not in ("failed", "done"):
            raise HTTPException(409, "This job is already queued or running.")
        db.update_job(job_id, status="queued", error=None)
        worker.submit(job_id)  # finished stages are skipped from the cache
        return view(job_id)

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: int):
        job = view(job_id)
        if job["status"] == "running":
            raise HTTPException(409, "Wait for the running job to finish, then remove it.")
        db.delete_job(job_id)
        if job["source_key"] and db.jobs_with_key(job["source_key"]) == 0:
            cache.remove(cfg, job["source_key"])
        return {"deleted": job_id}

    # ---- clips (M7) ---------------------------------------------------------------------------------
    def clip_view(row: dict) -> dict:
        d = row["data"]
        job = db.get_job(row["job_id"])
        keep = ("cid", "score", "duration", "key_moment", "start_reason", "end_reason", "hook_text", "why", "text",
                "flags", "meta", "layout", "layout_used", "caption_text", "edited", "error", "exporting", "notes",
                "provided_transcript_match")
        return {"id": row["id"], "job_id": row["job_id"], "job_title": job.get("title") or job["source"],
                "job_source": job["source"], "status": row["status"],
                "start": row["start"], "end": row["end"], "has_final": bool(d.get("files", {}).get("final")),
                "has_cover": bool(d.get("files", {}).get("cover")), "v": int(Path(row["path"]).stat().st_mtime) if row.get("path") and Path(row["path"]).is_file() else 0,
                **{k: d.get(k) for k in keep}}

    @app.get("/api/clips")
    def clips():
        return [clip_view(r) for r in db.list_clips()]

    @app.get("/api/clips/{clip_id}")
    def clip_detail_api(clip_id: int):
        return clip_view(get_clip(clip_id))

    def get_clip(clip_id: int) -> dict:
        try:
            return db.get_clip(clip_id)
        except LookupError:
            raise HTTPException(404, "Clip not found")

    @app.post("/api/clips/{clip_id}/edit")
    def edit_clip(clip_id: int, body: Edit):
        row = get_clip(clip_id)
        if row["status"] == "rendering" or row["data"].get("exporting"):
            raise HTTPException(409, "This clip is still rendering.")
        changes = {k: v for k, v in body.model_dump().items() if v is not None}
        if not changes:
            raise HTTPException(400, "Nothing to change.")
        job = db.get_job(row["job_id"])
        try:
            from ..clipwork import load_source
            clipops.validate(cfg, changes, row["data"], load_source(cfg, job["source_key"]).duration)
        except ValueError as e:
            raise HTTPException(400, str(e))
        db.update_clip(clip_id, status="rendering")
        worker.submit(("edit", clip_id, changes))
        return {"queued": clip_id}

    @app.post("/api/clips/{clip_id}/export")
    def export_clip(clip_id: int):
        row = get_clip(clip_id)
        if row["status"] != "ready" or row["data"].get("exporting"):
            raise HTTPException(409, "The clip is not ready, or an export is already running.")
        db.update_clip(clip_id, data={"exporting": True})
        worker.submit(("export", clip_id))
        return {"queued": clip_id}

    @app.get("/api/clips/{clip_id}/file/{kind}")
    def clip_file(clip_id: int, kind: str):
        row = get_clip(clip_id)
        path = row["data"].get("files", {}).get(kind)
        if kind not in ("review", "final", "cover") or not path or not Path(path).is_file():
            raise HTTPException(404, "File not available")
        name = f"{row['data'].get('meta', {}).get('title', 'clip')[:50]}".strip().replace("/", "-") or "clip"
        return FileResponse(path, media_type="image/jpeg" if kind == "cover" else "video/mp4",
                            filename=f"{name}.{'jpg' if kind == 'cover' else 'mp4'}" if kind != "review" else None)

    @app.delete("/api/clips/{clip_id}")
    def remove_clip(clip_id: int):
        row = get_clip(clip_id)
        if row["status"] == "rendering":
            raise HTTPException(409, "Wait for the render to finish.")
        clipops.delete_clip(cfg, db, clip_id)
        return {"deleted": clip_id}

    # ---- live monitor (M8) -------------------------------------------------------------------------
    @app.get("/api/live")
    def live_list():
        return live.list()

    @app.post("/api/webhooks/kick")
    async def kick_webhook(request: Request):
        """Accept Kick chat only after verifying its signature against the official rotating key."""
        import base64
        import urllib.request
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        raw = await request.body()
        message_id = request.headers.get("Kick-Event-Message-Id", "")
        timestamp = request.headers.get("Kick-Event-Message-Timestamp", "")
        signature = request.headers.get("Kick-Event-Signature", "")
        if not message_id or not timestamp or not signature:
            raise HTTPException(401, "Missing Kick signature headers")
        try:
            with urllib.request.urlopen("https://api.kick.com/public/v1/public-key", timeout=10) as response:
                key_doc = response.read().decode()
            try:
                parsed = json.loads(key_doc)
                key_doc = parsed.get("public_key", parsed.get("data", {}).get("public_key", key_doc))
            except json.JSONDecodeError:
                pass
            public_key = serialization.load_pem_public_key(key_doc.encode())
            public_key.verify(base64.b64decode(signature), f"{message_id}.{timestamp}.".encode() + raw,
                              padding.PKCS1v15(), hashes.SHA256())
            data = json.loads(raw)
        except Exception:
            raise HTTPException(401, "Kick webhook signature could not be verified")
        if request.headers.get("Kick-Event-Type") != "chat.message.sent":
            return {"ok": True, "accepted": 0}
        broadcaster = data.get("broadcaster", {})
        count = live.kick_event(broadcaster.get("username", ""), data.get("content", ""),
                                request.headers.get("Kick-Event-Message-Id") or data.get("message_id", ""),
                                (data.get("sender") or {}).get("user_id") or (data.get("sender") or {}).get("username", ""),
                                broadcaster.get("channel_slug", ""))
        return {"ok": True, "accepted": count}

    @app.post("/api/live")
    def live_start(body: NewLive):
        try:
            m = live.start(body.source.strip().strip('"'), body.chat_file, body.mode, body.language, clean_style(body.style))
        except StreamClipperError as e:
            raise HTTPException(400, str(e))
        return m.view()

    @app.delete("/api/live/{monitor_id}")
    def live_stop(monitor_id: int):
        if monitor_id not in live.monitors:
            raise HTTPException(404, "No such monitor")
        live.stop(monitor_id)
        return {"stopped": monitor_id}

    return app
