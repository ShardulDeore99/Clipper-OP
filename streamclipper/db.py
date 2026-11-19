"""SQLite store for jobs, per-stage timings and (from M3) clips."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL, kind TEXT NOT NULL, mode TEXT NOT NULL,
  title TEXT, source_key TEXT,
  status TEXT NOT NULL DEFAULT 'queued',      -- queued | running | done | failed
  stage TEXT, progress REAL DEFAULT 0,        -- progress = fraction of the current stage
  error TEXT, meta TEXT DEFAULT '{}',
  created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS stage_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL, stage TEXT NOT NULL,
  seconds REAL, cached INTEGER, finished REAL);
CREATE TABLE IF NOT EXISTS clips(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL, start REAL, end REAL, path TEXT,
  status TEXT, data TEXT DEFAULT '{}', created REAL);
"""
FIELDS = {"language", "title", "source", "kind", "mode", "source_key", "status", "stage", "progress", "error"}


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tx() as c:
            c.executescript(SCHEMA)
            if "language" not in [r["name"] for r in c.execute("PRAGMA table_info(jobs)")]:
                c.execute("ALTER TABLE jobs ADD COLUMN language TEXT")  # added in M2
            if "cid" not in [r["name"] for r in c.execute("PRAGMA table_info(clips)")]:
                c.execute("ALTER TABLE clips ADD COLUMN cid TEXT")  # added in M3

    @contextmanager
    def _tx(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    @staticmethod
    def _job(row) -> dict:
        d = dict(row)
        d["meta"] = json.loads(d["meta"] or "{}")
        return d

    def create_job(self, source: str, kind: str, mode: str, language: str | None = None) -> int:
        now = time.time()
        with self._tx() as c:
            cur = c.execute("INSERT INTO jobs(source,kind,mode,language,created,updated) VALUES(?,?,?,?,?,?)",
                            (source, kind, mode, language, now, now))
            return cur.lastrowid

    def get_job(self, job_id: int) -> dict:
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise LookupError(f"No job with id {job_id}")
        return self._job(row)

    def list_jobs(self) -> list[dict]:
        with self._tx() as c:
            return [self._job(r) for r in c.execute("SELECT * FROM jobs ORDER BY id DESC")]

    def pending_job_ids(self) -> list[int]:
        with self._tx() as c:
            return [r["id"] for r in c.execute(
                "SELECT id FROM jobs WHERE status IN ('queued','running') ORDER BY id")]

    def update_job(self, job_id: int, meta: dict | None = None, **fields) -> None:
        bad = set(fields) - FIELDS
        if bad:
            raise ValueError(f"Unknown job fields: {bad}")
        with self._tx() as c:
            if meta:
                row = c.execute("SELECT meta FROM jobs WHERE id=?", (job_id,)).fetchone()
                merged = {**json.loads(row["meta"] or "{}"), **meta}
                fields["meta"] = json.dumps(merged)
            fields["updated"] = time.time()
            sets = ", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))

    def record_stage(self, job_id: int, stage: str, seconds: float, cached: bool) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM stage_runs WHERE job_id=? AND stage=?", (job_id, stage))
            c.execute("INSERT INTO stage_runs(job_id,stage,seconds,cached,finished) VALUES(?,?,?,?,?)",
                      (job_id, stage, seconds, int(cached), time.time()))

    def stage_runs(self, job_id: int) -> list[dict]:
        with self._tx() as c:
            return [dict(r) for r in c.execute(
                "SELECT stage,seconds,cached FROM stage_runs WHERE job_id=? ORDER BY id", (job_id,))]

    def clear_stage_runs(self, job_id: int) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM stage_runs WHERE job_id=?", (job_id,))

    def delete_job(self, job_id: int) -> None:
        with self._tx() as c:
            for table in ("stage_runs", "clips"):
                c.execute(f"DELETE FROM {table} WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def jobs_with_key(self, key: str) -> int:
        with self._tx() as c:
            return c.execute("SELECT COUNT(*) FROM jobs WHERE source_key=?", (key,)).fetchone()[0]

    # ---- clips -----------------------------------------------------------------
    @staticmethod
    def _clip(row) -> dict:
        d = dict(row)
        d["data"] = json.loads(d["data"] or "{}")
        return d

    def upsert_clip(self, job_id: int, cid: str, start: float, end: float, path: str | None, status: str, data: dict) -> int:
        with self._tx() as c:
            row = c.execute("SELECT id FROM clips WHERE job_id=? AND cid=?", (job_id, cid)).fetchone()
            if row:
                c.execute("UPDATE clips SET start=?,end=?,path=?,status=?,data=? WHERE id=?",
                          (start, end, path, status, json.dumps(data), row["id"]))
                return row["id"]
            return c.execute("INSERT INTO clips(job_id,cid,start,end,path,status,data,created) VALUES(?,?,?,?,?,?,?,?)",
                             (job_id, cid, start, end, path, status, json.dumps(data), time.time())).lastrowid

    def get_clip(self, clip_id: int) -> dict:
        with self._tx() as c:
            row = c.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        if row is None:
            raise LookupError(f"No clip with id {clip_id}")
        return self._clip(row)

    def clip_by_cid(self, job_id: int, cid: str) -> dict | None:
        with self._tx() as c:
            row = c.execute("SELECT * FROM clips WHERE job_id=? AND cid=?", (job_id, cid)).fetchone()
        return self._clip(row) if row else None

    def list_clips(self, job_id: int | None = None) -> list[dict]:
        with self._tx() as c:
            q, args = "SELECT * FROM clips", ()
            if job_id is not None:
                q, args = q + " WHERE job_id=?", (job_id,)
            return [self._clip(r) for r in c.execute(q + " ORDER BY job_id DESC, start", args)]

    def update_clip(self, clip_id: int, status: str | None = None, data: dict | None = None,
                    start: float | None = None, end: float | None = None, path: str | None = None) -> None:
        with self._tx() as c:
            row = c.execute("SELECT data FROM clips WHERE id=?", (clip_id,)).fetchone()
            merged = {**json.loads(row["data"] or "{}"), **(data or {})}
            sets, args = ["data=?"], [json.dumps(merged)]
            for col, val in (("status", status), ("start", start), ("end", end), ("path", path)):
                if val is not None:
                    sets.append(f"{col}=?")
                    args.append(val)
            c.execute(f"UPDATE clips SET {', '.join(sets)} WHERE id=?", (*args, clip_id))

    def delete_clip(self, clip_id: int) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM clips WHERE id=?", (clip_id,))

    def delete_stale_clips(self, job_id: int, keep: set[str]) -> None:
        for row in self.list_clips(job_id):
            if row["cid"] not in keep and not row["data"].get("edited"):
                self.delete_clip(row["id"])
