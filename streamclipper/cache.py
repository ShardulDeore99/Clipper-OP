"""Per-stage cache: data/cache/<source_key>/<stage>/ with a done.json written only on success."""
import hashlib
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("streamclipper")


def cache_root(cfg, key: str) -> Path:
    return cfg.data_dir / "cache" / key


def stage_dir(cfg, key: str, stage: str) -> Path:
    p = cache_root(cfg, key) / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


def lookup(cfg, key: str, stage: str, inputs: dict, file_keys=()) -> dict | None:
    """Return cached outputs if inputs match and every output file still exists."""
    f = cache_root(cfg, key) / stage / "done.json"
    if not f.is_file():
        return None
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("Corrupt cache record %s; recomputing %s", f, stage)
        return None
    if rec.get("inputs") != inputs:
        return None
    out = rec["outputs"]
    if any(not Path(out[k]).is_file() for k in file_keys):
        return None
    return out


def store(cfg, key: str, stage: str, inputs: dict, outputs: dict, seconds: float) -> None:
    d = stage_dir(cfg, key, stage)
    tmp = d / "done.json.tmp"
    tmp.write_text(json.dumps({"inputs": inputs, "outputs": outputs, "seconds": round(seconds, 3)}),
                   encoding="utf-8")
    tmp.replace(d / "done.json")


def remove(cfg, key: str) -> None:
    shutil.rmtree(cache_root(cfg, key), ignore_errors=True)


def sig(path, content: bool = True) -> str:
    """Cheap change signature for an input file: size, plus a content hash for small files."""
    p = Path(path)
    st = p.stat()
    if content and st.st_size <= 16 * 1024 * 1024:
        return f"{st.st_size}:{hashlib.sha1(p.read_bytes()).hexdigest()[:16]}"
    return f"{st.st_size}:{int(st.st_mtime)}"
