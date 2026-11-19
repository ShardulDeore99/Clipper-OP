"""Stage 1: get the source onto this machine. Local files are used in place (no copy)."""
import hashlib
import logging
import re
import shutil
from pathlib import Path

from .. import cache
from ..errors import StageError

log = logging.getLogger("streamclipper")
NAME = "download"
FILE_KEYS = ("video",)
URL_FIND = re.compile(r"https?://[^\s<>\"'\u201c\u201d\u2018\u2019]+", re.I)
BARE_URL = re.compile(r"^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/\S*)?$", re.I)
JUNK = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\ufeff"))  # invisible copy-paste characters
FILE_EXTS = {"mp4", "mkv", "mov", "avi", "webm", "flv", "wmv", "m4v", "mp3", "wav", "m4a", "aac",
             "ogg", "flac", "txt", "pdf", "zip", "json"}
ANSI = re.compile(r"\x1b?\[[0-9;]*m")


def classify(source: str) -> tuple[str, str]:
    """Return ("url"|"file", normalized source).

    Tolerates what copy-paste does: quotes, angle brackets, invisible characters, text around the
    link, and links without https:// (www.site.com/..., youtu.be/...).
    """
    s = source.translate(JUNK).strip().strip("\"'\u201c\u201d\u2018\u2019<>").strip()
    if not s:
        raise StageError("add", "The source is empty.", "Paste a link or a file path.")
    m = URL_FIND.search(s)
    if m:
        return "url", m.group().rstrip(".,;")
    p = Path(s).expanduser()
    if p.is_file():
        return "file", str(p.resolve())
    tld = s.split("/")[0].rsplit(".", 1)[-1].lower()
    if BARE_URL.match(s) and (s.lower().startswith("www.") or "/" in s) and tld not in FILE_EXTS:
        return "url", "https://" + s
    raise StageError("add", f"'{s}' is not a link or an existing file.",
                     "Links start with https://. For a file, paste its full path, e.g. C:\\Videos\\stream.mp4")


def source_key(kind: str, source: str) -> str:
    """Links: hash of the URL. Files: hash of the content, so a renamed file still hits the cache."""
    if kind == "url":
        return hashlib.sha256(source.encode()).hexdigest()[:16]
    h = hashlib.sha256()
    with open(source, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_inputs(ctx) -> dict:
    return {"source": ctx.job["source"], "max_height": ctx.cfg["download"]["max_height"]}


def run(ctx, progress) -> dict:
    if ctx.job["kind"] == "file":
        src = Path(ctx.job["source"])
        return {"video": str(src), "title": src.stem}
    return _download(ctx, progress)


# YouTube sometimes answers 403 on the media request. Attempt 2 changes how we ask
# (IPv4 only, chunked requests, more retries). Only 403s are retried; other errors stop at once.
ATTEMPTS = [("default", {}),
            ("IPv4, chunked, more retries",
             {"source_address": "0.0.0.0", "http_chunk_size": 10_485_760, "retries": 20, "fragment_retries": 100})]


def clean_error(msg: str) -> str:
    return ANSI.sub("", msg).replace("ERROR:", "").strip()


def js_runtime() -> str | None:
    """YouTube needs a JavaScript runtime for yt-dlp to solve its challenge."""
    return next((n for n in ("deno", "node", "bun", "qjs") if shutil.which(n)), None)


def _download(ctx, progress) -> dict:
    try:
        import yt_dlp
    except ImportError:
        raise StageError(NAME, "yt-dlp is not installed.", "Run setup.ps1, or add a local file instead.")
    cfg_d = ctx.cfg["download"]
    h = cfg_d["max_height"]
    out_dir = cache.stage_dir(ctx.cfg, ctx.key, NAME)

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(min(d.get("downloaded_bytes", 0) / total, 0.99))

    base = {
        "format": f"bv*[height<={h}]+ba/b[height<={h}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True, "no_color": True,
        "progress_hooks": [hook],
    }
    if cfg_d.get("cookies_from_browser"):
        base["cookiesfrombrowser"] = (cfg_d["cookies_from_browser"],)
    if cfg_d.get("cookies_file"):
        base["cookiefile"] = cfg_d["cookies_file"]
    if rt := js_runtime():
        base["js_runtimes"] = {rt: {}}

    tried = []
    for label, extra in ATTEMPTS:
        try:
            return _attempt(yt_dlp, ctx.job["source"], {**base, **extra}, out_dir)
        except yt_dlp.utils.DownloadError as e:
            msg = clean_error(str(e))
            tried.append(f"{label}: {msg}")
            log.warning("Download attempt '%s' failed: %s", label, msg)
            if "403" not in msg:
                break
    hint = "Check the link."
    if "403" in tried[-1]:
        hint = (r"Update yt-dlp, then restart the app: .\.venv\Scripts\python -m streamclipper update-ytdlp"
                " . If it still fails, set cookies_from_browser = \"firefox\" in config.toml [download],"
                " or download the video yourself and add the file.")
        if not js_runtime():
            hint += " YouTube also needs a JavaScript runtime: winget install DenoLand.Deno"
    raise StageError(NAME, "Download failed. " + " | ".join(tried), hint)


def _attempt(yt_dlp, url: str, opts: dict, out_dir: Path) -> dict:
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("is_live"):
            raise StageError(NAME, "This is a live stream that is still running.",
                             "Add it after it ends (live monitoring comes in milestone 8).")
        result = ydl.process_ie_result(info, download=True)
    req = result.get("requested_downloads") or []
    path = Path(req[0]["filepath"]) if req else None
    if not path or not path.is_file():
        found = sorted(out_dir.glob("source.*"))
        path = found[0] if found else None
    if not path:
        raise StageError(NAME, "Download finished but no video file was found.")
    return {"video": str(path), "title": result.get("title") or path.stem}
