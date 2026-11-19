import argparse
import logging
import sys
import threading
import webbrowser

from . import RIGHTS_NOTICE, __version__
from .config import MODES, load_config
from .db import Database
from .errors import StreamClipperError
from .pipeline import run_job
from .stages.download import classify
from .stages.transcribe import LANGUAGES, fetch_models, normalize_language
from .startup import format_checks, require_ok, run_checks


def setup_logging(cfg) -> None:
    log_dir = cfg.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "streamclipper.log", encoding="utf-8")])


def add_job(cfg, db, source, mode=None, language=None) -> int:
    kind, source = classify(source)
    if language:
        language = normalize_language(language)
    return db.create_job(source, kind, mode or cfg["general"]["mode"], language)


def cmd_doctor(cfg, args) -> int:
    checks = run_checks(cfg)
    print(format_checks(checks))
    require_ok(checks)
    print("\nAll required checks passed.")
    return 0


def cmd_list(cfg, args) -> int:
    for j in Database(cfg.db_path).list_jobs():
        print(f"{j['id']:>4}  {j['status']:<8} {j['stage'] or '-':<10} {j['title'] or j['source']}")
    return 0


def _run(cfg, db, job_id) -> None:
    timer = run_job(cfg, db, job_id)
    print(f"Job {job_id} done in {timer.total:.1f}s (timing table above)")


def cmd_add(cfg, args) -> int:
    print(f"Added job {add_job(cfg, Database(cfg.db_path), args.source, args.mode, args.language)}")
    return 0


def cmd_run(cfg, args) -> int:
    db = Database(cfg.db_path)
    require_ok(run_checks(cfg))
    for job_id in ([args.job_id] if args.job_id else db.pending_job_ids()):
        _run(cfg, db, job_id)
    return 0


def cmd_process(cfg, args) -> int:
    db = Database(cfg.db_path)
    require_ok(run_checks(cfg))
    _run(cfg, db, add_job(cfg, db, args.source, args.mode, args.language))
    return 0


def cmd_update_ytdlp(cfg, args) -> int:
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "-U"] + (["--pre"] if args.nightly else [])
    print("Updating yt-dlp (network use). Restart StreamClipper afterwards.")
    code = subprocess.call(cmd + ["yt-dlp[default]"])
    if code:
        raise StreamClipperError("yt-dlp update failed. See the pip message above.")
    return 0


def cmd_export(cfg, args) -> int:
    from . import clipops
    db = Database(cfg.db_path)
    rows = [r for r in db.list_clips() if r["status"] == "ready" and (args.clip_id is None or r["id"] == args.clip_id)]
    if not rows:
        raise StreamClipperError("No ready clips to export.")
    for r in rows:
        print(f"clip {r['id']}: {clipops.export_final(cfg, db, r['id'])}")
    return 0


def cmd_live(cfg, args) -> int:
    from .live import LiveMonitor
    from .web.app import Worker
    require_ok(run_checks(cfg))
    db = Database(cfg.db_path)
    worker = Worker(cfg, db)
    worker.start()
    mon = LiveMonitor(cfg, db, worker.submit, args.source, 1, args.chat, args.mode, args.language)
    print(f"Monitoring {args.source}. Clips are made as moments happen (Ctrl+C to stop).")
    try:
        mon.start()
        while mon.is_alive():
            mon.join(1.0)
        if mon.error:
            raise StreamClipperError(mon.error)
        while not worker.q.empty() or worker.q.unfinished_tasks:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        mon.stop()
    return 0


def cmd_fetch_models(cfg, args) -> int:
    print("Downloading Whisper models (one-time, the only network use besides source downloads)...")
    print("Ready:", ", ".join(fetch_models(cfg)))
    return 0


def cmd_serve(cfg, args) -> int:
    import uvicorn
    from .web.app import create_app
    require_ok(run_checks(cfg))
    host, port = cfg["web"]["host"], cfg["web"]["port"]
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, [f"http://{host}:{port}"]).start()
    print(f"StreamClipper running at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
    return 0


def main(argv=None) -> int:
    if sys.version_info < (3, 11):
        sys.exit("StreamClipper needs Python 3.11 or newer.")
    p = argparse.ArgumentParser(prog="streamclipper", description=RIGHTS_NOTICE)
    p.add_argument("--config", help="path to a config.toml (default: the one next to this project)")
    p.add_argument("--debug", action="store_true", help="save debug output per clip (from M3)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check FFmpeg, libass, QSV, disk").set_defaults(fn=cmd_doctor)
    sub.add_parser("fetch-models", help="download Whisper models once").set_defaults(fn=cmd_fetch_models)
    e = sub.add_parser("export", help="final export (crf 18, 1080x1920) of ready clips")
    e.add_argument("clip_id", type=int, nargs="?")
    e.set_defaults(fn=cmd_export)
    lv = sub.add_parser("live", help="monitor a live stream and clip moments as they happen")
    lv.add_argument("source", help="live stream link or a file to replay")
    lv.add_argument("--chat", help="text file with one chat-message timestamp per line")
    lv.add_argument("--mode", choices=MODES)
    lv.add_argument("--language", choices=LANGUAGES)
    lv.set_defaults(fn=cmd_live)
    u = sub.add_parser("update-ytdlp", help="update yt-dlp (fixes most YouTube download errors)")
    u.add_argument("--nightly", action="store_true", help="use the nightly build (newest fixes)")
    u.set_defaults(fn=cmd_update_ytdlp)
    sub.add_parser("list", help="list jobs").set_defaults(fn=cmd_list)
    for name, fn, hlp in (("add", cmd_add, "add a source without running it"),
                          ("process", cmd_process, "add a source and run it now")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("source", help="link or file path")
        s.add_argument("--mode", choices=MODES)
        s.add_argument("--language", choices=LANGUAGES, help="default: config.toml [transcribe] language")
        s.set_defaults(fn=fn)
    r = sub.add_parser("run", help="run one job by id, or every queued job")
    r.add_argument("job_id", type=int, nargs="?")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config, {"general": {"debug": True}} if args.debug else None)
        setup_logging(cfg)
        if cfg["general"]["show_rights_notice"] and args.cmd in ("add", "process", "serve"):
            print(f"NOTICE: {RIGHTS_NOTICE}\n")
        return args.fn(cfg, args)
    except StreamClipperError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
