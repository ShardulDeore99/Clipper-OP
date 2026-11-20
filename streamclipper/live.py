"""M8 live monitor: a rolling recording of a live stream, triggers from audio bursts and chat spikes, record
on until a pause (or a max wait), merge close triggers, cut from the buffer, then send the clip through the
same pipeline (transcribe, snap boundaries, frame, captions, render, checks).

Everything is in stream time (seconds since the monitor started), so it behaves the same on a real stream
and on a file."""
import csv
import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from . import audio_features as af
from .credentials import get_secret
from .errors import StageError
from clip_engine.detectors.audio_detector import analyze_chunk

log = logging.getLogger("streamclipper")
CHUNK = 1600  # samples per 0.1 s at 16 kHz


LIVE_PY_VERSION = "v4"


def youtube_api_key() -> str | None:
    """Prefer the user's OS credential store; environment variable is a deployment fallback."""
    try:
        stored = get_secret()
        if stored:
            return stored
    except Exception:
        pass
    return os.getenv("YOUTUBE_API_KEY")


def resolve_stream_url(source: str) -> list[str]:
    """A direct media URL or a local file is used as is; page links go through yt-dlp."""
    if Path(source).is_file():
        return [source]
    if not re.match(r"https?://", source, re.I):
        raise StageError("live", "Give a live stream link, or a video file path.")
    if source.lower().split("?")[0].endswith((".m3u8", ".mpd", ".mp4", ".flv", ".ts", ".mkv")):
        return [source]
    try:
        import yt_dlp
    except ImportError:
        raise StageError("live", "yt-dlp is not installed.", "Run setup.ps1.")
    from .stages.download import clean_error, js_runtime
    # ffmpeg needs ONE url carrying video and audio, so pick a muxed format ourselves instead of letting a
    # strict selector fail with "Requested format is not available".
    opts = {"quiet": True, "no_warnings": True, "no_color": True, "noplaylist": True,
            "ignore_no_formats_error": True, "format": "best[height<=720]/best/worst"}
    rt = js_runtime()
    if rt:
        opts["js_runtimes"] = {rt: {}}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
    except yt_dlp.utils.DownloadError as e:
        hint = "Update yt-dlp: python -m streamclipper update-ytdlp"
        if not rt:
            hint += ". YouTube also needs a JavaScript runtime: winget install DenoLand.Deno (then restart the app)"
        raise StageError("live", f"Could not open the stream: {clean_error(str(e))}", hint)
    fmts = [f for f in (info.get("formats") or []) if f.get("url")]
    # HLS formats often leave vcodec/acodec unset (None), so classify by shape: no picture size = audio only.
    is_audio = lambda f: f.get("vcodec") == "none" or not (f.get("height") or f.get("width"))
    has_v = lambda f: not is_audio(f)
    has_a = lambda f: f.get("acodec") not in (None, "none")
    cap = lambda fs: [f for f in fs if (f.get("height") or 0) <= 720] or fs
    key = lambda f: (f.get("height") or 0, f.get("tbr") or 0)
    if info.get("url"):
        return [info["url"]]
    muxed = cap([f for f in fmts if has_v(f) and has_a(f)])
    if muxed:
        return [max(muxed, key=key)["url"]]
    video = cap([f for f in fmts if has_v(f) and not has_a(f)])
    audio = [f for f in fmts if is_audio(f)]
    if video and audio:  # separate video and audio streams: ffmpeg reads both
        return [min(video, key=lambda f: abs((f.get("height") or 0) - 720))["url"],
                max(audio, key=lambda f: f.get("abr") or f.get("tbr") or 0)["url"]]
    seen = ", ".join(f"{f.get('format_id')}:{f.get('height') or 'a'}" for f in fmts[:12]) or "none"
    hint = (f"Live now: {info.get('is_live')}; live status: {info.get('live_status')}; formats found: {seen}. "
            "Update yt-dlp and make sure the stream is live.")
    if not rt:
        hint = "No JavaScript runtime found: winget install DenoLand.Deno, restart the app. " + hint
    raise StageError("live", f"yt-dlp found no playable video+audio stream for this link. [live.py {LIVE_PY_VERSION}]", hint)


class ChatFeed:
    """Optional chat spikes from a text file, one timestamp per line: stream seconds, or epoch seconds if > 1e9."""

    def __init__(self, path, start_epoch: float):
        self.path, self.start, self.pos, self.times = (Path(path) if path else None), start_epoch, 0, deque(maxlen=5000)
        self.messages = deque(maxlen=5000)
        self.ids, self.id_order = set(), deque(maxlen=5000)
        self.total_count = 0
        self.lock = threading.Lock()
        self.provider_status = "file" if path else "waiting"
        self.provider_error = None
        self.last_message = None

    def add(self, text: str = "", user: str = "", message_id: str | None = None, t: float | None = None) -> None:
        """Store a chat event once; provider callbacks use monitor-relative receive time."""
        if message_id and message_id in self.ids:
            return
        if message_id:
            if len(self.id_order) == self.id_order.maxlen:
                self.ids.discard(self.id_order[0])
            self.id_order.append(message_id)
            self.ids.add(message_id)
        at = max(0.0, time.monotonic() - self.start_mono) if t is None else max(0.0, t)
        with self.lock:
            self.times.append(at)
            self.messages.append({"t": at, "user": user or "viewer", "text": text[:500]})
            self.total_count += 1
            self.last_message = {"user": user or "viewer", "text": text[:160], "t": round(at, 1)}

    def start_clock(self):
        self.start_mono = time.monotonic()

    def poll(self) -> None:
        if not self.path or not self.path.is_file():
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        for line in data.decode("utf-8", "replace").splitlines():
            try:
                v = float(line.strip().split(",")[0])
            except ValueError:
                continue
            self.add(t=v - self.start if v > 1e9 else v, user=f"file-{self.total_count + 1}")

    def spike(self, now: float, ratio: float) -> bool:
        with self.lock:
            times = list(self.times)
            msgs = [m for m in self.messages if now - 10 <= m["t"] <= now]
        recent = len(msgs) if msgs else sum(1 for t in times if now - 10 <= t <= now)
        unique = len({m["user"] for m in msgs}) if msgs else recent
        old = [sum(1 for t in times if a <= t < a + 10) for a in np.arange(max(0, now - 310), now - 10, 10)]
        base = max(float(np.median(old)) if old else 0.0, 0.5)
        # Suppress a single chatter spamming; a moment should draw distinct viewers.
        return recent >= 5 and unique >= min(3, recent) and recent >= ratio * base

    def snapshot(self, now: float) -> dict:
        with self.lock:
            recent = [m for m in self.messages if now - 10 <= m["t"] <= now]
            msgs = list(self.messages)
        laughs = sum(bool(re.search(r"\b(lol|lmao|lmfao|haha+|hehe+|kekw|lolol)\b", m["text"], re.I)) for m in recent)
        clip_cues = sum(bool(re.search(r"\b(clip that|clip it|clip|no way|what|bruh|omg|insane)\b", m["text"], re.I)) for m in recent)
        return {"count": len(recent), "unique": len({m["user"] for m in recent}),
                "laughs": laughs, "clip_cues": clip_cues, "last_message": self.last_message,
                "total": self.total_count}


def detect_platform(source: str) -> str:
    host = (urlparse(source).hostname or "").lower().removeprefix("www.")
    if host == "twitch.tv" or host.endswith(".twitch.tv"): return "twitch"
    if host in ("youtube.com", "youtu.be") or host.endswith(".youtube.com"): return "youtube"
    if host == "kick.com" or host.endswith(".kick.com"): return "kick"
    return "custom"


def platform_channel(source: str, platform: str) -> str | None:
    u = urlparse(source); bits = [p for p in u.path.split("/") if p]
    if platform == "twitch": return bits[0] if bits else None
    if platform == "kick": return bits[0] if bits else None
    if platform == "youtube":
        q = parse_qs(u.query)
        for key in ("v",):
            if q.get(key): return q[key][0]
        if bits and bits[0] in ("live", "embed") and len(bits) > 1: return bits[1]
        if bits and bits[-1] == "live" and bits[0].startswith("@"): return bits[0]
        return bits[-1] if bits and u.hostname and u.hostname.endswith("youtu.be") else None
    return None


class PlatformChat(threading.Thread):
    """Authorized platform chat connector. Kick events are delivered to its signed webhook endpoint."""
    def __init__(self, monitor, platform, channel):
        super().__init__(daemon=True); self.monitor, self.platform, self.channel = monitor, platform, channel

    def run(self):
        feed = self.monitor.chat
        while not self.monitor.stop_evt.is_set():
            try:
                feed.provider_status = "connecting" if not feed.provider_error else "reconnecting"
                if self.platform == "youtube": self._youtube()
                elif self.platform == "twitch": self._twitch()
                elif self.platform == "kick": self._kick()
                return
            except Exception as e:
                feed.provider_error = str(e)[:300]
                if isinstance(e, RuntimeError) or (isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429):
                    feed.provider_status = "setup required"
                    if not getattr(self, "noted_error", False):
                        self.monitor.note(f"{self.platform} chat setup: {str(e)[:180]}")
                        self.noted_error = True
                    return
                feed.provider_status = "reconnecting"
                if not getattr(self, "noted_error", False):
                    self.monitor.note(f"{self.platform} chat unavailable: {str(e)[:160]} (retrying)")
                    self.noted_error = True
                self.monitor.stop_evt.wait(5)

    def _kick(self):
        token = os.getenv("KICK_ACCESS_TOKEN")
        broadcaster_id = os.getenv("KICK_BROADCASTER_USER_ID")
        if not token or not broadcaster_id:
            self.monitor.chat.provider_status = "webhook setup required"
            self.monitor.chat.provider_error = "Set KICK_ACCESS_TOKEN (events:subscribe) and KICK_BROADCASTER_USER_ID; configure the public webhook URL in Kick Developer settings."
            return
        payload = json.dumps({"broadcaster_user_id": int(broadcaster_id), "method": "webhook",
                              "events": [{"name": "chat.message.sent", "version": 1}]}).encode()
        req = urllib.request.Request("https://api.kick.com/public/v1/events/subscriptions", data=payload,
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read())
        self.monitor.chat.provider_status = "webhook subscribed"
        if result.get("message"):
            self.monitor.chat.provider_error = str(result["message"])

    @staticmethod
    def _get(url, headers=None):
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=15) as res: return json.loads(res.read())

    def _youtube(self):
        key = youtube_api_key()
        if not key: raise RuntimeError("Set YOUTUBE_API_KEY to connect YouTube Live Chat.")
        base = "https://www.googleapis.com/youtube/v3/"
        video = self.channel
        if video.startswith("@"):
            channels = self._get(base + "channels?" + urllib.parse.urlencode({"part":"id", "forHandle":video[1:], "key":key}))
            channel_ids = [x["id"] for x in channels.get("items", [])]
            if not channel_ids: raise RuntimeError("YouTube channel handle was not found.")
            found = self._get(base + "search?" + urllib.parse.urlencode({"part":"id", "channelId":channel_ids[0], "eventType":"live", "type":"video", "maxResults":1, "key":key}))
            video = (found.get("items") or [{}])[0].get("id", {}).get("videoId")
            if not video: raise RuntimeError("This YouTube channel is not live right now.")
        details = self._get(base + "videos?" + urllib.parse.urlencode({"part":"liveStreamingDetails", "id":video, "key":key}))
        items = details.get("items", [])
        if not items: raise RuntimeError("YouTube video was not found or is not accessible.")
        chat_id = items[0].get("liveStreamingDetails", {}).get("activeLiveChatId")
        if not chat_id: raise RuntimeError("No active YouTube live chat was found for this video.")
        self.monitor.chat.provider_status = "connected"
        token = None
        while not self.monitor.stop_evt.is_set():
            params = {"part":"snippet,authorDetails", "liveChatId":chat_id, "key":key, "maxResults":200}
            if token: params["pageToken"] = token
            data = self._get(base + "liveChat/messages?" + urllib.parse.urlencode(params))
            for m in data.get("items", []):
                s = m.get("snippet", {}); a = m.get("authorDetails", {})
                self.monitor.chat.add(s.get("displayMessage", ""), a.get("channelId") or a.get("displayName", ""), m.get("id"))
            token = data.get("nextPageToken", token)
            self.monitor.chat.provider_status = "connected"
            self.monitor.stop_evt.wait(max(.5, data.get("pollingIntervalMillis", 5000) / 1000))

    def _twitch(self):
        token, client = os.getenv("TWITCH_USER_ACCESS_TOKEN"), os.getenv("TWITCH_CLIENT_ID")
        if not token or not client: raise RuntimeError("Set TWITCH_CLIENT_ID and TWITCH_USER_ACCESS_TOKEN (user:read:chat scope).")
        try: import asyncio, websockets
        except ImportError: raise RuntimeError("Install websocket-client support with setup.ps1 (websockets package).")
        async def consume():
            api_headers = {"Client-ID":client, "Authorization":"Bearer " + token, "Content-Type":"application/json"}
            identity = self._get("https://id.twitch.tv/oauth2/validate", {"Authorization":"OAuth " + token})
            broadcaster = self._get("https://api.twitch.tv/helix/users?" + urllib.parse.urlencode({"login":self.channel}), api_headers).get("data", [])
            if not broadcaster: raise RuntimeError("Twitch channel was not found.")
            async with websockets.connect("wss://eventsub.wss.twitch.tv/ws", ping_interval=20) as ws:
                welcome = json.loads(await ws.recv()); session = welcome["payload"]["session"]["id"]
                body = json.dumps({"type":"channel.chat.message", "version":"1", "condition":{"broadcaster_user_id":broadcaster[0]["id"], "user_id":identity["user_id"]}, "transport":{"method":"websocket", "session_id":session}}).encode()
                req = urllib.request.Request("https://api.twitch.tv/helix/eventsub/subscriptions", data=body, headers=api_headers, method="POST")
                with urllib.request.urlopen(req, timeout=15): pass
                self.monitor.chat.provider_status = "connected"
                while not self.monitor.stop_evt.is_set():
                    try: msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                    except asyncio.TimeoutError: continue
                    typ = msg.get("metadata", {}).get("message_type")
                    if typ == "notification":
                        e = msg.get("payload", {}).get("event", {}); payload = e.get("message", {})
                        self.monitor.chat.add(payload.get("text", ""), e.get("chatter_user_id") or e.get("chatter_user_name", ""), e.get("message_id"))
                    elif typ == "session_reconnect": self.monitor.chat.provider_status = "reconnecting"
        asyncio.run(consume())


def read_segments(csv_path: Path) -> list[tuple[Path, float, float]]:
    out = []
    if csv_path.is_file():
        for row in csv.reader(csv_path.read_text().splitlines()):
            if len(row) >= 3:
                try:
                    out.append((csv_path.parent / row[0], float(row[1]), float(row[2])))
                except ValueError:
                    pass
    return out


class LiveMonitor(threading.Thread):
    def __init__(self, cfg, db, submit, source: str, monitor_id: int = 1, chat_file: str | None = None,
                 mode: str | None = None, language: str | None = None, style: dict | None = None):
        super().__init__(daemon=True)
        self.style = style or {}
        self.cfg, self.db, self.submit, self.source = cfg, db, submit, source
        self.id, self.chat_file = monitor_id, chat_file
        self.mode, self.language = mode or cfg["general"]["mode"], language
        self.stop_evt = threading.Event()
        self.proc = None
        self.status, self.error = "starting", None
        self.events = deque(maxlen=50)
        self.telemetry = deque(maxlen=240)
        self.triggers = self.clips = 0
        self.platform = detect_platform(source)
        self.channel = platform_channel(source, self.platform)
        self.chat = ChatFeed(chat_file, time.time())
        self.chat.start_clock()
        self.chat_connector = None

    def note(self, text: str) -> None:
        self.events.appendleft({"at": time.strftime("%H:%M:%S"), "text": text})
        log.info("live %s: %s", self.id, text)

    def stop(self) -> None:
        self.stop_evt.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def view(self) -> dict:
        now = self.telemetry[-1]["t"] if self.telemetry else 0
        return {"id": self.id, "source": self.source, "status": self.status, "error": self.error,
                "triggers": self.triggers, "clips": self.clips, "events": list(self.events),
                "platform": self.platform, "channel": self.channel,
                "chat_status": self.chat.provider_status, "chat_error": self.chat.provider_error,
                "chat_stats": self.chat.snapshot(now), "timeline": list(self.telemetry)}

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:  # shown in the UI; never silent
            self.status, self.error = "error", str(e)
            log.exception("Live monitor %s failed", self.id)

    # ------------------------------------------------------------------------------------------------
    def _run(self) -> None:
        L, ff = self.cfg["live"], self.cfg["tools"]["ffmpeg"]
        urls = resolve_stream_url(self.source)
        if self.platform in ("youtube", "twitch", "kick") and self.channel:
            self.chat_connector = PlatformChat(self, self.platform, self.channel)
            self.chat_connector.start()
        elif self.chat_file:
            self.chat.provider_status = "file connected"
        va = 1 if len(urls) > 1 else 0  # input index that carries the audio
        self.buf = self.cfg.data_dir / "live" / f"{self.id}_{int(time.time())}"
        (self.buf / "clips").mkdir(parents=True)
        self.list_csv = self.buf / "list.csv"
        cmd = [ff, "-hide_banner", "-loglevel", "error",
               *[x for u in urls for x in ("-i", u)],
               "-map", "0:v:0", "-map", f"{va}:a:0", "-c", "copy", "-f", "segment", "-segment_time", "2",
               "-segment_format", "mpegts", "-segment_list", str(self.list_csv), "-segment_list_type", "csv",
               "-segment_list_flags", "+live", "-reset_timestamps", "1", str(self.buf / "seg_%06d.ts"),
               "-map", f"{va}:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.status = "recording"
        self.note(f"recording started (live.py {LIVE_PY_VERSION})")
        chat = self.chat
        hist, samples = deque(maxlen=3000), 0           # 5 minutes of 0.1 s levels
        burst_run, in_burst, quiet_run, thr = 0.0, False, 0.0, -60.0
        pending, last_clean, last_chart_second, live_score_peak = None, 0.0, -1, 0.0
        last_score_trigger = -1e9

        while not self.stop_evt.is_set():
            data = self._read(CHUNK * 2)
            if not data:
                break
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            db = float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-5))
            samples += len(x)
            now = samples / 16000.0
            hist.append(db)
            if len(hist) < 50:
                continue
            base = float(np.median(hist))
            floor = float(np.percentile(hist, 15))
            live_audio = analyze_chunk(x, 16000, base, floor)
            live_score_peak = max(live_score_peak, live_audio["audio_score"])
            if int(now * 10) % 10 == 0:                     # once a second
                thr = af.silence_threshold(np.array(hist))[0]
                chat.poll()
                if now - last_clean > 10:
                    self._clean(now, pending)
                    last_clean = now
            quiet_run = quiet_run + 0.1 if db < thr else 0.0
            delta = db - base
            burst_run = burst_run + 0.1 if delta >= L["burst_db"] else 0.0
            chat_stats = chat.snapshot(now)
            # Signals are normalized to a 0..100 score for the dashboard and cut metadata.
            audio_component = float(live_audio["audio_score"])
            chat_component = min(100.0, chat_stats["count"] * 8.0 + chat_stats["unique"] * 4.0 + chat_stats["laughs"] * 4.0 + chat_stats["clip_cues"] * 5.0)
            moment_score = round(0.62 * audio_component + 0.38 * chat_component)
            fired = None
            if burst_run >= L["burst_seconds"] and not in_burst:
                in_burst, fired = True, {"t": now - burst_run, "reason": f"audio burst (+{delta:.0f} dB)", "delta": delta,
                                         "score": moment_score, "signals": {"audio": round(audio_component), "chat": round(chat_component)}}
            if burst_run == 0.0:
                in_burst = False
            if fired is None and chat.times and int(now * 10) % 10 == 0 and chat.spike(now, L["chat_spike_ratio"]) and not pending:
                fired = {"t": now - 3, "reason": "chat spike", "delta": 0.0, "score": moment_score,
                         "signals": {"audio": round(audio_component), "chat": round(chat_component)}}
            if fired is None and not pending and now - last_score_trigger >= 10 and moment_score >= 72:
                fired = {"t": now - 3, "reason": f"combined moment score ({moment_score}/100)", "delta": delta,
                         "score": moment_score, "signals": {"audio": round(audio_component), "chat": round(chat_component)}}
                last_score_trigger = now
            if fired:
                self.triggers += 1
                if pending and now - pending["last"] <= L["merge_seconds"]:
                    pending["last"] = now
                    if fired.get("score", 0) > pending.get("score", 0):
                        pending["score"], pending["signals"] = fired.get("score", 0), fired.get("signals", {})
                    self.note(f"trigger merged into the clip at {pending['t0']:.0f} s")
                elif not pending:
                    pending = {"t0": fired["t"], "last": now, "burst_end": now,
                               **{k: fired[k] for k in ("reason", "delta", "score", "signals") if k in fired}}
                    self.note(f"trigger at {fired['t']:.0f} s: {fired['reason']}")
            elif pending and burst_run > 0:
                pending["last"] = pending["burst_end"] = now
            if pending:
                since = now - pending["last"]
                too_long = (now - pending["t0"] + L["lead_seconds"]) >= L["max_clip_seconds"] - 2
                if (since >= max(1.0, L["hold_seconds"]) and quiet_run >= 0.7) or since >= L["max_wait_seconds"] or too_long:
                    why = "pause" if quiet_run >= 0.7 else "max wait"
                    self._finish(dict(pending), now, why)
                    pending = None
            chart_second = int(now)
            if chart_second > last_chart_second:
                last_chart_second = chart_second
                recent_chat = sum(1 for t in chat.times if now - 10 <= t <= now)
                self.telemetry.append({"t": round(now, 1), "energy": round(max(-5.0, min(20.0, delta)), 1),
                                       "audio_score": live_score_peak,
                                       "chat": recent_chat, "unique_chat": chat_stats["unique"],
                                       "laughs": chat_stats["laughs"], "clip_cues": chat_stats["clip_cues"],
                                       "moment_score": moment_score, "trigger": bool(fired)})
                live_score_peak = 0.0
        if pending:  # the stream (or file) ended while a clip was waiting
            self._finish(dict(pending), samples / 16000.0, "end of stream")
        self.status = "stopped" if self.stop_evt.is_set() else "finished"
        self.note(self.status)
        self.stop_evt.set()  # also release a platform chat reader when the media source ends naturally
        self._wait_cutters()
        for pipe in (self.proc.stdout, self.proc.stderr):
            pipe.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = self.proc.stdout.read(n - len(buf))
            if not part:
                break
            buf += part
        return buf if len(buf) == n else b""

    def _clean(self, now: float, pending) -> None:
        """Delete recording older than buffer_seconds (but never what a waiting clip needs)."""
        keep_from = now - self.cfg["live"]["buffer_seconds"]
        if pending:
            keep_from = min(keep_from, pending["t0"] - self.cfg["live"]["lead_seconds"] - 10)
        for path, _, end in read_segments(self.list_csv):
            if end < keep_from and path.is_file():
                path.unlink(missing_ok=True)

    # --- cutting ----------------------------------------------------------------------------------
    def _finish(self, trig: dict, end_t: float, why: str) -> None:
        self.note(f"recording until {end_t:.0f} s ({why}); cutting")
        t = threading.Thread(target=self._cut, args=(trig, end_t), daemon=True)
        t.start()
        self._cutters = getattr(self, "_cutters", []) + [t]

    def _wait_cutters(self) -> None:
        for t in getattr(self, "_cutters", []):
            t.join(timeout=120)

    def _cut(self, trig: dict, end_t: float) -> None:
        try:
            L, ff = self.cfg["live"], self.cfg["tools"]["ffmpeg"]
            end_t = end_t + 1.0
            # enough context for the planner: the lead-in, and never less than the minimum clip length plus a margin
            start_t = max(0.0, min(trig["t0"] - L["lead_seconds"] - 2.0, end_t - (self.cfg["clips"]["min_seconds"] + 6)))
            end_t = min(end_t, start_t + L["max_clip_seconds"] + 4.0)
            deadline = time.time() + 20
            while time.time() < deadline:        # wait until the recording has caught up with end_t
                segs = read_segments(self.list_csv)
                if (segs and segs[-1][2] >= end_t) or (self.proc and self.proc.poll() is not None):
                    break
                time.sleep(0.3)
            segs = [s for s in read_segments(self.list_csv) if s[2] > start_t and s[1] < end_t and s[0].is_file()]
            if not segs:
                self.note("nothing in the buffer for this trigger (too old?)")
                return
            n = self.clips + 1
            lst = self.buf / f"concat_{n}.txt"
            lst.write_text("".join(f"file '{p.as_posix()}'\n" for p, _, _ in segs))
            rough = self.buf / "clips" / f"live_{self.id}_{int(time.time())}_{n}.mkv"
            p = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                                "-c", "copy", str(rough)], capture_output=True, text=True)
            if p.returncode != 0 or not rough.is_file():
                raise StageError("live", "Cutting the buffer failed: " + p.stderr[-300:])
            origin = segs[0][1]
            jid = self.db.create_job(str(rough), "file", self.mode, self.language)
            self.db.update_job(jid, title=f"Live moment at {int(trig['t0'])} s", meta={"style": self.style, "live_trigger": {
                "t": round(trig["t0"] - origin, 2), "burst_end": round(trig["burst_end"] - origin, 2),
                "delta_db": round(trig["delta"], 1), "reason": trig["reason"],
                "score": trig.get("score", 0), "signals": trig.get("signals", {}), "platform": self.platform}})
            self.clips += 1
            self.submit(jid)
            self.note(f"clip {n} queued as job {jid}")
        except Exception as e:
            self.note(f"cut failed: {e}")
            log.exception("Live cut failed")


class LiveManager:
    def __init__(self, cfg, db, submit):
        self.cfg, self.db, self.submit, self.monitors, self.next_id = cfg, db, submit, {}, 1

    def start(self, source, chat_file=None, mode=None, language=None, style=None) -> LiveMonitor:
        m = LiveMonitor(self.cfg, self.db, self.submit, source, self.next_id, chat_file or None, mode, language, style)
        self.monitors[self.next_id] = m
        self.next_id += 1
        m.start()
        return m

    def stop(self, monitor_id: int) -> None:
        self.monitors[monitor_id].stop()

    def list(self) -> list[dict]:
        return [m.view() for m in self.monitors.values()]

    def kick_event(self, username: str, text: str, message_id: str = "", chatter: str = "", slug: str = "") -> int:
        accepted = 0
        for m in self.monitors.values():
            names = {str(username).casefold(), str(slug).casefold()}
            if m.platform == "kick" and m.status == "recording" and m.channel and m.channel.casefold() in names:
                m.chat.add(text, chatter or username, message_id)
                if m.chat.provider_status != "connected": m.chat.provider_status = "connected"
                accepted += 1
        return accepted
