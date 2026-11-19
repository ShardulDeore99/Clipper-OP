# StreamClipper

A local tool that turns long videos, game streams and podcasts into vertical (9:16) clips of the best
moments, ready to review. Nothing is posted. Everything stays on this computer except the source download
(and the one-time model/font downloads in setup).

> You are responsible for having the rights or permission to clip any source you add.

## Run (Windows, PowerShell)
```
.\run.ps1      # first run sets everything up (Python venv, packages, Whisper models, caption font), then opens the UI
```
UI: http://127.0.0.1:8765. Paste a link or a file path, pick language and speed, press **Add source**.
When it finishes, clips appear under **My clips**: preview, trim, edit captions, change layout, download, delete.

## What it does
| Step | What happens |
|---|---|
| download, audio | link or file in, 16 kHz audio out (cached by source) |
| transcribe | faster-whisper, once per source. English stays English; Hinglish is written like `mai kyu aau nahi tu ja` |
| segment, score, plan | sentence/pause boundaries, cheap signals (loudness, laughter/shouting, keywords, chat spikes), then clips that start 2-4 s before the moment and end on a sentence end or pause. Min 12 s, target 20-35 s, max 60 s. Every boundary has a recorded reason |
| frame | face tracking on small frames, smoothed crop, layout choice (below) |
| render | captions (.ass, active word highlighted) + one encode per clip, loudness about -14 LUFS |
| check, save | flags with reasons; title, description, hashtags, cover frame |

Layouts: **gaming** (facecam 16:9 on top, game 16:9 below, blurred background), **normal** (16:9 centred with blur
above and below), single speaker (face-centred with headroom), two speakers (stacked), and a rounded facecam corner style.
Change the default in `config.toml` or per clip in the UI.

Checks that flag a clip (reason shown): starts/ends mid-sentence, over 1 s of silence at either end, a face cut off,
captions missing while speech exists, output not 1080x1920, loudness far from -14 LUFS.

Live monitor (UI panel or `streamclipper live <link>`): keeps a rolling recording, scores audio and chat activity,
and sends triggered clips through the same steps. It recognizes Twitch, YouTube, and Kick links automatically.
For YouTube chat, open **Profile** in the app and optionally save a YouTube API key. It is stored in the operating
system credential store and is never returned to the browser; the app also accepts `YOUTUBE_API_KEY` as an
environment-variable fallback. Without either, YouTube chat is simply unavailable while audio monitoring continues.
For Twitch, set `TWITCH_CLIENT_ID` and `TWITCH_USER_ACCESS_TOKEN` with the `user:read:chat` scope. An optional
timestamp log (seconds since monitor start, one timestamp per line) remains available as a fallback.

Kick chat uses Kick's signed `chat.message.sent` webhook. Set `KICK_ACCESS_TOKEN` with the `events:subscribe` scope
and `KICK_BROADCASTER_USER_ID`; the app creates the event subscription. In Kick Developer settings, set the webhook
URL to `https://YOUR-PUBLIC-HOST/api/webhooks/kick` and expose the local app through a public HTTPS tunnel. Kick
cannot deliver webhooks directly to localhost. Webhook signatures are checked against Kick's current public key.

On Windows, add credentials to the current PowerShell session before starting the app:

```powershell
$env:YOUTUBE_API_KEY = "your-youtube-api-key"
$env:TWITCH_CLIENT_ID = "your-twitch-client-id"
$env:TWITCH_USER_ACCESS_TOKEN = "your-user-token"
$env:KICK_ACCESS_TOKEN = "your-kick-token"
$env:KICK_BROADCASTER_USER_ID = "your-kick-broadcaster-id"
```

The live score combines audio intensity with message volume, unique chatters, laughter terms, and clip cues. It is
a transparent candidate score, not guaranteed joke recognition. The timeline shows the changing score, incoming chat,
and detected triggers. Messages are timestamped on receipt, so source stream delay can shift chat triggers slightly.

## FragBytez clip engine (Phases 1–3)
The `clip_engine` package adds audio activity and event timestamps, Whisper transcript-context cues for possible
humor/surprise/frustration, weighted scoring across available signals, and OpenCV scene-change/motion/face-presence
evidence. These are candidate signals, not guaranteed humor or emotion recognition. See [clip_engine/README.md](clip_engine/README.md).

## Commands
`doctor`, `process <link-or-file>`, `add`, `run [id]`, `list`, `serve`, `export [clip-id]`, `live <link>`,
`fetch-models`, `update-ytdlp [--nightly]`. Add `--debug` to save per clip: a contact sheet of 8 frames, the chosen
start/end with reasons, the layout, and the .ass/.srt (in `data/debug/`).
Run them as `.\.venv\Scripts\python -m streamclipper <command>`.

## Settings
Everything is in `config.toml` with comments. `[clips] max_per_source = 0` means unlimited clips; `[moments] min_score`
controls how picky the planner is. `glossary.txt` fixes names, game terms and slang in captions (`heard => written`).
Captions use Anton. `setup.ps1` tries to download it into `assets/fonts`; otherwise Impact or Arial Black is used.

## Troubleshooting
- Run `doctor` first. It checks FFmpeg, libass, a real caption render, QSV, yt-dlp and the face detector.
- YouTube `HTTP Error 403`: `update-ytdlp`, restart, install a JavaScript runtime (`winget install DenoLand.Deno`),
  or set `cookies_from_browser = "firefox"` in `config.toml`.
- Hinglish detected as English: choose **Hinglish** for that source.

## Tests
`python -m unittest discover -s tests -t . -v` (several minutes: they render real video). The web API test needs
`pip install -r requirements-dev.txt`.

## Screens
Three tabs: **Video or podcast** (paste a link or file), **Live stream** (stream link + optional chat log), **My clips**.
After you press Continue, a window asks for the *clip look* (layout, gaming style, caption position/look/size) with a live
preview. The same window is used for videos and live streams. Tick "Remember this look" to keep it for next time.
