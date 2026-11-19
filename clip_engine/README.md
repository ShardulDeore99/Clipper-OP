# FragBytez Clip Engine: Phases 1–3

Phase 1 adds CPU audio analysis for authorized local recordings. Phase 2 adds
Whisper transcript-context cues for possible humor, surprise and frustration,
plus weighted scoring that renormalizes across available signals. Phase 3 adds
candidate-centered OpenCV evidence for frame changes, motion and face presence.
These signals enrich the existing StreamClipper pipeline and its analysis
charts.

Transcript analysis uses inspectable text cues. It does not understand every
joke or use an LLM. Visual analysis reports activity and face presence; it does
not identify facial emotion or decide whether an event is funny. Candidates
still need human review.

```powershell
python -m clip_engine "C:\Videos\authorized-gameplay.mp4" --compact
python -m clip_engine "C:\Videos\audio.wav"
```

NumPy is already a project dependency. Video input requires FFmpeg on `PATH` or
an explicit `--ffmpeg` path. The source recording is read only; extracted WAV
data is temporary. The analysis is also included in the existing Video job
analysis chart and live monitor telemetry.

## Still to build

Platform chat ingestion, a trained semantic humor/emotion model, stronger
gameplay event recognition, feedback storage, and platform adapters remain
future phases. This module does not connect to Twitch, YouTube, or Kick and
does not fetch streams.
