"""CPU audio event analysis for authorized WAV recordings.

This phase uses measurable acoustic features only. The voice-band ratio is a
rough activity hint, not speech recognition or a claim that a sound is human.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..config.settings import AudioDetectorSettings, DEFAULT_AUDIO_SETTINGS


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def analyze_chunk(samples: np.ndarray, sample_rate: int, baseline_dbfs: float, noise_floor_dbfs: float) -> dict:
    """Score one live mono float chunk (normalized to [-1, 1])."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(x) or sample_rate <= 0:
        return {"audio_score": 0.0, "rms_dbfs": -120.0, "peak_dbfs": -120.0,
                "delta_db": 0.0, "voice_band_activity": 0.0, "silence": True}
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    dbfs = float(20 * np.log10(rms + 1e-6))
    peak_dbfs = float(20 * np.log10(peak + 1e-6))
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sample_rate)
    power = spectrum * spectrum
    voice_band = float(power[(freqs >= 250) & (freqs <= min(4000, sample_rate / 2))].sum() / (power.sum() + 1e-9))
    delta = max(0.0, dbfs - baseline_dbfs)
    relative = _clamp((dbfs - noise_floor_dbfs) / max(30.0, abs(noise_floor_dbfs) * 0.5) * 100)
    score = _clamp(0.50 * relative + 0.35 * _clamp(delta / 18 * 100) +
                   0.10 * _clamp((peak_dbfs + 36) / 30 * 100) + 0.05 * _clamp(voice_band * 125))
    return {"audio_score": round(score, 1), "rms_dbfs": round(dbfs, 1),
            "peak_dbfs": round(peak_dbfs, 1), "delta_db": round(delta, 1),
            "voice_band_activity": round(voice_band, 3), "silence": dbfs < -45.0}


def _merge_events(points: list[dict], settings: AudioDetectorSettings) -> list[dict]:
    active = [p for p in points if p["score"] >= settings.event_score_threshold and not p["silence"]]
    groups: list[list[dict]] = []
    for point in active:
        if groups and point["start"] - groups[-1][-1]["end"] <= settings.event_merge_gap_seconds:
            groups[-1].append(point)
        else:
            groups.append([point])
    return [{"start": round(g[0]["start"], 2), "end": round(g[-1]["end"], 2),
             "peak_score": round(max(p["score"] for p in g), 1),
             "peak_dbfs": round(max(p["peak_dbfs"] for p in g), 1),
             "reasons": sorted({reason for p in g for reason in p["reasons"]})}
            for g in groups]


def analyze_wav(path: str | Path, settings: AudioDetectorSettings = DEFAULT_AUDIO_SETTINGS) -> dict:
    """Analyze a PCM WAV and return normalized audio activity and event timestamps.

    Input may be mono or multichannel signed 16-bit PCM. Processing is chunked,
    so memory use stays bounded for long recordings.
    """
    points: list[dict] = []
    with wave.open(str(path), "rb") as audio:
        width, channels, sample_rate = audio.getsampwidth(), audio.getnchannels(), audio.getframerate()
        if width != 2 or channels < 1 or sample_rate <= 0:
            raise ValueError("Audio detector requires 16-bit PCM WAV audio.")
        frame_samples = max(1, round(sample_rate * settings.frame_seconds))
        chunk_size = frame_samples * 20
        frame_index = 0
        while raw := audio.readframes(chunk_size):
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768.0
            mono = samples.mean(axis=1)
            for start in range(0, len(mono), frame_samples):
                x = mono[start:start + frame_samples]
                if not len(x):
                    continue
                rms = float(np.sqrt(np.mean(x * x)))
                peak = float(np.max(np.abs(x)))
                dbfs = float(20 * np.log10(rms + 1e-6))
                peak_dbfs = float(20 * np.log10(peak + 1e-6))
                spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
                freqs = np.fft.rfftfreq(len(x), 1 / sample_rate)
                power = spectrum * spectrum
                total = float(power.sum()) + 1e-9
                voice_band = float(power[(freqs >= settings.voice_band_low_hz) &
                                          (freqs <= min(settings.voice_band_high_hz, sample_rate / 2))].sum() / total)
                start_t = frame_index * settings.frame_seconds
                points.append({"start": start_t, "end": start_t + len(x) / sample_rate,
                               "rms_dbfs": dbfs, "peak_dbfs": peak_dbfs,
                               "voice_band_activity": voice_band})
                frame_index += 1

    if not points:
        return {"duration": 0.0, "audio_score": 0.0, "features": [], "events": [],
                "silences": [], "signal_available": True}

    levels = np.array([p["rms_dbfs"] for p in points], dtype=np.float32)
    floor = float(np.percentile(levels, 15))
    high = float(np.percentile(levels, 95))
    silence_threshold = min(-30.0, floor + settings.silence_margin_db)
    baseline_frames = max(2, round(settings.baseline_seconds / settings.frame_seconds))
    excitement = []
    for i, point in enumerate(points):
        lo = max(0, i - baseline_frames)
        baseline = float(np.median(levels[lo:i])) if i > lo else floor
        delta = max(0.0, point["rms_dbfs"] - baseline)
        relative_level = _clamp((point["rms_dbfs"] - floor) / max(high - floor, 1.0) * 100)
        sudden_change = _clamp(delta / 18 * 100)
        peak_intensity = _clamp((point["peak_dbfs"] + 36) / 30 * 100)
        voice_activity = _clamp(point["voice_band_activity"] * 125)
        score = _clamp(0.40 * relative_level + 0.35 * sudden_change + 0.15 * peak_intensity + 0.10 * voice_activity)
        reasons = []
        if delta >= 6.0:
            reasons.append("sudden energy increase")
        if point["peak_dbfs"] >= -6.0:
            reasons.append("high audio peak")
        if point["voice_band_activity"] >= 0.7:
            reasons.append("strong voice-band activity (heuristic)")
        point.update({"score": round(score, 1), "baseline_dbfs": round(baseline, 1),
                      "delta_db": round(delta, 1), "silence": point["rms_dbfs"] <= silence_threshold,
                      "reasons": reasons})
        excitement.append(score)

    silences = []
    start = None
    for point in points:
        if point["silence"] and start is None:
            start = point["start"]
        elif not point["silence"] and start is not None:
            silences.append({"start": round(start, 2), "end": round(point["start"], 2)})
            start = None
    if start is not None:
        silences.append({"start": round(start, 2), "end": round(points[-1]["end"], 2)})

    return {"duration": round(points[-1]["end"], 3),
            "audio_score": round(float(np.mean(excitement)), 1),
            "signal_available": True,
            "summary": {"noise_floor_dbfs": round(floor, 1), "high_level_dbfs": round(high, 1),
                        "silence_threshold_dbfs": round(silence_threshold, 1)},
            "features": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}
                         for p in points],
            "events": _merge_events(points, settings), "silences": silences}
