"""Cheap audio features from the 16 kHz WAV. Read in chunks so a 3-hour stream stays small."""
import wave

import numpy as np

from .errors import StageError

FRAME = 0.02   # seconds per energy frame
HOP = 0.1      # seconds per stored value (max of 5 frames)


def energy_db(wav_path, chunk_seconds: int = 60) -> np.ndarray:
    """dBFS per 20 ms frame."""
    out = []
    with wave.open(str(wav_path)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise StageError("segment", "Expected 16-bit mono WAV.", "Re-run the audio stage.")
        n = int(w.getframerate() * FRAME)
        per_chunk = n * int(chunk_seconds / FRAME)
        while True:
            raw = w.readframes(per_chunk)
            frames = len(raw) // 2 // n
            if frames == 0:
                break
            x = np.frombuffer(raw, dtype=np.int16, count=frames * n).astype(np.float32) / 32768.0
            rms = np.sqrt((x.reshape(frames, n) ** 2).mean(axis=1))
            out.append(20 * np.log10(rms + 1e-5))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def pool(db: np.ndarray, factor: int = 5) -> np.ndarray:
    pad = (-len(db)) % factor
    if pad:
        db = np.concatenate([db, np.full(pad, -100.0, dtype=db.dtype)])
    return db.reshape(-1, factor).max(axis=1) if len(db) else db


def silence_threshold(db: np.ndarray) -> tuple[float, float]:
    """Adaptive: just above the noise floor. Returns (threshold, floor)."""
    if len(db) == 0:
        return -60.0, -100.0
    floor, high = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    return floor + 0.2 * (high - floor), floor


def silences(db: np.ndarray, thr: float, frame: float = FRAME, min_len: float = 0.3) -> list[list[float]]:
    quiet = np.concatenate([[False], db <= thr, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    runs = []
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) * frame >= min_len:
            runs.append([round(a * frame, 2), round(b * frame, 2)])
    return runs


def pseudo_words(db: np.ndarray, thr: float, frame: float = FRAME) -> list[dict]:
    """No transcript (gameplay with no talking): treat each non-silent run as a 'word' so boundaries still exist."""
    loud = np.concatenate([[False], db > thr, [False]])
    edges = np.flatnonzero(np.diff(loud.astype(np.int8)))
    words = []
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) * frame >= 0.15:
            words.append({"w": "", "start": round(a * frame, 3), "end": round(b * frame, 3)})
    return words
