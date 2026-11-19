"""Phase 1: synthetic audio detector checks, no external models required."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from clip_engine.detectors.audio_detector import analyze_wav


def write_tone(path: Path, seconds: float, burst: tuple[float, float] | None = None) -> None:
    rate = 16000
    t = np.arange(round(rate * seconds)) / rate
    signal = np.zeros_like(t)
    if burst:
        a, b = round(burst[0] * rate), round(burst[1] * rate)
        signal[a:b] = 0.7 * np.sin(2 * np.pi * 600 * t[a:b])
    data = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(data.tobytes())


class AudioDetectorTests(unittest.TestCase):
    def test_detects_high_energy_event_with_timestamps_and_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            write_tone(path, 8, (4, 5.5))
            result = analyze_wav(path)
        self.assertEqual(result["signal_available"], True)
        self.assertTrue(0 <= result["audio_score"] <= 100)
        self.assertTrue(result["events"])
        event = max(result["events"], key=lambda x: x["peak_score"])
        self.assertTrue(3.5 <= event["start"] <= 4.5, event)
        self.assertTrue(event["reasons"])

    def test_silence_is_reported_and_does_not_become_an_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silence.wav"
            write_tone(path, 4)
            result = analyze_wav(path)
        self.assertEqual(result["events"], [])
        self.assertTrue(result["silences"])

    def test_empty_wav_returns_an_available_empty_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.wav"
            write_tone(path, 0)
            result = analyze_wav(path)
        self.assertEqual(result["duration"], 0.0)
        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main()
