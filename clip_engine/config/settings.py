"""Settings for the first, CPU-only audio detection phase."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioDetectorSettings:
    frame_seconds: float = 0.5
    baseline_seconds: float = 8.0
    silence_margin_db: float = 6.0
    event_score_threshold: float = 58.0
    event_merge_gap_seconds: float = 1.5
    voice_band_low_hz: float = 250.0
    voice_band_high_hz: float = 4000.0


DEFAULT_AUDIO_SETTINGS = AudioDetectorSettings()
