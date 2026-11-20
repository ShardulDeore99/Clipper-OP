"""Speech-context and weighted event scoring checks."""
import unittest

from clip_engine.ai.transcript_analyzer import analyze_context, transcript_candidates
from clip_engine.intelligence.highlight_scorer import score_highlight


class TranscriptAnalyzerTests(unittest.TestCase):
    def test_laughter_and_surprise_are_reported_as_possible_humor_cues(self):
        words = [{"w": w, "start": i * 0.3, "end": i * 0.3 + 0.2}
                 for i, w in enumerate("I thought we won but no way hahaha what just happened!".split())]
        result = analyze_context(words, 1.0)
        self.assertGreaterEqual(result["humor_score"], 35)
        self.assertIn("possible humor cues", result["labels"])
        self.assertIn("laughter in transcript", result["labels"])

    def test_speech_candidate_generation_is_bounded_per_minute(self):
        words = [{"w": "hahaha", "start": i * 2.0, "end": i * 2.0 + 0.5} for i in range(8)]
        sentences = [{"start": w["start"], "end": w["end"], "text": w["w"]} for w in words]
        candidates = transcript_candidates(sentences, words)
        self.assertLessEqual(len(candidates), 2)

    def test_missing_signals_renormalize_weights(self):
        result = score_highlight({"audio": 80, "speech": 60, "chat": None, "visual": None})
        self.assertEqual(result["available_signals"], ["audio", "speech"])
        self.assertAlmostEqual(sum(result["normalized_weights"].values()), 1.0)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)


if __name__ == "__main__":
    unittest.main()
