import unittest

from clip_engine.ai.transcript_matcher import find_matches, has_target_match, parse_targets


class TranscriptMatchTests(unittest.TestCase):
    def test_matches_exact_spoken_line_and_returns_timestamps(self):
        words = [{"w": token, "start": i, "end": i + 0.5}
                 for i, token in enumerate("we made it through the first round but then the boss fell off the map".split())]
        matches = find_matches(words, "the boss fell off the map")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["start"], 9)
        self.assertEqual(matches[0]["end"], 14.5)
        self.assertTrue(has_target_match("the boss fell off the map and we won", "the boss fell off the map"))

    def test_unmatched_line_does_not_create_a_clip_candidate(self):
        words = [{"w": token, "start": i, "end": i + 0.5} for i, token in enumerate("game starts now lets play".split())]
        self.assertEqual(find_matches(words, "the dragon wins the final tournament"), [])
        self.assertFalse(has_target_match("game starts now lets play", "the dragon wins the final tournament"))

    def test_short_or_blank_targets_are_ignored(self):
        self.assertEqual(parse_targets("lol\nthis line has enough spoken words"), ["this line has enough spoken words"])


if __name__ == "__main__":
    unittest.main()
