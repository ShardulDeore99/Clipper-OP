"""M5: caption generation, time shift, glossary, placement, Hinglish rendering, and the render self-test."""
import re
import tempfile
import unittest
from pathlib import Path

from streamclipper import captions as C
from streamclipper.errors import StageError
from streamclipper.fonts import find_font, stage_font
from streamclipper.overlays import max_stickers, plan_stickers
from tests.helpers import temp_cfg


def W(text, start, dur=0.3):
    return {"w": text, "start": start, "end": round(start + dur, 3)}


def seq(text, t0=0.0, step=0.4):
    return [W(w, round(t0 + i * step, 3)) for i, w in enumerate(text.split())]


class CaptionTextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = temp_cfg(Path(self.tmp.name))
        self.font = find_font(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_words_are_shifted_to_clip_time(self):
        words = seq("aaa bbb ccc ddd", t0=100.0)
        out = C.clip_words(words, 100.0 - 0.15, 102.0)
        self.assertAlmostEqual(out[0]["start"], 0.15, places=2)
        self.assertTrue(all(w["start"] >= 0 for w in out))

    def test_glossary_fixes_names_and_phrases(self):
        rules = [(["steem"], ["Steam"]), (["half", "life"], ["Half-Life"])]
        out = C.apply_glossary(seq("play steem and half life now"), rules)
        self.assertEqual([w["w"] for w in out], ["play", "Steam", "and", "Half-Life", "now"])

    def test_glossary_file_parsing(self):
        f = Path(self.tmp.name) / "g.txt"
        f.write_text("# comment\nsteem => Steam\n\nhalf life => Half-Life\n")
        rules = C.load_glossary(f)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0][0], ["half", "life"])  # longest phrase first

    def test_groups_have_2_to_3_words_and_respect_sentences(self):
        words = seq("one two three four five six seven eight nine ten")
        words[3]["w"] = "four."
        groups = C.build_groups(words, (2, 3))
        self.assertTrue(all(1 <= len(g) <= 3 for g in groups))
        self.assertTrue(all(len(g) >= 2 for g in groups[:-1]))
        for g in groups:
            self.assertTrue(g[-1]["end"] - g[0]["start"] <= 1.8)
        flat = [w["w"] for g in groups for w in g]
        self.assertEqual(len(flat), 10)

    def test_ass_is_uppercase_highlighted_and_in_the_lower_middle(self):
        ass, stats = C.make_ass(self.cfg, self.font, seq("mai kyu aau nahi tu ja"))
        self.assertIn("MAI", ass)
        self.assertNotIn("mai", ass)
        self.assertIn("&H00FF00&", ass)  # lime highlight
        ys = [int(y) for y in re.findall(r"\\pos\(540,(\d+)\)", ass)]
        self.assertTrue(ys and all(1000 <= y <= 1500 for y in ys))
        self.assertEqual(stats["words"], 6)
        self.assertIn("Style: Cap,", ass)
        self.assertIn("PlayResY: 1920", ass)

    def test_captions_move_off_a_face(self):
        face_low = lambda t: [(300, 1150, 780, 1350)]  # a face sitting in the default caption position
        ass, _ = C.make_ass(self.cfg, self.font, seq("hello there my friend"), faces_fn=face_low)
        ys = {int(y) for y in re.findall(r"\\pos\(540,(\d+)\)", ass)}
        self.assertTrue(all(abs(y - 1250) > 100 for y in ys), ys)

    def test_devanagari_is_romanized_never_drawn_as_boxes(self):
        ass, stats = C.make_ass(self.cfg, self.font, [W("नहीं", 0.0), W("यार", 0.4)])
        self.assertIn("NAHI", ass)
        self.assertNotRegex(ass, r"[\u0900-\u097f]")

    def test_edited_captions_keep_timing(self):
        words = seq("a b c")
        self.assertEqual([w["w"] for w in C.apply_caption_edit(words, "x y z")], ["x", "y", "z"])
        spread = C.apply_caption_edit(words, "one two three four")
        self.assertEqual(len(spread), 4)
        self.assertEqual(spread[0]["start"], words[0]["start"])


class RenderSelfTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)
        self.font = find_font(self.cfg)
        stage_font(self.font, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hinglish_captions_really_render(self):
        words = seq("mai kyu aau nahi tu ja")
        ass, _ = C.make_ass(self.cfg, self.font, words)
        (self.root / "subs.ass").write_text(ass, encoding="utf-8")
        self.assertGreater(C.verify_render("ffmpeg", self.root, C.probe_time(words)), 300)

    def test_empty_captions_raise_instead_of_passing_silently(self):
        ass, _ = C.make_ass(self.cfg, self.font, [])
        (self.root / "subs.ass").write_text(ass, encoding="utf-8")
        with self.assertRaises(StageError):
            C.verify_render("ffmpeg", self.root, 1.0)

    def test_broken_font_or_file_raises(self):
        with self.assertRaises(StageError):
            C.verify_render("ffmpeg", self.root, 1.0)  # no subs.ass present


class StickerTests(unittest.TestCase):
    def test_caps_by_length(self):
        self.assertEqual((max_stickers(15), max_stickers(30), max_stickers(60)), (2, 4, 6))

    def test_spacing_cap_and_clear_ending(self):
        words = [W(w, 1.0 + i * 2.2) for i, w in enumerate(["omg", "wow", "insane", "gg", "wtf", "crazy", "clutch", "omg"])]
        s = plan_stickers(words, 14.0)
        self.assertLessEqual(len(s), 2)
        self.assertTrue(all(b["t0"] - a["t1"] >= 0.0 for a, b in zip(s, s[1:])))
        self.assertTrue(all(x["t1"] <= 14.0 - 1.5 for x in s))

    def test_not_over_a_face(self):
        face = lambda t: [(0, 0, 1080, 1000)]  # face covers every sticker spot
        self.assertEqual(plan_stickers([W("omg", 1.0)], 30.0, faces_fn=face), [])


if __name__ == "__main__":
    unittest.main()
