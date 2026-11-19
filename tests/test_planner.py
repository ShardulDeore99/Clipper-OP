"""M3: boundary snapping, clip planning, signals. Pure data: no ffmpeg, no models."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from streamclipper import audio_features as af
from streamclipper.boundaries import (boundary_points, build_sentences, cut_end, cut_start, snap_end, snap_start)
from streamclipper.planner import plan_clips
from streamclipper.signals import find_candidates, moment_curve

P = {"min_s": 12, "max_s": 60, "target_min": 20, "target_max": 35, "lead": (2.0, 4.0), "tail": (0.7, 1.0),
     "min_score": 0.0, "max_per_source": 0, "hook_words": ["wait", "listen", "bro"]}


def talk(script, t0=0.0, wlen=0.3, gap=1.0):
    """script: list of sentences (str). Returns words; sentences are separated by `gap` seconds."""
    words, t = [], t0
    for sent in script:
        for w in sent.split():
            words.append({"w": w, "start": round(t, 3), "end": round(t + wlen, 3)})
            t += wlen + 0.05
        t += gap - 0.05
    return words


def prep(script, **kw):
    words = talk(script, **kw)
    sents = build_sentences(words, 0.6)
    starts, ends = boundary_points(words, sents, 0.35)
    return words, sents, starts, ends


SCRIPT = ["So today we try the new boss.", "It looks really hard honestly.", "Wait, listen to this part.",
          "He jumps and misses completely!", "Bro what was that?", "I cannot believe it happened.",
          "Okay let us try again now.", "This time I will be careful.", "Here we go again guys.",
          "And that is the end of that.", "Thanks for watching everyone today.", "See you in the next one."]


class SnapTests(unittest.TestCase):
    def test_sentences_split_on_punctuation_and_pause(self):
        words = talk(["One two three.", "Four five"], gap=0.2) + talk(["six seven"], t0=20)
        s = build_sentences(words, 0.6)
        self.assertEqual([x["text"] for x in s], ["One two three.", "Four five", "six seven"])

    def test_start_snaps_to_sentence_start_in_lead_window(self):
        words, sents, starts, _ = prep(SCRIPT)
        key = sents[4]["start"] + 1.0
        p, why = snap_start(starts, key)
        self.assertEqual(p["kind"], "sentence")
        self.assertIn(p["t"], [s["start"] for s in sents])
        self.assertTrue(2.0 <= key - p["t"] <= 4.0 or "nearest" in why, why)

    def test_sentence_start_preferred_over_pause(self):
        pts = [{"t": 10.0, "kind": "pause", "gap": 0.5, "i": 1}, {"t": 9.4, "kind": "sentence", "gap": 1.0, "i": 0}]
        p, _ = snap_start(pts, 13.0)
        self.assertEqual(p["kind"], "sentence")

    def test_widens_when_nothing_in_window(self):
        pts = [{"t": 5.0, "kind": "sentence", "gap": 1.0, "i": 0}]
        p, why = snap_start(pts, 10.0)
        self.assertEqual(p["t"], 5.0)
        self.assertIn("nearest", why)
        self.assertIsNone(snap_start([], 10.0)[0])

    def test_end_is_never_before_the_payoff_and_has_room(self):
        words, sents, _, ends = prep(SCRIPT)
        payoff = sents[3]["end"]
        p, _ = snap_end(ends, payoff)
        self.assertGreaterEqual(p["t"], payoff - 0.05)
        e = cut_end(words, p["i"])
        self.assertTrue(0.7 <= e - words[p["i"]]["end"] <= 1.0)

    def test_cut_never_runs_into_the_next_word(self):
        words = [{"w": "a", "start": 0.0, "end": 0.3}, {"w": "b", "start": 0.5, "end": 0.8}]
        self.assertLess(cut_end(words, 0), 0.5)
        self.assertLess(cut_start(words, 1), 0.5)
        self.assertGreaterEqual(cut_start(words, 1), 0.3)


class PlanTests(unittest.TestCase):
    def cand(self, t, strength=0.8):
        return {"t": t, "strength": strength, "signals": {"energy": 0.8}, "delta_db": 10, "burst_end": t}

    def test_clip_starts_and_ends_on_sentence_boundaries(self):
        words, sents, starts, ends = prep(SCRIPT)
        clips, rej = plan_clips([self.cand(sents[4]["start"] + 1)], words, sents, starts, ends, P, 100)
        self.assertEqual(len(clips), 1, rej)
        c = clips[0]
        self.assertIn(words[c["i0"]]["start"], [s["start"] for s in sents])
        self.assertIn(words[c["i1"]]["end"], [s["end"] for s in sents])
        self.assertTrue(12 <= c["duration"] <= 60)
        self.assertTrue(c["start"] <= c["key_moment"] <= c["end"])
        self.assertTrue(c["start_reason"] and c["end_reason"])

    def test_short_clip_is_extended_to_the_minimum(self):
        words, sents, starts, ends = prep(SCRIPT)
        clips, _ = plan_clips([self.cand(sents[1]["start"] + 1)], words, sents, starts, ends, P, 100)
        self.assertGreaterEqual(clips[0]["duration"], 12)

    def test_too_long_is_trimmed_or_rejected_never_over_max(self):
        words, sents, starts, ends = prep(SCRIPT)
        p = dict(P, max_s=14)
        clips, rej = plan_clips([self.cand(sents[5]["start"] + 1)], words, sents, starts, ends, p, 100)
        for c in clips:
            self.assertLessEqual(c["duration"], 14)

    def test_overlapping_moments_merge_into_one_clip(self):
        words, sents, starts, ends = prep(SCRIPT)
        clips, _ = plan_clips([self.cand(sents[3]["start"] + 1), self.cand(sents[4]["start"] + 1, 0.7)],
                              words, sents, starts, ends, P, 100)
        self.assertEqual(len(clips), 1)

    def test_near_duplicates_are_dropped(self):
        a = talk(SCRIPT[:6])
        b = talk(SCRIPT[:6], t0=a[-1]["end"] + 30)
        words = a + b
        sents = build_sentences(words, 0.6)
        starts, ends = boundary_points(words, sents, 0.35)
        clips, rej = plan_clips([self.cand(a[10]["start"]), self.cand(b[10]["start"], 0.7)], words, sents, starts, ends, P, 200)
        self.assertEqual(len(clips), 1)
        self.assertTrue(any("duplicate" in r["reason"] for r in rej))

    def test_min_score_and_cap(self):
        words, sents, starts, ends = prep(SCRIPT)
        c = [self.cand(sents[4]["start"] + 1)]
        self.assertEqual(plan_clips(c, words, sents, starts, ends, dict(P, min_score=0.99), 100)[0], [])
        far = talk(SCRIPT, t0=200)
        allw = words + far
        s2 = build_sentences(allw, 0.6)
        st2, en2 = boundary_points(allw, s2, 0.35)
        cs = [self.cand(sents[4]["start"] + 1), self.cand(far[20]["start"], 0.5)]
        self.assertEqual(len(plan_clips(cs, allw, s2, st2, en2, P, 400)[0]), 2)  # 0 = unlimited
        self.assertEqual(len(plan_clips(cs, allw, s2, st2, en2, dict(P, max_per_source=1), 400)[0]), 1)


class SignalTests(unittest.TestCase):
    def test_loud_burst_becomes_the_top_candidate(self):
        rng = np.random.default_rng(1)
        sr, secs = 16000, 40
        x = rng.normal(0, 0.005, sr * secs)
        t = np.arange(sr * 3) / sr
        x[sr * 25:sr * 28] += 0.6 * np.sin(2 * np.pi * 300 * t)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.wav"
            with wave.open(str(f), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
            db = af.energy_db(f)
        db01 = af.pool(db)
        parts, curve, delta = moment_curve(secs, db01, [], [], [])
        cands = find_candidates(parts, curve, db01, delta, secs, 3)
        top = max(cands, key=lambda c: c["strength"])
        self.assertTrue(25 <= top["t"] <= 28.5, top)
        self.assertGreater(top["delta_db"], 10)

    def test_keywords_and_laughter_words_raise_strength(self):
        words = talk(["that was insane bro", "hahaha no way"])
        sents = build_sentences(words, 0.6)
        parts, curve, _ = moment_curve(10, np.full(100, -50.0), words, sents, ["insane", "no way"])
        self.assertGreater(parts["keyword"].max(), 0)
        self.assertEqual(parts["laugh"].max(), 1.0)

    def test_silence_detection(self):
        db = np.concatenate([np.full(100, -60.0), np.full(100, -20.0), np.full(100, -60.0)])
        thr, _ = af.silence_threshold(db)
        s = af.silences(db, thr, min_len=0.3)
        self.assertEqual(len(s), 2)


if __name__ == "__main__":
    unittest.main()
