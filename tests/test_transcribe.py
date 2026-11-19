"""Transcribe stage with a fake Whisper model: language routing, romanization, SRT, caching, errors."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from streamclipper import pipeline
from streamclipper.db import Database
from streamclipper.stages import audio, download
from streamclipper.errors import StageError
from streamclipper.pipeline import run_job
from streamclipper.stages import transcribe
from streamclipper.transcript import srt_time, to_srt
from tests.helpers import make_sample, temp_cfg


def W(word, start, end, p=0.9):
    return NS(word=word, start=start, end=end, probability=p)


HINDI = [NS(start=0.2, end=2.0, words=[W(" ये", 0.2, 0.5), W(" गेम", 0.5, 1.0), W(" बहुत", 1.0, 1.5), W(" मस्त", 1.5, 2.0)]),
         NS(start=2.5, end=4.0, words=[W(" bro", 2.5, 3.0), W(" चलो!", 3.0, 4.0)])]
ENGLISH = [NS(start=0.1, end=1.5, words=[W(" Hello", 0.1, 0.6), W(" everyone", 0.6, 1.5)])]


class FakeModel:
    def __init__(self, segments, detected="hi"):
        self.segments, self.detected, self.calls, self.closed = segments, detected, [], 0

    def transcribe(self, wav, **kw):
        self.calls.append(kw)
        outer = self

        def gen():
            try:
                yield from outer.segments
            finally:
                outer.closed += 1
        lang = kw.get("language") or self.detected
        return gen(), NS(language=lang, language_probability=0.9)


MIXED = [NS(start=0.0, end=2.0, words=[W(" Okay", 0.0, 0.4), W(" guys", 0.4, 0.8), W(" welcome", 0.8, 1.4), W(" back", 1.4, 2.0)]),
         NS(start=2.5, end=5.0, words=[W(" मैं", 2.5, 3.0), W(" क्यों", 3.0, 3.5), W(" आऊँ", 3.5, 4.0), W(" नहीं", 4.0, 4.5), W(" तू", 4.5, 4.8), W(" जा", 4.8, 5.0)]),
         NS(start=5.5, end=7.0, words=[W(" bro", 5.5, 6.0), W(" kyun", 6.0, 6.5), W(" nahin", 6.5, 7.0)])]


class MultilingualFake(FakeModel):
    """Like faster-whisper >= 1.1: has an explicit `multilingual` parameter."""

    def transcribe(self, wav, multilingual=False, **kw):
        return super().transcribe(wav, multilingual=multilingual, **kw)


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)
        self.db = Database(self.cfg.db_path)
        self.video = make_sample(self.root / "s.mp4", 5)
        patcher = mock.patch.object(pipeline, "STAGES", [download, audio, transcribe])  # M1+M2 stages only
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, model, language=None, mode="fast"):
        jid = self.db.create_job(str(self.video), "file", mode, language)
        with mock.patch.object(transcribe, "load_model", return_value=model):
            timer = run_job(self.cfg, self.db, jid)
        return jid, timer

    def _transcript(self, jid):
        import json
        key = self.db.get_job(jid)["source_key"]
        return json.loads((self.cfg.data_dir / "cache" / key / "transcribe" / "transcript.json").read_text("utf-8"))

    def test_hinglish_uses_hindi_with_roman_prompt_and_romanizes(self):
        m = FakeModel(HINDI)
        jid, _ = self._run(m, "hinglish")
        kw = m.calls[0]
        self.assertEqual((kw["language"], kw["word_timestamps"], kw["beam_size"], kw["vad_filter"]), ("hi", True, 1, True))
        self.assertIn("Aaj hum", kw["initial_prompt"])
        data = self._transcript(jid)
        self.assertEqual(data["segments"][0]["text"], "ye game bahut mast")
        self.assertEqual(data["segments"][1]["text"], "bro chalo!")
        self.assertEqual(data["language"], "hinglish")
        job = self.db.get_job(jid)
        self.assertEqual((job["status"], job["meta"]["words"], job["meta"]["language"]), ("done", 6, "hinglish"))

    def test_english_stays_english(self):
        m = FakeModel(ENGLISH, "en")
        jid, _ = self._run(m, "en")
        self.assertEqual(m.calls[0]["language"], "en")
        self.assertNotIn("initial_prompt", m.calls[0])
        self.assertEqual(self._transcript(jid)["segments"][0]["text"], "Hello everyone")

    def test_auto_hindi_detected_switches_to_hinglish_without_double_transcription(self):
        m = FakeModel(HINDI, "hi")
        jid, _ = self._run(m, "auto")
        self.assertEqual([c.get("language") for c in m.calls], [None, "hi"])
        self.assertEqual(m.closed, 1)  # detection pass was never iterated (no decoding); only pass 2 ran
        self.assertEqual(self._transcript(jid)["language"], "hinglish")

    def test_auto_switches_language_per_window_and_writes_casual_hinglish(self):
        m = MultilingualFake(MIXED, "en")
        jid, _ = self._run(m, "auto")
        self.assertEqual(len(m.calls), 1)  # one pass, language decided per window by the model
        self.assertIsNone(m.calls[0]["language"])
        self.assertTrue(m.calls[0]["multilingual"])
        data = self._transcript(jid)
        self.assertEqual([s["text"] for s in data["segments"]],
                         ["Okay guys welcome back", "mai kyu aau nahi tu ja", "bro kyu nahi"])
        self.assertEqual(data["language_mode"], "per-window")

    def test_explicit_english_leaves_text_alone(self):
        m = FakeModel([NS(start=0, end=1, words=[W(" nahin", 0, 0.5), W(" Main", 0.5, 1)])], "en")
        jid, _ = self._run(m, "en")
        self.assertEqual(self._transcript(jid)["segments"][0]["text"], "nahin Main")

    def test_auto_english_single_pass(self):
        m = FakeModel(ENGLISH, "en")
        self._run(m, "auto")
        self.assertEqual(len(m.calls), 1)

    def test_unsupported_language_fails_loudly(self):
        jid = self.db.create_job(str(self.video), "file", "fast", "auto")
        with mock.patch.object(transcribe, "load_model", return_value=FakeModel(ENGLISH, "es")):
            with self.assertRaises(StageError) as cm:
                run_job(self.cfg, self.db, jid)
        self.assertIn("'es'", str(cm.exception))
        self.assertEqual(self.db.get_job(jid)["status"], "failed")

    def test_transcribes_once_then_cached(self):
        m = FakeModel(HINDI)
        jid = self.db.create_job(str(self.video), "file", "fast", "hinglish")
        with mock.patch.object(transcribe, "load_model", return_value=m):
            run_job(self.cfg, self.db, jid)
            t2 = run_job(self.cfg, self.db, jid)
        self.assertEqual(len(m.calls), 1)
        self.assertTrue(dict((n, c) for n, _, c in t2.rows)["transcribe"])

    def test_changing_language_retranscribes(self):
        m = FakeModel(HINDI)
        jid = self.db.create_job(str(self.video), "file", "fast", "hinglish")
        with mock.patch.object(transcribe, "load_model", return_value=m):
            run_job(self.cfg, self.db, jid)
            self.db.update_job(jid, language="en")
            run_job(self.cfg, self.db, jid)
        self.assertEqual(len(m.calls), 2)

    def test_no_speech_is_reported_not_failed(self):
        jid, _ = self._run(FakeModel([]), "en")
        job = self.db.get_job(jid)
        self.assertEqual(job["status"], "done")
        self.assertFalse(job["meta"]["speech"])

    def test_models_are_restricted(self):
        for bad in ("medium", "large-v3"):
            cfg = temp_cfg(self.root)
            cfg["transcribe"]["model_fast"] = bad
            ctx = NS(cfg=cfg, job={"mode": "fast", "language": None})
            with self.assertRaises(StageError):
                transcribe.settings(ctx)
        ctx = NS(cfg=temp_cfg(self.root), job={"mode": "quality", "language": None})
        self.assertEqual(transcribe.settings(ctx)[0], "small")

    def test_missing_model_gives_actionable_error(self):
        with self.assertRaises(StageError) as cm:
            transcribe.load_model(self.cfg, "base")  # faster-whisper absent here, or model absent on disk
        self.assertTrue("faster-whisper" in str(cm.exception) or "fetch-models" in str(cm.exception))

    def test_srt_format(self):
        self.assertEqual(srt_time(3661.5), "01:01:01,500")
        srt = to_srt([{"start": 0.2, "end": 2.0, "text": "ye game bahut mast"}])
        self.assertEqual(srt, "1\n00:00:00,200 --> 00:00:02,000\nye game bahut mast\n")


if __name__ == "__main__":
    unittest.main()
