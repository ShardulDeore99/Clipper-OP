"""Integration: generate a sample video, run M1 stages, check audio + caching + loud failures."""
import tempfile
import time
import unittest
import wave
from pathlib import Path

from unittest import mock

from streamclipper import pipeline
from streamclipper.db import Database
from streamclipper.errors import StageError
from streamclipper.pipeline import run_job
from streamclipper.stages import audio, download
from streamclipper.startup import run_checks, require_ok
from tests.helpers import make_sample, temp_cfg


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)
        self.db = Database(self.cfg.db_path)
        patcher = mock.patch.object(pipeline, "STAGES", [download, audio])  # M1 stages only
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _job(self, path):
        return self.db.create_job(str(path), "file", "fast")

    def test_startup_checks_required_pass(self):
        require_ok(run_checks(self.cfg))

    def test_full_run_then_cached_rerun(self):
        video = make_sample(self.root / "sample.mp4", 6)
        jid = self._job(video)
        t1 = run_job(self.cfg, self.db, jid)
        job = self.db.get_job(jid)
        self.assertEqual(job["status"], "done")
        self.assertEqual((job["meta"]["width"], job["meta"]["height"]), (1280, 720))
        self.assertAlmostEqual(job["meta"]["duration"], 6, delta=0.5)
        self.assertEqual([r[0] for r in t1.rows], ["identify", "download", "audio"])

        key = job["source_key"]
        wav = self.cfg.data_dir / "cache" / key / "audio" / "audio.wav"
        with wave.open(str(wav)) as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))
            self.assertAlmostEqual(w.getnframes() / 16000, 6, delta=0.5)

        t2 = run_job(self.cfg, self.db, jid)  # second run: every stage from cache
        self.assertTrue(all(c for n, _, c in t2.rows if n != "identify"))
        print("\nRun 1:\n" + t1.table() + "\nRun 2 (cached):\n" + t2.table())

    def test_no_audio_fails_loudly(self):
        video = make_sample(self.root / "silent.mp4", 3, audio=False)
        jid = self._job(video)
        with self.assertRaises(StageError) as cm:
            run_job(self.cfg, self.db, jid)
        self.assertEqual(cm.exception.stage, "audio")
        job = self.db.get_job(jid)
        self.assertEqual(job["status"], "failed")
        self.assertIn("no audio", job["error"])

    def test_corrupt_file_fails_loudly(self):
        bad = self.root / "bad.mp4"
        bad.write_bytes(b"not a video" * 100)
        jid = self._job(bad)
        with self.assertRaises(StageError):
            run_job(self.cfg, self.db, jid)
        self.assertEqual(self.db.get_job(jid)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
