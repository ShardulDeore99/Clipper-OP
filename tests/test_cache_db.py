import tempfile
import unittest
from pathlib import Path

from streamclipper import cache
from streamclipper.db import Database
from streamclipper.stages.download import classify, source_key
from streamclipper.errors import StageError
from tests.helpers import temp_cfg


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hit_miss_and_missing_file(self):
        f = self.root / "out.wav"
        f.write_bytes(b"x" * 10)
        cache.store(self.cfg, "k", "audio", {"a": 1}, {"wav": str(f)}, 0.5)
        self.assertIsNotNone(cache.lookup(self.cfg, "k", "audio", {"a": 1}, ("wav",)))
        self.assertIsNone(cache.lookup(self.cfg, "k", "audio", {"a": 2}, ("wav",)))  # inputs changed
        f.unlink()
        self.assertIsNone(cache.lookup(self.cfg, "k", "audio", {"a": 1}, ("wav",)))  # file gone

    def test_file_key_is_content_based(self):
        a, b = self.root / "a.bin", self.root / "b.bin"
        a.write_bytes(b"same"); b.write_bytes(b"same")
        self.assertEqual(source_key("file", str(a)), source_key("file", str(b)))

    def test_classify(self):
        self.assertEqual(classify(' "https://x.test/v" ')[0], "url")
        for pasted, expected in [("youtube.com/watch?v=abc123", "https://youtube.com/watch?v=abc123"),
                                 ("www.youtube.com/watch?v=abc", "https://www.youtube.com/watch?v=abc"),
                                 ("youtu.be/abc123", "https://youtu.be/abc123"),
                                 ("<https://youtu.be/abc>", "https://youtu.be/abc"),
                                 ("  twitch.tv/videos/123 ", "https://twitch.tv/videos/123")]:
            self.assertEqual(classify(pasted), ("url", expected), pasted)
        for missing in ("C:/nope/missing.mp4", "clip.mp4", "videos/stream.mkv", "just some words"):
            with self.assertRaises(StageError, msg=missing):
                classify(missing)
        with self.assertRaises(StageError):
            classify("   ")

    def test_db_job_lifecycle(self):
        db = Database(self.cfg.db_path)
        jid = db.create_job("src", "file", "fast")
        db.update_job(jid, status="running", meta={"duration": 3})
        db.update_job(jid, meta={"width": 640})
        job = db.get_job(jid)
        self.assertEqual(job["meta"], {"duration": 3, "width": 640})
        self.assertEqual(db.pending_job_ids(), [jid])
        with self.assertRaises(ValueError):
            db.update_job(jid, evil="x")
        db.delete_job(jid)
        with self.assertRaises(LookupError):
            db.get_job(jid)


if __name__ == "__main__":
    unittest.main()
