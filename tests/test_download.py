"""Download stage with a fake yt_dlp module: 403 retry, error cleanup, no pointless retries."""
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from streamclipper.errors import StageError
from streamclipper.stages import download
from streamclipper.startup import ytdlp_age_days
from tests.helpers import temp_cfg


class FakeDownloadError(Exception):
    pass


def fake_module(outcomes, seen_opts, out_file):
    """outcomes: one entry per attempt, either an error message or None for success."""
    mod = types.ModuleType("yt_dlp")
    mod.utils = NS(DownloadError=FakeDownloadError)

    class YDL:
        def __init__(self, opts):
            seen_opts.append(opts)
            self.n = len(seen_opts) - 1

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def extract_info(self, url, download=False):
            if outcomes[self.n]:
                raise FakeDownloadError(outcomes[self.n])
            return {"title": "Fake video"}

        def process_ie_result(self, info, download=True):
            out_file.write_bytes(b"x" * 10)
            return {"title": "Fake video", "requested_downloads": [{"filepath": str(out_file)}]}

    mod.YoutubeDL = YDL
    return mod


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)
        self.ctx = NS(cfg=self.cfg, key="k", job={"source": "https://youtu.be/x", "kind": "url"})

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, outcomes):
        seen = []
        mod = fake_module(outcomes, seen, self.root / "v.mp4")
        with mock.patch.dict(sys.modules, {"yt_dlp": mod}):
            return download._download(self.ctx, lambda f: None), seen

    def test_403_then_success_uses_second_attempt(self):
        out, seen = self._run(["\x1b[0;31mERROR:\x1b[0m unable to download video data: HTTP Error 403: Forbidden", None])
        self.assertEqual(out["title"], "Fake video")
        self.assertEqual(len(seen), 2)
        self.assertNotIn("source_address", seen[0])
        self.assertEqual(seen[1]["source_address"], "0.0.0.0")
        self.assertTrue(seen[0]["no_color"])

    def test_persistent_403_gives_clean_actionable_error(self):
        with self.assertRaises(StageError) as cm:
            self._run(["[0;31mERROR:[0m HTTP Error 403: Forbidden"] * 2)
        text = str(cm.exception)
        self.assertNotIn("[0;31m", text)
        self.assertIn("update-ytdlp", text)
        self.assertIn("cookies_from_browser", text)

    def test_other_errors_are_not_retried(self):
        seen = []
        mod = fake_module(["Video unavailable", None], seen, self.root / "v.mp4")
        with mock.patch.dict(sys.modules, {"yt_dlp": mod}):
            with self.assertRaises(StageError):
                download._download(self.ctx, lambda f: None)
        self.assertEqual(len(seen), 1)

    def test_cookies_option_is_passed_when_configured(self):
        self.cfg["download"]["cookies_from_browser"] = "firefox"
        _, seen = self._run([None])
        self.assertEqual(seen[0]["cookiesfrombrowser"], ("firefox",))

    def test_classify_copy_paste_variants(self):
        for raw in ('\u201chttps://youtu.be/x\u201d', "<https://youtu.be/x>", "\ufeffyoutu.be/x", "see https://youtu.be/x now"):
            self.assertEqual(download.classify(raw), ("url", "https://youtu.be/x"), raw)
        for bad in ("notes.txt", "hello", "C:/nope/missing.mp4"):
            with self.assertRaises(StageError):
                download.classify(bad)

    def test_ytdlp_age(self):
        self.assertEqual(ytdlp_age_days("2026.08.19", date(2026, 11, 18)), 91)
        self.assertEqual(ytdlp_age_days("2026.08.04.234419", date(2026, 8, 5)), 1)
        self.assertIsNone(ytdlp_age_days("weird"))


if __name__ == "__main__":
    unittest.main()
