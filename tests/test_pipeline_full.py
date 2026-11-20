"""Full pipeline (M1-M7) on a synthetic video: real ffmpeg, fake Whisper, fake face detector."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from streamclipper import clipwork
from streamclipper.db import Database
from streamclipper.pipeline import run_job
from streamclipper.stages import transcribe
from tests.helpers import make_talk_video, temp_cfg
from tests.test_framing import MovingFace, box
from tests.test_transcribe import FakeModel, W

LINES = ["So today we try the new boss.", "It looks really hard honestly.", "Wait, listen to this part!",
         "He jumps and misses completely!", "Bro what was that?", "I cannot believe it happened.",
         "Okay let us try again now.", "This time I will be careful."]


def script_segments(seconds=60):
    segs, t, k = [], 1.0, 0
    while t + 3.4 < seconds:
        words, tt = [], t
        for w in LINES[k % len(LINES)].split():
            words.append(W(" " + w, round(tt, 2), round(tt + 0.3, 2)))
            tt += 0.4
        segs.append(NS(start=t, end=tt, words=words))
        t += 4.0
        k += 1
    return segs


def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                        "-show_entries", "format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(p.stdout)
    return d["streams"][0]["width"], d["streams"][0]["height"], d["streams"][0]["r_frame_rate"], float(d["format"]["duration"])


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)
        self.cfg["moments"]["min_score"] = 0.0
        self.db = Database(self.cfg.db_path)
        self.video = make_talk_video(self.root / "talk.mp4")
        self.addCleanup(self.tmp.cleanup)

    def run_all(self, language="en", debug=False):
        self.cfg["general"]["debug"] = debug
        jid = self.db.create_job(str(self.video), "file", "fast", language)
        face = MovingFace(lambda i: [box(0.4 + 0.002 * i, 0.4, 0.14, 0.3)])
        with mock.patch.object(transcribe, "load_model", return_value=FakeModel(script_segments(), "en")), \
             mock.patch.object(clipwork, "get_detector", return_value=face):
            timer = run_job(self.cfg, self.db, jid)
        return jid, timer

    def test_video_becomes_reviewable_clips(self):
        jid, timer = self.run_all(debug=True)
        job = self.db.get_job(jid)
        self.assertEqual(job["status"], "done", job.get("error"))
        clips = self.db.list_clips(jid)
        self.assertGreaterEqual(len(clips), 1)
        c = clips[0]
        d = c["data"]
        self.assertEqual(c["status"], "ready")
        # the loud burst at 30 s is inside the chosen clip
        self.assertTrue(c["start"] <= 31 <= c["end"], (c["start"], c["end"]))
        self.assertTrue(12 <= d["duration"] <= 60)
        # boundaries are real and explained
        self.assertTrue(d["start_reason"] and d["end_reason"])
        # output format: 1080x1920 / 30 fps, duration matches the plan
        w, h, fps, dur = probe(d["files"]["review"])
        self.assertEqual((w, h, fps), (1080, 1920, "30/1"))
        self.assertAlmostEqual(dur, d["duration"], delta=0.5)
        # captions really burned in (self-test pixels) and layout chosen from faces
        self.assertGreater(d["render"]["caption"]["verified_pixels"], 300)  # captions really drawn
        self.assertEqual(d["layout_used"], "single")
        # loudness normalised near -14 LUFS => no loudness flag
        self.assertFalse([f for f in d["flags"] if f["code"] == "loudness"], d["flags"])
        self.assertFalse([f for f in d["flags"] if f["code"] in ("wrong_size", "captions_missing", "render_failed")], d["flags"])
        # metadata + cover
        self.assertTrue(d["meta"]["title"] and d["meta"]["hashtags"])
        self.assertTrue(Path(d["files"]["cover"]).is_file())
        # --debug artefacts
        dbg = self.cfg.data_dir / "debug" / job["source_key"] / d["cid"]
        for name in ("plan.json", "contact.jpg", "subs.ass", "captions.srt"):
            self.assertTrue((dbg / name).is_file(), name)
        print("\nTiming table:\n" + timer.table())
        print("Clip:", round(c["start"], 1), "-", round(c["end"], 1), "|", d["start_reason"], "|", d["end_reason"])

    def test_rerun_is_cached_and_keeps_user_edits(self):
        jid, _ = self.run_all()
        clip = self.db.list_clips(jid)[0]
        self.db.update_clip(clip["id"], data={"edited": True, "marker": 1})
        face = MovingFace(lambda i: [box(0.4, 0.4, 0.14, 0.3)])
        with mock.patch.object(transcribe, "load_model", side_effect=AssertionError("transcribed twice")), \
             mock.patch.object(clipwork, "get_detector", return_value=face):
            timer = run_job(self.cfg, self.db, jid)
        cached = {n: c for n, _, c in timer.rows}
        self.assertTrue(all(cached[s] for s in ("download", "audio", "transcribe", "segment", "score", "plan", "frame", "render", "check")), cached)
        self.assertEqual(self.db.get_clip(clip["id"])["data"]["marker"], 1)


if __name__ == "__main__":
    unittest.main()
