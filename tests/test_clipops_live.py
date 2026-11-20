"""M7 editing and M8 live monitor, end to end on synthetic media (real ffmpeg, fake Whisper, fake faces)."""
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from streamclipper import clipops, clipwork
from streamclipper.db import Database
from streamclipper.live import LiveMonitor, read_segments
from streamclipper.pipeline import run_job
from streamclipper.stages import transcribe
from tests.helpers import make_talk_video, temp_cfg
from tests.test_framing import MovingFace, box
from tests.test_pipeline_full import probe, script_segments
from tests.test_transcribe import FakeModel


def patched(face=None):
    face = face or MovingFace(lambda i: [box(0.4, 0.4, 0.14, 0.3)])
    return (mock.patch.object(transcribe, "load_model", return_value=FakeModel(script_segments(), "en")),
            mock.patch.object(clipwork, "get_detector", return_value=face))


class ClipOpsTests(unittest.TestCase):
    def test_trim_captions_layout_export_delete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = temp_cfg(root)
            cfg["moments"]["min_score"] = 0.0
            db = Database(cfg.db_path)
            video = make_talk_video(root / "t.mp4")
            jid = db.create_job(str(video), "file", "fast", "en")
            p1, p2 = patched()
            with p1, p2:
                run_job(cfg, db, jid)
                clip = db.list_clips(jid)[0]
                cid, old = clip["id"], clip["data"]

                # validation
                for bad in ({"start": 20, "end": 10}, {"start": 10, "end": 11}, {"layout": "banana"}, {"volume": 3}):
                    with self.assertRaises(ValueError):
                        clipops.validate(cfg, bad, old, 60)

                # nudge the start by 1 s: re-render, reason recorded, boundary check speaks up
                clipops.apply_edit(cfg, db, cid, {"start": round(old["start"] + 1.0, 2)})
                row = db.get_clip(cid)
                self.assertEqual(row["status"], "ready")
                self.assertAlmostEqual(row["start"], old["start"] + 1.0, places=1)
                self.assertIn("manual trim", row["data"]["start_reason"])
                self.assertTrue(row["data"]["edited"])
                self.assertAlmostEqual(probe(row["path"])[3], row["end"] - row["start"], delta=0.5)

                # edit captions: new words burned in, timing kept
                clipops.apply_edit(cfg, db, cid, {"caption_text": "mai kyu aau nahi tu ja"})
                row = db.get_clip(cid)
                ass = (cfg.data_dir / "clips" / db.get_job(jid)["source_key"] / old["cid"] / "subs.ass").read_text()
                self.assertIn("KYU", ass)
                self.assertGreater(row["data"]["render"]["caption"]["verified_pixels"], 300)

                # turn off on-screen captions while retaining editable transcript data
                db.update_job(jid, meta={"style": {"caption_mode": "off"}})
                clipops.apply_edit(cfg, db, cid, {"layout": "fit_blur"})
                row = db.get_clip(cid)
                self.assertEqual(row["data"]["layout_used"], "fit_blur")
                self.assertFalse(row["data"]["render"]["caption"]["enabled"])
                self.assertEqual(row["data"]["render"]["caption"]["verified_pixels"], 0)
                self.assertNotIn("captions_missing", [f["code"] for f in row["data"]["flags"]])
                clip_dir = cfg.data_dir / "clips" / db.get_job(jid)["source_key"] / old["cid"]
                self.assertIn("MAI KYU", (clip_dir / "captions.srt").read_text())
                self.assertNotIn("MAI KYU", (clip_dir / "subs.ass").read_text())

                # final export: 1080x1920, 30 fps
                final = clipops.export_final(cfg, db, cid)
                self.assertEqual(probe(final)[:3], (1080, 1920, "30/1"))
                self.assertEqual(db.get_clip(cid)["data"]["files"]["final"], final)

                # delete removes the row and the files
                clipops.delete_clip(cfg, db, cid)
                with self.assertRaises(LookupError):
                    db.get_clip(cid)
                self.assertFalse((cfg.data_dir / "clips" / db.get_job(jid)["source_key"] / old["cid"]).exists())


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)
        self.db = Database(self.cfg.db_path)
        self.addCleanup(self.tmp.cleanup)

    def monitor(self, video, **kw):
        jobs = []
        m = LiveMonitor(self.cfg, self.db, jobs.append, str(video), 1, **kw)
        m.start()
        m.join(60)
        self.assertFalse(m.is_alive(), "monitor did not finish")
        m._wait_cutters()
        return m, jobs

    def test_burst_triggers_one_clip_that_flows_through_the_pipeline(self):
        video = make_talk_video(self.root / "live.mp4", 60, 30)
        m, jobs = self.monitor(video)
        self.assertIsNone(m.error, m.error)
        self.assertEqual(len(jobs), 1, list(m.events))
        job = self.db.get_job(jobs[0])
        trig = job["meta"]["live_trigger"]
        rough = Path(job["source"])
        self.assertTrue(rough.is_file())
        self.assertTrue(0 < trig["t"] < probe(rough)[3])              # trigger sits inside the cut
        self.assertLessEqual(probe(rough)[3], self.cfg["live"]["max_clip_seconds"] + 8)
        self.assertGreater(probe(rough)[3], 12)                        # enough context for the planner
        self.assertTrue(m.triggers >= 1)

        p1, p2 = patched()
        with p1, p2:                                                   # same pipeline as any other source
            run_job(self.cfg, self.db, jobs[0])
        clips = self.db.list_clips(jobs[0])
        self.assertEqual(len(clips), 1)
        c = clips[0]
        self.assertEqual(c["status"], "ready")
        self.assertEqual(probe(c["path"])[:3], (1080, 1920, "30/1"))
        self.assertTrue(c["start"] <= trig["t"] <= c["end"])

    def test_close_triggers_are_merged(self):
        video = make_talk_video(self.root / "live2.mp4", 70, [25, 31])   # two bursts 6 s apart (< merge window)
        m, jobs = self.monitor(video)
        self.assertEqual(len(jobs), 1, list(m.events))
        self.assertGreaterEqual(m.triggers, 2)

    def test_far_apart_triggers_make_separate_clips(self):
        video = make_talk_video(self.root / "live3.mp4", 100, [20, 70])
        m, jobs = self.monitor(video)
        self.assertEqual(len(jobs), 2, list(m.events))

    def test_chat_spike_alone_can_trigger(self):
        video = make_talk_video(self.root / "quiet.mp4", 60, 200)        # no audio burst at all
        chat = self.root / "chat.txt"
        chat.write_text("".join(f"{40 + i * 0.3:.2f}\n" for i in range(20)))   # 20 messages in 6 s around t=40
        m, jobs = self.monitor(video, chat_file=str(chat))
        self.assertGreaterEqual(len(jobs), 1, list(m.events))
        self.assertTrue(any("chat" in e["text"] for e in m.events))

    def test_bad_source_is_reported_not_silent(self):
        m = LiveMonitor(self.cfg, self.db, lambda j: None, "not a link", 1)
        m.start()
        m.join(10)
        self.assertEqual(m.status, "error")
        self.assertIn("link", m.error)


if __name__ == "__main__":
    unittest.main()
