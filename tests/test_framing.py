"""M4: layout choice, tracking, smoothing, and real ffmpeg renders of every layout (1080x1920)."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamclipper import framing as F
from streamclipper.faces import motion_centroid, shot_starts
from tests.helpers import make_sample, temp_cfg


def box(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class MovingFace:
    """Fake detector: a face drifting left to right, one sample per call."""

    def __init__(self, faces_fn):
        self.i, self.fn = 0, faces_fn

    def detect(self, gray):
        out = self.fn(self.i)
        self.i += 1
        return out


def size_of(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(p.stdout)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"]


class LayoutChoiceTests(unittest.TestCase):
    def test_no_face(self):
        self.assertEqual(F.choose_layout([[]] * 20)[0], "no_face")

    def test_single_big_face(self):
        self.assertEqual(F.choose_layout([[box(0.5, 0.4, 0.25, 0.4)]] * 20)[0], "single")

    def test_small_stable_corner_face_is_a_facecam(self):
        layout, info = F.choose_layout([[box(0.88, 0.8, 0.12, 0.2)]] * 20)
        self.assertEqual(layout, "gameplay_facecam")
        self.assertIn("facecam", info)

    def test_two_faces_is_two_speakers(self):
        faces = [[box(0.28, 0.45, 0.2, 0.35), box(0.72, 0.45, 0.2, 0.35)]] * 20
        layout, info = F.choose_layout(faces)
        self.assertEqual(layout, "two_speakers")
        self.assertLess(info["left"][0], info["right"][0])


class SmoothingTests(unittest.TestCase):
    def test_one_euro_removes_jitter_but_follows_a_move(self):
        rng = np.random.default_rng(0)
        xs = list(0.3 + rng.normal(0, 0.02, 40)) + list(0.7 + rng.normal(0, 0.02, 40))
        out = F.one_euro(xs, 0.25)
        self.assertLess(np.std(np.diff(out[:35])), np.std(np.diff(xs[:35])))
        self.assertGreater(out[-1], 0.6)

    def test_pan_speed_is_limited_except_at_shot_changes(self):
        out = F.limit_pan([0, 0, 1, 1], 0.25, 1.0)
        self.assertAlmostEqual(out[2], 0.25)
        self.assertEqual(F.limit_pan([0, 0, 1, 1], 0.25, 1.0, resets={2})[2], 1)

    def test_shot_detection(self):
        a = np.zeros((3, 36, 64), dtype=np.uint8)
        b = np.full((3, 36, 64), 200, dtype=np.uint8)
        self.assertEqual(shot_starts(np.concatenate([a, b])), [0, 3])

    def test_motion_centroid_follows_the_action(self):
        f = np.zeros((4, 36, 100), dtype=np.uint8)
        f[1:, :, 80:90] = np.arange(3)[:, None, None] * 90 + 100
        self.assertGreater(motion_centroid(f)[-1], 0.7)


class TrackingTests(unittest.TestCase):
    def test_crop_keeps_a_moving_face_inside_with_margin(self):
        cfg = temp_cfg(Path(tempfile.gettempdir()))
        n, W, H = 40, 1280, 720
        faces = [[box(0.2 + 0.6 * i / n, 0.4, 0.12, 0.28)] for i in range(n)]
        times = [i * 0.25 for i in range(n)]
        crop, prim = F.track_single(faces, times, set(), W, H, cfg["framing"], 0.25)
        self.assertEqual(crop["violations"], 0.0)
        xs = [p[1] for p in crop["path"]]
        self.assertGreater(xs[-1], xs[0] + 100)  # it followed the face
        self.assertLessEqual(crop["cw"], W)

    def test_expression_is_piecewise_linear(self):
        e = F._expr([[0, 0, 0], [1, 100, 0], [2, 100, 0]], 1)
        self.assertIn("if(lt(t,1.00)", e)


class RenderLayoutTests(unittest.TestCase):
    """Every layout must turn a 1280x720 source into a 1080x1920 / 30 fps video."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = temp_cfg(self.root)
        self.video = make_sample(self.root / "s.mp4", 5)

    def tearDown(self):
        self.tmp.cleanup()

    def run_layout(self, layout, faces_fn, style="stack"):
        self.cfg["framing"]["facecam_style"] = style
        fr = F.frame_clip(self.cfg, "ffmpeg", self.video, 1280, 720, 0, 4, MovingFace(faces_fn), layout)
        out = self.root / f"{layout}_{style}.mp4"
        flt = F.build_filter(fr, style)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.video), "-t", "2", "-filter_complex", flt,
                        "-map", "[v]", "-an", "-c:v", "libx264", str(out)], check=True)
        return fr, size_of(out)

    def test_single(self):
        fr, size = self.run_layout("auto", lambda i: [box(0.3 + 0.02 * i, 0.4, 0.14, 0.3)])
        self.assertEqual(fr["layout"], "single")
        self.assertEqual(size, (1080, 1920, "30/1"))
        self.assertEqual(fr["violations"], 0.0)
        self.assertTrue(F.faces_in_output(fr, 1.0))

    def test_two_speakers(self):
        fr, size = self.run_layout("auto", lambda i: [box(0.28, 0.45, 0.2, 0.35), box(0.72, 0.45, 0.2, 0.35)])
        self.assertEqual((fr["layout"], size[:2]), ("two_speakers", (1080, 1920)))

    def test_gameplay_facecam_stack_and_corner(self):
        for style in ("split", "stack", "corner"):
            fr, size = self.run_layout("auto", lambda i: [box(0.88, 0.8, 0.12, 0.2)], style)
            self.assertEqual(fr["layout"], "gameplay_facecam")
            self.assertEqual(size[:2], (1080, 1920), style)
            self.assertTrue(F.faces_in_output(fr, 1.0))

    def test_no_face_fits_over_a_blurred_copy(self):
        fr, size = self.run_layout("auto", lambda i: [])
        self.assertEqual((fr["layout"], size[:2]), ("no_face", (1080, 1920)))

    def test_normal_preset_fit_blur(self):
        fr, size = self.run_layout("fit_blur", lambda i: [box(0.5, 0.4, 0.2, 0.3)])
        self.assertEqual((fr["layout"], size[:2]), ("fit_blur", (1080, 1920)))
        self.assertEqual(F.faces_in_output(fr, 1.0), [])

    def test_forced_single_without_a_face_does_not_blind_crop(self):
        fr, size = self.run_layout("single", lambda i: [])
        self.assertEqual(fr["layout"], "no_face")
        self.assertIn("note", fr)
        self.assertEqual(size[:2], (1080, 1920))


if __name__ == "__main__":
    unittest.main()
