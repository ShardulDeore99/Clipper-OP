"""Frame sampling (small, grayscale, via ffmpeg) and face detection. Faces are boxes in 0-1 coordinates."""
import subprocess

import numpy as np

from .errors import StageError


class HaarDetector:
    """OpenCV Haar cascade: ships inside opencv, no model download. Best on frontal faces (facecams, podcasts)."""

    def __init__(self):
        try:
            import cv2
        except ImportError:
            raise StageError("frame", "OpenCV is not installed.", "Run setup.ps1, or set [framing] layout = \"no_face\".")
        self.cv2 = cv2
        try:
            base = cv2.data.haarcascades
            self.cascades = [cv2.CascadeClassifier(base + n) for n in ("haarcascade_frontalface_alt2.xml",
                                                                        "haarcascade_frontalface_default.xml")]
        except AttributeError:
            raise StageError("frame", f"This OpenCV build ({cv2.__version__}) has no face classifier.",
                             "Install the pinned one: pip install opencv-python-headless==4.10.0.84")
        if any(c.empty() for c in self.cascades):
            raise StageError("frame", "OpenCV face model files are missing.", "Reinstall opencv-python-headless.")

    def detect(self, gray: np.ndarray) -> list[tuple[float, float, float, float]]:
        h, w = gray.shape
        for c in self.cascades:
            found = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(20, h // 16),) * 2)
            if len(found):
                return [(x / w, y / h, (x + fw) / w, (y + fh) / h) for x, y, fw, fh in found]
        return []


def sample_frames(ffmpeg: str, video, start: float, dur: float, src_w: int, src_h: int, fps: float, height: int):
    """Grayscale frames at `fps`, shrunk to `height`. Returns (frames[n,h,w], times[n])."""
    w = max(2, int(round(height * src_w / src_h / 2)) * 2)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
           "-vf", f"fps={fps},scale={w}:{height},format=gray", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise StageError("frame", "Could not read frames for face detection: " + p.stderr.decode("utf-8", "replace")[-300:])
    n = len(p.stdout) // (w * height)
    if n == 0:
        raise StageError("frame", "No frames could be decoded for this clip.")
    frames = np.frombuffer(p.stdout, dtype=np.uint8, count=n * w * height).reshape(n, height, w)
    return frames, [round(i / fps, 3) for i in range(n)]


def shot_starts(frames: np.ndarray, thresh: float = 28.0) -> list[int]:
    """Indices where a new shot begins (big change between neighbouring samples)."""
    small = frames[:, ::6, ::6].astype(np.int16)
    return [0] + [i for i in range(1, len(small)) if np.abs(small[i] - small[i - 1]).mean() > thresh]


def motion_centroid(frames: np.ndarray) -> list[float]:
    """Where the action is, left-right (0-1), from frame differences. For gameplay with no face."""
    out, last = [], 0.5
    for i in range(len(frames)):
        if i == 0:
            out.append(last)
            continue
        d = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))
        col = (d > 24).sum(axis=0).astype(float)
        if col.sum() > 0.002 * d.size:
            last = float((col * np.arange(len(col))).sum() / col.sum() / len(col))
        out.append(last)
    return out
