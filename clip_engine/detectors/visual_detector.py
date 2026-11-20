"""Candidate-centered visual evidence using OpenCV frame sampling.

The detector reports scene change, motion and face presence. These are visual
activity clues; they do not classify humor or facial emotion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def analyze_candidates(video: str | Path, candidates: list[dict], sample_offsets=(-1.0, 0.0, 1.0)) -> list[dict]:
    try:
        import cv2
    except ImportError:
        return [{"available": False, "reason": "OpenCV unavailable"} for _ in candidates]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [{"available": False, "reason": "video frames unavailable"} for _ in candidates]
    cascade = None
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            cascade = None
    except (AttributeError, cv2.error):
        cascade = None

    output = []
    try:
        for candidate in candidates:
            t = max(0.0, float(candidate["t"]))
            frames = []
            face_counts = []
            for offset in sample_offsets:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + offset) * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(cv2.resize(frame, (min(320, w), max(1, round(h * min(320, w) / w)))), cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                if cascade is not None:
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5,
                                                     minSize=(max(16, gray.shape[1] // 20),) * 2)
                    face_counts.append(len(faces))
            if not frames:
                output.append({"available": False, "reason": "no decodable frames near candidate"})
                continue
            changes = [float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
                       for a, b in zip(frames, frames[1:])]
            scene_change = min(1.0, (max(changes, default=0.0) / 40.0))
            motion = min(1.0, (float(np.mean(changes)) / 25.0))
            # Face presence is evidence about the shot, not an emotion label.
            face_presence = min(1.0, max(face_counts, default=0) / 2.0) if cascade is not None else None
            score = min(100.0, 60 * scene_change + 40 * motion)
            evidence = []
            if scene_change >= 0.35:
                evidence.append("strong frame change")
            if motion >= 0.35:
                evidence.append("high local motion")
            if face_presence is not None and face_presence > 0:
                evidence.append("face visible (emotion not classified)")
            output.append({"available": True, "visual_score": round(score, 1),
                           "scene_change": round(scene_change, 3), "motion": round(motion, 3),
                           "face_presence": round(face_presence, 3) if face_presence is not None else None,
                           "evidence": evidence})
    finally:
        cap.release()
    return output
