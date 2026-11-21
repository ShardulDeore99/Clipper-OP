"""Everything that happens to ONE clip: frame -> captions + render -> check -> metadata/cover.
Each step caches by its inputs in data/clips/<source>/<clip>/, so a tweak re-runs only what it changes."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import cache
from .captions import clip_words, load_glossary, make_ass, probe_time, verify_render, build_groups, clean_token
from .checks import boundary_flags, edge_silence_flags, face_flags, flag, format_flags, loudness_flags
from .config import ROOT
from .errors import StageError
from .fonts import find_font, stage_font
from .framing import LAYOUTS, build_filter, faces_in_output, frame_clip
from .media import run
from .overlays import plan_stickers, sticker_events
from .publish_meta import make_meta
from .render import measure_loudness, render_clip
from .transcript import to_srt

log = logging.getLogger("streamclipper")
_DETECTOR = None


def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from .faces import HaarDetector
        _DETECTOR = HaarDetector()
    return _DETECTOR


@dataclass
class SourceInfo:
    key: str
    video: str
    width: int
    height: int
    duration: float
    words: list
    starts: list
    ends: list
    speech: bool
    language: str | None = None
    title: str = ""
    mode: str = "fast"
    extra: dict = field(default_factory=dict)


def load_source(cfg, key: str, mode: str = "fast", title: str = "") -> SourceInfo:
    root = cache.cache_root(cfg, key)

    def done(stage):
        return json.loads((root / stage / "done.json").read_text(encoding="utf-8"))["outputs"]
    dl, au = done("download"), done("audio")
    seg = json.loads(Path(done("segment")["segment_json"]).read_text(encoding="utf-8"))
    try:
        language = done("transcribe").get("language")
    except FileNotFoundError:
        language = None
    return SourceInfo(key, dl["video"], au["width"], au["height"], au["duration"], seg["words"], seg["starts"],
                      seg["ends"], not seg["pseudo"], language, title or dl.get("title", ""), mode)


class ClipWork:
    def __init__(self, cfg, src: SourceInfo, clip: dict):
        self.cfg, self.src, self.clip = cfg, src, clip
        self.dir = cfg.data_dir / "clips" / src.key / clip["cid"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dur = clip["end"] - clip["start"]

    # --- tiny per-step cache -------------------------------------------------
    def _hit(self, step, inputs, file_key=None):
        f = self.dir / f"{step}.json"
        if not f.is_file():
            return None
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if rec.get("inputs") != inputs:
            return None
        if file_key and not Path(rec["out"][file_key]).is_file():
            return None
        return rec["out"]

    def _save(self, step, inputs, out):
        (self.dir / f"{step}.json").write_text(json.dumps({"inputs": inputs, "out": out}), encoding="utf-8")
        return out

    # --- M4 ---------------------------------------------------------------------
    def frame(self) -> dict:
        layout = self.clip.get("layout") or self.cfg["framing"]["layout"] or "auto"
        if layout not in LAYOUTS:
            raise StageError("frame", f"Unknown layout '{layout}'.", f"Use one of {', '.join(LAYOUTS)}.")
        position_keys = ("facecam_style", "facecam_x", "facecam_y", "gameplay_x", "gameplay_y")
        positions = {k: self.clip[k] for k in position_keys if k in self.clip}
        inputs = {"start": self.clip["start"], "end": self.clip["end"], "layout": layout, "framing": self.cfg["framing"],
                  "positions": positions,
                  "video": cache.sig(self.src.video, content=False), "v": 1}
        path = self.dir / "framing.json"
        if self._hit("frame", inputs, "path"):
            return json.loads(path.read_text(encoding="utf-8"))
        forced = layout if layout != "auto" else self.cfg["framing"]["layout"]
        detector = None if forced == "no_face" else get_detector()
        fr = frame_clip(self.cfg, self.cfg["tools"]["ffmpeg"], self.src.video, self.src.width, self.src.height,
                        self.clip["start"], self.dur, detector, None if layout == "auto" else layout, positions)
        path.write_text(json.dumps(fr), encoding="utf-8")
        self._save("frame", inputs, {"path": str(path)})
        return fr

    # --- M5 + M6 ----------------------------------------------------------------
    def words(self) -> list[dict]:
        rules = load_glossary(ROOT / self.cfg["captions"]["glossary"])
        base = self.src.words if self.src.speech else []
        return clip_words(base, self.clip["start"], self.clip["end"], rules, self.clip.get("caption_text"))

    def render(self, final: bool = False) -> dict:
        fr = self.frame()
        words = self.words()
        c, r = self.cfg["captions"], self.cfg["render"]
        captions_enabled = c.get("enabled", True)
        visible_words = words if captions_enabled else []
        out_path = self.dir / ("final.mp4" if final else "review.mp4")
        inputs = {"start": self.clip["start"], "end": self.clip["end"], "fr": cache.sig(self.dir / "framing.json"),
                  "words": words, "final": final, "mode": self.src.mode, "captions": c, "render": r,
                  "video": cache.sig(self.src.video, content=False), "v": 1}
        hit = self._hit("render_final" if final else "render", inputs, "path")
        if hit:
            return hit
        font = find_font(self.cfg)
        stage_font(font, self.dir)
        faces_fn = lambda t: faces_in_output(fr, t)
        extra = sticker_events(plan_stickers(words, self.dur, faces_fn)) if r["overlays"] else []
        ass, stats = make_ass(self.cfg, font, visible_words, faces_fn, extra)
        (self.dir / "subs.ass").write_text(ass, encoding="utf-8")
        (self.dir / "captions.srt").write_text(self._srt(words), encoding="utf-8")
        verified = 0
        if visible_words:
            verification_time = probe_time(visible_words)
            if verification_time is not None:
                verified = verify_render(self.cfg["tools"]["ffmpeg"], self.dir, verification_time)
            else:
                log.info("Clip %s: transcript contains no glyphs supported by the caption renderer", self.clip["cid"])
        elif words and not captions_enabled:
            log.info("Clip %s: on-screen captions disabled", self.clip["cid"])
        else:
            log.info("Clip %s: no speech detected, rendering without captions", self.clip["cid"])
        loud = measure_loudness(self.cfg, self.src.video, self.clip["start"], self.dur)
        info = render_clip(self.cfg, self.src.video, self.clip["start"], self.dur, build_filter(fr, fr.get("style", "stack")),
                           bool(visible_words or extra), self.dir, out_path, final, self.src.mode, loud)
        out = {"path": str(out_path), "layout": fr["layout"], **info,
               "caption": {**stats, "verified_pixels": verified, "has_speech": bool(words),
                           "enabled": captions_enabled,
                           "font": font["family"], "font_fallback": font["is_fallback"]}}
        return self._save("render_final" if final else "render", inputs, out)

    def _srt(self, words) -> str:
        groups = build_groups([{**w, "w": clean_token(w["w"])[0]} for w in words], tuple(self.cfg["captions"]["words_per_caption"]))
        segs = [{"start": g[0]["start"], "end": g[-1]["end"], "text": " ".join(x["w"] for x in g)} for g in groups]
        return to_srt(segs)

    # --- M7 ---------------------------------------------------------------------
    def check(self) -> list[dict]:
        out = self.render()
        fr = self.frame()
        inputs = {"render": out["path"], "size": Path(out["path"]).stat().st_size, "start": self.clip["start"],
                  "end": self.clip["end"], "v": 1}
        hit = self._hit("check", inputs)
        if hit is not None:
            return hit["flags"]
        flags = []
        if self.src.speech:
            flags += boundary_flags(self.src.words, self.src.starts, self.src.ends, self.clip["start"], self.clip["end"])
        flags += edge_silence_flags(self.cfg, out["path"], self.dur)
        f2, _ = format_flags(self.cfg, out["path"])
        flags += f2
        flags += loudness_flags(self.cfg, measure_loudness(self.cfg, out["path"]))
        flags += face_flags(fr)
        cap = out["caption"]
        if cap.get("enabled", True) and cap["has_speech"] and not cap["verified_pixels"]:
            flags.append(flag("captions_missing", "Speech was found but captions were not rendered.", "error"))
        if not cap["has_speech"]:
            flags.append(flag("no_speech", "No speech detected: rendered without captions.", "info"))
        if cap["font_fallback"]:
            flags.append(flag("font_fallback", f"Caption font '{self.cfg['captions']['font']}' not found; used {cap['font']}.", "info"))
        if cap.get("glyphs_removed"):
            flags.append(flag("glyphs_removed", "Some caption characters are not supported by the font and were removed."))
        if fr.get("note"):
            flags.append(flag("layout_note", fr["note"], "info"))
        for m in self.clip.get("flags_plan", []):
            flags.append(flag("plan", m))
        return self._save("check", inputs, {"flags": flags})["flags"]

    def describe(self) -> dict:
        fr = self.frame()
        meta = make_meta({**self.clip, "end": self.clip["end"]}, self.src.language, self.src.title, fr)
        review = self.render()["path"]
        cover = self.dir / "cover.jpg"
        run([self.cfg["tools"]["ffmpeg"], "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{meta['cover_time']:.2f}",
             "-i", review, "-frames:v", "1", "-q:v", "3", str(cover)], "save")
        meta["cover"] = str(cover)
        return meta
