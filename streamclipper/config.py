import tomllib
from pathlib import Path

from .errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.toml"

# Keys M1 depends on. Later milestones add their own.
REQUIRED = {
    "general": ["mode", "data_dir", "debug", "show_rights_notice"],
    "tools": ["ffmpeg", "ffprobe"],
    "web": ["host", "port"],
    "download": ["max_height", "keep_original"],
    "audio": ["sample_rate", "channels"],
    "clips": ["max_per_source", "min_seconds", "target_min_seconds", "target_max_seconds", "max_seconds",
              "lead_in_seconds", "tail_out_seconds", "hook_text"],
    "moments": ["min_score", "candidates_per_minute", "pause_seconds", "pause_boundary", "keywords", "hook_words"],
    "framing": ["layout", "face_sample_fps", "analysis_height", "edge_margin_pct", "facecam_style"],
    "captions": ["font", "fallback_fonts", "font_size", "highlight_color", "words_per_caption", "max_lines", "y_center", "glossary"],
    "render": ["review_encoder", "review_crf", "export_crf", "export_size", "export_fps", "target_lufs",
               "max_parallel_renders", "overlays"],
    "live": ["buffer_seconds", "burst_db", "burst_seconds", "chat_spike_ratio", "lead_seconds", "max_wait_seconds",
             "merge_seconds", "hold_seconds", "max_clip_seconds"],
    "transcribe": ["language", "model_fast", "model_quality", "compute_type", "cpu_threads",
                   "beam_size", "vad_filter", "hinglish_prompt"],
    "intelligence": ["audio_weight", "chat_weight", "speech_weight", "visual_weight",
                     "humor_candidate_threshold", "high_priority_score", "potential_score", "review_score"],
}
MODES = ("fast", "quality")


class Config:
    def __init__(self, data: dict, path: Path):
        self._d, self.path = data, path

    def __getitem__(self, section: str) -> dict:
        return self._d[section]

    @property
    def data_dir(self) -> Path:
        p = Path(self["general"]["data_dir"])
        return p if p.is_absolute() else ROOT / p

    @property
    def db_path(self) -> Path:
        return self.data_dir / "streamclipper.db"


def load_config(path=None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_PATH
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name} is not valid TOML: {e}")
    for section, vals in (overrides or {}).items():
        data.setdefault(section, {}).update(vals)
    for section, keys in REQUIRED.items():
        for key in keys:
            if key not in data.get(section, {}):
                raise ConfigError(f"{path.name}: missing [{section}] {key}")
    if data["general"]["mode"] not in MODES:
        raise ConfigError(f"[general] mode must be one of {MODES}, got {data['general']['mode']!r}")
    if not isinstance(data["download"]["max_height"], int) or data["download"]["max_height"] < 144:
        raise ConfigError("[download] max_height must be a whole number of at least 144")
    if not isinstance(data["clips"]["max_per_source"], int) or data["clips"]["max_per_source"] < 0:
        raise ConfigError("[clips] max_per_source must be 0 (unlimited) or a positive number")
    weights = [data["intelligence"][f"{name}_weight"] for name in ("audio", "chat", "speech", "visual")]
    if any(not isinstance(w, (int, float)) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ConfigError("[intelligence] signal weights must be non-negative numbers with a positive sum")
    return Config(data, path)


# ---- per-source "clip look": chosen in the UI for a video or a live stream, stored in the job's meta --------
STYLE_CHOICES = {
    "layout": ("auto", "gameplay_facecam", "single", "two_speakers", "fit_blur"),
    "facecam_style": ("split", "stack", "corner"),
    "caption_pos": ("upper", "middle", "lower"),
    "caption_look": ("outline", "box"),
    "caption_theme": ("funny", "hype", "rage", "fail", "scary", "wholesome", "neutral"),
    "caption_size": ("small", "medium", "large"),
    "caption_mode": ("on", "off"),
}
LAYER_POSITION_KEYS = ("facecam_x", "facecam_y", "gameplay_x", "gameplay_y")
CAPTION_Y = {"upper": 1060, "middle": 1250, "lower": 1420}
CAPTION_PX = {"small": 72, "medium": 92, "large": 112}


def clean_style(style: dict | None) -> dict:
    """Keep only known keys with known values; an empty/unknown value means 'use config.toml'."""
    out = {}
    for k, v in (style or {}).items():
        if k in ("caption_x", "caption_y", *LAYER_POSITION_KEYS):
            limit = 1080 if k.endswith("_x") else 1920
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= limit:
                raise ConfigError(f"{k} must be a number between 0 and {limit}")
            out[k] = round(v)
            continue
        if k not in STYLE_CHOICES or v in (None, "", "default"):
            continue
        if v not in STYLE_CHOICES[k]:
            raise ConfigError(f"{k} must be one of {list(STYLE_CHOICES[k])}, got {v!r}")
        out[k] = v
    return out


def job_cfg(cfg: "Config", job: dict | None) -> "Config":
    """A copy of cfg with this job's clip-look choices applied (the global config is never changed)."""
    style = clean_style(((job or {}).get("meta") or {}).get("style"))
    if not style:
        return cfg
    import copy
    d = copy.deepcopy(cfg._d)
    if "layout" in style:
        d["framing"]["layout"] = style["layout"]
    if "facecam_style" in style:
        d["framing"]["facecam_style"] = style["facecam_style"]
    for k in LAYER_POSITION_KEYS:
        if k in style:
            d["framing"][k] = style[k]
    if "caption_pos" in style:
        d["captions"]["y_center"] = CAPTION_Y[style["caption_pos"]]
    if "caption_x" in style:
        d["captions"]["x_center"] = style["caption_x"]
    if "caption_y" in style:
        d["captions"]["y_center"] = style["caption_y"]
    if "caption_x" in style or "caption_y" in style:
        d["captions"]["position_locked"] = True
    if "caption_size" in style:
        d["captions"]["font_size"] = CAPTION_PX[style["caption_size"]]
    if "caption_look" in style:
        d["captions"]["look"] = style["caption_look"]
    if "caption_theme" in style:
        d["captions"]["highlight_color"] = style["caption_theme"]
    if style.get("caption_mode") == "off":
        d["captions"]["enabled"] = False
    return Config(d, cfg.path)
