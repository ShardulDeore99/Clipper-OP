"""Find a caption font and copy it next to the .ass file, so libass never depends on system lookup."""
import os
import shutil
from pathlib import Path

from .errors import StageError

ROOT = Path(__file__).resolve().parent.parent
# name -> (ASS family name, file names, bold flag for the style)
KNOWN = {
    "anton": ("Anton", ["Anton-Regular.ttf", "Anton.ttf"], 0),
    "impact": ("Impact", ["impact.ttf", "Impact.ttf"], 0),
    "arial black": ("Arial Black", ["ariblk.ttf", "Arial Black.ttf"], 0),
    "bebas neue": ("Bebas Neue", ["BebasNeue-Regular.ttf"], 0),
    "oswald": ("Oswald", ["Oswald-Bold.ttf", "Oswald[wght].ttf"], -1),
    "arial": ("Arial", ["arialbd.ttf"], -1),
    "dejavu sans": ("DejaVu Sans", ["DejaVuSans-Bold.ttf"], -1),
    "liberation sans": ("Liberation Sans", ["LiberationSans-Bold.ttf"], -1),
}
LAST_RESORT = ["dejavu sans", "liberation sans", "arial"]


def search_dirs() -> list[Path]:
    dirs = [ROOT / "assets" / "fonts"]
    if os.name == "nt":
        dirs += [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
                 Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"]
    else:
        dirs += [Path.home() / ".fonts", Path("/usr/share/fonts"), Path("/Library/Fonts"), Path("/usr/local/share/fonts")]
    return [d for d in dirs if d.is_dir()]


def _find_file(names: list[str]) -> Path | None:
    wanted = {n.lower() for n in names}
    for d in search_dirs():
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower() in wanted:
                    return Path(root) / f
    return None


def find_font(cfg) -> dict:
    c = cfg["captions"]
    order = [c["font"]] + list(c["fallback_fonts"]) + LAST_RESORT
    for i, name in enumerate(order):
        key = name.lower()
        if key in KNOWN:
            family, files, bold = KNOWN[key]
            path = _find_file(files)
            if path:
                return {"family": family, "path": str(path), "bold": bold, "requested": c["font"],
                        "is_fallback": i > 0}
    raise StageError("render", "No usable caption font was found.",
                     "Put Anton-Regular.ttf (free, Google Fonts) into the assets/fonts folder.")


def stage_font(font: dict, workdir: Path) -> Path:
    """Copy the font into <workdir>/fonts so the subtitles filter can use a short relative path."""
    d = workdir / "fonts"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / Path(font["path"]).name
    if not dest.exists():
        shutil.copy2(font["path"], dest)
    return d
