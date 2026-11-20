import tempfile
import unittest
from pathlib import Path

from streamclipper.config import load_config
from streamclipper.errors import ConfigError


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = load_config()
        self.assertEqual(cfg["download"]["max_height"], 720)
        self.assertEqual(cfg["clips"]["max_per_source"], 0)  # unlimited by default

    def test_missing_key_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text("[general]\nmode='fast'\n")
            with self.assertRaises(ConfigError):
                load_config(p)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(overrides={"general": {"mode": "turbo"}})

    def test_negative_clip_cap_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(overrides={"clips": {"max_per_source": -1}})


if __name__ == "__main__":
    unittest.main()


class ClipLookTests(unittest.TestCase):
    def test_job_style_overrides_a_copy_only(self):
        from streamclipper.config import clean_style, job_cfg, load_config
        cfg = load_config()
        job = {"meta": {"style": {"layout": "fit_blur", "caption_pos": "lower", "caption_look": "box", "caption_size": "large"}}}
        c = job_cfg(cfg, job)
        self.assertEqual((c["framing"]["layout"], c["captions"]["y_center"], c["captions"]["font_size"], c["captions"]["look"]),
                         ("fit_blur", 1420, 112, "box"))
        self.assertEqual(cfg["framing"]["layout"], "auto")  # the global config is untouched
        self.assertIs(job_cfg(cfg, {"meta": {}}), cfg)
        no_captions = job_cfg(cfg, {"meta": {"style": {"caption_mode": "off"}}})
        self.assertFalse(no_captions["captions"]["enabled"])
        self.assertNotIn("enabled", cfg["captions"])
        with self.assertRaises(Exception):
            clean_style({"layout": "nonsense"})
        with self.assertRaises(Exception):
            clean_style({"caption_mode": "sometimes"})
        self.assertEqual(clean_style({"layout": "", "caption_pos": "default"}), {})
