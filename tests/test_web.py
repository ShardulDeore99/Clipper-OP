"""API test. Skipped when fastapi/httpx are not installed (run setup.ps1 with requirements-dev.txt)."""
import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import make_sample, temp_cfg

HAVE = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx"))


@unittest.skipUnless(HAVE, "fastapi/httpx not installed")
class WebTests(unittest.TestCase):
    def test_add_file_job_runs_to_done(self):
        from fastapi.testclient import TestClient
        from streamclipper.web.app import create_app
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = make_sample(root / "s.mp4", 4)
            with TestClient(create_app(temp_cfg(root))) as c:
                self.assertEqual(c.get("/").status_code, 200)
                self.assertEqual(c.post("/api/jobs", json={"source": "C:/missing.mp4"}).status_code, 400)
                jid = c.post("/api/jobs", json={"source": str(video)}).json()["id"]
                for _ in range(60):
                    job = c.get("/api/jobs").json()[0]
                    if job["status"] in ("done", "failed"):
                        break
                    time.sleep(0.5)
                self.assertEqual(job["status"], "done", job.get("error"))
                meta = c.get("/api/meta").json()
                self.assertIn("gameplay_facecam", meta["layouts"])
                self.assertEqual(c.get("/api/clips").status_code, 200)
                self.assertEqual(c.get("/api/live").json(), [])
                self.assertEqual(c.post("/api/live", json={"source": "not a link"}).json()["status"] in ("starting", "error"), True)
                self.assertEqual(c.post("/api/clips/999/edit", json={"start": 1}).status_code, 404)
                self.assertEqual(c.delete(f"/api/jobs/{jid}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
