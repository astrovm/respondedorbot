from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_CONTAINER_QUADLET = REPO_ROOT / "quadlets" / "respondedorbot.container"


class QuadletTests(unittest.TestCase):
    def test_bot_container_allows_graceful_auto_update_shutdown(self):
        contents = BOT_CONTAINER_QUADLET.read_text(encoding="utf-8")

        self.assertIn("AutoUpdate=registry", contents)
        self.assertIn("StopTimeout=600", contents)
        self.assertIn("TimeoutStopSec=630", contents)


if __name__ == "__main__":
    unittest.main()
