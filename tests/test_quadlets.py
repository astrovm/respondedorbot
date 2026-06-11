from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_CONTAINER_QUADLET = REPO_ROOT / "quadlets" / "respondedorbot.container"
REDIS_CONTAINER_QUADLET = REPO_ROOT / "quadlets" / "respondedorbot-redis.container"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "podman.yml"


class QuadletTests(unittest.TestCase):
    def test_bot_container_allows_graceful_auto_update_shutdown(self):
        contents = BOT_CONTAINER_QUADLET.read_text(encoding="utf-8")

        self.assertIn("AutoUpdate=registry", contents)
        self.assertIn("TimeoutStopSec=630", contents)
        self.assertNotIn("StopTimeout=600", contents)

    def test_bot_container_mounts_external_workspace_read_only(self):
        contents = BOT_CONTAINER_QUADLET.read_text(encoding="utf-8")

        self.assertIn(
            "Volume=%h/respondedorbot/workspace:/app/workspace:ro",
            contents,
        )

    def test_redis_container_uses_pinned_stack_version(self):
        contents = REDIS_CONTAINER_QUADLET.read_text(encoding="utf-8")

        self.assertIn(
            "Image=docker.io/redis/redis-stack-server:7.4.0-v8",
            contents,
        )
        self.assertNotIn("redis-stack-server:latest", contents)
        self.assertNotIn("AutoUpdate=registry", contents)

    def test_container_publish_depends_on_quality_checks(self):
        contents = CI_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("checks:", contents)
        self.assertIn("needs: checks", contents)
        self.assertIn("uv sync --locked", contents)
        self.assertIn("uv run --locked ruff check api/ tests/", contents)
        self.assertIn("uv run --locked mypy api/", contents)
        self.assertIn("uv run --locked python -m pytest -q", contents)
        self.assertIn("group: ci-${{ github.ref }}", contents)
        self.assertIn("cancel-in-progress: true", contents)

    def test_main_publish_keeps_latest_and_sha_tags(self):
        contents = CI_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn('TAG_FLAGS="--tag ${REPO}:sha-${{ github.sha }}"', contents)
        self.assertIn('TAG_FLAGS="${TAG_FLAGS} --tag ${REPO}:latest"', contents)
        self.assertIn('podman push "${REPO}:sha-${{ github.sha }}"', contents)
        self.assertIn('podman push "${REPO}:latest"', contents)


if __name__ == "__main__":
    unittest.main()
