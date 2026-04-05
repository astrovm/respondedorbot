import os

import pytest
import redis as redis_module

from api import config as config_module
from api import index as index_module
from api.services import bcra as bcra_service


class _FastFailRedis:
    """Redis stand-in that raises ConnectionError immediately on any call."""

    def __getattr__(self, name: str):
        def raiser(*args, **kwargs):
            raise redis_module.ConnectionError("test: Redis not available")

        return raiser


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_artifacts():
    yield
    try:
        if os.path.isfile("test_api_key"):
            os.remove("test_api_key")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_caches(monkeypatch):
    bcra_service.reset_local_caches()
    index_module._provider_backoff_until.clear()
    monkeypatch.setattr(
        index_module.chat_config_db_service,
        "is_configured",
        lambda: False,
    )
    monkeypatch.setattr("time.sleep", lambda *_, **__: None)
    monkeypatch.setattr(index_module, "complete_with_providers", lambda *_, **__: "")
    monkeypatch.setattr(index_module, "config_redis", lambda *_, **__: _FastFailRedis())
    config_module.reset_cache()
    yield
