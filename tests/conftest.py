import os
from concurrent.futures import Future

import pytest
import redis as redis_module

from api.core import config as config_module
from api import index as index_module
from api.bot.chat_settings import reset_chat_config_cache
from api.providers.backoff import clear_all_cooldowns
from api.services import bcra as bcra_service


class _FastFailRedis:
    """Redis stand-in that raises ConnectionError immediately on any call."""

    def __getattr__(self, name: str):
        def raiser(*args, **kwargs):
            raise redis_module.ConnectionError("test: Redis not available")

        return raiser


class _NoopExecutor:
    """Prevent background work from escaping an individual test."""

    def submit(self, _fn):
        future = Future()
        future.set_result(None)
        return future


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
    clear_all_cooldowns()
    reset_chat_config_cache()
    monkeypatch.setenv(
        "BOT_SYSTEM_PROMPT",
        "sos el gordo, un bot argentino de prueba.\n\nReglas de prueba.",
    )
    monkeypatch.setattr(
        index_module.chat_config_db_service,
        "is_configured",
        lambda: False,
    )
    monkeypatch.setattr("time.sleep", lambda *_, **__: None)
    monkeypatch.setattr(
        index_module.app_runtime.providers,
        "complete",
        lambda *_, **__: "",
    )
    monkeypatch.setattr(
        index_module.app_runtime.config,
        "redis",
        lambda *_, **__: _FastFailRedis(),
    )
    monkeypatch.setattr(index_module, "_BACKGROUND_REFRESH_EXECUTOR", _NoopExecutor())
    config_module.reset_cache()
    yield
