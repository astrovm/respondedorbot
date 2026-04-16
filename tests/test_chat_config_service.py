import json
from unittest.mock import Mock

import redis

from api.chat_config_service import build_chat_config_service
from api.storage.chat_config_repository import ChatConfigRepository


def make_fake_redis(get_value: str | None):
    class FakeRedis:
        def get(self, key):
            return get_value

    return FakeRedis()


def test_get_chat_config_returns_repo_when_present():
    repo = Mock(spec=ChatConfigRepository)
    repo.is_configured.return_value = True
    repo.get_chat_config.return_value = {"link_mode": "delete"}

    service = build_chat_config_service(
        repository=repo,
        admin_reporter=lambda *a, **k: None,
        log_event=lambda *a, **k: None,
    )
    redis_client = make_fake_redis(None)
    cfg = service.get_chat_config(redis_client, "123")
    assert cfg["link_mode"] == "delete"
    repo.get_chat_config.assert_called_once()


def test_get_chat_config_loads_from_redis_and_migrates():
    repo = Mock(spec=ChatConfigRepository)
    repo.is_configured.return_value = True
    repo.get_chat_config.return_value = None

    payload = json.dumps({"link_mode": "delete"})
    redis_client = make_fake_redis(payload)

    admin_reporter = Mock()
    service = build_chat_config_service(
        repository=repo, admin_reporter=admin_reporter, log_event=lambda *a, **k: None
    )
    cfg = service.get_chat_config(redis_client, "123")
    assert cfg["link_mode"] == "delete"
    repo.set_chat_config.assert_called_once()


def test_set_chat_config_applies_updates_and_persists():
    repo = Mock(spec=ChatConfigRepository)
    repo.is_configured.return_value = True
    repo.get_chat_config.return_value = {"link_mode": "reply"}

    redis_client = make_fake_redis(None)
    service = build_chat_config_service(
        repository=repo,
        admin_reporter=lambda *a, **k: None,
        log_event=lambda *a, **k: None,
    )
    cfg = service.set_chat_config(redis_client, "123", link_mode="delete")
    assert cfg["link_mode"] == "delete"
    repo.set_chat_config.assert_called_once()
