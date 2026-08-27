from unittest.mock import Mock


from api.bot.chat_config_service import build_chat_config_service
from api.bot.chat_config_defaults import CHAT_CONFIG_DEFAULTS
from api.storage.chat_config_repository import ChatConfigRepository

def test_get_chat_config_returns_repo_when_present():
    repo = Mock(spec=ChatConfigRepository)
    repo.is_configured.return_value = True
    repo.get_chat_config.return_value = {"link_mode": "delete"}

    service = build_chat_config_service(
        repository=repo,
        admin_reporter=lambda *a, **k: None,
        log_event=lambda *a, **k: None,
    )
    cfg = service.get_chat_config("123")
    assert cfg["link_mode"] == "delete"
    repo.get_chat_config.assert_called_once()


def test_get_chat_config_returns_defaults_when_repo_has_no_record():
    repo = Mock(spec=ChatConfigRepository)
    repo.is_configured.return_value = True
    repo.get_chat_config.return_value = None

    admin_reporter = Mock()
    service = build_chat_config_service(
        repository=repo, admin_reporter=admin_reporter, log_event=lambda *a, **k: None
    )
    cfg = service.get_chat_config("123")

    assert cfg["link_mode"] == "reply"
    repo.set_chat_config.assert_not_called()
    admin_reporter.assert_not_called()


def test_get_chat_config_reports_repository_errors():
    repo = Mock(spec=ChatConfigRepository)
    repo.is_configured.return_value = True
    repository_error = RuntimeError("Postgres unavailable")
    repo.get_chat_config.side_effect = repository_error
    admin_reporter = Mock()
    service = build_chat_config_service(
        repository=repo,
        admin_reporter=admin_reporter,
        log_event=lambda *a, **k: None,
    )

    cfg = service.get_chat_config("123")

    assert cfg == CHAT_CONFIG_DEFAULTS
    admin_reporter.assert_called_once_with(
        "Error loading chat config",
        repository_error,
        {"chat_id": "123", "postgres_configured": True},
    )


def test_set_chat_config_applies_updates_and_persists():
    repo = Mock(spec=ChatConfigRepository)
    repo.is_configured.return_value = True
    repo.get_chat_config.return_value = {"link_mode": "reply"}

    service = build_chat_config_service(
        repository=repo,
        admin_reporter=lambda *a, **k: None,
        log_event=lambda *a, **k: None,
    )
    cfg = service.set_chat_config("123", link_mode="delete")
    assert cfg["link_mode"] == "delete"
    repo.set_chat_config.assert_called_once()
