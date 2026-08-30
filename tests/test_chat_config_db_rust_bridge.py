import json
from unittest.mock import MagicMock

import pytest

from api.services import chat_config_db


@pytest.fixture(autouse=True)
def _reset_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_config_db, "_SCHEMA_READY", False)


def _enable_rust(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, str]:
    bridge = MagicMock()
    database_url = "postgresql://synthetic.invalid/database?sslmode=disable"
    monkeypatch.setattr(chat_config_db, "_load_rust_chat_config", lambda: bridge)
    monkeypatch.setattr(
        chat_config_db.credits_db_service,
        "get_database_url",
        lambda: database_url,
    )
    return bridge, database_url


def test_rust_read_preserves_absent_rows_and_normalizes_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, database_url = _enable_rust(monkeypatch)
    bridge.chat_config_get.side_effect = ["null", json.dumps({"language": "en"})]
    defaults = {"language": "auto", "link_mode": "reply"}

    assert chat_config_db.get_chat_config("42", defaults) is None
    assert chat_config_db.get_chat_config("42", defaults) == {
        "language": "en",
        "link_mode": "reply",
    }
    bridge.chat_config_ensure_schema.assert_called_once_with(database_url)
    assert bridge.chat_config_get.call_count == 2


def test_rust_write_is_single_owner_and_returns_typed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, database_url = _enable_rust(monkeypatch)
    bridge.chat_config_set.return_value = json.dumps(
        {"language": "en", "link_mode": "reply"}
    )
    python_connect = MagicMock(side_effect=AssertionError("Python writer must not run"))
    monkeypatch.setattr(chat_config_db.credits_db_service, "connect", python_connect)

    assert chat_config_db.set_chat_config(
        "42", {"language": "en", "link_mode": "reply"}
    ) == {"language": "en", "link_mode": "reply"}
    bridge.chat_config_set.assert_called_once_with(
        database_url,
        "42",
        json.dumps({"language": "en", "link_mode": "reply"}),
    )
    python_connect.assert_not_called()


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("schema", "Rust chat configuration schema initialization failed"),
        ("read", "Rust chat configuration read failed"),
        ("write", "Rust chat configuration write failed"),
    ],
)
def test_rust_failures_do_not_fall_back_to_a_second_writer(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    message: str,
) -> None:
    bridge, _database_url = _enable_rust(monkeypatch)
    python_connect = MagicMock(side_effect=AssertionError("fallback must not run"))
    monkeypatch.setattr(chat_config_db.credits_db_service, "connect", python_connect)
    if method == "schema":
        bridge.chat_config_ensure_schema.side_effect = RuntimeError("synthetic")
    elif method == "read":
        bridge.chat_config_get.side_effect = RuntimeError("synthetic")
    else:
        bridge.chat_config_set.side_effect = RuntimeError("synthetic")

    with pytest.raises(chat_config_db.ChatConfigDBError, match=message):
        if method == "write":
            chat_config_db.set_chat_config("42", {"language": "en"})
        else:
            chat_config_db.get_chat_config("42", {"language": "auto"})
    python_connect.assert_not_called()
