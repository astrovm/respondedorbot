from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.bot import general_commands
from api.i18n import use_locale


class _FakeRustCommandNormalizer:
    def __init__(self, response: str | Exception | None) -> None:
        self.response = response
        self.input: str | None = None

    def normalize_command_text(self, message_text: str) -> str | None:
        self.input = message_text
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_python_fallback_matches_adapter_contract() -> None:
    path = Path(__file__).parents[1] / "contracts" / "command_normalization.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    with use_locale("es"):
        for case in contract["adapter"]:
            assert general_commands._normalize_command_python(
                general_commands._preprocess_command_text(case["input"])
            ) == case["expected_es"], case["name"]


def test_rust_receives_localized_and_romanized_adapter_text(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustCommandNormalizer("/MOUSUGUDESU")
    monkeypatch.setattr(general_commands, "_load_rust_command_normalizer", lambda: rust)

    with use_locale("es"):
        result = general_commands.convert_to_command("もうすぐです")

    assert result == "/MOUSUGUDESU"
    assert rust.input == "mousugudesu"


def test_rust_invalid_result_is_localized(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustCommandNormalizer(None)
    monkeypatch.setattr(general_commands, "_load_rust_command_normalizer", lambda: rust)

    with use_locale("en"):
        result = general_commands.convert_to_command("💥")

    assert result == "the command must contain letters or numbers"


def test_bridge_failure_uses_preprocessed_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rust = _FakeRustCommandNormalizer(ValueError("synthetic failure"))
    monkeypatch.setattr(general_commands, "_load_rust_command_normalizer", lambda: rust)

    with use_locale("es"):
        result = general_commands.convert_to_command("hola ñandú ñ")

    assert result == "/HOLA_NIANDU_ENIE"
    assert rust.input == "hola ñandú ñ"
    assert "using Python fallback" in caplog.text
