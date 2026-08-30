from __future__ import annotations

import json
from pathlib import Path

from api.bot import general_commands
from api.i18n import use_locale


def test_python_fallback_matches_shared_contract() -> None:
    path = Path(__file__).parents[1] / "contracts" / "base_conversion.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    with use_locale("es"):
        for case in contract["cases"]:
            assert general_commands._convert_base_python(case["input"]) == case[
                "expected_es"
            ], case["name"]


class _FakeRustConverter:
    def __init__(self, response: dict[str, object] | Exception) -> None:
        self.response = response
        self.input: str | None = None

    def convert_base(self, message_text: str) -> str:
        self.input = message_text
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def test_rust_success_is_localized(monkeypatch) -> None:
    rust = _FakeRustConverter(
        {
            "kind": "success",
            "number": "255",
            "source": 10,
            "result": "FF",
            "target": 16,
        }
    )
    monkeypatch.setattr(general_commands, "_load_rust_base_converter", lambda: rust)

    with use_locale("en"):
        result = general_commands.convert_base("255,10,16")

    assert result == "255 in base 10 is FF in base 16"
    assert rust.input == "255,10,16"


def test_rust_validation_is_localized(monkeypatch) -> None:
    rust = _FakeRustConverter({"kind": "source_range", "input": "99"})
    monkeypatch.setattr(general_commands, "_load_rust_base_converter", lambda: rust)

    with use_locale("es"):
        result = general_commands.convert_base("1,99,10")

    assert result == "base origen '99' tiene que ser entre 2 y 36 gordo"


def test_bridge_failure_uses_unicode_compatible_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustConverter(ValueError("synthetic unsupported Unicode"))
    monkeypatch.setattr(general_commands, "_load_rust_base_converter", lambda: rust)

    with use_locale("es"):
        result = general_commands.convert_base("１２,10,16")

    assert result == "ahi tenes boludo, １２ en base 10 es C en base 16"
    assert "using Python fallback" in caplog.text
