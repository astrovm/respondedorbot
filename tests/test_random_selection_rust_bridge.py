from __future__ import annotations

import json
import random

import pytest

from api.bot import general_commands
from api.i18n import use_locale


class _FakeRustRandomParser:
    def __init__(self, response: dict[str, object] | Exception) -> None:
        self.response = response
        self.input: str | None = None

    def parse_random_selection(self, message_text: str) -> str:
        self.input = message_text
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def test_rust_choices_use_the_existing_random_source(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustRandomParser({"kind": "choices", "values": ["alpha", "beta"]})
    monkeypatch.setattr(general_commands, "_load_rust_random_selection_parser", lambda: rust)
    choice_calls: list[list[str]] = []

    def choose_second(values: list[str]) -> str:
        choice_calls.append(values)
        return values[1]

    monkeypatch.setattr(random, "choice", choose_second)

    assert general_commands.select_random("alpha, beta") == "beta"
    assert rust.input == "alpha, beta"
    assert choice_calls == [["alpha", "beta"]]


def test_rust_range_preserves_arbitrary_precision_and_inclusive_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = "100000000000000000000"
    end = "100000000000000000002"
    rust = _FakeRustRandomParser({"kind": "range", "start": start, "end": end})
    monkeypatch.setattr(general_commands, "_load_rust_random_selection_parser", lambda: rust)
    range_calls: list[tuple[int, int]] = []

    def choose_last(first: int, last: int) -> int:
        range_calls.append((first, last))
        return last

    monkeypatch.setattr(random, "randint", choose_last)

    assert general_commands.select_random(f"{start}-{end}") == end
    assert range_calls == [(int(start), int(end))]


def test_rust_invalid_result_is_localized(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustRandomParser({"kind": "invalid"})
    monkeypatch.setattr(general_commands, "_load_rust_random_selection_parser", lambda: rust)

    with use_locale("en"):
        result = general_commands.select_random("invalid")

    assert result == "send options like 'pizza, steak, sushi' or a range like '1-10'"


def test_bridge_failure_uses_unicode_compatible_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rust = _FakeRustRandomParser(ValueError("synthetic unsupported Unicode"))
    monkeypatch.setattr(general_commands, "_load_rust_random_selection_parser", lambda: rust)

    def choose_last(_start: int, end: int) -> int:
        return end

    monkeypatch.setattr(random, "randint", choose_last)

    assert general_commands.select_random("１-３") == "3"
    assert "using Python fallback" in caplog.text
