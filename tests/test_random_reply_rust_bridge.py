from __future__ import annotations

import random

import pytest

from api.bot import general_commands
from api.i18n import Locale, use_locale


class _FakeRustRandomReply:
    def __init__(self, response: tuple[str, str] | Exception) -> None:
        self.response = response
        self.input: tuple[int, int] | None = None

    def evaluate_random_reply(self, response_sample: int, suffix_sample: int) -> tuple[str, str]:
        self.input = (response_sample, suffix_sample)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize(
    ("outcome", "name", "locale", "expected"),
    [
        (("yes", "none"), "test", "es", "si"),
        (("no", "address"), "test", "es", "no boludo"),
        (("no", "name"), "astro", "es", "no astro"),
        (("yes", "address"), "test", "en", "yes dude"),
    ],
)
def test_rust_outcome_uses_adapter_localization(
    monkeypatch: pytest.MonkeyPatch,
    outcome: tuple[str, str],
    name: str,
    locale: Locale,
    expected: str,
) -> None:
    rust = _FakeRustRandomReply(outcome)
    monkeypatch.setattr(general_commands, "_load_rust_random_reply_evaluator", lambda: rust)
    samples = iter([0, 2])
    monkeypatch.setattr(random, "randint", lambda _start, _end: next(samples))

    with use_locale(locale):
        result = general_commands.gen_random(name)

    assert result == expected
    assert rust.input == (0, 2)


def test_bridge_failure_reuses_samples_in_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rust = _FakeRustRandomReply(ValueError("synthetic failure"))
    monkeypatch.setattr(general_commands, "_load_rust_random_reply_evaluator", lambda: rust)
    samples = iter([0, 1])
    sample_calls: list[tuple[int, int]] = []

    def sample(start: int, end: int) -> int:
        sample_calls.append((start, end))
        return next(samples)

    monkeypatch.setattr(random, "randint", sample)

    with use_locale("es"):
        result = general_commands.gen_random("test")

    assert result == "no boludo"
    assert sample_calls == [(0, 1), (0, 2)]
    assert "using Python fallback" in caplog.text
