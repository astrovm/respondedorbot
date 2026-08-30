from __future__ import annotations

import re

import pytest

from api.links import context


class _FakeRustLinks:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.slice_calls: list[tuple[str, int, int]] = []

    def slice_telegram_utf16(self, text: str, offset: int, length: int) -> str:
        self.slice_calls.append((text, offset, length))
        if self.fail:
            raise ValueError("synthetic slice failure")
        return "https://example.test/from-rust"

    def trim_detected_url(self, raw_url: str) -> str:
        if self.fail:
            raise ValueError("synthetic trim failure")
        return raw_url.strip().rstrip(".")

    def select_unique_urls(self, candidates: list[str], max_links: int) -> list[str]:
        if self.fail:
            raise ValueError("synthetic selection failure")
        return list(dict.fromkeys(candidates))[:max_links]


def test_link_entity_parsing_uses_rust_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustLinks()
    monkeypatch.setattr(context, "_load_rust_link_parsing", lambda: rust)

    result = context.extract_urls_from_entities(
        "ignored source",
        [{"type": "url", "offset": 2, "length": 4}],
        normalize_url=lambda value: value,
    )

    assert result == ["https://example.test/from-rust"]
    assert rust.slice_calls == [("ignored source", 2, 4)]


def test_link_selection_uses_rust_stable_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context, "_load_rust_link_parsing", _FakeRustLinks)
    result = context.extract_message_urls(
        {"text": "https://a.test https://a.test https://b.test"},
        url_pattern=re.compile(r"(https?://[^\s]+)"),
        max_links=2,
        normalize_url=lambda value: value,
        extract_entities=lambda _text, _entities: [],
    )

    assert result == ["https://a.test", "https://b.test"]


def test_link_parsing_falls_back_after_bridge_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        context,
        "_load_rust_link_parsing",
        lambda: _FakeRustLinks(fail=True),
    )

    assert context.utf16_slice("a😀b", 3, 1) == "b"
    assert (
        context.normalize_detected_url(
            " https://example.test/path). ",
            normalize_url=lambda value: value,
        )
        == "https://example.test/path"
    )
