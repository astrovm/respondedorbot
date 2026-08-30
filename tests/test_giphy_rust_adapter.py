from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from api.bot import giphy


class _FakeRustGiphy:
    def __init__(self, outcomes: list[object], *, fail: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def giphy_search(self, *arguments: object) -> str:
        self.calls.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust Giphy failure")
        return json.dumps(self.outcomes.pop(0))


def test_rust_giphy_requests_are_authoritative(monkeypatch) -> None:
    rust = _FakeRustGiphy(
        [
            {"status": "success", "urls": [f"https://example.test/{index}.gif"]}
            for index in range(4)
        ]
    )
    monkeypatch.setattr(giphy, "_load_rust_giphy_adapter", lambda: rust)
    monkeypatch.setattr(giphy.random, "randint", MagicMock(side_effect=[3, 5, 8, 13]))
    python_get = MagicMock(side_effect=AssertionError("Python HTTP must not run"))
    monkeypatch.setattr(giphy.http_client, "get", python_get)
    monkeypatch.setenv("GIPHY_API_KEY", "synthetic-key")

    actual = giphy.fetch_giphy_pool("gm", logger=logging.getLogger("test.giphy"))

    assert actual == [f"https://example.test/{index}.gif" for index in range(4)]
    assert rust.calls == [
        ("synthetic-key", term, offset)
        for term, offset in zip(giphy.GIPHY_GM_TERMS, [3, 5, 8, 13], strict=True)
    ]
    python_get.assert_not_called()


def test_known_rust_failure_does_not_repeat_request(monkeypatch, caplog) -> None:
    rust = _FakeRustGiphy(
        [{"status": "transport_error", "kind": "timeout"} for _ in range(4)]
    )
    monkeypatch.setattr(giphy, "_load_rust_giphy_adapter", lambda: rust)
    monkeypatch.setattr(giphy.random, "randint", lambda _start, _end: 7)
    python_get = MagicMock(side_effect=AssertionError("request must not be repeated"))
    monkeypatch.setattr(giphy.http_client, "get", python_get)
    monkeypatch.setenv("GIPHY_API_KEY", "synthetic-key")

    with caplog.at_level(logging.ERROR, logger="test.giphy"):
        actual = giphy.fetch_giphy_pool("gn", logger=logging.getLogger("test.giphy"))

    assert actual == []
    assert caplog.text.count("rust_status=transport_error") == 4
    python_get.assert_not_called()


def test_invalid_bridge_result_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustGiphy([{"status": "unknown"} for _ in range(4)])
    monkeypatch.setattr(giphy, "_load_rust_giphy_adapter", lambda: rust)
    monkeypatch.setattr(giphy.random, "randint", lambda _start, _end: 11)
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {"images": {"original": {"url": "https://example.test/fallback.gif"}}}
        ]
    }
    python_get = MagicMock(return_value=response)
    monkeypatch.setattr(giphy.http_client, "get", python_get)
    monkeypatch.setenv("GIPHY_API_KEY", "synthetic-key")

    with caplog.at_level(logging.ERROR, logger="test.giphy"):
        actual = giphy.fetch_giphy_pool("gm", logger=logging.getLogger("test.giphy"))

    assert actual == ["https://example.test/fallback.gif"] * 4
    assert python_get.call_count == 4
    assert caplog.text.count("using Python fallback") == 4


def test_python_and_rust_translation_match_shared_success_contract() -> None:
    path = Path(__file__).parents[1] / "contracts" / "giphy_adapter.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    success = contract["response_cases"][0]
    expected = success["expected"]
    assert giphy._extract_giphy_urls(json.loads(success["body"])) == expected["urls"]
