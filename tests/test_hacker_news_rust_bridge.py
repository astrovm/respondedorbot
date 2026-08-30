from __future__ import annotations

import json

import pytest

from api.ai import prompt_context
from api.markets import hacker_news


class _FakeRustHackerNews:
    def normalize_hacker_news_item(self, title: str, url: str, description: str) -> str:
        return json.dumps(
            {
                "title": title,
                "url": url,
                "points": 7,
                "comments": 3,
                "comments_url": "https://news.ycombinator.com/item?id=7",
            }
        )

    def format_hacker_news_items(
        self,
        input_json: str,
        include_discussion: bool,
        no_data: str,
        comments_label: str,
    ) -> str:
        items = json.loads(input_json)
        return f"rust:{items[0]['title']}:{include_discussion}:{no_data}:{comments_label}"


def test_feed_parser_uses_rust_item_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hacker_news, "_load_rust_hacker_news", _FakeRustHackerNews)
    result = hacker_news._parse_feed(
        """<rss><channel><item><title>Synthetic story</title><link>https://example.test/story</link><description>ignored</description></item></channel></rss>""",
        max_items=10,
    )

    assert result == [
        {
            "title": "Synthetic story",
            "url": "https://example.test/story",
            "points": 7,
            "comments": 3,
            "comments_url": "https://news.ycombinator.com/item?id=7",
        }
    ]


def test_feed_parser_falls_back_when_rust_normalization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRust:
        def normalize_hacker_news_item(self, *_args: str) -> str:
            raise ValueError("synthetic bridge failure")

    monkeypatch.setattr(hacker_news, "_load_rust_hacker_news", FailingRust)
    result = hacker_news._parse_feed(
        """<rss><channel><item><title>Synthetic story</title><description>Points: 11&lt;br&gt;# Comments: 4</description></item></channel></rss>""",
        max_items=10,
    )

    assert result[0]["points"] == 11
    assert result[0]["comments"] == 4


def test_hacker_news_formatter_uses_localized_rust_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prompt_context,
        "_load_rust_hacker_news_formatter",
        _FakeRustHackerNews,
    )

    result = prompt_context.format_hacker_news_info([{"title": "Synthetic story"}])

    assert result == "rust:Synthetic story:True:sin datos por ahora:coms"


def test_hacker_news_formatter_falls_back_for_bridge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRust:
        def format_hacker_news_items(self, *_args: object) -> str:
            raise ValueError("synthetic bridge failure")

    monkeypatch.setattr(
        prompt_context,
        "_load_rust_hacker_news_formatter",
        FailingRust,
    )
    result = prompt_context.format_hacker_news_info(
        [{"title": "Synthetic story", "points": 5}],
        include_discussion=False,
    )

    assert result == "- Synthetic story (5 pts)"
