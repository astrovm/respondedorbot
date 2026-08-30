"""Build stable market, time, and link context for AI prompts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, tzinfo
import json
import logging
from typing import Any, Protocol, cast

from api.ai.pipeline import base_instructions
from api.core.rust_bridge import load_rust_bridge
from api.i18n import tr
from api.i18n.content import weather_description
from api.i18n.prompts import prompt


class _RustHackerNewsFormatter(Protocol):
    def format_hacker_news_items(
        self,
        input_json: str,
        include_discussion: bool,
        no_data: str,
        comments_label: str,
    ) -> str: ...


logger = logging.getLogger(__name__)


def _load_rust_hacker_news_formatter() -> _RustHackerNewsFormatter | None:
    module = load_rust_bridge("RUST_HACKER_NEWS_ENABLED")
    if module is None:
        return None
    return cast(_RustHackerNewsFormatter, module)


def clean_crypto_data(cryptos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for crypto in cryptos:
        cleaned.append(
            {
                "name": crypto["name"],
                "symbol": crypto["symbol"],
                "slug": crypto["slug"],
                "supply": {
                    "max": crypto["max_supply"],
                    "circulating": crypto["circulating_supply"],
                    "total": crypto["total_supply"],
                    "infinite": crypto["infinite_supply"],
                },
                "quote": {
                    "USD": {
                        "price": crypto["quote"]["USD"]["price"],
                        "volume_24h": crypto["quote"]["USD"]["volume_24h"],
                        "changes": {
                            "1h": crypto["quote"]["USD"]["percent_change_1h"],
                            "24h": crypto["quote"]["USD"]["percent_change_24h"],
                            "7d": crypto["quote"]["USD"]["percent_change_7d"],
                            "30d": crypto["quote"]["USD"]["percent_change_30d"],
                        },
                        "market_cap": crypto["quote"]["USD"]["market_cap"],
                        "dominance": crypto["quote"]["USD"]["market_cap_dominance"],
                    }
                },
            }
        )
    return cleaned


def get_weather_description(code: int) -> str:
    return weather_description(code) or tr("weather.unusual")


def _normalize_hacker_news_items(news: Iterable[object]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in news:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "title": str(item.get("title") or f"({tr('news.no_title')})").strip(),
                "url": str(item.get("url") or "").strip(),
                "points": item.get("points") if isinstance(item.get("points"), int) else None,
                "comments": (
                    item.get("comments") if isinstance(item.get("comments"), int) else None
                ),
                "comments_url": str(item.get("comments_url") or "").strip(),
            }
        )
    return items


def _format_hacker_news_python(
    items: list[dict[str, Any]],
    *,
    include_discussion: bool,
    no_data: str,
    comments_label: str,
) -> str:
    lines: list[str] = []
    for item in items:
        stats: list[str] = []
        if item["points"] is not None:
            stats.append(f"{item['points']} pts")
        if item["comments"] is not None:
            stats.append(f"{item['comments']} {comments_label}")

        stats_text = f" ({', '.join(stats)})" if stats else ""
        entry = f"- {item['title']}{stats_text}"
        if item["url"]:
            entry += f" → {item['url']}"
        if include_discussion and item["comments_url"]:
            entry += f" (HN: {item['comments_url']})"
        lines.append(entry)
    return "\n".join(lines) if lines else f"- {no_data}"


def format_hacker_news_info(
    news: Iterable[object] | None,
    include_discussion: bool = True,
) -> str:
    if not news:
        return f"- {tr('news.no_data')}"

    items = _normalize_hacker_news_items(news)

    no_data = tr("news.no_data")
    comments_label = tr("news.comments")
    rust = _load_rust_hacker_news_formatter()
    if rust is not None:
        try:
            return rust.format_hacker_news_items(
                json.dumps(items, separators=(",", ":")),
                include_discussion,
                no_data,
                comments_label,
            )
        except Exception:
            logger.exception("Rust Hacker News formatting failed; using Python")

    return _format_hacker_news_python(
        items,
        include_discussion=include_discussion,
        no_data=no_data,
        comments_label=comments_label,
    )


def format_weather_info(weather: dict[str, Any]) -> str:
    visibility_km = weather.get("visibility")
    visibility = (
        f"{visibility_km / 1000:.1f}km" if visibility_km is not None else tr("weather.no_data")
    )
    location = weather.get("location")
    location_line = f"- {tr('weather.location', value=location)}\n" if location else ""
    return (
        f"{location_line}"
        f"- {tr('weather.apparent', value=weather.get('apparent_temperature', '?'))}\n"
        f"- {tr('weather.rain', value=weather.get('precipitation_probability', '?'))}\n"
        f"- {tr('weather.condition', value=weather.get('description', tr('weather.no_data')))}\n"
        f"- {tr('weather.clouds', value=weather.get('cloud_cover', '?'))}\n"
        f"- {tr('weather.visibility', value=visibility)}"
    )


def _message_context_parts(
    *,
    chat_type: str,
    chat_title: str,
    first_name: str,
    username: str,
    current_time: datetime,
) -> list[str]:
    return [
        prompt("context.header"),
        prompt(
            "context.chat", chat_type=chat_type, chat_title=f" ({chat_title})" if chat_title else ""
        ),
        prompt(
            "context.user", first_name=first_name, username=f" ({username})" if username else ""
        ),
        prompt("context.time", time=current_time.strftime("%H:%M")),
    ]


def build_ai_messages(
    message: dict[str, Any],
    chat_history: list[dict[str, Any]],
    message_text: str,
    *,
    reply_context: str | None,
    enable_web_search: bool,
    summary_text: str | None,
    retrieved_messages: list[dict[str, Any]] | None,
    timezone_offset: int,
    make_timezone: Callable[[int], tzinfo],
    truncate_text: Callable[[str | None], str],
    build_links_context: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if summary_text:
        messages.append(
            {
                "role": "system",
                "content": prompt("context.summary", summary=summary_text),
            }
        )

    if retrieved_messages:
        retrieval_lines = [prompt("context.retrieved")]
        for item in retrieved_messages:
            role = str(item.get("role") or "user")
            text = str(item.get("text") or "")
            if text:
                retrieval_lines.append(f"- {role}: {text}")
        if len(retrieval_lines) > 1:
            messages.append({"role": "system", "content": "\n".join(retrieval_lines)})

    for history_message in chat_history:
        messages.append(
            {
                "role": history_message["role"],
                "content": [{"type": "text", "text": history_message["text"]}],
            }
        )

    sender = message.get("from", {})
    chat = message.get("chat", {})
    first_name = str(sender.get("first_name") or prompt("context.anonymous_user"))
    username = str(sender.get("username") or "")
    chat_type = str(chat.get("type") or "private")
    chat_title = str(chat.get("title") or "") if chat_type != "private" else ""
    current_time = datetime.now(make_timezone(timezone_offset))

    context_parts = _message_context_parts(
        chat_type=chat_type,
        chat_title=chat_title,
        first_name=first_name,
        username=username,
        current_time=current_time,
    )

    if reply_context and not (messages and messages[-1].get("role") == "assistant"):
        context_parts.extend(["", prompt("context.reply"), truncate_text(reply_context)])

    link_context = build_links_context(message)
    if link_context:
        context_parts.extend(["", link_context])

    instructions = [""] + base_instructions()
    if enable_web_search:
        instructions.append(prompt("context.web_search"))

    context_parts.extend(
        ["", prompt("context.message"), truncate_text(message_text)] + instructions
    )
    messages.append({"role": "user", "content": "\n".join(context_parts)})
    return messages
