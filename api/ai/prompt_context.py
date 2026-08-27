"""Build stable market, time, and link context for AI prompts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, tzinfo
from typing import Any

from api.ai.pipeline import base_instructions
from api.i18n import current_locale, tr


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
    descriptions = {
        0: "despejado",
        1: "mayormente despejado",
        2: "parcialmente nublado",
        3: "nublado",
        45: "neblina",
        48: "niebla",
        51: "llovizna leve",
        53: "llovizna moderada",
        55: "llovizna intensa",
        56: "llovizna helada leve",
        57: "llovizna helada intensa",
        61: "lluvia leve",
        63: "lluvia moderada",
        65: "lluvia intensa",
        66: "lluvia helada leve",
        67: "lluvia helada intensa",
        71: "nevada leve",
        73: "nevada moderada",
        75: "nevada intensa",
        77: "granizo",
        80: "lluvia leve intermitente",
        81: "lluvia moderada intermitente",
        82: "lluvia fuerte intermitente",
        85: "nevada leve intermitente",
        86: "nevada intensa intermitente",
        95: "tormenta",
        96: "tormenta con granizo leve",
        99: "tormenta con granizo intenso",
    }
    if current_locale() == "en":
        english = {
            0: "clear",
            1: "mostly clear",
            2: "partly cloudy",
            3: "cloudy",
            45: "foggy",
            48: "freezing fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "heavy drizzle",
            56: "light freezing drizzle",
            57: "heavy freezing drizzle",
            61: "light rain",
            63: "moderate rain",
            65: "heavy rain",
            66: "light freezing rain",
            67: "heavy freezing rain",
            71: "light snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "light rain showers",
            81: "moderate rain showers",
            82: "heavy rain showers",
            85: "light snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with light hail",
            99: "thunderstorm with heavy hail",
        }
        return english.get(code, tr("weather.unusual"))
    return descriptions.get(code, tr("weather.unusual"))


def format_hacker_news_info(
    news: Iterable[object] | None,
    include_discussion: bool = True,
) -> str:
    if not news:
        return f"- {tr('news.no_data')}"

    lines: list[str] = []
    for item in news:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or f"({tr('news.no_title')})").strip()
        url = str(item.get("url") or "").strip()
        stats: list[str] = []
        if isinstance(item.get("points"), int):
            stats.append(f"{item['points']} pts")
        if isinstance(item.get("comments"), int):
            stats.append(f"{item['comments']} {tr('news.comments')}")

        stats_text = f" ({', '.join(stats)})" if stats else ""
        entry = f"- {title}{stats_text}"
        if url:
            entry += f" → {url}"
        if include_discussion:
            hn_url = str(item.get("comments_url") or "").strip()
            if hn_url:
                entry += f" (HN: {hn_url})"
        lines.append(entry)

    return "\n".join(lines) if lines else f"- {tr('news.no_data')}"


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
    if current_locale() == "en":
        return [
            "CONTEXT:",
            f"- Chat: {chat_type}" + (f" ({chat_title})" if chat_title else ""),
            f"- User: {first_name}" + (f" ({username})" if username else ""),
            f"- Time: {current_time.strftime('%H:%M')}",
        ]
    return [
        "CONTEXTO:",
        f"- Chat: {chat_type}" + (f" ({chat_title})" if chat_title else ""),
        f"- Usuario: {first_name}" + (f" ({username})" if username else ""),
        f"- Hora: {current_time.strftime('%H:%M')}",
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
                "content": f"RESUMEN ACUMULADO DEL CHAT:\n{summary_text}",
            }
        )

    if retrieved_messages:
        retrieval_lines = [
            "RELEVANT EARLIER MESSAGES:"
            if current_locale() == "en"
            else "MENSAJES ANTERIORES RELEVANTES:"
        ]
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
    first_name = str(
        sender.get("first_name") or ("User" if current_locale() == "en" else "Usuario")
    )
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
        reply_label = (
            "MESSAGE BEING REPLIED TO:" if current_locale() == "en" else "MENSAJE AL QUE RESPONDE:"
        )
        context_parts.extend(["", reply_label, truncate_text(reply_context)])

    link_context = build_links_context(message)
    if link_context:
        context_parts.extend(["", link_context])

    instructions = (
        [
            "",
            "INSTRUCTIONS:",
            "- keep the bot persona",
            "- use casual English",
            "- answer without emojis or a final period",
            "- use one sentence unless a complex explanation needs more structure",
        ]
        if current_locale() == "en"
        else [""] + base_instructions()
    )
    if enable_web_search:
        instructions.append(
            "- use web search when you are unsure about a current fact"
            if current_locale() == "en"
            else "- si no estás seguro de algo podes buscarlo en internet"
        )

    message_label = "MESSAGE:" if current_locale() == "en" else "MENSAJE:"
    context_parts.extend(["", message_label, truncate_text(message_text)] + instructions)
    messages.append({"role": "user", "content": "\n".join(context_parts)})
    return messages
