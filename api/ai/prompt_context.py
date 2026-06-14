"""Build stable market, time, and link context for AI prompts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, tzinfo
from typing import Any

from api.ai.pipeline import INSTRUCCIONES_BASE


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
                        "dominance": crypto["quote"]["USD"][
                            "market_cap_dominance"
                        ],
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
    return descriptions.get(code, "clima raro")


def format_hacker_news_info(
    news: Iterable[object] | None,
    include_discussion: bool = True,
) -> str:
    if not news:
        return "- sin datos por ahora"

    lines: list[str] = []
    for item in news:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "(sin título)").strip()
        url = str(item.get("url") or "").strip()
        stats: list[str] = []
        if isinstance(item.get("points"), int):
            stats.append(f"{item['points']} pts")
        if isinstance(item.get("comments"), int):
            stats.append(f"{item['comments']} coms")

        stats_text = f" ({', '.join(stats)})" if stats else ""
        entry = f"- {title}{stats_text}"
        if url:
            entry += f" → {url}"
        if include_discussion:
            hn_url = str(item.get("comments_url") or "").strip()
            if hn_url:
                entry += f" (HN: {hn_url})"
        lines.append(entry)

    return "\n".join(lines) if lines else "- sin datos por ahora"


def format_weather_info(weather: dict[str, Any]) -> str:
    visibility_km = weather.get("visibility")
    visibility = (
        f"{visibility_km / 1000:.1f}km"
        if visibility_km is not None
        else "sin datos"
    )
    return (
        f"- Temperatura aparente: {weather.get('apparent_temperature', '?')}°C\n"
        f"- Probabilidad de lluvia: {weather.get('precipitation_probability', '?')}%\n"
        f"- Estado: {weather.get('description', 'sin datos')}\n"
        f"- Nubosidad: {weather.get('cloud_cover', '?')}%\n"
        f"- Visibilidad: {visibility}"
    )


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
        retrieval_lines = ["MENSAJES ANTERIORES RELEVANTES:"]
        for item in retrieved_messages:
            role = str(item.get("role") or "user")
            text = str(item.get("text") or "")
            if text:
                retrieval_lines.append(f"- {role}: {text}")
        if len(retrieval_lines) > 1:
            messages.append(
                {"role": "system", "content": "\n".join(retrieval_lines)}
            )

    for history_message in chat_history:
        messages.append(
            {
                "role": history_message["role"],
                "content": [
                    {"type": "text", "text": history_message["text"]}
                ],
            }
        )

    sender = message.get("from", {})
    chat = message.get("chat", {})
    first_name = str(sender.get("first_name") or "Usuario")
    username = str(sender.get("username") or "")
    chat_type = str(chat.get("type") or "private")
    chat_title = str(chat.get("title") or "") if chat_type != "private" else ""
    current_time = datetime.now(make_timezone(timezone_offset))

    context_parts = [
        "CONTEXTO:",
        f"- Chat: {chat_type}" + (f" ({chat_title})" if chat_title else ""),
        f"- Usuario: {first_name}" + (f" ({username})" if username else ""),
        f"- Hora: {current_time.strftime('%H:%M')}",
    ]

    if (
        reply_context
        and not (messages and messages[-1].get("role") == "assistant")
    ):
        context_parts.extend(
            ["", "MENSAJE AL QUE RESPONDE:", truncate_text(reply_context)]
        )

    link_context = build_links_context(message)
    if link_context:
        context_parts.extend(["", link_context])

    instructions = [""] + INSTRUCCIONES_BASE[:]
    if enable_web_search:
        instructions.append("- si no estás seguro de algo podes buscarlo en internet")

    context_parts.extend(
        ["", "MENSAJE:", truncate_text(message_text)] + instructions
    )
    messages.append({"role": "user", "content": "\n".join(context_parts)})
    return messages
