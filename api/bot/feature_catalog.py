"""Single source of truth for user-visible bot capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from api.bot.command_registry import (
    aliases_for,
    COMMAND_DESCRIPTIONS,
    COMMAND_GROUPS,
    command_descriptions,
)
from api.core.i18n import current_locale

CommandGroup = tuple[tuple[str, ...], str, bool, bool]


@dataclass(frozen=True)
class FeatureEntry:
    title: str
    description: str
    commands: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    category: str = "general"
    help_visible: bool = True
    telegram_visible: bool = False
    ai_visible: bool = True
    admin_only: bool = False
    implicit: bool = False


FEATURES: tuple[FeatureEntry, ...] = (
    FeatureEntry(
        "chat ia",
        "te contesto mensajes normales; en grupos respondo si me mencionan, me responden, usan trigger random o mandan comando ia",
        aliases_for("ask_ai"),
        ("/gordo explicame esto",),
        "ia",
        telegram_visible=True,
    ),
    FeatureEntry(
        "búsqueda web nativa",
        "en mensajes normales puedo buscar en internet cuando hace falta",
        examples=("buscá qué pasó con...",),
        category="ia",
        implicit=True,
    ),
    FeatureEntry(
        "crypto prices",
        "precios crypto por ranking, símbolo, moneda base y variación",
        aliases_for("get_prices"),
        (
            "/prices btc eth xmr",
            "/prices 20",
            "/prices 100 in eur",
            "/prices btc 7d",
            "/prices stables",
        ),
        "mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "clima",
        "clima actual para cualquier ciudad o ubicación",
        aliases_for("get_weather"),
        ("/clima Córdoba, Argentina",),
        "general",
        telegram_visible=True,
    ),
    FeatureEntry(
        "token cards",
        "si el mensaje completo es un address Solana/EVM o un $ticker, mando card con chart/imagen, stats, socials, links y botones",
        examples=("J8PS...pump", "$GLORP"),
        category="mercado",
        implicit=True,
    ),
    FeatureEntry(
        "dólar",
        "cotizaciones del dólar y variaciones por ventana",
        aliases_for("get_dollar_rates"),
        ("/usd 1h",),
        "mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "acciones",
        "precios de acciones por símbolo o empresa desde Yahoo Finance",
        aliases_for("get_stock_prices"),
        ("/acciones aapl tsla", "/acciones Mercado Libre"),
        "mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "petróleo",
        "precio Brent y WTI",
        aliases_for("get_oil_price"),
        category="mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "bcra",
        "variables económicas del BCRA",
        aliases_for("handle_bcra_variables"),
        category="mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "elección",
        "top 10 de elecciones globales en Polymarket por liquidez",
        aliases_for("get_polymarket_global_elections"),
        category="mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "arbitrajes",
        "rulo desde oficial, arbitraje tarjeta/crypto, power law, rainbow chart y sats",
        aliases_for("get_rulo", "get_devo", "powerlaw", "rainbow", "satoshi"),
        ("/devo 0.5, 100",),
        "mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "media",
        "transcribo voice/audio/video/video_note y describo fotos o stickers respondiendo al mensaje; también puedo procesar media cuando me hablan",
        aliases_for("handle_transcribe"),
        category="media",
        telegram_visible=True,
    ),
    FeatureEntry(
        "links",
        "arreglo links de X/Twitter, Bluesky, Instagram y Reddit según config; leo metadata, tweets y transcripts de YouTube como contexto",
        category="links",
        implicit=True,
    ),
    FeatureEntry(
        "tareas",
        "agendo recordatorios y tareas recurrentes por lenguaje natural; cualquiera de los comandos lista sin texto y crea con texto",
        aliases_for("task_command"),
        examples=("/tarea mañana recordame pagar el alquiler", "/tasks"),
        category="productividad",
        telegram_visible=True,
    ),
    FeatureEntry(
        "resúmenes y memoria",
        "resumo el chat, guardo resumen acumulado y recupero mensajes relevantes para responder con contexto",
        aliases_for("summary_command"),
        ("/resumen focus en crypto",),
        "memoria",
        telegram_visible=True,
    ),
    FeatureEntry(
        "utilidades",
        "random, conversión de bases, comandos Telegram, timestamp e instancia",
        aliases_for(
            "select_random",
            "convert_base",
            "convert_to_command",
            "get_timestamp",
            "get_instance_name",
        ),
        ("/random pizza, carne, sushi", "/convertbase 101, 2, 10"),
        "utilidades",
        telegram_visible=True,
    ),
    FeatureEntry(
        "gifs",
        "gif random de buenos días o buenas noches",
        aliases_for("get_good_morning", "get_good_night"),
        category="utilidades",
        telegram_visible=True,
    ),
    FeatureEntry(
        "config",
        "config por chat: idioma, links, followups, timezone, random replies y límite gratis por usuario/hora",
        aliases_for("config_command"),
        category="admin",
        telegram_visible=True,
    ),
    FeatureEntry(
        "idioma",
        "cambiá entre español e inglés",
        aliases_for("language_command"),
        category="config",
        telegram_visible=True,
    ),
    FeatureEntry(
        "créditos ia",
        "saldo, historial de gastos, topup con Telegram Stars y transferencia de créditos personales al grupo",
        aliases_for(
            "topup_command",
            "balance_command",
            "charges_command",
            "transfer_command",
        ),
        ("/charges 10", "/transfer 1.5"),
        "créditos",
        telegram_visible=True,
    ),
    FeatureEntry(
        "admin créditos",
        "mint y log de créditos, solo admin",
        aliases_for("printcredits_command", "creditlog_command"),
        category="admin",
        help_visible=False,
        telegram_visible=False,
        admin_only=True,
    ),
    FeatureEntry(
        "help",
        "muestro comandos y features",
        aliases_for("get_help"),
        category="utilidades",
        telegram_visible=True,
    ),
)

_CATEGORY_EN = {
    "ia": "AI",
    "mercado": "markets",
    "general": "general",
    "media": "media",
    "links": "links",
    "productividad": "productivity",
    "memoria": "memory",
    "utilidades": "utilities",
    "config": "settings",
    "créditos": "credits",
    "admin": "admin",
}

_FEATURE_EN: dict[str, tuple[str, str]] = {
    "chat ia": (
        "AI chat",
        "I answer normal messages; in groups I respond to mentions, replies, random triggers, and AI commands",
    ),
    "búsqueda web nativa": ("web search", "I can search the web when a current answer needs it"),
    "crypto prices": (
        "crypto prices",
        "crypto prices by ranking, symbol, base currency, and time window",
    ),
    "clima": ("weather", "current weather for any city or location"),
    "token cards": ("token cards", "send a Solana or EVM address, or a $ticker, for a market card"),
    "dólar": ("dollar", "dollar exchange rates and changes by time window"),
    "acciones": ("stocks", "stock prices by symbol or company from Yahoo Finance"),
    "petróleo": ("oil", "Brent and WTI oil prices"),
    "bcra": ("BCRA", "economic variables from Argentina's central bank"),
    "elección": ("elections", "top global election markets on Polymarket by liquidity"),
    "arbitrajes": (
        "arbitrage",
        "official-rate, card/crypto, power-law, rainbow-chart, and satoshi tools",
    ),
    "media": ("media", "transcribe audio and video or describe images and stickers"),
    "links": ("links", "fix supported social links and read linked content as context"),
    "tareas": ("tasks", "create reminders and recurring tasks with natural language"),
    "resúmenes y memoria": (
        "summaries and memory",
        "summarize chats and retrieve relevant prior messages",
    ),
    "utilidades": (
        "utilities",
        "random selection, base conversion, Telegram commands, timestamps, and instance info",
    ),
    "gifs": ("GIFs", "random good-morning and good-night GIFs"),
    "config": (
        "settings",
        "all chat settings, including language, links, timezone, replies, and group limits",
    ),
    "idioma": ("language", "switch between Spanish and English"),
    "créditos ia": (
        "AI credits",
        "balance, expense history, Telegram Stars top-ups, and group transfers",
    ),
    "admin créditos": ("credit admin", "mint and inspect credits; admin only"),
    "help": ("help", "show commands and features"),
}


def _strip_slash(command: str) -> str:
    return command.lstrip("/")


def command_aliases(command_groups: Sequence[CommandGroup] = COMMAND_GROUPS) -> set[str]:
    return {
        _strip_slash(alias)
        for aliases, _handler_name, _uses_ai, _takes_params in command_groups
        for alias in aliases
    }


def catalog_command_aliases(entries: Iterable[FeatureEntry] = FEATURES) -> set[str]:
    return {_strip_slash(command) for entry in entries for command in entry.commands}


def get_feature_for_command(command: str) -> Optional[FeatureEntry]:
    normalized = _strip_slash(command)
    for entry in FEATURES:
        if normalized in {_strip_slash(alias) for alias in entry.commands}:
            return entry
    return None


def telegram_command_descriptions(
    *,
    command_groups: Sequence[CommandGroup] = COMMAND_GROUPS,
    descriptions: Mapping[str, str] = COMMAND_DESCRIPTIONS,
    locale: str = "es",
) -> Dict[str, str]:
    if descriptions is COMMAND_DESCRIPTIONS:
        descriptions = command_descriptions(locale)
    allowed = command_aliases(command_groups)
    visible = {
        _strip_slash(command)
        for entry in FEATURES
        if entry.telegram_visible and not entry.admin_only
        for command in entry.commands
    }
    return {
        name: desc for name, desc in descriptions.items() if name in allowed and name in visible
    }


def render_help_text(entries: Iterable[FeatureEntry] = FEATURES) -> str:
    locale = current_locale()
    lines = ["what I can do:" if locale == "en" else "esto es lo que sé hacer, boludo:", ""]
    current_category = ""
    for entry in entries:
        if not entry.help_visible or entry.admin_only:
            continue
        category = (
            _CATEGORY_EN.get(entry.category, entry.category) if locale == "en" else entry.category
        )
        if category != current_category:
            if current_category:
                lines.append("")
            current_category = category
            lines.append(f"{category}:")
        title, description = (
            _FEATURE_EN.get(entry.title, (entry.title, entry.description))
            if locale == "en"
            else (entry.title, entry.description)
        )
        prefix = ", ".join(entry.commands) if entry.commands else title
        lines.append(f"- {prefix}: {description}")
        for example in entry.examples:
            lines.append(f"  {'example' if locale == 'en' else 'ejemplo'}: {example}")
    return "\n".join(lines).strip()


def render_ai_capabilities_prompt(entries: Iterable[FeatureEntry] = FEATURES) -> str:
    locale = current_locale()
    lines = (
        [
            "BOT CAPABILITIES:",
            "- if the user asks what you can do, answer from this list",
            "- do not invent commands; /buscar and /search do not exist",
            "- when an exact command exists, suggest that exact command",
        ]
        if locale == "en"
        else [
            "CAPACIDADES DEL BOT:",
            "- si el usuario pregunta que podes hacer, responde desde esta lista",
            "- no inventes comandos; /buscar y /search no existen",
            "- si existe comando exacto para algo, sugerilo con el comando exacto",
        ]
    )
    for entry in entries:
        if not entry.ai_visible:
            continue
        title, description = (
            _FEATURE_EN.get(entry.title, (entry.title, entry.description))
            if locale == "en"
            else (entry.title, entry.description)
        )
        label = ", ".join(entry.commands) if entry.commands else title
        if entry.admin_only:
            label = f"{label} ({'admin only' if locale == 'en' else 'solo admin'})"
        lines.append(f"- {label}: {description}")
    return "\n".join(lines)


__all__ = [
    "COMMAND_DESCRIPTIONS",
    "FEATURES",
    "FeatureEntry",
    "catalog_command_aliases",
    "command_aliases",
    "get_feature_for_command",
    "render_ai_capabilities_prompt",
    "render_help_text",
    "telegram_command_descriptions",
]
