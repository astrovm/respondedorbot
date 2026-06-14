"""Single source of truth for user-visible bot capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from api.bot.command_registry import (
    aliases_for,
    COMMAND_DESCRIPTIONS,
    COMMAND_GROUPS,
)

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
        "precios de acciones desde Yahoo Finance",
        aliases_for("get_stock_prices"),
        ("/acciones aapl tsla googl",),
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
        "mundial",
        "próximos 10 partidos del Mundial en Polymarket",
        aliases_for("get_polymarket_world_cup_games"),
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
        "agendo recordatorios y tareas recurrentes por lenguaje natural; /tareas lista y borra con botones",
        aliases_for("tasks_command"),
        examples=("mañana recordame pagar el alquiler", "/tareas"),
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
        "config por chat: links, followups, timezone, goles del Mundial, random replies y límite gratis por usuario/hora",
        aliases_for("config_command"),
        category="admin",
        telegram_visible=True,
    ),
    FeatureEntry(
        "créditos ia",
        "saldo, topup con Telegram Stars y transferencia de créditos personales al grupo",
        aliases_for("topup_command", "balance_command", "transfer_command"),
        ("/transfer 1.5",),
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


def _strip_slash(command: str) -> str:
    return command.lstrip("/")


def command_aliases(command_groups: Sequence[CommandGroup] = COMMAND_GROUPS) -> set[str]:
    return {
        _strip_slash(alias)
        for aliases, _handler_name, _uses_ai, _takes_params in command_groups
        for alias in aliases
    }


def catalog_command_aliases(entries: Iterable[FeatureEntry] = FEATURES) -> set[str]:
    return {
        _strip_slash(command)
        for entry in entries
        for command in entry.commands
    }


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
) -> Dict[str, str]:
    allowed = command_aliases(command_groups)
    visible = {
        _strip_slash(command)
        for entry in FEATURES
        if entry.telegram_visible and not entry.admin_only
        for command in entry.commands
    }
    return {
        name: desc
        for name, desc in descriptions.items()
        if name in allowed and name in visible
    }


def render_help_text(entries: Iterable[FeatureEntry] = FEATURES) -> str:
    lines = ["esto es lo que sé hacer, boludo:", ""]
    current_category = ""
    for entry in entries:
        if not entry.help_visible or entry.admin_only:
            continue
        if entry.category != current_category:
            if current_category:
                lines.append("")
            current_category = entry.category
            lines.append(f"{entry.category}:")
        prefix = ", ".join(entry.commands) if entry.commands else entry.title
        lines.append(f"- {prefix}: {entry.description}")
        for example in entry.examples:
            lines.append(f"  ejemplo: {example}")
    return "\n".join(lines).strip()


def render_ai_capabilities_prompt(entries: Iterable[FeatureEntry] = FEATURES) -> str:
    lines = [
        "CAPACIDADES DEL BOT:",
        "- si el usuario pregunta que podes hacer, responde desde esta lista",
        "- no inventes comandos; /buscar y /search no existen",
        "- si existe comando exacto para algo, sugerilo con el comando exacto",
    ]
    for entry in entries:
        if not entry.ai_visible:
            continue
        label = ", ".join(entry.commands) if entry.commands else entry.title
        if entry.admin_only:
            label = f"{label} (solo admin)"
        lines.append(f"- {label}: {entry.description}")
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
