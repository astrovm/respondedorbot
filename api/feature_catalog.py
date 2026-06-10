"""Single source of truth for user-visible bot capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from api.command_registry import COMMAND_GROUPS

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


COMMAND_DESCRIPTIONS: Dict[str, str] = {
    "ask": "te contesto cualquier gilada",
    "pregunta": "te contesto cualquier gilada",
    "che": "te contesto cualquier gilada",
    "gordo": "te contesto cualquier gilada",
    "config": "tocás la config del gordo y de los links",
    "convertbase": "te paso números entre bases",
    "random": "elijo por vos entre opciones o números",
    "prices": "precios crypto [1h/24h/7d/30d]",
    "price": "precios crypto [1h/24h/7d/30d]",
    "precios": "precios crypto [1h/24h/7d/30d]",
    "precio": "precios crypto [1h/24h/7d/30d]",
    "presios": "precios crypto [1h/24h/7d/30d]",
    "presio": "precios crypto [1h/24h/7d/30d]",
    "bresio": "precios crypto [1h/24h/7d/30d]",
    "bresios": "precios crypto [1h/24h/7d/30d]",
    "brecio": "precios crypto [1h/24h/7d/30d]",
    "brecios": "precios crypto [1h/24h/7d/30d]",
    "crypto": "precios crypto [1h/24h/7d/30d]",
    "criptos": "precios crypto [1h/24h/7d/30d]",
    "dolar": "cotizaciones del dolar [1h/6h/12h/24h/48h]",
    "dollar": "cotizaciones del dolar [1h/6h/12h/24h/48h]",
    "usd": "cotizaciones del dolar [1h/6h/12h/24h/48h]",
    "petroleo": "te paso el precio del Brent y del WTI",
    "oil": "te paso el precio del Brent y del WTI",
    "acciones": "precios de acciones [aapl tsla googl]",
    "stocks": "precios de acciones [aapl tsla googl]",
    "eleccion": "top 10 de elecciones globales en Polymarket por liquidez",
    "elecciones": "top 10 de elecciones globales en Polymarket por liquidez",
    "election": "top 10 de elecciones globales en Polymarket por liquidez",
    "elections": "top 10 de elecciones globales en Polymarket por liquidez",
    "rulo": "te armo los rulos desde el oficial",
    "devo": "te calculo el arbitraje entre tarjeta y crypto",
    "powerlaw": "te tiro el precio justo de btc según power law",
    "rainbow": "te tiro el precio justo de btc según rainbow chart",
    "satoshi": "te digo cuánto vale un satoshi",
    "sat": "te digo cuánto vale un satoshi",
    "sats": "te digo cuánto vale un satoshi",
    "time": "timestamp unix actual",
    "comando": "te lo convierto en comando de telegram",
    "command": "te lo convierto en comando de telegram",
    "instance": "nombre de esta instancia del bot",
    "help": "te muestro todos los comandos",
    "transcribe": "te transcribo audio o describo imagen",
    "describe": "te transcribo audio o describo imagen",
    "bcra": "te tiro las variables económicas del bcra",
    "variables": "te tiro las variables económicas del bcra",
    "topup": "cargás créditos IA con Telegram Stars por privado",
    "balance": "te muestro tu saldo IA",
    "transfer": "le pasás créditos tuyos al grupo",
    "gm": "gif de buenos días",
    "gn": "gif de buenas noches",
    "tareas": "listado de tareas programadas",
    "tasks": "listado de tareas programadas",
    "resumen": "resumí la conversación [enfoque opcional]",
    "summary": "resumí la conversación [enfoque opcional]",
    "tldr": "resumí la conversación [enfoque opcional]",
}


FEATURES: tuple[FeatureEntry, ...] = (
    FeatureEntry(
        "chat ia",
        "te contesto mensajes normales; en grupos respondo si me mencionan, me responden, usan trigger random o mandan comando ia",
        ("/ask", "/pregunta", "/che", "/gordo"),
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
        (
            "/prices",
            "/price",
            "/precios",
            "/precio",
            "/presios",
            "/presio",
            "/bresio",
            "/bresios",
            "/brecio",
            "/brecios",
            "/crypto",
            "/criptos",
        ),
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
        ("/dolar", "/dollar", "/usd"),
        ("/usd 1h",),
        "mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "acciones",
        "precios de acciones desde Yahoo Finance",
        ("/acciones", "/stocks"),
        ("/acciones aapl tsla googl",),
        "mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "petróleo",
        "precio Brent y WTI",
        ("/petroleo", "/oil"),
        category="mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "bcra",
        "variables económicas del BCRA",
        ("/bcra", "/variables"),
        category="mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "elección",
        "top 10 de elecciones globales en Polymarket por liquidez",
        ("/eleccion", "/elecciones", "/election", "/elections"),
        category="mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "arbitrajes",
        "rulo desde oficial, arbitraje tarjeta/crypto, power law, rainbow chart y sats",
        ("/rulo", "/devo", "/powerlaw", "/rainbow", "/satoshi", "/sat", "/sats"),
        ("/devo 0.5, 100",),
        "mercado",
        telegram_visible=True,
    ),
    FeatureEntry(
        "media",
        "transcribo voice/audio/video/video_note y describo fotos o stickers respondiendo al mensaje; también puedo procesar media cuando me hablan",
        ("/transcribe", "/describe"),
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
        ("/tareas", "/tasks"),
        examples=("mañana recordame pagar el alquiler", "/tareas"),
        category="productividad",
        telegram_visible=True,
    ),
    FeatureEntry(
        "resúmenes y memoria",
        "resumo el chat, guardo resumen acumulado y recupero mensajes relevantes para responder con contexto",
        ("/resumen", "/summary", "/tldr"),
        ("/resumen focus en crypto",),
        "memoria",
        telegram_visible=True,
    ),
    FeatureEntry(
        "utilidades",
        "random, conversión de bases, comandos Telegram, timestamp e instancia",
        ("/random", "/convertbase", "/comando", "/command", "/time", "/instance"),
        ("/random pizza, carne, sushi", "/convertbase 101, 2, 10"),
        "utilidades",
        telegram_visible=True,
    ),
    FeatureEntry(
        "gifs",
        "gif random de buenos días o buenas noches",
        ("/gm", "/gn"),
        category="utilidades",
        telegram_visible=True,
    ),
    FeatureEntry(
        "config",
        "config por chat: links, followups, replies a links arreglados, timezone, random replies y límite gratis por usuario/hora",
        ("/config",),
        category="admin",
        telegram_visible=True,
    ),
    FeatureEntry(
        "créditos ia",
        "saldo, topup con Telegram Stars y transferencia de créditos personales al grupo",
        ("/topup", "/balance", "/transfer"),
        ("/transfer 1.5",),
        "créditos",
        telegram_visible=True,
    ),
    FeatureEntry(
        "admin créditos",
        "mint y log de créditos, solo admin",
        ("/printcredits", "/creditlog"),
        category="admin",
        help_visible=False,
        telegram_visible=False,
        admin_only=True,
    ),
    FeatureEntry(
        "help",
        "muestro comandos y features",
        ("/help",),
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
