"""Command registration and message routing helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from os import environ
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

CommandHandler = Callable[..., Any]
CommandTuple = Tuple[CommandHandler, bool, bool]


@dataclass(frozen=True)
class CommandDefinition:
    aliases: Tuple[str, ...]
    handler_name: str
    uses_ai: bool
    takes_params: bool
    description: Optional[str] = None


COMMAND_DEFINITIONS: Tuple[CommandDefinition, ...] = (
    CommandDefinition(("/ask", "/pregunta", "/che", "/gordo"), "ask_ai", True, True, "te contesto cualquier gilada"),
    CommandDefinition(("/config",), "config_command", False, False, "tocás la config del gordo y de los links"),
    CommandDefinition(("/convertbase",), "convert_base", False, True, "te paso números entre bases"),
    CommandDefinition(("/random",), "select_random", False, True, "elijo por vos entre opciones o números"),
    CommandDefinition(
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
        "get_prices",
        False,
        True,
        "precios crypto [1h/24h/7d/30d]",
    ),
    CommandDefinition(("/dolar", "/dollar", "/usd"), "get_dollar_rates", False, True, "cotizaciones del dolar [1h/6h/12h/24h/48h]"),
    CommandDefinition(("/petroleo", "/oil"), "get_oil_price", False, False, "te paso el precio del Brent y del WTI"),
    CommandDefinition(("/acciones", "/stocks"), "get_stock_prices", False, True, "precios de acciones [aapl tsla googl]"),
    CommandDefinition(
        ("/eleccion", "/elecciones", "/election", "/elections"),
        "get_polymarket_global_elections",
        False,
        False,
        "top 10 de elecciones globales en Polymarket por liquidez",
    ),
    CommandDefinition(("/mundial", "/worldcup"), "get_polymarket_world_cup_games", False, False, "próximos 10 partidos del Mundial en Polymarket"),
    CommandDefinition(("/rulo",), "get_rulo", False, False, "te armo los rulos desde el oficial"),
    CommandDefinition(("/devo",), "get_devo", False, True, "te calculo el arbitraje entre tarjeta y crypto"),
    CommandDefinition(("/powerlaw",), "powerlaw", False, False, "te tiro el precio justo de btc según power law"),
    CommandDefinition(("/rainbow",), "rainbow", False, False, "te tiro el precio justo de btc según rainbow chart"),
    CommandDefinition(("/satoshi", "/sat", "/sats"), "satoshi", False, False, "te digo cuánto vale un satoshi"),
    CommandDefinition(("/time",), "get_timestamp", False, False, "timestamp unix actual"),
    CommandDefinition(("/comando", "/command"), "convert_to_command", False, True, "te lo convierto en comando de telegram"),
    CommandDefinition(("/instance",), "get_instance_name", False, False, "nombre de esta instancia del bot"),
    CommandDefinition(("/help",), "get_help", False, False, "te muestro todos los comandos"),
    CommandDefinition(("/transcribe", "/describe"), "handle_transcribe", False, False, "te transcribo audio o describo imagen"),
    CommandDefinition(("/bcra", "/variables"), "handle_bcra_variables", False, False, "te tiro las variables económicas del bcra"),
    CommandDefinition(("/topup",), "topup_command", False, False, "cargás créditos IA con Telegram Stars por privado"),
    CommandDefinition(("/balance",), "balance_command", False, False, "te muestro tu saldo IA"),
    CommandDefinition(("/printcredits",), "printcredits_command", False, True),
    CommandDefinition(("/creditlog",), "creditlog_command", False, True),
    CommandDefinition(("/transfer",), "transfer_command", False, True, "le pasás créditos tuyos al grupo"),
    CommandDefinition(("/gm",), "get_good_morning", False, False, "gif de buenos días"),
    CommandDefinition(("/gn",), "get_good_night", False, False, "gif de buenas noches"),
    CommandDefinition(("/tareas", "/tasks"), "tasks_command", False, False, "listado de tareas programadas"),
    CommandDefinition(("/resumen", "/summary", "/tldr"), "summary_command", False, True, "resumí la conversación [enfoque opcional]"),
)

COMMAND_GROUPS: Tuple[Tuple[Tuple[str, ...], str, bool, bool], ...] = tuple(
    (
        definition.aliases,
        definition.handler_name,
        definition.uses_ai,
        definition.takes_params,
    )
    for definition in COMMAND_DEFINITIONS
)

COMMAND_DESCRIPTIONS: Dict[str, str] = {
    alias.removeprefix("/"): definition.description
    for definition in COMMAND_DEFINITIONS
    if definition.description is not None
    for alias in definition.aliases
}


def aliases_for(*handler_names: str) -> Tuple[str, ...]:
    requested = set(handler_names)
    return tuple(
        alias
        for definition in COMMAND_DEFINITIONS
        if definition.handler_name in requested
        for alias in definition.aliases
    )


LINK_REPLACEMENT_DOMAINS = (
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "eeinstagram.com",
    "vxinstagram.com",
    "kkinstagram.com",
    "rxddit.com",
)


def build_command_registry(
    handlers: Mapping[str, CommandHandler],
) -> Dict[str, CommandTuple]:
    """Return the public command mapping used by the bot."""

    registry: Dict[str, CommandTuple] = {}
    for definition in COMMAND_DEFINITIONS:
        handler = handlers[definition.handler_name]
        for alias in definition.aliases:
            registry[alias] = (
                handler,
                definition.uses_ai,
                definition.takes_params,
            )
    return registry


def parse_command(message_text: str, bot_name: str) -> Tuple[str, str]:
    """Parse command and message text from input."""

    message_text = message_text.strip()
    if not message_text:
        return "", ""

    split_message = message_text.split(" ", 1)
    command = split_message[0].lower().replace(bot_name, "")

    if command.startswith("/"):
        command_body = command[1:]
        if command_body and all(char == "\u3164" for char in command_body):
            command = "/ask"

    if len(split_message) > 1:
        message_text = split_message[1].lstrip()
    else:
        message_text = ""

    return command, message_text


def should_gordo_respond(
    commands: Mapping[str, CommandTuple],
    command: str,
    message_text: str,
    message: Mapping[str, Any],
    chat_config: Mapping[str, Any],
    reply_metadata: Optional[Mapping[str, Any]],
    *,
    load_bot_config_fn: Callable[[], Mapping[str, Any]],
) -> bool:
    """Decide if the bot should respond to a message."""

    message_lower = message_text.lower()
    chat = message.get("chat") or {}
    chat_type = str(chat.get("type", ""))
    bot_username = environ.get("TELEGRAM_USERNAME")
    bot_name = f"@{bot_username}"

    is_command = command in commands
    reply = message.get("reply_to_message") or {}
    is_reply = (
        isinstance(reply, Mapping)
        and reply.get("from", {}).get("username", "") == bot_username
    )
    ignore_link_fix_followups = bool(chat_config.get("ignore_link_fix_followups", True))
    if not is_command and is_reply and ignore_link_fix_followups:
        reply_text = str(reply.get("text") or "")
        if any(domain in reply_text for domain in LINK_REPLACEMENT_DOMAINS):
            return False

    is_private = chat_type == "private"
    is_mention = bot_name in message_lower

    if (
        not is_command
        and is_reply
        and reply_metadata
        and reply_metadata.get("type") == "command"
        and not bool(reply_metadata.get("uses_ai", False))
        and not bool(chat_config.get("ai_command_followups", True))
    ):
        return False

    try:
        config = load_bot_config_fn()
        trigger_words = list(config.get("trigger_words", ["bot", "assistant"]))
    except ValueError:
        trigger_words = ["bot", "assistant"]

    if bool(chat_config.get("ai_random_replies", True)):
        is_trigger = (
            any(word in message_lower for word in trigger_words)
            and random.random() < 0.1
        )
    else:
        is_trigger = False

    return is_command or (
        not command.startswith("/")
        and (is_trigger or is_private or is_mention or is_reply)
    )


def should_auto_process_media(
    commands: Mapping[str, CommandTuple],
    command: str,
    message_text: str,
    message: Mapping[str, Any],
) -> bool:
    """Return whether incoming media should be auto transcribed/described."""

    chat = message.get("chat") or {}
    chat_type = str(chat.get("type", ""))
    if chat_type == "private":
        return True

    if command in commands:
        return True

    bot_username = str(environ.get("TELEGRAM_USERNAME") or "").strip()
    if not bot_username:
        return False

    bot_name = f"@{bot_username}"
    lowered_text = (message_text or "").lower()
    is_mention = bot_name.lower() in lowered_text

    reply = message.get("reply_to_message") or {}
    reply_from = reply.get("from") if isinstance(reply, Mapping) else {}
    is_reply_to_bot = str((reply_from or {}).get("username", "")) == bot_username

    return is_mention or is_reply_to_bot


__all__ = [
    "aliases_for",
    "COMMAND_DEFINITIONS",
    "COMMAND_DESCRIPTIONS",
    "COMMAND_GROUPS",
    "CommandDefinition",
    "CommandTuple",
    "build_command_registry",
    "parse_command",
    "should_auto_process_media",
    "should_gordo_respond",
]
