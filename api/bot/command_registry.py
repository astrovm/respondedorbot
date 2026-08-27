"""Command registration and message routing helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from os import environ
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from api.i18n import normalize_locale
from api.i18n.content import command_description

CommandHandler = Callable[..., Any]
CommandTuple = Tuple[CommandHandler, bool, bool]


@dataclass(frozen=True)
class CommandDefinition:
    aliases: Tuple[str, ...]
    handler_name: str
    uses_ai: bool
    takes_params: bool
    listed: bool = True


COMMAND_DEFINITIONS: Tuple[CommandDefinition, ...] = (
    CommandDefinition(("/ask", "/pregunta", "/che", "/gordo"), "ask_ai", True, True),
    CommandDefinition(("/config", "/configs", "/settings"), "config_command", False, False),
    CommandDefinition(("/language", "/idioma"), "language_command", False, True),
    CommandDefinition(("/convertbase",), "convert_base", False, True),
    CommandDefinition(("/random",), "select_random", False, True),
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
            "/c",
        ),
        "get_prices",
        False,
        True,
    ),
    CommandDefinition(("/crypto", "/criptos"), "get_crypto_prices", False, True),
    CommandDefinition(("/clima", "/weather"), "get_weather", False, True),
    CommandDefinition(("/dolar", "/dollar", "/usd"), "get_dollar_rates", False, True),
    CommandDefinition(("/petroleo", "/oil"), "get_oil_price", False, False),
    CommandDefinition(("/acciones", "/stocks"), "get_stock_prices", False, True),
    CommandDefinition(
        ("/eleccion", "/elecciones", "/election", "/elections"),
        "get_polymarket_global_elections",
        False,
        False,
    ),
    CommandDefinition(("/rulo",), "get_rulo", False, False),
    CommandDefinition(("/devo",), "get_devo", False, True),
    CommandDefinition(("/powerlaw",), "powerlaw", False, False),
    CommandDefinition(("/rainbow",), "rainbow", False, False),
    CommandDefinition(("/satoshi", "/sat", "/sats"), "satoshi", False, False),
    CommandDefinition(("/time",), "get_timestamp", False, False),
    CommandDefinition(("/comando", "/command"), "convert_to_command", False, True),
    CommandDefinition(("/instance",), "get_instance_name", False, False),
    CommandDefinition(("/help",), "get_help", False, False),
    CommandDefinition(("/transcribe", "/describe"), "handle_transcribe", False, False),
    CommandDefinition(("/bcra", "/variables"), "handle_bcra_variables", False, False),
    CommandDefinition(("/topup",), "topup_command", False, False),
    CommandDefinition(("/balance",), "balance_command", False, False),
    CommandDefinition(("/charges", "/history", "/gastos"), "charges_command", False, True),
    CommandDefinition(("/printcredits",), "printcredits_command", False, True, False),
    CommandDefinition(("/creditlog",), "creditlog_command", False, True, False),
    CommandDefinition(("/transfer",), "transfer_command", False, True),
    CommandDefinition(("/gm",), "get_good_morning", False, False),
    CommandDefinition(("/gn",), "get_good_night", False, False),
    CommandDefinition(("/tarea", "/tareas", "/task", "/tasks"), "task_command", True, True),
    CommandDefinition(("/resumen", "/summary", "/tldr"), "summary_command", False, True),
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
    alias.removeprefix("/"): command_description(definition.handler_name, "es")
    for definition in COMMAND_DEFINITIONS
    if definition.listed
    for alias in definition.aliases
}


def command_descriptions(locale: str = "es") -> Dict[str, str]:
    selected = normalize_locale(locale)
    return {
        alias.removeprefix("/"): command_description(definition.handler_name, selected)
        for definition in COMMAND_DEFINITIONS
        if definition.listed
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
        isinstance(reply, Mapping) and reply.get("from", {}).get("username", "") == bot_username
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
        is_trigger = any(word in message_lower for word in trigger_words) and random.random() < 0.1
    else:
        is_trigger = False

    return is_command or (
        not command.startswith("/") and (is_trigger or is_private or is_mention or is_reply)
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
