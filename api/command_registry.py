"""Command registration and message routing helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from os import environ
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CommandHandler = Callable[..., str]
CommandTuple = Tuple[CommandHandler, bool, bool]


@dataclass(frozen=True)
class CommandSpec:
    """Declarative command metadata."""

    handler: CommandHandler
    uses_ai: bool
    takes_params: bool


COMMAND_GROUPS: Tuple[Tuple[Tuple[str, ...], str, bool, bool], ...] = (
    (("/ask", "/pregunta", "/che", "/gordo"), "ask_ai", True, True),
    (("/agent",), "show_agent_thoughts", False, False),
    (("/config",), "config_command", False, False),
    (("/convertbase",), "convert_base", False, True),
    (("/random",), "select_random", False, True),
    (
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
        ),
        "get_prices",
        False,
        True,
    ),
    (("/dolar", "/dollar", "/usd"), "get_dollar_rates", False, False),
    (("/eleccion",), "get_polymarket_argentina_election", False, False),
    (("/rulo",), "get_rulo", False, False),
    (("/devo",), "get_devo", False, True),
    (("/powerlaw",), "powerlaw", False, False),
    (("/rainbow",), "rainbow", False, False),
    (("/satoshi", "/sat", "/sats"), "satoshi", False, False),
    (("/time",), "get_timestamp", False, False),
    (("/comando", "/command"), "convert_to_command", False, True),
    (("/buscar", "/search"), "search_command", False, True),
    (("/instance",), "get_instance_name", False, False),
    (("/help",), "get_help", False, False),
    (("/transcribe",), "handle_transcribe", False, False),
    (("/bcra", "/variables"), "handle_bcra_variables", False, False),
    (("/topup",), "topup_command", False, False),
    (("/balance",), "balance_command", False, False),
    (("/transfer",), "transfer_command", False, True),
)

LINK_REPLACEMENT_DOMAINS = (
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "kkinstagram.com",
    "eeinstagram.com",
    "rxddit.com",
)


def build_command_registry(
    handlers: Mapping[str, CommandHandler],
) -> Dict[str, CommandTuple]:
    """Return the public command mapping expected by legacy callers."""

    registry: Dict[str, CommandTuple] = {}
    for aliases, handler_name, uses_ai, takes_params in COMMAND_GROUPS:
        handler = handlers[handler_name]
        spec = CommandSpec(handler=handler, uses_ai=uses_ai, takes_params=takes_params)
        for alias in aliases:
            registry[alias] = (spec.handler, spec.uses_ai, spec.takes_params)
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

    reply = message.get("reply_to_message") or {}
    if isinstance(reply, Mapping) and reply.get("from", {}).get("username") == bot_username:
        reply_text = str(reply.get("text") or "")
        if any(domain in reply_text for domain in LINK_REPLACEMENT_DOMAINS):
            return False

    is_command = command in commands
    is_private = chat_type == "private"
    is_mention = bot_name in message_lower
    is_reply = isinstance(reply, Mapping) and reply.get("from", {}).get("username", "") == bot_username

    if (
        is_reply
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
    "CommandSpec",
    "CommandTuple",
    "build_command_registry",
    "parse_command",
    "should_auto_process_media",
    "should_gordo_respond",
]
