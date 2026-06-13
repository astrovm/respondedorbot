"""Telegram bot command menu helpers."""

from __future__ import annotations

import json
from typing import Callable, Mapping, Sequence, Tuple

from api.bot.feature_catalog import (
    COMMAND_DESCRIPTIONS,
    telegram_command_descriptions,
)

CommandGroup = Tuple[Tuple[str, ...], str, bool, bool]


def build_commands_list(
    command_groups: Sequence[CommandGroup],
    *,
    descriptions: Mapping[str, str] = COMMAND_DESCRIPTIONS,
) -> list[dict[str, str]]:
    if descriptions is COMMAND_DESCRIPTIONS:
        descriptions = telegram_command_descriptions(
            command_groups=command_groups,
            descriptions=descriptions,
        )

    commands_list: list[dict[str, str]] = []
    seen_commands: set[str] = set()

    for aliases, _handler_name, _uses_ai, _takes_params in command_groups:
        for alias in aliases:
            command = alias.lstrip("/")
            if command in seen_commands or command not in descriptions:
                continue
            seen_commands.add(command)
            commands_list.append(
                {
                    "command": command,
                    "description": descriptions[command],
                }
            )

    commands_list.sort(key=lambda item: item["command"])
    return commands_list


def update_bot_commands(
    *,
    token: str,
    request_fn: Callable[..., tuple[object, object]],
    command_groups: Sequence[CommandGroup],
    descriptions: Mapping[str, str] = COMMAND_DESCRIPTIONS,
    logger: Callable[[str], None] = print,
) -> bool:
    commands_list = build_commands_list(command_groups, descriptions=descriptions)
    _response, error = request_fn(
        "setMyCommands",
        method="POST",
        json_payload={"commands": json.dumps(commands_list)},
        token=token,
        expect_json=False,
    )
    if error:
        logger(f"Error updating bot commands: {error}")
        return False
    logger(f"Bot commands updated successfully: {len(commands_list)} commands")
    return True
