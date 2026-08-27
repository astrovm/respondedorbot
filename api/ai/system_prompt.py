from __future__ import annotations

from collections.abc import Callable
from typing import Any

from api.i18n import tr
from api.i18n.prompts import prompt

ConfigLoader = Callable[[], dict[str, Any]]


def build_system_message(
    context: dict[str, Any],
    *,
    tools_active: bool,
    tool_schemas: list[dict[str, Any]] | None,
    task_mode: bool,
    load_config: ConfigLoader,
) -> dict[str, Any]:
    del tool_schemas
    config = load_config()
    formatted_time = str((context.get("time") or {}).get("formatted", "")).strip()

    task_prefix = prompt("task_mode") if task_mode else ""

    tool_instruction = _build_tool_instruction() if tools_active else ""
    contextual_info = prompt(
        "system_context",
        tool_instruction=tool_instruction,
        formatted_time=formatted_time,
        language_instruction=tr("ai.language_instruction"),
    )
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": task_prefix + str(config.get("system_prompt", "")) + contextual_info,
            }
        ],
    }


def _build_tool_instruction() -> str:
    return prompt("tools")
