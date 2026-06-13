"""get_chat_members tool — returns known group members to the AI."""

from __future__ import annotations

import time
from typing import Any, Dict

from api.memory.state import get_chat_members
from api.tools.registry import ToolResult, register_tool


def _execute_get_chat_members(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    chat_id = context.get("chat_id")
    config_redis = context.get("config_redis")
    if not chat_id or not config_redis:
        return ToolResult(output="no disponible")
    redis_client = config_redis()
    if redis_client is None:
        return ToolResult(output="no disponible")
    members = get_chat_members(redis_client, str(chat_id))
    if not members:
        return ToolResult(output="no conozco a nadie en este chat todavia")
    lines = []
    now = int(time.time())
    for m in members:
        first_name = m.get("first_name") or ""
        username = m.get("username") or ""
        last_seen = int(m.get("last_seen") or 0)
        age = now - last_seen
        if age < 60:
            ago = "hace unos segundos"
        elif age < 3600:
            ago = f"hace {age // 60} min"
        elif age < 86400:
            ago = f"hace {age // 3600}h"
        else:
            ago = f"hace {age // 86400}d"
        name_part = f"{first_name} (@{username})" if username else first_name
        lines.append(f"- {name_part} — visto {ago}")
    return ToolResult(output="Miembros conocidos:\n" + "\n".join(lines))


register_tool(
    name="get_chat_members",
    description="Get list of known chat members. Returns users who have sent messages in this group.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    executor=_execute_get_chat_members,
    requires_context=["chat_id", "config_redis"],
)
