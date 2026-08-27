from __future__ import annotations

from collections.abc import Callable
from typing import Any

from api.core.i18n import tr

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

    task_prefix = ""
    if task_mode:
        task_prefix = (
            "EJECUTANDO TAREA PROGRAMADA:\n"
            "Responde la siguiente instruccion y nada mas.\n"
            "No hagas preguntas, no ofrezcas seguimientos, no pidas confirmacion.\n"
            "Genera tu respuesta y terminá.\n\n"
        )

    tool_instruction = _build_tool_instruction() if tools_active else ""
    contextual_info = f"""
{tool_instruction}

FECHA ACTUAL:
{formatted_time}

IDIOMA DE RESPUESTA:
{tr("ai.language_instruction")}
"""
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
    return (
        "\n\nHERRAMIENTAS:\n"
        "Llamalas directamente, sin pedir permiso ni narrar antes.\n"
        "No expliques que vas a hacer antes de usar una herramienta simple.\n"
        "Usa las herramientas cuando necesites datos actuales o externos.\n"
    )
