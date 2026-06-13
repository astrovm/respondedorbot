from __future__ import annotations

from collections.abc import Callable
from typing import Any

ConfigLoader = Callable[[], dict[str, Any]]
MarketFormatter = Callable[[dict[str, Any]], str]
WeatherFormatter = Callable[[dict[str, Any]], str]
NewsFormatter = Callable[[Any], str]
CapabilitiesRenderer = Callable[[], str]


def build_system_message(
    context: dict[str, Any],
    *,
    tools_active: bool,
    tool_schemas: list[dict[str, Any]] | None,
    task_mode: bool,
    load_config: ConfigLoader,
    format_market: MarketFormatter,
    format_weather: WeatherFormatter,
    format_news: NewsFormatter,
    render_capabilities: CapabilitiesRenderer,
) -> dict[str, Any]:
    config = load_config()
    market_info = format_market(context.get("market") or {})
    weather_source = context.get("weather")
    weather_info = format_weather(weather_source) if weather_source else ""
    news_info = format_news(context.get("hacker_news"))
    formatted_time = str((context.get("time") or {}).get("formatted", "")).strip()

    task_prefix = ""
    if task_mode:
        task_prefix = (
            "EJECUTANDO TAREA PROGRAMADA:\n"
            "Responde la siguiente instruccion y nada mas.\n"
            "No hagas preguntas, no ofrezcas seguimientos, no pidas confirmacion.\n"
            "Genera tu respuesta y terminá.\n\n"
        )

    tool_instruction = _build_tool_instruction(tool_schemas) if tools_active else ""
    contextual_info = f"""
{tool_instruction}
{render_capabilities()}

FECHA ACTUAL:
{formatted_time}

CONTEXTO DEL MERCADO:
{market_info}

CLIMA EN BUENOS AIRES:
{weather_info}

NOTICIAS DE HACKER NEWS:
{news_info}
"""
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": task_prefix
                + str(config.get("system_prompt", ""))
                + contextual_info,
            }
        ],
    }


def _build_tool_instruction(
    tool_schemas: list[dict[str, Any]] | None,
) -> str:
    if not tool_schemas:
        return ""

    summaries = []
    for entry in tool_schemas:
        function = entry.get("function", {})
        summaries.append(
            f"- {function.get('name', '')}: {function.get('description', '')}"
        )
    tool_summaries = "\n".join(summaries) + "\n"
    return (
        f"\n\nHERRAMIENTAS DISPONIBLES:\n{tool_summaries}"
        "Llamalas directamente, sin pedir permiso ni narrar antes.\n"
        "No expliques que vas a hacer antes de usar una herramienta simple.\n"
        "Usa las herramientas exactamente como estan nombradas arriba.\n"
        "\n"
        "task_set detalles:\n"
        "- task_set.text debe contener solo el contenido a ejecutar despues.\n"
        "- no incluyas tiempo ni frecuencia en text si ya van en delay_seconds, interval_seconds o trigger_config.\n"
        "- no reescribas pronombres ni cambies sujeto al guardar la tarea.\n"
        "- si el usuario dice 'decime', 'avisame' o pide que el bot hable de si mismo, preserva eso en el contenido restante.\n"
        "- ejemplo: 'A las 20:30 todos los dias decime cuanta aura farmeaste hoy' -> text=\"decime cuanta aura farmeaste hoy\" y trigger_config con hour=20, minute=30.\n"
        "- ejemplo: 'deci fumareeemooss todos los dias a las 4:20 am' -> text=\"deci fumareeemooss\" y trigger_config con hour=4, minute=20.\n"
        "- ejemplo: 'mañana recordame pagar el alquiler' -> text=\"recordame pagar el alquiler\" y el tiempo va en el parametro de schedule.\n"
        "- trigger_config con type='interval' y days=N para cada N dias.\n"
        "- trigger_config con type='cron', hour, minute para horarios especificos.\n"
        "- cron puede tener day_of_week='lun,mie,vie' o 'mon,wed,fri'; se normaliza internamente. Tambien puede usar day=1 para primer dia del mes.\n"
        "- si no especificas hora para cron, elegi una hora razonable segun el contexto.\n"
    )
