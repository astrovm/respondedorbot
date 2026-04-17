"""Telegram bot command menu helpers."""

from __future__ import annotations

import json
from typing import Callable, Dict, Mapping, Sequence, Tuple

CommandGroup = Tuple[Tuple[str, ...], str, bool, bool]

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
    "dolar": "cotizaciones del dolar [1h/6h/12h/24h/48h]",
    "dollar": "cotizaciones del dolar [1h/6h/12h/24h/48h]",
    "usd": "cotizaciones del dolar [1h/6h/12h/24h/48h]",
    "petroleo": "te paso el precio del Brent y del WTI",
    "oil": "te paso el precio del Brent y del WTI",
    "eleccion": "odds actuales de Polymarket para Diputados 2025",
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
    "buscar": "te busco en la web",
    "search": "te busco en la web",
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
}


def build_commands_list(
    command_groups: Sequence[CommandGroup],
    *,
    descriptions: Mapping[str, str] = COMMAND_DESCRIPTIONS,
) -> list[dict[str, str]]:
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
