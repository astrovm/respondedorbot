"""Chat configuration persistence and admin helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union

import redis

from api.chat_context import is_group_chat_type
from api.services.redis_helpers import redis_get_json, redis_setex_json


AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
ConfigLogger = Callable[[str, Optional[Mapping[str, Any]]], None]


CHAT_CONFIG_DEFAULTS = {
    "link_mode": "reply",
    "ai_random_replies": True,
    "ai_command_followups": True,
    "ignore_link_fix_followups": True,
    "timezone_offset": -3,
    "creditless_user_daily_limit": 5,
}

TIMEZONE_OFFSET_MIN = -12
TIMEZONE_OFFSET_MAX = 14

CHAT_ADMIN_STATUS_TTL = 300


def decode_redis_value(value: Any) -> Optional[str]:
    """Decode Redis values that may be bytes or text."""

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if value is not None:
        return str(value)
    return None


def load_chat_config_from_redis(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    log_event: ConfigLogger,
) -> Dict[str, Any]:
    """Load chat config from Redis for one-time migration to Postgres."""

    config = dict(CHAT_CONFIG_DEFAULTS)
    raw_value = redis_client.get(f"chat_config:{chat_id}")
    raw_value_text = decode_redis_value(raw_value)
    log_event(
        "Chat config raw value fetched",
        {"chat_id": chat_id, "raw_value": raw_value_text},
    )

    if raw_value_text:
        try:
            loaded = json.loads(raw_value_text)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if key in config:
                    config[key] = value
            return config

    log_event("No stored chat config found; using defaults", {"chat_id": chat_id})
    return config


def get_chat_config(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    chat_config_db_service: Any,
    admin_reporter: AdminReporter,
    log_event: ConfigLogger,
) -> Dict[str, Any]:
    """Load chat config from Postgres and migrate from Redis once when needed."""

    config = dict(CHAT_CONFIG_DEFAULTS)
    try:
        log_event("Loading chat config", {"chat_id": chat_id})
        if not chat_config_db_service.is_configured():
            log_event(
                "Chat config storage is not configured; using defaults",
                {"chat_id": chat_id},
            )
            return config

        pg_config = chat_config_db_service.get_chat_config(
            chat_id, CHAT_CONFIG_DEFAULTS
        )
        if isinstance(pg_config, dict):
            return pg_config

        redis_config = load_chat_config_from_redis(
            redis_client, chat_id, log_event=log_event
        )
        if redis_config != CHAT_CONFIG_DEFAULTS:
            try:
                chat_config_db_service.set_chat_config(chat_id, redis_config)
            except Exception as persist_error:
                admin_reporter(
                    "Error migrating chat config from Redis to Postgres",
                    persist_error,
                    {"chat_id": chat_id},
                )
            else:
                log_event(
                    "Migrated chat config from Redis to Postgres",
                    {"chat_id": chat_id, "config": redis_config},
                )
        return redis_config
    except Exception as error:
        admin_reporter(
            "Error loading chat config",
            error,
            {
                "chat_id": chat_id,
                "postgres_configured": chat_config_db_service.is_configured(),
            },
        )
    return config


def set_chat_config(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    chat_config_db_service: Any,
    admin_reporter: AdminReporter,
    log_event: ConfigLogger,
    **updates: Any,
) -> Dict[str, Any]:
    """Apply and persist a partial chat config update."""

    config = get_chat_config(
        redis_client,
        chat_id,
        chat_config_db_service=chat_config_db_service,
        admin_reporter=admin_reporter,
        log_event=log_event,
    )
    for key, value in updates.items():
        if key in config:
            config[key] = value

    try:
        log_event(
            "Saving chat config",
            {"chat_id": chat_id, "updates": updates, "config": config},
        )
        if not chat_config_db_service.is_configured():
            log_event(
                "Chat config storage is not configured; skipping persistence",
                {"chat_id": chat_id, "config": config},
            )
            return config

        chat_config_db_service.set_chat_config(chat_id, config)
    except Exception as error:
        admin_reporter(
            "Error saving chat config",
            error,
            {"chat_id": chat_id, "updates": updates},
        )

    return config


def coerce_bool(value: Any, *, default: bool) -> bool:
    """Normalize truthy config values that might be stored as strings."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "enabled"}:
            return True
        if lowered in {"false", "0", "no", "off", "disabled"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default if value is None else default


def _format_gmt_offset(offset: int) -> str:
    """Format UTC offset for display."""
    if offset == 0:
        return "UTC"
    sign = "+" if offset > 0 else ""
    return f"UTC{sign}{offset}"


@dataclass
class ChatConfigData:
    link_mode: str
    random_enabled: bool
    followups_enabled: bool
    ignore_link_fix_followups: bool
    timezone_offset: int
    creditless_limit: int


def parse_chat_config(config: Mapping[str, Any]) -> ChatConfigData:
    return ChatConfigData(
        link_mode=str(config.get("link_mode", "reply")),
        random_enabled=coerce_bool(config.get("ai_random_replies"), default=True),
        followups_enabled=coerce_bool(config.get("ai_command_followups"), default=True),
        ignore_link_fix_followups=coerce_bool(
            config.get("ignore_link_fix_followups"), default=True
        ),
        timezone_offset=int(config.get("timezone_offset", -3)),
        creditless_limit=int(config.get("creditless_user_daily_limit", 5)),
    )


def build_config_text(config: Mapping[str, Any], chat_type: str = "group") -> str:
    """Build the user-facing config summary text."""

    parsed = parse_chat_config(config)

    link_labels = {
        "delete": "borra el mensaje original y repostea el link arreglado",
        "reply": "responde al mensaje original con el link arreglado",
        "off": "no toca los links",
    }

    if parsed.creditless_limit < 0:
        creditless_label = "ilimitados"
    elif parsed.creditless_limit == 0:
        creditless_label = "ninguno"
    else:
        creditless_label = str(parsed.creditless_limit)

    is_group = is_group_chat_type(chat_type) if chat_type else True

    lines = [
        "config del gordo",
        "",
        "1. links arreglados",
        link_labels.get(parsed.link_mode, parsed.link_mode),
        "",
        "2. seguir charla",
        "si está activado, me contestás después de un comando y sigo el hilo como si nada",
        f"{'✅ activado' if parsed.followups_enabled else '▫️ desactivado'}",
        "",
        "3. ignorar replies a links arreglados",
        "si está activado, ignoro replies comunes a mensajes automáticos con fixupx/fxtwitter y similares",
        f"{'✅ activado' if parsed.ignore_link_fix_followups else '▫️ desactivado'}",
        "",
        "4. zona horaria",
        _format_gmt_offset(parsed.timezone_offset),
    ]

    if is_group:
        lines.extend([
            "",
            "5. ia random",
            "si está activado, a veces me meto solo en la charla aunque nadie me llame",
            f"{'✅ activado' if parsed.random_enabled else '▫️ desactivado'}",
            "",
            "6. limite ia gratis por usuario por dia",
            "cuantas veces puede usar ia del grupo un usuario sin creditos propios",
            creditless_label,
        ])

    lines.extend([
        "",
        "tocá los botones de abajo y dejalo como se te cante",
    ])
    return "\n".join(lines)


def build_config_keyboard(config: Mapping[str, Any], chat_type: str = "group") -> Dict[str, Any]:
    """Build the inline keyboard to toggle chat config values."""

    parsed = parse_chat_config(config)

    def choice_button(
        label: str, value: str, current: str, *, action: str
    ) -> Dict[str, str]:
        prefix = "✅" if current == value else "▫️"
        return {"text": f"{prefix} {label}", "callback_data": f"cfg:{action}:{value}"}

    def toggle_button(label: str, enabled: bool, action: str) -> Dict[str, str]:
        prefix = "✅" if enabled else "▫️"
        return {"text": f"{prefix} {label}", "callback_data": f"cfg:{action}:toggle"}

    def timezone_button(label: str, offset: int) -> Dict[str, str]:
        prefix = "✅ " if offset == parsed.timezone_offset else ""
        return {"text": f"{prefix}{label}", "callback_data": f"cfg:timezone:{offset}"}

    def creditless_button(label: str, value: int) -> Dict[str, str]:
        prefix = "✅ " if value == parsed.creditless_limit else ""
        return {"text": f"{prefix}{label}", "callback_data": f"cfg:creditless:{value}"}

    dec_offset = max(parsed.timezone_offset - 1, TIMEZONE_OFFSET_MIN)
    inc_offset = min(parsed.timezone_offset + 1, TIMEZONE_OFFSET_MAX)

    is_group = is_group_chat_type(chat_type) if chat_type else True

    keyboard = [
        [
            choice_button("responder link", "reply", parsed.link_mode, action="link"),
            choice_button("borrar link", "delete", parsed.link_mode, action="link"),
            choice_button("apagado", "off", parsed.link_mode, action="link"),
        ],
        [
            toggle_button(
                "seguir charla en comandos",
                parsed.followups_enabled,
                action="followups",
            )
        ],
        [
            toggle_button(
                "ignorar replies a links arreglados",
                parsed.ignore_link_fix_followups,
                action="linkfixfollowups",
            )
        ],
        [
            {"text": "➖ 1h", "callback_data": f"cfg:timezone:{dec_offset}"},
            {"text": f"🌍 {_format_gmt_offset(parsed.timezone_offset)}", "callback_data": f"cfg:timezone:{parsed.timezone_offset}"},
            {"text": "➕ 1h", "callback_data": f"cfg:timezone:{inc_offset}"},
        ],
    ]

    if is_group:
        keyboard.extend([
            [toggle_button("me meto en la charla", parsed.random_enabled, action="random")],
            [
                creditless_button("ninguno", 0),
                creditless_button("3", 3),
                creditless_button("5", 5),
                creditless_button("10", 10),
                creditless_button("∞", -1),
            ],
        ])
    return {"inline_keyboard": keyboard}

def chat_admin_cache_key(chat_id: str, user_id: Union[str, int]) -> str:
    return f"chat_admin:{chat_id}:{user_id}"


def is_chat_admin(
    chat_id: str,
    user_id: Optional[Union[str, int]],
    *,
    redis_client: Optional[redis.Redis],
    optional_redis_client: Callable[[], Optional[redis.Redis]],
    telegram_request: Callable[..., Any],
    log_event: ConfigLogger,
    redis_get_json_fn: Callable[[redis.Redis, str], Optional[Any]] = redis_get_json,
    redis_setex_json_fn: Callable[
        [redis.Redis, str, int, Any], bool
    ] = redis_setex_json,
) -> bool:
    """Return True if user_id is an admin of the chat."""

    if not chat_id or user_id is None:
        return False

    redis_client = redis_client or optional_redis_client()
    cache_key = chat_admin_cache_key(chat_id, user_id)

    if redis_client:
        cached_value = redis_get_json_fn(redis_client, cache_key)
        cached_flag = (
            cached_value.get("is_admin")
            if isinstance(cached_value, Mapping)
            else cached_value
        )
        if isinstance(cached_flag, bool):
            return cached_flag

    payload, error = telegram_request(
        "getChatMember",
        method="GET",
        params={"chat_id": chat_id, "user_id": user_id},
        log_errors=False,
    )

    is_admin = False
    if payload and payload.get("ok"):
        result = payload.get("result") or {}
        status = str(result.get("status") or "").lower()
        is_admin = status in {"administrator", "creator"}
    else:
        log_event(
            "Failed to verify chat admin status",
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "error": error or (payload or {}).get("description"),
            },
        )

    if redis_client:
        redis_setex_json_fn(
            redis_client,
            cache_key,
            CHAT_ADMIN_STATUS_TTL,
            {"is_admin": is_admin},
        )

    return is_admin


def report_unauthorized_config_attempt(
    chat_id: str,
    user: Mapping[str, Any],
    *,
    chat_type: Optional[str],
    action: str,
    log_event: ConfigLogger,
    callback_data: Optional[str] = None,
) -> None:
    """Log unauthorized attempts to mutate chat config."""

    context: Dict[str, Any] = {
        "chat_id": chat_id,
        "chat_type": chat_type,
        "user_id": user.get("id"),
        "username": user.get("username"),
        "action": action,
    }
    if callback_data:
        context["callback_data"] = callback_data
    log_event("Unauthorized config attempt", context)


__all__ = [
    "CHAT_ADMIN_STATUS_TTL",
    "CHAT_CONFIG_DEFAULTS",
    "ChatConfigData",
    "TIMEZONE_OFFSET_MAX",
    "TIMEZONE_OFFSET_MIN",
    "build_config_keyboard",
    "build_config_text",
    "coerce_bool",
    "decode_redis_value",
    "get_chat_config",
    "is_chat_admin",
    "is_group_chat_type",
    "parse_chat_config",
    "report_unauthorized_config_attempt",
    "set_chat_config",
]
