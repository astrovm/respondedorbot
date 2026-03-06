"""Chat configuration persistence and admin helpers."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Optional, Union

import redis

from api.services.redis_helpers import redis_get_json, redis_setex_json


AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
ConfigLogger = Callable[[str, Optional[Mapping[str, Any]]], None]


CHAT_CONFIG_KEY_PREFIX = "chat_config:"
CHAT_CONFIG_DEFAULTS = {
    "link_mode": "off",
    "ai_random_replies": True,
    "ai_command_followups": True,
}
CHAT_ADMIN_STATUS_TTL = 300


def decode_redis_value(value: Any) -> Optional[str]:
    """Decode Redis values that may be bytes or text."""

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if value is not None:
        return str(value)
    return None


def is_group_chat_type(chat_type: Optional[str]) -> bool:
    """Return True for group or supergroup chats."""

    return str(chat_type) in {"group", "supergroup"}


def chat_config_key(chat_id: str) -> str:
    return f"{CHAT_CONFIG_KEY_PREFIX}{chat_id}"


def legacy_link_mode_key(chat_id: str) -> str:
    return f"link_mode:{chat_id}"


def load_chat_config_from_redis(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    log_event: ConfigLogger,
) -> Dict[str, Any]:
    """Load chat config from Redis, keeping legacy link_mode compatibility."""

    config = dict(CHAT_CONFIG_DEFAULTS)
    raw_value = redis_client.get(chat_config_key(chat_id))
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

    legacy_value_text = decode_redis_value(redis_client.get(legacy_link_mode_key(chat_id)))
    if legacy_value_text:
        log_event(
            "Using legacy link mode config",
            {"chat_id": chat_id, "legacy_value": legacy_value_text},
        )
        config["link_mode"] = legacy_value_text
    else:
        log_event("No stored chat config found; using defaults", {"chat_id": chat_id})

    return config


def persist_chat_config_to_redis(
    redis_client: redis.Redis,
    chat_id: str,
    config: Mapping[str, Any],
    *,
    log_event: ConfigLogger,
) -> None:
    """Persist chat config in Redis and mirror legacy link_mode keys."""

    redis_client.set(chat_config_key(chat_id), json.dumps(dict(config)))
    link_mode = str(config.get("link_mode", "off"))
    legacy_key = legacy_link_mode_key(chat_id)
    if link_mode in {"reply", "delete"}:
        redis_client.set(legacy_key, link_mode)
        log_event("Persisted legacy link mode", {"chat_id": chat_id, "link_mode": link_mode})
        return

    redis_client.delete(legacy_key)
    log_event("Cleared legacy link mode", {"chat_id": chat_id})


def get_chat_config(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    chat_config_db_service: Any,
    admin_reporter: AdminReporter,
    log_event: ConfigLogger,
) -> Dict[str, Any]:
    """Load chat config using Postgres when available and Redis as fallback/migration source."""

    config = dict(CHAT_CONFIG_DEFAULTS)
    try:
        log_event("Loading chat config", {"chat_id": chat_id})
        if chat_config_db_service.is_configured():
            pg_config = chat_config_db_service.get_chat_config(chat_id, CHAT_CONFIG_DEFAULTS)
            if isinstance(pg_config, dict):
                return pg_config

            redis_config = load_chat_config_from_redis(
                redis_client, chat_id, log_event=log_event
            )
            try:
                chat_config_db_service.set_chat_config(chat_id, redis_config)
            except Exception as persist_error:
                admin_reporter(
                    "Error migrating chat config from Redis to Postgres",
                    persist_error,
                    {"chat_id": chat_id},
                )
                return config
            return redis_config

        return load_chat_config_from_redis(redis_client, chat_id, log_event=log_event)
    except Exception as error:
        admin_reporter(
            "Error loading chat config",
            error,
            {
                "chat_id": chat_id,
                "postgres_configured": chat_config_db_service.is_configured(),
            },
        )
        if not chat_config_db_service.is_configured():
            try:
                return load_chat_config_from_redis(redis_client, chat_id, log_event=log_event)
            except Exception as redis_error:
                admin_reporter(
                    "Error loading chat config from Redis",
                    redis_error,
                    {"chat_id": chat_id},
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
        if chat_config_db_service.is_configured():
            chat_config_db_service.set_chat_config(chat_id, config)
        else:
            persist_chat_config_to_redis(
                redis_client,
                chat_id,
                config,
                log_event=log_event,
            )
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


def build_config_text(config: Mapping[str, Any]) -> str:
    """Build the user-facing config summary text."""

    link_mode = str(config.get("link_mode", "off"))
    random_enabled = coerce_bool(config.get("ai_random_replies"), default=True)
    followups_enabled = coerce_bool(config.get("ai_command_followups"), default=True)

    link_labels = {
        "delete": "borrar mensaje original",
        "reply": "responder al mensaje original",
        "off": "apagado",
    }

    lines = [
        "Gordo config:",
        "",
        f"Arregla-links: {link_labels.get(link_mode, link_mode)}",
        f"Respuestas random de IA: {'✅ activadas' if random_enabled else '▫️ desactivadas'}",
        "Seguimientos para comandos no-IA: "
        f"{'✅ activados' if followups_enabled else '▫️ desactivados'}",
        "",
        "tocá los botones de abajo para cambiar la config",
    ]
    return "\n".join(lines)


def build_config_keyboard(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the inline keyboard to toggle chat config values."""

    link_mode = str(config.get("link_mode", "off"))
    random_enabled = coerce_bool(config.get("ai_random_replies"), default=True)
    followups_enabled = coerce_bool(config.get("ai_command_followups"), default=True)

    def choice_button(label: str, value: str, current: str, *, action: str) -> Dict[str, str]:
        prefix = "✅" if current == value else "▫️"
        return {"text": f"{prefix} {label}", "callback_data": f"cfg:{action}:{value}"}

    def toggle_button(label: str, enabled: bool, action: str) -> Dict[str, str]:
        prefix = "✅" if enabled else "▫️"
        return {"text": f"{prefix} {label}", "callback_data": f"cfg:{action}:toggle"}

    return {
        "inline_keyboard": [
            [
                choice_button("responder al mensaje original", "reply", link_mode, action="link"),
                choice_button("borrar mensaje original", "delete", link_mode, action="link"),
                choice_button("apagado", "off", link_mode, action="link"),
            ],
            [toggle_button("respuestas random de IA", random_enabled, action="random")],
            [
                toggle_button(
                    "seguimientos para comandos no-IA",
                    followups_enabled,
                    action="followups",
                )
            ],
        ]
    }


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
) -> bool:
    """Return True if user_id is an admin of the chat."""

    if not chat_id or user_id is None:
        return False

    redis_client = redis_client or optional_redis_client()
    cache_key = chat_admin_cache_key(chat_id, user_id)

    if redis_client:
        cached_value = redis_get_json(redis_client, cache_key)
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
        redis_setex_json(
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
    "build_config_keyboard",
    "build_config_text",
    "coerce_bool",
    "decode_redis_value",
    "get_chat_config",
    "is_chat_admin",
    "is_group_chat_type",
    "report_unauthorized_config_attempt",
    "set_chat_config",
]
