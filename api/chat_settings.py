"""Chat configuration compatibility wrappers and admin helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union

import redis

from api.chat_context import is_group_chat_type
from api.chat_config_defaults import (
    CHAT_ADMIN_STATUS_TTL,
    CHAT_CONFIG_DEFAULTS,
    TIMEZONE_OFFSET_MAX,
    TIMEZONE_OFFSET_MIN,
)
from api.chat_config_service import (
    build_chat_config_service,
    decode_redis_value,
)
from api.services.redis_helpers import redis_get_json, redis_setex_json
from api.storage.chat_config_repository import build_chat_config_repository

AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
ConfigLogger = Callable[[str, Optional[Mapping[str, Any]]], None]


def _build_service(
    *,
    chat_config_db_service: Any = None,
    admin_reporter: Optional[AdminReporter] = None,
    log_event: Optional[ConfigLogger] = None,
):
    repo = (
        chat_config_db_service
        if chat_config_db_service is not None
        else build_chat_config_repository()
    )
    return build_chat_config_service(
        repository=repo,
        admin_reporter=admin_reporter or (lambda *a, **k: None),
        log_event=log_event or (lambda *a, **k: None),
    )


_cached_service: Optional[Any] = None
_cached_service_key: Optional[tuple[int, int, int]] = None


def _service_cache_key(
    *,
    chat_config_db_service: Any = None,
    admin_reporter: Optional[AdminReporter] = None,
    log_event: Optional[ConfigLogger] = None,
) -> tuple[int, int, int]:
    return (
        id(chat_config_db_service),
        id(admin_reporter),
        id(log_event),
    )


def _get_cached_service(
    *,
    chat_config_db_service: Any = None,
    admin_reporter: Optional[AdminReporter] = None,
    log_event: Optional[ConfigLogger] = None,
):
    global _cached_service, _cached_service_key
    cache_key = _service_cache_key(
        chat_config_db_service=chat_config_db_service,
        admin_reporter=admin_reporter,
        log_event=log_event,
    )
    if _cached_service is None or _cached_service_key != cache_key:
        _cached_service = _build_service(
            chat_config_db_service=chat_config_db_service,
            admin_reporter=admin_reporter,
            log_event=log_event,
        )
        _cached_service_key = cache_key
    return _cached_service


def reset_chat_config_cache() -> None:
    global _cached_service, _cached_service_key
    _cached_service = None
    _cached_service_key = None


def get_chat_config(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    chat_config_db_service: Any = None,
    admin_reporter: Optional[AdminReporter] = None,
    log_event: Optional[ConfigLogger] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper that builds a ChatConfigService and delegates.

    Kept as a function so existing callsites (api.index and tests) keep the same
    signature while the implementation is moved into the service.
    """

    service = _get_cached_service(
        chat_config_db_service=chat_config_db_service,
        admin_reporter=admin_reporter,
        log_event=log_event,
    )
    return service.get_chat_config(redis_client, chat_id)


def set_chat_config(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    chat_config_db_service: Any = None,
    admin_reporter: Optional[AdminReporter] = None,
    log_event: Optional[ConfigLogger] = None,
    **updates: Any,
) -> Dict[str, Any]:
    service = _get_cached_service(
        chat_config_db_service=chat_config_db_service,
        admin_reporter=admin_reporter,
        log_event=log_event,
    )
    return service.set_chat_config(redis_client, chat_id, **updates)


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
    return default


def _format_utc_offset(offset: int) -> str:
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
        creditless_limit=int(
            config.get(
                "creditless_user_hourly_limit",
                config.get("creditless_user_daily_limit", 5),
            )
        ),
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
        "2. seguir charla en comandos",
        "si está activado, después de un comando sigo la conversación si me respondés",
        f"{'✅ activado' if parsed.followups_enabled else '▫️ desactivado'}",
        "",
        "3. ignorar replies a links arreglados",
        "si está activado, ignoro respuestas normales a mensajes automáticos con links arreglados",
        f"{'✅ activado' if parsed.ignore_link_fix_followups else '▫️ desactivado'}",
        "",
        "4. zona horaria",
        _format_utc_offset(parsed.timezone_offset),
    ]

    if is_group:
        lines.extend(
            [
                "",
                "5. respuestas random",
                "si está activado, a veces respondo solo en el grupo aunque nadie me llame",
                f"{'✅ activado' if parsed.random_enabled else '▫️ desactivado'}",
                "",
                "6. mensajes gratis por usuario por hora",
                "cuantos mensajes de ia paga el grupo por usuario cada hora",
                creditless_label,
            ]
        )

    lines.extend(
        [
            "",
            "tocá los botones de abajo para cambiar la config",
        ]
    )
    return "\n".join(lines)


def build_config_keyboard(
    config: Mapping[str, Any], chat_type: str = "group"
) -> Dict[str, Any]:
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

    def creditless_button(label: str, value: str) -> Dict[str, str]:
        return {"text": label, "callback_data": f"cfg:creditless:{value}"}

    creditless_current_label = (
        "∞" if parsed.creditless_limit < 0 else str(parsed.creditless_limit)
    )

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
            {
                "text": f"🌍 {_format_utc_offset(parsed.timezone_offset)}",
                "callback_data": "cfg:timezone:current",
            },
            {"text": "➕ 1h", "callback_data": f"cfg:timezone:{inc_offset}"},
        ],
    ]

    if is_group:
        keyboard.extend(
            [
                [
                    toggle_button(
                        "me meto en la charla", parsed.random_enabled, action="random"
                    )
                ],
                [
                    creditless_button("0", "none"),
                    creditless_button("-", "decrease"),
                    creditless_button(creditless_current_label, "current"),
                    creditless_button("+", "increase"),
                    creditless_button("∞", "unlimited"),
                ],
            ]
        )
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
