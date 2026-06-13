from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CallbackConfigDeps:
    set_chat_config: Callable[..., dict[str, Any]]
    coerce_bool: Callable[..., bool]
    guard_callback: Callable[..., bool]
    log_event: Callable[[str, Mapping[str, Any] | None], None]
    timezone_offset_min: int
    timezone_offset_max: int


@dataclass(frozen=True, slots=True)
class TaskCallbackDeps:
    guard_callback: Callable[..., bool]
    list_tasks: Callable[[str], list[dict[str, Any]]]
    cancel_task: Callable[[str], Any]
    is_group_chat_type: Callable[[str], bool]
    config_redis: Callable[..., Any]
    is_chat_admin: Callable[..., bool]
    answer_callback: Callable[..., None]
    build_tasks_message: Callable[..., tuple[str, dict[str, Any] | None]]
    edit_message: Callable[..., Any]
    logger: Any


@dataclass(frozen=True, slots=True)
class CallbackQueryDeps:
    guard_callback: Callable[..., bool]
    handle_topup: Callable[[dict[str, Any]], None]
    handle_task: Callable[[dict[str, Any]], None]
    handle_signal: Callable[..., bool]
    config_redis: Callable[..., Any]
    delete_msg: Callable[..., Any]
    edit_photo: Callable[..., Any]
    is_chat_admin: Callable[..., bool]
    answer_callback: Callable[..., None]
    admin_report: Callable[..., None]
    is_group_chat_type: Callable[[str], bool]
    send_msg: Callable[..., Any]
    report_unauthorized: Callable[..., None]
    denial_message: str
    get_chat_config: Callable[..., dict[str, Any]]
    config: CallbackConfigDeps
    build_config_text: Callable[..., str]
    build_config_keyboard: Callable[..., dict[str, Any]]
    edit_message: Callable[..., bool]


@dataclass(frozen=True, slots=True)
class CallbackContext:
    callback_id: str | None
    data: str
    chat_id: str
    chat_type: str
    message_id: int
    user: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConfigUpdateContext:
    redis_client: Any
    chat_id: str
    callback_id: str | None
    deps: CallbackConfigDeps


def handle_task_callback(
    callback_query: dict[str, Any],
    *,
    deps: TaskCallbackDeps,
) -> None:
    callback_data = callback_query.get("data")
    callback_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    user = callback_query.get("from") or {}

    if deps.guard_callback(callback_id, not callback_data or chat_id is None):
        return

    parts = str(callback_data).split(":", 2)
    if deps.guard_callback(callback_id, len(parts) != 3 or parts[0] != "task"):
        return

    task_id = parts[2]
    tasks = deps.list_tasks(str(chat_id))
    target_task = next(
        (task for task in tasks if str(task.get("id")) == str(task_id)),
        None,
    )
    if deps.guard_callback(
        callback_id,
        not target_task,
        text="esa tarea no existe",
        show_alert=True,
    ):
        return
    assert target_task is not None

    request_user_id = user.get("id")
    task_owner_id = target_task.get("user_id")
    is_owner = bool(
        task_owner_id and str(request_user_id) == str(task_owner_id)
    )

    chat_type = str(chat.get("type", ""))
    if deps.is_group_chat_type(chat_type):
        redis_client = deps.config_redis()
        is_admin = deps.is_chat_admin(
            str(chat_id),
            request_user_id,
            redis_client=redis_client,
        )
    else:
        is_admin = True

    if deps.guard_callback(
        callback_id,
        not is_owner and not is_admin,
        text="solo el creador o un admin pueden borrar esta tarea",
        show_alert=True,
    ):
        return

    deps.cancel_task(task_id)
    if callback_id:
        deps.answer_callback(callback_id, text=f"tarea {task_id} borrada")

    if message_id:
        tasks = deps.list_tasks(str(chat_id))
        new_text, new_keyboard = deps.build_tasks_message(tasks)
        try:
            deps.edit_message(str(chat_id), int(message_id), new_text, new_keyboard)
        except Exception as error:
            deps.logger.exception(
                "handle_task_callback: failed to edit task message "
                "chat_id=%s message_id=%s: %s",
                chat_id,
                message_id,
                error,
            )


def _update_timezone(
    config: dict[str, Any],
    value: str,
    *,
    context: ConfigUpdateContext,
) -> tuple[dict[str, Any], bool]:
    if context.deps.guard_callback(context.callback_id, value == "current"):
        return config, True
    try:
        offset = max(
            context.deps.timezone_offset_min,
            min(int(value), context.deps.timezone_offset_max),
        )
    except ValueError:
        context.deps.log_event(
            "Invalid timezone callback value",
            {"chat_id": context.chat_id, "value": value},
        )
        return config, False
    return (
        context.deps.set_chat_config(
            context.redis_client,
            context.chat_id,
            timezone_offset=offset,
        ),
        False,
    )


def _creditless_limit(config: Mapping[str, Any], value: str) -> int:
    current = int(
        config.get(
            "creditless_user_hourly_limit",
            config.get("creditless_user_daily_limit", 5),
        )
    )
    if value == "none":
        return 0
    if value == "decrease":
        return current if current < 0 else max(0, current - 1)
    if value == "increase":
        return current if current < 0 else current + 1
    if value == "unlimited":
        return -1
    limit = int(value)
    if limit < -1:
        raise ValueError
    return limit


def _update_creditless_limit(
    config: dict[str, Any],
    value: str,
    *,
    context: ConfigUpdateContext,
) -> tuple[dict[str, Any], bool]:
    if context.deps.guard_callback(context.callback_id, value == "current"):
        return config, True
    try:
        limit = _creditless_limit(config, value)
    except (TypeError, ValueError):
        context.deps.log_event(
            "Invalid creditless callback value",
            {"chat_id": context.chat_id, "value": value},
        )
        return config, False
    return (
        context.deps.set_chat_config(
            context.redis_client,
            context.chat_id,
            creditless_user_hourly_limit=limit,
        ),
        False,
    )


def update_callback_config(
    config: dict[str, Any],
    action: str,
    value: str,
    *,
    context: ConfigUpdateContext,
) -> tuple[dict[str, Any], bool]:
    toggle_fields = {
        "random": "ai_random_replies",
        "followups": "ai_command_followups",
        "linkfixfollowups": "ignore_link_fix_followups",
    }
    if action == "link" and value in {"reply", "delete", "off"}:
        config = context.deps.set_chat_config(
            context.redis_client,
            context.chat_id,
            link_mode=value,
        )
    elif field := toggle_fields.get(action):
        current = context.deps.coerce_bool(config.get(field), default=True)
        config = context.deps.set_chat_config(
            context.redis_client,
            context.chat_id,
            **{field: not current},
        )
    elif action == "timezone":
        return _update_timezone(
            config,
            value,
            context=context,
        )
    elif action == "creditless":
        return _update_creditless_limit(
            config,
            value,
            context=context,
        )
    return config, False


def _callback_context(
    callback_query: dict[str, Any],
    deps: CallbackQueryDeps,
) -> CallbackContext | None:
    callback_data = callback_query.get("data")
    callback_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    user = callback_query.get("from") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")

    if deps.guard_callback(
        callback_id,
        not callback_data or chat_id is None or message_id is None,
    ):
        return None
    return CallbackContext(
        callback_id=str(callback_id) if callback_id else None,
        data=str(callback_data),
        chat_id=str(chat_id),
        chat_type=str(chat.get("type", "")),
        message_id=int(str(message_id)),
        user=user,
    )


def _route_feature_callback(
    callback_query: dict[str, Any],
    context: CallbackContext,
    deps: CallbackQueryDeps,
) -> bool:
    if context.data.startswith("topup:"):
        deps.handle_topup(callback_query)
        return True
    if context.data.startswith("task:"):
        deps.handle_task(callback_query)
        return True
    if not context.data.startswith("sig:"):
        return False
    deps.handle_signal(
        callback_query,
        redis_client=deps.config_redis(),
        delete_msg=deps.delete_msg,
        edit_photo=deps.edit_photo,
        is_chat_admin=deps.is_chat_admin,
        answer_callback_query=deps.answer_callback,
        admin_report=deps.admin_report,
    )
    return True


def _authorize_config_callback(
    context: CallbackContext,
    redis_client: Any,
    deps: CallbackQueryDeps,
) -> bool:
    if not (
        context.data.startswith("cfg:")
        and deps.is_group_chat_type(context.chat_type)
    ):
        return True
    if deps.is_chat_admin(
        context.chat_id,
        context.user.get("id"),
        redis_client=redis_client,
    ):
        return True
    deps.guard_callback(context.callback_id, True)
    deps.send_msg(
        context.chat_id,
        deps.denial_message,
        str(context.message_id),
    )
    deps.report_unauthorized(
        context.chat_id,
        context.user,
        chat_type=context.chat_type,
        action="callback:config",
        callback_data=context.data,
    )
    return False


def _render_config_callback(
    config: dict[str, Any],
    context: CallbackContext,
    deps: CallbackQueryDeps,
) -> None:
    text = deps.build_config_text(config, context.chat_type)
    keyboard = deps.build_config_keyboard(config, context.chat_type)
    try:
        edit_succeeded = deps.edit_message(
            context.chat_id,
            context.message_id,
            text,
            keyboard,
        )
        if not edit_succeeded:
            deps.config.log_event(
                "Falling back to new config message",
                {
                    "chat_id": context.chat_id,
                    "message_id": context.message_id,
                },
            )
            deps.send_msg(context.chat_id, text, reply_markup=keyboard)
    finally:
        # Telegram keeps the button spinner active until the callback is answered.
        if context.callback_id:
            deps.answer_callback(context.callback_id)


def handle_callback_query(
    callback_query: dict[str, Any],
    *,
    deps: CallbackQueryDeps,
) -> None:
    context = _callback_context(callback_query, deps)
    if context is None or _route_feature_callback(callback_query, context, deps):
        return

    redis_client = deps.config_redis()
    if not _authorize_config_callback(context, redis_client, deps):
        return

    config = deps.get_chat_config(redis_client, context.chat_id)
    try:
        _, action, value = context.data.split(":", 2)
    except ValueError:
        deps.guard_callback(context.callback_id, True)
        return

    config, handled = update_callback_config(
        config,
        action,
        value,
        context=ConfigUpdateContext(
            redis_client=redis_client,
            chat_id=context.chat_id,
            callback_id=context.callback_id,
            deps=deps.config,
        ),
    )
    if handled:
        return
    _render_config_callback(config, context, deps)
