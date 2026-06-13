from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def handle_task_callback(
    callback_query: dict[str, Any],
    *,
    guard_callback: Callable[..., bool],
    list_tasks: Callable[[str], list[dict[str, Any]]],
    cancel_task: Callable[[str], Any],
    is_group_chat_type: Callable[[str], bool],
    config_redis: Callable[..., Any],
    is_chat_admin: Callable[..., bool],
    answer_callback: Callable[..., None],
    build_tasks_message: Callable[..., tuple[str, dict[str, Any] | None]],
    edit_message: Callable[..., Any],
    logger: Any,
) -> None:
    callback_data = callback_query.get("data")
    callback_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    user = callback_query.get("from") or {}

    if guard_callback(callback_id, not callback_data or chat_id is None):
        return

    parts = str(callback_data).split(":", 2)
    if guard_callback(callback_id, len(parts) != 3 or parts[0] != "task"):
        return

    task_id = parts[2]
    tasks = list_tasks(str(chat_id))
    target_task = next(
        (task for task in tasks if str(task.get("id")) == str(task_id)),
        None,
    )
    if guard_callback(
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
    if is_group_chat_type(chat_type):
        redis_client = config_redis()
        is_admin = is_chat_admin(
            str(chat_id),
            request_user_id,
            redis_client=redis_client,
        )
    else:
        is_admin = True

    if guard_callback(
        callback_id,
        not is_owner and not is_admin,
        text="solo el creador o un admin pueden borrar esta tarea",
        show_alert=True,
    ):
        return

    cancel_task(task_id)
    if callback_id:
        answer_callback(callback_id, text=f"tarea {task_id} borrada")

    if message_id:
        tasks = list_tasks(str(chat_id))
        new_text, new_keyboard = build_tasks_message(tasks)
        try:
            edit_message(str(chat_id), int(message_id), new_text, new_keyboard)
        except Exception as error:
            logger.exception(
                "handle_task_callback: failed to edit task message "
                "chat_id=%s message_id=%s: %s",
                chat_id,
                message_id,
                error,
            )


def update_callback_config(
    config: dict[str, Any],
    action: str,
    value: str,
    *,
    redis_client: Any,
    chat_id: str,
    callback_id: str | None,
    set_chat_config: Callable[..., dict[str, Any]],
    coerce_bool: Callable[..., bool],
    guard_callback: Callable[..., bool],
    log_config_event: Callable[[str, Mapping[str, Any] | None], None],
    timezone_offset_min: int,
    timezone_offset_max: int,
) -> tuple[dict[str, Any], bool]:
    if action == "link" and value in {"reply", "delete", "off"}:
        config = set_chat_config(redis_client, chat_id, link_mode=value)
    elif action == "random":
        current = coerce_bool(config.get("ai_random_replies"), default=True)
        config = set_chat_config(
            redis_client, chat_id, ai_random_replies=not current
        )
    elif action == "followups":
        current = coerce_bool(
            config.get("ai_command_followups"), default=True
        )
        config = set_chat_config(
            redis_client, chat_id, ai_command_followups=not current
        )
    elif action == "linkfixfollowups":
        current = coerce_bool(
            config.get("ignore_link_fix_followups"), default=True
        )
        config = set_chat_config(
            redis_client,
            chat_id,
            ignore_link_fix_followups=not current,
        )
    elif action == "timezone":
        if guard_callback(callback_id, value == "current"):
            return config, True
        try:
            offset = max(
                timezone_offset_min,
                min(int(value), timezone_offset_max),
            )
            config = set_chat_config(
                redis_client, chat_id, timezone_offset=offset
            )
        except ValueError:
            log_config_event(
                "Invalid timezone callback value",
                {"chat_id": chat_id, "value": value},
            )
    elif action == "creditless":
        try:
            current_limit = int(
                config.get(
                    "creditless_user_hourly_limit",
                    config.get("creditless_user_daily_limit", 5),
                )
            )
            if value == "none":
                limit = 0
            elif value == "decrease":
                limit = (
                    current_limit
                    if current_limit < 0
                    else max(0, current_limit - 1)
                )
            elif guard_callback(callback_id, value == "current"):
                return config, True
            elif value == "increase":
                limit = current_limit if current_limit < 0 else current_limit + 1
            elif value == "unlimited":
                limit = -1
            else:
                limit = int(value)
                if limit < -1:
                    raise ValueError
            config = set_chat_config(
                redis_client,
                chat_id,
                creditless_user_hourly_limit=limit,
            )
        except ValueError:
            log_config_event(
                "Invalid creditless callback value",
                {"chat_id": chat_id, "value": value},
            )
    return config, False


def handle_callback_query(
    callback_query: dict[str, Any],
    *,
    guard_callback: Callable[..., bool],
    handle_topup: Callable[[dict[str, Any]], None],
    handle_task: Callable[[dict[str, Any]], None],
    handle_signal: Callable[..., bool],
    config_redis: Callable[..., Any],
    delete_msg: Callable[..., Any],
    edit_photo: Callable[..., Any],
    is_chat_admin: Callable[..., bool],
    answer_callback: Callable[..., None],
    admin_report: Callable[..., None],
    is_group_chat_type: Callable[[str], bool],
    send_msg: Callable[..., Any],
    report_unauthorized: Callable[..., None],
    denial_message: str,
    get_chat_config: Callable[..., dict[str, Any]],
    set_chat_config: Callable[..., dict[str, Any]],
    coerce_bool: Callable[..., bool],
    log_config_event: Callable[[str, Mapping[str, Any] | None], None],
    timezone_offset_min: int,
    timezone_offset_max: int,
    build_config_text: Callable[..., str],
    build_config_keyboard: Callable[..., dict[str, Any]],
    edit_message: Callable[..., bool],
) -> None:
    callback_data = callback_query.get("data")
    callback_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    user = callback_query.get("from") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")

    if guard_callback(
        callback_id,
        not callback_data or chat_id is None or message_id is None,
    ):
        return

    callback_data_text = str(callback_data)
    # Route feature-owned payloads before treating the rest as config updates.
    if callback_data_text.startswith("topup:"):
        handle_topup(callback_query)
        return
    if callback_data_text.startswith("task:"):
        handle_task(callback_query)
        return
    if callback_data_text.startswith("sig:"):
        redis_client = config_redis()
        handle_signal(
            callback_query,
            redis_client=redis_client,
            delete_msg=delete_msg,
            edit_photo=edit_photo,
            is_chat_admin=is_chat_admin,
            answer_callback_query=answer_callback,
            admin_report=admin_report,
        )
        return

    redis_client = config_redis()
    chat_id_str = str(chat_id)
    chat_type = str(chat.get("type", ""))

    if callback_data_text.startswith("cfg:") and is_group_chat_type(chat_type):
        if not is_chat_admin(
            chat_id_str, user.get("id"), redis_client=redis_client
        ):
            guard_callback(callback_id, True)
            send_msg(chat_id_str, denial_message, str(message_id))
            report_unauthorized(
                chat_id_str,
                user,
                chat_type=chat_type,
                action="callback:config",
                callback_data=callback_data_text,
            )
            return

    config = get_chat_config(redis_client, chat_id_str)
    try:
        _, action, value = callback_data_text.split(":", 2)
    except ValueError:
        guard_callback(callback_id, True)
        return

    config, handled = update_callback_config(
        config,
        action,
        value,
        redis_client=redis_client,
        chat_id=chat_id_str,
        callback_id=callback_id,
        set_chat_config=set_chat_config,
        coerce_bool=coerce_bool,
        guard_callback=guard_callback,
        log_config_event=log_config_event,
        timezone_offset_min=timezone_offset_min,
        timezone_offset_max=timezone_offset_max,
    )
    if handled:
        return

    text = build_config_text(config, chat_type)
    keyboard = build_config_keyboard(config, chat_type)
    try:
        edit_succeeded = edit_message(
            chat_id_str, int(str(message_id)), text, keyboard
        )
        if not edit_succeeded:
            log_config_event(
                "Falling back to new config message",
                {"chat_id": chat_id_str, "message_id": message_id},
            )
            send_msg(chat_id_str, text, reply_markup=keyboard)
    finally:
        # Telegram keeps the button spinner active until the callback is answered.
        if callback_id:
            answer_callback(callback_id)
