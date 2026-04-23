from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple


class LinkReplacementDeps(Protocol):
    def replace_links(self, text: str) -> Tuple[str, bool, List[str]]: ...

    def build_message_links_context(self, message: Mapping[str, Any]) -> str: ...

    def delete_msg(self, chat_id: str, message_id: str) -> None: ...

    def send_msg(self, *args: Any, **kwargs: Any) -> Optional[int]: ...

    def save_message_to_redis(self, *args: Any, **kwargs: Any) -> None: ...


def handle_link_replacement(
    deps: LinkReplacementDeps,
    *,
    chat_config: Mapping[str, Any],
    message: Dict[str, Any],
    message_text: str,
    chat_id: str,
    message_id: str,
    redis_client: Any,
) -> bool:
    link_mode = str(chat_config.get("link_mode", "reply"))
    if link_mode == "off" or not message_text or message_text.startswith("/"):
        return False

    fixed_text, changed, original_links = deps.replace_links(message_text)
    if not changed:
        return False

    user_info = message.get("from", {})
    username = user_info.get("username")
    if username:
        shared_by = f"@{username}"
    else:
        name_parts = [
            part
            for part in (user_info.get("first_name"), user_info.get("last_name"))
            if part
        ]
        shared_by = " ".join(name_parts)

    if shared_by:
        fixed_text += f"\n\ncompartido por {shared_by}"

    link_context = deps.build_message_links_context({"text": fixed_text})
    stored_bot_message = fixed_text
    if link_context:
        stored_bot_message = f"{stored_bot_message}\n\n{link_context}"

    reply_id = message.get("reply_to_message", {}).get("message_id")
    reply_id = str(reply_id) if reply_id is not None else None

    if link_mode == "delete":
        deps.delete_msg(chat_id, message_id)
        if reply_id:
            sent_message_id = deps.send_msg(
                chat_id, fixed_text, reply_id, original_links
            )
        else:
            sent_message_id = deps.send_msg(chat_id, fixed_text, buttons=original_links)
        if sent_message_id is not None:
            deps.save_message_to_redis(
                chat_id,
                f"bot_{sent_message_id}",
                stored_bot_message,
                redis_client,
            )
        return True

    sent_message_id = deps.send_msg(
        chat_id, fixed_text, reply_id or message_id, original_links
    )
    if sent_message_id is not None:
        deps.save_message_to_redis(
            chat_id,
            f"bot_{sent_message_id}",
            stored_bot_message,
            redis_client,
        )
    return True


__all__ = ["handle_link_replacement"]
