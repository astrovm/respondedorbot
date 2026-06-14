"""Replace supported social links before normal message processing."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol

from api.links.service import LinkServiceProtocol


class LinkReplacementDeps(Protocol):
    @property
    def link_service(self) -> LinkServiceProtocol: ...

    def delete_msg(self, chat_id: str, message_id: str) -> None: ...

    def send_msg(self, *args: Any, **kwargs: Any) -> Optional[int]: ...

    def send_video(self, *args: Any, **kwargs: Any) -> Optional[int]: ...

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

    fixed_text, changed, original_links = deps.link_service.replace(message_text)
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

    link_context = deps.link_service.build_context({"text": fixed_text})
    stored_bot_message = fixed_text
    if link_context:
        stored_bot_message = f"{stored_bot_message}\n\n{link_context}"

    reply_id = message.get("reply_to_message", {}).get("message_id")
    reply_id = str(reply_id) if reply_id is not None else None
    oversized_video = deps.link_service.download_oversized_instagram_video(fixed_text)

    def send_replacement(target_reply_id: Optional[str]) -> Optional[int]:
        if oversized_video is not None:
            sent_video_id = deps.send_video(
                chat_id,
                oversized_video,
                caption=fixed_text,
                msg_id=target_reply_id or "",
                buttons=original_links,
            )
            if sent_video_id is not None:
                return sent_video_id
        if target_reply_id:
            return deps.send_msg(
                chat_id,
                fixed_text,
                target_reply_id,
                original_links,
            )
        return deps.send_msg(chat_id, fixed_text, buttons=original_links)

    if link_mode == "delete":
        deps.delete_msg(chat_id, message_id)
        sent_message_id = send_replacement(reply_id)
        if sent_message_id is not None:
            deps.save_message_to_redis(
                chat_id,
                f"bot_{sent_message_id}",
                stored_bot_message,
                redis_client,
            )
        return True

    sent_message_id = send_replacement(reply_id or message_id)
    if sent_message_id is not None:
        deps.save_message_to_redis(
            chat_id,
            f"bot_{sent_message_id}",
            stored_bot_message,
            redis_client,
        )
    return True


__all__ = ["handle_link_replacement"]
