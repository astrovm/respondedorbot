from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from api.logging_config import get_logger

logger = get_logger(__name__)


def extract_poll_text(poll: Mapping[str, Any]) -> str:
    question = str(poll.get("question", "")).strip()
    option_texts = []
    for option in poll.get("options") or []:
        option_text = option.get("text") if isinstance(option, Mapping) else None
        if option_text:
            option_texts.append(str(option_text).strip())

    if not option_texts:
        return question

    options = "\n".join(f"- {option_text}" for option_text in option_texts)
    return f"{question}\nOpciones:\n{options}" if question else f"Opciones:\n{options}"


def extract_message_text(message: dict[str, Any]) -> str:
    parts = []
    if message.get("text"):
        parts.append(str(message["text"]).strip())
    if message.get("caption"):
        parts.append(str(message["caption"]).strip())
    poll = message.get("poll")
    if isinstance(poll, dict):
        poll_text = extract_poll_text(poll)
        if poll_text:
            parts.append(poll_text)
    return "\n\n".join(parts)


def sticker_vision_file_id(sticker: Mapping[str, Any]) -> str | None:
    if sticker.get("is_animated") or sticker.get("is_video"):
        thumbnail = sticker.get("thumbnail") or sticker.get("thumb")
        if isinstance(thumbnail, Mapping):
            thumbnail_file_id = thumbnail.get("file_id")
            if thumbnail_file_id:
                return str(thumbnail_file_id)

    file_id = sticker.get("file_id")
    return str(file_id) if file_id else None


def extract_message_content(
    message: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    text = extract_message_text(message)
    photo_file_id = _extract_visual_file_id(message)
    audio_file_id = _extract_audio_file_id(message)
    return text, photo_file_id, audio_file_id


def _extract_visual_file_id(message: dict[str, Any]) -> str | None:
    if message.get("photo"):
        return str(message["photo"][-1]["file_id"])
    if message.get("sticker"):
        file_id = sticker_vision_file_id(message["sticker"])
        logger.debug("media detected type=sticker file_id=%s", file_id)
        return file_id

    replied = message.get("reply_to_message") or {}
    if replied.get("photo"):
        file_id = str(replied["photo"][-1]["file_id"])
        logger.debug("media detected type=photo_quoted file_id=%s", file_id)
        return file_id
    if replied.get("sticker"):
        file_id = sticker_vision_file_id(replied["sticker"])
        logger.debug("media detected type=sticker_quoted file_id=%s", file_id)
        return file_id
    return None


def _extract_audio_file_id(message: dict[str, Any]) -> str | None:
    for media_type in ("voice", "audio", "video", "video_note"):
        media = message.get(media_type)
        if media:
            file_id = str(media["file_id"])
            if media_type in {"video", "video_note"}:
                logger.debug(
                    "media detected type=%s file_id=%s", media_type, file_id
                )
            return file_id

    replied = message.get("reply_to_message") or {}
    for media_type in ("voice", "audio", "video", "video_note"):
        media = replied.get(media_type)
        if media:
            file_id = str(media["file_id"])
            logger.debug(
                "media detected type=%s_quoted file_id=%s", media_type, file_id
            )
            return file_id
    return None
