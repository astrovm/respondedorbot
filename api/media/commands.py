from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from api.i18n import tr


def transcription_error_message(
    error_code: str | None,
    *,
    download_message: str | None = None,
    transcribe_message: str | None = None,
) -> str | None:
    if not error_code:
        return None
    if error_code == "download":
        return download_message or tr("media.download_audio")
    return transcribe_message or tr("media.transcribe_audio")


def describe_replied_media(
    replied_msg: Mapping[str, Any],
    *,
    media_key: str,
    extract_file_id: Callable[[Any], str | None],
    prompt: str,
    success_prefix: str,
    download_error: str,
    describe_error: str,
    describe_media: Callable[
        ...,
        tuple[
            str | None,
            str | None,
            dict[str, Any] | None,
        ],
    ],
    sanitize_text: Callable[[str], str],
) -> tuple[str | None, dict[str, Any] | None]:
    media = replied_msg.get(media_key)
    if not media:
        return None, None
    file_id = extract_file_id(media)
    if not file_id:
        return None, None

    description, error_code, billing_segment = describe_media(file_id, prompt)
    if description:
        return f"{success_prefix}{sanitize_text(description)}", billing_segment
    if error_code == "download":
        return download_error, None
    return describe_error, None


def find_media_message(
    container: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any] | None:
    current: Mapping[str, Any] | None = container
    while isinstance(current, Mapping):
        value = current.get(key)
        if key == "photo":
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
                return current
        elif value:
            return current
        next_message = current.get("reply_to_message")
        current = next_message if isinstance(next_message, Mapping) else None
    return None


def handle_transcribe_with_message_result(
    message: dict[str, Any],
    *,
    extract_message_content: Callable[..., tuple[Any, str | None, str | None]],
    transcribe_audio_file: Callable[
        ...,
        tuple[
            str | None,
            str | None,
            dict[str, Any] | None,
        ],
    ],
    error_message: Callable[..., str | None],
    describe_media: Callable[
        ...,
        tuple[
            str | None,
            dict[str, Any] | None,
        ],
    ],
    sticker_file_id: Callable[[Mapping[str, Any]], str | None],
    logger: Any,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        if "reply_to_message" not in message:
            return (
                tr("media.reply_required"),
                [],
            )

        replied_msg = message["reply_to_message"]
        _, photo_file_id, audio_file_id = extract_message_content(replied_msg)

        if audio_file_id:
            text, error_code, billing_segment = transcribe_audio_file(
                audio_file_id,
                use_cache=True,
            )
            if text:
                return (
                    tr("media.audio_result", text=text),
                    [billing_segment] if billing_segment else [],
                )
            resolved_error = error_message(error_code)
            if resolved_error:
                return resolved_error, []
            return tr("media.transcribe_audio"), []

        if photo_file_id:
            photo_source = find_media_message(replied_msg, "photo")
            if photo_source:
                photo_response, billing_segment = describe_media(
                    photo_source,
                    media_key="photo",
                    extract_file_id=lambda media: (
                        media[-1]["file_id"]
                        if isinstance(media, Sequence)
                        and not isinstance(media, (str, bytes))
                        and media
                        else None
                    ),
                    prompt=tr("media.image_prompt"),
                    success_prefix=tr("media.image_result"),
                    download_error=tr("media.image_download"),
                    describe_error=tr("media.image_error"),
                )
                if photo_response:
                    return (
                        photo_response,
                        [billing_segment] if billing_segment else [],
                    )

            sticker_source = find_media_message(replied_msg, "sticker")
            if sticker_source:
                sticker_response, billing_segment = describe_media(
                    sticker_source,
                    media_key="sticker",
                    extract_file_id=lambda media: (
                        sticker_file_id(media) if isinstance(media, Mapping) else None
                    ),
                    prompt=tr("media.sticker_prompt"),
                    success_prefix=tr("media.sticker_result"),
                    download_error=tr("media.sticker_download"),
                    describe_error=tr("media.sticker_error"),
                )
                if sticker_response:
                    return (
                        sticker_response,
                        [billing_segment] if billing_segment else [],
                    )

        return (
            tr("media.none"),
            [],
        )
    except Exception as error:
        logger.exception("handle_transcribe failed: %s", error)
        return tr("media.command_error"), []
