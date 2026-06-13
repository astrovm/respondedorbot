from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


DEFAULT_TRANSCRIPTION_ERROR_MESSAGES = {
    "download": "no pude bajar el audio, mandalo de nuevo",
    "duration": "no pude medir la duración del audio",
    "transcribe": "no pude sacar nada de ese audio, probá más tarde",
}


def transcription_error_message(
    error_code: str | None,
    *,
    download_message: str | None = None,
    transcribe_message: str | None = None,
) -> str | None:
    if not error_code:
        return None
    if error_code == "download":
        return download_message or DEFAULT_TRANSCRIPTION_ERROR_MESSAGES["download"]
    return transcribe_message or DEFAULT_TRANSCRIPTION_ERROR_MESSAGES["transcribe"]


def describe_replied_media(
    replied_msg: Mapping[str, Any],
    *,
    media_key: str,
    extract_file_id: Callable[[Any], str | None],
    prompt: str,
    success_prefix: str,
    download_error: str,
    describe_error: str,
    describe_media: Callable[..., tuple[
        str | None,
        str | None,
        dict[str, Any] | None,
    ]],
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
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and value
            ):
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
    transcribe_audio_file: Callable[..., tuple[
        str | None,
        str | None,
        dict[str, Any] | None,
    ]],
    error_message: Callable[..., str | None],
    describe_media: Callable[..., tuple[
        str | None,
        dict[str, Any] | None,
    ]],
    sticker_file_id: Callable[[Mapping[str, Any]], str | None],
    logger: Any,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        if "reply_to_message" not in message:
            return (
                "respondeme un audio, video, imagen o sticker y te digo qué carajo hay ahí",
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
                    f"🎵 te saqué esto del audio: {text}",
                    [billing_segment] if billing_segment else [],
                )
            resolved_error = error_message(error_code)
            if resolved_error:
                return resolved_error, []
            return DEFAULT_TRANSCRIPTION_ERROR_MESSAGES["transcribe"], []

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
                    prompt=(
                        "describí lo que ves en esta imagen en detalle, "
                        "en minúsculas, sin emojis, sin markdown, "
                        "en lenguaje coloquial argentino"
                    ),
                    success_prefix="🖼️ en la imagen veo: ",
                    download_error="no pude bajar la imagen, mandala de nuevo",
                    describe_error=(
                        "no pude sacar qué mierda tiene la imagen, probá más tarde"
                    ),
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
                        sticker_file_id(media)
                        if isinstance(media, Mapping)
                        else None
                    ),
                    prompt=(
                        "describí lo que ves en este sticker en detalle, "
                        "en minúsculas, sin emojis, sin markdown, "
                        "en lenguaje coloquial argentino"
                    ),
                    success_prefix="🎨 en el sticker veo: ",
                    download_error="no pude bajar el sticker, mandalo de nuevo",
                    describe_error=(
                        "no pude sacar qué carajo tiene el sticker, probá más tarde"
                    ),
                )
                if sticker_response:
                    return (
                        sticker_response,
                        [billing_segment] if billing_segment else [],
                    )

        return (
            "ese mensaje no tiene audio, video, imagen ni sticker para laburar",
            [],
        )
    except Exception as error:
        logger.exception("handle_transcribe failed: %s", error)
        return "se trabó el /transcribe, probá más tarde", []
