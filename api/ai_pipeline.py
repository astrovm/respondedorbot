"""AI response cleanup and normalization helpers."""

from __future__ import annotations

import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence


GORDO_PREFIX_PATTERN = re.compile(r"^\s*gordo\b\s*:\s*", re.IGNORECASE)
LOG_PREVIEW_LIMIT = 160


def _preview_for_log(text: Optional[str], limit: int = LOG_PREVIEW_LIMIT) -> str:
    """Return a single-line preview suitable for debug logging."""

    preview = str(text or "").replace("\n", " ").strip()
    if len(preview) <= limit:
        return preview
    return preview[:limit] + "..."


def remove_gordo_prefix(text: Optional[str]) -> str:
    """Strip leading 'gordo:' persona prefix from each line of a response."""

    if not text:
        return ""

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        cleaned_lines.append(GORDO_PREFIX_PATTERN.sub("", line, count=1))

    return "\n".join(cleaned_lines).strip()


def clean_duplicate_response(response: str) -> str:
    """Remove duplicate consecutive text in AI responses."""

    if not response:
        return response

    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and (not cleaned_lines or stripped != cleaned_lines[-1]):
            cleaned_lines.append(stripped)

    cleaned_response = "\n".join(cleaned_lines)
    sentences = cleaned_response.split(". ")
    cleaned_sentences = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and (not cleaned_sentences or stripped != cleaned_sentences[-1]):
            cleaned_sentences.append(stripped)

    return ". ".join(cleaned_sentences).replace("..", ".")


def strip_leading_context(
    response: str,
    contexts: Optional[Sequence[Optional[str]]],
) -> str:
    """Remove leading echoes of known context strings from an AI response."""

    if not response or not contexts:
        return response

    trimmed_response = response
    normalized_contexts = [
        str(context).strip()
        for context in contexts
        if context is not None and str(context).strip()
    ]
    if not normalized_contexts:
        return trimmed_response

    changed = True
    max_passes = len(normalized_contexts)
    passes = 0

    while changed and passes < max_passes:
        changed = False
        passes += 1
        for context in normalized_contexts:
            if trimmed_response.lower().startswith(context.lower()):
                trimmed_response = trimmed_response[len(context) :].lstrip(" \t:-\n")
                changed = True
                break

    return trimmed_response


def strip_user_identity_prefix(response: str, user_identity: Optional[str]) -> str:
    """Remove a leading '<user>:' prefix that sometimes leaks into completions."""

    if not response or not user_identity:
        return response

    normalized_identity = str(user_identity).strip()
    if not normalized_identity:
        return response

    pattern = re.compile(rf"^\s*{re.escape(normalized_identity)}\s*:\s*", re.IGNORECASE)
    return pattern.sub("", response, count=1).lstrip()


def handle_ai_response(
    chat_id: str,
    handler_func: Callable[..., Any],
    messages: List[Dict[str, Any]],
    *,
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    context_texts: Optional[Sequence[Optional[str]]] = None,
    user_identity: Optional[str] = None,
    response_meta: Optional[Dict[str, Any]] = None,
    send_typing_fn: Callable[[str, str], None],
    telegram_token: Optional[str],
    reset_request_count_fn: Callable[[], Any],
    restore_request_count_fn: Callable[[Any], None],
    get_request_count_fn: Callable[[], int],
    strip_ai_fallback_marker_fn: Callable[[str], Any],
) -> str:
    """Handle AI API responses and apply the response cleanup pipeline."""

    if telegram_token:
        send_typing_fn(telegram_token, chat_id)
    time.sleep(random.uniform(0, 1))

    handler_name = getattr(handler_func, "__name__", "")
    is_ask_ai_handler = handler_name == "ask_ai"
    request_count_token: Optional[Any] = None
    if is_ask_ai_handler:
        request_count_token = reset_request_count_fn()

    user_name = ""
    if user_identity:
        user_name = user_identity.strip().split()[0] if user_identity.strip() else ""

    try:
        if image_data and is_ask_ai_handler:
            response = handler_func(
                messages,
                image_data=image_data,
                image_file_id=image_file_id,
                response_meta=response_meta,
                chat_id=chat_id,
                user_name=user_name,
            )
        elif is_ask_ai_handler:
            if response_meta is not None:
                try:
                    response = handler_func(
                        messages,
                        response_meta=response_meta,
                        chat_id=chat_id,
                        user_name=user_name,
                    )
                except TypeError:
                    response = handler_func(messages, response_meta=response_meta)
            else:
                response = handler_func(
                    messages,
                    chat_id=chat_id,
                    user_name=user_name,
                )
        else:
            if response_meta is not None:
                try:
                    response = handler_func(messages, response_meta=response_meta)
                except TypeError:
                    response = handler_func(messages)
            else:
                response = handler_func(messages)
    finally:
        if request_count_token is not None:
            if response_meta is not None:
                response_meta["provider_request_count"] = get_request_count_fn()
            restore_request_count_fn(request_count_token)

    response_text = str(response or "")
    response_text, used_ai_fallback = strip_ai_fallback_marker_fn(response_text)
    if response_meta is not None:
        response_meta["ai_fallback"] = used_ai_fallback

    persona_stripped_response = remove_gordo_prefix(response_text)
    context_stripped_response = strip_leading_context(
        persona_stripped_response, context_texts
    )
    prefix_stripped_response = strip_user_identity_prefix(
        context_stripped_response, user_identity
    )
    cleaned_response = clean_duplicate_response(prefix_stripped_response)

    if not cleaned_response.strip():
        print(
            "handle_ai_response: cleaned response empty after normalization "
            f"handler={handler_name or '<unknown>'} ai_fallback={used_ai_fallback} "
            f"raw_len={len(response_text)} "
            f"persona_len={len(persona_stripped_response)} "
            f"context_len={len(context_stripped_response)} "
            f"prefix_len={len(prefix_stripped_response)}"
        )
        print(
            "handle_ai_response: previews "
            f"raw='{_preview_for_log(response_text)}' "
            f"persona='{_preview_for_log(persona_stripped_response)}' "
            f"context='{_preview_for_log(context_stripped_response)}' "
            f"prefix='{_preview_for_log(prefix_stripped_response)}'"
        )
        return "me quedé reculando y no te pude responder, probá de nuevo"

    return cleaned_response


__all__ = [
    "clean_duplicate_response",
    "handle_ai_response",
    "remove_gordo_prefix",
]
