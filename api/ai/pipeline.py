"""AI response cleanup and normalization helpers."""

from __future__ import annotations

import re
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

_logger = logging.getLogger(__name__)

GORDO_PREFIX_PATTERN = re.compile(r"^\s*gordo\b\s*:\s*", re.IGNORECASE)
IDENTITY_USER_PATTERN = re.compile(r"\(@?(\w+)\)")
MARKDOWN_CODE_FENCE_PATTERN = re.compile(r"```(?:[^\n`]*)\n?(.*?)```", re.DOTALL)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
MARKDOWN_BOLD_PATTERN = re.compile(r"(?<!\w)(\*\*|__)(\S(?:.*?\S)?)\1(?!\w)")
MARKDOWN_ITALIC_PATTERN = re.compile(r"(?<![\w:/])(\*|_)([^\s*_](?:.*?[^\s*_])?)\1(?!\w)")
MARKDOWN_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
MARKDOWN_HEADER_PATTERN = re.compile(r"^\s{0,3}#{2,6}\s+", re.MULTILINE)
MARKDOWN_HRULE_PATTERN = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,})\s*$", re.MULTILINE)
MARKDOWN_BLOCKQUOTE_PATTERN = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
MARKDOWN_BULLET_PATTERN = re.compile(r"^\s{0,3}[-*]\s+", re.MULTILINE)
LOG_PREVIEW_LIMIT = 160

INSTRUCCIONES_BASE = [
    "INSTRUCCIONES:",
    "- mantené el personaje del gordo",
    "- usá lenguaje coloquial argentino",
    "- respondé en minúsculas, sin emojis, sin punto final",
    "- respondé en una sola frase salvo que sea necesario explicar algo complejo",
]


@dataclass(frozen=True)
class AIResponseRequest:
    chat_id: str
    handler: Callable[..., Any]
    messages: List[Dict[str, Any]]
    image_data: Optional[bytes] = None
    image_file_id: Optional[str] = None
    context_texts: Optional[Sequence[Optional[str]]] = None
    user_identity: Optional[str] = None
    response_meta: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    timezone_offset: int = -3
    reply_to_message_id: Optional[str] = None


@dataclass(frozen=True)
class AIResponseRuntime:
    send_typing: Callable[[str, str], None]
    telegram_token: Optional[str]
    reset_request_count: Callable[[], Any]
    restore_request_count: Callable[[Any], None]
    get_request_count: Callable[[], int]


@dataclass(frozen=True)
class _CleanupStages:
    raw: str
    persona: str
    context: str
    identity: str
    final: str


def _extract_user_name(user_identity: Optional[str]) -> str:
    if not user_identity:
        return ""
    m = IDENTITY_USER_PATTERN.search(user_identity)
    if m:
        return f"@{m.group(1)}"
    stripped = user_identity.strip()
    if stripped:
        return stripped.split()[0]
    return ""


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
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and (not cleaned_lines or stripped != cleaned_lines[-1]):
            cleaned_lines.append(stripped)

    cleaned_response = "\n".join(cleaned_lines)
    sentences = cleaned_response.split(". ")
    cleaned_sentences: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and (not cleaned_sentences or stripped != cleaned_sentences[-1]):
            cleaned_sentences.append(stripped)

    return ". ".join(cleaned_sentences).replace("..", ".")


def strip_markdown_formatting(text: str) -> str:
    """Remove common markdown markers while preserving their readable content."""

    if not text:
        return ""

    cleaned = MARKDOWN_CODE_FENCE_PATTERN.sub(r"\1", text)
    cleaned = MARKDOWN_IMAGE_PATTERN.sub(r"\1", cleaned)
    cleaned = MARKDOWN_LINK_PATTERN.sub(r"\1", cleaned)
    cleaned = MARKDOWN_INLINE_CODE_PATTERN.sub(r"\1", cleaned)
    cleaned = MARKDOWN_HEADER_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_HRULE_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_BLOCKQUOTE_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_BULLET_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_BOLD_PATTERN.sub(r"\2", cleaned)
    cleaned = MARKDOWN_ITALIC_PATTERN.sub(r"\2", cleaned)

    return "\n".join(line for line in cleaned.splitlines() if line.strip()).strip()


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


def _invoke_ask_ai(
    request: AIResponseRequest,
    *,
    user_name: str,
) -> Any:
    kwargs: Dict[str, Any] = {
        "response_meta": request.response_meta,
        "chat_id": request.chat_id,
        "user_name": user_name,
        "user_id": request.user_id,
        "timezone_offset": request.timezone_offset,
    }
    if request.image_data:
        kwargs["image_data"] = request.image_data
        kwargs["image_file_id"] = request.image_file_id
    return request.handler(request.messages, **kwargs)


def _generic_handler_kwargs(request: AIResponseRequest) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if request.response_meta is not None:
        kwargs["response_meta"] = request.response_meta
    if request.reply_to_message_id is not None:
        kwargs["reply_to_message_id"] = request.reply_to_message_id
    if request.image_data is not None:
        kwargs["image_data"] = request.image_data
        kwargs["image_file_id"] = request.image_file_id
    return kwargs


def _invoke_generic_handler(request: AIResponseRequest) -> Any:
    kwargs = _generic_handler_kwargs(request)
    if not kwargs:
        return request.handler(request.messages)

    try:
        parameters = inspect.signature(request.handler).parameters
    except (TypeError, ValueError):
        return request.handler(request.messages)

    accepted_kwargs = {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return request.handler(request.messages, **accepted_kwargs)


def _invoke_handler(request: AIResponseRequest, handler_name: str) -> Any:
    if handler_name == "ask_ai":
        return _invoke_ask_ai(
            request,
            user_name=_extract_user_name(request.user_identity),
        )
    return _invoke_generic_handler(request)


def _clean_response(request: AIResponseRequest, response: Any) -> _CleanupStages:
    raw = str(response or "")
    persona = remove_gordo_prefix(raw)
    context = strip_leading_context(persona, request.context_texts)
    identity = strip_user_identity_prefix(context, request.user_identity)
    final = strip_markdown_formatting(clean_duplicate_response(identity))
    return _CleanupStages(
        raw=raw,
        persona=persona,
        context=context,
        identity=identity,
        final=final,
    )


def _empty_response_fallback(
    request: AIResponseRequest,
    handler_name: str,
    cleanup: _CleanupStages,
) -> str:
    was_fallback = (
        bool(request.response_meta.get("ai_fallback"))
        if request.response_meta
        else False
    )
    if request.response_meta is not None:
        request.response_meta["ai_fallback"] = True
    _logger.warning(
        "cleaned response empty handler=%s ai_fallback=%s raw_len=%d",
        handler_name or "<unknown>",
        was_fallback,
        len(cleanup.raw),
    )
    _logger.debug(
        "empty response previews raw='%s' persona='%s' context='%s' prefix='%s'",
        _preview_for_log(cleanup.raw),
        _preview_for_log(cleanup.persona),
        _preview_for_log(cleanup.context),
        _preview_for_log(cleanup.identity),
    )
    return "me quedé reculando y no te pude responder, probá de nuevo"


def handle_ai_response(
    request: AIResponseRequest,
    runtime: AIResponseRuntime,
) -> str:
    """Handle AI API responses and apply the response cleanup pipeline."""

    if runtime.telegram_token:
        runtime.send_typing(runtime.telegram_token, request.chat_id)

    handler_name = getattr(request.handler, "__name__", "")
    is_ask_ai_handler = handler_name == "ask_ai"
    request_count_token: Optional[Any] = None
    if is_ask_ai_handler:
        request_count_token = runtime.reset_request_count()

    try:
        response = _invoke_handler(request, handler_name)
    finally:
        if request_count_token is not None:
            if request.response_meta is not None:
                request.response_meta["provider_request_count"] = (
                    runtime.get_request_count()
                )
            runtime.restore_request_count(request_count_token)

    cleanup = _clean_response(request, response)
    if cleanup.final.strip():
        return cleanup.final
    return _empty_response_fallback(request, handler_name, cleanup)


__all__ = [
    "INSTRUCCIONES_BASE",
    "AIResponseRequest",
    "AIResponseRuntime",
    "clean_duplicate_response",
    "handle_ai_response",
    "remove_gordo_prefix",
    "strip_markdown_formatting",
]
