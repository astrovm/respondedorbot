"""Prepare AI requests, enrich their context, and call the provider layer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import logging
from typing import Any, Protocol, cast

from api.billing.authorization import (
    AI_COST_AUTHORIZER_KEY,
    AI_SEGMENT_RECORDER_KEY,
    AIAuthorizationDenied,
)
from api.core.rust_bridge import load_rust_bridge
from api.i18n import current_locale, tr


logger = logging.getLogger(__name__)


class _RustAIRequestSanitization(Protocol):
    def ai_sanitize_assistant_text(self, text: str) -> str: ...


class _RustAIImageContextPlanning(Protocol):
    def ai_plan_image_context(
        self,
        has_image_data: bool,
        description: str | None,
        last_text_content: str | None,
        localized_context: str,
    ) -> tuple[str, str | None]: ...


def _load_rust_ai_request_sanitization() -> _RustAIRequestSanitization | None:
    module = load_rust_bridge("RUST_AI_REQUEST_SANITIZATION_ENABLED")
    if module is None:
        return None
    return cast(_RustAIRequestSanitization, module)


def _load_rust_ai_image_context_planning() -> _RustAIImageContextPlanning | None:
    module = load_rust_bridge("RUST_AI_IMAGE_CONTEXT_PLANNING_ENABLED")
    if module is None:
        return None
    return cast(_RustAIImageContextPlanning, module)


def _plan_image_context(
    has_image_data: bool,
    description: str | None,
    last_text_content: str | None,
    localized_context: str,
) -> tuple[str, str | None]:
    rust = _load_rust_ai_image_context_planning()
    if rust is not None:
        try:
            action, updated_content = rust.ai_plan_image_context(
                has_image_data,
                description,
                last_text_content,
                localized_context,
            )
            action = str(action)
            if action not in {"no_image", "description_failed", "description_ready"}:
                raise ValueError(f"invalid Rust image-context action: {action}")
            if updated_content is not None and not isinstance(updated_content, str):
                raise ValueError("Rust image-context content must be text or null")
            if action == "description_ready" and description is None:
                raise ValueError("Rust cannot complete a missing image description")
            if action != "description_ready" and updated_content is not None:
                raise ValueError("Rust returned content for an incomplete image description")
            if updated_content is not None and last_text_content is None:
                raise ValueError("Rust returned image context without a text message tail")
            return action, updated_content
        except Exception:
            logger.exception(
                "Rust AI image-context planning failed; using Python fallback"
            )

    if not has_image_data:
        return "no_image", None
    if description is None:
        return "description_failed", None
    return (
        "description_ready",
        (
            f"{last_text_content}\n\n{localized_context}"
            if last_text_content is not None
            else None
        ),
    )


def _python_sanitize_assistant_text(text: str) -> str:
    content = text.lower()
    content = "".join(
        character for character in content if not (0x1F000 <= ord(character) <= 0x1FFFF)
    )
    return content.rstrip(".")


def _sanitize_assistant_text(text: str) -> str:
    rust = _load_rust_ai_request_sanitization()
    if rust is not None:
        try:
            return str(rust.ai_sanitize_assistant_text(text))
        except Exception:
            logger.exception(
                "Rust AI request sanitization failed; using Python fallback"
            )
    return _python_sanitize_assistant_text(text)


def _copy_billing_context(
    response_meta: dict[str, Any] | None,
    tool_context: dict[str, Any],
) -> None:
    if response_meta is None:
        return
    for key in (AI_COST_AUTHORIZER_KEY, AI_SEGMENT_RECORDER_KEY):
        if key in response_meta:
            tool_context[key] = response_meta[key]


def sanitize_bot_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize old assistant messages before sending them back to the model."""

    if message.get("role") != "assistant":
        return message
    content = message.get("content", "")
    if isinstance(content, str):
        content = _sanitize_assistant_text(content)
    elif isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = _sanitize_assistant_text(part["text"])
    return {**message, "content": content}


def get_stable_ai_context(
    timezone_offset: int,
    *,
    get_time_context: Callable[[int], dict[str, Any]],
) -> dict[str, Any]:
    """Build the small local context included in every request."""

    return {"time": get_time_context(timezone_offset)}


def build_ai_request(
    messages: list[dict[str, Any]],
    *,
    chat_id: str | None,
    user_name: str | None,
    user_id: int | None,
    timezone_offset: int,
    task_mode: bool,
    enable_web_search: bool,
    sanitize_message: Callable[[dict[str, Any]], dict[str, Any]],
    get_context: Callable[[int], dict[str, Any]],
    get_prices: Callable[..., Any],
    get_stock_prices: Callable[[str], str],
    select_random: Callable[[str], str],
    get_dollar_rates: Callable[[str], str | None],
    get_weather_context: Callable[[str], dict[str, Any] | None],
    get_hacker_news_context: Callable[[int], list[dict[str, Any]]],
    get_bot_capabilities: Callable[[], str],
    config_redis: Callable[..., Any],
    get_tool_schemas: Callable[..., list[dict[str, Any]]],
    build_system_message: Callable[..., dict[str, Any]],
    fetch_urls: Callable[[list[dict[str, Any]]], str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    dict[str, Any],
]:
    """Turn stored chat messages into one complete provider request."""

    # Old bot replies are cleaned before they become model context again.
    messages = [sanitize_message(message) for message in messages or []]
    context_data = get_context(timezone_offset)

    # Tool functions receive this dictionary when the model calls them.
    tool_context: dict[str, Any] = {
        "get_prices": get_prices,
        "get_stock_prices": get_stock_prices,
        "select_random": select_random,
        "get_dollar_rates": get_dollar_rates,
        "get_weather_context": get_weather_context,
        "get_hacker_news_context": get_hacker_news_context,
        "get_bot_capabilities": get_bot_capabilities,
        "config_redis": config_redis,
        "timezone_offset": timezone_offset,
        "locale": current_locale(),
    }
    if enable_web_search:
        tool_context["web_search_enabled"] = True
    if chat_id:
        tool_context["chat_id"] = chat_id
    if user_name:
        tool_context["user_name"] = user_name
    if user_id is not None:
        tool_context["user_id"] = user_id

    extra_tools = get_tool_schemas(tool_context, task_mode=task_mode)
    system_message = build_system_message(
        context_data,
        tools_active=bool(extra_tools),
        tool_schemas=extra_tools,
        task_mode=task_mode,
    )

    # URL contents are added as system context, not mixed into the user's words.
    fetched_contents = fetch_urls(messages) if enable_web_search else ""
    if fetched_contents:
        messages = [*messages, {"role": "system", "content": fetched_contents}]

    return system_message, messages, extra_tools, tool_context


def inject_image_context(
    messages: list[dict[str, Any]],
    image_data: bytes | None,
    image_file_id: str | None,
    response_meta: dict[str, Any] | None,
    *,
    describe_image: Callable[[bytes, str, str | None], Any],
    append_billing_segment: Callable[[dict[str, Any] | None, Any], None],
    logger: Any,
) -> None:
    """Describe an image once, then append that description to the prompt."""

    if image_data is None:
        return

    logger.info("vision model processing image")
    user_text = tr("media.image_prompt")
    image_result = describe_image(image_data, user_text, image_file_id)
    image_description = image_result.text if image_result else None
    description = str(image_description) if image_description else None
    image_context = (
        tr("media.image_context", description=description) if description else ""
    )
    last_content = None
    if messages and isinstance(messages[-1].get("content"), str):
        last_content = messages[-1]["content"]
    action, updated_content = _plan_image_context(
        True,
        description,
        last_content,
        image_context,
    )

    if action == "description_ready":
        # Vision is a separate billable provider call after Rust accepts the result.
        append_billing_segment(response_meta, image_result)
        if updated_content is not None:
            messages[-1]["content"] = updated_content
        logger.info("vision model described image, continuing ai flow")
    elif action == "description_failed":
        print("Failed to describe image, continuing without description...")


def ask_ai(
    messages: list[dict[str, Any]],
    *,
    image_data: bytes | None,
    image_file_id: str | None,
    response_meta: dict[str, Any] | None,
    enable_web_search: bool,
    chat_id: str | None,
    user_name: str | None,
    user_id: int | None,
    timezone_offset: int,
    task_mode: bool,
    build_request: Callable[
        ...,
        tuple[
            dict[str, Any],
            list[dict[str, Any]],
            list[dict[str, Any]] | None,
            dict[str, Any],
        ],
    ],
    inject_image: Callable[..., None],
    complete: Callable[..., str | None],
    fallback: Callable[[list[dict[str, Any]]], str],
    admin_report: Callable[..., None],
    logger: Any,
) -> str:
    """Run the full non-streaming request and fall back to a local reply."""

    try:
        system_message, messages, extra_tools, tool_context = build_request(
            messages,
            chat_id=chat_id,
            user_name=user_name,
            user_id=user_id,
            timezone_offset=timezone_offset,
            task_mode=task_mode,
            enable_web_search=enable_web_search,
        )
        _copy_billing_context(response_meta, tool_context)

        if image_data is not None:
            inject_image(messages, image_data, image_file_id, response_meta)

        response = complete(
            system_message,
            messages,
            response_meta=response_meta,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools or None,
            tool_context=tool_context,
        )
        response = str(response or "")
        if response:
            logger.info(
                "ask_ai response len=%d preview='%s'",
                len(response),
                response[:160].replace("\n", " "),
            )
            return response

        if response_meta is not None:
            response_meta["ai_fallback"] = True
        return fallback(messages)
    except AIAuthorizationDenied as error:
        if response_meta is not None:
            response_meta["authorization_denied"] = True
            response_meta["ai_fallback"] = True
        return str(error)
    except Exception as error:
        # A provider/config error should not make the Telegram handler crash.
        error_context = {
            "messages_count": len(messages),
            "messages_preview": [message.get("content", "")[:100] for message in messages],
        }
        admin_report("Error in ask_ai", error, error_context)
        if response_meta is not None:
            response_meta["ai_fallback"] = True
        return fallback(messages)


def ask_ai_stream(
    messages: list[dict[str, Any]],
    *,
    enable_web_search: bool,
    chat_id: str | None,
    user_name: str | None,
    user_id: int | None,
    timezone_offset: int,
    response_meta: dict[str, Any] | None,
    build_request: Callable[
        ...,
        tuple[
            dict[str, Any],
            list[dict[str, Any]],
            list[dict[str, Any]] | None,
            dict[str, Any],
        ],
    ],
    stream: Callable[..., Iterator[tuple[str, str]]],
) -> Iterator[tuple[str, str]]:
    system_message, messages, extra_tools, tool_context = build_request(
        messages,
        chat_id=chat_id,
        user_name=user_name,
        user_id=user_id,
        timezone_offset=timezone_offset,
        task_mode=False,
        enable_web_search=enable_web_search,
    )
    _copy_billing_context(response_meta, tool_context)
    stream_kwargs: dict[str, Any] = {
        "enable_web_search": enable_web_search,
        "extra_tools": extra_tools,
        "tool_context": tool_context,
    }
    if response_meta is not None:
        stream_kwargs["response_meta"] = response_meta
    token_iterator = stream(
        system_message,
        messages,
        **stream_kwargs,
    )

    def track_authorization_denial() -> Iterator[tuple[str, str]]:
        try:
            yield from token_iterator
        except AIAuthorizationDenied:
            if response_meta is not None:
                response_meta["authorization_denied"] = True
                response_meta["ai_fallback"] = True
            raise

    return track_authorization_denial()


@dataclass
class AIRequestServiceDeps:
    """Functions supplied by the composition root.

    Listing dependencies explicitly makes this service easy to test without
    network, Redis, or real provider clients.
    """

    get_weather_context: Callable[[str], dict[str, Any] | None]
    get_time_context: Callable[[int], dict[str, Any]]
    get_hacker_news_context: Callable[[int], list[dict[str, Any]]]
    get_prices: Callable[..., Any]
    get_stock_prices: Callable[[str], str]
    select_random: Callable[[str], str]
    get_dollar_rates: Callable[[str], str | None]
    get_bot_capabilities: Callable[[], str]
    config_redis: Callable[..., Any]
    get_tool_schemas: Callable[..., list[dict[str, Any]]]
    build_system_message: Callable[..., dict[str, Any]]
    fetch_urls: Callable[[list[dict[str, Any]]], str]
    describe_image: Callable[[bytes, str, str | None], Any]
    append_billing_segment: Callable[[dict[str, Any] | None, Any], None]
    complete: Callable[..., str | None]
    stream: Callable[..., Iterator[tuple[str, str]]]
    fallback: Callable[[list[dict[str, Any]]], str]
    admin_report: Callable[..., None]
    logger: Any


class AIRequestService:
    """Production-facing API for normal and streaming AI requests."""

    def __init__(self, deps: AIRequestServiceDeps) -> None:
        self._deps = deps

    def get_stable_context(self, timezone_offset: int = -3) -> dict[str, Any]:
        return get_stable_ai_context(
            timezone_offset,
            get_time_context=self._deps.get_time_context,
        )

    def build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        chat_id: str | None = None,
        user_name: str | None = None,
        user_id: int | None = None,
        timezone_offset: int = -3,
        task_mode: bool = False,
        enable_web_search: bool = True,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]] | None,
        dict[str, Any],
    ]:
        return build_ai_request(
            messages,
            chat_id=chat_id,
            user_name=user_name,
            user_id=user_id,
            timezone_offset=timezone_offset,
            task_mode=task_mode,
            enable_web_search=enable_web_search,
            sanitize_message=sanitize_bot_message,
            get_context=self.get_stable_context,
            get_prices=self._deps.get_prices,
            get_stock_prices=self._deps.get_stock_prices,
            select_random=self._deps.select_random,
            get_dollar_rates=self._deps.get_dollar_rates,
            get_weather_context=self._deps.get_weather_context,
            get_hacker_news_context=self._deps.get_hacker_news_context,
            get_bot_capabilities=self._deps.get_bot_capabilities,
            config_redis=self._deps.config_redis,
            get_tool_schemas=self._deps.get_tool_schemas,
            build_system_message=self._deps.build_system_message,
            fetch_urls=self._deps.fetch_urls,
        )

    def inject_image_context(
        self,
        messages: list[dict[str, Any]],
        image_data: bytes | None,
        image_file_id: str | None,
        response_meta: dict[str, Any] | None,
    ) -> None:
        inject_image_context(
            messages,
            image_data,
            image_file_id,
            response_meta,
            describe_image=self._deps.describe_image,
            append_billing_segment=self._deps.append_billing_segment,
            logger=self._deps.logger,
        )

    def ask(
        self,
        messages: list[dict[str, Any]],
        image_data: bytes | None = None,
        image_file_id: str | None = None,
        response_meta: dict[str, Any] | None = None,
        enable_web_search: bool = True,
        chat_id: str | None = None,
        user_name: str | None = None,
        user_id: int | None = None,
        timezone_offset: int = -3,
        task_mode: bool = False,
    ) -> str:
        return ask_ai(
            messages,
            image_data=image_data,
            image_file_id=image_file_id,
            response_meta=response_meta,
            enable_web_search=enable_web_search,
            chat_id=chat_id,
            user_name=user_name,
            user_id=user_id,
            timezone_offset=timezone_offset,
            task_mode=task_mode,
            build_request=self.build_request,
            inject_image=self.inject_image_context,
            complete=self._deps.complete,
            fallback=self._deps.fallback,
            admin_report=self._deps.admin_report,
            logger=self._deps.logger,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_web_search: bool = True,
        chat_id: str | None = None,
        user_name: str | None = None,
        user_id: int | None = None,
        timezone_offset: int = -3,
        response_meta: dict[str, Any] | None = None,
    ) -> Iterator[tuple[str, str]]:
        return ask_ai_stream(
            messages,
            enable_web_search=enable_web_search,
            chat_id=chat_id,
            user_name=user_name,
            user_id=user_id,
            timezone_offset=timezone_offset,
            response_meta=response_meta,
            build_request=self.build_request,
            stream=self._deps.stream,
        )
