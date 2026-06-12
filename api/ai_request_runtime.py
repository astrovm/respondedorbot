from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


def sanitize_bot_message(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") != "assistant":
        return message
    content = message.get("content", "")
    if isinstance(content, str):
        content = content.lower()
        content = "".join(
            character
            for character in content
            if not (0x1F000 <= ord(character) <= 0x1FFFF)
        )
        content = content.rstrip(".")
    elif isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = part["text"].lower()
                part["text"] = "".join(
                    character
                    for character in part["text"]
                    if not (0x1F000 <= ord(character) <= 0x1FFFF)
                )
                part["text"] = part["text"].rstrip(".")
    return {**message, "content": content}


def get_stable_ai_context(
    timezone_offset: int,
    *,
    cache: dict[int, tuple[int, dict[str, Any]]],
    ttl: int,
    now: Callable[[], float],
    get_market_context: Callable[[], dict[str, Any]],
    get_weather_context: Callable[[], dict[str, Any] | None],
    get_time_context: Callable[[int], dict[str, Any]],
    get_hacker_news_context: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    timestamp = int(now())
    cached = cache.get(timezone_offset)
    if cached and timestamp - cached[0] <= ttl:
        return cached[1]

    context = {
        "market": get_market_context(),
        "weather": get_weather_context(),
        "time": get_time_context(timezone_offset),
        "hacker_news": get_hacker_news_context(),
    }
    cache[timezone_offset] = (timestamp, context)
    return context


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
    messages = [sanitize_message(message) for message in messages or []]
    context_data = get_context(timezone_offset)

    tool_context: dict[str, Any] = {
        "get_prices": get_prices,
        "config_redis": config_redis,
        "timezone_offset": timezone_offset,
    }
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
    if image_data is None:
        return

    logger.info("vision model processing image")
    user_text = (
        "describí lo que ves en esta imagen en detalle, "
        "en minúsculas, sin emojis, sin markdown, en lenguaje coloquial argentino"
    )
    image_result = describe_image(image_data, user_text, image_file_id)
    image_description = image_result.text if image_result else None

    if image_description:
        append_billing_segment(response_meta, image_result)
        image_context = f"[Imagen: {image_description}]"
        if messages:
            last_message = messages[-1]
            if isinstance(last_message.get("content"), str):
                last_message["content"] += f"\n\n{image_context}"
        logger.info("vision model described image, continuing ai flow")
    else:
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
    build_request: Callable[..., tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]] | None,
        dict[str, Any],
    ]],
    inject_image: Callable[..., None],
    complete: Callable[..., str | None],
    fallback: Callable[[list[dict[str, Any]]], str],
    admin_report: Callable[..., None],
    logger: Any,
) -> str:
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
    except Exception as error:
        error_context = {
            "messages_count": len(messages),
            "messages_preview": [
                message.get("content", "")[:100] for message in messages
            ],
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
    build_request: Callable[..., tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]] | None,
        dict[str, Any],
    ]],
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
    return stream(
        system_message,
        messages,
        enable_web_search=enable_web_search,
        extra_tools=extra_tools,
        tool_context=tool_context,
    )
