from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import redis

from api.memory_compaction import IncrementalSummarySource


def call_summary_model(
    messages: list[dict[str, Any]],
    *,
    get_client: Callable[[], Any],
    estimate_tokens: Callable[[list[dict[str, Any]]], int],
    estimate_cost: Callable[[int, int, str], int],
    model: str,
    max_tokens: int,
    logger: Any,
) -> tuple[str | None, int]:
    client = get_client()
    if client is None:
        logger.warning("summary: no openrouter client available")
        return None, 0

    prompt_tokens = estimate_tokens(messages)
    logger.info(
        "summary: calling model=%s max_tokens=%d prompt_tokens_est=%d",
        model,
        max_tokens,
        prompt_tokens,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        if response and response.choices and response.choices[0].message:
            text = str(response.choices[0].message.content or "").strip()
            usage = response.usage or {}
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            finish_reason = response.choices[0].finish_reason
            cost = estimate_cost(input_tokens, output_tokens, model)
            logger.info(
                "summary: model=%s input=%d output=%d finish_reason=%s "
                "cost_usd_micros=%d text_len=%d",
                model,
                input_tokens,
                output_tokens,
                finish_reason,
                cost,
                len(text),
            )
            if not text:
                logger.warning("summary: model=%s returned empty text", model)
            elif finish_reason == "length":
                logger.warning(
                    "summary: model=%s hit max_tokens, output truncated",
                    model,
                )
            return text, cost
        logger.warning("summary: model=%s returned empty response", model)
    except Exception as error:
        logger.warning("summary: model=%s failed: %s", model, error)

    logger.error("summary: model failed")
    return None, 0


def build_chat_messages(
    bot_personality: str,
    messages: list[dict[str, Any]],
    prompt_text: str,
    prior_summary: str | None = None,
) -> list[dict[str, Any]]:
    api_messages: list[dict[str, Any]] = [
        {"role": "system", "content": bot_personality}
    ]
    if prior_summary:
        api_messages.append({"role": "assistant", "content": prior_summary})
    for message in messages:
        content = message.get("content") or message.get("text", "")
        if content:
            api_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": content,
                }
            )
    api_messages.append({"role": "user", "content": prompt_text})
    return api_messages


def compact_conversation(
    messages: list[dict[str, Any]],
    prior_summary: str | None,
    *,
    load_personality: Callable[[], str],
    call_model: Callable[
        [list[dict[str, Any]]],
        tuple[str | None, int],
    ],
    sanitize_text: Callable[[str], str],
    no_markdown_prompt: str,
    max_summary_messages: int,
    truncate_lines: int,
) -> tuple[str, int]:
    if len(messages) > max_summary_messages:
        messages = messages[-max_summary_messages:]
    api_messages = build_chat_messages(
        load_personality(),
        messages,
        (
            "actualizá el resumen previo con los mensajes nuevos. "
            "usá formato denso: temas, hechos clave, decisiones y pendientes. "
            "omití saludos y chat casual. mantené el idioma original. "
            f"{no_markdown_prompt}"
        ),
        prior_summary=prior_summary,
    )

    result, cost = call_model(api_messages)
    if result:
        return f"[contexto anterior: {sanitize_text(result)}]", cost

    fallback_lines = []
    for message in messages:
        content = message.get("content") or message.get("text", "")
        if content:
            fallback_lines.append(f"{message.get('role', 'user')}: {content}")
    truncated = "\n".join(fallback_lines[:truncate_lines])
    return f"[contexto anterior truncado: {truncated}]", 0


def build_summary_messages(
    source: IncrementalSummarySource,
    prompt_text: str,
    *,
    load_personality: Callable[[], str],
) -> list[dict[str, Any]]:
    return build_chat_messages(
        load_personality(),
        source.delta_messages,
        prompt_text,
        prior_summary=source.prior_summary,
    )


def wrap_provider_stream(
    provider_name: str,
    token_iter: Iterator[str],
    *,
    logger: Any,
) -> Iterator[tuple[str, str]]:
    try:
        for token in token_iter:
            yield provider_name, token
    except Exception:
        logger.exception("summary_stream: provider=%s failed", provider_name)
        raise


def stream_summary_command(
    chat_id: str,
    redis_client: redis.Redis,
    prompt_text: str,
    *,
    get_history: Callable[[str, redis.Redis], list[dict[str, Any]]],
    prepare_memory: Callable[..., tuple[
        list[dict[str, Any]],
        str | None,
        list[dict[str, Any]],
        int,
    ]],
    load_personality: Callable[[], str],
    build_provider: Callable[[], Any],
    sanitize_text: Callable[[str], str],
    max_tokens: int,
    logger: Any,
) -> tuple[Iterator[tuple[str, str]], str | None]:
    history = get_history(chat_id, redis_client)

    if not history:
        logger.info("summary_stream: no history for chat_id=%s", chat_id)

        def empty() -> Iterator[tuple[str, str]]:
            yield "none", "no hay mensajes para resumir"

        return empty(), None

    visible_history, summary_text, _retrieved_messages, summary_cost = (
        prepare_memory(
            redis_client,
            chat_id,
            history,
            prompt_text,
        )
    )
    source = IncrementalSummarySource(
        prior_summary=summary_text,
        delta_messages=visible_history,
        is_zero_delta=not visible_history,
        next_marker=None,
    )
    logger.info(
        "summary_stream: chat_id=%s history=%d visible=%d zero_delta=%s "
        "has_prior=%s compaction_cost_usd_micros=%d",
        chat_id,
        len(history),
        len(source.delta_messages),
        source.is_zero_delta,
        bool(source.prior_summary),
        summary_cost,
    )
    if source.is_zero_delta and source.prior_summary:
        sanitized = sanitize_text(source.prior_summary)

        def yield_cached() -> Iterator[tuple[str, str]]:
            yield "cache", sanitized

        return yield_cached(), None

    api_messages = build_summary_messages(
        source,
        prompt_text,
        load_personality=load_personality,
    )
    provider = build_provider()
    logger.info(
        "summary_stream: chat_id=%s provider_available=%s messages=%d",
        chat_id,
        provider.is_available(),
        len(api_messages),
    )
    if not provider.is_available():

        def unavailable() -> Iterator[tuple[str, str]]:
            yield "none", "no pude generar el resumen"

        return unavailable(), source.next_marker

    system_message = api_messages[0]
    messages = api_messages[1:]
    stream = provider.stream(
        system_message,
        messages,
        enable_web_search=False,
        max_tokens=max_tokens,
    )
    return (
        wrap_provider_stream(provider.name, stream, logger=logger),
        None,
    )


def estimate_summary_cost_usd_micros(
    input_tokens: int,
    output_tokens: int,
    model: str,
    *,
    pricing_by_model: Mapping[str, Mapping[str, int]],
) -> int:
    pricing = pricing_by_model.get(model, {})
    input_rate = pricing.get("input_per_million", 100_000)
    output_rate = pricing.get("output_per_million", 400_000)
    return (input_tokens * input_rate + output_tokens * output_rate) // 1_000_000
