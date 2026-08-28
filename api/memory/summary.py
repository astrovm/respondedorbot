"""Keep long conversations useful without sending every old message to the AI.

Old messages are compressed into a summary. Recent messages stay verbatim, and
search can bring back older messages that are relevant to the current request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Any

import redis

from api.i18n import tr
from api.i18n.prompts import prompt
from api.memory import compaction as memory_compaction
from api.ai.pricing import (
    AIUsageResult,
    calculate_billing_for_segments,
    credit_units_from_usd_micros,
    ensure_mapping,
)
from api.providers.pricing import needs_published_provider_pricing
from api.memory.background import DurableCompactionQueue
from api.memory.compaction import CompactionPlan, IncrementalSummarySource


def call_summary_model(
    messages: list[dict[str, Any]],
    *,
    get_client: Callable[[], Any],
    estimate_tokens: Callable[[list[dict[str, Any]]], int],
    estimate_cost: Callable[[int, int, str], int],
    model: str,
    max_tokens: int,
    logger: Any,
    get_provider_pricing: Callable[[str, str, str | None], Mapping[str, int] | None] | None = None,
) -> tuple[str | None, int, dict[str, Any] | None]:
    """Ask the summary model for text and return its measured credit cost."""

    client = get_client()
    if client is None:
        logger.warning("summary: no openrouter client available")
        return None, 0, None

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
            usage = ensure_mapping(response.usage) or {}
            input_tokens = int(usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(usage.get("completion_tokens", 0) or 0)
            finish_reason = response.choices[0].finish_reason
            resolved_model = str(getattr(response, "model", None) or model)
            upstream_provider = getattr(response, "provider", None)
            service_tier = getattr(response, "service_tier", None)
            response_metadata = {
                "provider": "openrouter",
                "provider_generation_id": getattr(response, "id", None),
                "provider_request_id": getattr(response, "_request_id", None),
            }
            if resolved_model != model:
                response_metadata["requested_model"] = model
            if upstream_provider:
                response_metadata["upstream_provider"] = str(upstream_provider)
                response_metadata.update(
                    {"service_tier": str(service_tier)} if service_tier else {}
                )
                if get_provider_pricing is not None and needs_published_provider_pricing(usage):
                    provider_pricing = get_provider_pricing(
                        resolved_model,
                        str(upstream_provider),
                        str(service_tier) if service_tier else None,
                    )
                    if provider_pricing:
                        response_metadata["provider_pricing"] = dict(provider_pricing)
                        response_metadata["provider_pricing_source"] = "openrouter_endpoints"
            segment = AIUsageResult(
                kind="summary",
                text=text,
                model=resolved_model,
                usage=usage,
                source="openrouter",
                metadata=response_metadata,
            ).billing_segment()
            billing = calculate_billing_for_segments([segment])
            cost = int(
                Decimal(str(billing["raw_usd_micros_exact"])).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
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
            return text, cost, segment
        logger.warning("summary: model=%s returned empty response", model)
    except Exception as error:
        logger.warning("summary: model=%s failed: %s", model, error)

    logger.error("summary: model failed")
    return None, 0, None


def build_chat_messages(
    bot_personality: str,
    messages: list[dict[str, Any]],
    prompt_text: str,
    prior_summary: str | None = None,
) -> list[dict[str, Any]]:
    api_messages: list[dict[str, Any]] = [{"role": "system", "content": bot_personality}]
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
        tuple[str | None, int] | tuple[str | None, int, dict[str, Any] | None],
    ],
    sanitize_text: Callable[[str], str],
    max_summary_messages: int,
    truncate_lines: int,
) -> tuple[str, int] | tuple[str, int, dict[str, Any] | None]:
    """Merge new messages into the previous compact conversation summary.

    If the model is unavailable, a small plain-text transcript is kept instead
    so the bot loses less context and the request can still continue.
    """

    if len(messages) > max_summary_messages:
        messages = messages[-max_summary_messages:]
    api_messages = build_chat_messages(
        load_personality(),
        messages,
        prompt("summary.compact", no_markdown=prompt("no_markdown")),
        prior_summary=prior_summary,
    )

    model_result = call_model(api_messages)
    result, cost = model_result[:2]
    billing_segment = model_result[2] if len(model_result) > 2 else None
    if result:
        compacted = tr("summary.context", summary=sanitize_text(result))
        return (compacted, cost, billing_segment) if len(model_result) > 2 else (compacted, cost)

    fallback_lines = []
    for message in messages:
        content = message.get("content") or message.get("text", "")
        if content:
            fallback_lines.append(f"{message.get('role', 'user')}: {content}")
    truncated = "\n".join(fallback_lines[:truncate_lines])
    fallback = f"[contexto anterior truncado: {truncated}]"
    return (fallback, 0, billing_segment) if len(model_result) > 2 else (fallback, 0)


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
    prepare_memory: Callable[..., Any],
    load_personality: Callable[[], str],
    build_provider: Callable[[], Any],
    sanitize_text: Callable[[str], str],
    max_tokens: int,
    logger: Any,
    model: str,
    response_meta: dict[str, Any] | None = None,
) -> tuple[Iterator[tuple[str, str]], str | None]:
    """Build and stream the `/summary` response for one chat.

    A chat with no new messages can return its cached summary immediately.
    Otherwise the configured provider streams a fresh summary token by token.
    """

    history = get_history(chat_id, redis_client)

    def record_usage(result: AIUsageResult) -> None:
        if response_meta is None:
            return
        segment = result.billing_segment()
        segment["kind"] = "summary"
        response_meta.setdefault("billing_segments", []).append(segment)

    def record_internal_cache(text: str) -> None:
        record_usage(
            AIUsageResult(
                kind="summary",
                text=text,
                model=model,
                source="cache",
                cached=True,
                metadata={"pricing_basis": "internal_cache"},
            )
        )

    if not history:
        logger.info("summary_stream: no history for chat_id=%s", chat_id)
        empty_text = tr("summary.empty")
        record_internal_cache(empty_text)

        def empty() -> Iterator[tuple[str, str]]:
            yield "none", empty_text

        return empty(), None

    prepared = prepare_memory(
        redis_client,
        chat_id,
        history,
        prompt_text,
    )
    visible_history, summary_text, _retrieved_messages, summary_cost = prepared[:4]
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
        record_internal_cache(sanitized)

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
            yield "none", tr("summary.error")

        return unavailable(), source.next_marker

    system_message = api_messages[0]
    messages = api_messages[1:]
    stream = provider.stream(
        system_message,
        messages,
        enable_web_search=False,
        max_tokens=max_tokens,
        on_usage_result=record_usage,
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


@dataclass
class SummaryServiceDeps:
    """Storage, provider, and tuning values required by SummaryService."""

    state: Any
    config: Any
    provider: Any
    estimate_tokens: Callable[[list[dict[str, Any]]], int]
    sanitize_text: Callable[[str], str]
    logger: Any
    model: str
    max_tokens: int
    compaction_threshold: int
    compaction_keep: int
    max_summary_messages: int
    truncate_lines: int
    pricing_by_model: Mapping[str, Mapping[str, int]]
    redis_factory: Callable[[], Any]
    credits: Any
    compaction_timeout_seconds: float


class SummaryService:
    """Coordinates conversation compaction and user-requested summaries."""

    def __init__(self, deps: SummaryServiceDeps) -> None:
        self._deps = deps
        self._background = DurableCompactionQueue(
            redis_factory=deps.redis_factory,
            compact=self.compact_conversation_for_billing,
            get_summary=deps.state.get_chat_summary,
            get_marker=deps.state.get_chat_compacted_until,
            save_result=deps.state.save_chat_compaction_result,
            estimate_reserve=self.estimate_compaction_reserve,
            settle_reservation=deps.credits.settle_ai_reservation_once,
            logger=deps.logger,
            admin_report=deps.provider.admin_report,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> int:
        return estimate_summary_cost_usd_micros(
            input_tokens,
            output_tokens,
            model,
            pricing_by_model=self._deps.pricing_by_model,
        )

    def call_model(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, int]:
        result, cost, _segment = self.call_model_with_billing(messages)
        return result, cost

    def call_model_with_billing(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, int, dict[str, Any] | None]:
        return call_summary_model(
            messages,
            get_client=lambda: self._deps.provider.get_openrouter_client(
                timeout=self._deps.compaction_timeout_seconds
            ),
            estimate_tokens=self._deps.estimate_tokens,
            estimate_cost=self.estimate_cost,
            model=self._deps.model,
            max_tokens=self._deps.max_tokens,
            logger=self._deps.logger,
            get_provider_pricing=self._deps.provider.get_openrouter_provider_pricing,
        )

    def load_personality(self) -> str:
        try:
            value = self._deps.config.load_bot_config().get("system_prompt", "")
            return value if isinstance(value, str) else ""
        except Exception:
            return ""

    def compact_conversation(
        self,
        messages: list[dict[str, Any]],
        prior_summary: str | None = None,
    ) -> tuple[str, int]:
        result = compact_conversation(
            messages,
            prior_summary,
            load_personality=self.load_personality,
            call_model=self.call_model,
            sanitize_text=self._deps.sanitize_text,
            max_summary_messages=self._deps.max_summary_messages,
            truncate_lines=self._deps.truncate_lines,
        )
        return result[0], result[1]

    def compact_conversation_for_billing(
        self,
        messages: list[dict[str, Any]],
        prior_summary: str | None = None,
    ) -> tuple[str, int, dict[str, Any] | None]:
        result = compact_conversation(
            messages,
            prior_summary,
            load_personality=self.load_personality,
            call_model=self.call_model_with_billing,
            sanitize_text=self._deps.sanitize_text,
            max_summary_messages=self._deps.max_summary_messages,
            truncate_lines=self._deps.truncate_lines,
        )
        if len(result) == 3:
            return result
        return result[0], result[1], None

    def estimate_compaction_reserve(self, plan: CompactionPlan) -> int:
        api_messages = build_chat_messages(
            self.load_personality(),
            plan.messages,
            prompt("summary.estimate"),
            prior_summary=plan.prior_summary,
        )
        input_tokens = self._deps.estimate_tokens(api_messages)
        cost = self.estimate_cost(input_tokens, self._deps.max_tokens, self._deps.model)
        return credit_units_from_usd_micros(cost)

    def schedule_compaction(self, plan: CompactionPlan | None, billing: Any) -> bool:
        if plan is None:
            return False
        return self._background.enqueue(plan, billing)

    def start_background_worker(self) -> None:
        self._background.start()

    def stop_background_worker(self) -> None:
        self._background.stop()

    def run_pending_compactions_once(self) -> int:
        return self._background.run_pending_once()

    def build_messages(
        self,
        source: IncrementalSummarySource,
        prompt_text: str,
    ) -> list[dict[str, Any]]:
        return build_summary_messages(
            source,
            prompt_text,
            load_personality=self.load_personality,
        )

    build_incremental_source = staticmethod(memory_compaction.build_incremental_summary_source)

    def build_provider(self) -> Any:
        return self._deps.provider.build_provider(
            model=self._deps.model,
            max_tool_rounds=1,
        )

    def stream_command(
        self,
        chat_id: str,
        redis_client: redis.Redis,
        prompt_text: str,
        *,
        response_meta: dict[str, Any] | None = None,
    ) -> tuple[Iterator[tuple[str, str]], str | None]:
        return stream_summary_command(
            chat_id,
            redis_client,
            prompt_text,
            get_history=self._deps.state.get_history,
            prepare_memory=self.prepare_memory,
            load_personality=self.load_personality,
            build_provider=self.build_provider,
            sanitize_text=self._deps.sanitize_text,
            max_tokens=self._deps.max_tokens,
            logger=self._deps.logger,
            model=self._deps.model,
            response_meta=response_meta,
        )

    def resolve_compaction_params(
        self,
        threshold: int | None = None,
        keep: int | None = None,
    ) -> tuple[int, int]:
        return memory_compaction.resolve_compaction_params(
            threshold,
            keep,
            default_threshold=self._deps.compaction_threshold,
            default_keep=self._deps.compaction_keep,
        )

    def compact_memory(
        self,
        redis_client: redis.Redis | None,
        chat_id: str | None,
        messages: list[dict[str, Any]],
        existing_summary: str | None,
        compacted_until: str | None,
        compact_fn: Callable[
            [list[dict[str, Any]], str | None],
            tuple[str, int],
        ]
        | None = None,
        compaction_threshold: int | None = None,
        compaction_keep: int | None = None,
    ) -> tuple[str | None, list[dict[str, Any]], str | None, int]:
        """Replace sufficiently old messages with one durable summary."""

        threshold, keep = self.resolve_compaction_params(
            compaction_threshold,
            compaction_keep,
        )
        return memory_compaction.compact_chat_memory(
            redis_client,
            chat_id,
            messages,
            existing_summary,
            compacted_until,
            compact_fn=compact_fn or self.compact_conversation,
            compaction_threshold=threshold,
            compaction_keep=keep,
            build_source=memory_compaction.build_incremental_summary_source,
            save_summary=self._deps.state.save_chat_summary,
            save_marker=self._deps.state.save_chat_compacted_until,
        )

    def prepare_memory(
        self,
        redis_client: redis.Redis | None,
        chat_id: str | None,
        chat_history: list[dict[str, Any]],
        query_text: str,
        reply_to_message_id: str | None = None,
        compaction_threshold: int | None = None,
        compaction_keep: int | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        str | None,
        list[dict[str, Any]],
        int,
        CompactionPlan | None,
    ]:
        """Prepare the smallest useful memory package for the next AI request.

        The result separates recent visible history, the compact summary, and
        older messages retrieved because they match the user's current query.
        """

        threshold, keep = self.resolve_compaction_params(
            compaction_threshold,
            compaction_keep,
        )
        return memory_compaction.prepare_chat_memory(
            redis_client,
            chat_id,
            chat_history,
            query_text,
            reply_to_message_id=reply_to_message_id,
            compaction_threshold=threshold,
            compaction_keep=keep,
            get_summary=self._deps.state.get_chat_summary,
            get_marker=self._deps.state.get_chat_compacted_until,
            fetch_full_history=self._deps.state.fetch_for_compaction,
            search_history=self._deps.state.search_history,
        )
