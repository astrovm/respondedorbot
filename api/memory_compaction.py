from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import redis


@dataclass
class IncrementalSummarySource:
    prior_summary: str | None
    delta_messages: list[dict[str, Any]]
    is_zero_delta: bool
    next_marker: str | None


def build_incremental_summary_source(
    history: list[dict[str, Any]],
    existing_summary: str | None,
    compacted_until: str | None,
) -> IncrementalSummarySource:
    start_index = 0
    marker_found = False
    if compacted_until:
        for index, message in enumerate(history):
            if str(message.get("id")) == compacted_until:
                start_index = index + 1
                marker_found = True
                break
    if compacted_until and not marker_found:
        start_index = 0
    delta = history[start_index:]
    return IncrementalSummarySource(
        prior_summary=existing_summary,
        delta_messages=delta,
        is_zero_delta=not delta,
        next_marker=str(delta[-1].get("id")) if delta else None,
    )


def resolve_compaction_params(
    threshold: int | None,
    keep: int | None,
    *,
    default_threshold: int,
    default_keep: int,
) -> tuple[int, int]:
    return (
        threshold if threshold is not None else default_threshold,
        keep if keep is not None else default_keep,
    )


def compact_chat_memory(
    redis_client: redis.Redis | None,
    chat_id: str | None,
    messages: list[dict[str, Any]],
    existing_summary: str | None,
    compacted_until: str | None,
    *,
    compact_fn: Callable[
        [list[dict[str, Any]], str | None], tuple[str, int]
    ],
    compaction_threshold: int,
    compaction_keep: int,
    build_source: Callable[
        [list[dict[str, Any]], str | None, str | None],
        IncrementalSummarySource,
    ],
    save_summary: Callable[[redis.Redis, str, str], None],
    save_marker: Callable[[redis.Redis, str, str], None],
) -> tuple[str | None, list[dict[str, Any]], str | None, int]:
    if not messages:
        return existing_summary, [], compacted_until, 0

    source = build_source(messages, existing_summary, compacted_until)
    if source.is_zero_delta or len(source.delta_messages) <= compaction_threshold:
        return existing_summary, source.delta_messages, compacted_until, 0

    dropped = source.delta_messages[:-compaction_keep]
    if not dropped:
        return existing_summary, source.delta_messages, compacted_until, 0
    summary, cost = compact_fn(dropped, source.prior_summary)
    if not summary:
        return (
            existing_summary,
            source.delta_messages[-compaction_keep:],
            compacted_until,
            0,
        )

    marker = str(dropped[-1].get("id")) if dropped else compacted_until
    if redis_client is not None and chat_id:
        save_summary(redis_client, chat_id, summary)
        if marker:
            save_marker(redis_client, chat_id, marker)
    return summary, source.delta_messages[-compaction_keep:], marker, cost


def prepare_chat_memory(
    redis_client: redis.Redis | None,
    chat_id: str | None,
    chat_history: list[dict[str, Any]],
    query_text: str,
    *,
    reply_to_message_id: str | None,
    compaction_threshold: int,
    compaction_keep: int,
    get_summary: Callable[[redis.Redis, str], str | None],
    get_marker: Callable[[redis.Redis, str], str | None],
    fetch_full_history: Callable[..., list[dict[str, Any]]],
    compact_memory: Callable[..., tuple[
        str | None,
        list[dict[str, Any]],
        str | None,
        int,
    ]],
    search_history: Callable[..., list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str | None, list[dict[str, Any]], int]:
    summary = (
        get_summary(redis_client, chat_id)
        if redis_client is not None and chat_id
        else None
    )
    searchable = (
        fetch_full_history(
            redis_client,
            chat_id,
        )
        if redis_client is not None and chat_id
        else []
    )
    base_history = searchable if len(searchable) > len(chat_history) else chat_history
    marker = (
        get_marker(redis_client, chat_id)
        if redis_client is not None and chat_id and summary
        else None
    )
    summary, visible, _marker, cost = compact_memory(
        redis_client,
        chat_id,
        base_history,
        summary,
        marker,
        compaction_threshold=compaction_threshold,
        compaction_keep=compaction_keep,
    )
    recent_ids = {
        str(message.get("id"))
        for message in visible
        if message.get("id") is not None
    }
    retrieved = (
        search_history(
            redis_client,
            chat_id,
            query_text,
            reply_to_message_id=reply_to_message_id,
            limit=5,
            exclude_message_ids=recent_ids,
        )
        if redis_client is not None and chat_id and query_text.strip()
        else []
    )
    return visible, summary, retrieved, cost
