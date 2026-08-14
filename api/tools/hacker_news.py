"""Fetch Hacker News stories requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.ai.prompt_context import format_hacker_news_info
from api.tools.registry import ToolResult, register_tool


def _execute_hacker_news(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_news = context.get("get_hacker_news_context")
    if get_news is None:
        return ToolResult(output="Hacker News lookup not available")

    raw_limit = params.get("limit", 5)
    try:
        limit = max(1, min(10, int(raw_limit)))
    except TypeError, ValueError:
        limit = 5
    stories = get_news(limit)
    return ToolResult(output=format_hacker_news_info(stories))


register_tool(
    name="hacker_news",
    description="Get current top technology stories from Hacker News.",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of stories to return. Defaults to 5.",
            },
        },
        "required": [],
    },
    executor=_execute_hacker_news,
    requires_context=["get_hacker_news_context"],
)
