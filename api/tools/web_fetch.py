"""web_fetch tool — wraps fetch_url_content for agentic use."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool


def _execute_web_fetch(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    # Lazy import to avoid circular dependency: agent_tools imports from tools/.
    from api.agent_tools import fetch_url_content

    url = params.get("url", "")
    if not url:
        return ToolResult(output="no se proporciono una url")
    result = fetch_url_content(str(url))
    error = result.get("error")
    if error:
        return ToolResult(output=f"error obteniendo {url}: {error}")
    title = result.get("title") or ""
    content = result.get("content") or ""
    output_parts = []
    if title:
        output_parts.append(f"Titulo: {title}")
    output_parts.append(content)
    return ToolResult(
        output="\n".join(output_parts), metadata={"url": result.get("url", url)}
    )


register_tool(
    name="web_fetch",
    description="Fetch and extract text content from a URL. Returns the page title and visible text content.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch and read",
            },
        },
        "required": ["url"],
    },
    executor=_execute_web_fetch,
)
