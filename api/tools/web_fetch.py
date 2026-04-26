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
    from api.utils.links import fetch_tweet_content

    url = params.get("url", "")
    if not url:
        return ToolResult(output="no se proporciono una url")
    tweet = fetch_tweet_content(str(url))
    if tweet:
        tweet_error = tweet.get("error")
        if tweet_error:
            return ToolResult(output=tweet_error, metadata={"url": tweet.get("url", url)})
        output_parts = []
        author = tweet.get("author") or ""
        date = tweet.get("date") or ""
        if author or date:
            heading = f"Tweet de {author}" if author else "Tweet"
            if date:
                heading = f"{heading} · {date}"
            output_parts.append(heading)
        text = tweet.get("text") or ""
        if text:
            output_parts.append(text)
        return ToolResult(
            output="\n".join(output_parts) or "tweet sin texto legible",
            metadata={"url": tweet.get("url", url)},
        )
    result = fetch_url_content(str(url))
    error = result.get("error")
    if error:
        return ToolResult(output=f"error obteniendo {url}: {error}")
    title = result.get("title") or ""
    content = result.get("content") or ""
    if "Something went wrong" in content and "Try again" in content:
        return ToolResult(
            output="error obteniendo la pagina: X devolvio una pagina de error",
            metadata={"url": result.get("url", url)},
        )
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
