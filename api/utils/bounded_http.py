"""Read streamed HTTP responses with a hard memory bound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundedHttpResponse:
    url: str
    status: int | None
    content_type: str
    body: bytes
    encoding: str | None
    apparent_encoding: str | None
    truncated: bool


def read_bounded_response(
    response: Any,
    *,
    max_bytes: int,
    fallback_url: str,
) -> BoundedHttpResponse:
    """Consume at most ``max_bytes`` and always close the response."""

    url = str(getattr(response, "url", "") or fallback_url)
    status = getattr(response, "status_code", None)
    content_type = str(response.headers.get("Content-Type", "")).lower()
    encoding = getattr(response, "encoding", None)
    apparent_encoding = getattr(response, "apparent_encoding", None)
    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                truncated = True
                break
            chunks.append(chunk[:remaining])
            total += min(len(chunk), remaining)
            if len(chunk) > remaining:
                truncated = True
                break
    finally:
        try:
            response.close()
        except Exception:
            pass
    return BoundedHttpResponse(
        url=url,
        status=status if isinstance(status, int) else None,
        content_type=content_type,
        body=b"".join(chunks),
        encoding=str(encoding) if encoding else None,
        apparent_encoding=(
            str(apparent_encoding) if apparent_encoding else None
        ),
        truncated=truncated,
    )


def decode_bounded_body(response: BoundedHttpResponse) -> str:
    try:
        return response.body.decode(
            response.encoding or response.apparent_encoding or "utf-8",
            errors="replace",
        )
    except (LookupError, UnicodeError):
        return response.body.decode("utf-8", errors="replace")
