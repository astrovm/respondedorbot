from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from urllib.parse import urlparse

from api.agent_tools import normalize_http_url


def _message_has_domain_link(message: str, domain: str) -> bool:
    if not message:
        return False

    normalized_domain = domain.lower().strip(".")
    if not normalized_domain:
        return False

    candidates = re.findall(
        r"(https?://[^\s<>()]+|www\.[^\s<>()]+|(?<!@)\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>()]*)?)",
        message,
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        cleaned_candidate = candidate.rstrip(".,!?;:)]}'\"")
        normalized_url = normalize_http_url(cleaned_candidate)
        if not normalized_url:
            continue
        hostname = (urlparse(normalized_url).hostname or "").lower()
        if hostname == normalized_domain or hostname.endswith(f".{normalized_domain}"):
            return True

    return False


class TelegramGateway:
    def __init__(
        self,
        telegram_request: Callable[..., Tuple[Optional[Dict[str, Any]], Optional[str]]],
        message_has_domain_link: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self._telegram_request = telegram_request
        self._message_has_domain_link = (
            message_has_domain_link or _message_has_domain_link
        )

    def send_message(
        self,
        chat_id: str,
        msg: str,
        msg_id: str = "",
        buttons: Optional[List[str]] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": msg}
        if self._message_has_domain_link(msg, "polymarket.com"):
            payload["disable_web_page_preview"] = True
        if msg_id:
            payload["reply_to_message_id"] = msg_id

        markup = reply_markup
        if markup is None and buttons:
            keyboard = [[{"text": "abrir en la app", "url": url}] for url in buttons]
            markup = {"inline_keyboard": keyboard}

        if markup is not None:
            payload["reply_markup"] = markup

        payload_response, error = self._telegram_request(
            "sendMessage", method="POST", json_payload=payload
        )
        if error or not payload_response:
            return None

        result = payload_response.get("result")
        if isinstance(result, dict):
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                return message_id

        return None

    def delete_message(self, chat_id: str, msg_id: str) -> None:
        self._telegram_request(
            "deleteMessage",
            method="GET",
            params={"chat_id": chat_id, "message_id": msg_id},
            log_errors=False,
            expect_json=False,
        )
