from __future__ import annotations

import re
from os import environ
from typing import Any, Callable, Dict, List, Optional, Tuple

from urllib.parse import urlparse

import requests

from api.agent_tools import normalize_http_url


_MAX_TELEGRAM_TEXT_LENGTH = 4096


def _truncate_telegram_text(text: str) -> str:
    if len(text) <= _MAX_TELEGRAM_TEXT_LENGTH:
        return text
    truncated = text[: _MAX_TELEGRAM_TEXT_LENGTH - 3]
    last_newline = truncated.rfind("\n")
    if last_newline > _MAX_TELEGRAM_TEXT_LENGTH * 0.8:
        truncated = truncated[:last_newline]
    return truncated + "..."


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
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": _truncate_telegram_text(msg)}
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

    def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        msg_id: str = "",
        caption: str = "",
    ) -> Optional[int]:
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "animation": animation_url,
        }
        if msg_id:
            payload["reply_to_message_id"] = msg_id
        if caption:
            payload["caption"] = caption

        payload_response, error = self._telegram_request(
            "sendAnimation", method="POST", json_payload=payload
        )
        if error or not payload_response:
            return None

        result = payload_response.get("result")
        if isinstance(result, dict):
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                return message_id

        return None

    def download_file(self, file_id: str) -> Optional[bytes]:
        token = environ.get("TELEGRAM_TOKEN")
        if not token:
            return None

        payload_response, error = self._telegram_request(
            "getFile",
            method="GET",
            params={"file_id": file_id},
            log_errors=False,
        )
        if error or not payload_response:
            return None

        result = payload_response.get("result")
        if not isinstance(result, dict):
            return None

        file_path = result.get("file_path")
        if not isinstance(file_path, str):
            return None

        file_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
        try:
            download_response = requests.get(file_url, timeout=30)
            download_response.raise_for_status()
            return download_response.content
        except requests.RequestException:
            return None
