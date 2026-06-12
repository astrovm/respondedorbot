from __future__ import annotations

import json
import re
from os import environ
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from urllib.parse import urlparse

from api.agent_tools import normalize_http_url
from api.services import http_client


_MAX_TELEGRAM_TEXT_LENGTH = 4096


def _redact_telegram_tokens(value: str) -> str:
    return re.sub(r"/bot[^/\s]+/", "/bot<redacted>/", value)


def telegram_request(
    endpoint: str,
    *,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    data_payload: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    timeout: int = 5,
    token: Optional[str] = None,
    log_errors: bool = True,
    expect_json: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Perform a Telegram Bot API request and return its parsed payload."""

    resolved_token = token or environ.get("TELEGRAM_TOKEN")
    if not resolved_token:
        error_msg = "Telegram token not configured"
        if log_errors:
            print(error_msg)
        return None, error_msg

    url = f"https://api.telegram.org/bot{resolved_token}/{endpoint}"
    method_upper = method.upper()
    try:
        if method_upper == "GET" and json_payload is None:
            response = http_client.get(url, params=params, timeout=timeout)
        elif method_upper == "POST" and params is None and files is None:
            response = http_client.post(url, json=json_payload, timeout=timeout)
        elif method_upper == "POST" and files is not None:
            response = http_client.post(
                url,
                data=data_payload,
                files=files,
                timeout=timeout,
            )
        else:
            response = http_client.request(
                method_upper,
                url,
                params=params,
                json=json_payload,
                data=data_payload,
                files=files,
                timeout=timeout,
            )
        response.raise_for_status()
        if not expect_json:
            return {}, None
    except requests.RequestException as error:
        error_payload = None
        error_description = str(error)
        if error.response is not None:
            try:
                parsed_error = error.response.json()
                if isinstance(parsed_error, dict):
                    error_payload = parsed_error
                    error_description = str(parsed_error.get("description") or error)
            except Exception:
                pass
        is_not_modified = "message is not modified" in error_description.lower()
        if log_errors and not is_not_modified:
            detail = _redact_telegram_tokens(str(error))
            response_body = ""
            if error.response is not None:
                try:
                    body = _redact_telegram_tokens(error.response.text)
                    response_body = f" response={body[:500]!r}"
                except Exception:
                    pass
            print(f"Telegram request to {endpoint} failed: {detail}{response_body}")
        return error_payload, error_description

    try:
        payload = response.json()
    except ValueError as exc:
        if log_errors:
            print(f"Telegram request to {endpoint} returned invalid JSON: {exc}")
        return None, str(exc)

    if not isinstance(payload, dict):
        if log_errors:
            print(f"Telegram request to {endpoint} returned unexpected payload type")
        return None, "unexpected response"

    if not payload.get("ok"):
        description = str(payload.get("description") or "telegram request failed")
        if log_errors:
            print(f"Telegram request to {endpoint} returned ok=false: {description}")
        return payload, description

    return payload, None


def send_typing(token: str, chat_id: str) -> None:
    telegram_request(
        "sendChatAction",
        method="GET",
        params={"chat_id": chat_id, "action": "typing"},
        token=token,
        log_errors=False,
        expect_json=False,
    )


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
            if '<a href="https://polymarket.com/' in msg:
                payload["parse_mode"] = "HTML"
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

    def send_photo(
        self,
        chat_id: str,
        photo: bytes,
        *,
        caption: str = "",
        msg_id: str = "",
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        payload: Dict[str, Any] = {"chat_id": chat_id}
        if msg_id:
            payload["reply_to_message_id"] = msg_id
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "HTML"
            payload["disable_web_page_preview"] = True
        if reply_markup is not None:
            payload["reply_markup"] = json.dumps(reply_markup)

        payload_response, error = self._telegram_request(
            "sendPhoto",
            method="POST",
            data_payload=payload,
            files={"photo": ("chart.png", photo, "image/png")},
        )
        if error or not payload_response:
            return None

        result = payload_response.get("result")
        if isinstance(result, dict):
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                return message_id

        return None

    def edit_photo(
        self,
        chat_id: str,
        message_id: str,
        photo: bytes,
        *,
        caption: str = "",
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> bool:
        media: Dict[str, Any] = {"type": "photo", "media": "attach://photo"}
        if caption:
            media["caption"] = caption
            media["parse_mode"] = "HTML"
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "media": json.dumps(media),
        }
        if reply_markup is not None:
            payload["reply_markup"] = json.dumps(reply_markup)

        payload_response, error = self._telegram_request(
            "editMessageMedia",
            method="POST",
            data_payload=payload,
            files={"photo": ("chart.png", photo, "image/png")},
        )
        if error and "message is not modified" in error.lower():
            return True
        return error is None and bool(payload_response)

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
            download_response = http_client.get(file_url, timeout=30)
            download_response.raise_for_status()
            return download_response.content
        except Exception:
            return None
