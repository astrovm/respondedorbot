"""Send Telegram Bot API requests and higher-level message operations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from os import environ
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import requests
from urllib.parse import urlparse

from api.links.agent_tools import normalize_http_url
from api.services import http_client


_MAX_TELEGRAM_TEXT_LENGTH = 4096


@dataclass(frozen=True)
class _TelegramRequestData:
    method: str
    params: Optional[Dict[str, Any]]
    json_payload: Optional[Dict[str, Any]]
    data_payload: Optional[Dict[str, Any]]
    files: Optional[Dict[str, Any]]
    timeout: int


def _redact_telegram_tokens(value: str) -> str:
    return re.sub(r"/bot[^/\s]+/", "/bot<redacted>/", value)


def _send_telegram_request(
    url: str,
    request: _TelegramRequestData,
) -> requests.Response:
    method = request.method.upper()
    if method == "GET" and request.json_payload is None:
        return http_client.get(url, params=request.params, timeout=request.timeout)
    if method == "POST" and request.params is None and request.files is None:
        return http_client.post(
            url,
            json=request.json_payload,
            timeout=request.timeout,
        )
    if method == "POST" and request.files is not None:
        return http_client.post(
            url,
            data=request.data_payload,
            files=request.files,
            timeout=request.timeout,
        )
    return http_client.request(
        method,
        url,
        params=request.params,
        json=request.json_payload,
        data=request.data_payload,
        files=request.files,
        timeout=request.timeout,
    )


def _request_error_details(
    error: requests.RequestException,
) -> tuple[Optional[Dict[str, Any]], str]:
    description = str(error)
    if error.response is None:
        return None, description

    try:
        payload = error.response.json()
    except (requests.RequestException, ValueError):
        return None, description
    if not isinstance(payload, dict):
        return None, description

    typed_payload = dict(payload)
    return typed_payload, str(typed_payload.get("description") or error)


def _log_request_error(
    endpoint: str,
    error: requests.RequestException,
    description: str,
) -> None:
    if "message is not modified" in description.lower():
        return

    response_body = ""
    if error.response is not None:
        try:
            body = _redact_telegram_tokens(error.response.text)
        except requests.RequestException:
            body = ""
        if body:
            response_body = f" response={body[:500]!r}"

    detail = _redact_telegram_tokens(str(error))
    print(f"Telegram request to {endpoint} failed: {detail}{response_body}")


def _parse_telegram_payload(
    endpoint: str,
    response: requests.Response,
    *,
    log_errors: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw_payload = response.json()
    except ValueError as error:
        if log_errors:
            print(f"Telegram request to {endpoint} returned invalid JSON: {error}")
        return None, str(error)

    if not isinstance(raw_payload, Mapping):
        if log_errors:
            print(f"Telegram request to {endpoint} returned unexpected payload type")
        return None, "unexpected response"

    payload = dict(raw_payload)
    if payload.get("ok"):
        return payload, None

    description = str(payload.get("description") or "telegram request failed")
    if log_errors:
        print(f"Telegram request to {endpoint} returned ok=false: {description}")
    return payload, description


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
    request = _TelegramRequestData(
        method=method,
        params=params,
        json_payload=json_payload,
        data_payload=data_payload,
        files=files,
        timeout=timeout,
    )
    try:
        response = _send_telegram_request(url, request)
        response.raise_for_status()
    except requests.RequestException as error:
        error_payload, description = _request_error_details(error)
        if log_errors:
            _log_request_error(endpoint, error, description)
        return error_payload, description

    if not expect_json:
        return {}, None
    return _parse_telegram_payload(endpoint, response, log_errors=log_errors)


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

    def request(self, endpoint: str, **kwargs: Any) -> Any:
        return self._telegram_request(endpoint, **kwargs)

    def redact_tokens(self, value: str) -> str:
        return _redact_telegram_tokens(value)

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

    def send_video(
        self,
        chat_id: str,
        video: bytes,
        *,
        caption: str = "",
        msg_id: str = "",
        buttons: Optional[List[str]] = None,
    ) -> Optional[int]:
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "supports_streaming": "true",
        }
        if msg_id:
            payload["reply_to_message_id"] = msg_id
        if caption:
            payload["caption"] = caption[:1024]
        if buttons:
            payload["reply_markup"] = json.dumps(
                {
                    "inline_keyboard": [
                        [{"text": "abrir en la app", "url": url}] for url in buttons
                    ]
                }
            )

        payload_response, error = self._telegram_request(
            "sendVideo",
            method="POST",
            data_payload=payload,
            files={"video": ("instagram.mp4", video, "video/mp4")},
            timeout=60,
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
