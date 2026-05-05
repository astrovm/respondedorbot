"""Shared HTTP session wrapper for connection reuse."""

from __future__ import annotations

from typing import Any, Optional

import requests

_SESSION: Optional[requests.Session] = None


def get_http_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    return get_http_session().request(method, url, **kwargs)


def get(url: str, **kwargs: Any) -> requests.Response:
    return get_http_session().get(url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return get_http_session().post(url, **kwargs)
