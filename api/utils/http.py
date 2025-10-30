"""HTTP helper utilities for resilient requests."""

from typing import Any, Optional
import warnings

import requests
from requests import Response
from requests.exceptions import SSLError
from urllib3.exceptions import InsecureRequestWarning

__all__ = ["request_with_ssl_fallback"]


def request_with_ssl_fallback(
    url: str,
    *,
    method: str = "get",
    session: Optional[requests.sessions.Session] = None,
    suppress_warning: bool = True,
    **kwargs: Any,
) -> Response:
    """Perform an HTTP request and retry without SSL verification on :class:`SSLError`."""

    requester = getattr(session or requests, method.lower())
    try:
        return requester(url, **kwargs)
    except SSLError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["verify"] = False
        if suppress_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InsecureRequestWarning)
                return requester(url, **fallback_kwargs)
        return requester(url, **fallback_kwargs)
