from unittest.mock import MagicMock


def test_get_http_session_reuses_single_session(monkeypatch):
    import api.services.http_client as http_client

    sessions = []

    class DummySession:
        def __init__(self):
            sessions.append(self)

    monkeypatch.setattr(http_client.requests, "Session", DummySession)
    monkeypatch.setattr(http_client, "_SESSION", None)

    first = http_client.get_http_session()
    second = http_client.get_http_session()

    assert first is second
    assert len(sessions) == 1


def test_http_wrappers_delegate_to_shared_session(monkeypatch):
    import api.services.http_client as http_client

    session = MagicMock()
    monkeypatch.setattr(http_client, "get_http_session", lambda: session)

    http_client.get("https://example.test", timeout=1)
    http_client.post("https://example.test", json={"ok": True}, timeout=2)
    http_client.request("PUT", "https://example.test", timeout=3)

    session.get.assert_called_once_with("https://example.test", timeout=1)
    session.post.assert_called_once_with(
        "https://example.test", json={"ok": True}, timeout=2
    )
    session.request.assert_called_once_with("PUT", "https://example.test", timeout=3)
