from tests.support import *  # noqa: F401,F403


def test_is_social_frontend():
    from api.index import is_social_frontend

    assert is_social_frontend("twitter.com")
    assert is_social_frontend("mobile.twitter.com")
    assert is_social_frontend("xcancel.com")
    assert not is_social_frontend("example.com")


@patch("api.utils.links.request_with_ssl_fallback")
def test_replace_links(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = (
        "<meta property='og:title' content='foo'>"
        "<meta property='og:image' content='https://example.com/image.png'>"
    )
    mock_get.return_value = mock_response
    text = (
        "Check https://twitter.com/foo?utm_source=share and http://x.com/bar?s=20 and https://bsky.app/baz?share=1 and "
        "https://www.instagram.com/qux?igsh=abc123 and https://www.reddit.com/r/foo?st=abc and https://old.reddit.com/r/bar?utm_name=bar and "
        "https://www.tiktok.com/@bar?lang=en and https://vm.tiktok.com/ZMHGacxknMW5J-gEiNC/?share=copy"
    )
    fixed, changed, originals = replace_links(text)
    assert changed
    assert "https://twitter.com/foo" in fixed
    assert "http://x.com/bar" in fixed
    assert "https://fxbsky.app/baz" in fixed
    assert "https://kksave.com/qux" in fixed
    assert "https://www.rxddit.com/r/foo" in fixed
    assert "https://old.rxddit.com/r/bar" in fixed
    assert "https://www.tiktok.com/@bar" in fixed
    assert "https://vm.tiktok.com/ZMHGacxknMW5J-gEiNC/" in fixed
    assert "fxtwitter.com" not in fixed
    assert "fixupx.com" not in fixed
    assert "?" not in fixed
    expected = {
        "https://bsky.app/baz",
        "https://www.instagram.com/qux",
        "https://www.reddit.com/r/foo",
        "https://old.reddit.com/r/bar",
    }
    assert set(originals) == expected
    assert all("?" not in url and "#" not in url for url in originals)


@patch("api.utils.links.request_with_ssl_fallback")
def test_replace_links_skips_when_embed_fails(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    text = "Check https://www.reddit.com/r/foo"
    fixed, changed, originals = replace_links(text)
    mock_get.assert_called_once()
    assert not changed
    assert fixed == text
    assert originals == []


@patch("api.utils.links.request_with_ssl_fallback")
def test_replace_links_skips_when_no_metadata(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<html></html>"
    mock_get.return_value = mock_response
    text = "Check https://www.reddit.com/r/foo"
    fixed, changed, originals = replace_links(text)
    assert not changed
    assert fixed == text
    assert originals == []


@patch("api.utils.links.request_with_ssl_fallback")
def test_replace_links_skips_when_only_twitter_metadata(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = (
        "<meta name='twitter:card' content='summary'>"
        "<meta name='twitter:image' content='https://example.com/img.png'>"
    )
    mock_get.return_value = mock_response
    text = "Check https://www.instagram.com/qux?igsh=abc123"
    fixed, changed, originals = replace_links(text)
    assert not changed
    assert fixed == "Check https://www.instagram.com/qux"
    assert originals == []


def test_replace_links_instagram_falls_back_to_eeinstagram_when_kk_fails():
    checks = []

    def checker(url):
        checks.append(url)
        if "kksave.com" in url:
            return False
        if "eeinstagram.com" in url:
            return True
        return False

    from api.utils.links import replace_links as links_replace_links

    fixed, changed, originals = links_replace_links(
        "Check https://www.instagram.com/qux?igsh=abc123",
        embed_checker=checker,
    )

    assert fixed == "Check https://eeinstagram.com/qux"
    assert changed is True
    assert originals == ["https://www.instagram.com/qux"]
    assert checks == [
        "https://kksave.com/qux",
        "https://eeinstagram.com/qux",
    ]


def test_handle_msg_link_reply():
    message = {
        "message_id": 1,
        "chat": {"id": 123, "type": "group"},
        "from": {"first_name": "John", "username": "john"},
        "text": "check https://twitter.com/foo/status/1",
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg", return_value=901) as mock_send,
        patch("api.index.delete_msg") as mock_delete,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch(
            "api.index.build_message_links_context",
            return_value="LINKS DEL MENSAJE:\n1. https://fxtwitter.com/foo/status/1\ntitulo: foo",
        ) as mock_links_context,
        patch("api.index.save_message_to_redis") as mock_save,
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = (
            "<meta property='og:title' content='foo'>"
            "<meta property='og:image' content='https://example.com/image.png'>"
        )
        mock_get.return_value = mock_response

        result = handle_msg(message)

        assert result == "ok"
        expected = "check https://fxtwitter.com/foo/status/1\n\ncompartido por @john"
        mock_send.assert_called_once_with(
            "123", expected, "1", ["https://twitter.com/foo/status/1"]
        )
        mock_delete.assert_not_called()
        mock_links_context.assert_called_once_with({"text": expected})
        mock_save.assert_called_once_with(
            "123",
            "bot_901",
            "check https://fxtwitter.com/foo/status/1\n\ncompartido por @john\n\nLINKS DEL MENSAJE:\n1. https://fxtwitter.com/foo/status/1\ntitulo: foo",
            redis_client,
        )


def test_handle_msg_link_reply_instagram():
    message = {
        "message_id": 3,
        "chat": {"id": 789, "type": "group"},
        "from": {"first_name": "Lu", "username": "lu"},
        "text": "mirá https://www.instagram.com/qux?igsh=abc",
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg", return_value=903) as mock_send,
        patch("api.index.delete_msg") as mock_delete,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch(
            "api.index.build_message_links_context",
            return_value="LINKS DEL MENSAJE:\n1. https://kksave.com/qux\ntitulo: foo",
        ) as mock_links_context,
        patch("api.index.save_message_to_redis") as mock_save,
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = (
            "<meta property='og:title' content='foo'>"
            "<meta property='og:image' content='https://example.com/image.png'>"
        )
        mock_get.return_value = mock_response

        result = handle_msg(message)

        assert result == "ok"
        expected = "mirá https://kksave.com/qux\n\ncompartido por @lu"
        mock_send.assert_called_once_with(
            "789", expected, "3", ["https://www.instagram.com/qux"]
        )
        mock_delete.assert_not_called()
        mock_links_context.assert_called_once_with({"text": expected})
        mock_save.assert_called_once_with(
            "789",
            "bot_903",
            "mirá https://kksave.com/qux\n\ncompartido por @lu\n\nLINKS DEL MENSAJE:\n1. https://kksave.com/qux\ntitulo: foo",
            redis_client,
        )


def test_handle_msg_link_delete():
    message = {
        "message_id": 2,
        "chat": {"id": 456, "type": "group"},
        "from": {"first_name": "Ana", "username": "ana"},
        "text": "look https://x.com/bar/status/1",
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg", return_value=902) as mock_send,
        patch("api.index.delete_msg") as mock_delete,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "delete"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch(
            "api.index.build_message_links_context",
            return_value="LINKS DEL MENSAJE:\n1. https://fixupx.com/bar/status/1\ntitulo: foo",
        ) as mock_links_context,
        patch("api.index.save_message_to_redis") as mock_save,
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = (
            "<meta property='og:title' content='foo'>"
            "<meta property='og:image' content='https://example.com/image.png'>"
        )
        mock_get.return_value = mock_response

        result = handle_msg(message)

        assert result == "ok"
        expected = "look https://fixupx.com/bar/status/1\n\ncompartido por @ana"
        mock_delete.assert_called_once_with("456", "2")
        mock_send.assert_called_once_with(
            "456", expected, buttons=["https://x.com/bar/status/1"]
        )
        mock_links_context.assert_called_once_with({"text": expected})
        mock_save.assert_called_once_with(
            "456",
            "bot_902",
            "look https://fixupx.com/bar/status/1\n\ncompartido por @ana\n\nLINKS DEL MENSAJE:\n1. https://fixupx.com/bar/status/1\ntitulo: foo",
            redis_client,
        )


@patch("api.index.config_redis")
def test_handle_msg_link_without_preview(mock_redis):
    message = {
        "message_id": 5,
        "chat": {"id": 321, "type": "group"},
        "from": {"id": 1},
        "text": "https://example.com",
    }
    with (
        patch("api.index.send_msg") as mock_send,
        patch("api.index.replace_links") as mock_replace,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch("api.index.should_gordo_respond", return_value=False),
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_replace.return_value = ("https://example.com", False, [])

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()


def test_replace_links_checks_preview(monkeypatch):
    from api.index import replace_links

    mock_can = MagicMock(return_value=True)
    monkeypatch.setattr("api.index.can_embed_url", mock_can)
    text, changed, originals = replace_links("https://x.com/foo/status/123")
    assert text == "https://fixupx.com/foo/status/123"
    assert changed is True
    assert originals == ["https://x.com/foo/status/123"]
    mock_can.assert_called_once_with("https://fixupx.com/foo/status/123")


def test_replace_links_strips_xcom_i_status(monkeypatch):
    from api.index import replace_links

    mock_can = MagicMock(return_value=True)
    monkeypatch.setattr("api.index.can_embed_url", mock_can)
    text, changed, originals = replace_links(
        "https://x.com/i/status/1848434048944783554"
    )
    assert text == "https://fixupx.com/status/1848434048944783554"
    assert changed is True
    assert originals == ["https://x.com/i/status/1848434048944783554"]
    mock_can.assert_called_once_with("https://fixupx.com/status/1848434048944783554")


def test_replace_links_skips_twitter_user_profiles(monkeypatch):
    from api.index import replace_links

    mock_can = MagicMock(return_value=True)
    monkeypatch.setattr("api.index.can_embed_url", mock_can)
    text, changed, originals = replace_links("https://twitter.com/foo")
    assert text == "https://twitter.com/foo"
    assert changed is False
    assert originals == []
    mock_can.assert_not_called()


@patch("api.utils.links.request_with_ssl_fallback")
def test_xcom_link_replacement_with_metadata(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = (
        "<meta property='og:title' content='foo'>"
        "<meta property='og:image' content='https://example.com/image.png'>"
    )
    mock_get.return_value = mock_response
    fixed, changed, originals = replace_links("https://x.com/foo/status/123")
    assert changed is True
    assert fixed == "https://fixupx.com/foo/status/123"
    assert originals == ["https://x.com/foo/status/123"]
    mock_get.assert_called_once_with(
        "https://fixupx.com/foo/status/123",
        allow_redirects=True,
        timeout=10,
        headers=ANY,
    )


@patch("api.utils.links.request_with_ssl_fallback")
def test_xcancel_link_replacement(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = (
        "<meta property='og:title' content='foo'>"
        "<meta property='og:image' content='https://example.com/image.png'>"
    )
    mock_get.return_value = mock_response
    fixed, changed, originals = replace_links("https://xcancel.com/foo/status/123")
    assert changed is True
    assert fixed == "https://fixupx.com/foo/status/123"
    assert originals == ["https://xcancel.com/foo/status/123"]
    mock_get.assert_called_once_with(
        "https://fixupx.com/foo/status/123",
        allow_redirects=True,
        timeout=10,
        headers=ANY,
    )


def test_can_embed_url_logs_missing_meta(monkeypatch, capsys):
    from api.index import can_embed_url

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<html></html>"
    monkeypatch.setattr(
        "api.utils.links.request_with_ssl_fallback", lambda *a, **kw: mock_response
    )

    result = can_embed_url("http://example.com")
    assert result is False
    captured = capsys.readouterr().out
    assert "missing required metadata" in captured
    assert "og:title/twitter:title or og:description/twitter:description" in captured
    assert (
        "og:image/twitter:image or og:video/twitter:player or twitter:card" in captured
    )


def test_can_embed_url_rejects_title_without_card_or_media(monkeypatch):
    from api.index import can_embed_url

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<meta property='og:title' content='Title'>"
    monkeypatch.setattr(
        "api.utils.links.request_with_ssl_fallback", lambda *a, **kw: mock_response
    )

    assert can_embed_url("http://example.com") is False


def test_can_embed_url_allows_title_and_image(monkeypatch):
    from api.index import can_embed_url

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = (
        "<meta property='og:title' content='Title'>"
        "<meta property='og:image' content='https://example.com/img.png'>"
    )
    monkeypatch.setattr(
        "api.utils.links.request_with_ssl_fallback", lambda *a, **kw: mock_response
    )

    assert can_embed_url("http://example.com") is True


def test_can_embed_url_allows_twitter_card_text_preview(monkeypatch):
    from api.index import can_embed_url

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = (
        "<meta name='twitter:card' content='tweet'>"
        "<meta name='twitter:title' content='Agustin Cortes (@agucortes)'>"
        "<meta property='og:description' content='Texto del post'>"
    )
    monkeypatch.setattr(
        "api.utils.links.request_with_ssl_fallback", lambda *a, **kw: mock_response
    )

    assert can_embed_url("https://fixupx.com/status/2032173338240467235") is True


def test_can_embed_url_rejects_twitter_card_only(monkeypatch):
    from api.index import can_embed_url

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = (
        "<meta name='twitter:card' content='summary'>"
        "<meta name='twitter:image' content='https://example.com/img.png'>"
    )
    monkeypatch.setattr(
        "api.utils.links.request_with_ssl_fallback", lambda *a, **kw: mock_response
    )

    assert can_embed_url("http://example.com") is False


def test_can_embed_url_falls_back_to_get_when_eeinstagram_head_not_allowed(monkeypatch):
    from api.index import can_embed_url

    head_response = MagicMock()
    head_response.status_code = 405
    head_response.headers = {"Content-Type": "application/json"}

    get_response = MagicMock()
    get_response.status_code = 200
    get_response.headers = {"Content-Type": "text/html"}
    get_response.text = (
        "<meta property='og:title' content='Instagram post'>"
        "<meta property='og:image' content='https://example.com/preview.jpg'>"
    )

    def fake_request(url, **kwargs):
        if kwargs.get("method") == "head":
            return head_response
        return get_response

    monkeypatch.setattr("api.utils.links.request_with_ssl_fallback", fake_request)

    assert can_embed_url("https://eeinstagram.com/reel/DUEZt-wEXNw/") is True


def test_can_embed_url_allows_eeinstagram_image_only_metadata(monkeypatch):
    from api.index import can_embed_url

    head_response = MagicMock()
    head_response.status_code = 405
    head_response.headers = {"Content-Type": "application/json"}

    get_response = MagicMock()
    get_response.status_code = 200
    get_response.headers = {"Content-Type": "text/html"}
    get_response.text = (
        "<meta property='og:image' content='https://example.com/preview.jpg'>"
    )

    def fake_request(url, **kwargs):
        if kwargs.get("method") == "head":
            return head_response
        return get_response

    monkeypatch.setattr("api.utils.links.request_with_ssl_fallback", fake_request)

    assert can_embed_url("https://eeinstagram.com/p/DVUqOBgDEor/") is True


def test_can_embed_url_allows_eeinstagram_redirect(monkeypatch):
    from api.index import can_embed_url

    head_response = MagicMock()
    head_response.status_code = 307
    head_response.headers = {"Location": "https://scontent.cdninstagram.com/video.mp4"}

    def fake_request(url, **kwargs):
        if kwargs.get("method") == "head":
            return head_response
        raise AssertionError("GET should not be called for eeinstagram")

    monkeypatch.setattr("api.utils.links.request_with_ssl_fallback", fake_request)

    assert can_embed_url("https://eeinstagram.com/reel/DOmco1zjuVi/") is True


def test_can_embed_url_allows_eeinstagram_post_redirect(monkeypatch):
    from api.index import can_embed_url

    head_response = MagicMock()
    head_response.status_code = 307
    head_response.headers = {"Location": "https://scontent.cdninstagram.com/video.mp4"}

    def fake_request(url, **kwargs):
        if kwargs.get("method") == "head":
            return head_response
        raise AssertionError("GET should not be called for eeinstagram")

    monkeypatch.setattr("api.utils.links.request_with_ssl_fallback", fake_request)

    assert can_embed_url("https://eeinstagram.com/p/DQ5RaKnjE8J/") is True


@patch("api.utils.links.request_with_ssl_fallback")
def test_can_embed_url_allows_direct_media(mock_get):
    from api.index import can_embed_url
    from api.utils.links import TELEGRAM_PREVIEW_USER_AGENT

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "video/mp4"}
    mock_response.text = ""
    mock_get.return_value = mock_response

    result = can_embed_url("http://example.com/video")
    assert result is True

    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["User-Agent"] == TELEGRAM_PREVIEW_USER_AGENT


def test_handle_msg_link_already_fixed():
    message = {
        "message_id": 6,
        "chat": {"id": 654, "type": "group"},
        "from": {"id": 1},
        "text": "https://fixupx.com/foo",
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg") as mock_send,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch("api.index.should_gordo_respond", return_value=False),
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()
        mock_get.assert_not_called()


def test_handle_msg_original_link_no_check():
    message = {
        "message_id": 7,
        "chat": {"id": 987, "type": "group"},
        "from": {"id": 1},
        "text": "https://vm.tiktok.com/foo",
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg") as mock_send,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch("api.index.should_gordo_respond", return_value=False),
        patch("api.index.replace_links") as mock_replace,
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_replace.return_value = ("https://vm.tiktok.com/foo", False, [])

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()
        mock_get.assert_not_called()


def test_handle_msg_link_already_fixed_subdomain():
    message = {
        "message_id": 8,
        "chat": {"id": 999, "type": "group"},
        "from": {"id": 1},
        "text": "https://old.rxddit.com/r/foo",
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg") as mock_send,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch("api.index.should_gordo_respond", return_value=False),
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()
        mock_get.assert_not_called()


def test_handle_msg_replaced_link_adds_button():
    message = {
        "message_id": 9,
        "chat": {"id": 111, "type": "group"},
        "from": {"id": 1},
        "text": "https://x.com/foo/status/1?utm_source=bar",
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg") as mock_send,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch("api.index.should_gordo_respond") as mock_should,
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = (
            "<meta property='og:title' content='foo'>"
            "<meta property='og:image' content='https://example.com/image.png'>"
        )
        mock_get.return_value = mock_resp

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_called_once_with(
            "111",
            "https://fixupx.com/foo/status/1",
            "9",
            ["https://x.com/foo/status/1"],
        )
        mock_should.assert_not_called()


def test_handle_msg_replaced_link_replies_to_original_message():
    message = {
        "message_id": 10,
        "chat": {"id": 222, "type": "group"},
        "from": {"id": 1, "username": "user"},
        "text": "https://x.com/foo/status/1",
        "reply_to_message": {"message_id": 1},
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg") as mock_send,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "reply"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch("api.index.should_gordo_respond") as mock_should,
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = (
            "<meta property='og:title' content='foo'>"
            "<meta property='og:image' content='https://example.com/image.png'>"
        )
        mock_get.return_value = mock_resp

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_called_once_with(
            "222",
            "https://fixupx.com/foo/status/1\n\ncompartido por @user",
            "1",
            ["https://x.com/foo/status/1"],
        )
        mock_should.assert_not_called()


def test_handle_msg_replaced_link_delete_mode_replies_to_original_message():
    message = {
        "message_id": 11,
        "chat": {"id": 333, "type": "group"},
        "from": {"id": 1, "username": "user"},
        "text": "https://x.com/foo/status/1",
        "reply_to_message": {"message_id": 2},
    }
    with (
        patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}),
        patch("api.index.config_redis") as mock_redis,
        patch("api.index.send_msg") as mock_send,
        patch("api.index.delete_msg") as mock_delete,
        patch(
            "api.index.get_chat_config",
            return_value={**CHAT_CONFIG_DEFAULTS, "link_mode": "delete"},
        ),
        patch("api.index.initialize_commands", return_value={}),
        patch("api.index.should_gordo_respond") as mock_should,
        patch("api.utils.links.request_with_ssl_fallback") as mock_get,
    ):
        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = (
            "<meta property='og:title' content='foo'>"
            "<meta property='og:image' content='https://example.com/image.png'>"
        )
        mock_get.return_value = mock_resp

        result = handle_msg(message)

        assert result == "ok"
        mock_delete.assert_called_once_with("333", "11")
        mock_send.assert_called_once_with(
            "333",
            "https://fixupx.com/foo/status/1\n\ncompartido por @user",
            "2",
            ["https://x.com/foo/status/1"],
        )
        mock_should.assert_not_called()
