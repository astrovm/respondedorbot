from tests.support import *  # noqa: F401,F403


class FakeWebhookRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    def setex(self, key, ttl, value):
        self.data[key] = value
        return True

    def delete(self, key):
        self.data.pop(key, None)
        return 1

    def expire(self, key, ttl):
        return key in self.data


def test_responder_no_args():
    with app.test_request_context("/?"):
        response = responder()
        assert response == ("falta key", 200)


def test_responder_wrong_key():
    with app.test_request_context("/?key=wrong_key"), patch(
        "os.environ.get"
    ) as mock_env, patch("api.index.admin_report") as mock_admin:
        mock_env.return_value = "correct_key"
        response = responder()
        assert response == ("key incorrecta", 400)
        mock_admin.assert_called_once_with("intento con key inválida")


def test_responder_valid_key_with_webhook_check():
    with app.test_request_context("/?key=valid_key&check_webhook=true"), patch(
        "os.environ.get"
    ) as mock_env, patch("api.index.verify_webhook") as mock_verify:
        mock_env.return_value = "valid_key"
        mock_verify.return_value = True
        response = responder()
        assert response == ("webhook joya", 200)


def test_responder_valid_key_with_webhook_update():
    with app.test_request_context("/?key=valid_key&update_webhook=true"), patch(
        "os.environ.get"
    ) as mock_env, patch("api.index.set_telegram_webhook") as mock_set:
        mock_env.side_effect = lambda key, default=None: {
            "WEBHOOK_AUTH_KEY": "valid_key",
            "FUNCTION_URL": "https://example.com",
        }.get(key, default)
        mock_set.return_value = True
        response = responder()
        assert response == ("webhook acomodado", 200)


def test_responder_valid_key_with_valid_message():
    message_data = {
        "message": {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "hello",
        }
    }

    with app.test_request_context(
        "/?key=valid_key", method="POST", json=message_data
    ), patch("os.environ.get") as mock_env, patch(
        "api.index.handle_msg"
    ) as mock_handle, patch(
        "api.index.config_redis"
    ) as mock_redis, patch(
        "api.index.is_secret_token_valid"
    ) as mock_token:
        mock_env.side_effect = lambda key, default=None: {
            "WEBHOOK_AUTH_KEY": "valid_key"
        }.get(key, default)
        mock_handle.return_value = "ok"
        mock_redis.return_value = MagicMock()
        mock_token.return_value = True
        response = responder()
        assert response == ("ok", 200)


def test_responder_exception_handling():
    with app.test_request_context("/?key=valid_key"), patch(
        "os.environ.get"
    ) as mock_env, patch("api.index.process_request_parameters") as mock_process, patch(
        "api.index.admin_report"
    ) as mock_admin:
        mock_env.side_effect = lambda key, default=None: {
            "WEBHOOK_AUTH_KEY": "valid_key"
        }.get(key, default)
        mock_process.side_effect = Exception("Test error")

        response = responder()
        assert response == ("error crítico", 500)
        mock_admin.assert_called_once()


def test_process_request_parameters():
    from api.index import process_request_parameters

    with app.test_request_context("/?check_webhook=true"):
        with patch("api.index.verify_webhook") as mock_verify:
            mock_verify.return_value = True
            response, status = process_request_parameters(request)
            assert status == 200
            assert "webhook joya" in response

    with app.test_request_context("/?update_webhook=true"):
        with patch("api.index.set_telegram_webhook") as mock_set, patch(
            "os.environ.get"
        ) as mock_env:
            mock_set.return_value = True
            mock_env.return_value = "https://example.com/webhook"  # Mock FUNCTION_URL
            response, status = process_request_parameters(request)
            assert status == 200
            assert "webhook acomodado" in response

def test_process_request_parameters_handles_pre_checkout_query():
    from api.index import process_request_parameters

    with app.test_request_context("/", json={"pre_checkout_query": {"id": "pcq_1"}}):
        with patch("api.index.is_secret_token_valid", return_value=True), patch(
            "api.index.handle_pre_checkout_query"
        ) as mock_handle:
            response, status = process_request_parameters(request)

    assert status == 200
    assert response == "ok"
    mock_handle.assert_called_once_with({"id": "pcq_1"})


def test_process_request_parameters_returns_ok_for_completed_duplicate_without_reprocessing():
    from api.index import process_request_parameters

    redis_client = FakeWebhookRedis()
    redis_client.setex(
        "webhook:idempotency:message:123:1:completed",
        60,
        json.dumps({"status": "completed"}),
    )

    with app.test_request_context(
        "/",
        method="POST",
        json={
            "message": {
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "text": "hola",
            }
        },
    ):
        with patch("api.index.is_secret_token_valid", return_value=True), patch(
            "api.index._optional_redis_client", return_value=redis_client
        ), patch("api.index.handle_msg") as mock_handle:
            response, status = process_request_parameters(request)

    assert (response, status) == ("ok", 200)
    mock_handle.assert_not_called()


def test_process_request_parameters_returns_retry_for_in_flight_duplicate():
    from api.index import process_request_parameters

    redis_client = FakeWebhookRedis()
    redis_client.set(
        "webhook:idempotency:message:123:1:lock",
        "owner-1",
        ex=60,
        nx=False,
    )

    with app.test_request_context(
        "/",
        method="POST",
        json={
            "message": {
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "text": "hola",
            }
        },
    ):
        with patch("api.index.is_secret_token_valid", return_value=True), patch(
            "api.index._optional_redis_client", return_value=redis_client
        ), patch("api.index.handle_msg") as mock_handle:
            response, status = process_request_parameters(request)

    assert (response, status) == ("retry", 503)
    mock_handle.assert_not_called()


def test_process_request_parameters_returns_retry_when_message_arrives_without_redis():
    from api.index import process_request_parameters

    with app.test_request_context(
        "/",
        method="POST",
        json={
            "message": {
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "text": "hola",
            }
        },
    ):
        with patch("api.index.is_secret_token_valid", return_value=True), patch(
            "api.index._optional_redis_client", return_value=None
        ), patch("api.index.handle_msg") as mock_handle:
            response, status = process_request_parameters(request)

    assert (response, status) == ("retry", 503)
    mock_handle.assert_not_called()


def test_verify_webhook():
    from api.index import verify_webhook

    with patch("api.index.get_telegram_webhook_info") as mock_get_info, patch(
        "os.environ.get"
    ) as mock_env:

        # Mock environment variables
        def mock_env_get(key):
            env_vars = {
                "FUNCTION_URL": "https://my-app.vercel.app",
                "WEBHOOK_AUTH_KEY": "test_key",
                "TELEGRAM_TOKEN": "test_token",
            }
            return env_vars.get(key)

        mock_env.side_effect = mock_env_get

        # Test when webhook is correctly set
        mock_get_info.return_value = {"url": "https://my-app.vercel.app?key=test_key"}
        assert verify_webhook() == True

        # Test when webhook URL doesn't match
        mock_get_info.return_value = {"url": "https://old-app.vercel.app?key=test_key"}
        assert verify_webhook() == False

        # Test when webhook info has error
        mock_get_info.return_value = {"error": "Something went wrong"}
        assert verify_webhook() == False

        # Test when missing environment variables
        mock_env.side_effect = lambda key: None
        assert verify_webhook() == False


def test_is_secret_token_valid():
    from api.index import is_secret_token_valid

    with patch("api.index.config_redis") as mock_config_redis:

        mock_instance = MagicMock()
        mock_config_redis.return_value = mock_instance

        # Test valid token
        mock_instance.get.return_value = "valid_token"
        # Create a mock request with the right headers
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "valid_token"
        result = is_secret_token_valid(mock_request)
        assert result == True

        # Test invalid token
        mock_instance.get.return_value = "valid_token"  # Redis has valid_token
        mock_request.headers.get.return_value = "invalid_token"
        assert is_secret_token_valid(mock_request) == False

        # Test missing token
        mock_instance.get.return_value = "valid_token"  # Redis has valid_token
        mock_request.headers.get.return_value = None
        assert is_secret_token_valid(mock_request) == False


def test_set_telegram_webhook():
    from api.index import set_telegram_webhook

    with patch("requests.get") as mock_request, patch(
        "redis.Redis"
    ) as mock_redis, patch("os.environ.get") as mock_env:

        # Fix: lambda to handle both arguments
        mock_env.side_effect = lambda key, default=None: {
            "WEBHOOK_AUTH_KEY": "test_key",
            "TELEGRAM_TOKEN": "test_token",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_PASSWORD": None,
        }.get(key, default)

        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.set.return_value = True
        mock_instance.ping.return_value = True  # Add ping success

        # Test successful webhook set
        mock_request.return_value.raise_for_status.return_value = None
        assert set_telegram_webhook("https://test.url") == True

        # Test failed webhook set
        mock_request.side_effect = requests.exceptions.RequestException
        assert set_telegram_webhook("https://test.url") == False


def test_send_typing_basic():
    from api.index import send_typing

    with patch("requests.get") as mock_get:
        send_typing("test_token", "12345")
        mock_get.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendChatAction",
            params={"chat_id": "12345", "action": "typing"},
            timeout=5,
        )


def test_telegram_request_requires_token():
    from api.index import _telegram_request

    with patch("api.index.environ.get") as mock_env:
        mock_env.return_value = None
        payload, error = _telegram_request("sendMessage")

    mock_env.assert_called_once_with("TELEGRAM_TOKEN")
    assert payload is None
    assert error == "Telegram token not configured"


def test_telegram_request_handles_ok_false():
    from api.index import _telegram_request

    with patch("api.index.environ.get") as mock_env, patch(
        "requests.get"
    ) as mock_get:
        mock_env.return_value = "token123"
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": False, "description": "bad"}
        mock_get.return_value = response

        payload, error = _telegram_request("foo")

    mock_get.assert_called_once_with(
        "https://api.telegram.org/bottoken123/foo",
        params=None,
        timeout=5,
    )
    assert payload == {"ok": False, "description": "bad"}
    assert error == "bad"


def test_send_msg_basic():
    from api.index import send_msg

    with patch("requests.post") as mock_post, patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_token"
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 42}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = send_msg("12345", "hello")

        mock_post.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendMessage",
            json={"chat_id": "12345", "text": "hello"},
            timeout=5,
        )
        assert result == 42


def test_send_msg_with_buttons():
    from api.index import send_msg

    with patch("requests.post") as mock_post, patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_token"
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 77}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = send_msg("12345", "hello", buttons=["https://twitter.com/foo"])

        mock_post.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendMessage",
            json={
                "chat_id": "12345",
                "text": "hello",
                "reply_markup": {
                    "inline_keyboard": [
                        [{"text": "abrir en la app", "url": "https://twitter.com/foo"}]
                    ]
                },
            },
            timeout=5,
        )
        assert result == 77


def test_send_msg_disables_preview_for_polymarket_link():
    from api.index import send_msg

    with patch("requests.post") as mock_post, patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_token"
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 101}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        polymarket_text = (
            "Polymarket - ¿Quién gana más bancas en Diputados 2025?\n"
            "https://polymarket.com/event/which-party-wins-most-seats-in-argentina-deputies-election"
        )

        result = send_msg("12345", polymarket_text)

        mock_post.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendMessage",
            json={
                "chat_id": "12345",
                "text": polymarket_text,
                "disable_web_page_preview": True,
            },
            timeout=5,
        )
        assert result == 101


def test_send_msg_does_not_disable_preview_for_lookalike_domain():
    from api.index import send_msg

    with patch("requests.post") as mock_post, patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_token"
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 102}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        lookalike_text = "https://evilpolymarket.com/event/not-the-real-domain"

        result = send_msg("12345", lookalike_text)

        mock_post.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendMessage",
            json={"chat_id": "12345", "text": lookalike_text},
            timeout=5,
        )
        assert result == 102


def test_admin_report_basic():
    from api.index import admin_report

    with patch("api.index.send_msg") as mock_send_msg, patch(
        "os.environ.get"
    ) as mock_env:
        mock_env.side_effect = lambda key, default=None: {
            "ADMIN_CHAT_ID": "12345",
            "FRIENDLY_INSTANCE_NAME": "test_instance",
        }.get(key, default)

        admin_report("test message")
        mock_send_msg.assert_called_once_with(
            "12345", "reporte admin desde test_instance: test message"
        )
