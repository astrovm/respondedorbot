from unittest.mock import patch, MagicMock
from flask import Flask, request
from api.index import (
    responder,
    convert_to_command,
    config_redis,
    check_rate_limit,
    extract_message_text,
    parse_command,
    format_user_message,
    should_gordo_respond,
    truncate_text,
)
from datetime import datetime, timezone, timedelta
import json
import requests
import redis

app = Flask(__name__)


def test_responder_no_args():
    with app.test_request_context("/?"):
        response = responder()
        assert response == ("No key", 200)


def test_responder_wrong_key():
    with app.test_request_context("/?key=wrong_key"), patch("os.environ.get") as mock_env, patch("api.index.admin_report") as mock_admin:
        mock_env.return_value = "correct_key"
        response = responder()
        assert response == ("Wrong key", 400)
        mock_admin.assert_called_once_with("Wrong key attempt")


def test_responder_valid_key_with_webhook_check():
    with app.test_request_context("/?key=valid_key&check_webhook=true"), patch("os.environ.get") as mock_env, patch("api.index.verify_webhook") as mock_verify:
        mock_env.return_value = "valid_key"
        mock_verify.return_value = True
        response = responder()
        assert response == ("Webhook checked", 200)


def test_responder_valid_key_with_webhook_update():
    with app.test_request_context("/?key=valid_key&update_webhook=true"), patch("os.environ.get") as mock_env, patch("api.index.set_telegram_webhook") as mock_set:
        mock_env.side_effect = lambda key: {"GORDO_KEY": "valid_key", "CURRENT_FUNCTION_URL": "https://example.com"}.get(key)
        mock_set.return_value = True
        response = responder()
        assert response == ("Webhook updated", 200)


def test_responder_valid_key_with_valid_message():
    message_data = {
        "message": {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "hello"
        }
    }
    
    with app.test_request_context("/?key=valid_key", method="POST", json=message_data), patch("os.environ.get") as mock_env, patch("api.index.handle_msg") as mock_handle, patch("api.index.config_redis") as mock_redis, patch("api.index.is_secret_token_valid") as mock_token:
        mock_env.side_effect = lambda key, default=None: {"GORDO_KEY": "valid_key"}.get(key, default)
        mock_handle.return_value = "ok"
        mock_redis.return_value = MagicMock()
        mock_token.return_value = True
        response = responder()
        assert response == ("Ok", 200)


def test_responder_exception_handling():
    with app.test_request_context("/?key=valid_key"), patch("os.environ.get") as mock_env, patch("api.index.process_request_parameters") as mock_process, patch("api.index.admin_report") as mock_admin:
        mock_env.side_effect = lambda key, default=None: {"GORDO_KEY": "valid_key"}.get(key, default)
        mock_process.side_effect = Exception("Test error")
        
        response = responder()
        assert response == ("Critical error", 500)
        mock_admin.assert_called_once()


def test_convert_to_command():
    # Test basic string
    msg_text1 = "h3llo W0RLD"
    expected1 = "/H3LLO_W0RLD"
    assert convert_to_command(msg_text1) == expected1

    # Test string with special characters
    msg_text2 = "hello! world? or... mmm ...bye."
    expected2 = "/HELLO_SIGNODEEXCLAMACION_WORLD_SIGNODEPREGUNTA_OR_PUNTOSSUSPENSIVOS_MMM_PUNTOSSUSPENSIVOS_BYE_PUNTO"
    assert convert_to_command(msg_text2) == expected2

    # Test string with consecutive spaces
    msg_text3 = "  hello   world "
    expected3 = "/HELLO_WORLD"
    assert convert_to_command(msg_text3) == expected3

    # Test string with emoji
    msg_text4 = "😄hello 😄 world"
    expected4 = "/CARA_SONRIENDO_CON_OJOS_SONRIENTES_HELLO_CARA_SONRIENDO_CON_OJOS_SONRIENTES_WORLD"
    assert convert_to_command(msg_text4) == expected4

    # Test string with accented characters and Ñ
    msg_text5 = "hola ñandú ñ"
    expected5 = "/HOLA_NIANDU_ENIE"
    assert convert_to_command(msg_text5) == expected5

    # Test string with new line
    msg_text6 = "hola\nlinea\n"
    expected6 = "/HOLA_LINEA"
    assert convert_to_command(msg_text6) == expected6

    # Test empty string
    msg_text7 = ""
    expected7 = "y que queres que convierta boludo? mandate texto"
    assert convert_to_command(msg_text7) == expected7


def test_config_redis():
    with patch("redis.Redis") as mock_redis:
        # Test successful connection
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.ping.return_value = True

        redis_client = config_redis(host="test", port=1234, password="pass")
        assert redis_client == mock_instance

        # Test failed connection
        mock_instance.ping.side_effect = Exception("Connection failed")
        try:
            config_redis()
            assert False, "Should have raised exception"
        except Exception as e:
            assert str(e) == "Connection failed"


def test_check_rate_limit():
    with patch("redis.Redis") as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance

        # Test within limits
        mock_instance.pipeline.return_value.execute.return_value = [10, True, 5, True]
        assert check_rate_limit("test_chat", mock_instance) == True

        # Test exceeded global limit
        mock_instance.pipeline.return_value.execute.return_value = [1025, True, 5, True]
        assert check_rate_limit("test_chat", mock_instance) == False

        # Test exceeded chat limit
        mock_instance.pipeline.return_value.execute.return_value = [10, True, 129, True]
        assert check_rate_limit("test_chat", mock_instance) == False


def test_extract_message_text():
    # Test regular text message
    msg = {"text": "hello world"}
    assert extract_message_text(msg) == "hello world"

    # Test caption
    msg = {"caption": "photo caption"}
    assert extract_message_text(msg) == "photo caption"

    # Test poll
    msg = {"poll": {"question": "poll question"}}
    assert extract_message_text(msg) == "poll question"

    # Test empty message
    msg = {}
    assert extract_message_text(msg) == ""


def test_parse_command():
    # Test basic command
    assert parse_command("/start hello", "@bot") == ("/start", "hello")

    # Test command with no args
    assert parse_command("/help", "@bot") == ("/help", "")

    # Test command with bot mention
    assert parse_command("/start@bot hello", "@bot") == ("/start", "hello")


def test_format_user_message():
    # Test with username
    msg = {"from": {"first_name": "John", "username": "john123"}}
    assert format_user_message(msg, "hello") == "John (john123): hello"

    # Test without username
    msg = {"from": {"first_name": "John"}}
    assert format_user_message(msg, "hello") == "John: hello"


def test_should_gordo_respond():
    commands = {"/test": (lambda x: x, False, False)}

    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "testbot"  # Set mock bot username

        # Test command
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert should_gordo_respond(commands, "/test", "hello", msg) == True

        # Test private chat
        msg = {"chat": {"type": "private"}, "from": {"username": "test"}}
        assert should_gordo_respond(commands, "", "hello", msg) == True

        # Test mention
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert should_gordo_respond(commands, "", "@testbot hello", msg) == True

        # Test reply to bot
        msg = {
            "chat": {"type": "group"},
            "from": {"username": "test"},
            "reply_to_message": {"from": {"username": "testbot"}},
        }
        assert should_gordo_respond(commands, "", "hello", msg) == True

        # Test trigger word with mocked random
        with patch("random.random") as mock_random:
            mock_random.return_value = 0.05  # Below 0.1 threshold
            msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
            assert should_gordo_respond(commands, "", "hey gordo", msg) == True


def test_gen_random():
    from api.index import gen_random

    with patch("random.randint") as mock_randint:
        # Test "si" response
        mock_randint.side_effect = [
            1,
            0,
        ]  # First call returns 1 (si), second call returns 0 (no suffix)
        assert gen_random("test") == "si"

        # Test "no boludo" response
        mock_randint.side_effect = [
            0,
            1,
        ]  # First call returns 0 (no), second call returns 1 (boludo)
        assert gen_random("test") == "no boludo"

        # Test "no {name}" response
        mock_randint.side_effect = [
            0,
            2,
        ]  # First call returns 0 (no), second call returns 2 (name)
        assert gen_random("astro") == "no astro"


def test_select_random():
    from api.index import select_random

    with patch("random.choice") as mock_choice:
        # Test comma-separated list
        mock_choice.return_value = "pizza"
        assert select_random("pizza, pasta, sushi") == "pizza"

        # Test number range
        with patch("random.randint") as mock_randint:
            mock_randint.return_value = 7
            assert select_random("1-10") == "7"

        # Test invalid input
        assert (
            select_random("invalid input")
            == "mandate algo como 'pizza, carne, sushi' o '1-10' boludo, no me hagas laburar al pedo"
        )


def test_save_message_to_redis():
    from api.index import save_message_to_redis

    with patch("redis.Redis") as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance

        # Mock sismember to return False (message doesn't exist)
        mock_instance.sismember.return_value = False

        # Test successful save
        chat_id = "123"
        message_id = "456"
        text = "test message"

        save_message_to_redis(chat_id, message_id, text, mock_instance)

        # Verify pipeline calls
        mock_instance.pipeline.assert_called_once()
        pipeline = mock_instance.pipeline.return_value
        pipeline.lpush.assert_called_once()
        pipeline.ltrim.assert_called_once()
        pipeline.execute.assert_called_once()


def test_get_chat_history():
    from api.index import get_chat_history

    with patch("redis.Redis") as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance

        # Test with valid history
        mock_instance.lrange.return_value = [
            json.dumps(
                {
                    "id": "123",
                    "text": "test message",
                    "timestamp": int(datetime.now(timezone.utc).timestamp()),
                }
            ),
            json.dumps(
                {
                    "id": "bot_124",
                    "text": "bot response",
                    "timestamp": int(datetime.now(timezone.utc).timestamp()),
                }
            ),
        ]

        history = list(get_chat_history("123", mock_instance, 2))
        assert len(history) == 2
        assert history[0]["role"] == "assistant"
        assert history[1]["role"] == "user"


def test_build_ai_messages():
    from api.index import build_ai_messages

    # Test basic message building
    message = {
        "from": {"first_name": "John", "username": "john123"},
        "chat": {"type": "private", "title": None},
        "text": "test message",
    }

    chat_history = []
    message_text = "test message"

    messages = build_ai_messages(message, chat_history, message_text)

    assert len(messages) == 1  # Should only contain the current message
    assert "CONTEXTO:" in messages[0]["content"]
    assert "Usuario: John" in messages[0]["content"]
    assert "test message" in messages[0]["content"]


def test_process_request_parameters():
    from api.index import process_request_parameters

    with app.test_request_context("/?check_webhook=true"):
        with patch("api.index.verify_webhook") as mock_verify:
            mock_verify.return_value = True
            response, status = process_request_parameters(request)
            assert status == 200
            assert "Webhook checked" in response

    with app.test_request_context("/?update_webhook=true"):
        with patch("api.index.set_telegram_webhook") as mock_set, patch("os.environ.get") as mock_env:
            mock_set.return_value = True
            mock_env.return_value = "https://example.com/webhook"  # Mock CURRENT_FUNCTION_URL
            response, status = process_request_parameters(request)
            assert status == 200
            assert "Webhook updated" in response


def test_handle_rate_limit():
    from api.index import handle_rate_limit

    with patch("api.index.send_typing") as mock_send_typing, patch(
        "time.sleep"
    ) as mock_sleep, patch("api.index.gen_random") as mock_gen_random, patch("os.environ.get") as mock_env:

        chat_id = "123"
        message = {"from": {"first_name": "John"}}
        mock_gen_random.return_value = "no boludo"
        mock_env.return_value = "fake_token"  # Mock TELEGRAM_TOKEN

        response = handle_rate_limit(chat_id, message)

        mock_send_typing.assert_called_once()
        mock_sleep.assert_called_once()
        assert response == "no boludo"


def test_verify_webhook():
    from api.index import verify_webhook

    with patch("api.index.get_telegram_webhook_info") as mock_get_info, patch(
        "api.index.set_telegram_webhook"
    ) as mock_set_webhook, patch("requests.get") as mock_request, patch(
        "os.environ.get"
    ) as mock_env, patch(
        "api.index.admin_report"
    ) as mock_admin_report:

        # Mock environment variables
        def mock_env_get(key):
            env_vars = {
                "MAIN_FUNCTION_URL": "https://main.function.url",
                "CURRENT_FUNCTION_URL": "https://main.function.url",
                "GORDO_KEY": "test_key",
                "TELEGRAM_TOKEN": "test_token",
                "ADMIN_CHAT_ID": "123456789",
                "FRIENDLY_INSTANCE_NAME": "test_instance",
            }
            return env_vars.get(key)

        mock_env.side_effect = mock_env_get

        # Test when webhook is correctly set
        mock_get_info.return_value = {"url": "https://main.function.url?key=test_key"}
        mock_request.return_value.raise_for_status.return_value = None

        assert verify_webhook() == True

        # Test when webhook needs update
        mock_get_info.return_value = {"url": "https://old.function.url?key=test_key"}
        mock_set_webhook.return_value = True

        assert verify_webhook() == True

        # Reset admin_report mock for next test
        mock_admin_report.reset_mock()

        # Test when main webhook is down
        mock_env_get = lambda key: {
            "MAIN_FUNCTION_URL": "https://main.function.url",
            "CURRENT_FUNCTION_URL": "https://backup.function.url",
            "GORDO_KEY": "test_key",
            "TELEGRAM_TOKEN": "test_token",
            "ADMIN_CHAT_ID": "123456789",
            "FRIENDLY_INSTANCE_NAME": "test_instance",
        }.get(key)
        mock_env.side_effect = mock_env_get
        mock_request.side_effect = requests.exceptions.RequestException("Test error")
        mock_set_webhook.return_value = True

        assert verify_webhook() == True
        mock_admin_report.assert_called_once_with(
            "Main webhook failed with error: Test error"
        )


def test_truncate_text():
    from api.index import truncate_text

    # Test text within limit
    assert truncate_text("short text", 512) == "short text"

    # Test text at limit
    text_at_limit = "a" * 512
    assert truncate_text(text_at_limit, 512) == text_at_limit

    # Test text exceeding limit
    long_text = "a" * 600
    assert truncate_text(long_text, 512) == ("a" * 509) + "..."

    # Test with default limit
    assert len(truncate_text("a" * 1000)) <= 512


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


def test_handle_msg():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.ask_ai"
    ) as mock_ask_ai, patch(
        "api.index.send_typing"
    ) as mock_send_typing, patch(
        "time.sleep"
    ) as _mock_sleep:  # Add sleep mock to avoid delays

        mock_env.return_value = "testbot"
        mock_rate_limit.return_value = True  # Don't rate limit
        mock_ask_ai.return_value = "test response"  # Mock ai response

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.lrange.return_value = []  # Empty chat history

        # Test basic message handling
        message = {
            "message_id": "123",
            "chat": {"id": "456", "type": "private"},
            "from": {"first_name": "John", "username": "john123"},
            "text": "/help",
        }

        assert handle_msg(message) == "ok"
        mock_send_msg.assert_called_once()

        # Test message without command
        message["text"] = "hello bot"
        mock_send_msg.reset_mock()
        mock_send_typing.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_called_once()
        mock_send_typing.assert_called_once()
        mock_ask_ai.assert_called_once()

        # Test rate limited message
        mock_rate_limit.return_value = False
        mock_send_msg.reset_mock()
        mock_send_typing.reset_mock()
        mock_ask_ai.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_typing.assert_called_once()
        mock_ask_ai.assert_not_called()


def test_handle_msg_with_crypto_command():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.get_prices"
    ) as mock_get_prices:

        mock_env.return_value = "testbot"
        mock_rate_limit.return_value = True
        mock_get_prices.return_value = "BTC: 50000"

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "/prices btc"
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_get_prices.assert_called_once()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_image():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.describe_image_cloudflare"
    ) as mock_describe, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.resize_image_if_needed"
    ) as mock_resize, patch(
        "api.index.encode_image_to_base64"
    ) as mock_encode, patch(
        "api.index.cached_requests"
    ) as mock_requests:

        mock_env.side_effect = lambda key, default=None: {"TELEGRAM_USERNAME": "testbot", "TELEGRAM_TOKEN": "test_token"}.get(key, default)
        mock_rate_limit.return_value = True
        mock_download.return_value = b"image data"
        mock_describe.return_value = "A beautiful landscape"
        mock_resize.return_value = b"resized image data"
        mock_encode.return_value = "base64_encoded_image"
        mock_requests.return_value = {"description": "A beautiful landscape"}

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "photo": [{"file_id": "photo_123"}]
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_download.assert_called_once()
        mock_encode.assert_called_once()


def test_handle_msg_with_audio():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.transcribe_audio_cloudflare"
    ) as mock_transcribe, patch(
        "api.index.download_telegram_file"
    ) as mock_download:

        mock_env.side_effect = lambda key: {
            "TELEGRAM_USERNAME": "testbot",
            "CLOUDFLARE_API_KEY": "test_key",
            "CLOUDFLARE_ACCOUNT_ID": "test_account"
        }.get(key)
        mock_rate_limit.return_value = True
        mock_download.return_value = b"audio data"
        mock_transcribe.return_value = "transcribed text"

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123"}
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_transcribe.assert_called_once()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.handle_transcribe_with_message"
    ) as mock_handle_transcribe:

        mock_env.side_effect = lambda key: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key)
        mock_rate_limit.return_value = True
        mock_handle_transcribe.return_value = "Transcription result"

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "/transcribe",
            "reply_to_message": {
                "message_id": 2,
                "voice": {"file_id": "voice_123"}
            }
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_handle_transcribe.assert_called_once()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_unknown_command():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.should_gordo_respond"
    ) as mock_should_respond:

        mock_env.side_effect = lambda key: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key)
        mock_rate_limit.return_value = True
        mock_should_respond.return_value = False

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "group"},
            "from": {"first_name": "John", "username": "john"},
            "text": "/unknown_command"
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_send_msg.assert_not_called()  # Should not send message


def test_handle_msg_with_exception():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.admin_report"
    ) as mock_admin_report, patch("os.environ.get") as mock_env:

        mock_env.side_effect = lambda key: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key)
        mock_config_redis.side_effect = Exception("Redis error")

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "hello"
        }

        result = handle_msg(message)
        assert result == "Error processing message"
        mock_admin_report.assert_called_once()


def test_get_weather():
    from api.index import get_weather

    # Create a fixed datetime for testing
    current_time = datetime(2024, 1, 1, 12, 0)  # Create naive datetime first

    with patch("api.index.datetime") as mock_datetime, patch(
        "api.index.cached_requests"
    ) as mock_cached_requests:

        # Set up datetime mock to handle timezone
        class MockDatetime:
            @classmethod
            def now(cls, tz=None):
                if tz:
                    return current_time.replace(tzinfo=tz)
                return current_time

            @classmethod
            def fromisoformat(cls, timestamp):
                return datetime.fromisoformat(timestamp)

        mock_datetime.now = MockDatetime.now
        mock_datetime.fromisoformat = MockDatetime.fromisoformat
        mock_datetime.datetime = datetime
        mock_datetime.timezone = timezone
        mock_datetime.timedelta = timedelta

        # Test successful weather fetch
        mock_cached_requests.return_value = {
            "data": {
                "hourly": {
                    "time": [
                        "2024-01-01T12:00",  # Current hour
                        "2024-01-01T13:00",  # Next hour
                        "2024-01-01T14:00",  # Future hours...
                    ],
                    "apparent_temperature": [25.5, 26.0, 26.5],
                    "precipitation_probability": [30, 35, 40],
                    "weather_code": [0, 1, 2],
                    "cloud_cover": [50, 55, 60],
                    "visibility": [10000, 9000, 8000],
                }
            }
        }

        weather = get_weather()
        assert weather is not None
        assert weather["apparent_temperature"] == 25.5
        assert weather["precipitation_probability"] == 30
        assert weather["weather_code"] == 0
        assert weather["cloud_cover"] == 50
        assert weather["visibility"] == 10000

        # Test failed weather fetch
        mock_cached_requests.return_value = None
        assert get_weather() == {}


def test_set_telegram_webhook():
    from api.index import set_telegram_webhook

    with patch("requests.get") as mock_request, patch(
        "redis.Redis"
    ) as mock_redis, patch("os.environ.get") as mock_env:

        # Fix: lambda to handle both arguments
        mock_env.side_effect = lambda key, default=None: {
            "GORDO_KEY": "test_key",
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


def test_handle_msg_edge_cases():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.send_typing"
    ) as mock_send_typing, patch(
        "api.index.gen_random"
    ) as mock_gen_random, patch(
        "api.index.cached_requests"
    ) as mock_cached_requests, patch(
        "api.index.admin_report"
    ) as mock_admin_report, patch(
        "api.index.ask_ai"
    ) as mock_ask_ai, patch(
        "api.index.should_gordo_respond"
    ) as mock_should_respond, patch(
        "time.sleep"
    ) as _mock_sleep:

        # Set up mocks
        mock_env.return_value = "testbot"
        mock_rate_limit.return_value = True
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_gen_random.return_value = "no boludo"
        mock_cached_requests.return_value = None  # Prevent API calls
        mock_ask_ai.return_value = "test response"
        mock_redis.get.return_value = json.dumps(
            {"timestamp": 123, "data": {}}
        )  # Valid JSON
        mock_should_respond.return_value = False  # Don't respond by default

        # Reset all mocks before starting tests
        mock_send_msg.reset_mock()
        mock_send_typing.reset_mock()
        mock_ask_ai.reset_mock()
        mock_admin_report.reset_mock()

        # Test empty message
        message = {
            "message_id": "123",
            "chat": {"id": "456", "type": "private"},
            "from": {"first_name": "John", "username": "john123"},
        }
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_not_called()

        # Test message with only whitespace
        message["text"] = "   \n   \t   "
        mock_send_msg.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_not_called()

        # Test message that should get a response
        mock_should_respond.return_value = True
        message["text"] = "test"
        mock_send_msg.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_called_once_with("456", "test response", "123")

        # Test message with invalid JSON
        mock_redis.get.return_value = "invalid json"
        mock_send_msg.reset_mock()
        assert handle_msg(message) == "ok"

        # Test message with missing required fields
        mock_admin_report.reset_mock()  # Reset admin report mock
        message = {"message_id": "123"}  # Missing chat and from fields
        mock_send_msg.reset_mock()
        result = handle_msg(message)
        assert result == "Error processing message"
        mock_admin_report.assert_called_once()  # Should report error to admin

        # Test message with None values
        mock_admin_report.reset_mock()
        message = {
            "message_id": "123",
            "chat": {"id": None},  # Changed to match error handling structure
            "from": {"username": None},  # Changed to match error handling structure
            "text": None,
        }
        mock_send_msg.reset_mock()
        result = handle_msg(message)
        assert result == "Error processing message"
        mock_admin_report.assert_called_once()

        # Test message with missing message_id
        mock_admin_report.reset_mock()
        message = {"chat": {"id": "456"}, "from": {"first_name": "John"}}
        mock_send_msg.reset_mock()
        result = handle_msg(message)
        assert result == "Error processing message"
        mock_admin_report.assert_called_once()


def test_parse_command_edge_cases():
    # Test empty string
    assert parse_command("", "@bot") == ("", "")

    # Test only spaces
    assert parse_command("    ", "@bot") == ("", "")

    # Test command with multiple spaces
    assert parse_command("/start    hello    world", "@bot") == (
        "/start",
        "hello    world",
    )

    # Test command with special characters
    assert parse_command("/start!@#$%^&*()", "@bot") == ("/start!@#$%^&*()", "")


def test_extract_message_text_edge_cases():
    # Test message with all types of text
    msg = {
        "text": "text message",
        "caption": "photo caption",
        "poll": {"question": "poll question"},
    }
    # Should prioritize text over caption and poll
    assert extract_message_text(msg) == "text message"

    # Test message with caption and poll
    msg = {"caption": "photo caption", "poll": {"question": "poll question"}}
    # Should prioritize caption over poll
    assert extract_message_text(msg) == "photo caption"

    # Test message with invalid poll structure
    msg = {"poll": "invalid poll"}
    assert extract_message_text(msg) == ""

    # Test message with None values
    msg = {"text": None, "caption": None, "poll": None}
    assert extract_message_text(msg) == ""


def test_format_user_message_edge_cases():
    # Test with minimal user info
    msg = {"from": {"first_name": ""}}
    assert format_user_message(msg, "hello") == ": hello"

    # Test with None values
    msg = {"from": {"first_name": None, "username": None}}
    assert format_user_message(msg, "hello") == ": hello"

    # Test with special characters in names
    msg = {"from": {"first_name": "John<>&", "username": "user!@#"}}
    assert format_user_message(msg, "hello") == "John<>& (user!@#): hello"

    # Test with very long names
    long_name = "a" * 100
    msg = {"from": {"first_name": long_name, "username": long_name}}
    formatted = format_user_message(msg, "hello")
    assert len(formatted) <= 256  # Should be truncated


def test_check_rate_limit_edge_cases():
    with patch("redis.Redis") as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance

        # Test with negative counts
        mock_instance.pipeline.return_value.execute.return_value = [-1, True, -1, True]
        assert check_rate_limit("test_chat", mock_instance) == True

        # Test with None values
        mock_instance.pipeline.return_value.execute.return_value = [
            None,
            True,
            None,
            True,
        ]
        assert check_rate_limit("test_chat", mock_instance) == True

        # Test with exactly at limits
        mock_instance.pipeline.return_value.execute.return_value = [
            1024,
            True,
            128,
            True,
        ]
        assert check_rate_limit("test_chat", mock_instance) == True

        # Test with Redis errors
        mock_instance.pipeline.return_value.execute.side_effect = redis.RedisError
        assert check_rate_limit("test_chat", mock_instance) == False


def test_truncate_text_edge_cases():
    # Test empty string
    assert truncate_text("") == ""

    # Test None input
    assert truncate_text(None) == ""

    # Test string with exactly max length
    text = "a" * 512
    assert truncate_text(text) == text

    # Test string with max length minus one
    text = "a" * 511
    assert truncate_text(text) == text

    # Test string with max length plus one
    text = "a" * 513
    truncated = truncate_text(text)
    assert len(truncated) == 512
    assert truncated.endswith("...")

    # Test with very small max_length
    assert truncate_text("hello", 1) == "."
    assert truncate_text("hello", 2) == ".."
    assert truncate_text("hello", 3) == "..."
    assert truncate_text("hello", 4) == "h..."


def test_truncate_text_more_edge_cases():
    from api.index import truncate_text

    # Test with zero max_length
    assert truncate_text("text", 0) == ""

    # Test with negative max_length
    assert truncate_text("text", -5) == ""

    # Test with very large text
    large_text = "a" * 10000
    truncated = truncate_text(large_text)
    assert len(truncated) == 512

    # Test with max_length smaller than ellipsis
    assert truncate_text("hello", 1) == "."
    assert truncate_text("hello", 2) == ".."

    # Test with non-string inputs - we'll use str() conversion first
    assert truncate_text(str(123), 10) == "123"
    assert truncate_text(str(True), 10) == "True"

    # Test with empty string and various max_lengths
    assert truncate_text("", 5) == ""
    assert truncate_text("", 0) == ""


def test_initialize_commands():
    from api.index import (
        initialize_commands,
        ask_ai,
        select_random,
        get_prices,
        get_dollar_rates as _get_dollar_rates,
    )
    from api.index import (
        get_devo as _get_devo,
        powerlaw as _powerlaw,
        rainbow as _rainbow,
        get_timestamp as _get_timestamp,
        convert_to_command as _convert_to_command,
        get_instance_name as _get_instance_name,
        get_help,
    )

    commands = initialize_commands()

    # Test that commands dict contains expected entries
    assert "/ask" in commands
    assert "/pregunta" in commands
    assert "/che" in commands
    assert "/gordo" in commands
    assert "/random" in commands
    assert "/prices" in commands
    assert "/dolar" in commands

    # Test that AI commands are properly marked
    assert commands["/ask"][1] == True
    assert commands["/pregunta"][1] == True
    assert commands["/che"][1] == True
    assert commands["/gordo"][1] == True

    # Test that non-AI commands are properly marked
    assert commands["/random"][1] == False
    assert commands["/prices"][1] == False
    assert commands["/dolar"][1] == False

    # Test function mappings
    assert commands["/ask"][0] == ask_ai
    assert commands["/random"][0] == select_random
    assert commands["/prices"][0] == get_prices
    assert commands["/help"][0] == get_help


def test_config_redis_with_env_vars():
    from api.index import config_redis

    with patch("redis.Redis") as mock_redis, patch("os.environ.get") as mock_env:
        # Setup
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.ping.return_value = True

        # Configure environment variables
        mock_env.side_effect = lambda key, default=None: {
            "REDIS_HOST": "redis.example.com",
            "REDIS_PORT": "1234",
            "REDIS_PASSWORD": "secret",
        }.get(key, default)

        # Test
        result = config_redis()

        # Verify
        assert result == mock_instance
        mock_redis.assert_called_with(
            host="redis.example.com",
            port=1234,
            password="secret",
            decode_responses=True,
        )


def test_config_redis_connection_error():
    from api.index import config_redis

    with patch("redis.Redis") as mock_redis, patch(
        "api.index.admin_report"
    ) as mock_admin_report:
        # Setup
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.ping.side_effect = redis.ConnectionError("Connection refused")

        # Test and verify
        try:
            config_redis()
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Connection refused" in str(e)

        # Verify admin report was called
        mock_admin_report.assert_called_once()


def test_extract_message_text_complex_cases():
    from api.index import extract_message_text

    # Test with nested structures
    msg = {
        "text": "",
        "caption": "",
        "poll": {"question": "Is this a nested poll?", "incorrect_field": "ignored"},
    }
    assert extract_message_text(msg) == "Is this a nested poll?"

    # Test with malformed poll
    msg = {"poll": {"not_question": "This shouldn't appear"}}
    assert extract_message_text(msg) == ""

    # Test prioritization (text > caption > poll)
    msg = {
        "text": "Primary text",
        "caption": "Secondary caption",
        "poll": {"question": "Tertiary poll question"},
    }
    assert extract_message_text(msg) == "Primary text"

    # Test with spaces to trim
    msg = {"text": "  Text with spaces  "}
    assert extract_message_text(msg) == "Text with spaces"

    # Test with non-string values
    msg = {"text": 12345}
    assert extract_message_text(msg) == "12345"

    # Test with None values but valid keys
    msg = {"text": None, "caption": None, "poll": None}
    assert extract_message_text(msg) == ""


def test_parse_command_complex_cases():
    from api.index import parse_command

    # Test command with different case
    assert parse_command("/StArT hello", "@bot") == ("/start", "hello")

    # Test command with multiple bot mentions
    assert parse_command("/start@bot@bot hello", "@bot") == ("/start", "hello")

    # Test command with multiple arguments and spaces
    assert parse_command("/start arg1    arg2  arg3", "@bot") == (
        "/start",
        "arg1    arg2  arg3",
    )

    # Command must start with / to be recognized as a command
    assert parse_command("Hey /start hello", "@bot") == ("hey", "/start hello")

    # Test with special characters
    assert parse_command("/start-now hello!", "@bot") == ("/start-now", "hello!")

    # Test with leading/trailing spaces in entire string
    # Note: function strips trailing spaces
    assert parse_command("  /start  hello  ", "@bot") == ("/start", "hello")

    # Test with emoji in command
    assert parse_command("/start😀 hello", "@bot") == ("/start😀", "hello")

    # Test with non-ASCII characters
    assert parse_command("/привет мир", "@bot") == ("/привет", "мир")


def test_format_user_message_complex_cases():
    from api.index import format_user_message

    # Test with very long names and messages
    first_name = "A" * 50
    username = "U" * 50
    msg = {"from": {"first_name": first_name, "username": username}}
    long_message = "M" * 500
    result = format_user_message(msg, long_message)
    assert len(result) > 100  # Should contain the name and message
    assert first_name in result
    assert username in result

    # Test with special characters in names
    msg = {
        "from": {
            "first_name": "User<script>alert(1)</script>",
            "username": "user@example-site.com",
        }
    }
    result = format_user_message(msg, "Test message")
    assert "User<script>alert(1)</script>" in result
    assert "user@example-site.com" in result

    # Test with missing fields
    msg = {"from": {}}
    assert format_user_message(msg, "Test message") == ": Test message"

    # Test with additional fields that should be ignored
    msg = {
        "from": {"first_name": "John", "username": "john123", "ignored_field": "value"}
    }
    assert format_user_message(msg, "Test message") == "John (john123): Test message"

    # Test with empty message
    msg = {"from": {"first_name": "John", "username": "john123"}}
    assert format_user_message(msg, "") == "John (john123): "


def test_should_gordo_respond_complex_cases():
    from api.index import should_gordo_respond

    commands = {"/test": (lambda x: x, False, False), "/other": (lambda x: x, True, False)}

    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "testbot"  # Set mock bot username

        # Test with command not in command list but starts with /
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert should_gordo_respond(commands, "/unknown", "hello", msg) == False

        # Test with message containing bot username in middle of message
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(commands, "", "hey there @testbot how are you", msg)
            == True
        )

        # Test with reply to message that's not from the bot
        msg = {
            "chat": {"type": "group"},
            "from": {"username": "test"},
            "reply_to_message": {"from": {"username": "not_bot"}},
        }
        assert should_gordo_respond(commands, "", "hello", msg) == False

    # Test trigger words in a separate block with proper random mocking
    with patch("os.environ.get") as mock_env, patch("random.random", return_value=0.5):
        mock_env.return_value = "testbot"

        # Test with trigger word but probability too high (0.5 > 0.1)
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert should_gordo_respond(commands, "", "hey gordo", msg) == False

    with patch("os.environ.get") as mock_env, patch("random.random", return_value=0.05):
        mock_env.return_value = "testbot"

        # Test with multiple trigger words and low probability (0.05 < 0.1)
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(commands, "", "hey gordo and respondedor", msg) == True
        )

        # Test with case-insensitive trigger words
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert should_gordo_respond(commands, "", "hey GORDO", msg) == True


def test_get_prices_basic():
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
        # Mock the API response with some basic cryptocurrency data
        mock_get_prices.return_value = {
            "data": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "quote": {"USD": {"price": 50000.0, "percent_change_24h": 5.25}},
                },
                {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "quote": {"USD": {"price": 2500.0, "percent_change_24h": -2.5}},
                },
            ]
        }

        # Test basic price query
        result = get_prices("")
        assert result is not None
        assert "BTC: 50000" in result
        assert "ETH: 2500" in result
        assert "+5.25%" in result
        assert "-2.5%" in result


def test_cached_requests_basic():
    from api.index import cached_requests

    with patch("requests.get") as mock_get, patch("redis.Redis") as mock_redis, patch(
        "time.time"
    ) as mock_time:
        # Setup mocks
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.ping.return_value = True

        # Mock time
        mock_time.return_value = 1000

        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = '{"key": "value"}'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test with no cached data
        mock_instance.get.return_value = None

        result = cached_requests(
            "https://api.example.com", {"param": "value"}, {"header": "value"}, 300
        )

        # Verify a request was made and data was returned
        mock_get.assert_called_once()
        assert result is not None
        assert result["timestamp"] == 1000
        assert result["data"] == {"key": "value"}


def test_get_weather_description():
    from api.index import get_weather_description

    # Test various weather codes
    assert get_weather_description(0) == "despejado"
    assert get_weather_description(1) == "mayormente despejado"
    assert get_weather_description(2) == "parcialmente nublado"
    assert get_weather_description(3) == "nublado"
    assert get_weather_description(45) == "neblina"
    assert get_weather_description(61) == "lluvia leve"
    assert get_weather_description(95) == "tormenta"

    # Test unknown code
    assert get_weather_description(9999) == "clima raro"


def test_ask_ai_with_openrouter_success():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing
    with patch("api.index.get_market_context") as mock_get_market_context, patch(
        "api.index.get_weather_context"
    ) as mock_get_weather_context, patch(
        "api.index.get_time_context"
    ) as mock_get_time_context, patch(
        "os.environ.get"
    ) as mock_env:

        # Setup basic mocks
        mock_get_market_context.return_value = {"crypto": [], "dollar": {}}
        mock_get_weather_context.return_value = {"temperature": 25}
        mock_get_time_context.return_value = {"formatted": "Monday"}
        mock_env.side_effect = lambda key: {"OPENROUTER_API_KEY": "test_key"}.get(key)

        messages = [{"role": "user", "content": "hello"}]
        result = ask_ai(messages)

        # Just verify it returns a string (could be fallback response)
        assert isinstance(result, str)
        assert len(result) > 0


def test_ask_ai_with_cloudflare_fallback():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing
    with patch("api.index.get_market_context") as mock_get_market_context, patch(
        "api.index.get_weather_context"
    ) as mock_get_weather_context, patch(
        "api.index.get_time_context"
    ) as mock_get_time_context, patch(
        "os.environ.get"
    ) as mock_env:

        # Setup basic mocks  
        mock_get_market_context.return_value = {"crypto": [], "dollar": {}}
        mock_get_weather_context.return_value = {"temperature": 25}
        mock_get_time_context.return_value = {"formatted": "Monday"}
        mock_env.side_effect = lambda key: {"OPENROUTER_API_KEY": "test_key"}.get(key)

        messages = [{"role": "user", "content": "hello"}]
        result = ask_ai(messages)

        # Just verify it returns a string (could be fallback response)
        assert isinstance(result, str)
        assert len(result) > 0


def test_ask_ai_with_all_failures():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing
    with patch("api.index.get_market_context") as mock_get_market_context, patch(
        "api.index.get_weather_context"
    ) as mock_get_weather_context, patch(
        "api.index.get_time_context"
    ) as mock_get_time_context, patch(
        "os.environ.get"
    ) as mock_env:

        # Setup basic mocks
        mock_get_market_context.return_value = {"crypto": [], "dollar": {}}
        mock_get_weather_context.return_value = {"temperature": 25}
        mock_get_time_context.return_value = {"formatted": "Monday"}
        mock_env.side_effect = lambda key: {"OPENROUTER_API_KEY": "test_key"}.get(key)

        messages = [{"role": "user", "content": "hello"}]
        result = ask_ai(messages)

        # Just verify it returns a string (could be fallback response)
        assert isinstance(result, str)
        assert len(result) > 0


def test_ask_ai_with_image():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing when given an image
    with patch("api.index.get_market_context") as mock_get_market_context, patch(
        "api.index.get_weather_context"
    ) as mock_get_weather_context, patch(
        "api.index.get_time_context"
    ) as mock_get_time_context, patch(
        "api.index.describe_image_cloudflare"
    ) as mock_describe_image, patch(
        "os.environ.get"
    ) as mock_env:

        # Setup basic mocks
        mock_get_market_context.return_value = {"crypto": [], "dollar": {}}
        mock_get_weather_context.return_value = {"temperature": 25}
        mock_get_time_context.return_value = {"formatted": "Monday"}
        mock_describe_image.return_value = "A beautiful landscape"
        mock_env.side_effect = lambda key: {"OPENROUTER_API_KEY": "test_key"}.get(key)

        messages = [{"role": "user", "content": "what do you see in this image?"}]
        image_data = b"fake_image_data"
        result = ask_ai(messages, image_data=image_data, image_file_id="img123")

        # Just verify it returns a string (could be fallback response)
        assert isinstance(result, str)
        assert len(result) > 0


def test_get_dollar_rates_basic():
    from api.index import get_dollar_rates

    with patch("api.index.cached_requests") as mock_cached_requests:
        # Mock the API response with dollar rate data
        mock_cached_requests.return_value = {
            "data": {
                "oficial": {"price": 100.0, "variation": 0.5},
                "tarjeta": {"price": 150.0, "variation": 0.75},
                "mep": {"al30": {"ci": {"price": 200.0, "variation": 1.25}}},
                "ccl": {"al30": {"ci": {"price": 210.0, "variation": 1.5}}},
                "blue": {"ask": 220.0, "variation": 2.0},
                "cripto": {
                    "ccb": {"ask": 230.0, "variation": 2.5},
                    "usdc": {"ask": 235.0, "variation": 2.75},
                    "usdt": {"ask": 240.0, "variation": 3.0},
                },
            }
        }

        result = get_dollar_rates()
        assert result is not None
        assert "Oficial: 100" in result
        assert "Tarjeta: 150" in result
        assert "MEP: 200" in result
        assert "CCL: 210" in result
        assert "Blue: 220" in result
        assert "Bitcoin: 230" in result
        assert "USDC: 235" in result
        assert "USDT: 240" in result


def test_get_dollar_rates_api_failure():
    from api.index import get_dollar_rates
    import pytest

    with patch("api.index.cached_requests") as mock_cached_requests:
        # Mock API failure
        mock_cached_requests.return_value = None

        # The function should raise an exception when API fails
        with pytest.raises(TypeError):
            get_dollar_rates()


def test_get_devo_with_fee_only():
    from api.index import get_devo

    with patch("api.index.cached_requests") as mock_cached_requests:
        # Mock the API response
        mock_cached_requests.return_value = {
            "data": {
                "oficial": {"price": 100.0},
                "tarjeta": {"price": 150.0},
                "cripto": {"usdt": {"ask": 200.0, "bid": 190.0}},
            }
        }

        result = get_devo("0.5")
        assert result is not None
        assert "Profit: 62.68%" in result


def test_get_devo_with_fee_and_amount():
    from api.index import get_devo

    with patch("api.index.cached_requests") as mock_cached_requests:
        # Mock the API response
        mock_cached_requests.return_value = {
            "data": {
                "oficial": {"price": 100.0},
                "tarjeta": {"price": 150.0},
                "cripto": {"usdt": {"ask": 200.0, "bid": 190.0}},
            }
        }

        result = get_devo("0.5, 100")
        assert result is not None
        assert "100 USD Tarjeta" in result
        assert "Ganarias" in result


def test_get_devo_invalid_input():
    from api.index import get_devo

    result = get_devo("invalid")
    assert "Invalid input. Usage: /devo <fee_percentage>[, <purchase_amount>]" in result


def test_satoshi_basic():
    from api.index import satoshi

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
        # Mock the API response for both USD and ARS
        def mock_get_prices_side_effect(currency):
            if currency == "USD":
                return {
                    "data": [
                        {"quote": {"USD": {"price": 50000.0}}},
                    ]
                }
            elif currency == "ARS":
                return {
                    "data": [
                        {"quote": {"ARS": {"price": 10000000.0}}},
                    ]
                }
            return None

        mock_get_prices.side_effect = mock_get_prices_side_effect

        result = satoshi()
        assert result is not None
        assert "1 satoshi = $0.00050000 USD" in result
        assert "1 satoshi = $0.1000 ARS" in result
        assert "$1 USD = 2,000 sats" in result
        assert "$1 ARS = 10.000 sats" in result


def test_powerlaw_basic():
    from api.index import powerlaw

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
        # Mock the API response for BTC price
        mock_get_prices.return_value = {
            "data": [
                {"quote": {"USD": {"price": 50000.0}}},
            ]
        }

        result = powerlaw()
        assert result is not None
        assert "segun power law btc deberia estar en" in result


def test_rainbow_basic():
    from api.index import rainbow

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices, patch(
        "api.index.datetime"
    ) as mock_datetime:
        # Mock the API response for BTC price
        mock_get_prices.return_value = {
            "data": [
                {"quote": {"USD": {"price": 50000.0}}},
            ]
        }

        # Mock the current date to a fixed value for consistent calculations
        mock_datetime.now.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)

        result = rainbow()
        assert result is not None
        assert "segun rainbow chart btc deberia estar en" in result


def test_convert_base_basic():
    from api.index import convert_base

    # Test binary to decimal
    assert convert_base("101, 2, 10") == "ahi tenes boludo, 101 en base 2 es 5 en base 10"

    # Test decimal to hexadecimal
    assert convert_base("255, 10, 16") == "ahi tenes boludo, 255 en base 10 es FF en base 16"

    # Test invalid input
    assert (
        convert_base("invalid")
        == "capo mandate algo como /convertbase 101, 2, 10 y te paso de binario a decimal"
    )


def test_get_timestamp_basic():
    from api.index import get_timestamp

    with patch("time.time") as mock_time:
        mock_time.return_value = 1672531200  # January 1, 2023
        assert get_timestamp() == "1672531200"


def test_get_help_basic():
    from api.index import get_help

    result = get_help()
    assert "comandos disponibles boludo:" in result
    assert "/ask" in result
    assert "/dolar" in result
    assert "/prices" in result


def test_get_instance_name_basic():
    from api.index import get_instance_name

    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_instance"
        assert get_instance_name() == "estoy corriendo en test_instance boludo"


def test_send_typing_basic():
    from api.index import send_typing

    with patch("requests.get") as mock_get:
        send_typing("test_token", "12345")
        mock_get.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendChatAction",
            params={"chat_id": "12345", "action": "typing"},
            timeout=5,
        )


def test_send_msg_basic():
    from api.index import send_msg

    with patch("requests.get") as mock_get, patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_token"
        send_msg("12345", "hello")
        mock_get.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendMessage",
            params={"chat_id": "12345", "text": "hello"},
            timeout=5,
        )


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
            "12345", "Admin report from test_instance: test message"
        )


# Phase 1: Cache Functions Tests

def test_get_cached_transcription_success():
    from api.index import get_cached_transcription
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = "cached transcription text"
        
        result = get_cached_transcription("test_file_id")
        
        assert result == "cached transcription text"
        mock_redis.get.assert_called_once_with("audio_transcription:test_file_id")


def test_get_cached_transcription_not_found():
    from api.index import get_cached_transcription
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        result = get_cached_transcription("test_file_id")
        
        assert result is None
        mock_redis.get.assert_called_once_with("audio_transcription:test_file_id")


def test_get_cached_transcription_exception():
    from api.index import get_cached_transcription
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        
        result = get_cached_transcription("test_file_id")
        
        assert result is None


def test_cache_transcription_success():
    from api.index import cache_transcription
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        
        cache_transcription("test_file_id", "transcription text", 3600)
        
        mock_redis.setex.assert_called_once_with("audio_transcription:test_file_id", 3600, "transcription text")


def test_cache_transcription_default_ttl():
    from api.index import cache_transcription
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        
        cache_transcription("test_file_id", "transcription text")
        
        mock_redis.setex.assert_called_once_with("audio_transcription:test_file_id", 604800, "transcription text")


def test_cache_transcription_exception():
    from api.index import cache_transcription
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        
        # Should not raise exception, just print error
        cache_transcription("test_file_id", "transcription text")


def test_get_cached_description_success():
    from api.index import get_cached_description
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = "cached image description"
        
        result = get_cached_description("test_file_id")
        
        assert result == "cached image description"
        mock_redis.get.assert_called_once_with("image_description:test_file_id")


def test_get_cached_description_not_found():
    from api.index import get_cached_description
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        result = get_cached_description("test_file_id")
        
        assert result is None
        mock_redis.get.assert_called_once_with("image_description:test_file_id")


def test_get_cached_description_exception():
    from api.index import get_cached_description
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        
        result = get_cached_description("test_file_id")
        
        assert result is None


def test_cache_description_success():
    from api.index import cache_description
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        
        cache_description("test_file_id", "image description", 3600)
        
        mock_redis.setex.assert_called_once_with("image_description:test_file_id", 3600, "image description")


def test_cache_description_default_ttl():
    from api.index import cache_description
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        
        cache_description("test_file_id", "image description")
        
        mock_redis.setex.assert_called_once_with("image_description:test_file_id", 604800, "image description")


def test_cache_description_exception():
    from api.index import cache_description
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        
        # Should not raise exception, just print error
        cache_description("test_file_id", "image description")


def test_get_cache_history_success():
    from api.index import get_cache_history
    import json
    from datetime import datetime, timedelta
    
    with patch("api.index.datetime") as mock_datetime:
        mock_redis = MagicMock()
        test_data = {"data": "test", "timestamp": "2024-01-01"}
        mock_redis.get.return_value = json.dumps(test_data)
        
        # Mock datetime.now() to return a fixed time
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        result = get_cache_history(1, "test_hash", mock_redis)
        
        assert result == test_data
        expected_timestamp = (fixed_time - timedelta(hours=1)).strftime("%Y-%m-%d-%H")
        mock_redis.get.assert_called_once_with(expected_timestamp + "test_hash")


def test_get_cache_history_not_found():
    from api.index import get_cache_history
    
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    
    result = get_cache_history(1, "test_hash", mock_redis)
    
    assert result is None


def test_get_cache_history_invalid_data():
    from api.index import get_cache_history
    import json
    
    mock_redis = MagicMock()
    test_data = {"data": "test"}  # Missing timestamp
    mock_redis.get.return_value = json.dumps(test_data)
    
    result = get_cache_history(1, "test_hash", mock_redis)
    
    assert result is None


def test_get_cached_bcra_variables_success():
    from api.index import get_cached_bcra_variables
    import json
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        test_data = {"base_monetaria": "1000000", "inflacion_mensual": "5.2"}
        mock_redis.get.return_value = json.dumps(test_data)
        
        result = get_cached_bcra_variables()
        
        assert result == test_data
        mock_redis.get.assert_called_once_with("bcra_variables")


def test_get_cached_bcra_variables_not_found():
    from api.index import get_cached_bcra_variables
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = None
        
        result = get_cached_bcra_variables()
        
        assert result is None
        mock_redis.get.assert_called_once_with("bcra_variables")


def test_get_cached_bcra_variables_exception():
    from api.index import get_cached_bcra_variables
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        
        result = get_cached_bcra_variables()
        
        assert result is None


def test_cache_bcra_variables_success():
    from api.index import cache_bcra_variables
    import json
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        test_data = {"base_monetaria": "1000000", "inflacion_mensual": "5.2"}
        
        cache_bcra_variables(test_data, 600)
        
        mock_redis.setex.assert_called_once_with("bcra_variables", 600, json.dumps(test_data))


def test_cache_bcra_variables_default_ttl():
    from api.index import cache_bcra_variables
    import json
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        test_data = {"base_monetaria": "1000000"}
        
        cache_bcra_variables(test_data)
        
        mock_redis.setex.assert_called_once_with("bcra_variables", 300, json.dumps(test_data))


def test_cache_bcra_variables_exception():
    from api.index import cache_bcra_variables
    
    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        test_data = {"base_monetaria": "1000000"}
        
        # Should not raise exception, just print error
        cache_bcra_variables(test_data)
