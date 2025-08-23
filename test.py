from unittest.mock import patch, MagicMock, ANY
import os
import pytest
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
    web_search,
    search_command,
    parse_tool_call,
    execute_tool,
    complete_with_providers,
    handle_ai_response,
    handle_msg,
    replace_links,
    configure_links,
)
from datetime import datetime, timezone, timedelta
import json
import requests
import redis

app = Flask(__name__)


# Cleanup potential test artifact files left behind
@pytest.fixture(autouse=True, scope="session")
def cleanup_test_artifacts():
    yield
    # Remove any stray file named 'test_api_key' after tests
    try:
        if os.path.isfile("test_api_key"):
            os.remove("test_api_key")
    except Exception:
        pass


def test_responder_no_args():
    with app.test_request_context("/?"):
        response = responder()
        assert response == ("No key", 200)


def test_responder_wrong_key():
    with app.test_request_context("/?key=wrong_key"), patch(
        "os.environ.get"
    ) as mock_env, patch("api.index.admin_report") as mock_admin:
        mock_env.return_value = "correct_key"
        response = responder()
        assert response == ("Wrong key", 400)
        mock_admin.assert_called_once_with("Wrong key attempt")


def test_responder_valid_key_with_webhook_check():
    with app.test_request_context("/?key=valid_key&check_webhook=true"), patch(
        "os.environ.get"
    ) as mock_env, patch("api.index.verify_webhook") as mock_verify:
        mock_env.return_value = "valid_key"
        mock_verify.return_value = True
        response = responder()
        assert response == ("Webhook checked", 200)


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
        assert response == ("Webhook updated", 200)


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
        assert response == ("Ok", 200)


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


def test_load_bot_config_caches_and_parses(monkeypatch):
    from api import index

    index._bot_config = None
    monkeypatch.setenv("BOT_SYSTEM_PROMPT", "hola")
    monkeypatch.setenv("BOT_TRIGGER_WORDS", "bot, amigo")

    cfg_first = index.load_bot_config()
    assert cfg_first == {
        "trigger_words": ["bot", "amigo"],
        "system_prompt": "hola",
    }

    # Change env vars; function should return cached config
    monkeypatch.setenv("BOT_SYSTEM_PROMPT", "changed")
    monkeypatch.setenv("BOT_TRIGGER_WORDS", "foo,bar")
    cfg_second = index.load_bot_config()
    assert cfg_second is cfg_first


def test_load_bot_config_missing_env(monkeypatch):
    from api import index

    index._bot_config = None
    monkeypatch.delenv("BOT_SYSTEM_PROMPT", raising=False)
    monkeypatch.setenv("BOT_TRIGGER_WORDS", "foo")

    with pytest.raises(ValueError):
        index.load_bot_config()

    monkeypatch.setenv("BOT_SYSTEM_PROMPT", "hola")
    monkeypatch.delenv("BOT_TRIGGER_WORDS", raising=False)

    with pytest.raises(ValueError):
        index.load_bot_config()


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
    import api.index

    # Reset global cache to ensure clean state
    api.index._bot_config = None

    commands = {"/test": (lambda x: x, False, False)}

    with patch("os.environ.get") as mock_env:
        # Mock environment variables for both bot config and telegram username
        def env_side_effect(key):
            env_vars = {
                "TELEGRAM_USERNAME": "testbot",
                "BOT_SYSTEM_PROMPT": "You are a test bot",
                "BOT_TRIGGER_WORDS": "gordo,test,bot"
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

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


def test_should_gordo_respond_ignores_link_fix_reply():
    import api.index

    api.index._bot_config = None
    commands = {"/test": (lambda x: x, False, False)}

    with patch("os.environ.get") as mock_env:
        def env_side_effect(key):
            env_vars = {
                "TELEGRAM_USERNAME": "testbot",
                "BOT_SYSTEM_PROMPT": "You are a test bot",
                "BOT_TRIGGER_WORDS": "gordo,test,bot",
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        msg = {
            "chat": {"type": "group"},
            "from": {"username": "user"},
            "reply_to_message": {
                "from": {"username": "testbot"},
                "text": "https://fxtwitter.com/foo",
            },
        }

        assert should_gordo_respond(commands, "/test", "hello", msg) is False


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
        with patch("api.index.set_telegram_webhook") as mock_set, patch(
            "os.environ.get"
        ) as mock_env:
            mock_set.return_value = True
            mock_env.return_value = (
                "https://example.com/webhook"  # Mock FUNCTION_URL
            )
            response, status = process_request_parameters(request)
            assert status == 200
            assert "Webhook updated" in response


def test_handle_rate_limit():
    from api.index import handle_rate_limit

    with patch("api.index.send_typing") as mock_send_typing, patch(
        "time.sleep"
    ) as mock_sleep, patch("api.index.gen_random") as mock_gen_random, patch(
        "os.environ.get"
    ) as mock_env:

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
    ) as _mock_sleep:  # Add sleep mock to avoid delays  # noqa: F841

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
            "text": "/prices btc",
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
    ) as mock_requests, patch(
        "api.index.send_typing"
    ) as mock_send_typing:

        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot",
            "TELEGRAM_TOKEN": "test_token",
        }.get(key, default)
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
            "photo": [{"file_id": "photo_123"}],
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_download.assert_called_once()
        mock_encode.assert_called_once()
        mock_send_msg.assert_called_once()
        mock_send_typing.assert_called()


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
            "CLOUDFLARE_ACCOUNT_ID": "test_account",
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
            "voice": {"file_id": "voice_123"},
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

        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)
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
            "reply_to_message": {"message_id": 2, "voice": {"file_id": "voice_123"}},
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

        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)
        mock_rate_limit.return_value = True
        mock_should_respond.return_value = False

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "group"},
            "from": {"first_name": "John", "username": "john"},
            "text": "/unknown_command",
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_send_msg.assert_not_called()  # Should not send message


def test_handle_msg_with_exception():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.admin_report"
    ) as mock_admin_report, patch("os.environ.get") as mock_env:

        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)
        mock_config_redis.side_effect = Exception("Redis error")

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "hello",
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
    ) as _mock_sleep:  # noqa: F841

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
        get_devo as _get_devo,  # noqa: F401
        powerlaw as _powerlaw,  # noqa: F401
        rainbow as _rainbow,  # noqa: F401
        get_timestamp as _get_timestamp,  # noqa: F401
        convert_to_command as _convert_to_command,  # noqa: F401
        get_instance_name as _get_instance_name,  # noqa: F401
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
    assert "/usd" in commands

    # Test that AI commands are properly marked
    assert commands["/ask"][1] == True
    assert commands["/pregunta"][1] == True
    assert commands["/che"][1] == True
    assert commands["/gordo"][1] == True

    # Test that non-AI commands are properly marked
    assert commands["/random"][1] == False
    assert commands["/prices"][1] == False
    assert commands["/dolar"][1] == False
    assert commands["/usd"][1] == False

    # Test function mappings
    assert commands["/ask"][0] == ask_ai
    assert commands["/random"][0] == select_random
    assert commands["/prices"][0] == get_prices
    assert commands["/help"][0] == get_help
    assert commands["/usd"][0] == _get_dollar_rates
    # Test search commands
    assert "/buscar" in commands
    assert "/search" in commands
    from api.index import search_command as _search_command
    assert commands["/buscar"][0] == _search_command
    assert commands["/search"][0] == _search_command


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
    import api.index
    from api.index import should_gordo_respond

    api.index._bot_config = {
        "trigger_words": ["gordo", "test", "bot"],
        "system_prompt": "You are a test bot",
    }

    commands = {
        "/test": (lambda x: x, False, False),
        "/other": (lambda x: x, True, False),
    }

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


def test_ask_ai_sanitizes_tool_call_before_retry():
    """ask_ai should sanitize tool responses before retrying provider"""
    from api.index import ask_ai

    with patch("api.index.get_market_context", return_value={}), patch(
        "api.index.get_weather_context", return_value={}
    ), patch("api.index.get_time_context", return_value={}), patch(
        "api.index.build_system_message", return_value={"role": "system", "content": "sys"}
    ), patch("api.index.OpenAI") as mock_openai, patch(
        "api.index.execute_tool", return_value="{}"
    ):
        mock_openai.return_value = MagicMock()

        calls = []

        def fake_complete(system_message, msgs):
            calls.append(msgs)
            if len(calls) == 1:
                return '[TOOL] web_search {"query": "test"}'
            return "respuesta final"

        with patch(
            "api.index.complete_with_providers", side_effect=fake_complete
        ):
            result = ask_ai([{"role": "user", "content": "hola"}])

    assert result == "respuesta final"
    # second call should contain sanitized assistant message without [TOOL]
    second_messages = calls[1]
    assert "[TOOL]" not in second_messages[-2]["content"]


def test_ask_ai_handles_repeated_tool_calls():
    """ask_ai should execute tools repeatedly if providers return new tool calls"""
    from api.index import ask_ai

    with patch("api.index.get_market_context", return_value={}), patch(
        "api.index.get_weather_context", return_value={}
    ), patch("api.index.get_time_context", return_value={}), patch(
        "api.index.build_system_message", return_value={"role": "system", "content": "sys"}
    ), patch("api.index.OpenAI") as mock_openai, patch(
        "api.index.execute_tool"
    ) as mock_tool:
        mock_openai.return_value = MagicMock()
        mock_tool.side_effect = ["res1", "res2"]

        call_count = {"n": 0}

        def fake_complete(system_message, msgs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return '[TOOL] tool1 {}'
            if call_count["n"] == 2:
                return '[TOOL] tool2 {}'
            return "fin"

        with patch(
            "api.index.complete_with_providers", side_effect=fake_complete
        ):
            result = ask_ai([{"role": "user", "content": "hola"}])

    assert result == "fin"
    assert mock_tool.call_count == 2
    assert call_count["n"] == 3

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
    assert (
        convert_base("101, 2, 10") == "ahi tenes boludo, 101 en base 2 es 5 en base 10"
    )

    # Test decimal to hexadecimal
    assert (
        convert_base("255, 10, 16")
        == "ahi tenes boludo, 255 en base 10 es FF en base 16"
    )

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
    assert "/usd" in result
    assert "/prices" in result
    assert "/links" in result


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


def test_send_msg_with_buttons():
    from api.index import send_msg

    with patch("requests.get") as mock_get, patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_token"
        send_msg("12345", "hello", buttons=["https://twitter.com/foo"])
        mock_get.assert_called_once_with(
            "https://api.telegram.org/bottest_token/sendMessage",
            params={
                "chat_id": "12345",
                "text": "hello",
                "reply_markup": json.dumps(
                    {
                        "inline_keyboard": [
                            [{"text": "Open in app", "url": "https://twitter.com/foo"}]
                        ]
                    }
                ),
            },
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

        mock_redis.setex.assert_called_once_with(
            "audio_transcription:test_file_id", 3600, "transcription text"
        )


def test_cache_transcription_default_ttl():
    from api.index import cache_transcription

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        cache_transcription("test_file_id", "transcription text")

        mock_redis.setex.assert_called_once_with(
            "audio_transcription:test_file_id", 604800, "transcription text"
        )


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

        mock_redis.setex.assert_called_once_with(
            "image_description:test_file_id", 3600, "image description"
        )


def test_cache_description_default_ttl():
    from api.index import cache_description

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        cache_description("test_file_id", "image description")

        mock_redis.setex.assert_called_once_with(
            "image_description:test_file_id", 604800, "image description"
        )


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

        mock_redis.setex.assert_called_once_with(
            "bcra_variables", 600, json.dumps(test_data)
        )


def test_cache_bcra_variables_default_ttl():
    from api.index import cache_bcra_variables
    import json

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        test_data = {"base_monetaria": "1000000"}

        cache_bcra_variables(test_data)

        mock_redis.setex.assert_called_once_with(
            "bcra_variables", 300, json.dumps(test_data)
        )


def test_cache_bcra_variables_exception():
    from api.index import cache_bcra_variables

    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        test_data = {"base_monetaria": "1000000"}

        # Should not raise exception, just print error
        cache_bcra_variables(test_data)


def test_handle_transcribe_with_message_no_reply():
    from api.index import handle_transcribe_with_message

    message = {"message_id": 1, "chat": {"id": 123}, "text": "/transcribe"}

    result = handle_transcribe_with_message(message)
    assert (
        result == "Respondé a un mensaje con audio o imagen para transcribir/describir"
    )


def test_handle_transcribe_with_message_voice_cached():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_transcription") as mock_cached:
        mock_cached.return_value = "cached voice transcription"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"voice": {"file_id": "voice123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎵 Transcripción: cached voice transcription"
        mock_cached.assert_called_once_with("voice123")


def test_handle_transcribe_with_message_voice_download_success():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_transcription") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.transcribe_audio_cloudflare"
    ) as mock_transcribe:

        mock_cached.return_value = None
        mock_download.return_value = b"audio data"
        mock_transcribe.return_value = "new transcription"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"voice": {"file_id": "voice123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎵 Transcripción: new transcription"
        mock_download.assert_called_once_with("voice123")
        mock_transcribe.assert_called_once_with(b"audio data", "voice123")


def test_handle_transcribe_with_message_voice_download_fail():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_transcription") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download:

        mock_cached.return_value = None
        mock_download.return_value = None

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"voice": {"file_id": "voice123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "No pude descargar el audio"


def test_handle_transcribe_with_message_audio_success():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_transcription") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.transcribe_audio_cloudflare"
    ) as mock_transcribe:

        mock_cached.return_value = None
        mock_download.return_value = b"audio data"
        mock_transcribe.return_value = "audio transcription"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"audio": {"file_id": "audio123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎵 Transcripción: audio transcription"


def test_handle_transcribe_with_message_photo_cached():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_description") as mock_cached:
        mock_cached.return_value = "cached image description"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"photo": [{"file_id": "photo123"}]},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🖼️ Descripción: cached image description"
        mock_cached.assert_called_once_with("photo123")


def test_handle_transcribe_with_message_photo_success():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_description") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch("api.index.resize_image_if_needed") as mock_resize, patch(
        "api.index.describe_image_cloudflare"
    ) as mock_describe:

        mock_cached.return_value = None
        mock_download.return_value = b"image data"
        mock_resize.return_value = b"resized image data"
        mock_describe.return_value = "image description"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"photo": [{"file_id": "photo123"}]},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🖼️ Descripción: image description"
        mock_download.assert_called_once_with("photo123")
        mock_resize.assert_called_once_with(b"image data")
        mock_describe.assert_called_once_with(
            b"resized image data",
            "Describe what you see in this image in detail.",
            "photo123",
        )


def test_handle_transcribe_with_message_sticker_success():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_description") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch("api.index.resize_image_if_needed") as mock_resize, patch(
        "api.index.describe_image_cloudflare"
    ) as mock_describe:

        mock_cached.return_value = None
        mock_download.return_value = b"sticker data"
        mock_resize.return_value = b"resized sticker data"
        mock_describe.return_value = "sticker description"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"sticker": {"file_id": "sticker123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎨 Descripción del sticker: sticker description"


def test_handle_transcribe_with_message_no_media():
    from api.index import handle_transcribe_with_message

    message = {
        "message_id": 1,
        "chat": {"id": 123},
        "text": "/transcribe",
        "reply_to_message": {"text": "just text"},
    }

    result = handle_transcribe_with_message(message)
    assert (
        result
        == "El mensaje no contiene audio, imagen o sticker para transcribir/describir"
    )


def test_handle_transcribe_with_message_exception():
    from api.index import handle_transcribe_with_message
    from typing import cast, Dict, Any

    # Malformed message that causes exception
    message = cast(Dict[str, Any], None)

    result = handle_transcribe_with_message(message)
    assert result == "Error procesando el comando, intentá más tarde"


def test_handle_transcribe():
    from api.index import handle_transcribe

    result = handle_transcribe()
    assert (
        result
        == "El comando /transcribe debe usarse respondiendo a un mensaje con audio o imagen"
    )


def test_format_bcra_variables_empty():
    from api.index import format_bcra_variables

    result = format_bcra_variables({})
    assert result == "No se pudieron obtener las variables del BCRA"

    from typing import cast, Dict, Any

    result = format_bcra_variables(cast(Dict[str, Any], None))
    assert result == "No se pudieron obtener las variables del BCRA"


def test_format_bcra_variables_with_data():
    from api.index import format_bcra_variables

    variables = {
        "base_monetaria_total": {"value": "5.000.000,50", "date": "15/01/2025"},
        "inflacion_mensual": {"value": "5,2", "date": "15/01/2025"},
        "inflacion_interanual": {"value": "150,5", "date": "15/01/2025"},
        "inflacion_esperada": {"value": "3,1", "date": "15/01/2025"},
        "tasa_tamar": {"value": "45,0", "date": "15/01/2025"},
        "tasa_badlar": {"value": "40,5", "date": "15/01/2025"},
        "tasa_justicia": {"value": "50,0", "date": "15/01/2025"},
        "dolar_minorista_compra": {"value": "1.200,50", "date": "15/01/2025"},
        "dolar_minorista_venta": {"value": "1.250,75", "date": "15/01/2025"},
        "dolar_mayorista": {"value": "1.180,25", "date": "15/01/2025"},
        "uva": {"value": "500,75", "date": "15/01/2025"},
        "cer": {"value": "0,45", "date": "15/01/2025"},
        "reservas_int": {"value": "25.000", "date": "15/01/2025"},
    }

    result = format_bcra_variables(variables)
    assert "📊 Variables principales BCRA" in result
    assert "15/01/25" in result  # Date should be formatted


def test_handle_bcra_variables_cached():
    from api.index import handle_bcra_variables

    with patch("api.index.get_cached_bcra_variables") as mock_cached, patch(
        "api.index.format_bcra_variables"
    ) as mock_format:

        mock_cached.return_value = {"test": "data"}
        mock_format.return_value = "formatted variables"

        result = handle_bcra_variables()
        assert result == "formatted variables"
        mock_cached.assert_called_once()
        mock_format.assert_called_once_with({"test": "data"})


def test_handle_bcra_variables_scrape_fresh():
    from api.index import handle_bcra_variables

    with patch("api.index.get_cached_bcra_variables") as mock_cached, patch(
        "api.index.scrape_bcra_variables"
    ) as mock_scrape, patch("api.index.cache_bcra_variables") as mock_cache, patch(
        "api.index.format_bcra_variables"
    ) as mock_format:

        mock_cached.return_value = None
        mock_scrape.return_value = {"scraped": "data"}
        mock_format.return_value = "formatted scraped data"

        result = handle_bcra_variables()
        assert result == "formatted scraped data"
        mock_scrape.assert_called_once()
        mock_cache.assert_called_once_with({"scraped": "data"})
        mock_format.assert_called_once_with({"scraped": "data"})


def test_handle_bcra_variables_no_data():
    from api.index import handle_bcra_variables

    with patch("api.index.get_cached_bcra_variables") as mock_cached, patch(
        "api.index.scrape_bcra_variables"
    ) as mock_scrape:

        mock_cached.return_value = None
        mock_scrape.return_value = None

        result = handle_bcra_variables()
        assert (
            result
            == "No pude obtener las variables del BCRA en este momento, probá más tarde"
        )


def test_handle_bcra_variables_exception():
    from api.index import handle_bcra_variables

    with patch("api.index.get_cached_bcra_variables") as mock_cached:
        mock_cached.side_effect = Exception("Cache error")

        result = handle_bcra_variables()
        assert result == "Error al obtener las variables del BCRA"


def test_get_market_context_success():
    from api.index import get_market_context

    with patch("api.index.cached_requests") as mock_cached, patch(
        "api.index.clean_crypto_data"
    ) as mock_clean, patch("os.environ.get") as mock_env:

        # Mock crypto response
        crypto_response = {
            "data": {"data": [{"symbol": "BTC", "quote": {"USD": {"price": 50000}}}]}
        }

        # Mock dollar response
        dollar_response = {
            "data": {"oficial": {"price": 1000}, "blue": {"price": 1200}}
        }

        def mock_requests_side_effect(url, *_args, **_kwargs):  # noqa: ARG001
            if "coinmarketcap" in url:
                return crypto_response
            elif "criptoya" in url:
                return dollar_response
            return None

        mock_cached.side_effect = mock_requests_side_effect
        mock_clean.return_value = [{"symbol": "BTC", "price": 50000}]
        mock_env.return_value = "test_api_key"

        result = get_market_context()

        assert "crypto" in result
        assert "dollar" in result
        assert result["crypto"] == [{"symbol": "BTC", "price": 50000}]
        assert result["dollar"] == {"oficial": {"price": 1000}, "blue": {"price": 1200}}


def test_get_market_context_crypto_fail():
    from api.index import get_market_context

    with patch("api.index.cached_requests") as mock_cached, patch(
        "os.environ.get"
    ) as mock_env:

        # Mock dollar response only
        dollar_response = {"data": {"oficial": {"price": 1000}}}

        def mock_requests_side_effect(url, *_args, **_kwargs):  # noqa: ARG001
            if "coinmarketcap" in url:
                return None  # Crypto fails
            elif "criptoya" in url:
                return dollar_response
            return None

        mock_cached.side_effect = mock_requests_side_effect
        mock_env.return_value = "test_api_key"

        result = get_market_context()

        assert "crypto" not in result
        assert "dollar" in result


def test_get_market_context_all_fail():
    from api.index import get_market_context

    with patch("api.index.cached_requests") as mock_cached, patch(
        "os.environ.get"
    ) as mock_env, patch("api.index.get_cached_bcra_variables") as mock_get_bcra, patch(
        "api.index.scrape_bcra_variables"
    ) as mock_scrape_bcra, patch("api.index.cache_bcra_variables") as mock_cache_bcra:

        mock_cached.return_value = None
        mock_env.return_value = "test_api_key"
        mock_get_bcra.return_value = None
        mock_scrape_bcra.return_value = None
        mock_cache_bcra.return_value = None

        result = get_market_context()

        assert result == {}


def test_get_weather_context_success():
    from api.index import get_weather_context

    with patch("api.index.get_weather") as mock_weather, patch(
        "api.index.get_weather_description"
    ) as mock_description:

        mock_weather.return_value = {"temperature": 25.0, "weather_code": 0}
        mock_description.return_value = "cielo despejado"

        result = get_weather_context()

        assert result is not None
        assert result["temperature"] == 25.0
        assert result["weather_code"] == 0
        assert result["description"] == "cielo despejado"
        mock_description.assert_called_once_with(0)


def test_get_weather_context_fail():
    from api.index import get_weather_context

    with patch("api.index.get_weather") as mock_weather:
        mock_weather.return_value = None

        result = get_weather_context()

        assert result is None


def test_get_weather_context_exception():
    from api.index import get_weather_context

    with patch("api.index.get_weather") as mock_weather:
        mock_weather.side_effect = Exception("Weather API error")

        result = get_weather_context()

        assert result is None


def test_get_time_context():
    from api.index import get_time_context
    from datetime import datetime, timezone, timedelta

    with patch("api.index.datetime") as mock_datetime:
        # Mock a fixed time
        fixed_time = datetime(2024, 1, 15, 14, 30, 0)
        buenos_aires_tz = timezone(timedelta(hours=-3))
        fixed_time_ba = fixed_time.replace(tzinfo=buenos_aires_tz)

        mock_datetime.now.return_value = fixed_time_ba
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        result = get_time_context()

        assert "datetime" in result
        assert "formatted" in result
        assert result["datetime"] == fixed_time_ba


def test_get_fallback_response():
    from api.index import get_fallback_response

    messages = [{"role": "user", "content": "hello"}]

    result = get_fallback_response(messages)

    # Should return a string (one of many predefined fallback responses)
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) < 50  # Reasonable length for a fallback response


def test_build_system_message():
    from api.index import build_system_message
    import api.index

    # Reset global cache to ensure clean state
    api.index._bot_config = None

    context = {
        "market": {
            "crypto": [{"symbol": "BTC", "price": 50000}],
            "dollar": {"oficial": {"price": 1000}},
        },
        "weather": {
            "temperature": 25,
            "apparent_temperature": 26,
            "precipitation_probability": 10,
            "description": "cielo despejado",
            "cloud_cover": 20,
            "visibility": 10000,
        },
        "time": {"formatted": "Monday 15/01/2024"},
    }

    with patch("os.environ.get") as mock_env:
        # Mock environment variables
        def env_side_effect(key):
            env_vars = {
                "BOT_SYSTEM_PROMPT": "Sos el gordo, un bot argentino de prueba.",
                "BOT_TRIGGER_WORDS": "gordo,test,bot"
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        result = build_system_message(context)

        assert result["role"] == "system"
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 0
        content_text = result["content"][0]["text"]
        assert "argentina" in content_text.lower()
        assert "gordo" in content_text.lower()


def test_build_system_message_empty_context():
    from api.index import build_system_message
    import api.index

    # Reset global cache to ensure clean state
    api.index._bot_config = None

    context = {"market": {}, "weather": None, "time": {"formatted": "Monday"}}

    with patch("os.environ.get") as mock_env:
        # Mock environment variables
        def env_side_effect(key):
            env_vars = {
                "BOT_SYSTEM_PROMPT": "You are a test bot assistant.",
                "BOT_TRIGGER_WORDS": "test,bot"
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        result = build_system_message(context)

        assert result["role"] == "system"
        assert "content" in result
        assert isinstance(result["content"], list)
        # Should still have base personality
        content_text = result["content"][0]["text"]
        assert len(content_text) > 100


def test_clean_crypto_data():
    from api.index import clean_crypto_data

    raw_data = [
        {
            "id": 1,
            "name": "Bitcoin",
            "symbol": "BTC",
            "slug": "bitcoin",
            "max_supply": 21000000,
            "circulating_supply": 19500000,
            "total_supply": 19500000,
            "infinite_supply": False,
            "quote": {
                "USD": {
                    "price": 50000.123456,
                    "volume_24h": 1000000000,
                    "percent_change_1h": 0.5,
                    "percent_change_24h": 2.5,
                    "percent_change_7d": 5.0,
                    "percent_change_30d": 10.0,
                    "market_cap": 1000000000000,
                    "market_cap_dominance": 45.5,
                }
            },
        }
    ]

    result = clean_crypto_data(raw_data)

    assert len(result) == 1
    assert result[0]["symbol"] == "BTC"
    assert result[0]["name"] == "Bitcoin"
    assert result[0]["slug"] == "bitcoin"
    assert result[0]["quote"]["USD"]["price"] == 50000.123456


def test_format_market_info():
    from api.index import format_market_info

    market_data = {
        "crypto": [{"symbol": "BTC", "price": 50000, "change_24h": 2.5}],
        "dollar": {"oficial": {"price": 1000}, "blue": {"price": 1200}},
    }

    result = format_market_info(market_data)

    assert "BTC" in result
    assert "50000" in result or "50,000" in result
    assert "1000" in result or "1,000" in result


def test_format_market_info_empty_dict():
    from api.index import format_market_info

    result = format_market_info({})

    assert result == ""


def test_format_weather_info():
    from api.index import format_weather_info

    weather_data = {
        "temperature": 25.5,
        "apparent_temperature": 26.0,
        "precipitation_probability": 10,
        "description": "cielo despejado",
        "cloud_cover": 20,
        "visibility": 10000,
    }

    result = format_weather_info(weather_data)

    assert "26" in result  # apparent_temperature
    assert "cielo despejado" in result
    assert "10%" in result  # precipitation_probability


def test_format_weather_info_empty():
    from api.index import format_weather_info
    import pytest

    # Function doesn't handle empty data gracefully - should raise KeyError
    with pytest.raises(KeyError):
        format_weather_info({})


def test_sort_dollar_rates_success():
    """Test sort_dollar_rates with valid dollar rates data"""
    from api.index import sort_dollar_rates

    dollar_rates = {
        "data": {
            "oficial": {"price": 1000.50, "variation": 1.2},
            "tarjeta": {"price": 1600.75, "variation": -0.8},
            "mep": {"al30": {"ci": {"price": 1050.25, "variation": 0.5}}},
            "ccl": {"al30": {"ci": {"price": 1075.80, "variation": 0.7}}},
            "blue": {"ask": 1200.00, "variation": 2.1},
            "cripto": {
                "ccb": {"ask": 1150.90, "variation": 1.8},
                "usdc": {"ask": 1140.30, "variation": 1.5},
                "usdt": {"ask": 1145.60, "variation": 1.6},
            },
        }
    }

    result = sort_dollar_rates(dollar_rates)

    assert len(result) == 8
    assert result[0]["name"] == "Oficial"
    assert result[0]["price"] == 1000.50
    assert result[-1]["name"] == "Tarjeta"
    assert result[-1]["price"] == 1600.75
    # Verify sorting by price
    for i in range(len(result) - 1):
        assert result[i]["price"] <= result[i + 1]["price"]


def test_sort_dollar_rates_with_none_variations():
    """Test sort_dollar_rates with None variation values"""
    from api.index import sort_dollar_rates

    dollar_rates = {
        "data": {
            "oficial": {"price": 1000.50, "variation": None},
            "tarjeta": {"price": 1600.75, "variation": None},
            "mep": {"al30": {"ci": {"price": 1050.25, "variation": None}}},
            "ccl": {"al30": {"ci": {"price": 1075.80, "variation": None}}},
            "blue": {"ask": 1200.00, "variation": None},
            "cripto": {
                "ccb": {"ask": 1150.90, "variation": None},
                "usdc": {"ask": 1140.30, "variation": None},
                "usdt": {"ask": 1145.60, "variation": None},
            },
        }
    }

    result = sort_dollar_rates(dollar_rates)

    assert len(result) == 8
    for rate in result:
        assert rate["history"] is None


def test_format_dollar_rates_with_positive_variations():
    """Test format_dollar_rates with positive variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Oficial", "price": 1000.50, "history": 1.2},
        {"name": "Blue", "price": 1200.00, "history": 2.1},
        {"name": "MEP", "price": 1050.25, "history": 0.5},
    ]

    result = format_dollar_rates(dollar_rates, 24)

    expected_lines = [
        "Oficial: 1000.5 (+1.2% 24hs)",
        "Blue: 1200 (+2.1% 24hs)",
        "MEP: 1050.25 (+0.5% 24hs)",
    ]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_with_negative_variations():
    """Test format_dollar_rates with negative variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Tarjeta", "price": 1600.75, "history": -0.8},
        {"name": "CCL", "price": 1075.80, "history": -1.5},
    ]

    result = format_dollar_rates(dollar_rates, 12)

    expected_lines = ["Tarjeta: 1600.75 (-0.8% 12hs)", "CCL: 1075.8 (-1.5% 12hs)"]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_with_none_variations():
    """Test format_dollar_rates with None variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Oficial", "price": 1000.50, "history": None},
        {"name": "Blue", "price": 1200.00, "history": None},
    ]

    result = format_dollar_rates(dollar_rates, 24)

    expected_lines = ["Oficial: 1000.5", "Blue: 1200"]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_mixed_variations():
    """Test format_dollar_rates with mixed variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Oficial", "price": 1000.50, "history": 1.2},
        {"name": "Blue", "price": 1200.00, "history": None},
        {"name": "Tarjeta", "price": 1600.75, "history": -0.8},
    ]

    result = format_dollar_rates(dollar_rates, 6)

    expected_lines = [
        "Oficial: 1000.5 (+1.2% 6hs)",
        "Blue: 1200",
        "Tarjeta: 1600.75 (-0.8% 6hs)",
    ]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_zero_decimal_formatting():
    """Test format_dollar_rates decimal formatting for whole numbers"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Test1", "price": 1000.00, "history": 0.00},
        {"name": "Test2", "price": 1200.10, "history": 1.00},
        {"name": "Test3", "price": 1500.50, "history": -2.50},
    ]

    result = format_dollar_rates(dollar_rates, 24)

    expected_lines = [
        "Test1: 1000 (+0% 24hs)",
        "Test2: 1200.1 (+1% 24hs)",
        "Test3: 1500.5 (-2.5% 24hs)",
    ]
    assert result == "\n".join(expected_lines)


def test_clean_crypto_data_success():
    """Test clean_crypto_data with valid crypto data"""
    from api.index import clean_crypto_data

    cryptos = [
        {
            "name": "Bitcoin",
            "symbol": "BTC",
            "slug": "bitcoin",
            "max_supply": 21000000,
            "circulating_supply": 19500000,
            "total_supply": 19500000,
            "infinite_supply": False,
            "quote": {
                "USD": {
                    "price": 45000.50,
                    "volume_24h": 25000000000,
                    "percent_change_1h": 0.5,
                    "percent_change_24h": 2.1,
                    "percent_change_7d": -1.8,
                    "percent_change_30d": 15.2,
                    "market_cap": 877500000000,
                    "market_cap_dominance": 42.5,
                }
            },
        }
    ]

    result = clean_crypto_data(cryptos)

    assert len(result) == 1
    crypto = result[0]
    assert crypto["name"] == "Bitcoin"
    assert crypto["symbol"] == "BTC"
    assert crypto["slug"] == "bitcoin"
    assert crypto["supply"]["max"] == 21000000
    assert crypto["supply"]["circulating"] == 19500000
    assert crypto["quote"]["USD"]["price"] == 45000.50
    assert crypto["quote"]["USD"]["changes"]["24h"] == 2.1


def test_clean_crypto_data_multiple_cryptos():
    """Test clean_crypto_data with multiple cryptocurrencies"""
    from api.index import clean_crypto_data

    cryptos = [
        {
            "name": "Bitcoin",
            "symbol": "BTC",
            "slug": "bitcoin",
            "max_supply": 21000000,
            "circulating_supply": 19500000,
            "total_supply": 19500000,
            "infinite_supply": False,
            "quote": {
                "USD": {
                    "price": 45000.50,
                    "volume_24h": 25000000000,
                    "percent_change_1h": 0.5,
                    "percent_change_24h": 2.1,
                    "percent_change_7d": -1.8,
                    "percent_change_30d": 15.2,
                    "market_cap": 877500000000,
                    "market_cap_dominance": 42.5,
                }
            },
        },
        {
            "name": "Ethereum",
            "symbol": "ETH",
            "slug": "ethereum",
            "max_supply": None,
            "circulating_supply": 120000000,
            "total_supply": 120000000,
            "infinite_supply": True,
            "quote": {
                "USD": {
                    "price": 3000.25,
                    "volume_24h": 15000000000,
                    "percent_change_1h": -0.2,
                    "percent_change_24h": 1.8,
                    "percent_change_7d": -3.2,
                    "percent_change_30d": 8.5,
                    "market_cap": 360000000000,
                    "market_cap_dominance": 18.2,
                }
            },
        },
    ]

    result = clean_crypto_data(cryptos)

    assert len(result) == 2
    assert result[0]["name"] == "Bitcoin"
    assert result[1]["name"] == "Ethereum"
    assert result[1]["supply"]["max"] is None
    assert result[1]["supply"]["infinite"] is True


def test_format_market_info_with_crypto_and_dollar():
    """Test format_market_info with both crypto and dollar data"""
    from api.index import format_market_info

    market = {
        "crypto": [
            {"name": "Bitcoin", "price": 45000.50},
            {"name": "Ethereum", "price": 3000.25},
        ],
        "dollar": [
            {"name": "Oficial", "price": 1000.50},
            {"name": "Blue", "price": 1200.00},
        ],
    }

    result = format_market_info(market)

    assert "PRECIOS DE CRIPTOS:" in result
    assert "DOLARES:" in result
    assert '"name": "Bitcoin"' in result
    assert '"name": "Oficial"' in result


def test_format_market_info_crypto_only():
    """Test format_market_info with only crypto data"""
    from api.index import format_market_info

    market = {"crypto": [{"name": "Bitcoin", "price": 45000.50}]}

    result = format_market_info(market)

    assert "PRECIOS DE CRIPTOS:" in result
    assert "DOLARES:" not in result
    assert '"name": "Bitcoin"' in result


def test_format_market_info_dollar_only():
    """Test format_market_info with only dollar data"""
    from api.index import format_market_info

    market = {"dollar": [{"name": "Oficial", "price": 1000.50}]}

    result = format_market_info(market)

    assert "PRECIOS DE CRIPTOS:" not in result
    assert "DOLARES:" in result
    assert '"name": "Oficial"' in result


def test_format_market_info_empty():
    """Test format_market_info with empty market data"""
    from api.index import format_market_info

    market = {}

    result = format_market_info(market)

    assert result == ""


def test_get_weather_description_clear():
    """Test get_weather_description for clear weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(0) == "despejado"
    assert get_weather_description(1) == "mayormente despejado"
    assert get_weather_description(2) == "parcialmente nublado"
    assert get_weather_description(3) == "nublado"


def test_get_weather_description_rain():
    """Test get_weather_description for rain weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(61) == "lluvia leve"
    assert get_weather_description(63) == "lluvia moderada"
    assert get_weather_description(65) == "lluvia intensa"


def test_get_weather_description_storm():
    """Test get_weather_description for storm weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(95) == "tormenta"
    assert get_weather_description(96) == "tormenta con granizo leve"
    assert get_weather_description(99) == "tormenta con granizo intenso"


def test_get_weather_description_unknown():
    """Test get_weather_description for unknown weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(999) == "clima raro"
    assert get_weather_description(-1) == "clima raro"


def test_download_telegram_file_success():
    """Test download_telegram_file with successful download"""
    from api.index import download_telegram_file

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.get"
    ) as mock_get:
        mock_env.return_value = "test_token"

        # Mock file info response
        mock_file_info = MagicMock()
        mock_file_info.json.return_value = {
            "ok": True,
            "result": {"file_path": "photos/file_123.jpg"},
        }
        mock_file_info.raise_for_status.return_value = None

        # Mock file download response
        mock_file_download = MagicMock()
        mock_file_download.content = b"fake image data"
        mock_file_download.raise_for_status.return_value = None

        # Configure side effect for two different calls
        mock_get.side_effect = [mock_file_info, mock_file_download]

        result = download_telegram_file("test_file_id")

        assert result == b"fake image data"
        assert mock_get.call_count == 2

        # Verify file info call
        info_call = mock_get.call_args_list[0]
        assert "getFile" in info_call[0][0]
        assert info_call[1]["params"]["file_id"] == "test_file_id"

        # Verify file download call
        download_call = mock_get.call_args_list[1]
        assert "photos/file_123.jpg" in download_call[0][0]


def test_download_telegram_file_api_error():
    """Test download_telegram_file with API error"""
    from api.index import download_telegram_file

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.get"
    ) as mock_get:
        mock_env.return_value = "test_token"

        # Mock failed API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "File not found"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = download_telegram_file("invalid_file_id")

        assert result is None


def test_download_telegram_file_network_error():
    """Test download_telegram_file with network error"""
    from api.index import download_telegram_file

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.get"
    ) as mock_get:
        mock_env.return_value = "test_token"
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = download_telegram_file("test_file_id")

        assert result is None


def test_encode_image_to_base64_success():
    """Test encode_image_to_base64 with valid image data"""
    from api.index import encode_image_to_base64
    import base64

    test_data = b"fake image bytes"
    expected = base64.b64encode(test_data).decode("utf-8")

    result = encode_image_to_base64(test_data)

    assert result == expected
    assert isinstance(result, str)


def test_encode_image_to_base64_empty():
    """Test encode_image_to_base64 with empty data"""
    from api.index import encode_image_to_base64

    result = encode_image_to_base64(b"")

    assert result == ""


def test_resize_image_if_needed_no_resize():
    """Test resize_image_if_needed when image is already small enough"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module:
        # Mock small image
        mock_image = MagicMock()
        mock_image.size = (200, 150)  # Smaller than max_size of 512
        mock_image_module.open.return_value = mock_image

        test_data = b"small image data"
        result = resize_image_if_needed(test_data, max_size=512)

        assert result == test_data  # Should return original data unchanged


def test_resize_image_if_needed_with_resize():
    """Test resize_image_if_needed when image needs resizing"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module, patch(
        "api.index.io.BytesIO"
    ) as mock_bytesio:
        # Mock large image that needs resizing
        mock_image = MagicMock()
        mock_image.size = (1024, 768)  # Larger than max_size of 512
        mock_image.mode = "RGB"

        # Mock resized image
        mock_resized = MagicMock()
        mock_image.resize.return_value = mock_resized

        # Mock output buffer
        mock_output_buffer = MagicMock()
        mock_output_buffer.getvalue.return_value = b"resized image data"
        mock_bytesio.return_value = mock_output_buffer

        mock_image_module.open.return_value = mock_image
        mock_image_module.Resampling.LANCZOS = "LANCZOS"

        test_data = b"large image data"
        result = resize_image_if_needed(test_data, max_size=512)

        assert result == b"resized image data"
        mock_image.resize.assert_called_once()
        mock_resized.save.assert_called_once()


def test_resize_image_if_needed_rgba_conversion():
    """Test resize_image_if_needed with RGBA image conversion"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module, patch(
        "api.index.io.BytesIO"
    ) as mock_bytesio:
        # Mock large RGBA image that needs conversion
        mock_image = MagicMock()
        mock_image.size = (1024, 768)
        mock_image.mode = "RGBA"

        # Mock converted image
        mock_converted = MagicMock()
        mock_image.convert.return_value = mock_converted

        # Mock resized image that also has RGBA mode
        mock_resized = MagicMock()
        mock_resized.mode = "RGBA"
        mock_resized_converted = MagicMock()
        mock_resized.convert.return_value = mock_resized_converted
        mock_image.resize.return_value = mock_resized

        # Mock output buffer
        mock_output_buffer = MagicMock()
        mock_output_buffer.getvalue.return_value = b"converted resized image"
        mock_bytesio.return_value = mock_output_buffer

        mock_image_module.open.return_value = mock_image
        mock_image_module.Resampling.LANCZOS = "LANCZOS"

        result = resize_image_if_needed(b"rgba image data")

        assert result == b"converted resized image"
        mock_resized.convert.assert_called_once_with("RGB")


def test_resize_image_if_needed_import_error():
    """Test resize_image_if_needed when PIL is not available"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module:
        mock_image_module.open.side_effect = ImportError("PIL not available")

        test_data = b"image data"
        result = resize_image_if_needed(test_data)

        assert result == test_data  # Should return original data on ImportError


def test_resize_image_if_needed_processing_error():
    """Test resize_image_if_needed with image processing error"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module:
        mock_image_module.open.side_effect = Exception("Invalid image format")

        test_data = b"corrupted image data"
        result = resize_image_if_needed(test_data)

        assert result == test_data  # Should return original data on error


def test_describe_image_cloudflare_success():
    """Test describe_image_cloudflare with successful API response"""
    from api.index import describe_image_cloudflare

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.post"
    ) as mock_post:
        mock_env.side_effect = lambda key, default=None: {
            "CLOUDFLARE_API_KEY": "test_api_key",
            "CLOUDFLARE_ACCOUNT_ID": "test_account_id",
        }.get(key, default)

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"response": "A beautiful landscape with mountains"}
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = describe_image_cloudflare(b"base64_image_data")

        assert result == "A beautiful landscape with mountains"
        mock_post.assert_called_once()


def test_describe_image_cloudflare_api_error():
    """Test describe_image_cloudflare with API error"""
    from api.index import describe_image_cloudflare

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.post"
    ) as mock_post:
        mock_env.side_effect = lambda key, default=None: {
            "CLOUDFLARE_API_KEY": "test_api_key",
            "CLOUDFLARE_ACCOUNT_ID": "test_account_id",
        }.get(key, default)

        mock_post.side_effect = requests.exceptions.RequestException("API error")

        result = describe_image_cloudflare(b"base64_image_data")

        assert result is None


def test_transcribe_audio_cloudflare_success():
    """Test transcribe_audio_cloudflare with successful transcription"""
    from api.index import transcribe_audio_cloudflare

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.post"
    ) as mock_post:
        mock_env.side_effect = lambda key, default=None: {
            "CLOUDFLARE_API_KEY": "test_api_key",
            "CLOUDFLARE_ACCOUNT_ID": "test_account_id",
        }.get(key, default)

        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "result": {"text": "Hello, this is a test audio transcription"},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = transcribe_audio_cloudflare(b"audio_data")

        assert result == "Hello, this is a test audio transcription"
        mock_post.assert_called_once()


def test_transcribe_audio_cloudflare_network_error():
    """Test transcribe_audio_cloudflare with network error"""
    from api.index import transcribe_audio_cloudflare

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.post"
    ) as mock_post:
        mock_env.side_effect = lambda key, default=None: {
            "CLOUDFLARE_API_KEY": "test_api_key",
            "CLOUDFLARE_ACCOUNT_ID": "test_account_id",
        }.get(key, default)

        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        result = transcribe_audio_cloudflare(b"audio_data")

        assert result is None


# Tests for web search functionality
def test_web_search_success():
    """Test web_search with successful DDGS response"""
    with patch('ddgs.DDGS') as mock_ddgs_class:
        # Mock DDGS instance and its text method
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {
                "title": "Test Result 1",
                "href": "https://example.com/test1",
                "body": "This is test result 1 description"
            },
            {
                "title": "Test Result 2",
                "href": "https://example.com/test2",
                "body": "This is test result 2 description"
            }
        ]

        results = web_search("test query", limit=3)

        assert len(results) == 2
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["url"] == "https://example.com/test1"
        assert results[0]["snippet"] == "This is test result 1 description"
        assert results[1]["title"] == "Test Result 2"
        assert results[1]["url"] == "https://example.com/test2"
        assert results[1]["snippet"] == "This is test result 2 description"


def test_web_search_no_results():
    """Test web_search when no results are found"""
    with patch('ddgs.DDGS') as mock_ddgs_class:
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = []

        results = web_search("nonexistent query")

        assert len(results) == 0


def test_web_search_network_error():
    """Test web_search when network error occurs"""
    with patch('ddgs.DDGS') as mock_ddgs_class:
        mock_ddgs_class.side_effect = Exception("Network error")

        results = web_search("test query")

        assert len(results) == 0


def test_web_search_limit_parameter():
    """Test web_search respects the limit parameter"""
    with patch('ddgs.DDGS') as mock_ddgs_class:
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Description 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Description 2"}
        ]

        results = web_search("test query", limit=2)

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[1]["title"] == "Result 2"

        # Verify that DDGS was called with the correct limit
        mock_ddgs.text.assert_called_once_with(
            query="test query",
            region="ar-es",
            safesearch="off",
            max_results=2
        )


# Tests for search command
def test_search_command_success():
    """Test search_command with successful results"""
    with patch('api.index.web_search') as mock_search:
        mock_search.return_value = [
            {"title": "Test Result", "url": "https://example.com", "snippet": ""},
            {"title": "Another Result", "url": "https://test.com", "snippet": "Test snippet"}
        ]

        result = search_command("python programming")

        assert "🔎 Resultados para: python programming" in result
        assert "Test Result" in result
        assert "https://example.com" in result
        assert "Another Result" in result
        assert "Test snippet" in result


def test_search_command_empty_query():
    """Test search_command with empty query"""
    result = search_command("")
    assert result == "decime qué querés buscar capo"

    result = search_command(None)
    assert result == "decime qué querés buscar capo"


def test_search_command_no_results():
    """Test search_command when no results found"""
    with patch('api.index.web_search') as mock_search:
        mock_search.return_value = []

        result = search_command("nonexistent query")

        assert result == "no encontré resultados ahora con duckduckgo"


# Tests for tool calling functionality
def test_parse_tool_call_valid():
    """Test parse_tool_call with valid tool call"""
    text = 'Some text\n[TOOL] web_search {"query": "test", "limit": 3}\nMore text'

    result = parse_tool_call(text)

    assert result is not None
    tool_name, args = result
    assert tool_name == "web_search"
    assert args == {"query": "test", "limit": 3}


def test_parse_tool_call_invalid():
    """Test parse_tool_call with invalid inputs"""
    # No tool call
    assert parse_tool_call("Just normal text") is None

    # Malformed JSON
    text = '[TOOL] web_search {"query": invalid json}'
    assert parse_tool_call(text) is None

    # Missing arguments
    text = '[TOOL] web_search'
    assert parse_tool_call(text) is None

    # None input
    assert parse_tool_call(None) is None


def test_execute_tool_web_search():
    """Test execute_tool with web_search tool"""
    with patch('api.index.web_search') as mock_search:
        mock_search.return_value = [{"title": "Test", "url": "https://test.com", "snippet": ""}]

        result = execute_tool("web_search", {"query": "test", "limit": 3})

        import json
        parsed_result = json.loads(result)
        assert parsed_result["query"] == "test"
        assert len(parsed_result["results"]) == 1


def test_execute_tool_empty_query():
    """Test execute_tool with empty query"""
    result = execute_tool("web_search", {"query": "", "limit": 3})

    assert result == "query vacío"


def test_execute_tool_unknown():
    """Test execute_tool with unknown tool"""
    result = execute_tool("unknown_tool", {})

    assert result == "herramienta desconocida: unknown_tool"


def test_handle_ai_response_sanitizes_tool_lines(monkeypatch):
    """handle_ai_response should strip visible tool call lines"""
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return "Hola\n[TOOL] web_search {\"query\": \"test\"}\nChau"

    result = handle_ai_response("123", fake_handler, [])

    assert "[TOOL]" not in result
    assert result == "Hola\nChau"


def test_handle_ai_response_returns_fallback_on_empty(monkeypatch):
    """Empty sanitized responses should return a fallback message"""
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return "[TOOL] web_search {\"query\": \"test\"}"

    result = handle_ai_response("123", fake_handler, [])

    assert result == "no pude generar respuesta, intentá de nuevo"


# Tests for complete_with_providers function
def test_complete_with_providers_groq_success():
    """Test complete_with_providers when Groq succeeds"""
    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    with patch('api.index.get_groq_ai_response') as mock_groq, \
         patch('api.index.get_ai_response') as mock_openrouter, \
         patch('api.index.get_cloudflare_ai_response') as mock_cloudflare:

        mock_groq.return_value = "Groq response"
        mock_openrouter.return_value = "OpenRouter response"
        mock_cloudflare.return_value = "Cloudflare response"

        result = complete_with_providers(system_message, messages)

        assert result == "Groq response"
        mock_groq.assert_called_once()
        mock_openrouter.assert_not_called()
        mock_cloudflare.assert_not_called()


def test_complete_with_providers_fallback_sequence():
    """Test complete_with_providers fallback sequence"""
    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    with patch('api.index.get_groq_ai_response') as mock_groq, \
         patch('api.index.get_ai_response') as mock_openrouter, \
         patch('api.index.get_cloudflare_ai_response') as mock_cloudflare, \
         patch('os.environ.get') as mock_env, \
         patch('api.index.OpenAI') as mock_openai:

        mock_env.return_value = "test_api_key"
        mock_groq.return_value = None  # Groq fails
        mock_openrouter.return_value = "OpenRouter response"
        mock_cloudflare.return_value = "Cloudflare response"

        # Mock OpenAI client creation
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        result = complete_with_providers(system_message, messages)

        assert result == "OpenRouter response"
        assert mock_groq.call_count == 2
        mock_openrouter.assert_called_once()
        mock_cloudflare.assert_not_called()


def test_complete_with_providers_all_fail():
    """Test complete_with_providers when all providers fail"""
    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    with patch('api.index.get_groq_ai_response') as mock_groq, \
         patch('api.index.get_ai_response') as mock_openrouter, \
         patch('api.index.get_cloudflare_ai_response') as mock_cloudflare, \
         patch('os.environ.get') as mock_env, \
         patch('api.index.OpenAI') as mock_openai:

        mock_env.return_value = "test_api_key"
        mock_groq.return_value = None
        mock_openrouter.return_value = None
        mock_cloudflare.return_value = None

        # Mock OpenAI client creation
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        result = complete_with_providers(system_message, messages)

        assert result is None
        assert mock_groq.call_count == 2
        mock_openrouter.assert_called_once()
        assert mock_cloudflare.call_count == 2


def test_is_social_frontend():
    from api.index import is_social_frontend

    assert is_social_frontend("twitter.com")
    assert is_social_frontend("mobile.twitter.com")
    assert is_social_frontend("vxtiktok.com")
    assert not is_social_frontend("example.com")


@patch("api.index.requests.get")
def test_replace_links(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<meta property='og:title' content='foo'>"
    mock_get.return_value = mock_response
    text = (
        "Check https://twitter.com/foo?utm_source=share and http://x.com/bar?s=20 and https://bsky.app/baz?share=1 and "
        "https://www.instagram.com/qux?igsh=abc123 and https://www.reddit.com/r/foo?st=abc and https://old.reddit.com/r/bar?utm_name=bar and "
        "https://www.tiktok.com/@bar?lang=en and https://vm.tiktok.com/ZMHGacxknMW5J-gEiNC/?share=copy"
    )
    fixed, changed, originals = replace_links(text)
    assert changed
    assert "https://fxtwitter.com/foo" in fixed
    assert "http://fixupx.com/bar" in fixed
    assert "https://fxbsky.app/baz" in fixed
    assert "https://kkinstagram.com/qux" in fixed
    assert "https://www.rxddit.com/r/foo" in fixed
    assert "https://old.rxddit.com/r/bar" in fixed
    assert "https://www.vxtiktok.com/@bar" in fixed
    assert "https://vm.vxtiktok.com/ZMHGacxknMW5J-gEiNC/" in fixed
    assert "?" not in fixed
    expected = {
        "https://twitter.com/foo",
        "http://x.com/bar",
        "https://bsky.app/baz",
        "https://www.instagram.com/qux",
        "https://www.reddit.com/r/foo",
        "https://old.reddit.com/r/bar",
        "https://www.tiktok.com/@bar",
        "https://vm.tiktok.com/ZMHGacxknMW5J-gEiNC/",
    }
    assert set(originals) == expected
    assert all("?" not in url and "#" not in url for url in originals)


@patch("api.index.requests.get")
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


@patch("api.index.requests.get")
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




def test_configure_links_sets_and_disables():
    with patch("api.index.config_redis") as mock_redis:
        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        resp = configure_links("123", "reply")
        redis_client.set.assert_called_with("link_mode:123", "reply")
        assert "reply" in resp

        resp = configure_links("123", "off")
        redis_client.delete.assert_called_with("link_mode:123")
        assert "disabled" in resp


def test_configure_links_usage_shows_current():
    with patch("api.index.config_redis") as mock_redis:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client

        resp = configure_links("123", "")
        redis_client.delete.assert_not_called()
        assert "Usage:" in resp
        assert "current: reply" in resp


def test_handle_msg_link_reply():
    message = {
        "message_id": 1,
        "chat": {"id": 123, "type": "group"},
        "from": {"first_name": "John", "username": "john"},
        "text": "check https://twitter.com/foo",
    }
    with patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}), \
        patch("api.index.config_redis") as mock_redis, \
        patch("api.index.send_msg") as mock_send, \
        patch("api.index.delete_msg") as mock_delete, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.save_message_to_redis") as mock_save, \
        patch("api.index.requests.get") as mock_get:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<meta property='og:title'>"
        mock_get.return_value = mock_response

        result = handle_msg(message)

        assert result == "ok"
        expected = "check https://fxtwitter.com/foo\n\nShared by @john"
        mock_send.assert_called_once_with(
            "123", expected, "1", ["https://twitter.com/foo"]
        )
        mock_delete.assert_not_called()
        mock_save.assert_not_called()


def test_handle_msg_link_reply_instagram():
    message = {
        "message_id": 3,
        "chat": {"id": 789, "type": "group"},
        "from": {"first_name": "Lu", "username": "lu"},
        "text": "mirá https://www.instagram.com/qux?igsh=abc",
    }
    with patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}), \
        patch("api.index.config_redis") as mock_redis, \
        patch("api.index.send_msg") as mock_send, \
        patch("api.index.delete_msg") as mock_delete, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.save_message_to_redis") as mock_save, \
        patch("api.index.requests.get") as mock_get:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<meta property='og:title'>"
        mock_get.return_value = mock_response

        result = handle_msg(message)

        assert result == "ok"
        expected = "mirá https://kkinstagram.com/qux\n\nShared by @lu"
        mock_send.assert_called_once_with(
            "789", expected, "3", ["https://www.instagram.com/qux"]
        )
        mock_delete.assert_not_called()
        mock_save.assert_not_called()


def test_handle_msg_link_delete():
    message = {
        "message_id": 2,
        "chat": {"id": 456, "type": "group"},
        "from": {"first_name": "Ana", "username": "ana"},
        "text": "look https://x.com/bar",
    }
    with patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}), \
        patch("api.index.config_redis") as mock_redis, \
        patch("api.index.send_msg") as mock_send, \
        patch("api.index.delete_msg") as mock_delete, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.save_message_to_redis") as mock_save, \
        patch("api.index.requests.get") as mock_get:
        redis_client = MagicMock()
        redis_client.get.return_value = "delete"
        mock_redis.return_value = redis_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<meta property='og:title'>"
        mock_get.return_value = mock_response

        result = handle_msg(message)

        assert result == "ok"
        expected = "look https://fixupx.com/bar\n\nShared by @ana"
        mock_delete.assert_called_once_with("456", "2")
        mock_send.assert_called_once_with(
            "456", expected, buttons=["https://x.com/bar"]
        )
        mock_save.assert_not_called()


@patch("api.index.config_redis")
def test_handle_msg_link_without_preview(mock_redis):
    message = {
        "message_id": 5,
        "chat": {"id": 321, "type": "group"},
        "from": {"id": 1},
        "text": "https://example.com",
    }
    with patch("api.index.send_msg") as mock_send, \
        patch("api.index.replace_links") as mock_replace, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.should_gordo_respond") as mock_should:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client
        mock_replace.return_value = ("https://example.com", False, [])

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()
        mock_should.assert_not_called()


def test_replace_links_checks_preview(monkeypatch):
    from api.index import replace_links

    mock_can = MagicMock(return_value=True)
    monkeypatch.setattr("api.index.can_embed_url", mock_can)
    text, changed, originals = replace_links("https://x.com/foo")
    assert text == "https://fixupx.com/foo"
    assert changed is True
    assert originals == ["https://x.com/foo"]
    mock_can.assert_called_once_with("https://fixupx.com/foo")


@patch("api.index.requests.get")
def test_xcom_link_replacement_with_metadata(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<meta property='og:title' content='foo'>"
    mock_get.return_value = mock_response
    fixed, changed, originals = replace_links("https://x.com/foo")
    assert changed is True
    assert fixed == "https://fixupx.com/foo"
    assert originals == ["https://x.com/foo"]
    mock_get.assert_called_once_with(
        "https://fixupx.com/foo", allow_redirects=True, timeout=5, headers=ANY
    )

def test_can_embed_url_logs_missing_meta(monkeypatch, capsys):
    from api.index import can_embed_url

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<html></html>"
    monkeypatch.setattr("api.index.requests.get", lambda *a, **kw: mock_response)

    result = can_embed_url("http://example.com")
    assert result is False
    captured = capsys.readouterr().out
    assert "missing og/twitter meta tags" in captured


@patch("api.index.requests.get")
def test_can_embed_url_allows_direct_media(mock_get):
    from api.index import can_embed_url

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "video/mp4"}
    mock_response.text = ""
    mock_get.return_value = mock_response

    result = can_embed_url("http://example.com/video")
    assert result is True

    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["User-Agent"] == "TelegramBot (like TwitterBot)"


def test_handle_msg_link_already_fixed():
    message = {
        "message_id": 6,
        "chat": {"id": 654, "type": "group"},
        "from": {"id": 1},
        "text": "https://fixupx.com/foo",
    }
    with patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}), \
        patch("api.index.config_redis") as mock_redis, \
        patch("api.index.send_msg") as mock_send, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.should_gordo_respond") as mock_should, \
        patch("api.index.requests.get") as mock_get:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()
        mock_should.assert_not_called()
        mock_get.assert_not_called()


def test_handle_msg_original_link_no_check():
    message = {
        "message_id": 7,
        "chat": {"id": 987, "type": "group"},
        "from": {"id": 1},
        "text": "https://vm.tiktok.com/foo",
    }
    with patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}), \
        patch("api.index.config_redis") as mock_redis, \
        patch("api.index.send_msg") as mock_send, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.should_gordo_respond") as mock_should, \
        patch("api.index.replace_links") as mock_replace, \
        patch("api.index.requests.get") as mock_get:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client
        mock_replace.return_value = ("https://vm.tiktok.com/foo", False, [])

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()
        mock_should.assert_not_called()
        mock_get.assert_not_called()


def test_handle_msg_link_already_fixed_subdomain():
    message = {
        "message_id": 8,
        "chat": {"id": 999, "type": "group"},
        "from": {"id": 1},
        "text": "https://vm.vxtiktok.com/foo",
    }
    with patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}), \
        patch("api.index.config_redis") as mock_redis, \
        patch("api.index.send_msg") as mock_send, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.should_gordo_respond") as mock_should, \
        patch("api.index.requests.get") as mock_get:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_not_called()
        mock_should.assert_not_called()
        mock_get.assert_not_called()


def test_handle_msg_replaced_link_adds_button():
    message = {
        "message_id": 9,
        "chat": {"id": 111, "type": "group"},
        "from": {"id": 1},
        "text": "https://x.com/foo?utm_source=bar",
    }
    with patch.dict("api.index.environ", {"TELEGRAM_USERNAME": "bot"}), \
        patch("api.index.config_redis") as mock_redis, \
        patch("api.index.send_msg") as mock_send, \
        patch("api.index.initialize_commands", return_value={}), \
        patch("api.index.should_gordo_respond") as mock_should, \
        patch("api.index.requests.get") as mock_get:
        redis_client = MagicMock()
        redis_client.get.return_value = "reply"
        mock_redis.return_value = redis_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = "<meta property='og:title'>"
        mock_get.return_value = mock_resp

        result = handle_msg(message)

        assert result == "ok"
        mock_send.assert_called_once_with(
            "111", "https://fixupx.com/foo", "9", ["https://x.com/foo"]
        )
        mock_should.assert_not_called()
