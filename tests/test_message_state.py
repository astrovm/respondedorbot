from tests.support import *


def test_format_user_message():
    # Test with username
    msg = {"from": {"first_name": "John", "username": "john123"}}
    assert format_user_message(msg, "hello") == "John (john123): hello"

    # Test without username
    msg = {"from": {"first_name": "John"}}
    assert format_user_message(msg, "hello") == "John: hello"


def test_format_user_message_with_reply_context():
    msg = {"from": {"first_name": "John", "username": "john123"}}
    context = "Jane (jane77): texto original"
    result = format_user_message(msg, "respuesta", context)
    assert "en respuesta a" in result
    assert "Jane (jane77): texto original" in result


def test_save_message_to_redis():
    from api.index import save_message_to_redis
    from api.message_state import CHAT_STATE_TTL

    with patch("redis.Redis") as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance

        # Mock sismember to return False (message doesn't exist)
        mock_instance.sismember.return_value = False
        pipeline = mock_instance.pipeline.return_value
        pipeline.execute.return_value = [None, None, None, True, True, []]

        # Test successful save
        chat_id = "123"
        message_id = "456"
        text = "test message"

        save_message_to_redis(chat_id, message_id, text, mock_instance)

        # Verify pipeline calls
        mock_instance.pipeline.assert_called_once()
        pipeline.lpush.assert_called_once()
        pipeline.ltrim.assert_called_once()
        pipeline.expire.assert_any_call(f"chat_history:{chat_id}", CHAT_STATE_TTL)
        pipeline.expire.assert_any_call(f"chat_message_ids:{chat_id}", CHAT_STATE_TTL)
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


def test_build_reply_context_text():
    reply_msg = {
        "from": {"first_name": "Jane", "username": "jane77"},
        "text": "mensaje original",
    }
    message = {"reply_to_message": reply_msg}

    context = build_reply_context_text(message)
    assert context == "Jane (jane77): mensaje original"

    media_reply = {
        "from": {"first_name": "Pablo"},
        "photo": [{"file_id": "abc"}],
    }
    assert (
        build_reply_context_text({"reply_to_message": media_reply})
        == "Pablo: una foto sin texto"
    )

    assert build_reply_context_text({}) is None


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


def test_truncate_text_edge_cases():
    # Test empty string
    assert truncate_text("") == ""

    # Test None input
    assert truncate_text(None) == ""

    # Test string with exactly max length
    text = "a" * 1024
    assert truncate_text(text) == text

    # Test string with max length minus one
    text = "a" * 1023
    assert truncate_text(text) == text

    # Test string with max length plus one
    text = "a" * 1025
    truncated = truncate_text(text)
    assert len(truncated) == 1024
    assert truncated.endswith("...")

    # Test with very small max_length


def test_save_and_get_chat_summary_round_trip():
    from api.message_state import get_chat_summary, save_chat_summary

    redis_client = MagicMock()
    store = {}

    redis_client.setex.side_effect = lambda key, ttl, value: store.__setitem__(key, value)
    redis_client.get.side_effect = lambda key: store.get(key)

    save_chat_summary(redis_client, "123", "summary text")

    assert get_chat_summary(redis_client, "123") == "summary text"


def test_save_and_get_chat_compacted_until_round_trip():
    from api.message_state import get_chat_compacted_until, save_chat_compacted_until

    redis_client = MagicMock()
    store = {}

    redis_client.setex.side_effect = lambda key, ttl, value: store.__setitem__(key, value)
    redis_client.get.side_effect = lambda key: store.get(key)

    save_chat_compacted_until(redis_client, "123", "msg_55")

    assert get_chat_compacted_until(redis_client, "123") == "msg_55"


def test_save_message_to_redis_writes_search_document():
    from api.message_state import save_message_to_redis

    redis_client = MagicMock()
    pipe = MagicMock()
    pipe.execute.return_value = [None, None, None, None, None, None, []]
    redis_client.pipeline.return_value = pipe
    redis_client.sismember.return_value = False
    redis_client.smembers.return_value = set()
    redis_client.execute_command.side_effect = Exception("Index already exists")

    save_message_to_redis(
        "chat1",
        "42",
        "astro: hola mundo",
        redis_client,
        admin_reporter=MagicMock(),
        role="user",
    )

    pipe.hset.assert_called()
    pipe.expire.assert_any_call("chatmsg:chat1:42", ANY)


def test_search_chat_history_prioritizes_reply_thread_matches():
    import api.message_state as message_state
    from api.message_state import search_chat_history

    message_state._SEARCH_INDEX_READY = False
    redis_client = MagicMock()
    redis_client.execute_command.side_effect = [
        Exception("Index already exists"),
        [
            2,
            "chatmsg:1:10",
            [
                "message_id",
                "10",
                "text",
                "wallet error happened",
                "reply_to_message_id",
                "99",
                "timestamp",
                "10",
            ],
            "chatmsg:1:11",
            [
                "message_id",
                "11",
                "text",
                "wallet error generic",
                "reply_to_message_id",
                "1",
                "timestamp",
                "11",
            ],
        ],
    ]

    results = search_chat_history(
        redis_client,
        chat_id="1",
        query_text="wallet error",
        reply_to_message_id="99",
        limit=5,
    )

    assert results[0]["reply_to_message_id"] == "99"
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
