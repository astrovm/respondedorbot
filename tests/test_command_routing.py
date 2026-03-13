from tests.support import *  # noqa: F401,F403

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

    # Test string with Japanese characters
    msg_text7 = "もうすぐです"
    expected7 = "/MOUSUGUDESU"
    assert convert_to_command(msg_text7) == expected7

    # Test string with halfwidth katakana
    msg_text8 = "ｶﾀｶﾅ"
    expected8 = "/KATAKANA"
    assert convert_to_command(msg_text8) == expected8


def test_should_use_groq_compound_tools_truthy(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    assert should_use_groq_compound_tools() is True

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_test_key")
    assert should_use_groq_compound_tools() is True

    monkeypatch.delenv("GROQ_FREE_API_KEY", raising=False)
    assert should_use_groq_compound_tools() is False

def test_optional_redis_client_success():
    from api.index import _optional_redis_client

    with patch("api.index.config_redis") as mock_config:
        sentinel = MagicMock()
        mock_config.return_value = sentinel

        result = _optional_redis_client(db=2)

    mock_config.assert_called_once_with(db=2)
    assert result is sentinel


def test_optional_redis_client_handles_failure():
    from api.index import _optional_redis_client

    with patch("api.index.config_redis") as mock_config:
        mock_config.side_effect = Exception("boom")

        result = _optional_redis_client()

    mock_config.assert_called_once()
    assert result is None


def test_hash_cache_key_is_stable():
    from api.index import _hash_cache_key

    payload_one = {"a": 1, "b": 2}
    payload_two = {"b": 2, "a": 1}

    key_one = _hash_cache_key("prefix", payload_one)
    key_two = _hash_cache_key("prefix", payload_two)
    other_key = _hash_cache_key("other", payload_one)

    assert key_one == key_two
    assert key_one.startswith("prefix:")
    assert key_one != other_key


def test_check_global_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")
    redis_client = MagicMock()
    redis_client.get.side_effect = [b"999", b"499999"]
    assert check_global_rate_limit(redis_client) is True

    redis_client = MagicMock()
    redis_client.get.side_effect = [b"1000", b"10"]
    assert check_global_rate_limit(redis_client) is False

    redis_client = MagicMock()
    redis_client.get.side_effect = [b"1", b"500000"]
    assert check_global_rate_limit(redis_client) is False


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


def test_parse_command_hangul_filler_alias():
    command, args = parse_command("/ㅤ pregunta", "@bot")
    assert command == "/ask"
    assert args == "pregunta"


def test_should_gordo_respond():
    import api.index

    # Reset global cache to ensure clean state
    config_module.reset_cache()

    commands = {"/test": (lambda x: x, False, False)}
    chat_config = {
        "link_mode": "off",
        "ai_random_replies": True,
        "ai_command_followups": True,
        "ignore_link_fix_followups": True,
    }

    with patch("os.environ.get") as mock_env:
        # Mock environment variables for both bot config and telegram username
        def env_side_effect(key):
            env_vars = {
                "TELEGRAM_USERNAME": "testbot",
                "BOT_SYSTEM_PROMPT": "You are a test bot",
                "BOT_TRIGGER_WORDS": "gordo,test,bot",
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        # Test command
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(commands, "/test", "hello", msg, chat_config, None)
            is True
        )

        # Test private chat
        msg = {"chat": {"type": "private"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(commands, "", "hello", msg, chat_config, None)
            is True
        )

        # Test mention
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(
                commands, "", "@testbot hello", msg, chat_config, None
            )
            is True
        )

        # Test reply to bot
        msg = {
            "chat": {"type": "group"},
            "from": {"username": "test"},
            "reply_to_message": {"from": {"username": "testbot"}},
        }
        assert (
            should_gordo_respond(commands, "", "hello", msg, chat_config, None)
            is True
        )

        # Test trigger word with mocked random
        with patch("random.random") as mock_random:
            mock_random.return_value = 0.05  # Below 0.1 threshold
            msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
            assert (
                should_gordo_respond(
                    commands, "", "hey gordo", msg, chat_config, None
                )
                is True
            )


def test_should_gordo_respond_ignores_link_fix_reply_when_toggle_enabled():
    import api.index

    config_module.reset_cache()
    commands = {"/test": (lambda x: x, False, False)}
    chat_config = {
        "link_mode": "off",
        "ai_random_replies": True,
        "ai_command_followups": True,
        "ignore_link_fix_followups": True,
    }

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

        assert (
            should_gordo_respond(
                commands, "", "hello", msg, chat_config, None
            )
            is False
        )


def test_should_gordo_respond_allows_link_fix_reply_when_toggle_disabled():
    import api.index

    config_module.reset_cache()
    commands = {"/test": (lambda x: x, False, False)}
    chat_config = {
        "link_mode": "off",
        "ai_random_replies": True,
        "ai_command_followups": True,
        "ignore_link_fix_followups": False,
    }

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
                "text": "https://fixupx.com/foo",
            },
        }

        assert (
            should_gordo_respond(
                commands, "", "investiga eso", msg, chat_config, None
            )
            is True
        )


def test_should_gordo_respond_allows_commands_on_link_fix_replies():
    import api.index

    config_module.reset_cache()
    commands = {"/ask": (lambda x: x, True, True)}
    chat_config = {
        "link_mode": "off",
        "ai_random_replies": True,
        "ai_command_followups": False,
        "ignore_link_fix_followups": True,
    }

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

        assert (
            should_gordo_respond(
                commands, "/ask", "investiga eso", msg, chat_config, None
            )
            is True
        )


def test_should_gordo_respond_respects_random_toggle():
    commands = {}
    chat_config = {
        "link_mode": "off",
        "ai_random_replies": False,
        "ai_command_followups": True,
        "ignore_link_fix_followups": True,
    }
    msg = {"chat": {"type": "group"}, "from": {"username": "test"}}

    with patch("os.environ.get") as mock_env, patch(
        "random.random", return_value=0.05
    ) as mock_random:
        mock_env.return_value = "testbot"
        assert (
            should_gordo_respond(
                commands, "", "hey gordo", msg, chat_config, None
            )
            is False
        )
        mock_random.assert_not_called()


def test_should_gordo_respond_blocks_followups_when_disabled():
    commands = {"/test": (lambda x: x, False, False)}
    chat_config = {
        "link_mode": "off",
        "ai_random_replies": True,
        "ai_command_followups": False,
        "ignore_link_fix_followups": True,
    }
    msg = {
        "chat": {"type": "group"},
        "from": {"username": "user"},
        "reply_to_message": {"from": {"username": "testbot"}},
    }
    reply_metadata = {"type": "command", "uses_ai": False}

    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "testbot"
        assert (
            should_gordo_respond(
                commands, "", "hola", msg, chat_config, reply_metadata
            )
            is False
        )


def test_should_auto_process_media_requires_direct_invocation_in_groups():
    commands = {"/ask": (lambda x: x, True, True)}

    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "testbot"

        plain_group_msg = {
            "chat": {"type": "group"},
            "from": {"username": "user"},
        }
        assert (
            should_auto_process_media(commands, "", "", plain_group_msg) is False
        )

        mention_group_msg = {
            "chat": {"type": "group"},
            "from": {"username": "user"},
            "text": "@testbot mirá esto",
        }
        assert (
            should_auto_process_media(
                commands, "", mention_group_msg["text"], mention_group_msg
            )
            is True
        )

        reply_group_msg = {
            "chat": {"type": "group"},
            "from": {"username": "user"},
            "reply_to_message": {"from": {"username": "testbot"}},
        }
        assert (
            should_auto_process_media(commands, "", "", reply_group_msg) is True
        )


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


def test_remove_gordo_prefix():
    sample = "gordo: hola capo\n  Gordo : me fui a dormir\nsin prefijo"
    expected = "hola capo\nme fui a dormir\nsin prefijo"
    assert remove_gordo_prefix(sample) == expected

    assert remove_gordo_prefix("sin etiqueta") == "sin etiqueta"
    assert remove_gordo_prefix(None) == ""


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


def test_format_balance_command_private_includes_topup_hint():
    from api.index import _format_balance_command

    with patch("api.index._fetch_balance", return_value=420):
        text = _format_balance_command("private", 1, 2)

    assert "tenés 42.0 créditos ia" in text
    assert "mandale /topup" in text


def test_format_balance_command_group_includes_topup_and_transfer_hints():
    from api.index import _format_balance_command

    with patch("api.index._fetch_balance", side_effect=[300, 1200]):
        text = _format_balance_command("group", 1, 2)

    assert "lo tuyo: 30.0" in text
    assert "lo del grupo: 120.0" in text
    assert "/topup" in text
    assert "/transfer <monto>" in text


def test_get_ai_onboarding_credits_default_is_30_units(monkeypatch):
    from api.index import get_ai_onboarding_credits

    monkeypatch.delenv("AI_ONBOARDING_CREDITS", raising=False)
    assert get_ai_onboarding_credits() == 30


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


def test_check_global_rate_limit_edge_cases(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")
    redis_client = MagicMock()
    redis_client.get.side_effect = [b"-1", b"-1"]
    assert check_global_rate_limit(redis_client) is True

    redis_client = MagicMock()
    redis_client.get.side_effect = [None, None]
    assert check_global_rate_limit(redis_client) is True

    redis_client = MagicMock()
    redis_client.get.side_effect = redis.RedisError
    assert check_global_rate_limit(redis_client) is True


def test_check_global_rate_limit_uses_groq_chat_budget(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")
    redis_client = MagicMock()
    redis_client.get.side_effect = [b"999", b"499999"]
    assert check_global_rate_limit(redis_client) is True

    redis_client = MagicMock()
    redis_client.get.side_effect = [b"1000", b"10"]
    assert check_global_rate_limit(redis_client) is False

    redis_client = MagicMock()
    redis_client.get.side_effect = [b"1", b"500000"]
    assert check_global_rate_limit(redis_client) is False


def test_groq_rate_limits_match_developer_plan_constants():
    assert index.GROQ_RATE_LIMITS["chat"] == {
        "rpm": 1000,
        "rpd": 500_000,
        "tpm": 250_000,
        "model": "moonshotai/kimi-k2-instruct-0905",
    }
    assert index.GROQ_RATE_LIMITS["compound"] == {
        "rpm": 200,
        "rpd": 20_000,
        "tpm": 200_000,
        "model": "groq/compound",
    }
    assert index.GROQ_RATE_LIMITS["vision"] == {
        "rpm": 1000,
        "rpd": 500_000,
        "tpm": 300_000,
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    }
    assert index.GROQ_RATE_LIMITS["transcribe"] == {
        "rpm": 400,
        "rpd": 200_000,
        "ash": 400_000,
        "asd": 4_000_000,
        "model": "whisper-large-v3-turbo",
    }
    assert index.GROQ_FREE_RATE_LIMITS["chat"] == {
        "rpm": 60,
        "rpd": 1_000,
        "tpm": 10_000,
        "tpd": 300_000,
        "model": "moonshotai/kimi-k2-instruct-0905",
    }
    assert index.GROQ_FREE_RATE_LIMITS["compound"] == {
        "rpm": 30,
        "rpd": 250,
        "tpm": 70_000,
        "model": "groq/compound",
    }
    assert index.GROQ_FREE_RATE_LIMITS["vision"] == {
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 30_000,
        "tpd": 500_000,
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    }
    assert index.GROQ_FREE_RATE_LIMITS["transcribe"] == {
        "rpm": 20,
        "rpd": 2_000,
        "ash": 7_200,
        "asd": 28_800,
        "model": "whisper-large-v3-turbo",
    }


def test_check_global_rate_limit_enforces_chat_tpm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")
    redis_client = MagicMock()
    redis_client.get.side_effect = [b"0", b"0", b"249999"]

    assert check_global_rate_limit(
        redis_client,
        scope="chat",
        token_count=2,
    ) is False


def test_check_global_rate_limit_enforces_transcribe_audio_limits(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")
    redis_client = MagicMock()
    redis_client.get.side_effect = [b"0", b"0", b"399999", b"3999999"]

    assert check_global_rate_limit(
        redis_client,
        scope="transcribe",
        audio_seconds=2.0,
    ) is False


def test_check_global_rate_limit_uses_paid_when_free_is_exhausted(monkeypatch):
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_key")
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")

    redis_client = MagicMock()
    redis_client.get.side_effect = [
        b"60",
        b"10",
        b"0",
        b"999",
        b"499999",
    ]

    assert check_global_rate_limit(redis_client) is True


def test_check_global_rate_limit_enforces_free_daily_token_budget(monkeypatch):
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_key")

    redis_client = MagicMock()
    redis_client.get.side_effect = [b"0", b"0", b"0", b"299999"]

    assert check_global_rate_limit(
        redis_client,
        scope="chat",
        token_count=2,
    ) is False


def test_reserve_and_reconcile_groq_rate_limit_releases_unused_tokens():
    class FakePipeline:
        def __init__(self, store):
            self.store = store
            self.ops = []

        def incrby(self, key, delta):
            self.ops.append(("incrby", key, int(delta)))
            return self

        def expire(self, key, ttl, nx=True):
            self.ops.append(("expire", key, int(ttl), bool(nx)))
            return self

        def execute(self):
            results = []
            for op in self.ops:
                if op[0] == "incrby":
                    _, key, delta = op
                    self.store[key] = int(self.store.get(key, 0)) + delta
                    results.append(self.store[key])
                else:
                    results.append(True)
            self.ops = []
            return results

    class FakeRedis:
        def __init__(self):
            self.store = {}

        def get(self, key):
            value = self.store.get(key)
            return str(value).encode("utf-8") if value is not None else None

        def pipeline(self):
            return FakePipeline(self.store)

    fake_redis = FakeRedis()
    reservation = index._reserve_groq_rate_limit(
        index.GROQ_PAID_ACCOUNT,
        "chat",
        request_count=1,
        token_count=1000,
        redis_client=fake_redis,
    )

    assert reservation is not None

    token_key = index._groq_rate_limit_metric_minute_key(
        index.GROQ_PAID_ACCOUNT,
        "chat",
        "tokens",
    )
    assert fake_redis.store[token_key] == 1000

    index._reconcile_groq_rate_limit(
        reservation,
        actual_request_count=1,
        actual_token_count=400,
        redis_client=fake_redis,
    )

    assert fake_redis.store[token_key] == 400


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
    assert "/price" in commands
    assert "/bresio" in commands
    assert "/bresios" in commands
    assert "/brecio" in commands
    assert "/brecios" in commands
    assert "/dolar" in commands
    assert "/usd" in commands
    assert "/topup" in commands
    assert "/balance" in commands
    assert "/printcredits" in commands
    assert "/creditlog" in commands
    assert "/purgeailog" in commands
    assert "/transfer" in commands

    # Test that AI commands are properly marked
    assert commands["/ask"][1] == True
    assert commands["/pregunta"][1] == True
    assert commands["/che"][1] == True
    assert commands["/gordo"][1] == True

    # Test that non-AI commands are properly marked
    assert commands["/random"][1] == False
    assert commands["/prices"][1] == False
    assert commands["/price"][1] == False
    assert commands["/dolar"][1] == False
    assert commands["/usd"][1] == False

    # Test function mappings
    assert commands["/ask"][0] == ask_ai
    assert commands["/random"][0] == select_random
    assert commands["/prices"][0] == get_prices
    assert commands["/price"][0] == get_prices
    assert commands["/bresio"][0] == get_prices
    assert commands["/bresios"][0] == get_prices
    assert commands["/brecio"][0] == get_prices
    assert commands["/brecios"][0] == get_prices
    assert commands["/help"][0] == get_help
    assert commands["/usd"][0] == _get_dollar_rates
    # Test search commands
    assert "/buscar" in commands
    assert "/search" in commands
    from api.index import search_command as _search_command

    assert commands["/buscar"][0] == _search_command
    assert commands["/search"][0] == _search_command
    assert commands["/buscar"][1] is True
    assert commands["/search"][1] is True


def test_price_alias_command_dispatches_to_get_prices():
    from api.index import initialize_commands, parse_command, get_prices

    command, args = parse_command("/price 1 btc in usd", "@bot")
    commands = initialize_commands()

    assert command == "/price"
    assert args == "1 btc in usd"

    handler_func, uses_ai, takes_params = commands[command]
    assert handler_func == get_prices
    assert uses_ai == False
    assert takes_params == True


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


def test_should_gordo_respond_complex_cases():
    import api.index
    from api.index import should_gordo_respond

    config_module.set_cache(
        {
            "trigger_words": ["gordo", "test", "bot"],
            "system_prompt": "You are a test bot",
        }
    )

    commands = {
        "/test": (lambda x: x, False, False),
        "/other": (lambda x: x, True, False),
    }
    chat_config = {
        "link_mode": "off",
        "ai_random_replies": True,
        "ai_command_followups": True,
        "ignore_link_fix_followups": True,
    }

    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "testbot"  # Set mock bot username

        # Test with command not in command list but starts with /
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(
                commands, "/unknown", "hello", msg, chat_config, None
            )
            is False
        )

        # Test with message containing bot username in middle of message
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(
                commands,
                "",
                "hey there @testbot how are you",
                msg,
                chat_config,
                None,
            )
            is True
        )

        # Test with reply to message that's not from the bot
        msg = {
            "chat": {"type": "group"},
            "from": {"username": "test"},
            "reply_to_message": {"from": {"username": "not_bot"}},
        }
        assert (
            should_gordo_respond(commands, "", "hello", msg, chat_config, None)
            is False
        )

    # Test trigger words in a separate block with proper random mocking
    with patch("os.environ.get") as mock_env, patch("random.random", return_value=0.5):
        mock_env.return_value = "testbot"

        # Test with trigger word but probability too high (0.5 > 0.1)
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(commands, "", "hey gordo", msg, chat_config, None)
            is False
        )

    with patch("os.environ.get") as mock_env, patch("random.random", return_value=0.05):
        mock_env.return_value = "testbot"

        # Test with multiple trigger words and low probability (0.05 < 0.1)
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(
                commands,
                "",
                "hey gordo and respondedor",
                msg,
                chat_config,
                None,
            )
            is True
        )

        # Test with case-insensitive trigger words
        msg = {"chat": {"type": "group"}, "from": {"username": "test"}}
        assert (
            should_gordo_respond(commands, "", "hey GORDO", msg, chat_config, None)
            is True
        )


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
    assert "esto es lo que sé hacer, boludo:" in result
    assert "/ask" in result
    assert "/dolar" in result
    assert "/usd" in result
    assert "/prices" in result
    assert "/config" in result


def test_get_instance_name_basic():
    from api.index import get_instance_name

    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "test_instance"
        assert get_instance_name() == "estoy corriendo en test_instance boludo"


def test_format_hacker_news_info_variants():
    from api.index import format_hacker_news_info

    stories = [
        {
            "title": "Historia Tres",
            "url": "https://example.com/tres",
            "points": 99,
            "comments": 12,
            "comments_url": "https://news.ycombinator.com/item?id=4",
        }
    ]

    with_discussion = format_hacker_news_info(stories)
    assert "Historia Tres" in with_discussion
    assert "99 pts" in with_discussion
    assert "HN:" in with_discussion

    without_discussion = format_hacker_news_info(stories, include_discussion=False)
    assert "HN:" not in without_discussion


def test_ensure_callback_updates_enabled_updates_webhook():
    from api import index as index_module

    index_module._WEBHOOK_CALLBACKS_CHECKED = False
    with patch.dict(
        index_module.environ,
        {
            "TELEGRAM_TOKEN": "token",
            "WEBHOOK_AUTH_KEY": "secret",
            "FUNCTION_URL": "https://example.com",
        },
        clear=True,
    ), patch(
        "api.index.get_telegram_webhook_info",
        return_value={
            "url": "https://example.com?key=secret",
            "allowed_updates": ["message"],
        },
    ) as mock_info, patch(
        "api.index.set_telegram_webhook",
        return_value=True,
    ) as mock_set, patch("api.index._log_config_event") as mock_log:
        ensure_callback_updates_enabled()

    mock_info.assert_called_once_with("token")
    mock_set.assert_called_once_with("https://example.com")
    mock_log.assert_called_once()
    assert index_module._WEBHOOK_CALLBACKS_CHECKED is True


def test_ensure_callback_updates_enabled_skips_when_allowed():
    from api import index as index_module

    index_module._WEBHOOK_CALLBACKS_CHECKED = False
    with patch.dict(
        index_module.environ,
        {
            "TELEGRAM_TOKEN": "token",
            "WEBHOOK_AUTH_KEY": "secret",
            "FUNCTION_URL": "https://example.com",
        },
        clear=True,
    ), patch(
        "api.index.get_telegram_webhook_info",
        return_value={
            "url": "https://example.com?key=secret",
            "allowed_updates": ["message", "callback_query", "pre_checkout_query"],
        },
    ), patch("api.index.set_telegram_webhook") as mock_set:
        ensure_callback_updates_enabled()

    mock_set.assert_not_called()
    assert index_module._WEBHOOK_CALLBACKS_CHECKED is True


def test_cached_requests_retries_on_failure(monkeypatch):
    from api.index import cached_requests

    calls = {"n": 0}

    class FakeResp:
        status_code = 200

        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(
        url, params=None, headers=None, timeout=5, verify=True
    ):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.RequestException("boom")
        return FakeResp(text=json.dumps({"ok": True}))

    monkeypatch.setattr("requests.get", fake_get)

    with patch("api.index.config_redis") as mock_redis:
        # Fake Redis client that stores in dict
        store = {}

        class R:
            def get(self, k):
                return store.get(k)

            def set(self, k, v):
                store[k] = v
                return True

        mock_redis.return_value = R()

        out = cached_requests("https://ex", {"a": 1}, None, 60)
        assert out is not None
        assert calls["n"] == 2  # one failure + one retry


def test_get_oil_price_success():
    mock_text_brent = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-03-08,100,102,99,103.23,10\n"
        "2026-03-09,100,102,99,101.17,10\n"
    )
    mock_text_wti = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-03-08,100,102,99,92.31,10\n"
        "2026-03-09,100,102,99,98.77,10\n"
    )

    with patch("api.index.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(text=mock_text_brent, raise_for_status=lambda: None),
            MagicMock(text=mock_text_wti, raise_for_status=lambda: None),
        ]

        result = get_oil_price()

    assert result.splitlines() == [
        "Brent: 101.17 USD (-2% 24hs)",
        "WTI: 98.77 USD (+7% 24hs)",
    ]




def test_get_oil_price_success_without_csv_header():
    mock_text_brent = "CB.F,20260309,161651,103.23,119.46,100.02,101.17,,\n"
    mock_text_wti = "CL.F,20260309,161655,92.31,119.43,96.45,98.77,,\n"

    with patch("api.index.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(text=mock_text_brent, raise_for_status=lambda: None),
            MagicMock(text=mock_text_wti, raise_for_status=lambda: None),
        ]

        result = get_oil_price()

    assert result.splitlines() == [
        "Brent: 101.17 USD (-2% 24hs)",
        "WTI: 98.77 USD (+7% 24hs)",
    ]

def test_get_oil_price_failure():
    with patch("api.index.requests.get", side_effect=Exception("boom")):
        result = get_oil_price()

    assert result == "no pude traer el precio del petróleo boludo"
