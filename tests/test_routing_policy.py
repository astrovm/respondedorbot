from unittest.mock import MagicMock, patch

from tests.support import *  # noqa: F401,F403


def _build_policy(has_credits_fn):
    from api import index
    from api.command_registry import should_gordo_respond as base_should_respond
    from api.routing_policy import RoutingPolicy

    return RoutingPolicy(
        base_policy=base_should_respond,
        has_ai_credits_for_random_reply=has_credits_fn,
        load_bot_config_fn=index.load_bot_config,
    )


def test_routing_policy_allows_private_chat_messages(monkeypatch):
    from api import config as config_module

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    config_module.set_cache(
        {
            "trigger_words": ["gordo", "test", "bot"],
            "system_prompt": "You are a test bot",
        }
    )

    has_credits = MagicMock(return_value=False)
    policy = _build_policy(has_credits)

    message = {"chat": {"type": "private"}, "from": {"username": "user"}}

    assert (
        policy.should_respond(
            {},
            "",
            "hola",
            message,
            {
                "link_mode": "off",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
            },
            None,
        )
        is True
    )
    has_credits.assert_not_called()


def test_routing_policy_blocks_random_group_reply_without_credits(monkeypatch):
    from api import config as config_module

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    config_module.set_cache(
        {
            "trigger_words": ["gordo", "test", "bot"],
            "system_prompt": "You are a test bot",
        }
    )

    has_credits = MagicMock(return_value=False)
    policy = _build_policy(has_credits)

    message = {"chat": {"type": "group"}, "from": {"username": "user"}}

    with patch("random.random", return_value=0.05):
        assert (
            policy.should_respond(
                {},
                "",
                "hey gordo",
                message,
                {
                    "link_mode": "off",
                    "ai_random_replies": True,
                    "ai_command_followups": True,
                    "ignore_link_fix_followups": True,
                },
                None,
            )
            is False
        )

    has_credits.assert_called_once_with(message)


def test_routing_policy_allows_reply_to_bot_message(monkeypatch):
    from api import config as config_module

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    config_module.set_cache(
        {
            "trigger_words": ["gordo", "test", "bot"],
            "system_prompt": "You are a test bot",
        }
    )

    has_credits = MagicMock(return_value=False)
    policy = _build_policy(has_credits)

    message = {
        "chat": {"type": "group"},
        "from": {"username": "user"},
        "reply_to_message": {"from": {"username": "testbot"}},
    }

    assert (
        policy.should_respond(
            {},
            "",
            "hola",
            message,
            {
                "link_mode": "off",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
            },
            None,
        )
        is True
    )
    has_credits.assert_not_called()
