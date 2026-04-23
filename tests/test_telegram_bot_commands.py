from tests.support import *


def test_build_commands_list_sorts_deduplicates_and_skips_undocumented_aliases():
    from api.telegram_bot_commands import build_commands_list

    command_groups = (
        (("/zeta", "/alpha", "/alpha"), "handler", False, False),
        (("/beta",), "handler", False, False),
        (("/missing",), "handler", False, False),
    )

    result = build_commands_list(
        command_groups,
        descriptions={
            "alpha": "alpha desc",
            "beta": "beta desc",
            "zeta": "zeta desc",
        },
    )

    assert result == [
        {"command": "alpha", "description": "alpha desc"},
        {"command": "beta", "description": "beta desc"},
        {"command": "zeta", "description": "zeta desc"},
    ]


def test_update_bot_commands_posts_serialized_sorted_commands():
    from api.telegram_bot_commands import update_bot_commands

    request_fn = MagicMock(return_value=(None, None))

    ok = update_bot_commands(
        token="abc",
        request_fn=request_fn,
        command_groups=(
            (("/beta",), "handler", False, False),
            (("/alpha",), "handler", False, False),
        ),
        descriptions={
            "alpha": "alpha desc",
            "beta": "beta desc",
        },
        logger=MagicMock(),
    )

    assert ok is True
    request_fn.assert_called_once_with(
        "setMyCommands",
        method="POST",
        json_payload={
            "commands": json.dumps(
                [
                    {"command": "alpha", "description": "alpha desc"},
                    {"command": "beta", "description": "beta desc"},
                ]
            )
        },
        token="abc",
        expect_json=False,
    )


def test_update_bot_commands_logs_error_and_returns_false():
    from api.telegram_bot_commands import update_bot_commands

    logger = MagicMock()

    ok = update_bot_commands(
        token="abc",
        request_fn=MagicMock(return_value=(None, "boom")),
        command_groups=((("/alpha",), "handler", False, False),),
        descriptions={"alpha": "alpha desc"},
        logger=logger,
    )

    assert ok is False
    logger.assert_called_once_with("Error updating bot commands: boom")


def test_build_commands_list_includes_tldr_alias_when_documented():
    from api.command_registry import COMMAND_GROUPS
    from api.telegram_bot_commands import build_commands_list

    commands = build_commands_list(COMMAND_GROUPS)
    names = {item["command"] for item in commands}

    assert "resumen" in names
    assert "summary" in names
    assert "tldr" in names
