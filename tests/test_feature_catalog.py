from api.bot.command_registry import COMMAND_GROUPS
from api.bot.feature_catalog import (
    COMMAND_DESCRIPTIONS,
    catalog_command_aliases,
    command_aliases,
    render_ai_capabilities_prompt,
    render_help_text,
    telegram_command_descriptions,
)


def test_catalog_covers_all_real_command_aliases():
    assert command_aliases(COMMAND_GROUPS) <= catalog_command_aliases()


def test_removed_search_commands_are_not_described():
    assert "buscar" not in COMMAND_DESCRIPTIONS
    assert "search" not in COMMAND_DESCRIPTIONS


def test_telegram_descriptions_only_include_real_visible_commands():
    descriptions = telegram_command_descriptions(command_groups=COMMAND_GROUPS)

    assert "buscar" not in descriptions
    assert "search" not in descriptions
    assert "printcredits" not in descriptions
    assert "creditlog" not in descriptions
    assert {"crypto", "criptos", "instance"} <= set(descriptions)


def test_help_mentions_implicit_features():
    help_text = render_help_text()

    assert "búsqueda web nativa" in help_text
    assert "/buscar" not in help_text
    assert "/search" not in help_text
    assert "$ticker" in help_text
    assert "address Solana/EVM" in help_text
    assert "stickers" in help_text
    assert "YouTube" in help_text
    assert "/tareas" in help_text


def test_ai_capabilities_prompt_mentions_key_features_without_removed_commands():
    prompt = render_ai_capabilities_prompt()

    assert "CAPACIDADES DEL BOT" in prompt
    assert "/buscar y /search no existen" in prompt
    assert "búsqueda web nativa" in prompt
    assert "$ticker" in prompt
    assert "address Solana/EVM" in prompt
    assert "Telegram Stars" in prompt
