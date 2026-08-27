from api.bot.chat_config_defaults import CHAT_SETTING_DEFINITIONS
from api.bot.chat_settings import build_config_keyboard, build_config_text
from api.bot.command_registry import command_descriptions
from api.bot.feature_catalog import render_help_text
from api.billing.ai import AIMessageBilling
from api.core.i18n import catalog_errors, resolve_locale, use_locale
from api.markets.crypto import get_prices
from api.tasks.scheduler import format_interval
from api.tools.registry import execute_tool


def test_translation_catalogs_have_matching_keys_and_placeholders():
    assert catalog_errors() == []


def test_locale_resolution_uses_telegram_language_only_for_private_auto_mode():
    assert resolve_locale("auto", telegram_language_code="en-US", chat_type="private") == "en"
    assert resolve_locale("auto", telegram_language_code="en", chat_type="group") == "es"
    assert resolve_locale("es", telegram_language_code="en", chat_type="private") == "es"
    assert resolve_locale("en", telegram_language_code="es", chat_type="group") == "en"


def test_config_lists_every_registered_setting_in_english():
    config = {setting.key: setting.default for setting in CHAT_SETTING_DEFINITIONS}
    config["language"] = "en"

    text = build_config_text(config, "private")
    keyboard = build_config_keyboard(config, "private")

    assert text.startswith("Bot settings")
    assert text.count("only available in groups") == 2
    assert "Language used for bot messages and responses" in text
    assert "Free messages per user per hour" in text
    assert len(keyboard["inline_keyboard"]) == 5


def test_english_locale_reaches_help_commands_markets_and_tasks():
    with use_locale("en"):
        help_text = render_help_text()
        descriptions = command_descriptions("en")
        market_error = get_prices(
            "",
            change_fields={"24h": "percent_change_24h"},
            fetch_prices=lambda _currency: None,
            fetch_quotes=lambda *_args, **_kwargs: None,
        )
        interval = format_interval(3600)
        tool_error = execute_tool("weather", {"location": ""}, {})
        media_success = AIMessageBilling.is_transcribe_success_response(
            "🖼️ image: a black cat"
        )

    assert help_text.startswith("what I can do:")
    assert descriptions["settings"] == "open all bot settings"
    assert descriptions["language"] == "change the bot language [es|en]"
    assert market_error == "I could not load crypto prices"
    assert interval == "every 1 hour"
    assert tool_error.output == "weather is unavailable"
    assert media_success


def test_english_creditlog_has_no_spanish_labels():
    from api.admin.commands import build_creditlog_lines

    with use_locale("en"):
        text = "\n".join(build_creditlog_lines([{"metadata": {}}]))

    assert "latest AI settlements:" in text
    assert "no date | cmd=no command | status=ok" in text
    assert "reserved=0.00 charged=0.00 refund=0.00 extra=0.00 debt=0.00" in text
    assert "requests: no segments" in text
    assert "models: no models" in text
    assert "sin fecha" not in text
    assert "reservado=" not in text
