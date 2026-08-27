"""Structural catalog of bot capabilities; localized prose lives in ``api.i18n``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from api.bot.command_registry import (
    COMMAND_DESCRIPTIONS,
    COMMAND_GROUPS,
    aliases_for,
    command_descriptions,
)
from api.i18n import current_locale
from api.i18n.content import category_name, feature_examples, feature_text, help_text

CommandGroup = tuple[tuple[str, ...], str, bool, bool]


@dataclass(frozen=True)
class FeatureEntry:
    key: str
    commands: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    category: str = "general"
    help_visible: bool = True
    telegram_visible: bool = False
    ai_visible: bool = True
    admin_only: bool = False
    implicit: bool = False


FEATURES: tuple[FeatureEntry, ...] = (
    FeatureEntry("ai_chat", aliases_for("ask_ai"), (), "ai", telegram_visible=True),
    FeatureEntry("web_search", category="ai", implicit=True),
    FeatureEntry(
        "crypto",
        aliases_for("get_prices"),
        (),
        "markets",
        telegram_visible=True,
    ),
    FeatureEntry("weather", aliases_for("get_weather"), telegram_visible=True),
    FeatureEntry("token_cards", category="markets", implicit=True),
    FeatureEntry("dollar", aliases_for("get_dollar_rates"), (), "markets", telegram_visible=True),
    FeatureEntry(
        "stocks",
        aliases_for("get_stock_prices"),
        (),
        "markets",
        telegram_visible=True,
    ),
    FeatureEntry("oil", aliases_for("get_oil_price"), category="markets", telegram_visible=True),
    FeatureEntry(
        "bcra", aliases_for("handle_bcra_variables"), category="markets", telegram_visible=True
    ),
    FeatureEntry(
        "elections",
        aliases_for("get_polymarket_global_elections"),
        category="markets",
        telegram_visible=True,
    ),
    FeatureEntry(
        "arbitrage",
        aliases_for("get_rulo", "get_devo", "powerlaw", "rainbow", "satoshi"),
        (),
        "markets",
        telegram_visible=True,
    ),
    FeatureEntry(
        "media", aliases_for("handle_transcribe"), category="media", telegram_visible=True
    ),
    FeatureEntry("links", category="links", implicit=True),
    FeatureEntry(
        "tasks",
        aliases_for("task_command"),
        (),
        "productivity",
        telegram_visible=True,
    ),
    FeatureEntry(
        "memory",
        aliases_for("summary_command"),
        (),
        "memory",
        telegram_visible=True,
    ),
    FeatureEntry(
        "utilities",
        aliases_for(
            "select_random",
            "convert_base",
            "convert_to_command",
            "get_timestamp",
            "get_instance_name",
        ),
        (),
        "utilities",
        telegram_visible=True,
    ),
    FeatureEntry(
        "gifs",
        aliases_for("get_good_morning", "get_good_night"),
        category="utilities",
        telegram_visible=True,
    ),
    FeatureEntry(
        "config", aliases_for("config_command"), category="settings", telegram_visible=True
    ),
    FeatureEntry(
        "language", aliases_for("language_command"), category="settings", telegram_visible=True
    ),
    FeatureEntry(
        "credits",
        aliases_for("topup_command", "balance_command", "charges_command", "transfer_command"),
        (),
        "credits",
        telegram_visible=True,
    ),
    FeatureEntry(
        "credit_admin",
        aliases_for("printcredits_command", "creditlog_command"),
        category="admin",
        help_visible=False,
        admin_only=True,
    ),
    FeatureEntry("help", aliases_for("get_help"), category="utilities", telegram_visible=True),
)


def _strip_slash(command: str) -> str:
    return command.lstrip("/")


def command_aliases(command_groups: Sequence[CommandGroup] = COMMAND_GROUPS) -> set[str]:
    return {
        _strip_slash(alias)
        for aliases, _handler, _uses_ai, _takes_params in command_groups
        for alias in aliases
    }


def catalog_command_aliases(entries: Iterable[FeatureEntry] = FEATURES) -> set[str]:
    return {_strip_slash(command) for entry in entries for command in entry.commands}


def get_feature_for_command(command: str) -> Optional[FeatureEntry]:
    normalized = _strip_slash(command)
    for entry in FEATURES:
        if normalized in {_strip_slash(alias) for alias in entry.commands}:
            return entry
    return None


def telegram_command_descriptions(
    *,
    command_groups: Sequence[CommandGroup] = COMMAND_GROUPS,
    descriptions: Mapping[str, str] = COMMAND_DESCRIPTIONS,
    locale: str = "es",
) -> Dict[str, str]:
    if descriptions is COMMAND_DESCRIPTIONS:
        descriptions = command_descriptions(locale)
    allowed = command_aliases(command_groups)
    visible = {
        _strip_slash(command)
        for entry in FEATURES
        if entry.telegram_visible and not entry.admin_only
        for command in entry.commands
    }
    return {
        name: desc for name, desc in descriptions.items() if name in allowed and name in visible
    }


def render_help_text(entries: Iterable[FeatureEntry] = FEATURES) -> str:
    locale = current_locale()
    localized_title = help_text("title", locale)
    assert isinstance(localized_title, str)
    lines = [localized_title, ""]
    current_category = ""
    for entry in entries:
        if not entry.help_visible or entry.admin_only:
            continue
        category = category_name(entry.category, locale)
        if category != current_category:
            if current_category:
                lines.append("")
            current_category = category
            lines.append(f"{category}:")
        title, description = feature_text(entry.key, locale)
        prefix = ", ".join(entry.commands) if entry.commands else title
        lines.append(f"- {prefix}: {description}")
        example_label = help_text("example", locale)
        assert isinstance(example_label, str)
        for example in feature_examples(entry.key, locale):
            lines.append(f"  {example_label}: {example}")
    return "\n".join(lines).strip()


def render_ai_capabilities_prompt(entries: Iterable[FeatureEntry] = FEATURES) -> str:
    locale = current_locale()
    header = help_text("capabilities", locale)
    assert isinstance(header, tuple)
    lines = list(header)
    for entry in entries:
        if not entry.ai_visible:
            continue
        title, description = feature_text(entry.key, locale)
        label = ", ".join(entry.commands) if entry.commands else title
        if entry.admin_only:
            admin_label = help_text("admin", locale)
            assert isinstance(admin_label, str)
            label = f"{label} ({admin_label})"
        lines.append(f"- {label}: {description}")
    return "\n".join(lines)


__all__ = [
    "COMMAND_DESCRIPTIONS",
    "FEATURES",
    "FeatureEntry",
    "catalog_command_aliases",
    "command_aliases",
    "get_feature_for_command",
    "render_ai_capabilities_prompt",
    "render_help_text",
    "telegram_command_descriptions",
]
