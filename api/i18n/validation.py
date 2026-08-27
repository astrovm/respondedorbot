"""Validate translation catalog structure and interpolation fields."""

from __future__ import annotations

from string import Formatter

from typing import Mapping

from api.i18n import CATALOGS, ES, SUPPORTED_LOCALES, Locale
from api.i18n.content import (
    CATEGORY_NAMES,
    COMMAND_DESCRIPTIONS,
    FEATURE_EXAMPLES,
    FEATURE_TEXT,
    HELP_TEXT,
)


def _localized_group_errors(
    group_name: str,
    group: Mapping[str, Mapping[Locale, object]],
) -> list[str]:
    errors: list[str] = []
    expected_locales = set(SUPPORTED_LOCALES)
    for key, translations in group.items():
        actual_locales = set(translations)
        if actual_locales != expected_locales:
            errors.append(
                f"{group_name}.{key} locales {sorted(actual_locales)} != {sorted(expected_locales)}"
            )
    return errors


def catalog_errors() -> list[str]:
    errors: list[str] = []
    expected_keys = set(ES)
    formatter = Formatter()
    for locale, catalog in CATALOGS.items():
        missing = expected_keys - set(catalog)
        extra = set(catalog) - expected_keys
        if missing:
            errors.append(f"{locale} missing keys: {sorted(missing)}")
        if extra:
            errors.append(f"{locale} extra keys: {sorted(extra)}")
        for key in expected_keys & set(catalog):
            expected_fields = {
                field for _literal, field, _spec, _conversion in formatter.parse(ES[key]) if field
            }
            actual_fields = {
                field
                for _literal, field, _spec, _conversion in formatter.parse(catalog[key])
                if field
            }
            if actual_fields != expected_fields:
                errors.append(
                    f"{locale}.{key} placeholders {sorted(actual_fields)} != "
                    f"{sorted(expected_fields)}"
                )
    expected_locales = set(SUPPORTED_LOCALES)
    errors.extend(_localized_group_errors("command", COMMAND_DESCRIPTIONS))
    errors.extend(_localized_group_errors("feature", FEATURE_TEXT))
    errors.extend(_localized_group_errors("feature_examples", FEATURE_EXAMPLES))
    errors.extend(_localized_group_errors("category", CATEGORY_NAMES))
    if set(HELP_TEXT) != expected_locales:
        errors.append(f"help locales {sorted(HELP_TEXT)} != {sorted(expected_locales)}")
    return errors


__all__ = ["catalog_errors"]
