"""Chat configuration default constants shared by services and UI helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True, slots=True)
class ChatSettingDefinition:
    key: str
    default: object
    action: str
    group_only: bool = False


CHAT_SETTING_DEFINITIONS: tuple[ChatSettingDefinition, ...] = (
    ChatSettingDefinition("language", "auto", "language"),
    ChatSettingDefinition("link_mode", "reply", "link"),
    ChatSettingDefinition("ai_command_followups", True, "followups"),
    ChatSettingDefinition(
        "ignore_link_fix_followups",
        True,
        "linkfixfollowups",
    ),
    ChatSettingDefinition("timezone_offset", -3, "timezone"),
    ChatSettingDefinition("ai_random_replies", True, "random", group_only=True),
    ChatSettingDefinition(
        "creditless_user_hourly_limit",
        5,
        "creditless",
        group_only=True,
    ),
)

CHAT_CONFIG_DEFAULTS: Dict[str, object] = {
    setting.key: setting.default for setting in CHAT_SETTING_DEFINITIONS
}

TIMEZONE_OFFSET_MIN = -12
TIMEZONE_OFFSET_MAX = 14

CHAT_ADMIN_STATUS_TTL = 300
