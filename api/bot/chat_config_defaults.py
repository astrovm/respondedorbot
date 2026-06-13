"""Chat configuration default constants shared by services and UI helpers."""

from __future__ import annotations

from typing import Dict

CHAT_CONFIG_DEFAULTS: Dict[str, object] = {
    "link_mode": "reply",
    "ai_random_replies": True,
    "ai_command_followups": True,
    "ignore_link_fix_followups": True,
    "timezone_offset": -3,
    "creditless_user_hourly_limit": 5,
}

TIMEZONE_OFFSET_MIN = -12
TIMEZONE_OFFSET_MAX = 14

CHAT_ADMIN_STATUS_TTL = 300
