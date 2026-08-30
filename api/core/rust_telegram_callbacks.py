"""Optional Rust callback-envelope parser used during the migration."""

from __future__ import annotations

from typing import Protocol, cast

from api.core.rust_bridge import load_rust_bridge


class RustTelegramCallbacks(Protocol):
    def telegram_parse_callback_context(self, callback_json: str) -> str: ...


def load_rust_telegram_callbacks() -> RustTelegramCallbacks | None:
    module = load_rust_bridge("RUST_TELEGRAM_CALLBACKS_ENABLED")
    if module is None:
        return None
    return cast(RustTelegramCallbacks, module)
