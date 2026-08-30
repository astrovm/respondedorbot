"""Optional Rust Telegram payment validator used during the migration."""

from __future__ import annotations

from typing import Protocol, cast

from api.core.rust_bridge import load_rust_bridge


class RustTelegramPayments(Protocol):
    def telegram_evaluate_pre_checkout(
        self,
        query_json: str,
        billing_available: bool,
        pack_id: str | None,
        pack_xtr_amount: int | None,
    ) -> str: ...


def load_rust_telegram_payments() -> RustTelegramPayments | None:
    module = load_rust_bridge("RUST_TELEGRAM_PAYMENTS_ENABLED")
    if module is None:
        return None
    return cast(RustTelegramPayments, module)
