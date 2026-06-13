from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from api import billing_callbacks
from api.ai_billing import (
    AIBillingPack,
    build_insufficient_credits_message,
    build_topup_keyboard,
    get_ai_billing_pack,
    get_ai_billing_packs,
    get_ai_onboarding_credits,
    maybe_grant_onboarding_credits,
    parse_topup_payload,
)
from api.credit_units import format_credit_units


class BillingService:
    def __init__(
        self,
        *,
        credits: Any,
        admin_report: Callable[..., None],
        telegram: Any,
        telegram_request: Callable[..., tuple[Any, str | None]],
        guard_callback: Callable[..., bool],
        answer_callback: Callable[..., None],
        answer_pre_checkout: Callable[..., None],
        extract_user_id: Callable[[Mapping[str, Any]], int | None],
    ) -> None:
        self.credits = credits
        self.admin_report = admin_report
        self.telegram = telegram
        self.telegram_request = telegram_request
        self.guard_callback = guard_callback
        self.answer_callback = answer_callback
        self.answer_pre_checkout = answer_pre_checkout
        self.extract_user_id = extract_user_id

    get_onboarding_credits = staticmethod(get_ai_onboarding_credits)
    get_packs = staticmethod(get_ai_billing_packs)
    get_pack = staticmethod(get_ai_billing_pack)
    build_topup_keyboard = staticmethod(build_topup_keyboard)
    parse_topup_payload = staticmethod(parse_topup_payload)
    build_insufficient_message = staticmethod(build_insufficient_credits_message)

    def is_available(self) -> bool:
        return bool(self.credits.is_configured())

    def fetch_balance(
        self,
        scope_type: Literal["user", "chat"],
        scope_id: int,
    ) -> int:
        return int(self.credits.get_balance(scope_type, int(scope_id)))

    def maybe_grant_onboarding(self, user_id: int | None) -> None:
        maybe_grant_onboarding_credits(
            self.credits,
            self.admin_report,
            user_id,
        )

    def unavailable_alert(self) -> str:
        return "el cobro de ia está hecho pelota, avisale al admin"

    def unavailable_message(self) -> str:
        return billing_callbacks.billing_unavailable_message()

    def send_invoice(
        self,
        *,
        chat_id: str,
        user_id: int,
        pack: AIBillingPack,
    ) -> bool:
        return billing_callbacks.send_stars_invoice(
            chat_id=chat_id,
            user_id=user_id,
            pack=pack,
            format_credits=format_credit_units,
            telegram_request=self.telegram_request,
        )

    def handle_topup_callback(self, callback_query: dict[str, Any]) -> None:
        billing_callbacks.handle_topup_callback(
            callback_query,
            guard_callback=self.guard_callback,
            billing_available=self.is_available,
            get_pack=self.get_pack,
            send_invoice=self.send_invoice,
            answer_callback=self.answer_callback,
            unavailable_alert=self.unavailable_alert,
        )

    def handle_pre_checkout(self, query: dict[str, Any]) -> None:
        billing_callbacks.handle_pre_checkout_query(
            query,
            billing_available=self.is_available,
            answer_query=self.answer_pre_checkout,
            unavailable_alert=self.unavailable_alert,
            parse_payload=self.parse_topup_payload,
            get_pack=self.get_pack,
        )

    def handle_successful_payment(self, message: dict[str, Any]) -> str:
        return billing_callbacks.handle_successful_payment(
            message,
            billing_available=self.is_available,
            unavailable_message=self.unavailable_message,
            send_message=self.telegram.send_message,
            extract_user_id=self.extract_user_id,
            parse_payload=self.parse_topup_payload,
            get_pack=self.get_pack,
            record_payment=self.credits.record_star_payment,
            admin_report=self.admin_report,
            format_credits=format_credit_units,
        )


__all__ = ["BillingService"]
