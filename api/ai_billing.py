"""AI credits billing helpers used by commands and message flow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import environ
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]


AI_BILLING_DEFAULT_PACKS = [
    {"id": "p100", "credits": 100, "xtr": 50},
    {"id": "p250", "credits": 250, "xtr": 125},
    {"id": "p500", "credits": 500, "xtr": 250},
    {"id": "p1000", "credits": 1000, "xtr": 500},
    {"id": "p2500", "credits": 2500, "xtr": 1250},
]


def is_group_chat_type(chat_type: Optional[str]) -> bool:
    return str(chat_type) in {"group", "supergroup"}


def get_ai_credits_per_response() -> int:
    """Return credits charged per AI response."""

    raw_value = str(environ.get("AI_CREDITS_PER_RESPONSE") or "1").strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


def get_ai_onboarding_credits() -> int:
    """Return onboarding credits granted once per user."""

    raw_value = str(environ.get("AI_ONBOARDING_CREDITS") or "3").strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 3
    return max(0, parsed)


def get_ai_billing_packs() -> List[Dict[str, int]]:
    """Load Stars billing packs from env or defaults."""

    raw_value = str(environ.get("AI_STARS_PACKS_JSON") or "").strip()
    if not raw_value:
        return list(AI_BILLING_DEFAULT_PACKS)

    try:
        loaded = json.loads(raw_value)
    except json.JSONDecodeError:
        return list(AI_BILLING_DEFAULT_PACKS)

    if not isinstance(loaded, list):
        return list(AI_BILLING_DEFAULT_PACKS)

    normalized: List[Dict[str, int]] = []
    for item in loaded:
        if not isinstance(item, Mapping):
            continue
        pack_id = str(item.get("id", "")).strip()
        try:
            credits = int(item.get("credits"))  # type: ignore[arg-type]
            xtr = int(item.get("xtr"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not pack_id or credits <= 0 or xtr <= 0:
            continue
        normalized.append({"id": pack_id, "credits": credits, "xtr": xtr})

    return normalized or list(AI_BILLING_DEFAULT_PACKS)


def get_ai_billing_pack(pack_id: str) -> Optional[Dict[str, int]]:
    """Return the pack dict matching pack_id."""

    for pack in get_ai_billing_packs():
        if str(pack.get("id")) == str(pack_id):
            return pack
    return None


def build_topup_keyboard() -> Dict[str, Any]:
    """Build inline keyboard with top-up packs."""

    rows: List[List[Dict[str, str]]] = []
    for pack in get_ai_billing_packs():
        pack_id = str(pack["id"])
        rows.append(
            [
                {
                    "text": f"{pack['credits']} créditos - {pack['xtr']} ⭐",
                    "callback_data": f"topup:{pack_id}",
                }
            ]
        )
    return {"inline_keyboard": rows}


def parse_topup_payload(payload: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse invoice payload for top-up purchases."""

    if not payload:
        return None, None
    parts = str(payload).split(":")
    if len(parts) < 2 or parts[0] != "topup":
        return None, None

    user_id: Optional[int] = None
    if len(parts) >= 3:
        try:
            user_id = int(parts[2])
        except (TypeError, ValueError):
            user_id = None
    return parts[1], user_id


def build_insufficient_credits_message(
    *,
    chat_type: str,
    user_balance: int,
    chat_balance: int,
) -> str:
    """Build a user-facing paywall message when no credits are available."""

    if is_group_chat_type(chat_type):
        return (
            "sin créditos para IA en este grupo, che.\n"
            f"- Tu saldo: {user_balance}\n"
            f"- Saldo del grupo: {chat_balance}\n"
            "agregá créditos con /topup (por privado) y si querés pasá al grupo con /transfer <monto>.\n"
            "podés ver todo con /balance"
        )

    return (
        "te quedaste sin créditos IA.\n"
        f"saldo actual: {user_balance}\n"
        "agregá créditos con /topup para recargar con Stars ⭐"
    )


def extract_numeric_chat_id(chat_id: str) -> Optional[int]:
    try:
        return int(chat_id)
    except (TypeError, ValueError):
        return None


def extract_user_id(message: Mapping[str, Any]) -> Optional[int]:
    user = message.get("from") if message else None
    if not isinstance(user, Mapping):
        return None
    try:
        return int(user.get("id"))
    except (TypeError, ValueError):
        return None


def maybe_grant_onboarding_credits(
    credits_db_service: Any,
    admin_reporter: AdminReporter,
    user_id: Optional[int],
) -> None:
    """Grant onboarding credits when configured and not yet granted."""

    if user_id is None:
        return

    onboarding_credits = get_ai_onboarding_credits()
    if onboarding_credits <= 0:
        return

    try:
        credits_db_service.grant_onboarding_if_needed(user_id, onboarding_credits)
    except Exception as error:
        admin_reporter(
            "falló la acreditación de onboarding",
            error,
            {"user_id": user_id},
        )


def format_balance_command(
    credits_db_service: Any,
    *,
    chat_type: str,
    user_id: int,
    chat_id: int,
) -> str:
    """Format the /balance response for private and group chats."""

    user_balance = credits_db_service.get_balance("user", int(user_id))
    if is_group_chat_type(chat_type):
        chat_balance = credits_db_service.get_balance("chat", int(chat_id))
        return (
            "saldos IA:\n"
            f"- tu saldo personal: {user_balance}\n"
            f"- saldo del grupo: {chat_balance}\n"
            "si no te alcanza el saldo personal, se usa el del grupo.\n"
            "para cargar créditos: /topup (por privado)\n"
            "si querés pasar créditos al grupo: /transfer <monto>"
        )

    return f"tu saldo personal de IA es: {user_balance}\npara cargar créditos: /topup"


@dataclass
class AIMessageBilling:
    """Charge/refund helper for a single handled message."""

    credits_db_service: Any
    admin_reporter: AdminReporter
    gen_random_fn: Callable[[str], str]
    build_insufficient_credits_message_fn: Callable[..., str]
    maybe_grant_onboarding_credits_fn: Callable[[Optional[int]], None]
    get_ai_credits_per_response_fn: Callable[[], int]
    command: str
    chat_id: str
    chat_type: str
    user_id: Optional[int]
    numeric_chat_id: Optional[int]
    message: Mapping[str, Any]
    onboarding_checked: bool = False
    billing_not_configured_message: str = "el cobro IA no está configurado, avisale al admin."
    billing_missing_scope_message: str = "no pude identificar usuario/chat para cobrar IA."
    billing_charge_error_message: str = "error cobrando créditos IA, intentá de nuevo."
    charge_errors: List[str] = field(default_factory=list)

    def _resolve_ai_charge_context(self) -> Tuple[Optional[int], Optional[str]]:
        if not self.credits_db_service.is_configured():
            return None, self.billing_not_configured_message
        if self.user_id is None or self.numeric_chat_id is None:
            return None, self.billing_missing_scope_message
        return (
            self.numeric_chat_id if is_group_chat_type(self.chat_type) else None,
            None,
        )

    def _build_insufficient_credits_reply(self, charge_result: Mapping[str, Any]) -> str:
        random_name = str(self.message.get("from", {}).get("first_name") or "boludo")
        random_response = self.gen_random_fn(random_name)
        credits_message = self.build_insufficient_credits_message_fn(
            chat_type=self.chat_type,
            user_balance=int(charge_result.get("user_balance", 0)),
            chat_balance=int(charge_result.get("chat_balance", 0)),
        )
        return f"{random_response}\n\n{credits_message}"

    def charge_one_ai_request(
        self, usage_tag: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Charge one AI request, returning charge metadata or a user-facing error."""

        chat_scope_id, context_error = self._resolve_ai_charge_context()
        if context_error:
            return None, context_error
        if self.user_id is None:
            return None, self.billing_missing_scope_message

        if not self.onboarding_checked:
            self.maybe_grant_onboarding_credits_fn(self.user_id)
            self.onboarding_checked = True

        credits_cost = self.get_ai_credits_per_response_fn()
        try:
            charge_result = self.credits_db_service.charge_ai_credits(
                user_id=self.user_id,
                chat_id=chat_scope_id,
                amount=credits_cost,
            )
        except Exception as error:
            self.admin_reporter(
                "Error charging IA credits",
                error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "command": self.command,
                    "usage_tag": usage_tag,
                },
            )
            return None, self.billing_charge_error_message

        if not charge_result.get("ok"):
            return None, self._build_insufficient_credits_reply(charge_result)

        return {
            "credits_cost": credits_cost,
            "chat_scope_id": chat_scope_id,
            "source": str(charge_result.get("source") or "user"),
        }, None

    def refund_ai_charge_meta(
        self,
        charge_meta: Optional[Mapping[str, Any]],
        reason: str,
    ) -> None:
        """Refund a previously applied AI charge."""

        if not charge_meta or self.user_id is None:
            return

        try:
            self.credits_db_service.refund_ai_charge(
                user_id=self.user_id,
                chat_id=charge_meta.get("chat_scope_id"),
                amount=int(charge_meta.get("credits_cost", 1)),
                source="chat"
                if str(charge_meta.get("source") or "user") == "chat"
                else "user",
            )
        except Exception as refund_error:
            self.admin_reporter(
                "falló el reintegro de créditos IA",
                refund_error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reason": reason,
                    "command": self.command,
                },
            )

    @staticmethod
    def is_transcribe_success_response(text: Optional[str]) -> bool:
        """Return True when a media transcription/description completed successfully."""

        if not text:
            return False
        success_prefixes = (
            "🎵 Transcripción: ",
            "🖼️ Descripción: ",
            "🎨 Descripción del sticker: ",
        )
        return text.startswith(success_prefixes)


__all__ = [
    "AIMessageBilling",
    "build_insufficient_credits_message",
    "build_topup_keyboard",
    "extract_numeric_chat_id",
    "extract_user_id",
    "format_balance_command",
    "get_ai_billing_pack",
    "get_ai_billing_packs",
    "get_ai_credits_per_response",
    "get_ai_onboarding_credits",
    "maybe_grant_onboarding_credits",
    "parse_topup_payload",
]
