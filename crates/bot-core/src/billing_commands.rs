//! Pure plans and bilingual replies for user-facing billing commands.

use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, format_credit_units, parse_credit_units};
use crate::locale::Locale;
use crate::telegram_actions::{SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferCommandContext {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub user_id: Option<i64>,
    pub locale: Locale,
    pub is_group: bool,
    pub billing_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferCommandPlan {
    NotHandled,
    Reply(TelegramAction),
    Transfer {
        user_id: i64,
        chat_id: i64,
        amount: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferResult {
    pub transferred: bool,
    pub user_balance: i64,
    pub chat_balance: i64,
}

fn reply(context: TransferCommandContext, text: &str) -> TransferCommandPlan {
    let mut message = SendMessage::new(context.chat_id, text);
    message.reply_to_message_id = Some(context.message_id);
    TransferCommandPlan::Reply(TelegramAction::SendMessage(message))
}

#[must_use]
pub fn plan_transfer_command(
    message_text: &str,
    bot_name: &str,
    context: TransferCommandContext,
) -> TransferCommandPlan {
    let parsed = parse_command(message_text, bot_name);
    if parsed.command != "/transfer" {
        return TransferCommandPlan::NotHandled;
    }
    if !context.billing_available {
        return reply(
            context,
            match context.locale {
                Locale::Es => "el cobro de ia no está andando, avisale al admin",
                Locale::En => "AI billing is unavailable, please tell the admin",
            },
        );
    }
    if !context.is_group {
        return reply(
            context,
            match context.locale {
                Locale::Es => "esto es para grupos, capo: /transfer <monto>",
                Locale::En => "this command is for groups: /transfer <amount>",
            },
        );
    }
    let Some(user_id) = context.user_id else {
        return reply(
            context,
            match context.locale {
                Locale::Es => "no te pude sacar bien el usuario o el grupo para transferir",
                Locale::En => "I could not identify the user or group for the transfer",
            },
        );
    };
    let amount_token = parsed
        .message_text
        .split(' ')
        .next()
        .unwrap_or_default()
        .trim();
    let Some(amount) = parse_credit_units(amount_token).map(CreditUnits::value) else {
        return reply(
            context,
            match context.locale {
                Locale::Es => "mandalo bien: /transfer <monto>",
                Locale::En => "usage: /transfer <amount>",
            },
        );
    };
    if amount <= 0 {
        return reply(
            context,
            match context.locale {
                Locale::Es => "el monto tiene que ser mayor a 0, no me rompas las bolas",
                Locale::En => "the amount must be greater than 0",
            },
        );
    }
    TransferCommandPlan::Transfer {
        user_id,
        chat_id: context.chat_id.0,
        amount,
    }
}

#[must_use]
pub fn transfer_result_reply(amount: i64, result: TransferResult, locale: Locale) -> String {
    let user_balance = format_credit_units(CreditUnits::new(result.user_balance));
    if !result.transferred {
        return match locale {
            Locale::Es => format!(
                "Saldo personal insuficiente.\nDisponible: {user_balance} créditos\nProbá un monto menor o cargá con /topup."
            ),
            Locale::En => {
                format!(
                    "Insufficient personal balance.\nAvailable: {user_balance} credits\nTry a smaller amount or add credits with /topup."
                )
            }
        };
    }

    let amount = format_credit_units(CreditUnits::new(amount));
    let chat_balance = format_credit_units(CreditUnits::new(result.chat_balance));
    match locale {
        Locale::Es => format!(
            "Transferencia al grupo: {amount} créditos\n\nSaldo personal: {user_balance} créditos\nSaldo del grupo: {chat_balance} créditos"
        ),
        Locale::En => format!(
            "Transferred to group: {amount} credits\n\nPersonal balance: {user_balance} credits\nGroup balance: {chat_balance} credits"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TransferCommandContext, TransferCommandPlan, TransferResult, plan_transfer_command,
        transfer_result_reply,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    fn context(locale: Locale) -> TransferCommandContext {
        TransferCommandContext {
            chat_id: ChatId(-202),
            message_id: MessageId(12),
            user_id: Some(55),
            locale,
            is_group: true,
            billing_available: true,
        }
    }

    fn reply_text(plan: TransferCommandPlan) -> String {
        let TransferCommandPlan::Reply(TelegramAction::SendMessage(message)) = plan else {
            return String::new();
        };
        assert_eq!(message.reply_to_message_id, Some(MessageId(12)));
        message.text
    }

    #[test]
    fn transfer_plan_parses_fractional_credits_and_bot_suffix() {
        assert_eq!(
            plan_transfer_command("/transfer@mybot 0.1", "@mybot", context(Locale::Es)),
            TransferCommandPlan::Transfer {
                user_id: 55,
                chat_id: -202,
                amount: 10,
            }
        );
        assert_eq!(
            plan_transfer_command("/balance", "@mybot", context(Locale::Es)),
            TransferCommandPlan::NotHandled
        );
    }

    #[test]
    fn transfer_plan_preserves_guard_and_validation_order() {
        let mut unavailable = context(Locale::En);
        unavailable.billing_available = false;
        unavailable.is_group = false;
        assert_eq!(
            reply_text(plan_transfer_command("/transfer bad", "", unavailable)),
            "AI billing is unavailable, please tell the admin"
        );

        let mut private = context(Locale::Es);
        private.is_group = false;
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", private)),
            "esto es para grupos, capo: /transfer <monto>"
        );

        let mut missing_user = context(Locale::En);
        missing_user.user_id = None;
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", missing_user)),
            "I could not identify the user or group for the transfer"
        );

        assert_eq!(
            reply_text(plan_transfer_command(
                "/transfer 1.555",
                "",
                context(Locale::Es)
            )),
            "mandalo bien: /transfer <monto>"
        );
        assert_eq!(
            reply_text(plan_transfer_command(
                "/transfer -1",
                "",
                context(Locale::En)
            )),
            "the amount must be greater than 0"
        );
    }

    #[test]
    fn transfer_result_replies_match_both_outcomes_and_locales() {
        assert_eq!(
            transfer_result_reply(
                10,
                TransferResult {
                    transferred: true,
                    user_balance: 285,
                    chat_balance: 1_215,
                },
                Locale::Es,
            ),
            "Transferencia al grupo: 0.10 créditos\n\nSaldo personal: 2.85 créditos\nSaldo del grupo: 12.15 créditos"
        );
        assert_eq!(
            transfer_result_reply(
                150,
                TransferResult {
                    transferred: true,
                    user_balance: 70,
                    chat_balance: 230,
                },
                Locale::En,
            ),
            "Transferred to group: 1.50 credits\n\nPersonal balance: 0.70 credits\nGroup balance: 2.30 credits"
        );
        assert_eq!(
            transfer_result_reply(
                150,
                TransferResult {
                    transferred: false,
                    user_balance: 70,
                    chat_balance: 80,
                },
                Locale::Es,
            ),
            "Saldo personal insuficiente.\nDisponible: 0.70 créditos\nProbá un monto menor o cargá con /topup."
        );
    }
}
