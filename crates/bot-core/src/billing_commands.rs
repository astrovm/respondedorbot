//! Pure plans and bilingual replies for user-facing billing commands.

use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, display_credit_units, parse_credit_units};
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
    /// The sender of the message `/transfer` replies to, unless that is this
    /// bot. Replying to a person sends the credits to them instead of the group.
    pub recipient: Option<TransferRecipient>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferRecipient {
    pub user_id: i64,
    pub is_bot: bool,
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
    TransferToUser {
        user_id: i64,
        recipient_id: i64,
        amount: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferResult {
    pub transferred: bool,
    pub user_balance: i64,
    pub chat_balance: i64,
}

/// Shared reply for every credit command while billing storage is unavailable.
#[must_use]
pub const fn billing_unavailable(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => {
            "Los créditos de IA no están disponibles en este momento. Probá más tarde o avisale al admin"
        }
        Locale::En => "AI credits are unavailable right now. Try again later or tell the admin",
    }
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
            crate::billing_commands::billing_unavailable(context.locale),
        );
    }
    if !context.is_group {
        return reply(
            context,
            match context.locale {
                Locale::Es => {
                    "Esto es para grupos, capo. Usalo ahí: /transfer <monto> se lo pasa al grupo, o respondé a alguien para pasárselo a esa persona"
                }
                Locale::En => {
                    "This command is for groups. Use it there: /transfer <amount> moves credits to the group, or reply to someone to send them to that person"
                }
            },
        );
    }
    let Some(user_id) = context.user_id else {
        return reply(
            context,
            match context.locale {
                Locale::Es => "No pude identificar tu usuario o el grupo para transferir",
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
                Locale::Es => {
                    "Mandalo así: /transfer <monto>\nEjemplo: /transfer 1.5\nRespondé a alguien para pasárselo a esa persona"
                }
                Locale::En => {
                    "Usage: /transfer <amount>\nExample: /transfer 1.5\nReply to someone to send them the credits"
                }
            },
        );
    };
    if amount <= 0 {
        return reply(
            context,
            match context.locale {
                Locale::Es => "El monto tiene que ser mayor a 0, no me rompas las bolas",
                Locale::En => "The amount must be greater than 0",
            },
        );
    }
    if let Some(recipient) = context.recipient {
        if recipient.is_bot {
            return reply(
                context,
                match context.locale {
                    Locale::Es => "Los bots no usan créditos, pasáselos a una persona",
                    Locale::En => "Bots can't use credits. Send them to a person",
                },
            );
        }
        if recipient.user_id == user_id {
            return reply(
                context,
                match context.locale {
                    Locale::Es => "No te podés pasar créditos a vos mismo",
                    Locale::En => "You can't send credits to yourself",
                },
            );
        }
        return TransferCommandPlan::TransferToUser {
            user_id,
            recipient_id: recipient.user_id,
            amount,
        };
    }
    TransferCommandPlan::Transfer {
        user_id,
        chat_id: context.chat_id.0,
        amount,
    }
}

#[must_use]
pub fn transfer_result_reply(amount: i64, result: TransferResult, locale: Locale) -> String {
    let user_balance = display_credit_units(CreditUnits::new(result.user_balance));
    if !result.transferred {
        return match locale {
            Locale::Es => format!(
                "No te alcanza el saldo personal: tenés {user_balance} créditos\nProbá con un monto menor o cargá con /topup"
            ),
            Locale::En => {
                format!(
                    "Not enough personal balance: you have {user_balance} credits\nTry a smaller amount or add credits with /topup"
                )
            }
        };
    }

    let amount = display_credit_units(CreditUnits::new(amount));
    let chat_balance = display_credit_units(CreditUnits::new(result.chat_balance));
    match locale {
        Locale::Es => format!(
            "Pasaste {amount} créditos al grupo\n\nTu saldo: {user_balance} créditos\nSaldo del grupo: {chat_balance} créditos"
        ),
        Locale::En => format!(
            "Moved {amount} credits to the group\n\nYour balance: {user_balance} credits\nGroup balance: {chat_balance} credits"
        ),
    }
}

/// Reply after sending credits to another user. Only the sender's balance is
/// shown: the reply is public and the recipient's balance is theirs.
#[must_use]
pub fn user_transfer_result_reply(
    amount: i64,
    recipient_name: &str,
    result: TransferResult,
    locale: Locale,
) -> String {
    if !result.transferred {
        return transfer_result_reply(amount, result, locale);
    }
    let amount = display_credit_units(CreditUnits::new(amount));
    let user_balance = display_credit_units(CreditUnits::new(result.user_balance));
    match locale {
        Locale::Es => format!(
            "Le pasaste {amount} créditos a {recipient_name}\n\nTu saldo: {user_balance} créditos"
        ),
        Locale::En => format!(
            "Sent {amount} credits to {recipient_name}\n\nYour balance: {user_balance} credits"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TransferCommandContext, TransferCommandPlan, TransferRecipient, TransferResult,
        plan_transfer_command, transfer_result_reply, user_transfer_result_reply,
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
            recipient: None,
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
            "AI credits are unavailable right now. Try again later or tell the admin"
        );

        let mut private = context(Locale::Es);
        private.is_group = false;
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", private)),
            "Esto es para grupos, capo. Usalo ahí: /transfer <monto> se lo pasa al grupo, o respondé a alguien para pasárselo a esa persona"
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
            "Mandalo así: /transfer <monto>\nEjemplo: /transfer 1.5\nRespondé a alguien para pasárselo a esa persona"
        );
        assert_eq!(
            reply_text(plan_transfer_command(
                "/transfer -1",
                "",
                context(Locale::En)
            )),
            "The amount must be greater than 0"
        );
    }

    #[test]
    fn replying_to_a_person_sends_the_credits_to_them() {
        let to = |user_id, is_bot| TransferCommandContext {
            recipient: Some(TransferRecipient { user_id, is_bot }),
            ..context(Locale::Es)
        };
        assert_eq!(
            plan_transfer_command("/transfer 1.5", "", to(77, false)),
            TransferCommandPlan::TransferToUser {
                user_id: 55,
                recipient_id: 77,
                amount: 150,
            }
        );
        // Amount checks still come first.
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 0", "", to(77, false))),
            "El monto tiene que ser mayor a 0, no me rompas las bolas"
        );
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", to(55, false))),
            "No te podés pasar créditos a vos mismo"
        );
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", to(77, true))),
            "Los bots no usan créditos, pasáselos a una persona"
        );
        let english = |user_id, is_bot| TransferCommandContext {
            recipient: Some(TransferRecipient { user_id, is_bot }),
            ..context(Locale::En)
        };
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", english(55, false))),
            "You can't send credits to yourself"
        );
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", english(1, true))),
            "Bots can't use credits. Send them to a person"
        );
    }

    #[test]
    fn user_transfer_replies_show_only_the_senders_balance() {
        let sent = TransferResult {
            transferred: true,
            user_balance: 70,
            chat_balance: 9_999,
        };
        assert_eq!(
            user_transfer_result_reply(150, "Ana", sent, Locale::Es),
            "Le pasaste 1.50 créditos a Ana\n\nTu saldo: 0.70 créditos"
        );
        assert_eq!(
            user_transfer_result_reply(150, "@ana", sent, Locale::En),
            "Sent 1.50 credits to @ana\n\nYour balance: 0.70 credits"
        );
        let short = TransferResult {
            transferred: false,
            ..sent
        };
        assert_eq!(
            user_transfer_result_reply(150, "Ana", short, Locale::En),
            "Not enough personal balance: you have 0.70 credits\nTry a smaller amount or add credits with /topup"
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
            "Pasaste 0.10 créditos al grupo\n\nTu saldo: 2.85 créditos\nSaldo del grupo: 12.15 créditos"
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
            "Moved 1.50 credits to the group\n\nYour balance: 0.70 credits\nGroup balance: 2.30 credits"
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
            "No te alcanza el saldo personal: tenés 0.70 créditos\nProbá con un monto menor o cargá con /topup"
        );
    }

    #[test]
    fn transfer_guards_are_localized_in_the_other_language() {
        let private = TransferCommandContext {
            is_group: false,
            ..context(Locale::En)
        };
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", private)),
            "This command is for groups. Use it there: /transfer <amount> moves credits to the group, or reply to someone to send them to that person"
        );
        let anonymous = TransferCommandContext {
            user_id: None,
            ..context(Locale::Es)
        };
        assert_eq!(
            reply_text(plan_transfer_command("/transfer 1", "", anonymous)),
            "No pude identificar tu usuario o el grupo para transferir"
        );
        assert_eq!(
            reply_text(plan_transfer_command("/transfer", "", context(Locale::En))),
            "Usage: /transfer <amount>\nExample: /transfer 1.5\nReply to someone to send them the credits"
        );
        assert_eq!(
            reply_text(plan_transfer_command(
                "/transfer 0",
                "",
                context(Locale::En)
            )),
            "The amount must be greater than 0"
        );
        // A transfer plan carries no reply text.
        assert_eq!(
            reply_text(plan_transfer_command(
                "/transfer 1",
                "",
                context(Locale::En)
            )),
            ""
        );
        assert_eq!(
            transfer_result_reply(
                100,
                TransferResult {
                    transferred: false,
                    user_balance: 50,
                    chat_balance: 0,
                },
                Locale::Es
            ),
            "No te alcanza el saldo personal: tenés 0.50 créditos\nProbá con un monto menor o cargá con /topup"
        );
    }

    #[test]
    fn remaining_transfer_messages_are_localized() {
        assert_eq!(
            reply_text(plan_transfer_command(
                "/transfer 0",
                "",
                context(Locale::Es)
            )),
            "El monto tiene que ser mayor a 0, no me rompas las bolas"
        );
        assert_eq!(
            transfer_result_reply(
                100,
                TransferResult {
                    transferred: false,
                    user_balance: 50,
                    chat_balance: 0,
                },
                Locale::En
            ),
            "Not enough personal balance: you have 0.50 credits\nTry a smaller amount or add credits with /topup"
        );
    }
}
