//! Native plans and replies for privileged billing commands.

use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, format_credit_units, parse_credit_units};
use crate::locale::Locale;
use crate::telegram_actions::{SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrintCreditsPlan {
    NotHandled,
    Reply(TelegramAction),
    Mint { user_id: i64, amount: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrintCreditsContext {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub user_id: i64,
    pub admin_user_id: Option<i64>,
    pub billing_available: bool,
    pub locale: Locale,
}

fn reply(chat_id: ChatId, message_id: MessageId, text: &str) -> PrintCreditsPlan {
    let mut message = SendMessage::new(chat_id, text);
    message.reply_to_message_id = Some(message_id);
    PrintCreditsPlan::Reply(TelegramAction::SendMessage(message))
}

#[must_use]
pub fn plan_printcredits_command(
    message_text: &str,
    bot_name: &str,
    context: PrintCreditsContext,
) -> PrintCreditsPlan {
    let parsed = parse_command(message_text, bot_name);
    if parsed.command != "/printcredits" {
        return PrintCreditsPlan::NotHandled;
    }
    if context.admin_user_id != Some(context.user_id) {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "este comando es solo para el admin",
                Locale::En => "this command is only for the admin",
            },
        );
    }
    if !context.billing_available {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "los créditos IA no están disponibles ahora",
                Locale::En => "AI credits are not available right now",
            },
        );
    }
    let amount_token = parsed
        .message_text
        .split_once(' ')
        .map_or(parsed.message_text.as_str(), |(token, _)| token)
        .trim();
    let Some(amount) = parse_credit_units(amount_token) else {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "mandalo bien: /printcredits <monto>",
                Locale::En => "usage: /printcredits <amount>",
            },
        );
    };
    if amount.value() <= 0 {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "el monto tiene que ser mayor a 0",
                Locale::En => "the amount must be greater than 0",
            },
        );
    }
    PrintCreditsPlan::Mint {
        user_id: context.user_id,
        amount: amount.value(),
    }
}

#[must_use]
pub fn printcredits_result_reply(amount: i64, balance: i64, locale: Locale) -> String {
    let amount = format_credit_units(CreditUnits::new(amount));
    let balance = format_credit_units(CreditUnits::new(balance));
    match locale {
        Locale::Es => format!("listo, te imprimí {amount} créditos\nte quedaron {balance}"),
        Locale::En => format!("minted {amount} credits\nyour balance is {balance}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PrintCreditsContext, PrintCreditsPlan, plan_printcredits_command, printcredits_result_reply,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    fn reply_text(plan: PrintCreditsPlan) -> Option<String> {
        match plan {
            PrintCreditsPlan::Reply(TelegramAction::SendMessage(message)) => Some(message.text),
            PrintCreditsPlan::NotHandled
            | PrintCreditsPlan::Mint { .. }
            | PrintCreditsPlan::Reply(_) => None,
        }
    }

    fn plan(text: &str, admin: Option<i64>, billing: bool, locale: Locale) -> PrintCreditsPlan {
        plan_printcredits_command(
            text,
            "@bot",
            PrintCreditsContext {
                chat_id: ChatId(202),
                message_id: MessageId(12),
                user_id: 99,
                admin_user_id: admin,
                billing_available: billing,
                locale,
            },
        )
    }

    #[test]
    fn preserves_authorization_billing_and_input_guard_order() {
        assert_eq!(
            reply_text(plan("/printcredits bad", None, false, Locale::Es)),
            Some("este comando es solo para el admin".to_owned())
        );
        assert_eq!(
            reply_text(plan("/printcredits bad", Some(99), false, Locale::En)),
            Some("AI credits are not available right now".to_owned())
        );
        assert_eq!(
            reply_text(plan("/printcredits bad", Some(99), true, Locale::Es)),
            Some("mandalo bien: /printcredits <monto>".to_owned())
        );
        assert_eq!(
            reply_text(plan("/printcredits -1", Some(99), true, Locale::En)),
            Some("the amount must be greater than 0".to_owned())
        );
    }

    #[test]
    fn parses_exact_credit_units_and_formats_bilingual_success() {
        assert_eq!(
            plan(
                "/printcredits@bot 100.0 ignored",
                Some(99),
                true,
                Locale::Es
            ),
            PrintCreditsPlan::Mint {
                user_id: 99,
                amount: 10_000,
            }
        );
        assert_eq!(
            printcredits_result_reply(10_000, 12_000, Locale::Es),
            "listo, te imprimí 100.00 créditos\nte quedaron 120.00"
        );
        assert_eq!(
            printcredits_result_reply(10_000, 12_000, Locale::En),
            "minted 100.00 credits\nyour balance is 120.00"
        );
        assert_eq!(
            plan("/other 1", Some(99), true, Locale::Es),
            PrintCreditsPlan::NotHandled
        );
    }
}
