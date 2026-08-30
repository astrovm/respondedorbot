//! Native plans for commands that require no external state.

use crate::base_conversion::{BaseConversion, convert_base};
use crate::command_parsing::parse_command;
use crate::locale::Locale;
use crate::telegram_actions::{SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatelessCommandPlan {
    NotHandled,
    LegacyFallbackRequired,
    Action(TelegramAction),
}

fn render_base_conversion(result: &BaseConversion, locale: Locale) -> String {
    match (locale, result) {
        (
            Locale::Es,
            BaseConversion::Success {
                number,
                source,
                result,
                target,
            },
        ) => format!("ahi tenes boludo, {number} en base {source} es {result} en base {target}"),
        (
            Locale::En,
            BaseConversion::Success {
                number,
                source,
                result,
                target,
            },
        ) => format!("{number} in base {source} is {result} in base {target}"),
        (Locale::Es, BaseConversion::Usage) => {
            "capo mandate algo como /convertbase 101, 2, 10 y te paso de binario a decimal"
                .to_owned()
        }
        (Locale::En, BaseConversion::Usage) => {
            "use /convertbase 101, 2, 10 to convert binary to decimal".to_owned()
        }
        (Locale::Es, BaseConversion::AlphanumericRequired) => {
            "el numero tiene que ser alfanumerico boludo".to_owned()
        }
        (Locale::En, BaseConversion::AlphanumericRequired) => {
            "the number must be alphanumeric".to_owned()
        }
        (Locale::Es, BaseConversion::SourceRange { input }) => {
            format!("base origen '{input}' tiene que ser entre 2 y 36 gordo")
        }
        (Locale::En, BaseConversion::SourceRange { input }) => {
            format!("source base '{input}' must be between 2 and 36")
        }
        (Locale::Es, BaseConversion::TargetRange { input }) => {
            format!("base destino '{input}' tiene que ser entre 2 y 36 boludo")
        }
        (Locale::En, BaseConversion::TargetRange { input }) => {
            format!("target base '{input}' must be between 2 and 36")
        }
        (Locale::Es, BaseConversion::NumbersRequired) => {
            "mandate numeros posta gordo, no me hagas perder el tiempo".to_owned()
        }
        (Locale::En, BaseConversion::NumbersRequired) => "send valid numbers".to_owned(),
    }
}

#[must_use]
pub fn plan_stateless_command(
    chat_id: ChatId,
    message_id: MessageId,
    message_text: &str,
    bot_name: &str,
    locale: Locale,
) -> StatelessCommandPlan {
    let parsed = parse_command(message_text, bot_name);
    if parsed.command != "/convertbase" {
        return StatelessCommandPlan::NotHandled;
    }
    let Ok(result) = convert_base(&parsed.message_text) else {
        return StatelessCommandPlan::LegacyFallbackRequired;
    };
    let mut message = SendMessage::new(chat_id, &render_base_conversion(&result, locale));
    message.reply_to_message_id = Some(message_id);
    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
}

#[cfg(test)]
mod tests {
    use super::{StatelessCommandPlan, plan_stateless_command};
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    fn message_text(plan: StatelessCommandPlan) -> Option<String> {
        match plan {
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message)) => {
                Some(message.text)
            }
            StatelessCommandPlan::NotHandled
            | StatelessCommandPlan::LegacyFallbackRequired
            | StatelessCommandPlan::Action(_) => None,
        }
    }

    #[test]
    fn plans_spanish_and_english_base_conversion_replies() {
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(42),
                MessageId(7),
                "/convertbase@mybot 101, 2, 10",
                "@mybot",
                Locale::Es,
            )),
            Some("ahi tenes boludo, 101 en base 2 es 5 en base 10".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(42),
                MessageId(7),
                "/convertbase 101, 2, 10",
                "@mybot",
                Locale::En,
            )),
            Some("101 in base 2 is 5 in base 10".to_owned())
        );
    }

    #[test]
    fn plans_every_localized_validation_response() {
        let cases = [
            (
                "bad",
                "use /convertbase 101, 2, 10 to convert binary to decimal",
            ),
            ("10!,2,10", "the number must be alphanumeric"),
            ("10,1,10", "source base '1' must be between 2 and 36"),
            ("10,2,40", "target base '40' must be between 2 and 36"),
            ("10,no,2", "send valid numbers"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                message_text(plan_stateless_command(
                    ChatId(1),
                    MessageId(2),
                    &format!("/convertbase {input}"),
                    "@bot",
                    Locale::En,
                )),
                Some(expected.to_owned())
            );
        }
    }

    #[test]
    fn unknown_commands_are_ignored_and_unicode_uses_legacy_fallback() {
        assert_eq!(
            plan_stateless_command(ChatId(1), MessageId(2), "/other value", "@bot", Locale::Es,),
            StatelessCommandPlan::NotHandled
        );
        assert_eq!(
            plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/convertbase １２, 10, 2",
                "@bot",
                Locale::Es,
            ),
            StatelessCommandPlan::LegacyFallbackRequired
        );
    }

    #[test]
    fn reply_action_preserves_chat_and_message_identity() {
        let plan = plan_stateless_command(
            ChatId(-10042),
            MessageId(77),
            "/convertbase 1,2,10",
            "@bot",
            Locale::Es,
        );
        assert!(matches!(
            &plan,
            StatelessCommandPlan::Action(TelegramAction::SendMessage(_))
        ));
        let StatelessCommandPlan::Action(TelegramAction::SendMessage(message)) = plan else {
            return;
        };
        assert_eq!(message.chat_id, ChatId(-10042));
        assert_eq!(message.reply_to_message_id, Some(MessageId(77)));
    }
}
