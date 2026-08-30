//! Native plans for commands that require no external state.

use crate::base_conversion::{BaseConversion, convert_base};
use crate::command_normalization::normalize_command_text;
use crate::command_parsing::parse_command;
use crate::help_catalog::render_help_text;
use crate::locale::Locale;
use crate::telegram_actions::{SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatelessCommandPlan {
    NotHandled,
    LegacyFallbackRequired,
    Action(TelegramAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatelessRuntimeContext<'a> {
    pub unix_timestamp: i64,
    pub instance_name: Option<&'a str>,
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
    if parsed.command == "/help" {
        let mut message = SendMessage::new(chat_id, render_help_text(locale));
        message.reply_to_message_id = Some(message_id);
        return StatelessCommandPlan::Action(TelegramAction::SendMessage(message));
    }
    if matches!(parsed.command.as_str(), "/comando" | "/command") {
        if parsed.message_text.is_empty() {
            let text = match locale {
                Locale::Es => "y que queres que convierta boludo? mandate texto",
                Locale::En => "send the text you want to convert",
            };
            let mut message = SendMessage::new(chat_id, text);
            message.reply_to_message_id = Some(message_id);
            return StatelessCommandPlan::Action(TelegramAction::SendMessage(message));
        }
        // The legacy adapter first expands emoji with locale-specific names and
        // romanizes Japanese text. Keep non-ASCII input on that path until those
        // preprocessing contracts have native implementations.
        if !parsed.message_text.is_ascii() {
            return StatelessCommandPlan::LegacyFallbackRequired;
        }
        let text = normalize_command_text(&parsed.message_text).unwrap_or_else(|| match locale {
            Locale::Es => {
                "no me mandes giladas boludo, tiene que tener letras o numeros".to_owned()
            }
            Locale::En => "the command must contain letters or numbers".to_owned(),
        });
        let mut message = SendMessage::new(chat_id, &text);
        message.reply_to_message_id = Some(message_id);
        return StatelessCommandPlan::Action(TelegramAction::SendMessage(message));
    }
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

#[must_use]
pub fn plan_runtime_stateless_command(
    chat_id: ChatId,
    message_id: MessageId,
    message_text: &str,
    bot_name: &str,
    locale: Locale,
    context: StatelessRuntimeContext<'_>,
) -> StatelessCommandPlan {
    let parsed = parse_command(message_text, bot_name);
    let text = match parsed.command.as_str() {
        "/time" => context.unix_timestamp.to_string(),
        "/instance" => match locale {
            Locale::Es => format!(
                "estoy corriendo en {} boludo",
                context.instance_name.unwrap_or("None")
            ),
            Locale::En => format!(
                "I am running on {}",
                context.instance_name.unwrap_or("None")
            ),
        },
        _ => return StatelessCommandPlan::NotHandled,
    };
    let mut message = SendMessage::new(chat_id, &text);
    message.reply_to_message_id = Some(message_id);
    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
}

#[cfg(test)]
mod tests {
    use super::{
        StatelessCommandPlan, StatelessRuntimeContext, plan_runtime_stateless_command,
        plan_stateless_command,
    };
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
    fn plans_complete_localized_help_replies() {
        let spanish = message_text(plan_stateless_command(
            ChatId(1),
            MessageId(2),
            "/help",
            "@bot",
            Locale::Es,
        ));
        assert!(spanish.is_some_and(|text| {
            text.starts_with("esto es lo que sé hacer, boludo:") && text.contains("/transfer")
        }));
        let english = message_text(plan_stateless_command(
            ChatId(1),
            MessageId(2),
            "/help@bot",
            "@bot",
            Locale::En,
        ));
        assert!(english.is_some_and(|text| {
            text.starts_with("what I can do:") && text.contains("/weather London")
        }));
    }

    #[test]
    fn plans_ascii_command_conversion_aliases_and_localized_guards() {
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/command@bot hello! world? or... bye.",
                "@bot",
                Locale::En,
            )),
            Some(
                "/HELLO_SIGNODEEXCLAMACION_WORLD_SIGNODEPREGUNTA_OR_PUNTOSSUSPENSIVOS_BYE_PUNTO"
                    .to_owned()
            )
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/comando",
                "@bot",
                Locale::Es,
            )),
            Some("y que queres que convierta boludo? mandate texto".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/command 💥",
                "@bot",
                Locale::En,
            )),
            None
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/command --",
                "@bot",
                Locale::En,
            )),
            Some("the command must contain letters or numbers".to_owned())
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

    #[test]
    fn plans_time_and_localized_instance_without_reading_global_state() {
        assert_eq!(
            message_text(plan_runtime_stateless_command(
                ChatId(1),
                MessageId(2),
                "/time",
                "@bot",
                Locale::Es,
                StatelessRuntimeContext {
                    unix_timestamp: 1_672_531_200,
                    instance_name: Some("synthetic"),
                },
            )),
            Some("1672531200".to_owned())
        );
        assert_eq!(
            message_text(plan_runtime_stateless_command(
                ChatId(1),
                MessageId(2),
                "/instance@bot",
                "@bot",
                Locale::En,
                StatelessRuntimeContext {
                    unix_timestamp: 0,
                    instance_name: Some("test instance"),
                },
            )),
            Some("I am running on test instance".to_owned())
        );
        assert_eq!(
            message_text(plan_runtime_stateless_command(
                ChatId(1),
                MessageId(2),
                "/instance",
                "@bot",
                Locale::Es,
                StatelessRuntimeContext {
                    unix_timestamp: 0,
                    instance_name: None,
                },
            )),
            Some("estoy corriendo en None boludo".to_owned())
        );
    }

    #[test]
    fn runtime_planner_ignores_other_commands() {
        assert_eq!(
            plan_runtime_stateless_command(
                ChatId(1),
                MessageId(2),
                "/convertbase 1,2,10",
                "@bot",
                Locale::Es,
                StatelessRuntimeContext {
                    unix_timestamp: 0,
                    instance_name: None,
                },
            ),
            StatelessCommandPlan::NotHandled
        );
    }
}
