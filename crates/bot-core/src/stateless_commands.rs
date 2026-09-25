//! Native plans for commands that require no external state.

use crate::base_conversion::{BaseConversion, convert_base};
use crate::command_normalization::{normalize_command_text, preprocess_command_text};
use crate::command_parsing::parse_command;
use crate::help_catalog::render_help_page;
use crate::locale::Locale;
use crate::telegram_actions::{SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatelessCommandPlan {
    NotHandled,
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
        ) => format!("Ahí tenés, boludo: {number} en base {source} es {result} en base {target}"),
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
            "Capo, mandate algo como /convertbase 101, 2, 10 y te paso de binario a decimal"
                .to_owned()
        }
        (Locale::En, BaseConversion::Usage) => {
            "Use /convertbase 101, 2, 10 to convert binary to decimal".to_owned()
        }
        (Locale::Es, BaseConversion::AlphanumericRequired) => {
            "El número tiene que ser alfanumérico, boludo".to_owned()
        }
        (Locale::En, BaseConversion::AlphanumericRequired) => {
            "The number must be alphanumeric".to_owned()
        }
        (Locale::Es, BaseConversion::SourceRange { input }) => {
            format!("La base de origen '{input}' tiene que estar entre 2 y 36, gordo")
        }
        (Locale::En, BaseConversion::SourceRange { input }) => {
            format!("Source base '{input}' must be between 2 and 36")
        }
        (Locale::Es, BaseConversion::TargetRange { input }) => {
            format!("La base de destino '{input}' tiene que estar entre 2 y 36, boludo")
        }
        (Locale::En, BaseConversion::TargetRange { input }) => {
            format!("Target base '{input}' must be between 2 and 36")
        }
        (Locale::Es, BaseConversion::NumbersRequired) => {
            "Mandate números posta, gordo, no me hagas perder el tiempo".to_owned()
        }
        (Locale::En, BaseConversion::NumbersRequired) => "Send valid numbers".to_owned(),
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
    plan_stateless_command_with_reply(chat_id, message_id, message_text, None, bot_name, locale)
}

#[must_use]
pub fn plan_stateless_command_with_reply(
    chat_id: ChatId,
    message_id: MessageId,
    message_text: &str,
    replied_message_text: Option<&str>,
    bot_name: &str,
    locale: Locale,
) -> StatelessCommandPlan {
    let parsed = parse_command(message_text, bot_name);
    if parsed.command == "/help" {
        let (text, keyboard) = render_help_page(locale, "home");
        let mut message = SendMessage::new(chat_id, &text);
        message.reply_markup = Some(keyboard);
        message.reply_to_message_id = Some(message_id);
        return StatelessCommandPlan::Action(TelegramAction::SendMessage(message));
    }
    if matches!(parsed.command.as_str(), "/comando" | "/command") {
        let conversion_text = if parsed.message_text.is_empty() {
            replied_message_text.unwrap_or_default().trim()
        } else {
            &parsed.message_text
        };
        if conversion_text.is_empty() {
            let text = match locale {
                Locale::Es => "¿Y qué querés que convierta, boludo? Mandate texto",
                Locale::En => "Send the text you want to convert",
            };
            let mut message = SendMessage::new(chat_id, text);
            message.reply_to_message_id = Some(message_id);
            return StatelessCommandPlan::Action(TelegramAction::SendMessage(message));
        }
        let preprocessed = preprocess_command_text(conversion_text, locale);
        let text = normalize_command_text(&preprocessed).unwrap_or_else(|| match locale {
            Locale::Es => {
                "No me mandes giladas, boludo: tiene que tener letras o números".to_owned()
            }
            Locale::En => "The command must contain letters or numbers".to_owned(),
        });
        let mut message = SendMessage::new(chat_id, &text);
        message.reply_to_message_id = Some(message_id);
        return StatelessCommandPlan::Action(TelegramAction::SendMessage(message));
    }
    if parsed.command != "/convertbase" {
        return StatelessCommandPlan::NotHandled;
    }
    let Ok(result) = convert_base(&parsed.message_text) else {
        return StatelessCommandPlan::NotHandled;
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
        "/instance" => match (locale, context.instance_name) {
            (Locale::Es, Some(name)) => format!("Estoy corriendo en {name}, boludo"),
            (Locale::En, Some(name)) => format!("I am running on {name}"),
            (Locale::Es, None) => "No tengo nombre de instancia configurado".to_owned(),
            (Locale::En, None) => "This instance has no name configured".to_owned(),
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
        plan_stateless_command, plan_stateless_command_with_reply,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    fn message_text(plan: StatelessCommandPlan) -> Option<String> {
        match plan {
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message)) => {
                Some(message.text)
            }
            StatelessCommandPlan::NotHandled | StatelessCommandPlan::Action(_) => None,
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
            Some("Ahí tenés, boludo: 101 en base 2 es 5 en base 10".to_owned())
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
        assert!(spanish.is_some_and(|text| { text.starts_with("👋 Ayuda\n\n") }));
        let english = message_text(plan_stateless_command(
            ChatId(1),
            MessageId(2),
            "/help@bot",
            "@bot",
            Locale::En,
        ));
        assert!(english.is_some_and(|text| { text.starts_with("👋 Help\n\n") }));
    }

    #[test]
    fn plans_command_conversion_aliases_transliteration_and_localized_guards() {
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
            Some("¿Y qué querés que convierta, boludo? Mandate texto".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/command 💥",
                "@bot",
                Locale::En,
            )),
            Some("/COLLISION".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/comando 💥",
                "@bot",
                Locale::Es,
            )),
            Some("/COLISION".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/command テスト",
                "@bot",
                Locale::Es,
            )),
            Some("/TESUTO".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/comando Привет мир",
                "@bot",
                Locale::Es,
            )),
            Some("/PRIVET_MIR".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/command --",
                "@bot",
                Locale::En,
            )),
            Some("The command must contain letters or numbers".to_owned())
        );
    }

    #[test]
    fn command_conversion_uses_replied_text_only_when_inline_text_is_empty() {
        assert_eq!(
            message_text(plan_stateless_command_with_reply(
                ChatId(1),
                MessageId(2),
                "/comando@testbot",
                Some("quoted content"),
                "testbot",
                Locale::Es,
            )),
            Some("/QUOTED_CONTENT".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command_with_reply(
                ChatId(1),
                MessageId(2),
                "/command inline text",
                Some("quoted content"),
                "testbot",
                Locale::En,
            )),
            Some("/INLINE_TEXT".to_owned())
        );
        assert_eq!(
            message_text(plan_stateless_command_with_reply(
                ChatId(1),
                MessageId(2),
                "/command",
                Some("   "),
                "testbot",
                Locale::En,
            )),
            Some("Send the text you want to convert".to_owned())
        );
    }

    #[test]
    fn plans_every_localized_validation_response() {
        let cases = [
            (
                "bad",
                "Use /convertbase 101, 2, 10 to convert binary to decimal",
            ),
            ("10!,2,10", "The number must be alphanumeric"),
            ("10,1,10", "Source base '1' must be between 2 and 36"),
            ("10,2,40", "Target base '40' must be between 2 and 36"),
            ("10,no,2", "Send valid numbers"),
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
    fn unknown_commands_are_ignored_and_compatibility_digits_are_native() {
        assert_eq!(
            plan_stateless_command(ChatId(1), MessageId(2), "/other value", "@bot", Locale::Es,),
            StatelessCommandPlan::NotHandled
        );
        assert_eq!(
            message_text(plan_stateless_command(
                ChatId(1),
                MessageId(2),
                "/convertbase １２, 10, 2",
                "@bot",
                Locale::Es,
            )),
            Some("Ahí tenés, boludo: １２ en base 10 es 1100 en base 2".to_owned())
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
            Some("No tengo nombre de instancia configurado".to_owned())
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
