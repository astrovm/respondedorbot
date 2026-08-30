//! Pure planning for the `/language` and `/idioma` commands.

use crate::chat_config::ChatConfig;
use crate::command_parsing::parse_command;
use crate::locale::Locale;
use crate::telegram_actions::{
    InlineKeyboardButton, InlineKeyboardMarkup, SendMessage, TelegramAction,
};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageCommandPlan {
    NotHandled,
    LegacyGroupRequired,
    Action {
        action: TelegramAction,
        updated_config: Option<ChatConfig>,
    },
}

fn language_keyboard() -> InlineKeyboardMarkup {
    InlineKeyboardMarkup {
        inline_keyboard: vec![vec![
            InlineKeyboardButton {
                text: "Español".to_owned(),
                url: None,
                callback_data: Some("cfg:language:es".to_owned()),
            },
            InlineKeyboardButton {
                text: "English".to_owned(),
                url: None,
                callback_data: Some("cfg:language:en".to_owned()),
            },
        ]],
    }
}

#[must_use]
pub fn plan_language_command(
    chat_id: ChatId,
    message_id: MessageId,
    message_text: &str,
    bot_name: &str,
    locale: Locale,
    config: &ChatConfig,
    is_group: bool,
) -> LanguageCommandPlan {
    let parsed = parse_command(message_text, bot_name);
    if !matches!(parsed.command.as_str(), "/language" | "/idioma") {
        return LanguageCommandPlan::NotHandled;
    }
    if is_group {
        return LanguageCommandPlan::LegacyGroupRequired;
    }

    let requested = parsed.message_text.trim().to_lowercase();
    let (text, updated_config) = if requested.is_empty() {
        let language = match locale {
            Locale::Es => "Español",
            Locale::En => "English",
        };
        let text = match locale {
            Locale::Es => format!("idioma actual: {language}"),
            Locale::En => format!("current language: {language}"),
        };
        (text, None)
    } else if !matches!(requested.as_str(), "es" | "en") {
        let text = match locale {
            Locale::Es => "mandalo bien: /language [es|en]",
            Locale::En => "usage: /language [es|en]",
        };
        (text.to_owned(), None)
    } else {
        let mut updated = config.clone();
        updated.language.clone_from(&requested);
        let text = if requested == "es" {
            "listo, ahora hablo en español"
        } else {
            "done, I will speak English now"
        };
        (text.to_owned(), Some(updated))
    };
    let mut message = SendMessage::new(chat_id, &text);
    message.reply_to_message_id = Some(message_id);
    message.reply_markup = Some(language_keyboard());
    LanguageCommandPlan::Action {
        action: TelegramAction::SendMessage(message),
        updated_config,
    }
}

#[cfg(test)]
mod tests {
    use super::{LanguageCommandPlan, plan_language_command};
    use crate::{
        chat_config::ChatConfig,
        locale::Locale,
        telegram_actions::TelegramAction,
        telegram_input::{ChatId, MessageId},
    };

    fn plan(text: &str, locale: Locale, is_group: bool) -> LanguageCommandPlan {
        plan_language_command(
            ChatId(1),
            MessageId(2),
            text,
            "@mybot",
            locale,
            &ChatConfig::default(),
            is_group,
        )
    }

    #[test]
    fn plans_current_usage_and_persisted_language_changes() {
        let cases = [
            ("/language", Locale::Es, "idioma actual: Español", None),
            ("/idioma nope", Locale::En, "usage: /language [es|en]", None),
            (
                "/language@mybot en",
                Locale::Es,
                "done, I will speak English now",
                Some("en"),
            ),
            (
                "/idioma ES",
                Locale::En,
                "listo, ahora hablo en español",
                Some("es"),
            ),
        ];
        for (input, locale, expected_text, expected_update) in cases {
            let planned = plan(input, locale, false);
            assert!(matches!(planned, LanguageCommandPlan::Action { .. }));
            let LanguageCommandPlan::Action {
                action: TelegramAction::SendMessage(message),
                updated_config,
            } = planned
            else {
                return;
            };
            assert_eq!(message.text, expected_text);
            assert_eq!(
                updated_config
                    .as_ref()
                    .map(|config| config.language.as_str()),
                expected_update
            );
            assert_eq!(
                message
                    .reply_markup
                    .as_ref()
                    .and_then(|markup| markup.inline_keyboard.first())
                    .map(Vec::len),
                Some(2)
            );
        }
    }

    #[test]
    fn leaves_groups_and_other_commands_for_their_owners() {
        assert_eq!(
            plan("/language en", Locale::Es, true),
            LanguageCommandPlan::LegacyGroupRequired
        );
        assert_eq!(
            plan("/other", Locale::Es, false),
            LanguageCommandPlan::NotHandled
        );
    }
}
