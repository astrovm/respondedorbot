//! Typed rendering and action planning for chat configuration.

use crate::chat_config::ChatConfig;
use crate::command_parsing::parse_command;
use crate::locale::Locale;
use crate::telegram_actions::{
    InlineKeyboardButton, InlineKeyboardMarkup, SendMessage, TelegramAction,
};
use crate::telegram_input::{ChatId, MessageId};

pub const TIMEZONE_OFFSET_MIN: i64 = -12;
pub const TIMEZONE_OFFSET_MAX: i64 = 14;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigCommandPlan {
    NotHandled,
    Action(TelegramAction),
}

fn offset_text(offset: i64) -> String {
    match offset.cmp(&0) {
        std::cmp::Ordering::Equal => "UTC".to_owned(),
        std::cmp::Ordering::Greater => format!("UTC+{offset}"),
        std::cmp::Ordering::Less => format!("UTC{offset}"),
    }
}

fn enabled_text(enabled: bool, locale: Locale) -> String {
    let state = match (enabled, locale) {
        (true, Locale::Es) => "activado",
        (false, Locale::Es) => "desactivado",
        (true, Locale::En) => "on",
        (false, Locale::En) => "off",
    };
    format!("{} {state}", if enabled { "✅" } else { "▫️" })
}

fn render_config_text(config: &ChatConfig, locale: Locale, is_group: bool) -> String {
    let language = match locale {
        Locale::Es => "Español",
        Locale::En => "English",
    };
    let link_mode = match (locale, config.link_mode.as_str()) {
        (Locale::Es, "delete") => "borro el original y reposteo el link arreglado",
        (Locale::Es, "off") => "no toco los links",
        (Locale::Es, _) => "respondo con el link arreglado",
        (Locale::En, "delete") => "Delete the original and repost the fixed link",
        (Locale::En, "off") => "Do not modify links",
        (Locale::En, _) => "Reply with the fixed link",
    };
    let unavailable = match locale {
        Locale::Es => "solo disponible en grupos",
        Locale::En => "only available in groups",
    };
    let random = if is_group {
        enabled_text(config.ai_random_replies, locale)
    } else {
        unavailable.to_owned()
    };
    let creditless = if is_group {
        if config.creditless_user_hourly_limit < 0 {
            "∞".to_owned()
        } else {
            config.creditless_user_hourly_limit.to_string()
        }
    } else {
        unavailable.to_owned()
    };
    match locale {
        Locale::Es => format!(
            "config del gordo\n\n1. idioma\nidioma de los mensajes y respuestas del bot\n{language}\n\n2. links arreglados\nqué hago con links compatibles\n{link_mode}\n\n3. seguir charla en comandos\nsigo la conversación cuando respondés a un comando\n{}\n\n4. ignorar replies a links arreglados\nignoro respuestas normales a links que arreglé\n{}\n\n5. zona horaria\nhora usada en comandos y tareas\n{}\n\n6. respuestas random\na veces respondo en el grupo aunque nadie me llame\n{random}\n\n7. mensajes gratis por usuario por hora\nmensajes de IA que paga el grupo para cada usuario\n{creditless}\n\ntocá los botones de abajo para cambiar la config",
            enabled_text(config.ai_command_followups, locale),
            enabled_text(config.ignore_link_fix_followups, locale),
            offset_text(config.timezone_offset),
        ),
        Locale::En => format!(
            "Bot settings\n\n1. Language\nLanguage used for bot messages and responses\n{language}\n\n2. Fixed links\nWhat I do with supported links\n{link_mode}\n\n3. Command follow-ups\nContinue the conversation when you reply to a command\n{}\n\n4. Ignore replies to fixed links\nIgnore normal replies to links I fixed\n{}\n\n5. Timezone\nTime used in commands and tasks\n{}\n\n6. Random replies\nSometimes join group conversations without being called\n{random}\n\n7. Free messages per user per hour\nAI messages paid by the group for each user\n{creditless}\n\nUse the buttons below to change the settings",
            enabled_text(config.ai_command_followups, locale),
            enabled_text(config.ignore_link_fix_followups, locale),
            offset_text(config.timezone_offset),
        ),
    }
}

fn button(text: String, callback_data: String) -> InlineKeyboardButton {
    InlineKeyboardButton {
        text,
        url: None,
        callback_data: Some(callback_data),
    }
}

fn selected_label(selected: bool, label: &str) -> String {
    format!("{} {label}", if selected { "✅" } else { "▫️" })
}

fn render_config_keyboard(
    config: &ChatConfig,
    locale: Locale,
    is_group: bool,
) -> InlineKeyboardMarkup {
    let language = match config.language.as_str() {
        "es" => Locale::Es,
        "en" => Locale::En,
        _ => locale,
    };
    let labels = match locale {
        Locale::Es => (
            "responder link",
            "borrar link",
            "apagado",
            "seguir charla",
            "ignorar replies",
            "me meto en la charla",
        ),
        Locale::En => (
            "reply with link",
            "replace link",
            "off",
            "command follow-ups",
            "ignore replies",
            "join conversations",
        ),
    };
    let mut rows = vec![
        vec![
            button(
                selected_label(language == Locale::Es, "Español"),
                "cfg:language:es".to_owned(),
            ),
            button(
                selected_label(language == Locale::En, "English"),
                "cfg:language:en".to_owned(),
            ),
        ],
        vec![
            button(
                selected_label(config.link_mode == "reply", labels.0),
                "cfg:link:reply".to_owned(),
            ),
            button(
                selected_label(config.link_mode == "delete", labels.1),
                "cfg:link:delete".to_owned(),
            ),
            button(
                selected_label(config.link_mode == "off", labels.2),
                "cfg:link:off".to_owned(),
            ),
        ],
        vec![button(
            selected_label(config.ai_command_followups, labels.3),
            "cfg:followups:toggle".to_owned(),
        )],
        vec![button(
            selected_label(config.ignore_link_fix_followups, labels.4),
            "cfg:linkfixfollowups:toggle".to_owned(),
        )],
        vec![
            button(
                "➖ 1h".to_owned(),
                format!(
                    "cfg:timezone:{}",
                    config
                        .timezone_offset
                        .saturating_sub(1)
                        .max(TIMEZONE_OFFSET_MIN)
                ),
            ),
            button(
                format!("🌍 {}", offset_text(config.timezone_offset)),
                "cfg:timezone:current".to_owned(),
            ),
            button(
                "➕ 1h".to_owned(),
                format!(
                    "cfg:timezone:{}",
                    config
                        .timezone_offset
                        .saturating_add(1)
                        .min(TIMEZONE_OFFSET_MAX)
                ),
            ),
        ],
    ];
    if is_group {
        rows.push(vec![button(
            selected_label(config.ai_random_replies, labels.5),
            "cfg:random:toggle".to_owned(),
        )]);
        rows.push(vec![
            button("0".to_owned(), "cfg:creditless:none".to_owned()),
            button("-".to_owned(), "cfg:creditless:decrease".to_owned()),
            button(
                if config.creditless_user_hourly_limit < 0 {
                    "∞".to_owned()
                } else {
                    config.creditless_user_hourly_limit.to_string()
                },
                "cfg:creditless:current".to_owned(),
            ),
            button("+".to_owned(), "cfg:creditless:increase".to_owned()),
            button("∞".to_owned(), "cfg:creditless:unlimited".to_owned()),
        ]);
    }
    InlineKeyboardMarkup {
        inline_keyboard: rows,
    }
}

#[must_use]
pub fn plan_config_command(
    chat_id: ChatId,
    message_id: MessageId,
    message_text: &str,
    bot_name: &str,
    locale: Locale,
    config: &ChatConfig,
    is_group: bool,
) -> ConfigCommandPlan {
    let parsed = parse_command(message_text, bot_name);
    if !matches!(
        parsed.command.as_str(),
        "/config" | "/configs" | "/settings"
    ) {
        return ConfigCommandPlan::NotHandled;
    }
    let mut message = SendMessage::new(chat_id, &render_config_text(config, locale, is_group));
    message.reply_to_message_id = Some(message_id);
    message.reply_markup = Some(render_config_keyboard(config, locale, is_group));
    ConfigCommandPlan::Action(TelegramAction::SendMessage(message))
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::{ConfigCommandPlan, TIMEZONE_OFFSET_MAX, plan_config_command};
    use crate::{
        chat_config::ChatConfig,
        locale::Locale,
        telegram_actions::TelegramAction,
        telegram_input::{ChatId, MessageId},
    };

    fn message(plan: ConfigCommandPlan) -> Option<crate::telegram_actions::SendMessage> {
        match plan {
            ConfigCommandPlan::Action(TelegramAction::SendMessage(message)) => Some(message),
            ConfigCommandPlan::NotHandled | ConfigCommandPlan::Action(_) => None,
        }
    }

    fn sort_json_keys(value: &mut Value) {
        match value {
            Value::Array(values) => values.iter_mut().for_each(sort_json_keys),
            Value::Object(map) => {
                let mut entries = std::mem::take(map).into_iter().collect::<Vec<_>>();
                entries.sort_by(|left, right| left.0.cmp(&right.0));
                for (_, value) in &mut entries {
                    sort_json_keys(value);
                }
                map.extend(entries);
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }

    fn parity_hash(message: &crate::telegram_actions::SendMessage) -> Option<String> {
        let mut value = serde_json::json!({
            "text": message.text,
            "reply_markup": message.reply_markup,
        });
        sort_json_keys(&mut value);
        let encoded = serde_json::to_string(&value).ok()?;
        let mut hash = String::with_capacity(64);
        for byte in Sha256::digest(encoded.as_bytes()) {
            if write!(&mut hash, "{byte:02x}").is_err() {
                return None;
            }
        }
        Some(hash)
    }

    #[test]
    fn spanish_group_defaults_match_text_and_seven_keyboard_rows() {
        let message = message(plan_config_command(
            ChatId(1),
            MessageId(2),
            "/configs",
            "@mybot",
            Locale::Es,
            &ChatConfig::default(),
            true,
        ));
        assert!(message.is_some());
        let Some(message) = message else { return };
        assert_eq!(
            message.text,
            "config del gordo\n\n1. idioma\nidioma de los mensajes y respuestas del bot\nEspañol\n\n2. links arreglados\nqué hago con links compatibles\nrespondo con el link arreglado\n\n3. seguir charla en comandos\nsigo la conversación cuando respondés a un comando\n✅ activado\n\n4. ignorar replies a links arreglados\nignoro respuestas normales a links que arreglé\n✅ activado\n\n5. zona horaria\nhora usada en comandos y tareas\nUTC-3\n\n6. respuestas random\na veces respondo en el grupo aunque nadie me llame\n✅ activado\n\n7. mensajes gratis por usuario por hora\nmensajes de IA que paga el grupo para cada usuario\n5\n\ntocá los botones de abajo para cambiar la config"
        );
        assert_eq!(
            parity_hash(&message).as_deref(),
            Some("c7f3bbd9a744addf3cfdf79e21a8c533069417caf6b21d4dd3e143ce8e5bb4eb")
        );
        assert_eq!(
            message
                .reply_markup
                .map(|markup| markup.inline_keyboard.len()),
            Some(7)
        );
    }

    #[test]
    fn english_private_hides_group_buttons_and_marks_values_unavailable() {
        let config = ChatConfig {
            language: "en".to_owned(),
            ..ChatConfig::default()
        };
        let message = message(plan_config_command(
            ChatId(1),
            MessageId(2),
            "/settings@mybot",
            "@mybot",
            Locale::En,
            &config,
            false,
        ));
        assert!(message.is_some());
        let Some(message) = message else { return };
        assert_eq!(
            message.text,
            "Bot settings\n\n1. Language\nLanguage used for bot messages and responses\nEnglish\n\n2. Fixed links\nWhat I do with supported links\nReply with the fixed link\n\n3. Command follow-ups\nContinue the conversation when you reply to a command\n✅ on\n\n4. Ignore replies to fixed links\nIgnore normal replies to links I fixed\n✅ on\n\n5. Timezone\nTime used in commands and tasks\nUTC-3\n\n6. Random replies\nSometimes join group conversations without being called\nonly available in groups\n\n7. Free messages per user per hour\nAI messages paid by the group for each user\nonly available in groups\n\nUse the buttons below to change the settings"
        );
        assert_eq!(
            parity_hash(&message).as_deref(),
            Some("b9039bb642d144fe2f2a28fceed429d515ee4e88805c47f0156f54154185ad51")
        );
        assert_eq!(
            message
                .reply_markup
                .map(|markup| markup.inline_keyboard.len()),
            Some(5)
        );
    }

    #[test]
    fn custom_values_render_bounds_disabled_states_and_unlimited_limit() {
        let config = ChatConfig {
            link_mode: "delete".to_owned(),
            ai_command_followups: false,
            ignore_link_fix_followups: false,
            timezone_offset: TIMEZONE_OFFSET_MAX,
            ai_random_replies: false,
            creditless_user_hourly_limit: -1,
            ..ChatConfig::default()
        };
        let message = message(plan_config_command(
            ChatId(1),
            MessageId(2),
            "/config",
            "@mybot",
            Locale::En,
            &config,
            true,
        ));
        assert!(message.is_some());
        let Some(message) = message else { return };
        assert!(
            message
                .text
                .contains("Delete the original and repost the fixed link")
        );
        assert!(message.text.contains("▫️ off"));
        assert!(message.text.contains("UTC+14"));
        assert!(message.text.contains('∞'));
        let next_timezone = message
            .reply_markup
            .and_then(|markup| markup.inline_keyboard.get(4).cloned())
            .and_then(|row| row.get(2).cloned())
            .and_then(|button| button.callback_data);
        assert_eq!(next_timezone.as_deref(), Some("cfg:timezone:14"));
    }

    #[test]
    fn ignores_unrelated_commands() {
        assert_eq!(
            plan_config_command(
                ChatId(1),
                MessageId(2),
                "/other",
                "@mybot",
                Locale::Es,
                &ChatConfig::default(),
                true,
            ),
            ConfigCommandPlan::NotHandled
        );
    }
}
