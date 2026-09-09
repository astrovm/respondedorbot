//! Typed rendering and action planning for chat configuration.

use crate::chat_config::ChatConfig;
use crate::command_parsing::parse_command;
use crate::locale::Locale;
use crate::menu_ui::{back, button, close, localized};
use crate::telegram_actions::{InlineKeyboardMarkup, SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

pub const TIMEZONE_OFFSET_MIN: i64 = -12;
pub const TIMEZONE_OFFSET_MAX: i64 = 14;

fn offset_text(offset: i64) -> String {
    match offset.cmp(&0) {
        std::cmp::Ordering::Equal => "UTC".to_owned(),
        std::cmp::Ordering::Greater => format!("UTC+{offset}"),
        std::cmp::Ordering::Less => format!("UTC{offset}"),
    }
}

fn selected_label(selected: bool, label: &str) -> String {
    if selected {
        format!("✓ {label}")
    } else {
        label.to_owned()
    }
}

fn toggle_label(enabled: bool, locale: Locale) -> String {
    selected_label(
        enabled,
        localized(
            locale,
            if enabled { "Activado" } else { "Desactivado" },
            if enabled { "On" } else { "Off" },
        ),
    )
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
        Locale::Es => ("Responder con link", "Reemplazar original", "Desactivado"),
        Locale::En => ("Reply with link", "Replace original", "off"),
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
            toggle_label(config.ai_command_followups, locale),
            "cfg:followups:toggle".to_owned(),
        )],
        vec![button(
            toggle_label(config.ignore_link_fix_followups, locale),
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
            toggle_label(config.ai_random_replies, locale),
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

const PAGES: [&str; 7] = [
    "language",
    "link",
    "followups",
    "linkfixfollowups",
    "timezone",
    "random",
    "creditless",
];

fn titles(locale: Locale) -> [&'static str; 7] {
    match locale {
        Locale::Es => [
            "Idioma",
            "Links",
            "Seguir conversaciones",
            "Ignorar replies a links",
            "Zona horaria",
            "Respuestas random",
            "Mensajes gratis por hora",
        ],
        Locale::En => [
            "Language",
            "Links",
            "Follow conversations",
            "Ignore replies to links",
            "Timezone",
            "Random replies",
            "Free messages per hour",
        ],
    }
}

#[must_use]
pub fn render_config(
    config: &ChatConfig,
    locale: Locale,
    is_group: bool,
) -> (String, InlineKeyboardMarkup) {
    render_config_page(config, locale, is_group, "home")
}

#[must_use]
pub fn render_config_page(
    config: &ChatConfig,
    locale: Locale,
    is_group: bool,
    page: &str,
) -> (String, InlineKeyboardMarkup) {
    if page == "help" {
        return (localized(locale, "Configuración\n\nLos cambios se guardan al tocar una opción. En grupos, solo los admins pueden cambiarlos.", "Settings\n\nChanges are saved when you choose an option. In groups, only admins can change settings.").to_owned(), InlineKeyboardMarkup { inline_keyboard: vec![vec![back(locale, "cfg:page:home"), close(locale, "cfg:page:close")]] });
    }
    let names = titles(locale);
    let index = PAGES
        .iter()
        .position(|name| *name == page)
        .filter(|index| is_group || *index < 5);
    if let Some(index) = index {
        let descriptions = match locale {
            Locale::Es => [
                "Idioma de mis mensajes y respuestas.",
                "Qué hago con los links compatibles.",
                "Sigo la conversación cuando respondés a un comando.",
                "Ignoro respuestas normales a links que arreglé.",
                "Hora usada en comandos y tareas.",
                "A veces respondo en el grupo aunque nadie me llame.",
                "Mensajes de IA que paga el grupo por usuario, por hora.",
            ],
            Locale::En => [
                "Language used for my messages and responses.",
                "What I do with supported links.",
                "Continue the conversation when you reply to a command.",
                "Ignore normal replies to links I fixed.",
                "Time used in commands and tasks.",
                "Sometimes join group conversations without being called.",
                "AI messages paid by the group per user, per hour.",
            ],
        };
        let options = render_config_keyboard(config, locale, is_group)
            .inline_keyboard
            .remove(index);
        let mut rows = if index == 1 {
            options.into_iter().map(|b| vec![b]).collect::<Vec<_>>()
        } else {
            vec![options]
        };
        rows.push(vec![
            back(locale, "cfg:page:home"),
            close(locale, "cfg:page:close"),
        ]);
        return (
            format!("{}\n\n{}", names[index], descriptions[index]),
            InlineKeyboardMarkup {
                inline_keyboard: rows,
            },
        );
    }
    let language = match config.language.as_str() {
        "en" => "English",
        "es" => "Español",
        _ => localized(locale, "Español", "English"),
    };
    let yes_no = |value| {
        localized(
            locale,
            if value { "Sí" } else { "No" },
            if value { "On" } else { "Off" },
        )
    };
    let links = match config.link_mode.as_str() {
        "delete" => localized(locale, "Reemplazar original", "Replace original"),
        "off" => localized(locale, "Desactivados", "Off"),
        _ => localized(locale, "Responder con link", "Reply with link"),
    };
    let values = [
        language.to_owned(),
        links.to_owned(),
        yes_no(config.ai_command_followups).to_owned(),
        yes_no(config.ignore_link_fix_followups).to_owned(),
        offset_text(config.timezone_offset),
        yes_no(config.ai_random_replies).to_owned(),
        if config.creditless_user_hourly_limit < 0 {
            "∞".to_owned()
        } else {
            config.creditless_user_hourly_limit.to_string()
        },
    ];
    let mut rows = (0..if is_group { 7 } else { 5 })
        .map(|i| {
            vec![button(
                format!("{} · {}", names[i], values[i]),
                format!("cfg:page:{}", PAGES[i]),
            )]
        })
        .collect::<Vec<_>>();
    rows.push(vec![
        button(localized(locale, "Ayuda", "Help"), "cfg:page:help"),
        close(locale, "cfg:page:close"),
    ]);
    (
        localized(
            locale,
            "⚙️ Configuración\n\nElegí qué querés cambiar.",
            "⚙️ Settings\n\nChoose a setting to change.",
        )
        .to_owned(),
        InlineKeyboardMarkup {
            inline_keyboard: rows,
        },
    )
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
) -> Option<TelegramAction> {
    let parsed = parse_command(message_text, bot_name);
    if !matches!(
        parsed.command.as_str(),
        "/config" | "/configs" | "/settings"
    ) {
        return None;
    }
    let (text, reply_markup) = render_config(config, locale, is_group);
    let mut message = SendMessage::new(chat_id, &text);
    message.reply_to_message_id = Some(message_id);
    message.reply_markup = Some(reply_markup);
    Some(TelegramAction::SendMessage(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_is_compact_and_shows_current_values_only_in_buttons() {
        for locale in [Locale::Es, Locale::En] {
            for group in [false, true] {
                let (text, keyboard) = render_config(&ChatConfig::default(), locale, group);
                assert!(text.lines().count() <= 3);
                assert!(!text.contains("UTC"));
                assert_eq!(keyboard.inline_keyboard.len(), if group { 8 } else { 6 });
                assert!(keyboard.inline_keyboard[4][0].text.contains("UTC-3"));
                assert_eq!(
                    keyboard.inline_keyboard[0][0].callback_data.as_deref(),
                    Some("cfg:page:language")
                );
                assert_eq!(
                    keyboard.inline_keyboard[keyboard.inline_keyboard.len() - 1][1]
                        .callback_data
                        .as_deref(),
                    Some("cfg:page:close")
                );
                for page in PAGES.iter().take(if group { 7 } else { 5 }) {
                    let (detail, keyboard) =
                        render_config_page(&ChatConfig::default(), locale, group, page);
                    assert!(!detail.contains("Elegí"));
                    assert!(
                        keyboard
                            .inline_keyboard
                            .iter()
                            .flatten()
                            .any(|b| b.callback_data.as_deref() == Some("cfg:page:home"))
                    );
                }
            }
        }
    }

    #[test]
    fn details_preserve_selected_options_and_timezone_bounds() {
        let config = ChatConfig {
            link_mode: "delete".to_owned(),
            timezone_offset: TIMEZONE_OFFSET_MAX,
            creditless_user_hourly_limit: -1,
            ..ChatConfig::default()
        };
        let (_, links) = render_config_page(&config, Locale::En, true, "link");
        assert!(links.inline_keyboard[1][0].text.starts_with("✓"));
        let (_, timezone) = render_config_page(&config, Locale::En, true, "timezone");
        assert_eq!(
            timezone.inline_keyboard[0][2].callback_data.as_deref(),
            Some("cfg:timezone:14")
        );
        let (_, home) = render_config(&config, Locale::En, true);
        assert!(home.inline_keyboard[6][0].text.ends_with('∞'));
        let config = ChatConfig {
            timezone_offset: TIMEZONE_OFFSET_MIN,
            ..config
        };
        let (_, timezone) = render_config_page(&config, Locale::Es, false, "timezone");
        assert_eq!(
            timezone.inline_keyboard[0][0].callback_data.as_deref(),
            Some("cfg:timezone:-12")
        );
    }

    #[test]
    fn unavailable_group_pages_return_home() {
        let config = ChatConfig::default();
        for page in ["random", "creditless", "unknown"] {
            assert_eq!(
                render_config_page(&config, Locale::En, false, page),
                render_config(&config, Locale::En, false)
            );
        }
    }

    #[test]
    fn command_aliases_reply_to_original_message() {
        for command in ["/config", "/configs", "/settings@mybot"] {
            let Some(TelegramAction::SendMessage(message)) = plan_config_command(
                ChatId(1),
                MessageId(2),
                command,
                "@mybot",
                Locale::Es,
                &ChatConfig::default(),
                true,
            ) else {
                unreachable!("missing config")
            };
            assert_eq!(message.reply_to_message_id, Some(MessageId(2)));
            assert!(message.reply_markup.is_some());
        }
        assert!(
            plan_config_command(
                ChatId(1),
                MessageId(2),
                "/other",
                "@mybot",
                Locale::Es,
                &ChatConfig::default(),
                true
            )
            .is_none()
        );
    }
}
