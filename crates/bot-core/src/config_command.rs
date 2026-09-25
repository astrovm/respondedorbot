//! Typed rendering and action planning for chat configuration.

use crate::chat_config::ChatConfig;
use crate::command_parsing::parse_command;
use crate::locale::Locale;
use crate::menu_ui::{back, button, close, localized};
use crate::telegram_actions::{
    InlineKeyboardButton, InlineKeyboardMarkup, SendMessage, TelegramAction,
};
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

fn on_off_text(enabled: bool, locale: Locale) -> &'static str {
    localized(
        locale,
        if enabled { "Activado" } else { "Desactivado" },
        if enabled { "On" } else { "Off" },
    )
}

/// Explicit on/off buttons, so the selected state never reads as an action.
fn toggle_buttons(action: &str, enabled: bool, locale: Locale) -> Vec<InlineKeyboardButton> {
    vec![
        button(
            selected_label(enabled, on_off_text(true, locale)),
            format!("cfg:{action}:on"),
        ),
        button(
            selected_label(!enabled, on_off_text(false, locale)),
            format!("cfg:{action}:off"),
        ),
    ]
}

fn link_mode_text(mode: &str, locale: Locale) -> &'static str {
    match mode {
        "delete" => localized(locale, "Reemplazar original", "Replace original"),
        "off" => on_off_text(false, locale),
        _ => localized(locale, "Responder con link", "Reply with link"),
    }
}

fn creditless_text(limit: i64) -> String {
    if limit < 0 {
        "∞".to_owned()
    } else {
        limit.to_string()
    }
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
        ["reply", "delete", "off"]
            .into_iter()
            .map(|mode| {
                button(
                    selected_label(config.link_mode == mode, link_mode_text(mode, locale)),
                    format!("cfg:link:{mode}"),
                )
            })
            .collect(),
        toggle_buttons("followups", config.ai_command_followups, locale),
        toggle_buttons("linkfixfollowups", config.ignore_link_fix_followups, locale),
        vec![
            button(
                "-1 h".to_owned(),
                format!(
                    "cfg:timezone:{}",
                    config
                        .timezone_offset
                        .saturating_sub(1)
                        .max(TIMEZONE_OFFSET_MIN)
                ),
            ),
            button(
                offset_text(config.timezone_offset),
                "cfg:timezone:current".to_owned(),
            ),
            button(
                "+1 h".to_owned(),
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
        rows.push(toggle_buttons("random", config.ai_random_replies, locale));
        rows.push(vec![
            button("0".to_owned(), "cfg:creditless:none".to_owned()),
            button("-1".to_owned(), "cfg:creditless:decrease".to_owned()),
            button(
                format!("{}/h", creditless_text(config.creditless_user_hourly_limit)),
                "cfg:creditless:current".to_owned(),
            ),
            button("+1".to_owned(), "cfg:creditless:increase".to_owned()),
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

fn descriptions(locale: Locale) -> [&'static str; 7] {
    match locale {
        Locale::Es => [
            "El idioma de mis mensajes, menús y respuestas.",
            "Qué hago cuando alguien manda un link de X, Bluesky, Instagram o Reddit.\n\n• Responder con link: contesto con el link arreglado.\n• Reemplazar original: borro el mensaje y lo vuelvo a mandar arreglado.\n• Desactivado: no toco los links.",
            "Si respondés a una respuesta mía de un comando, sigo la conversación sin que tengas que usar /ask.",
            "Si alguien responde a un link que arreglé, no lo tomo como una pregunta para mí.",
            "La hora que uso para tareas, recordatorios y fechas.",
            "De vez en cuando me meto en la charla del grupo aunque nadie me llame.",
            "Cuántos mensajes de IA por hora puede usar cada persona con el saldo del grupo.\n\n0 = nadie, ∞ = sin límite",
        ],
        Locale::En => [
            "The language of my messages, menus and replies.",
            "What I do when someone sends an X, Bluesky, Instagram or Reddit link.\n\n• Reply with link: I reply with the fixed link.\n• Replace original: I delete the message and send it again, fixed.\n• Off: I leave links alone.",
            "When you reply to one of my command answers, I keep the conversation going without /ask.",
            "When someone replies to a link I fixed, I do not treat it as a question for me.",
            "The time I use for tasks, reminders and dates.",
            "Every now and then I join the group conversation without being called.",
            "How many AI messages per hour each person can use from the group balance.\n\n0 = nobody, ∞ = no limit",
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
        return (
            localized(
                locale,
                "Cómo funciona\n\nCada cambio se guarda al instante. En grupos, solo los admins pueden cambiar la configuración.",
                "How it works\n\nEvery change is saved right away. In groups, only admins can change settings.",
            )
            .to_owned(),
            InlineKeyboardMarkup {
                inline_keyboard: vec![vec![
                    back(locale, "cfg:page:home"),
                    close(locale, "cfg:page:close"),
                ]],
            },
        );
    }
    let names = titles(locale);
    let index = PAGES
        .iter()
        .position(|name| *name == page)
        .filter(|index| is_group || *index < 5);
    if let Some(index) = index {
        let descriptions = descriptions(locale);
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
    let values = [
        language.to_owned(),
        link_mode_text(&config.link_mode, locale).to_owned(),
        on_off_text(config.ai_command_followups, locale).to_owned(),
        on_off_text(config.ignore_link_fix_followups, locale).to_owned(),
        offset_text(config.timezone_offset),
        on_off_text(config.ai_random_replies, locale).to_owned(),
        creditless_text(config.creditless_user_hourly_limit),
    ];
    let mut rows = (0..if is_group { 7 } else { 5 })
        .map(|i| {
            vec![button(
                format!("{}: {}", names[i], values[i]),
                format!("cfg:page:{}", PAGES[i]),
            )]
        })
        .collect::<Vec<_>>();
    rows.push(vec![
        button(localized(locale, "Ayuda", "Help"), "cfg:page:help"),
        close(locale, "cfg:page:close"),
    ]);
    (
        if is_group {
            localized(locale, "Configuración del grupo", "Group settings")
        } else {
            localized(locale, "Configuración", "Settings")
        }
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
