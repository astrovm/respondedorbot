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

fn payer_text(group_pays_first: bool, locale: Locale) -> &'static str {
    if group_pays_first {
        localized(locale, "Grupo", "Group")
    } else {
        localized(locale, "Cada uno", "Members")
    }
}

fn render_config_keyboard(
    config: &ChatConfig,
    locale: Locale,
    is_group: bool,
) -> InlineKeyboardMarkup {
    let mut rows = vec![
        vec![
            button(
                selected_label(config.language == "es", "Español"),
                "cfg:language:es".to_owned(),
            ),
            button(
                selected_label(config.language == "en", "English"),
                "cfg:language:en".to_owned(),
            ),
            button(
                selected_label(
                    !matches!(config.language.as_str(), "es" | "en"),
                    localized(locale, "Automático", "Automatic"),
                ),
                "cfg:language:auto".to_owned(),
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
        toggle_buttons("answerlinks", !config.ignore_link_fix_followups, locale),
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
            button(
                selected_label(config.group_pays_first, payer_text(true, locale)),
                "cfg:payer:group".to_owned(),
            ),
            button(
                selected_label(!config.group_pays_first, payer_text(false, locale)),
                "cfg:payer:members".to_owned(),
            ),
        ]);
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
        rows.push(toggle_buttons(
            "grouptasks",
            config.group_pays_tasks,
            locale,
        ));
        rows.push(toggle_buttons(
            "answermedia",
            !config.ignore_media_replies,
            locale,
        ));
    }
    InlineKeyboardMarkup {
        inline_keyboard: rows,
    }
}

/// The first `PRIVATE_PAGES` pages also exist in private chats.
const PRIVATE_PAGES: usize = 5;

const PAGES: [&str; 10] = [
    "language",
    "link",
    "followups",
    "answerlinks",
    "timezone",
    "random",
    "payer",
    "creditless",
    "grouptasks",
    "answermedia",
];

fn titles(locale: Locale) -> [&'static str; 10] {
    match locale {
        Locale::Es => [
            "Idioma",
            "Links",
            "Seguir conversaciones",
            "Contestar replies a links",
            "Zona horaria",
            "Respuestas random",
            "Quién paga primero",
            "Mensajes por hora con créditos del grupo",
            "Tareas con créditos del grupo",
            "Contestar replies con solo multimedia",
        ],
        Locale::En => [
            "Language",
            "Links",
            "Follow conversations",
            "Answer replies to links",
            "Timezone",
            "Random replies",
            "Who pays first",
            "Messages per hour on group credits",
            "Tasks on group credits",
            "Answer media-only replies",
        ],
    }
}

fn descriptions(locale: Locale) -> [&'static str; 10] {
    match locale {
        Locale::Es => [
            "El idioma de mis mensajes, menús y respuestas.",
            "Qué hago cuando alguien manda un link de X, Bluesky, Instagram o Reddit.\n\n• Responder con link: contesto con el link arreglado.\n• Reemplazar original: borro el mensaje y lo vuelvo a mandar arreglado.\n• Desactivado: no toco los links.",
            "Si respondés a una respuesta mía de un comando, sigo la conversación sin que tengas que usar /ask.",
            "Si alguien responde a un link que arreglé, lo tomo como una pregunta para mí.\n\n• Activado: contesto.\n• Desactivado: lo ignoro.",
            "La hora que uso para tareas, recordatorios y fechas.",
            "De vez en cuando me meto en la charla del grupo aunque nadie me llame.",
            "Quién paga los mensajes de IA de alguien que tiene créditos propios.\n\n• Grupo: paga el grupo hasta el límite por hora, después sus créditos.\n• Cada uno: paga con sus créditos, y el grupo cuando no le alcanzan.",
            "Cuántos mensajes de IA por hora puede usar cada persona con los créditos del grupo.\n\n0 = nadie, ∞ = sin límite",
            "Si está activado, las tareas programadas se pagan igual que un mensaje de IA de quien las creó: cuenta para su límite por hora y sigue la opción de quién paga primero. Si no, las tareas usan solo los créditos personales de quien las creó.",
            "Si alguien me responde solo con multimedia, como un sticker, GIF, foto, video o videomensaje, sin texto, le contesto.\n\n• Activado: contesto.\n• Desactivado: no contesto.\n\nLos audios y las notas de voz los sigo escuchando, y /describe y /transcribe andan igual.",
        ],
        Locale::En => [
            "The language of my messages, menus and replies.",
            "What I do when someone sends an X, Bluesky, Instagram or Reddit link.\n\n• Reply with link: I reply with the fixed link.\n• Replace original: I delete the message and send it again, fixed.\n• Off: I leave links alone.",
            "When you reply to one of my command answers, I keep the conversation going without /ask.",
            "When someone replies to a link I fixed, I treat it as a question for me.\n\n• On: I answer.\n• Off: I ignore it.",
            "The time I use for tasks, reminders and dates.",
            "Every now and then I join the group conversation without being called.",
            "Who pays for the AI messages of someone with their own credits.\n\n• Group: the group pays up to the hourly limit, then their credits.\n• Members: they pay with their credits, and the group when theirs run out.",
            "How many AI messages per hour each person can use from the group credits.\n\n0 = nobody, ∞ = no limit",
            "When on, scheduled tasks are paid like an AI message from whoever created them: they count toward their hourly limit and follow the who pays first option. When off, tasks use only the personal credits of whoever created them.",
            "When someone replies to me with only media, like a sticker, GIF, photo, video or video message, and no text, I answer.\n\n• On: I answer.\n• Off: I don't answer.\n\nI still listen to audio and voice messages, and /describe and /transcribe keep working.",
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
        .filter(|index| is_group || *index < PRIVATE_PAGES);
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
        _ => localized(locale, "Automático", "Automatic"),
    };
    let values = [
        language.to_owned(),
        link_mode_text(&config.link_mode, locale).to_owned(),
        on_off_text(config.ai_command_followups, locale).to_owned(),
        on_off_text(!config.ignore_link_fix_followups, locale).to_owned(),
        offset_text(config.timezone_offset),
        on_off_text(config.ai_random_replies, locale).to_owned(),
        payer_text(config.group_pays_first, locale).to_owned(),
        creditless_text(config.creditless_user_hourly_limit),
        on_off_text(config.group_pays_tasks, locale).to_owned(),
        on_off_text(!config.ignore_media_replies, locale).to_owned(),
    ];
    let mut rows = (0..if is_group { PAGES.len() } else { PRIVATE_PAGES })
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
                assert_eq!(keyboard.inline_keyboard.len(), if group { 11 } else { 6 });
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
                for page in PAGES
                    .iter()
                    .take(if group { PAGES.len() } else { PRIVATE_PAGES })
                {
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
        assert!(home.inline_keyboard[7][0].text.ends_with('∞'));
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
    fn payer_page_marks_who_pays_first() {
        for (group_pays_first, selected, es, en) in [
            (
                false,
                1,
                "Quién paga primero: Cada uno",
                "Who pays first: Members",
            ),
            (
                true,
                0,
                "Quién paga primero: Grupo",
                "Who pays first: Group",
            ),
        ] {
            let config = ChatConfig {
                group_pays_first,
                ..ChatConfig::default()
            };
            let (text, page) = render_config_page(&config, Locale::En, true, "payer");
            assert_eq!(
                text,
                "Who pays first\n\nWho pays for the AI messages of someone with their own credits.\n\n• Group: the group pays up to the hourly limit, then their credits.\n• Members: they pay with their credits, and the group when theirs run out."
            );
            let (spanish, _) = render_config_page(&config, Locale::Es, true, "payer");
            assert!(spanish.starts_with("Quién paga primero\n\nQuién paga los mensajes de IA"));
            let options = &page.inline_keyboard[0];
            assert_eq!(
                options
                    .iter()
                    .map(|b| b.callback_data.as_deref())
                    .collect::<Vec<_>>(),
                [Some("cfg:payer:group"), Some("cfg:payer:members")]
            );
            assert!(options[selected].text.starts_with('✓'));
            assert!(!options[1 - selected].text.starts_with('✓'));
            for (locale, label) in [(Locale::Es, es), (Locale::En, en)] {
                let (_, home) = render_config(&config, locale, true);
                assert_eq!(home.inline_keyboard[6][0].text, label);
            }
        }
    }

    #[test]
    fn media_replies_page_shows_answering_not_ignoring() {
        for (ignore_media_replies, selected, es, en) in [
            (
                false,
                0,
                "Contestar replies con solo multimedia: Activado",
                "Answer media-only replies: On",
            ),
            (
                true,
                1,
                "Contestar replies con solo multimedia: Desactivado",
                "Answer media-only replies: Off",
            ),
        ] {
            let config = ChatConfig {
                ignore_media_replies,
                ..ChatConfig::default()
            };
            let (text, page) = render_config_page(&config, Locale::En, true, "answermedia");
            assert!(text.starts_with("Answer media-only replies\n\nWhen someone replies to me"));
            let (spanish, _) = render_config_page(&config, Locale::Es, true, "answermedia");
            assert!(spanish.starts_with("Contestar replies con solo multimedia\n\nSi alguien me"));
            let options = &page.inline_keyboard[0];
            assert_eq!(
                options
                    .iter()
                    .map(|b| b.callback_data.as_deref())
                    .collect::<Vec<_>>(),
                [Some("cfg:answermedia:on"), Some("cfg:answermedia:off")]
            );
            assert!(options[selected].text.starts_with('✓'));
            assert!(!options[1 - selected].text.starts_with('✓'));
            for (locale, label) in [(Locale::Es, es), (Locale::En, en)] {
                let (_, home) = render_config(&config, locale, true);
                assert_eq!(home.inline_keyboard[9][0].text, label);
            }
        }
    }

    #[test]
    fn link_replies_page_shows_answering_not_ignoring() {
        for (ignore, selected, label) in [
            (true, 1, "Answer replies to links: Off"),
            (false, 0, "Answer replies to links: On"),
        ] {
            let config = ChatConfig {
                ignore_link_fix_followups: ignore,
                ..ChatConfig::default()
            };
            let (_, page) = render_config_page(&config, Locale::En, false, "answerlinks");
            assert!(page.inline_keyboard[0][selected].text.starts_with('✓'));
            assert_eq!(
                page.inline_keyboard[0][0].callback_data.as_deref(),
                Some("cfg:answerlinks:on")
            );
            let (_, home) = render_config(&config, Locale::En, false);
            assert_eq!(home.inline_keyboard[3][0].text, label);
        }
    }

    #[test]
    fn group_tasks_page_marks_whether_the_group_pays() {
        for (group_pays_tasks, selected, es, en) in [
            (
                true,
                0,
                "Tareas con créditos del grupo: Activado",
                "Tasks on group credits: On",
            ),
            (
                false,
                1,
                "Tareas con créditos del grupo: Desactivado",
                "Tasks on group credits: Off",
            ),
        ] {
            let config = ChatConfig {
                group_pays_tasks,
                ..ChatConfig::default()
            };
            let (text, page) = render_config_page(&config, Locale::En, true, "grouptasks");
            assert!(text.starts_with("Tasks on group credits\n\nWhen on, scheduled tasks"));
            let (spanish, _) = render_config_page(&config, Locale::Es, true, "grouptasks");
            assert!(spanish.starts_with("Tareas con créditos del grupo\n\nSi está activado"));
            let options = &page.inline_keyboard[0];
            assert_eq!(
                options
                    .iter()
                    .map(|b| b.callback_data.as_deref())
                    .collect::<Vec<_>>(),
                [Some("cfg:grouptasks:on"), Some("cfg:grouptasks:off")]
            );
            assert!(options[selected].text.starts_with('✓'));
            for (locale, label) in [(Locale::Es, es), (Locale::En, en)] {
                let (_, home) = render_config(&config, locale, true);
                assert_eq!(home.inline_keyboard[8][0].text, label);
            }
        }
    }

    #[test]
    fn language_page_offers_automatic() {
        for (language, selected, home_label) in [
            ("auto", 2, "Idioma: Automático"),
            ("es", 0, "Idioma: Español"),
            ("en", 1, "Idioma: English"),
        ] {
            let config = ChatConfig {
                language: language.to_owned(),
                ..ChatConfig::default()
            };
            let (_, page) = render_config_page(&config, Locale::Es, false, "language");
            let options = &page.inline_keyboard[0];
            assert_eq!(
                options[2].callback_data.as_deref(),
                Some("cfg:language:auto")
            );
            assert!(options[selected].text.starts_with('✓'));
            let (_, home) = render_config(&config, Locale::Es, false);
            assert_eq!(home.inline_keyboard[0][0].text, home_label);
        }
        let (_, page) = render_config_page(&ChatConfig::default(), Locale::En, false, "language");
        assert_eq!(page.inline_keyboard[0][2].text, "✓ Automatic");
    }

    #[test]
    fn unavailable_group_pages_return_home() {
        let config = ChatConfig::default();
        for page in [
            "random",
            "payer",
            "creditless",
            "grouptasks",
            "answermedia",
            "unknown",
        ] {
            assert_eq!(
                render_config_page(&config, Locale::En, false, page),
                render_config(&config, Locale::En, false)
            );
        }
    }

    #[test]
    fn command_aliases_reply_to_original_message() {
        for command in ["/config", "/configs", "/settings@mybot"] {
            let plan = plan_config_command(
                ChatId(1),
                MessageId(2),
                command,
                "@mybot",
                Locale::Es,
                &ChatConfig::default(),
                true,
            );
            assert!(
                matches!(
                    &plan,
                    Some(TelegramAction::SendMessage(message))
                        if message.reply_to_message_id == Some(MessageId(2))
                            && message.reply_markup.is_some()
                ),
                "{plan:?}"
            );
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

    #[test]
    fn help_page_explains_saving_and_links_back_home() {
        let config = ChatConfig::default();
        for (locale, heading, back_label) in [
            (
                Locale::Es,
                "Cómo funciona\n\nCada cambio se guarda al instante.",
                "‹ Volver",
            ),
            (
                Locale::En,
                "How it works\n\nEvery change is saved right away.",
                "‹ Back",
            ),
        ] {
            let (text, keyboard) = render_config_page(&config, locale, true, "help");
            assert!(text.starts_with(heading), "{text}");
            let buttons = keyboard
                .inline_keyboard
                .concat()
                .into_iter()
                .map(|button| (button.text, button.callback_data))
                .collect::<Vec<_>>();
            assert_eq!(buttons.len(), 2);
            assert_eq!(
                buttons[0],
                (back_label.to_owned(), Some("cfg:page:home".to_owned()))
            );
            assert_eq!(buttons[1].1.as_deref(), Some("cfg:page:close"));
        }
    }
}
