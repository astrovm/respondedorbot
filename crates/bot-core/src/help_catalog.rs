//! Localized help categories for the interactive `/help` menu.

use crate::locale::Locale;

/// Compact help pages. Full aliases remain available through the command parser.
#[must_use]
pub fn render_help_page(
    locale: Locale,
    page: &str,
) -> (String, crate::telegram_actions::InlineKeyboardMarkup) {
    use crate::menu_ui::{back, button, close, localized};
    let entries = [
        (
            "markets",
            "Mercados",
            "Markets",
            "/p · $ticker — todos los activos\n/c — crypto\n/s — acciones y otros activos\n\n/p rkh 1m\n/p btc in eur\n/s Rockhopper\n\n/dolar · /petroleo · /bcra · /elecciones\n/rulo · /devo · /powerlaw · /rainbow · /satoshi",
            "/p · $ticker — all assets\n/c — crypto\n/s — stocks and other assets\n\n/p rkh 1m\n/p btc in eur\n/s Rockhopper\n\n/dolar · /petroleo · /bcra · /elecciones\n/rulo · /devo · /powerlaw · /rainbow · /satoshi",
        ),
        (
            "ai",
            "IA y media",
            "AI and media",
            "/ask — preguntame lo que quieras\nTambién podés mencionarme o responderme.\n\n/transcribe — respondé a un audio, video o imagen\n/resumen — resumir el chat\n\nPuedo buscar en la web cuando hace falta.",
            "/ask — ask me anything\nYou can also mention me or reply to me.\n\n/transcribe — reply to audio, video or an image\n/resumen — summarize the chat\n\nI can search the web when needed.",
        ),
        (
            "tasks",
            "Tareas",
            "Tasks",
            "/tarea — ver tus tareas\n/tarea mañana recordame pagar el alquiler\n\nCreá recordatorios o tareas recurrentes con tus palabras.",
            "/tarea — view your tasks\n/tarea remind me tomorrow to pay rent\n\nCreate reminders or recurring tasks in your own words.",
        ),
        (
            "credits",
            "Créditos",
            "Credits",
            "/balance — saldo\n/topup — cargar con Telegram Stars\n/charges — historial de gastos\n/transfer 1.5 — pasar créditos al grupo",
            "/balance — balance\n/topup — add credits with Telegram Stars\n/charges — spending history\n/transfer 1.5 — transfer credits to the group",
        ),
        (
            "tools",
            "Utilidades",
            "Utilities",
            "/clima Córdoba — clima actual\n/random pizza, sushi — elegir una opción\n/convertbase 101, 2, 10 — convertir bases\n/time — timestamp\n/comando — convertir a comando Telegram\n/instance — instancia del bot\n/gm · /gn — GIF de saludo",
            "/clima London — current weather\n/random pizza, sushi — pick an option\n/convertbase 101, 2, 10 — convert bases\n/time — timestamp\n/comando — convert to a Telegram command\n/instance — bot instance\n/gm · /gn — greeting GIF",
        ),
        (
            "settings",
            "Configuración y links",
            "Settings and links",
            "/config — ajustes de este chat\n/language — idioma\n\nArreglo links de X, Bluesky, Instagram y Reddit según tu configuración.",
            "/config — settings for this chat\n/language — language\n\nI fix X, Bluesky, Instagram and Reddit links according to your settings.",
        ),
    ];
    if let Some((_, es, en, body_es, body_en)) = entries.iter().find(|entry| entry.0 == page) {
        return (
            format!(
                "{}\n\n{}",
                localized(locale, es, en),
                localized(locale, body_es, body_en)
            ),
            crate::telegram_actions::InlineKeyboardMarkup {
                inline_keyboard: vec![vec![back(locale, "help:home"), close(locale, "help:close")]],
            },
        );
    }
    let mut rows = entries
        .iter()
        .map(|(id, es, en, _, _)| vec![button(localized(locale, es, en), format!("help:{id}"))])
        .collect::<Vec<_>>();
    rows.push(vec![close(locale, "help:close")]);
    (
        localized(
            locale,
            "Ayuda\n\n¿Qué querés hacer?",
            "Help\n\nWhat would you like to do?",
        )
        .to_owned(),
        crate::telegram_actions::InlineKeyboardMarkup {
            inline_keyboard: rows,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_help_category_is_localized_and_has_navigation() {
        for locale in [Locale::Es, Locale::En] {
            let (home, keyboard) = render_help_page(locale, "home");
            assert!(home.lines().count() <= 3);
            assert_eq!(keyboard.inline_keyboard.len(), 7);
            for row in &keyboard.inline_keyboard[..6] {
                let page = row[0]
                    .callback_data
                    .as_deref()
                    .unwrap_or_default()
                    .strip_prefix("help:")
                    .unwrap_or_default();
                let (text, keyboard) = render_help_page(locale, page);
                assert!(text.contains('/'));
                assert!(text.chars().count() < 1000);
                assert_eq!(
                    keyboard.inline_keyboard[0][0].callback_data.as_deref(),
                    Some("help:home")
                );
                assert_eq!(
                    keyboard.inline_keyboard[0][1].callback_data.as_deref(),
                    Some("help:close")
                );
            }
        }
    }
}
