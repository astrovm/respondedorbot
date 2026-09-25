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
            "/p o $ticker: cualquier activo\n/c: solo crypto\n/s: acciones y fondos\n\nEjemplos\n/p btc\n/p btc en eur\n/p rkh 1m\n/s apple\n/c eth, sol\n\nArgentina\n/dolar, /bcra, /rulo, /devo\n\nMás\n/petroleo, /elecciones, /powerlaw, /rainbow, /satoshi",
            "/p or $ticker: any asset\n/c: crypto only\n/s: stocks and funds\n\nExamples\n/p btc\n/p btc in eur\n/p rkh 1m\n/s apple\n/c eth, sol\n\nArgentina\n/dollar, /bcra, /rulo, /devo\n\nMore\n/oil, /elections, /powerlaw, /rainbow, /satoshi",
        ),
        (
            "ai",
            "IA y media",
            "AI and media",
            "/ask: preguntame lo que quieras\nTambién podés mencionarme o responder a un mensaje mío.\n\n/transcribe: respondé a un audio, video, imagen o link de YouTube\n/resumen: resumir la charla del chat\n\nSi hace falta, busco en la web.",
            "/ask: ask me anything\nYou can also mention me or reply to one of my messages.\n\n/transcribe: reply to audio, video, an image or a YouTube link\n/summary: summarize the chat\n\nI search the web when needed.",
        ),
        (
            "tasks",
            "Tareas",
            "Tasks",
            "/tarea: ver y cancelar tus tareas\n\nEjemplos\n/tarea mañana a las 9 recordame pagar el alquiler\n/tarea todos los lunes a las 8 mandá el dólar\n/tarea cada 2 horas recordame tomar agua",
            "/task: view and cancel your tasks\n\nExamples\n/task tomorrow at 9 remind me to pay rent\n/task every Monday at 8 send the dollar rate\n/task every 2 hours remind me to drink water",
        ),
        (
            "credits",
            "Créditos",
            "Credits",
            "/balance: ver tu saldo\n/topup: cargar con Telegram Stars\n/gastos: ver en qué gastaste\n/transfer 1.5: pasar créditos al grupo",
            "/balance: check your balance\n/topup: add credits with Telegram Stars\n/charges: see what you spent\n/transfer 1.5: move credits to the group",
        ),
        (
            "tools",
            "Utilidades",
            "Utilities",
            "/clima Córdoba: clima actual\n/random pizza, sushi: elijo por vos\n/convertbase 101, 2, 10: convertir entre bases\n/comando hola mundo: convertir en /comando\n/time: timestamp Unix\n/gm y /gn: GIF de saludo\n/instance: qué instancia responde",
            "/weather London: current weather\n/random pizza, sushi: I pick for you\n/convertbase 101, 2, 10: convert between bases\n/command hello world: turn text into a /command\n/time: Unix timestamp\n/gm and /gn: greeting GIF\n/instance: which instance is answering",
        ),
        (
            "settings",
            "Configuración",
            "Settings",
            "/config: ajustes de este chat\n/idioma: cambiar el idioma\n\nArreglo los links de X, Bluesky, Instagram y Reddit para que se vean bien en Telegram. Elegí cómo en /config.",
            "/config: settings for this chat\n/language: change the language\n\nI fix X, Bluesky, Instagram and Reddit links so they preview properly in Telegram. Choose how in /config.",
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
            "Ayuda\n\nHablame con /ask, mencionándome o respondiendo a un mensaje mío.",
            "Help\n\nTalk to me with /ask, by mentioning me or by replying to one of my messages.",
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
            assert!(home.lines().count() <= 4);
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
