//! Shared plain-text conventions for Telegram menus.
use crate::locale::Locale;
use crate::telegram_actions::InlineKeyboardButton;

#[must_use]
pub const fn localized(locale: Locale, es: &'static str, en: &'static str) -> &'static str {
    match locale {
        Locale::Es => es,
        Locale::En => en,
    }
}

#[must_use]
pub fn button(text: impl Into<String>, callback: impl Into<String>) -> InlineKeyboardButton {
    InlineKeyboardButton {
        text: text.into(),
        callback_data: Some(callback.into()),
        url: None,
        copy_text: None,
    }
}

#[must_use]
pub fn back(locale: Locale, callback: impl Into<String>) -> InlineKeyboardButton {
    button(localized(locale, "‹ Volver", "‹ Back"), callback)
}

#[must_use]
pub fn close(locale: Locale, callback: impl Into<String>) -> InlineKeyboardButton {
    button(localized(locale, "Cerrar", "Close"), callback)
}
