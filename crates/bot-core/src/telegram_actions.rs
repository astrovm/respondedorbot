//! Typed outbound Telegram actions produced by application logic.

use serde::Serialize;

use crate::telegram_input::{ChatId, MessageId};

pub const MAX_TELEGRAM_TEXT_LENGTH: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ParseMode {
    #[serde(rename = "HTML")]
    Html,
    #[serde(rename = "MarkdownV2")]
    MarkdownV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InlineKeyboardButton {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_data: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InlineKeyboardMarkup {
    pub inline_keyboard: Vec<Vec<InlineKeyboardButton>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SendMessage {
    pub chat_id: ChatId,
    pub text: String,
    pub reply_to_message_id: Option<MessageId>,
    pub parse_mode: Option<ParseMode>,
    pub disable_web_page_preview: bool,
    pub reply_markup: Option<InlineKeyboardMarkup>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelegramAction {
    SendMessage(SendMessage),
    SendTyping {
        chat_id: ChatId,
    },
    EditMessage {
        chat_id: ChatId,
        message_id: MessageId,
        text: String,
        reply_markup: Option<InlineKeyboardMarkup>,
    },
    DeleteMessage {
        chat_id: ChatId,
        message_id: MessageId,
    },
    AnswerCallback {
        callback_id: String,
        text: Option<String>,
        show_alert: bool,
    },
    AnswerPreCheckout {
        query_id: String,
        ok: bool,
        error_message: Option<String>,
    },
}

#[must_use]
pub fn truncate_text(text: &str) -> String {
    if text.chars().count() <= MAX_TELEGRAM_TEXT_LENGTH {
        return text.to_owned();
    }
    let mut truncated = text
        .chars()
        .take(MAX_TELEGRAM_TEXT_LENGTH - 3)
        .collect::<String>();
    let newline_threshold = MAX_TELEGRAM_TEXT_LENGTH * 4 / 5;
    if let Some(last_newline) = truncated.rfind('\n')
        && truncated[..last_newline].chars().count() > newline_threshold
    {
        truncated.truncate(last_newline);
    }
    truncated.push_str("...");
    truncated
}

impl SendMessage {
    #[must_use]
    pub fn new(chat_id: ChatId, text: &str) -> Self {
        Self {
            chat_id,
            text: truncate_text(text),
            reply_to_message_id: None,
            parse_mode: None,
            disable_web_page_preview: false,
            reply_markup: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        InlineKeyboardButton, InlineKeyboardMarkup, MAX_TELEGRAM_TEXT_LENGTH, SendMessage,
        truncate_text,
    };
    use crate::telegram_input::ChatId;

    #[test]
    fn truncation_preserves_short_and_unicode_text() {
        assert_eq!(truncate_text("hola"), "hola");
        let text = "🦀".repeat(MAX_TELEGRAM_TEXT_LENGTH + 10);
        let actual = truncate_text(&text);
        assert_eq!(actual.chars().count(), MAX_TELEGRAM_TEXT_LENGTH);
        assert!(actual.ends_with("..."));
    }

    #[test]
    fn truncation_prefers_a_late_line_boundary() {
        let prefix = "a".repeat(MAX_TELEGRAM_TEXT_LENGTH * 4 / 5 + 1);
        let text = format!("{prefix}\n{}", "b".repeat(1000));
        assert_eq!(truncate_text(&text), format!("{prefix}..."));

        let early = format!("line\n{}", "b".repeat(MAX_TELEGRAM_TEXT_LENGTH + 10));
        assert_eq!(
            truncate_text(&early).chars().count(),
            MAX_TELEGRAM_TEXT_LENGTH
        );
    }

    #[test]
    fn send_message_constructor_applies_safe_defaults() {
        assert_eq!(
            SendMessage::new(ChatId(42), "hello"),
            SendMessage {
                chat_id: ChatId(42),
                text: "hello".to_owned(),
                reply_to_message_id: None,
                parse_mode: None,
                disable_web_page_preview: false,
                reply_markup: None,
            }
        );
    }

    #[test]
    fn keyboard_types_serialize_without_untyped_maps() {
        let keyboard = InlineKeyboardMarkup {
            inline_keyboard: vec![vec![InlineKeyboardButton {
                text: "Open".to_owned(),
                url: Some("https://example.test".to_owned()),
                callback_data: None,
            }]],
        };
        let actual = serde_json::to_value(keyboard);
        assert!(actual.is_ok());
        let actual = actual.unwrap_or(serde_json::Value::Null);
        assert_eq!(
            actual,
            serde_json::json!({
                "inline_keyboard":[[{"text":"Open","url":"https://example.test"}]]
            })
        );
    }
}
