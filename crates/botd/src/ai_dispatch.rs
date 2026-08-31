//! Object-safe boundary between Telegram routing and one native AI transaction.

use bot_core::locale::Locale;
use bot_core::telegram_input::{ChatId, MessageId, UserId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiReplyMetadata {
    pub kind: String,
    pub uses_ai: bool,
}

impl AiReplyMetadata {
    #[must_use]
    pub fn is_non_ai_command(&self) -> bool {
        self.kind == "command" && !self.uses_ai
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiConversationInput {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub chat_type: String,
    pub chat_title: String,
    pub sender_id: UserId,
    pub sender_first_name: String,
    pub sender_username: String,
    pub message_text: String,
    pub command: String,
    pub reply_to_message_id: Option<MessageId>,
    pub reply_context: Option<String>,
    pub photo_file_id: Option<String>,
    pub audio_file_id: Option<String>,
    pub locale: Locale,
    pub timezone_offset_hours: i64,
    pub timestamp: i64,
    pub spontaneous: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiPreparation {
    Silent {
        diagnostics: Vec<String>,
    },
    Reply {
        text: String,
        completion_id: Option<String>,
        diagnostics: Vec<String>,
    },
}

impl AiPreparation {
    #[must_use]
    pub fn silent() -> Self {
        Self::Silent {
            diagnostics: Vec::new(),
        }
    }

    #[must_use]
    pub fn reply(text: impl Into<String>, completion_id: Option<String>) -> Self {
        Self::Reply {
            text: text.into(),
            completion_id,
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiDelivery {
    pub completion_id: String,
    pub delivered: bool,
    pub sent_message_id: Option<MessageId>,
}

/// The implementation may reserve credits during `prepare`, but must not treat
/// the transaction as delivered until `complete_delivery` is called.
pub trait AiConversationSource {
    fn reply_metadata(
        &mut self,
        chat_id: &str,
        message_id: &str,
    ) -> Result<Option<AiReplyMetadata>, String>;

    fn prepare(&mut self, input: AiConversationInput) -> Result<AiPreparation, String>;

    fn prepare_streaming(
        &mut self,
        input: AiConversationInput,
        _on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<AiPreparation, String> {
        self.prepare(input)
    }

    fn record_ignored(&mut self, _input: AiConversationInput) -> Result<(), String> {
        Ok(())
    }

    fn complete_delivery(&mut self, delivery: AiDelivery) -> Result<(), String>;
}

#[must_use]
pub fn reply_context(
    first_name: Option<&str>,
    username: Option<&str>,
    text: Option<&str>,
) -> Option<String> {
    let text = text.map(str::trim).filter(|value| !value.is_empty())?;
    let first_name = first_name.unwrap_or_default().trim();
    let username = username.unwrap_or_default().trim();
    let identity = if username.is_empty() {
        first_name.to_owned()
    } else if first_name.is_empty() {
        format!("({username})")
    } else {
        format!("{first_name} ({username})")
    };
    Some(if identity.is_empty() {
        text.to_owned()
    } else {
        format!("{identity}: {text}")
    })
}

#[cfg(test)]
mod tests {
    use super::{AiReplyMetadata, reply_context};

    #[test]
    fn metadata_distinguishes_non_ai_command_followups() {
        assert!(
            AiReplyMetadata {
                kind: "command".to_owned(),
                uses_ai: false,
            }
            .is_non_ai_command()
        );
        for value in [
            AiReplyMetadata {
                kind: "command".to_owned(),
                uses_ai: true,
            },
            AiReplyMetadata {
                kind: "ai".to_owned(),
                uses_ai: false,
            },
        ] {
            assert!(!value.is_non_ai_command());
        }
    }

    #[test]
    fn reply_context_matches_the_legacy_identity_shape() {
        assert_eq!(
            reply_context(Some("Gordo"), Some("testbot"), Some(" earlier answer ")),
            Some("Gordo (testbot): earlier answer".to_owned())
        );
        assert_eq!(
            reply_context(None, Some("testbot"), Some("answer")),
            Some("(testbot): answer".to_owned())
        );
        assert_eq!(
            reply_context(None, None, Some("answer")),
            Some("answer".to_owned())
        );
        assert_eq!(reply_context(Some("Gordo"), None, Some("  ")), None);
    }
}
