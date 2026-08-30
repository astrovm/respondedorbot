//! Backward-compatible Redis conversation-state write preparation.

use serde::Serialize;

pub const MESSAGE_HISTORY_SCHEMA_VERSION: u8 = 1;
pub const CHAT_HISTORY_MAX_MESSAGES: usize = 200;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MessageWritePlan {
    pub keys: MessageWriteKeys,
    pub message_id: String,
    pub history_entry: String,
    pub chat_id: String,
    pub role: String,
    pub user_id: String,
    pub username: String,
    pub text: String,
    pub timestamp: i64,
    pub reply_to_message_id: String,
    pub mentions_bot: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MessageWriteKeys {
    pub history: String,
    pub order: String,
    pub legacy_ids: String,
    pub sequence: String,
    pub search_document: String,
}

#[derive(Serialize)]
struct HistoryEntry<'a> {
    schema_version: u8,
    id: &'a str,
    text: &'a str,
    timestamp: i64,
    role: &'a str,
}

/// Truncate stored message text using the legacy character-count rule.
#[must_use]
pub fn truncate_text(text: Option<&str>, max_length: usize) -> String {
    let Some(text) = text else {
        return String::new();
    };
    if max_length == 0 {
        return String::new();
    }
    if max_length <= 3 {
        return ".".repeat(max_length);
    }
    if text.chars().count() <= max_length {
        return text.to_owned();
    }
    let mut truncated = text.chars().take(max_length - 3).collect::<String>();
    truncated.push_str("...");
    truncated
}

/// Prepare all compatibility-sensitive values for the existing atomic Lua write.
#[allow(clippy::too_many_arguments)]
pub fn prepare_message_write(
    chat_id: &str,
    message_id: &str,
    text: &str,
    timestamp: i64,
    role: Option<&str>,
    user_id: Option<&str>,
    username: Option<&str>,
    reply_to_message_id: Option<&str>,
    mentions_bot: bool,
) -> Result<MessageWritePlan, serde_json::Error> {
    let default_role = if message_id.starts_with("bot_") {
        "assistant"
    } else {
        "user"
    };
    let role = role
        .filter(|value| !value.is_empty())
        .unwrap_or(default_role);
    let text = truncate_text(Some(text), 4096);
    let history_entry = serde_json::to_string(&HistoryEntry {
        schema_version: MESSAGE_HISTORY_SCHEMA_VERSION,
        id: message_id,
        text: &text,
        timestamp,
        role,
    })?;

    Ok(MessageWritePlan {
        keys: MessageWriteKeys {
            history: format!("chat_history:{chat_id}"),
            order: format!("chat_message_order:{chat_id}"),
            legacy_ids: format!("chat_message_ids:{chat_id}"),
            sequence: format!("chat_message_sequence:{chat_id}"),
            search_document: format!("chatmsg:{chat_id}:{message_id}"),
        },
        message_id: message_id.to_owned(),
        history_entry,
        chat_id: chat_id.to_owned(),
        role: role.to_owned(),
        user_id: user_id.unwrap_or_default().to_owned(),
        username: username.unwrap_or_default().to_owned(),
        text,
        timestamp,
        reply_to_message_id: reply_to_message_id.unwrap_or_default().to_owned(),
        mentions_bot: if mentions_bot { "1" } else { "0" }.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::{MESSAGE_HISTORY_SCHEMA_VERSION, prepare_message_write, truncate_text};

    #[test]
    fn truncates_by_unicode_characters_with_legacy_small_limit_rules() {
        assert_eq!(truncate_text(None, 10), "");
        assert_eq!(truncate_text(Some("hello"), 0), "");
        assert_eq!(truncate_text(Some("hello"), 2), "..");
        assert_eq!(truncate_text(Some("hello"), 4), "h...");
        assert_eq!(truncate_text(Some("😀😀😀😀😀"), 4), "😀...");
        assert_eq!(truncate_text(Some("short"), 10), "short");
    }

    #[test]
    fn prepares_versioned_user_message_for_existing_redis_schema() {
        let result = prepare_message_write(
            "-1001",
            "42",
            "hello",
            1_788_000_000,
            None,
            Some("7"),
            Some("astro"),
            Some("41"),
            true,
        );
        assert!(result.is_ok());
        let Ok(plan) = result else {
            return;
        };
        assert_eq!(plan.keys.history, "chat_history:-1001");
        assert_eq!(plan.keys.order, "chat_message_order:-1001");
        assert_eq!(plan.keys.legacy_ids, "chat_message_ids:-1001");
        assert_eq!(plan.keys.sequence, "chat_message_sequence:-1001");
        assert_eq!(plan.keys.search_document, "chatmsg:-1001:42");
        assert_eq!(plan.role, "user");
        assert_eq!(plan.mentions_bot, "1");
        let value = serde_json::from_str::<serde_json::Value>(&plan.history_entry).ok();
        assert_eq!(
            value
                .as_ref()
                .and_then(|item| item["schema_version"].as_u64()),
            Some(u64::from(MESSAGE_HISTORY_SCHEMA_VERSION))
        );
        assert_eq!(
            value.as_ref().and_then(|item| item["id"].as_str()),
            Some("42")
        );
        assert_eq!(
            value.as_ref().and_then(|item| item["text"].as_str()),
            Some("hello")
        );
    }

    #[test]
    fn defaults_bot_role_but_preserves_explicit_truthy_role() {
        let defaulted_result =
            prepare_message_write("1", "bot_2", "text", 10, Some(""), None, None, None, false);
        assert!(defaulted_result.is_ok());
        let Ok(defaulted) = defaulted_result else {
            return;
        };
        assert_eq!(defaulted.role, "assistant");
        let explicit_result = prepare_message_write(
            "1",
            "bot_2",
            "text",
            10,
            Some("tool"),
            None,
            None,
            None,
            false,
        );
        assert!(explicit_result.is_ok());
        let Ok(explicit) = explicit_result else {
            return;
        };
        assert_eq!(explicit.role, "tool");
        assert_eq!(explicit.mentions_bot, "0");
    }
}
