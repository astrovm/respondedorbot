//! Backward-compatible Redis conversation-state write preparation.

use serde::Serialize;
use std::cmp::Reverse;
use std::collections::HashSet;

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchCandidate {
    pub index: usize,
    pub message_id: String,
    pub text: String,
    pub reply_to_message_id: String,
    pub timestamp: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RankedSearchCandidate {
    pub index: usize,
    pub reply_score: u8,
    pub overlap_score: usize,
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

/// Escape the broad full-text query using the existing RediSearch contract.
#[must_use]
pub fn escape_search_text(query_text: &str) -> String {
    query_text
        .split_whitespace()
        .filter_map(|token| {
            let cleaned = token
                .chars()
                .filter(|character| {
                    character.is_alphanumeric() || matches!(character, '_' | '@' | '.' | '-')
                })
                .collect::<String>()
                .replace('@', "\\@");
            (!cleaned.is_empty()).then_some(cleaned)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Escape one TAG query value using the legacy ASCII-safe rule.
#[must_use]
pub fn escape_search_tag(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if !character.is_ascii_alphanumeric() && character != '_' {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Rank adapter-parsed RediSearch rows without losing their original payloads.
#[must_use]
pub fn rank_search_candidates(
    candidates: &[SearchCandidate],
    search_text: &str,
    reply_to_message_id: Option<&str>,
    excluded_message_ids: &HashSet<String>,
    limit: usize,
) -> Vec<RankedSearchCandidate> {
    let query_tokens = search_text
        .to_lowercase()
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<HashSet<_>>();
    let mut ranked = candidates
        .iter()
        .filter(|candidate| {
            candidate.message_id.is_empty() || !excluded_message_ids.contains(&candidate.message_id)
        })
        .map(|candidate| {
            let text_tokens = candidate
                .text
                .to_lowercase()
                .split_whitespace()
                .map(ToOwned::to_owned)
                .collect::<HashSet<_>>();
            (
                RankedSearchCandidate {
                    index: candidate.index,
                    reply_score: u8::from(
                        reply_to_message_id
                            .is_some_and(|reply_id| candidate.reply_to_message_id == reply_id),
                    ),
                    overlap_score: query_tokens.intersection(&text_tokens).count(),
                },
                candidate.timestamp,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by_key(|(candidate, timestamp)| {
        (
            Reverse(candidate.reply_score),
            Reverse(candidate.overlap_score),
            Reverse(*timestamp),
            candidate.index,
        )
    });
    ranked.truncate(limit);
    ranked
        .into_iter()
        .map(|(candidate, _timestamp)| candidate)
        .collect()
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
    use std::collections::HashSet;

    use super::{
        MESSAGE_HISTORY_SCHEMA_VERSION, SearchCandidate, escape_search_tag, escape_search_text,
        prepare_message_write, rank_search_candidates, truncate_text,
    };

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

    #[test]
    fn escapes_search_text_and_tag_values_for_redisearch() {
        assert_eq!(
            escape_search_text("wallet, @test_bot riesgo!"),
            "wallet \\@test_bot riesgo"
        );
        assert_eq!(escape_search_text("¿qué pasó?"), "qué pasó");
        assert_eq!(escape_search_tag("-1001"), "\\-1001");
        assert_eq!(escape_search_tag("safe_value"), "safe_value");
    }

    #[test]
    fn ranks_reply_overlap_and_timestamp_and_filters_exclusions() {
        let candidates = vec![
            SearchCandidate {
                index: 0,
                message_id: "10".to_owned(),
                text: "wallet error happened".to_owned(),
                reply_to_message_id: "99".to_owned(),
                timestamp: 10,
            },
            SearchCandidate {
                index: 1,
                message_id: "11".to_owned(),
                text: "wallet error generic".to_owned(),
                reply_to_message_id: "1".to_owned(),
                timestamp: 11,
            },
            SearchCandidate {
                index: 2,
                message_id: "12".to_owned(),
                text: "wallet error excluded".to_owned(),
                reply_to_message_id: "99".to_owned(),
                timestamp: 12,
            },
        ];
        let excluded = HashSet::from(["12".to_owned()]);
        let ranked = rank_search_candidates(&candidates, "wallet error", Some("99"), &excluded, 5);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].index, 0);
        assert_eq!(ranked[0].reply_score, 1);
        assert_eq!(ranked[1].index, 1);
        assert_eq!(ranked[1].overlap_score, 2);
    }
}
