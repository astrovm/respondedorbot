//! Redis and PostgreSQL adapters for foreground native AI conversations.

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_message_state::{RedisMessageState, SearchRow};
use bot_core::ai_prompt::{HistoryMessage, PromptRole, RetrievedMessage};
use bot_core::ai_request::sanitize_assistant_text;
use bot_core::command_state::{
    BOT_MESSAGE_METADATA_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT, CHAT_STATE_TTL_SECONDS,
};
use bot_core::message_state::{
    CHAT_HISTORY_MAX_MESSAGES, bot_message_metadata_key, chat_compacted_until_key,
    chat_members_key, chat_summary_key, prepare_chat_member_payload, prepare_message_write,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::ai_dispatch::{AiConversationInput, AiReplyMetadata};
use crate::compaction_scheduler::MemoryCompactionPlan;
use crate::compaction_scheduler::PayerSource;
use crate::conversation::{
    ConversationBilling, ConversationMemory, ConversationState, ProviderSegmentRequest,
    ReserveDecision, ReserveRequest, SettlementRequest,
};

const COMPACTION_THRESHOLD: usize = 40;
const COMPACTION_KEEP: usize = 25;

pub struct RedisConversationState {
    state: RedisMessageState,
}

impl RedisConversationState {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, String> {
        RedisMessageState::new(endpoint)
            .map(|state| Self { state })
            .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug, Deserialize)]
struct StoredHistoryEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    timestamp: i64,
    #[serde(default)]
    role: String,
}

impl ConversationState for RedisConversationState {
    fn reply_metadata(
        &mut self,
        chat_id: &str,
        message_id: &str,
    ) -> Result<Option<AiReplyMetadata>, String> {
        let key = bot_message_metadata_key(chat_id, message_id);
        let Some(payload) = self
            .state
            .get_value(&key)
            .map_err(|error| error.to_string())?
        else {
            return Ok(None);
        };
        let value: Value = match serde_json::from_str(&payload) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let Some(value) = value.as_object() else {
            return Ok(None);
        };
        Ok(Some(AiReplyMetadata {
            kind: value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            uses_ai: value
                .get("uses_ai")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }))
    }

    fn load_memory(
        &mut self,
        chat_id: &str,
        search_text: &str,
        _reply_to_message_id: Option<&str>,
        _max_history_messages: usize,
    ) -> Result<ConversationMemory, String> {
        let limit = i64::try_from(CHAT_HISTORY_MAX_MESSAGES)
            .map_err(|_| "history limit exceeds the Redis range".to_owned())?;
        let entries = self
            .state
            .get_history_entries(chat_id, limit)
            .map_err(|error| error.to_string())?;
        let parsed = decode_history(entries);
        let summary = self
            .state
            .get_value(&chat_summary_key(chat_id))
            .map_err(|error| error.to_string())?
            .filter(|value| !value.is_empty());
        let marker = if summary.is_some() {
            self.state
                .get_value(&chat_compacted_until_key(chat_id))
                .map_err(|error| error.to_string())?
                .filter(|value| !value.is_empty())
        } else {
            None
        };
        let (visible, compaction_plan) = build_compaction_view(&parsed, &summary, &marker, chat_id);
        let recent_ids = visible
            .iter()
            .map(|entry| entry.id.clone())
            .filter(|id| !id.is_empty())
            .collect::<std::collections::HashSet<_>>();
        let history = visible
            .into_iter()
            .filter(|entry| !entry.text.is_empty())
            .map(|entry| HistoryMessage {
                role: role(&entry.role, &entry.id),
                text: if role(&entry.role, &entry.id) == PromptRole::Assistant {
                    sanitize_assistant_text(&entry.text)
                } else {
                    entry.text
                },
            })
            .collect();
        let retrieved = if search_text.trim().is_empty() {
            Vec::new()
        } else {
            self.state
                .search_messages(chat_id, search_text, 5)
                .map(|rows| decode_retrieved(rows, &recent_ids))
                .unwrap_or_default()
        };
        Ok(ConversationMemory {
            summary,
            history,
            retrieved,
            compaction_plan,
        })
    }

    fn record_incoming(&mut self, input: &AiConversationInput) -> Result<(), String> {
        if input.message_text.is_empty() {
            return Ok(());
        }
        let chat_id = input.chat_id.0.to_string();
        let message_id = input.message_id.0.to_string();
        let user_id = input.sender_id.0.to_string();
        let identity = user_identity(input);
        let text = match (&input.reply_context, identity.is_empty()) {
            (Some(reply), false) if input.locale == bot_core::locale::Locale::Es => {
                format!(
                    "{identity} (en respuesta a {reply}): {}",
                    input.message_text
                )
            }
            (Some(reply), false) => {
                format!("{identity} (replying to {reply}): {}", input.message_text)
            }
            (Some(reply), true) if input.locale == bot_core::locale::Locale::Es => {
                format!("(en respuesta a {reply}): {}", input.message_text)
            }
            (Some(reply), true) => format!("(replying to {reply}): {}", input.message_text),
            (None, _) => format!("{identity}: {}", input.message_text),
        };
        let plan = prepare_message_write(
            &chat_id,
            &message_id,
            &text,
            input.timestamp,
            Some("user"),
            Some(&user_id),
            Some(&input.sender_username),
            input
                .reply_to_message_id
                .map(|id| id.0.to_string())
                .as_deref(),
            input.message_text.contains('@') || input.message_text.starts_with('/'),
        )
        .map_err(|error| error.to_string())?;
        let _stored = self
            .state
            .save_message(&plan, CHAT_STATE_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT)
            .map_err(|error| error.to_string())?;
        if matches!(input.chat_type.as_str(), "group" | "supergroup") {
            let payload = prepare_chat_member_payload(
                &input.sender_first_name,
                &input.sender_username,
                input.timestamp,
            )
            .map_err(|error| error.to_string())?;
            self.state
                .save_chat_member(
                    &chat_members_key(&chat_id),
                    &user_id,
                    &payload,
                    CHAT_STATE_TTL_SECONDS,
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn load_summary_memory(
        &mut self,
        chat_id: &str,
        max_history_messages: usize,
    ) -> Result<ConversationMemory, String> {
        let limit = i64::try_from(max_history_messages)
            .map_err(|_| "summary history limit exceeds the Redis range".to_owned())?;
        let entries = self
            .state
            .get_history_entries(chat_id, limit)
            .map_err(|error| error.to_string())?;
        let summary = self
            .state
            .get_value(&chat_summary_key(chat_id))
            .map_err(|error| error.to_string())?
            .filter(|value| !value.is_empty());
        let marker = if summary.is_some() {
            self.state
                .get_value(&chat_compacted_until_key(chat_id))
                .map_err(|error| error.to_string())?
                .filter(|value| !value.is_empty())
        } else {
            None
        };
        Ok(decode_summary_memory(entries, summary, marker))
    }

    fn record_outgoing(
        &mut self,
        input: &AiConversationInput,
        sent_message_id: Option<i64>,
        text: &str,
    ) -> Result<(), String> {
        let chat_id = input.chat_id.0.to_string();
        let stored_id = format!("bot_{}", sent_message_id.unwrap_or(input.message_id.0));
        let plan = prepare_message_write(
            &chat_id,
            &stored_id,
            text,
            input.timestamp,
            Some("assistant"),
            None,
            None,
            None,
            false,
        )
        .map_err(|error| error.to_string())?;
        let _stored = self
            .state
            .save_message(&plan, CHAT_STATE_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT)
            .map_err(|error| error.to_string())?;
        if let Some(sent_message_id) = sent_message_id {
            self.state
                .set_value(
                    &bot_message_metadata_key(&chat_id, &sent_message_id.to_string()),
                    &json!({"type": "ai"}).to_string(),
                    BOT_MESSAGE_METADATA_TTL_SECONDS,
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

fn decode_summary_memory(
    entries: Vec<String>,
    summary: Option<String>,
    marker: Option<String>,
) -> ConversationMemory {
    let mut parsed = decode_history(entries);
    if let Some(marker) = marker
        && let Some(index) = parsed.iter().position(|entry| entry.id == marker)
    {
        parsed.drain(..=index);
    }
    let history = parsed
        .into_iter()
        .filter(|entry| !entry.text.is_empty())
        .map(|entry| HistoryMessage {
            role: role(&entry.role, &entry.id),
            text: if role(&entry.role, &entry.id) == PromptRole::Assistant {
                sanitize_assistant_text(&entry.text)
            } else {
                entry.text
            },
        })
        .collect();
    ConversationMemory {
        summary,
        history,
        retrieved: Vec::new(),
        compaction_plan: None,
    }
}

fn stored_compaction_message(entry: &StoredHistoryEntry) -> Value {
    json!({
        "id": entry.id,
        "text": entry.text,
        "timestamp": entry.timestamp,
        "role": entry.role,
    })
}

fn build_compaction_view(
    history: &[StoredHistoryEntry],
    summary: &Option<String>,
    marker: &Option<String>,
    chat_id: &str,
) -> (Vec<StoredHistoryEntry>, Option<MemoryCompactionPlan>) {
    let start_index = marker.as_deref().map_or(0, |marker| {
        history
            .iter()
            .position(|entry| entry.id == marker)
            .map_or(0, |index| index + 1)
    });
    let delta = history.get(start_index..).unwrap_or_default();
    let dropped_count = delta.len().saturating_sub(COMPACTION_KEEP);
    let plan = (delta.len() > COMPACTION_THRESHOLD && dropped_count > 0)
        .then(|| {
            let dropped = &delta[..dropped_count];
            let target_marker = dropped.last()?.id.clone();
            (!target_marker.is_empty()).then(|| MemoryCompactionPlan {
                chat_id: chat_id.to_owned(),
                messages: dropped.iter().map(stored_compaction_message).collect(),
                prior_summary: summary.clone(),
                expected_marker: marker.clone(),
                target_marker,
            })
        })
        .flatten();
    let visible = if plan.is_some() && summary.is_some() {
        delta[dropped_count..].to_vec()
    } else {
        delta.to_vec()
    };
    (visible, plan)
}

pub struct PostgresConversationBilling {
    repository: BillingRepository,
    payer_by_operation: std::collections::HashMap<String, PayerSource>,
}

impl PostgresConversationBilling {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            repository: BillingRepository::new(database_url),
            payer_by_operation: std::collections::HashMap::new(),
        }
    }
}

impl ConversationBilling for PostgresConversationBilling {
    fn reserve(&mut self, request: ReserveRequest) -> Result<ReserveDecision, String> {
        let amount = i32::try_from(request.amount)
            .map_err(|_| "AI reservation exceeds the database range".to_owned())?;
        let requested_source = self
            .payer_by_operation
            .get(&request.operation_id)
            .copied()
            .map(|source| match source {
                PayerSource::User => "user",
                PayerSource::Chat => "chat",
            });
        self.repository
            .charge_ai_credits(
                request.user_id,
                request.chat_id,
                amount,
                "ai_reserve",
                &request.metadata,
                requested_source,
                Some(&request.reservation_id),
                &request.operation_id,
            )
            .map(|result| {
                let source = match result.source.as_deref() {
                    Some("user") => Some(PayerSource::User),
                    Some("chat") => Some(PayerSource::Chat),
                    _ => None,
                };
                if result.ok
                    && let Some(source) = source
                {
                    self.payer_by_operation
                        .insert(request.operation_id.clone(), source);
                }
                ReserveDecision {
                    authorized: result.ok,
                    user_balance: result.user_balance,
                    chat_balance: result.chat_balance,
                    source,
                }
            })
            .map_err(|error| error.to_string())
    }

    fn record_segment(&mut self, request: ProviderSegmentRequest) -> Result<(), String> {
        let metadata = json!({
            "operation_id": request.operation_id,
            "segment_id": request.segment_id,
            "segment": request.segment,
        });
        self.repository
            .record_ai_provider_usage(request.user_id, request.chat_id, &metadata)
            .map(|_inserted| ())
            .map_err(|error| error.to_string())
    }

    fn settle(&mut self, request: SettlementRequest) -> Result<(), String> {
        self.payer_by_operation.remove(&request.operation_id);
        let metadata = Map::from_iter([
            ("operation_id".to_owned(), json!(request.operation_id)),
            ("reason".to_owned(), json!(request.reason)),
            ("delivered".to_owned(), json!(request.delivered)),
        ]);
        self.repository
            .settle_ai_operation_once(
                request.user_id,
                request.chat_id,
                metadata
                    .get("operation_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                request.actual_credit_units,
                &metadata,
            )
            .map(|_result| ())
            .map_err(|error| error.to_string())
    }

    fn personal_balance(&mut self, user_id: i64) -> Result<Option<i64>, String> {
        self.repository
            .get_balance("user", user_id)
            .map(Some)
            .map_err(|error| error.to_string())
    }
}

fn decode_history(entries: Vec<String>) -> Vec<StoredHistoryEntry> {
    let mut entries = entries
        .into_iter()
        .filter_map(|entry| serde_json::from_str::<StoredHistoryEntry>(&entry).ok())
        .collect::<Vec<_>>();
    entries.sort_by_key(history_sort_key);
    entries
}

fn history_sort_key(entry: &StoredHistoryEntry) -> (i64, i64) {
    let (raw_id, assistant_offset) = entry
        .id
        .strip_prefix("bot_")
        .map_or((entry.id.as_str(), 0_i64), |id| (id, 1_i64));
    raw_id.parse::<i64>().map_or(
        (entry.timestamp.saturating_mul(2), assistant_offset),
        |id| (id.saturating_mul(2), assistant_offset),
    )
}

fn role(stored_role: &str, id: &str) -> PromptRole {
    match stored_role {
        "assistant" => PromptRole::Assistant,
        "system" => PromptRole::System,
        "tool" => PromptRole::Tool,
        "user" => PromptRole::User,
        _ if id.starts_with("bot_") => PromptRole::Assistant,
        _ => PromptRole::User,
    }
}

fn decode_retrieved(
    rows: Vec<SearchRow>,
    recent_ids: &std::collections::HashSet<String>,
) -> Vec<RetrievedMessage> {
    rows.into_iter()
        .filter(|row| {
            row.fields
                .get("message_id")
                .is_none_or(|id| !recent_ids.contains(id.as_str()))
        })
        .filter_map(|row| {
            let text = row.fields.get("text")?.to_owned();
            (!text.is_empty()).then(|| RetrievedMessage {
                role: row
                    .fields
                    .get("role")
                    .cloned()
                    .unwrap_or_else(|| "user".to_owned()),
                text,
            })
        })
        .collect()
}

fn user_identity(input: &AiConversationInput) -> String {
    if input.sender_username.is_empty() {
        input.sender_first_name.clone()
    } else {
        format!("{} ({})", input.sender_first_name, input.sender_username)
    }
}

#[cfg(test)]
mod tests {
    use bot_core::ai_prompt::PromptRole;

    use super::{
        StoredHistoryEntry, build_compaction_view, decode_history, decode_summary_memory,
        history_sort_key, role,
    };

    #[test]
    fn history_decoder_sorts_user_then_bot_and_preserves_legacy_roles() {
        let entries = decode_history(vec![
            r#"{"id":"10","text":"later","timestamp":10,"role":"user"}"#.to_owned(),
            r#"{"id":"bot_9","text":"answer","timestamp":9}"#.to_owned(),
            r#"{"id":"9","text":"question","timestamp":9}"#.to_owned(),
            "malformed".to_owned(),
        ]);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].id, "9");
        assert_eq!(entries[1].id, "bot_9");
        assert_eq!(entries[2].id, "10");
        assert_eq!(
            role(&entries[1].role, &entries[1].id),
            PromptRole::Assistant
        );
        assert!(history_sort_key(&entries[0]) < history_sort_key(&entries[1]));
    }

    #[test]
    fn summary_memory_uses_only_entries_after_a_valid_marker() {
        let entries = vec![
            r#"{"id":"1","text":"old","timestamp":1,"role":"user"}"#.to_owned(),
            r#"{"id":"2","text":"marker","timestamp":2,"role":"assistant"}"#.to_owned(),
            r#"{"id":"3","text":"new","timestamp":3,"role":"user"}"#.to_owned(),
        ];
        let compacted = decode_summary_memory(
            entries.clone(),
            Some("prior summary".to_owned()),
            Some("2".to_owned()),
        );
        assert_eq!(compacted.history.len(), 1);
        assert_eq!(compacted.history[0].text, "new");

        let missing_marker = decode_summary_memory(
            entries,
            Some("prior summary".to_owned()),
            Some("missing".to_owned()),
        );
        assert_eq!(missing_marker.history.len(), 3);
    }

    #[test]
    fn plans_only_dropped_delta_and_keeps_current_context_rules() -> Result<(), &'static str> {
        let history = (1..=50)
            .map(|id| StoredHistoryEntry {
                id: id.to_string(),
                text: format!("message {id}"),
                timestamp: id,
                role: "user".to_owned(),
            })
            .collect::<Vec<_>>();
        let (first_visible, first_plan) = build_compaction_view(&history, &None, &None, "chat");
        assert_eq!(first_visible.len(), 50);
        let Some(first_plan) = first_plan else {
            return Err("first compaction should be planned");
        };
        assert_eq!(first_plan.messages.len(), 25);
        assert_eq!(first_plan.target_marker, "25");
        assert_eq!(first_plan.expected_marker, None);

        let (incremental_visible, incremental_plan) = build_compaction_view(
            &history,
            &Some("prior".to_owned()),
            &Some("5".to_owned()),
            "chat",
        );
        assert_eq!(incremental_visible.len(), 25);
        assert_eq!(incremental_visible[0].id, "26");
        let Some(incremental_plan) = incremental_plan else {
            return Err("incremental compaction should be planned");
        };
        assert_eq!(incremental_plan.messages.len(), 20);
        assert_eq!(incremental_plan.target_marker, "25");
        assert_eq!(incremental_plan.expected_marker.as_deref(), Some("5"));
        Ok(())
    }
}
