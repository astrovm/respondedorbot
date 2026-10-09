//! Redis and PostgreSQL adapters for foreground native AI conversations.

use std::collections::{HashMap, HashSet};

use bot_adapters::billing_read::{AiChargeResult, BillingRepository};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_creditless_cap::{
    CREDITLESS_CAP_TTL_SECONDS, RedisCreditlessCap, creditless_cap_key,
};
use bot_adapters::redis_message_state::{RedisMessageState, SearchRow};
use bot_core::ai_pricing::calculate_billing_for_segments;
use bot_core::ai_prompt::{HistoryMessage, PromptRole, RetrievedMessage};
use bot_core::ai_request::sanitize_assistant_text;
use bot_core::command_state::{
    BOT_MESSAGE_METADATA_SCHEMA_VERSION, BOT_MESSAGE_METADATA_TTL_SECONDS,
    CHAT_HISTORY_WRITE_LIMIT, CHAT_STATE_TTL_SECONDS,
};
use bot_core::message_state::{
    CHAT_HISTORY_MAX_MESSAGES, MESSAGE_HISTORY_SCHEMA_VERSION, bot_message_metadata_key,
    chat_compacted_until_key, chat_members_key, chat_summary_key, prepare_chat_member_payload,
    prepare_message_write,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::ai_dispatch::{AiConversationInput, AiReplyMetadata};
use crate::compaction_scheduler::MemoryCompactionPlan;
use crate::compaction_scheduler::PayerSource;
use crate::conversation::{
    ConversationBilling, ConversationMemory, ConversationState, ProviderSegmentRequest,
    ReserveDecision, ReserveDenial, ReserveRequest, SettlementRequest,
};
use crate::error_text;
use crate::reconciliation::ActiveOperationRegistry;

const COMPACTION_THRESHOLD: usize = 40;
// The configured history size is a small constant, far inside the Redis range.
const CHAT_HISTORY_LIMIT: i64 = CHAT_HISTORY_MAX_MESSAGES as i64;
const COMPACTION_KEEP: usize = 25;

pub struct RedisConversationState {
    state: RedisMessageState,
}

impl RedisConversationState {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, String> {
        RedisMessageState::new(endpoint)
            .map(|state| Self { state })
            .map_err(error_text)
    }
}

#[derive(Clone, Debug, Deserialize)]
struct StoredHistoryEntry {
    schema_version: u8,
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
        let Some(payload) = self.state.get_value(&key).map_err(error_text)? else {
            return Ok(None);
        };
        let value: Value = match serde_json::from_str(&payload) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let Some(value) = value.as_object() else {
            return Ok(None);
        };
        if value.get("schema_version").and_then(Value::as_u64)
            != Some(u64::from(BOT_MESSAGE_METADATA_SCHEMA_VERSION))
        {
            return Ok(None);
        }
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
        current_message_id: Option<&str>,
        max_history_messages: usize,
    ) -> Result<ConversationMemory, String> {
        let (entries, summary, marker) = self
            .state
            .get_history_with_summary(
                chat_id,
                CHAT_HISTORY_LIMIT,
                &chat_summary_key(chat_id),
                &chat_compacted_until_key(chat_id),
            )
            .map_err(error_text)?;
        let parsed = decode_history(entries);
        let summary = summary.filter(|value| !value.is_empty());
        // A marker without its summary is stale and must not trim history.
        let marker = marker.filter(|value| summary.is_some() && !value.is_empty());
        // Compaction planning sees the full delta; only the prompt view is trimmed.
        let (visible, compaction_plan) = build_compaction_view(&parsed, &summary, &marker, chat_id);
        let visible = prompt_history_window(visible, current_message_id, max_history_messages);
        let recent_ids = visible
            .iter()
            .map(|entry| entry.id.clone())
            .chain(current_message_id.map(str::to_owned))
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
        .map_err(error_text)?;
        let _stored = self
            .state
            .save_message(&plan, CHAT_STATE_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT)
            .map_err(error_text)?;
        if matches!(input.chat_type.as_str(), "group" | "supergroup") {
            let payload = prepare_chat_member_payload(
                &input.sender_first_name,
                &input.sender_username,
                input.timestamp,
            )
            .map_err(error_text)?;
            self.state
                .save_chat_member(
                    &chat_members_key(&chat_id),
                    &user_id,
                    &payload,
                    CHAT_STATE_TTL_SECONDS,
                )
                .map_err(error_text)?;
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
        let (entries, summary, marker) = self
            .state
            .get_history_with_summary(
                chat_id,
                limit,
                &chat_summary_key(chat_id),
                &chat_compacted_until_key(chat_id),
            )
            .map_err(error_text)?;
        let summary = summary.filter(|value| !value.is_empty());
        // A marker without its summary is stale and must not trim history.
        let marker = marker.filter(|value| summary.is_some() && !value.is_empty());
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
        .map_err(error_text)?;
        let _stored = self
            .state
            .save_message(&plan, CHAT_STATE_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT)
            .map_err(error_text)?;
        if let Some(sent_message_id) = sent_message_id {
            self.state
                .set_value(
                    &bot_message_metadata_key(&chat_id, &sent_message_id.to_string()),
                    &json!({
                        "schema_version": BOT_MESSAGE_METADATA_SCHEMA_VERSION,
                        "type": "ai"
                    })
                    .to_string(),
                    BOT_MESSAGE_METADATA_TTL_SECONDS,
                )
                .map_err(error_text)?;
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

/// Drops the message being answered (it is sent separately as the current
/// message) and keeps only the newest `max_history_messages` entries.
fn prompt_history_window(
    visible: Vec<StoredHistoryEntry>,
    current_message_id: Option<&str>,
    max_history_messages: usize,
) -> Vec<StoredHistoryEntry> {
    let mut visible = visible
        .into_iter()
        .filter(|entry| current_message_id.is_none_or(|current| entry.id != current))
        .collect::<Vec<_>>();
    let excess = visible.len().saturating_sub(max_history_messages);
    visible.drain(..excess);
    visible
}

pub struct PostgresConversationBilling {
    repository: BillingRepository,
    creditless_cap: Option<RedisCreditlessCap>,
    payer_by_operation: HashMap<String, PayerSource>,
    /// Credit units held per reservation of each operation, so a group-paid
    /// operation can move to the member in one piece.
    held_by_operation: HashMap<String, HashMap<String, i64>>,
    cap_key_by_operation: HashMap<String, String>,
    cap_checked_operations: HashSet<String>,
    onboarding_checked_operations: HashSet<String>,
    active_operations: Option<ActiveOperationRegistry>,
    active_marked_operations: HashSet<String>,
}

impl PostgresConversationBilling {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            repository: BillingRepository::new(database_url),
            creditless_cap: None,
            payer_by_operation: HashMap::new(),
            held_by_operation: HashMap::new(),
            cap_key_by_operation: HashMap::new(),
            cap_checked_operations: HashSet::new(),
            onboarding_checked_operations: HashSet::new(),
            active_operations: None,
            active_marked_operations: HashSet::new(),
        }
    }

    #[must_use]
    pub fn with_creditless_cap(mut self, creditless_cap: RedisCreditlessCap) -> Self {
        self.creditless_cap = Some(creditless_cap);
        self
    }

    #[must_use]
    pub fn with_active_operations(mut self, active_operations: ActiveOperationRegistry) -> Self {
        self.active_operations = Some(active_operations);
        self
    }

    fn release_operation_state(&mut self, operation_id: &str) {
        self.payer_by_operation.remove(operation_id);
        self.held_by_operation.remove(operation_id);
        self.onboarding_checked_operations.remove(operation_id);
        self.cap_checked_operations.remove(operation_id);
        self.cap_key_by_operation.remove(operation_id);
        if self.active_marked_operations.remove(operation_id)
            && let Some(active_operations) = self.active_operations.as_ref()
        {
            active_operations.mark_inactive(operation_id);
        }
    }

    fn charge_reservation(
        &self,
        request: &ReserveRequest,
        amount: i32,
        source: Option<&str>,
    ) -> Result<AiChargeResult, String> {
        self.repository
            .charge_ai_credits(
                request.user_id,
                request.chat_id,
                amount,
                "ai_reserve",
                &request.metadata,
                source,
                Some(&request.reservation_id),
                &request.operation_id,
            )
            .map_err(error_text)
    }

    /// With the group paying first, the group pays until the member reaches
    /// their hourly limit, then their own credits do. The message is counted
    /// before anything is charged, so two at once can't both fit under it.
    fn group_first_source(
        &mut self,
        request: &ReserveRequest,
        chat_id: i64,
    ) -> Result<&'static str, String> {
        let limit = request.creditless_user_hourly_limit;
        let Some(creditless_cap) = self.creditless_cap.as_ref().filter(|_| limit >= 0) else {
            return Ok("chat");
        };
        let cap_key = reservation_cap_key(request, chat_id);
        // Recorded before Redis is asked, so if the answer is lost after the
        // message was counted, aborting the operation still gives it back.
        self.cap_key_by_operation
            .insert(request.operation_id.clone(), cap_key.clone());
        let count = creditless_cap
            .admit_once(&cap_key, &request.operation_id, CREDITLESS_CAP_TTL_SECONDS)
            .map_err(error_text)?;
        if count > limit {
            creditless_cap
                .refund_once(&cap_key, &request.operation_id)
                .map_err(error_text)?;
            self.cap_key_by_operation.remove(&request.operation_id);
            return Ok("user");
        }
        // Already counted, so the check after charging the group skips it.
        self.cap_checked_operations
            .insert(request.operation_id.clone());
        Ok("chat")
    }

    /// With the group paying first, a later reservation the group can't
    /// cover moves the whole operation to the member: the group gets back
    /// what it held and its hourly slot, and the member's credits hold all of
    /// it. An operation settles with one payer, so it can't be split.
    fn move_operation_to_member(
        &mut self,
        request: &ReserveRequest,
        amount: i32,
        refused: AiChargeResult,
    ) -> Result<AiChargeResult, String> {
        let held = self
            .held_by_operation
            .get(&request.operation_id)
            .map_or(0, |holds| holds.values().sum::<i64>());
        let total = held + i64::from(amount);
        if refused.user_balance < total {
            return Ok(refused);
        }
        let held = i32::try_from(held)
            .map_err(|_| "AI reservation exceeds the database range".to_owned())?;
        let total = i32::try_from(total)
            .map_err(|_| "AI reservation exceeds the database range".to_owned())?;
        let mut refund_metadata = request.metadata.clone();
        refund_metadata.insert("reason".to_owned(), json!("group_short_moved_to_member"));
        self.repository
            .refund_ai_charge(
                request.user_id,
                request.chat_id,
                held,
                "chat",
                "ai_refund",
                &refund_metadata,
                Some(&format!("{}:moved_to_member", request.operation_id)),
                &request.operation_id,
            )
            .map_err(error_text)?;
        self.refund_creditless_cap(&request.operation_id)?;
        self.cap_key_by_operation.remove(&request.operation_id);
        self.held_by_operation.remove(&request.operation_id);
        self.payer_by_operation.remove(&request.operation_id);
        self.repository
            .charge_ai_credits(
                request.user_id,
                request.chat_id,
                total,
                "ai_reserve",
                &request.metadata,
                Some("user"),
                Some(&format!("{}:member", request.reservation_id)),
                &request.operation_id,
            )
            .map_err(error_text)
    }

    fn refund_creditless_cap(&self, operation_id: &str) -> Result<(), String> {
        if let Some(cap_key) = self.cap_key_by_operation.get(operation_id)
            && let Some(creditless_cap) = self.creditless_cap.as_ref()
        {
            creditless_cap
                .refund_once(cap_key, operation_id)
                .map_err(error_text)?;
        }
        Ok(())
    }
}

/// The member's hourly counter in the chat the message came from.
fn reservation_cap_key(request: &ReserveRequest, chat_id: i64) -> String {
    let origin_chat_id = request
        .metadata
        .get("origin_chat_id")
        .map_or_else(|| chat_id.to_string(), value_as_key_component);
    creditless_cap_key(&origin_chat_id, request.user_id)
}

impl ConversationBilling for PostgresConversationBilling {
    fn reserve(&mut self, request: ReserveRequest) -> Result<ReserveDecision, String> {
        if self
            .onboarding_checked_operations
            .insert(request.operation_id.clone())
        {
            let _grant = self
                .repository
                .grant_onboarding_if_needed(request.user_id, 300);
        }
        let amount = i32::try_from(request.amount)
            .map_err(|_| "AI reservation exceeds the database range".to_owned())?;
        let cached_source = self
            .payer_by_operation
            .get(&request.operation_id)
            .copied()
            .map(|source| match source {
                PayerSource::User => "user",
                PayerSource::Chat => "chat",
            });
        let group_first_chat = request
            .chat_id
            .filter(|_| cached_source.is_none() && request.group_pays_first);
        let requested_source = match group_first_chat {
            Some(chat_id) => Some(self.group_first_source(&request, chat_id)?),
            None => cached_source,
        };
        let mut result = self.charge_reservation(&request, amount, requested_source)?;
        if group_first_chat.is_some() && !result.ok {
            if requested_source == Some("user") {
                // The member used up what the group pays for this hour and
                // their own credits don't cover the rest.
                self.release_operation_state(&request.operation_id);
                return Ok(ReserveDecision {
                    authorized: false,
                    user_balance: result.user_balance,
                    chat_balance: result.chat_balance,
                    source: None,
                    denial: Some(ReserveDenial::CreditlessHourlyCap {
                        limit: request.creditless_user_hourly_limit,
                    }),
                });
            }
            // The group's balance ran out, so the member's own credits pay
            // and the message no longer counts toward the group's hourly limit.
            self.refund_creditless_cap(&request.operation_id)?;
            self.cap_key_by_operation.remove(&request.operation_id);
            result = self.charge_reservation(&request, amount, Some("user"))?;
        } else if cached_source == Some("chat") && request.group_pays_first && !result.ok {
            result = self.move_operation_to_member(&request, amount, result)?;
        }
        if result.ok {
            self.held_by_operation
                .entry(request.operation_id.clone())
                .or_default()
                .insert(request.reservation_id.clone(), result.amount);
        }
        let source = match result.source.as_deref() {
            Some("user") => Some(PayerSource::User),
            Some("chat") => Some(PayerSource::Chat),
            _ => None,
        };

        if result.ok
            && source == Some(PayerSource::Chat)
            && request.creditless_user_hourly_limit >= 0
            && let Some(chat_id) = request.chat_id
            && let Some(creditless_cap) = self.creditless_cap.as_ref()
            && self
                .cap_checked_operations
                .insert(request.operation_id.clone())
        {
            let cap_key = reservation_cap_key(&request, chat_id);
            self.cap_key_by_operation
                .insert(request.operation_id.clone(), cap_key.clone());
            let count = creditless_cap
                .admit_once(&cap_key, &request.operation_id, CREDITLESS_CAP_TTL_SECONDS)
                .map_err(error_text)?;
            if count > request.creditless_user_hourly_limit {
                let mut refund_metadata = request.metadata.clone();
                refund_metadata.insert("reason".to_owned(), json!("creditless_hourly_cap"));
                refund_metadata.insert("settlement_id".to_owned(), json!(&request.reservation_id));
                let refund_id = format!("{}:creditless_cap_refund", request.reservation_id);
                let refund = self
                    .repository
                    .refund_ai_charge(
                        request.user_id,
                        request.chat_id,
                        amount,
                        "chat",
                        "ai_refund",
                        &refund_metadata,
                        Some(&refund_id),
                        &request.operation_id,
                    )
                    .map_err(error_text)?;
                self.release_operation_state(&request.operation_id);
                return Ok(ReserveDecision {
                    authorized: false,
                    user_balance: refund.user_balance,
                    chat_balance: refund.chat_balance,
                    source,
                    denial: Some(ReserveDenial::CreditlessHourlyCap {
                        limit: request.creditless_user_hourly_limit,
                    }),
                });
            }
        }

        if result.ok
            && let Some(source) = source
        {
            self.payer_by_operation
                .insert(request.operation_id.clone(), source);
            if self
                .active_marked_operations
                .insert(request.operation_id.clone())
                && let Some(active_operations) = self.active_operations.as_ref()
            {
                active_operations.mark_active(&request.operation_id);
            }
        }
        Ok(ReserveDecision {
            authorized: result.ok,
            user_balance: result.user_balance,
            chat_balance: result.chat_balance,
            source,
            denial: None,
        })
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
            .map_err(error_text)
    }

    fn settle(&mut self, request: SettlementRequest) -> Result<(), String> {
        let operation_id = request.operation_id.clone();
        let result = (|| {
            let mut metadata = Map::from_iter([
                ("operation_id".to_owned(), json!(operation_id)),
                ("reason".to_owned(), json!(request.reason)),
                ("delivered".to_owned(), json!(request.delivered)),
            ]);
            let mut pricing_complete = true;
            if !request.billing_segments.is_empty() {
                let pricing =
                    calculate_billing_for_segments(&Value::Array(request.billing_segments.clone()))
                        .map_err(error_text)?;
                pricing_complete =
                    pricing.get("pricing_complete").and_then(Value::as_bool) == Some(true);
                metadata.insert(
                    "billing_segments".to_owned(),
                    Value::Array(request.billing_segments),
                );
                copy_pricing_metadata(&mut metadata, &pricing);
            }
            if pricing_complete {
                // Refund the ephemeral hourly allowance before committing the
                // durable zero-cost settlement. A Redis failure then leaves the
                // database operation replayable, and the Redis operation marker
                // makes a successful refund safe to retry.
                if request.actual_credit_units == 0 {
                    self.refund_creditless_cap(&operation_id)?;
                }
                self.repository
                    .settle_ai_operation_once(
                        request.user_id,
                        request.chat_id,
                        &operation_id,
                        request.actual_credit_units,
                        &metadata,
                    )
                    .map_err(error_text)?;
            }
            Ok(())
        })();

        // In-memory admission state is only a guard around the live provider
        // operation. Always release it, including pricing, database, and Redis
        // failures, so the durable reconciler can repair an unsettled reserve.
        self.release_operation_state(&operation_id);
        result
    }

    fn abort_operation(&mut self, operation_id: &str) -> Result<(), String> {
        let cap_refund = self.refund_creditless_cap(operation_id);
        self.release_operation_state(operation_id);
        cap_refund
    }

    fn release_operation(&mut self, operation_id: &str) {
        self.release_operation_state(operation_id);
    }

    fn personal_balance(&mut self, user_id: i64) -> Result<Option<i64>, String> {
        self.repository
            .get_balance("user", user_id)
            .map(Some)
            .map_err(error_text)
    }
}

fn copy_pricing_metadata(metadata: &mut Map<String, Value>, pricing: &Value) {
    for key in [
        "pricing_version",
        "raw_usd_micros",
        "model_breakdown",
        "tool_breakdown",
        "segment_breakdown",
        "pricing_complete",
    ] {
        if let Some(value) = pricing.get(key) {
            metadata.insert(key.to_owned(), value.clone());
        }
    }
}

fn value_as_key_component(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| value.to_string(), ToOwned::to_owned)
}

fn decode_history(entries: Vec<String>) -> Vec<StoredHistoryEntry> {
    let mut entries = entries
        .into_iter()
        .filter_map(|entry| serde_json::from_str::<StoredHistoryEntry>(&entry).ok())
        .filter(|entry| entry.schema_version == MESSAGE_HISTORY_SCHEMA_VERSION)
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
    use crate::test_env::fresh_op;
    use std::time::{SystemTime, UNIX_EPOCH};

    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_core::ai_prompt::PromptRole;
    use bot_core::locale::Locale;
    use bot_core::telegram_input::{ChatId, MessageId, UserId};
    use serde_json::json;

    use super::{
        MESSAGE_HISTORY_SCHEMA_VERSION, PayerSource, PostgresConversationBilling,
        RedisConversationState, StoredHistoryEntry, build_compaction_view, copy_pricing_metadata,
        decode_history, decode_retrieved, decode_summary_memory, history_sort_key,
        prompt_history_window, role, user_identity,
    };
    use crate::ai_dispatch::AiConversationInput;
    use crate::conversation::{
        ConversationBilling, ConversationState, ReserveDenial, ReserveRequest, SettlementRequest,
    };
    use crate::reconciliation::ActiveOperationRegistry;

    #[test]
    fn charge_metadata_keeps_the_pricing_details_and_nothing_else() {
        let pricing = json!({
            "pricing_version": "synthetic-version",
            "raw_usd_micros": 120,
            "model_breakdown": [{"model": "synthetic/model"}],
            "tool_breakdown": [],
            "segment_breakdown": [{"segment": 0}],
            "pricing_complete": true,
            "charged_credit_units": 3,
        });
        let mut metadata = serde_json::Map::new();
        copy_pricing_metadata(&mut metadata, &pricing);
        assert_eq!(
            serde_json::Value::Object(metadata),
            json!({
                "pricing_version": "synthetic-version",
                "raw_usd_micros": 120,
                "model_breakdown": [{"model": "synthetic/model"}],
                "tool_breakdown": [],
                "segment_breakdown": [{"segment": 0}],
                "pricing_complete": true,
            })
        );
    }

    #[test]
    fn settlement_releases_all_ephemeral_guards_when_pricing_fails() {
        let operation_id = "ai:42:7:88";
        let active = ActiveOperationRegistry::default();
        active.mark_active(operation_id);
        let mut billing = PostgresConversationBilling::new("postgresql://synthetic.invalid/db")
            .with_active_operations(active.clone());
        billing
            .payer_by_operation
            .insert(operation_id.to_owned(), PayerSource::User);
        billing
            .active_marked_operations
            .insert(operation_id.to_owned());
        billing
            .cap_key_by_operation
            .insert(operation_id.to_owned(), "cap-key".to_owned());

        let result = billing.settle(SettlementRequest {
            user_id: 88,
            chat_id: None,
            operation_id: operation_id.to_owned(),
            actual_credit_units: 1,
            delivered: true,
            reason: "synthetic".to_owned(),
            billing_segments: vec![json!("invalid segment")],
        });

        assert!(result.is_err());
        assert!(!active.is_active(operation_id));
        assert!(!billing.payer_by_operation.contains_key(operation_id));
        assert!(!billing.cap_key_by_operation.contains_key(operation_id));
    }

    fn integration_redis_endpoint() -> Option<RedisEndpoint> {
        let port = std::env::var("TEST_REDIS_PORT").ok()?.parse().ok()?;
        Some(RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or(String::from("127.0.0.1")),
            port,
            // Empty passwords are ignored by the Redis client.
            password: std::env::var("TEST_REDIS_PASSWORD").ok(),
        })
    }

    fn conversation_input(chat_id: i64, message_id: i64, locale: Locale) -> AiConversationInput {
        AiConversationInput {
            chat_id: ChatId(chat_id),
            message_id: MessageId(message_id),
            chat_type: "supergroup".to_owned(),
            chat_title: "Synthetic chat".to_owned(),
            sender_id: UserId(42),
            sender_first_name: "Synthetic".to_owned(),
            sender_username: "synthetic_user".to_owned(),
            message_text: "synthetic message".to_owned(),
            command: String::new(),
            reply_to_message_id: Some(MessageId(message_id - 1)),
            reply_context: Some("earlier synthetic message".to_owned()),
            has_reply: true,
            visual_media_kind: None,
            audio_media_kind: None,
            photo_file_id: None,
            audio_file_id: None,
            audio_duration_seconds: None,
            locale,
            timezone_offset_hours: -3,
            creditless_user_hourly_limit: 10,
            group_pays_first: false,
            timestamp: 1_700_000_000 + message_id,
            spontaneous: false,
            link_context: None,
        }
    }

    #[test]
    fn redis_conversation_state_round_trips_live_conversation_memory() -> TestResult {
        integration_redis_endpoint().map_or(Ok(()), round_trip_live_conversation_memory)
    }

    fn round_trip_live_conversation_memory(endpoint: RedisEndpoint) -> TestResult {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let chat_id = -i64::try_from(nonce % 1_000_000_000)?;
        let mut state = RedisConversationState::new(&endpoint)?;
        let spanish = conversation_input(chat_id, 2, Locale::Es);
        state.record_incoming(&spanish)?;
        state.record_outgoing(&spanish, Some(3), "synthetic assistant reply")?;

        let english = conversation_input(chat_id, 4, Locale::En);
        state.record_incoming(&english)?;
        state.record_outgoing(&english, None, "second synthetic reply")?;

        let memory = state.load_memory(&chat_id.to_string(), "synthetic", Some("2"), None, 20)?;
        assert_eq!(memory.history.len(), 4);
        let current =
            state.load_memory(&chat_id.to_string(), "synthetic", Some("2"), Some("4"), 20)?;
        assert_eq!(current.history.len(), 3);
        let capped = state.load_memory(&chat_id.to_string(), "", None, None, 2)?;
        assert_eq!(capped.history.len(), 2);
        assert_eq!(capped.history[1].text, "second synthetic reply");
        assert!(memory.history.iter().any(|entry| {
            entry.role == PromptRole::Assistant && entry.text == "synthetic assistant reply"
        }));
        assert_eq!(
            state.reply_metadata(&chat_id.to_string(), "3")?,
            Some(crate::ai_dispatch::AiReplyMetadata {
                kind: "ai".to_owned(),
                uses_ai: false,
            })
        );
        assert!(state.reply_metadata(&chat_id.to_string(), "999")?.is_none());

        let malformed_key =
            bot_core::message_state::bot_message_metadata_key(&chat_id.to_string(), "malformed");
        state.state.set_value(&malformed_key, "not json", 60)?;
        assert!(
            state
                .reply_metadata(&chat_id.to_string(), "malformed")?
                .is_none()
        );
        state.state.set_value(&malformed_key, "[]", 60)?;
        assert!(
            state
                .reply_metadata(&chat_id.to_string(), "malformed")?
                .is_none()
        );
        state
            .state
            .set_value(&malformed_key, r#"{"type":"ai"}"#, 60)?;
        assert!(
            state
                .reply_metadata(&chat_id.to_string(), "malformed")?
                .is_none()
        );

        let summary_key = bot_core::message_state::chat_summary_key(&chat_id.to_string());
        let marker_key = bot_core::message_state::chat_compacted_until_key(&chat_id.to_string());
        state
            .state
            .set_value(&summary_key, "synthetic summary", 60)?;
        state.state.set_value(&marker_key, "2", 60)?;
        let compacted = state.load_memory(&chat_id.to_string(), "", None, None, 20)?;
        assert_eq!(compacted.summary.as_deref(), Some("synthetic summary"));

        let summary = state.load_summary_memory(&chat_id.to_string(), 20)?;
        assert!(!summary.history.is_empty());

        let mut empty = conversation_input(chat_id, 5, Locale::En);
        empty.message_text.clear();
        state.record_incoming(&empty)?;

        let mut anonymous = conversation_input(chat_id, 6, Locale::Es);
        anonymous.chat_type = "private".to_owned();
        anonymous.sender_first_name.clear();
        anonymous.sender_username.clear();
        state.record_incoming(&anonymous)?;
        anonymous.message_id = MessageId(7);
        anonymous.locale = Locale::En;
        state.record_incoming(&anonymous)?;
        anonymous.message_id = MessageId(8);
        anonymous.reply_context = None;
        state.record_incoming(&anonymous)?;
        Ok(())
    }

    #[test]
    fn retrieved_rows_and_user_identity_normalize_optional_fields() {
        use std::collections::{BTreeMap, HashSet};

        let rows = vec![
            bot_adapters::redis_message_state::SearchRow {
                key: "recent".to_owned(),
                fields: BTreeMap::from([
                    ("message_id".to_owned(), "2".to_owned()),
                    ("text".to_owned(), "duplicate".to_owned()),
                ]),
            },
            bot_adapters::redis_message_state::SearchRow {
                key: "missing-role".to_owned(),
                fields: BTreeMap::from([("text".to_owned(), "useful result".to_owned())]),
            },
            bot_adapters::redis_message_state::SearchRow {
                key: "empty".to_owned(),
                fields: BTreeMap::from([("text".to_owned(), String::new())]),
            },
        ];
        let decoded = decode_retrieved(rows, &HashSet::from(["2".to_owned()]));
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].role, "user");
        assert_eq!(decoded[0].text, "useful result");

        let mut input = conversation_input(1, 2, Locale::En);
        input.sender_username.clear();
        assert_eq!(user_identity(&input), "Synthetic");
    }

    #[test]
    fn prompt_history_window_drops_current_message_and_keeps_newest_entries() {
        let entries = decode_history(
            (1..=5)
                .map(|id| {
                    format!(r#"{{"schema_version":1,"id":"{id}","text":"m{id}","timestamp":{id}}}"#)
                })
                .collect(),
        );
        let ids = |window: Vec<StoredHistoryEntry>| {
            window.into_iter().map(|entry| entry.id).collect::<Vec<_>>()
        };
        assert_eq!(
            ids(prompt_history_window(entries.clone(), Some("5"), 3)),
            vec!["2", "3", "4"]
        );
        assert_eq!(
            ids(prompt_history_window(entries.clone(), None, 2)),
            vec!["4", "5"]
        );
        assert_eq!(
            ids(prompt_history_window(entries.clone(), Some("missing"), 10)).len(),
            5
        );
        assert!(prompt_history_window(entries, None, 0).is_empty());
    }

    #[test]
    fn history_decoder_requires_current_records_and_sorts_user_then_bot() {
        let entries = decode_history(vec![
            r#"{"schema_version":1,"id":"10","text":"later","timestamp":10,"role":"user"}"#
                .to_owned(),
            r#"{"schema_version":1,"id":"bot_9","text":"answer","timestamp":9}"#.to_owned(),
            r#"{"schema_version":1,"id":"9","text":"question","timestamp":9}"#.to_owned(),
            r#"{"id":"8","text":"old","timestamp":8,"role":"user"}"#.to_owned(),
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
            r#"{"schema_version":1,"id":"1","text":"old","timestamp":1,"role":"user"}"#.to_owned(),
            r#"{"schema_version":1,"id":"2","text":"marker","timestamp":2,"role":"assistant"}"#
                .to_owned(),
            r#"{"schema_version":1,"id":"3","text":"new","timestamp":3,"role":"user"}"#.to_owned(),
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
                schema_version: MESSAGE_HISTORY_SCHEMA_VERSION,
                id: id.to_string(),
                text: format!("message {id}"),
                timestamp: id,
                role: "user".to_owned(),
            })
            .collect::<Vec<_>>();
        let (first_visible, first_plan) = build_compaction_view(&history, &None, &None, "chat");
        assert_eq!(first_visible.len(), 50);
        let first_plan = first_plan.ok_or("first compaction should be planned")?;
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
        let incremental_plan =
            incremental_plan.ok_or("incremental compaction should be planned")?;
        assert_eq!(incremental_plan.messages.len(), 20);
        assert_eq!(incremental_plan.target_marker, "25");
        assert_eq!(incremental_plan.expected_marker.as_deref(), Some("5"));
        Ok(())
    }

    #[test]
    fn unknown_stored_roles_fall_back_by_message_id() {
        assert_eq!(role("", "12"), PromptRole::User);
        assert_eq!(role("synthetic", "bot_12"), PromptRole::Assistant);
        assert_eq!(role("tool", "bot_12"), PromptRole::Tool);
    }

    #[test]
    fn aborting_releases_guards_even_when_the_creditless_refund_fails() -> TestResult {
        let operation_id = "ai:42:7:99";
        let unreachable_cap =
            bot_adapters::redis_creditless_cap::RedisCreditlessCap::new(&RedisEndpoint {
                host: "127.0.0.1".to_owned(),
                port: 1,
                password: None,
            })?;
        let active = ActiveOperationRegistry::default();
        active.mark_active(operation_id);
        let mut billing = PostgresConversationBilling::new("postgresql://synthetic.invalid/db")
            .with_creditless_cap(unreachable_cap)
            .with_active_operations(active.clone());
        billing
            .payer_by_operation
            .insert(operation_id.to_owned(), PayerSource::Chat);
        billing
            .active_marked_operations
            .insert(operation_id.to_owned());
        billing
            .cap_key_by_operation
            .insert(operation_id.to_owned(), "synthetic-cap-key".to_owned());

        let result = billing.abort_operation(operation_id);

        assert!(
            matches!(&result, Err(error) if !error.is_empty()),
            "{result:?}"
        );
        assert!(!active.is_active(operation_id));
        assert!(!billing.payer_by_operation.contains_key(operation_id));
        assert!(!billing.cap_key_by_operation.contains_key(operation_id));
        // Without a cap key there is nothing to refund, so a retry succeeds.
        assert_eq!(billing.abort_operation(operation_id), Ok(()));
        Ok(())
    }

    fn user_reserve(
        user_id: i64,
        operation_id: &str,
        reservation: &str,
        amount: i64,
    ) -> ReserveRequest {
        ReserveRequest {
            user_id,
            chat_id: None,
            operation_id: operation_id.to_owned(),
            reservation_id: format!("{operation_id}:{reservation}"),
            amount,
            creditless_user_hourly_limit: 10,
            group_pays_first: false,
            metadata: serde_json::Map::from_iter([(
                "operation_id".to_owned(),
                json!(operation_id),
            )]),
        }
    }

    #[test]
    fn postgres_reserves_keep_the_first_payer_and_leave_denials_unmarked() -> TestResult {
        std::env::var("TEST_DATABASE_URL").map_or(Ok(()), reserve_against_postgres)
    }

    fn reserve_against_postgres(database_url: String) -> TestResult {
        let database_url = database_url.as_str();
        bot_adapters::billing_schema::BillingSchemaRepository::new(database_url).ensure_schema()?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let suffix = i64::try_from(nonce % 100_000_000)?;
        let user_id = 6_310_000_000_000_i64 + suffix;
        let denied_operation = format!("synthetic-denied:{nonce}");
        let paid_operation = format!("synthetic-paid:{nonce}");
        let active = ActiveOperationRegistry::default();
        let mut billing =
            PostgresConversationBilling::new(database_url).with_active_operations(active.clone());

        let repository = bot_adapters::billing_read::BillingRepository::new(database_url);
        repository.mint_user_credits(user_id, 1_000, None, &fresh_op())?;

        // More than the minted credits plus any onboarding grant.
        let denied = billing.reserve(user_reserve(user_id, &denied_operation, "reserve", 5_000))?;
        assert!(!denied.authorized);
        assert_eq!(denied.source, None);
        assert!(denied.denial.is_none());
        assert!(denied.user_balance >= 1_000);
        assert!(!active.is_active(&denied_operation));
        assert!(!billing.payer_by_operation.contains_key(&denied_operation));

        let first = billing.reserve(user_reserve(user_id, &paid_operation, "first", 100))?;
        assert!(first.authorized);
        assert_eq!(first.source, Some(PayerSource::User));
        assert!(active.is_active(&paid_operation));

        // Later reserves for the same operation request the recorded payer.
        let second = billing.reserve(user_reserve(user_id, &paid_operation, "second", 50))?;
        assert!(second.authorized);
        assert_eq!(second.source, Some(PayerSource::User));
        assert_eq!(second.user_balance, first.user_balance - 50);
        assert_eq!(
            billing.personal_balance(user_id)?,
            Some(second.user_balance)
        );

        billing.release_operation(&paid_operation);
        assert!(!active.is_active(&paid_operation));
        assert!(!billing.payer_by_operation.contains_key(&paid_operation));
        Ok(())
    }

    #[test]
    fn unreachable_ledger_fails_settlement_but_still_releases_guards() {
        let operation_id = "ai:42:7:77";
        let active = ActiveOperationRegistry::default();
        let mut billing = PostgresConversationBilling::new(
            "postgresql://synthetic:synthetic@127.0.0.1:1/synthetic?sslmode=disable",
        )
        .with_active_operations(active.clone());
        active.mark_active(operation_id);
        billing
            .payer_by_operation
            .insert(operation_id.to_owned(), PayerSource::User);
        billing
            .active_marked_operations
            .insert(operation_id.to_owned());

        let result = billing.settle(SettlementRequest {
            user_id: 88,
            chat_id: None,
            operation_id: operation_id.to_owned(),
            actual_credit_units: 5,
            delivered: true,
            reason: "synthetic".to_owned(),
            billing_segments: Vec::new(),
        });

        assert!(
            matches!(&result, Err(error) if !error.is_empty()),
            "{result:?}"
        );
        assert!(!active.is_active(operation_id));
        assert!(!billing.payer_by_operation.contains_key(operation_id));
        assert!(!billing.active_marked_operations.contains(operation_id));
    }

    #[test]
    fn pending_provider_cost_leaves_the_operation_open_for_reconciliation() {
        let operation_id = "ai:42:7:66";
        let active = ActiveOperationRegistry::default();
        // Any ledger access would fail against this address.
        let mut billing = PostgresConversationBilling::new(
            "postgresql://synthetic:synthetic@127.0.0.1:1/synthetic?sslmode=disable",
        )
        .with_active_operations(active.clone());
        active.mark_active(operation_id);
        billing
            .active_marked_operations
            .insert(operation_id.to_owned());

        let result = billing.settle(SettlementRequest {
            user_id: 88,
            chat_id: None,
            operation_id: operation_id.to_owned(),
            actual_credit_units: 0,
            delivered: true,
            reason: "synthetic".to_owned(),
            billing_segments: vec![json!({
                "kind": "chat",
                "model": "deepseek/deepseek-v4.1-flash",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "source": "openrouter",
                "metadata": {
                    "provider": "openrouter",
                    "provider_generation_id": "generation-pending",
                    "provider_usage_pending": true
                }
            })],
        });

        assert_eq!(result, Ok(()));
        assert!(!active.is_active(operation_id));
        assert!(!billing.active_marked_operations.contains(operation_id));
    }

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn unreachable_redis() -> RedisEndpoint {
        RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port: 1,
            password: None,
        }
    }

    #[test]
    fn unreachable_redis_fails_every_conversation_state_operation() -> TestResult {
        let mut state = RedisConversationState::new(&unreachable_redis())?;
        let input = conversation_input(-100, 2, Locale::En);
        let failures = [
            state.reply_metadata("-100", "1").err(),
            state.load_memory("-100", "", None, None, 20).err(),
            state.load_summary_memory("-100", 20).err(),
            state.record_incoming(&input).err(),
            state.record_outgoing(&input, Some(3), "reply").err(),
        ];
        for failure in failures {
            assert!(
                matches!(&failure, Some(error) if !error.is_empty()),
                "{failure:?}"
            );
        }
        // Oversized limits are rejected before Redis is contacted.
        assert_eq!(
            state.load_summary_memory("-100", usize::MAX).err(),
            Some("summary history limit exceeds the Redis range".to_owned())
        );
        Ok(())
    }

    fn unreachable_billing() -> PostgresConversationBilling {
        PostgresConversationBilling::new(
            "postgresql://synthetic:synthetic@127.0.0.1:1/synthetic?sslmode=disable",
        )
    }

    fn user_request(operation_id: &str, amount: i64) -> ReserveRequest {
        ReserveRequest {
            user_id: 88,
            chat_id: None,
            operation_id: operation_id.to_owned(),
            reservation_id: format!("{operation_id}:reserve"),
            amount,
            creditless_user_hourly_limit: 10,
            group_pays_first: false,
            metadata: serde_json::Map::new(),
        }
    }

    #[test]
    fn reservations_reject_oversized_amounts_and_ledger_failures() {
        let mut billing = unreachable_billing();
        assert_eq!(
            billing
                .reserve(user_request("ai:88:1", i64::from(i32::MAX) + 1))
                .err(),
            Some("AI reservation exceeds the database range".to_owned())
        );
        let failed = billing.reserve(user_request("ai:88:2", 10));
        assert!(
            matches!(&failed, Err(error) if !error.is_empty()),
            "{failed:?}"
        );
        assert!(billing.payer_by_operation.is_empty());
        let balance = billing.personal_balance(88);
        assert!(
            matches!(&balance, Err(error) if !error.is_empty()),
            "{balance:?}"
        );
        let recorded = billing.record_segment(crate::conversation::ProviderSegmentRequest {
            user_id: 88,
            chat_id: None,
            operation_id: "ai:88:2".to_owned(),
            segment_id: "ai:88:2:segment".to_owned(),
            segment: json!({"kind": "chat"}),
        });
        assert!(
            matches!(&recorded, Err(error) if !error.is_empty()),
            "{recorded:?}"
        );
    }

    #[test]
    fn zero_cost_settlement_refunds_the_allowance_before_the_ledger() {
        let operation_id = "ai:42:7:55";
        let mut billing = unreachable_billing();
        billing
            .payer_by_operation
            .insert(operation_id.to_owned(), PayerSource::User);
        let result = billing.settle(SettlementRequest {
            user_id: 88,
            chat_id: None,
            operation_id: operation_id.to_owned(),
            actual_credit_units: 0,
            delivered: false,
            reason: "synthetic".to_owned(),
            billing_segments: Vec::new(),
        });
        // Without a creditless admission the refund is a no-op, so the
        // ledger write is attempted (and fails here).
        assert!(
            matches!(&result, Err(error) if !error.is_empty()),
            "{result:?}"
        );
        assert!(billing.payer_by_operation.is_empty());
    }

    #[test]
    fn postgres_chat_payer_is_capped_per_hour_and_kept_for_the_operation() -> TestResult {
        std::env::var("TEST_DATABASE_URL")
            .ok()
            .zip(integration_redis_endpoint())
            .map_or(Ok(()), chat_payer_scenario)
    }

    fn chat_payer_scenario((database_url, endpoint): (String, RedisEndpoint)) -> TestResult {
        use bot_adapters::redis_creditless_cap::{RedisCreditlessCap, creditless_cap_key};

        bot_adapters::billing_schema::BillingSchemaRepository::new(&database_url)
            .ensure_schema()?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let suffix = i64::try_from(nonce % 100_000_000)?;
        let user_id = 6_320_000_000_000_i64 + suffix;
        let chat_id = -6_330_000_000_000_i64 - suffix;
        let repository = bot_adapters::billing_read::BillingRepository::new(&database_url);
        repository.mint_user_credits(user_id, 1_000, None, &fresh_op())?;
        assert!(
            repository
                .transfer_user_to_chat(user_id, chat_id, 1_000, &fresh_op())?
                .transferred
        );
        let request = |operation_id: &str, reservation: &str, amount: i64, limit: i64| {
            ReserveRequest {
                user_id,
                chat_id: Some(chat_id),
                operation_id: operation_id.to_owned(),
                reservation_id: format!("{operation_id}:{reservation}"),
                amount,
                creditless_user_hourly_limit: limit,
                group_pays_first: false,
                // Without origin metadata the cap is keyed by the paying chat.
                metadata: serde_json::Map::new(),
            }
        };
        let cap_key = creditless_cap_key(&chat_id.to_string(), user_id);
        let cap_reader = RedisCreditlessCap::new(&endpoint)?;
        let active = ActiveOperationRegistry::default();
        let mut billing = PostgresConversationBilling::new(&database_url)
            .with_creditless_cap(RedisCreditlessCap::new(&endpoint)?)
            .with_active_operations(active.clone());

        // More than any onboarding grant, so the chat pays.
        let paid = format!("synthetic-chat-paid:{nonce}");
        let first = billing.reserve(request(&paid, "first", 400, 5))?;
        assert!(first.authorized);
        assert_eq!(first.source, Some(PayerSource::Chat));
        assert_eq!(first.chat_balance, 600);
        assert!(active.is_active(&paid));
        assert_eq!(cap_reader.count(&cap_key)?, Some(1));

        // The operation keeps its payer and its single hourly admission.
        let second = billing.reserve(request(&paid, "second", 100, 5))?;
        assert!(second.authorized);
        assert_eq!(second.source, Some(PayerSource::Chat));
        assert_eq!(second.chat_balance, 500);
        assert_eq!(cap_reader.count(&cap_key)?, Some(1));

        // A new operation past a zero hourly limit is refunded and denied.
        let blocked = format!("synthetic-chat-blocked:{nonce}");
        let denied = billing.reserve(request(&blocked, "first", 400, 0))?;
        assert!(!denied.authorized);
        assert_eq!(denied.source, Some(PayerSource::Chat));
        assert_eq!(
            denied.denial,
            Some(ReserveDenial::CreditlessHourlyCap { limit: 0 })
        );
        assert_eq!(denied.chat_balance, 500);
        assert!(!billing.payer_by_operation.contains_key(&blocked));

        // A cap that cannot be reached fails the reservation.
        let mut unreachable_cap = PostgresConversationBilling::new(&database_url)
            .with_creditless_cap(RedisCreditlessCap::new(&unreachable_redis())?);
        let failed = unreachable_cap.reserve(request(
            &format!("synthetic-chat-unreachable:{nonce}"),
            "first",
            400,
            5,
        ));
        assert!(
            matches!(&failed, Err(error) if !error.is_empty()),
            "{failed:?}"
        );
        Ok(())
    }

    #[test]
    fn postgres_group_pays_first_until_the_hourly_limit_then_own_credits() -> TestResult {
        std::env::var("TEST_DATABASE_URL")
            .ok()
            .zip(integration_redis_endpoint())
            .map_or(Ok(()), group_first_scenario)
    }

    fn group_first_scenario((database_url, endpoint): (String, RedisEndpoint)) -> TestResult {
        use bot_adapters::redis_creditless_cap::RedisCreditlessCap;

        bot_adapters::billing_schema::BillingSchemaRepository::new(&database_url)
            .ensure_schema()?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let suffix = i64::try_from(nonce % 100_000_000)?;
        let member = 6_340_000_000_000_i64 + suffix;
        let broke = 6_350_000_000_000_i64 + suffix;
        let rich = 6_360_000_000_000_i64 + suffix;
        let funder = 6_370_000_000_000_i64 + suffix;
        let mover = 6_390_000_000_000_i64 + suffix;
        let chat_id = -6_380_000_000_000_i64 - suffix;
        let repository = bot_adapters::billing_read::BillingRepository::new(&database_url);
        repository.mint_user_credits(member, 500, None, &fresh_op())?;
        repository.mint_user_credits(rich, 1_000, None, &fresh_op())?;
        repository.mint_user_credits(funder, 1_000, None, &fresh_op())?;
        assert!(
            repository
                .transfer_user_to_chat(funder, chat_id, 1_000, &fresh_op())?
                .transferred
        );
        let request = |user_id: i64, operation_id: &str, amount: i64, limit: i64| ReserveRequest {
            user_id,
            chat_id: Some(chat_id),
            operation_id: operation_id.to_owned(),
            reservation_id: format!("{operation_id}:first"),
            amount,
            creditless_user_hourly_limit: limit,
            group_pays_first: true,
            metadata: serde_json::Map::new(),
        };
        let mut billing = PostgresConversationBilling::new(&database_url)
            .with_creditless_cap(RedisCreditlessCap::new(&endpoint)?);
        let counter = RedisCreditlessCap::new(&endpoint)?;
        let used = |user_id: i64| {
            counter.count(&bot_adapters::redis_creditless_cap::creditless_cap_key(
                &chat_id.to_string(),
                user_id,
            ))
        };

        // Under the hourly limit the group pays even though the member has credits.
        let first = format!("synthetic-group-first:{nonce}");
        let paid = billing.reserve(request(member, &first, 100, 1))?;
        assert!(paid.authorized);
        assert_eq!(paid.source, Some(PayerSource::Chat));
        assert_eq!(paid.chat_balance, 900);
        assert_eq!(used(member)?, Some(1));
        // The same operation keeps the group as its payer.
        let again = billing.reserve(ReserveRequest {
            reservation_id: format!("{first}:second"),
            ..request(member, &first, 50, 1)
        })?;
        assert_eq!(again.source, Some(PayerSource::Chat));
        assert_eq!(again.chat_balance, 850);

        // At the limit the member's own credits take over.
        let own = billing.reserve(request(member, &format!("synthetic-own:{nonce}"), 100, 1))?;
        assert!(own.authorized);
        assert_eq!(own.source, Some(PayerSource::User));
        assert_eq!(own.chat_balance, 850);
        // Paying with their own credits doesn't use up the group's hour.
        assert_eq!(used(member)?, Some(1));

        // At the limit with too few credits of their own, the limit stops them.
        let blocked = format!("synthetic-blocked:{nonce}");
        let denied = billing.reserve(request(broke, &blocked, 400, 0))?;
        assert!(!denied.authorized);
        assert_eq!(denied.source, None);
        assert_eq!(
            denied.denial,
            Some(ReserveDenial::CreditlessHourlyCap { limit: 0 })
        );
        assert_eq!(denied.chat_balance, 850);
        assert!(!billing.payer_by_operation.contains_key(&blocked));

        // When the group can't cover it, the member's own credits pay.
        let big = billing.reserve(request(rich, &format!("synthetic-big:{nonce}"), 900, 5))?;
        assert!(big.authorized);
        assert_eq!(big.source, Some(PayerSource::User));
        assert_eq!(big.chat_balance, 850);
        assert_eq!(used(rich)?, Some(0));

        // Without an hourly limit the group always pays first.
        let unlimited = billing.reserve(request(
            rich,
            &format!("synthetic-unlimited:{nonce}"),
            10,
            -1,
        ))?;
        assert_eq!(unlimited.source, Some(PayerSource::Chat));
        assert_eq!(unlimited.chat_balance, 840);

        // The group pays the start of a message but can't cover the rest: the
        // whole message moves to the member, and the group gets it all back.
        repository.mint_user_credits(mover, 2_000, None, &fresh_op())?;
        let moved_op = format!("synthetic-moved:{nonce}");
        let base = billing.reserve(request(mover, &moved_op, 800, 5))?;
        assert_eq!(base.source, Some(PayerSource::Chat));
        assert_eq!(base.chat_balance, 40);
        assert_eq!(used(mover)?, Some(1));
        let extension = billing.reserve(ReserveRequest {
            reservation_id: format!("{moved_op}:extension"),
            ..request(mover, &moved_op, 100, 5)
        })?;
        assert!(extension.authorized);
        assert_eq!(extension.source, Some(PayerSource::User));
        assert_eq!(extension.chat_balance, 840);
        assert_eq!(used(mover)?, Some(0));
        let before_settlement = repository.get_balance("user", mover)?;
        billing.settle(SettlementRequest {
            user_id: mover,
            chat_id: Some(chat_id),
            operation_id: moved_op.clone(),
            actual_credit_units: 850,
            delivered: true,
            reason: "synthetic".to_owned(),
            billing_segments: Vec::new(),
        })?;
        // One payer at settlement: the member gets back what wasn't used.
        assert_eq!(
            repository.get_balance("user", mover)?,
            before_settlement + 50
        );
        assert_eq!(repository.get_balance("chat", chat_id)?, 840);

        // When the member can't cover it either, the group keeps its hold.
        let stuck_op = format!("synthetic-stuck:{nonce}");
        let stuck_base = billing.reserve(request(broke, &stuck_op, 100, 5))?;
        assert_eq!(stuck_base.source, Some(PayerSource::Chat));
        let stuck = billing.reserve(ReserveRequest {
            reservation_id: format!("{stuck_op}:extension"),
            ..request(broke, &stuck_op, 100_000, 5)
        })?;
        assert!(!stuck.authorized);
        assert_eq!(stuck.chat_balance, 740);

        // A counter that cannot be read fails the reservation.
        let mut unreachable_cap = PostgresConversationBilling::new(&database_url)
            .with_creditless_cap(RedisCreditlessCap::new(&unreachable_redis())?);
        let failed = unreachable_cap.reserve(request(
            member,
            &format!("synthetic-group-unreachable:{nonce}"),
            10,
            5,
        ));
        assert!(
            matches!(&failed, Err(error) if !error.is_empty()),
            "{failed:?}"
        );
        // Its key is already recorded, so aborting can still give the slot back
        // if Redis counted it before the answer was lost.
        assert!(
            unreachable_cap
                .cap_key_by_operation
                .contains_key(&format!("synthetic-group-unreachable:{nonce}"))
        );
        Ok(())
    }
}
