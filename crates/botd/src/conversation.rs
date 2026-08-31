//! Native foreground AI conversation transaction.

use std::collections::HashMap;

use bot_core::ai_pricing::calculate_billing_for_segments;
use bot_core::ai_prompt::{
    ConversationPromptInput, HistoryMessage, PromptContent, PromptMessage, PromptRole,
    RetrievedMessage, build_conversation_prompt, build_system_prompt,
};
use bot_core::ai_reserve::{
    EstimatedMessage, TokenEstimateValue, chat_output_token_limit,
    estimate_chat_reserve_credit_units,
};
use bot_core::ai_response_cleanup::cleanup_response;
use bot_core::locale::Locale;
use chrono::{DateTime, FixedOffset, Offset, Utc};
use serde_json::{Map, Value, json};

use crate::ai_dispatch::{
    AiConversationInput, AiConversationSource, AiDelivery, AiPreparation, AiReplyMetadata,
};
use crate::chat_tool_loop::{
    ChatRoundStream, ChatToolLoopError, NativeToolRuntime, run_chat_tool_loop,
};
use crate::media::{MediaExecution, MediaKind, MediaRuntime};

const MAX_HISTORY_MESSAGES: usize = 40;
const SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE: i64 = 4_000;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConversationMemory {
    pub summary: Option<String>,
    pub history: Vec<HistoryMessage>,
    pub retrieved: Vec<RetrievedMessage>,
}

pub trait ConversationState {
    fn reply_metadata(
        &mut self,
        chat_id: &str,
        message_id: &str,
    ) -> Result<Option<AiReplyMetadata>, String>;

    fn load_memory(
        &mut self,
        chat_id: &str,
        search_text: &str,
        reply_to_message_id: Option<&str>,
        max_history_messages: usize,
    ) -> Result<ConversationMemory, String>;

    fn record_incoming(&mut self, input: &AiConversationInput) -> Result<(), String>;

    fn record_outgoing(
        &mut self,
        input: &AiConversationInput,
        sent_message_id: Option<i64>,
        text: &str,
    ) -> Result<(), String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReserveRequest {
    pub user_id: i64,
    pub chat_id: Option<i64>,
    pub operation_id: String,
    pub reservation_id: String,
    pub amount: i64,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReserveDecision {
    pub authorized: bool,
    pub user_balance: i64,
    pub chat_balance: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderSegmentRequest {
    pub user_id: i64,
    pub chat_id: Option<i64>,
    pub operation_id: String,
    pub segment_id: String,
    pub segment: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementRequest {
    pub user_id: i64,
    pub chat_id: Option<i64>,
    pub operation_id: String,
    pub actual_credit_units: i64,
    pub delivered: bool,
    pub reason: String,
}

pub trait ConversationBilling {
    fn reserve(&mut self, request: ReserveRequest) -> Result<ReserveDecision, String>;

    fn record_segment(&mut self, request: ProviderSegmentRequest) -> Result<(), String>;

    fn settle(&mut self, request: SettlementRequest) -> Result<(), String>;
}

pub trait ConversationToolFactory {
    type Tools: NativeToolRuntime;

    fn create(&mut self, input: &AiConversationInput) -> Result<Self::Tools, String>;
}

#[derive(Debug, Clone)]
struct PendingConversation {
    input: AiConversationInput,
    text: String,
    segments: Vec<Value>,
    provider_failed: bool,
}

#[derive(Debug, Default)]
struct PreparedConversationMedia {
    execution: Option<MediaExecution>,
    segments: Vec<Value>,
    reserve_decision: Option<ReserveDecision>,
    diagnostics: Vec<String>,
}

pub struct NativeConversation<Provider, Tools, State, Billing> {
    provider: Provider,
    tools: Tools,
    state: State,
    billing: Billing,
    persona: String,
    model: String,
    max_tool_rounds: usize,
    media: Option<Box<dyn MediaRuntime>>,
    pending: HashMap<String, PendingConversation>,
}

impl<Provider, Tools, State, Billing> NativeConversation<Provider, Tools, State, Billing> {
    #[must_use]
    pub fn new(
        provider: Provider,
        tools: Tools,
        state: State,
        billing: Billing,
        persona: &str,
        model: &str,
        max_tool_rounds: usize,
    ) -> Self {
        Self {
            provider,
            tools,
            state,
            billing,
            persona: persona.to_owned(),
            model: model.to_owned(),
            max_tool_rounds,
            media: None,
            pending: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_media(mut self, media: Box<dyn MediaRuntime>) -> Self {
        self.media = Some(media);
        self
    }

    fn preparation_error(locale: Locale) -> &'static str {
        match locale {
            Locale::Es => "me quedé reculando y no te pude responder, probá de nuevo",
            Locale::En => "I could not answer, try again",
        }
    }

    fn insufficient(
        locale: Locale,
        input: &AiConversationInput,
        decision: &ReserveDecision,
    ) -> String {
        if matches!(input.chat_type.as_str(), "group" | "supergroup") {
            match locale {
                Locale::Es => format!(
                    "se quedaron secos de créditos ia en este grupo, boludo.\n- lo tuyo: {}\n- lo del grupo: {}\nmetele /topup por privado y si querés pasá saldo al grupo con /transfer <monto>\nsi querés ver bien la miseria, mandá /balance",
                    format_credit_units(decision.user_balance),
                    format_credit_units(decision.chat_balance),
                ),
                Locale::En => format!(
                    "this group is out of AI credits\n- yours: {}\n- group: {}\nuse /topup in private and /transfer <amount> to fund the group\nuse /balance to see the balances",
                    format_credit_units(decision.user_balance),
                    format_credit_units(decision.chat_balance),
                ),
            }
        } else {
            match locale {
                Locale::Es => format!(
                    "te quedaste seco de créditos ia, boludo.\nsaldo: {}\nmetele /topup si querés que siga laburando",
                    format_credit_units(decision.user_balance),
                ),
                Locale::En => format!(
                    "you are out of AI credits\nbalance: {}\nuse /topup to add more",
                    format_credit_units(decision.user_balance),
                ),
            }
        }
    }
}

impl<Provider, ToolFactory, State, Billing>
    NativeConversation<Provider, ToolFactory, State, Billing>
where
    Provider: ChatRoundStream,
    ToolFactory: ConversationToolFactory,
    State: ConversationState,
    Billing: ConversationBilling,
{
    fn reserve(
        &mut self,
        input: &AiConversationInput,
        operation_id: &str,
        reservation_kind: &str,
        amount: i64,
        estimated_prompt_messages: usize,
    ) -> Result<ReserveDecision, String> {
        let mut metadata = Map::from_iter([
            ("operation_id".to_owned(), json!(operation_id)),
            ("message_id".to_owned(), json!(input.message_id.0)),
            ("origin_chat_id".to_owned(), json!(input.chat_id.0)),
            ("usage_tag".to_owned(), json!(reservation_kind)),
            ("reserved_credit_units".to_owned(), json!(amount)),
            (
                "estimated_prompt_messages".to_owned(),
                json!(estimated_prompt_messages),
            ),
            ("model".to_owned(), json!(self.model)),
        ]);
        metadata.insert("credit_scale".to_owned(), json!(100));
        self.billing.reserve(ReserveRequest {
            user_id: input.sender_id.0,
            chat_id: group_chat_id(input),
            operation_id: operation_id.to_owned(),
            reservation_id: format!("{operation_id}:{reservation_kind}"),
            amount,
            metadata,
        })
    }

    fn settle_immediately(
        &mut self,
        input: &AiConversationInput,
        operation_id: &str,
        reason: &str,
    ) -> Result<(), String> {
        self.billing.settle(SettlementRequest {
            user_id: input.sender_id.0,
            chat_id: group_chat_id(input),
            operation_id: operation_id.to_owned(),
            actual_credit_units: 0,
            delivered: false,
            reason: reason.to_owned(),
        })
    }

    fn prompt(
        &mut self,
        input: &AiConversationInput,
        media_context: Option<&MediaExecution>,
    ) -> Result<(Vec<PromptMessage>, ToolFactory::Tools), String> {
        let tools = self.tools.create(input)?;
        let memory = self.state.load_memory(
            &input.chat_id.0.to_string(),
            &input.message_text,
            input
                .reply_to_message_id
                .map(|id| id.0.to_string())
                .as_deref(),
            MAX_HISTORY_MESSAGES,
        )?;
        let date = formatted_date(input.timestamp, input.timezone_offset_hours);
        let mut messages = vec![PromptMessage::text(
            PromptRole::System,
            build_system_prompt(&self.persona, input.locale, &date, true, false),
        )];
        let mut conversation = build_conversation_prompt(&ConversationPromptInput {
            locale: input.locale,
            chat_type: input.chat_type.clone(),
            chat_title: input.chat_title.clone(),
            first_name: input.sender_first_name.clone(),
            username: input.sender_username.clone(),
            formatted_time: formatted_time(input.timestamp, input.timezone_offset_hours),
            message_text: input.message_text.clone(),
            reply_context: input.reply_context.clone(),
            link_context: None,
            enable_web_search: true,
            summary: memory.summary,
            history: memory.history,
            retrieved: memory.retrieved,
        });
        if let Some(media) = media_context {
            append_media_context(&mut conversation, media, input.locale);
        }
        messages.extend(conversation);
        Ok((messages, tools))
    }

    fn prepare_media(
        &mut self,
        input: &AiConversationInput,
        operation_id: &str,
    ) -> Result<PreparedConversationMedia, String> {
        if self.media.is_none() {
            return Ok(PreparedConversationMedia::default());
        }
        let selected = input
            .audio_file_id
            .as_deref()
            .map(|file_id| (MediaKind::Audio, file_id, input.audio_duration_seconds))
            .or_else(|| {
                input
                    .photo_file_id
                    .as_deref()
                    .map(|file_id| (MediaKind::Image, file_id, None))
            });
        let Some((kind, file_id, duration)) = selected else {
            return Ok(PreparedConversationMedia::default());
        };
        let prepared = match self
            .media
            .as_mut()
            .ok_or_else(|| "native media runtime disappeared".to_owned())?
            .prepare(kind, file_id, duration)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                return Ok(PreparedConversationMedia {
                    diagnostics: vec![format!("AI media preparation: {error}")],
                    ..PreparedConversationMedia::default()
                });
            }
        };
        let reserve_amount = prepared.reserve_credit_units();
        let reservation_kind = match kind {
            MediaKind::Image => "image_context_media",
            MediaKind::Audio => "auto_audio_media",
        };
        let decision = if reserve_amount > 0 {
            Some(self.reserve(input, operation_id, reservation_kind, reserve_amount, 1)?)
        } else {
            None
        };
        if decision
            .as_ref()
            .is_some_and(|decision| !decision.authorized)
        {
            return Ok(PreparedConversationMedia {
                reserve_decision: decision,
                ..PreparedConversationMedia::default()
            });
        }
        let prompt = media_prompt(kind, input.locale);
        match self
            .media
            .as_mut()
            .ok_or_else(|| "native media runtime disappeared".to_owned())?
            .execute(prepared, prompt)
        {
            Ok(execution) => {
                let segments = execution
                    .billing_segment
                    .clone()
                    .into_iter()
                    .collect::<Vec<_>>();
                Ok(PreparedConversationMedia {
                    execution: Some(execution),
                    segments,
                    reserve_decision: decision,
                    diagnostics: Vec::new(),
                })
            }
            Err(error) => Ok(PreparedConversationMedia {
                reserve_decision: decision,
                diagnostics: vec![format!("AI media provider: {error}")],
                ..PreparedConversationMedia::default()
            }),
        }
    }

    fn prepare_transaction(
        &mut self,
        input: AiConversationInput,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<AiPreparation, String> {
        self.state.record_incoming(&input)?;
        let operation_id = operation_id(&input);
        let admission = vec![PromptMessage::text(PromptRole::User, &input.message_text)];
        let base_amount = estimate_reserve(&admission, &self.model)?;
        let base = self.reserve(
            &input,
            &operation_id,
            "ai_response_base",
            base_amount,
            admission.len(),
        )?;
        if !base.authorized {
            return Ok(if input.spontaneous {
                AiPreparation::silent()
            } else {
                AiPreparation::reply(Self::insufficient(input.locale, &input, &base), None)
            });
        }

        let prepared_media = self.prepare_media(&input, &operation_id)?;
        let mut segments = prepared_media.segments;
        let mut diagnostics = prepared_media.diagnostics;
        if let Some(decision) = prepared_media
            .reserve_decision
            .as_ref()
            .filter(|decision| !decision.authorized)
        {
            self.settle_immediately(&input, &operation_id, "ai_response_media_reserve_failed")?;
            return Ok(if input.spontaneous {
                AiPreparation::silent()
            } else {
                AiPreparation::reply(Self::insufficient(input.locale, &input, decision), None)
            });
        }

        let (messages, mut tools) = match self.prompt(&input, prepared_media.execution.as_ref()) {
            Ok(value) => value,
            Err(error) => {
                let actual = price_segments(&segments)?;
                record_and_settle(
                    &mut self.billing,
                    &input,
                    &operation_id,
                    &segments,
                    actual,
                    false,
                    "ai_request_preparation_failed",
                )?;
                return Err(error);
            }
        };
        let full_amount = estimate_reserve(&messages, &self.model)?;
        if full_amount > base_amount {
            let extension = self.reserve(
                &input,
                &operation_id,
                "ai_response_context_extension",
                full_amount - base_amount,
                messages.len(),
            )?;
            if !extension.authorized {
                let actual = price_segments(&segments)?;
                record_and_settle(
                    &mut self.billing,
                    &input,
                    &operation_id,
                    &segments,
                    actual,
                    false,
                    "ai_response_reserve_adjustment_failed",
                )?;
                return Ok(if input.spontaneous {
                    AiPreparation::silent()
                } else {
                    AiPreparation::reply(Self::insufficient(input.locale, &input, &extension), None)
                });
            }
        }

        let loop_result = run_chat_tool_loop(
            &self.provider,
            &mut tools,
            &messages,
            false,
            self.max_tool_rounds,
            |token| {
                on_token(token).map_err(bot_adapters::openrouter_chat::OpenRouterChatError::Stream)
            },
        );
        let (raw_text, chat_segments, chat_diagnostics, provider_failed) = match loop_result {
            Ok(result) => (
                result.text,
                result.billing_segments,
                result.diagnostics,
                result.stopped_at_limit,
            ),
            Err(error) => partial_failure(error),
        };
        segments.extend(chat_segments);
        diagnostics.extend(chat_diagnostics);
        let identity = user_identity(&input);
        let cleaned = cleanup_response(
            &raw_text,
            std::slice::from_ref(&input.reply_context),
            (!identity.is_empty()).then_some(identity.as_str()),
        )
        .final_text;
        let fallback = cleaned.trim().is_empty() || provider_failed;
        let text = if fallback {
            Self::preparation_error(input.locale).to_owned()
        } else {
            cleaned
        };
        if input.spontaneous && fallback {
            let actual = price_segments(&segments)?;
            record_and_settle(
                &mut self.billing,
                &input,
                &operation_id,
                &segments,
                actual,
                false,
                "ai_response_provider_usage_before_fallback",
            )?;
            return Ok(AiPreparation::Silent { diagnostics });
        }
        self.pending.insert(
            operation_id.clone(),
            PendingConversation {
                input,
                text: text.clone(),
                segments,
                provider_failed,
            },
        );
        Ok(AiPreparation::Reply {
            text,
            completion_id: Some(operation_id),
            diagnostics,
        })
    }
}

impl<Provider, ToolFactory, State, Billing> AiConversationSource
    for NativeConversation<Provider, ToolFactory, State, Billing>
where
    Provider: ChatRoundStream,
    ToolFactory: ConversationToolFactory,
    State: ConversationState,
    Billing: ConversationBilling,
{
    fn reply_metadata(
        &mut self,
        chat_id: &str,
        message_id: &str,
    ) -> Result<Option<AiReplyMetadata>, String> {
        self.state.reply_metadata(chat_id, message_id)
    }

    fn prepare(&mut self, input: AiConversationInput) -> Result<AiPreparation, String> {
        self.prepare_transaction(input, &mut |_token| Ok(()))
    }

    fn prepare_streaming(
        &mut self,
        input: AiConversationInput,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<AiPreparation, String> {
        self.prepare_transaction(input, on_token)
    }

    fn record_ignored(&mut self, input: AiConversationInput) -> Result<(), String> {
        self.state.record_incoming(&input)
    }

    fn complete_delivery(&mut self, delivery: AiDelivery) -> Result<(), String> {
        let Some(pending) = self.pending.remove(&delivery.completion_id) else {
            return Ok(());
        };
        let actual = price_segments(&pending.segments)?;
        let reason = if delivery.delivered {
            if pending.provider_failed {
                "ai_response_provider_usage_before_fallback"
            } else {
                "ai_response_success"
            }
        } else if pending.segments.is_empty() {
            "ai_response_delivery_failure_refund"
        } else {
            "ai_response_provider_usage_before_delivery_failure"
        };
        record_and_settle(
            &mut self.billing,
            &pending.input,
            &delivery.completion_id,
            &pending.segments,
            actual,
            delivery.delivered,
            reason,
        )?;
        if delivery.delivered {
            self.state.record_outgoing(
                &pending.input,
                delivery.sent_message_id.map(|id| id.0),
                &pending.text,
            )?;
        }
        Ok(())
    }
}

fn media_prompt(kind: MediaKind, locale: Locale) -> &'static str {
    match (kind, locale) {
        (MediaKind::Image, Locale::Es) => {
            "describí lo que ves en esta imagen en detalle, en minúsculas, sin emojis, sin markdown, en lenguaje coloquial argentino"
        }
        (MediaKind::Image, Locale::En) => {
            "Describe this image in detail in English, without markdown or emojis."
        }
        (MediaKind::Audio, _) => "Transcribe this audio exactly as spoken.",
    }
}

fn append_media_context(messages: &mut [PromptMessage], media: &MediaExecution, locale: Locale) {
    let context = match (media.kind, locale) {
        (MediaKind::Image, Locale::Es) => format!("[Imagen: {}]", media.text),
        (MediaKind::Image, Locale::En) => format!("[Image: {}]", media.text),
        (MediaKind::Audio, Locale::Es) => format!("[Transcripción de audio: {}]", media.text),
        (MediaKind::Audio, Locale::En) => format!("[Audio transcription: {}]", media.text),
    };
    let Some(last) = messages.last_mut() else {
        return;
    };
    match &mut last.content {
        PromptContent::Text(text) => text.push_str(&format!("\n\n{context}")),
        PromptContent::TextParts(parts) => parts.push(context),
        PromptContent::Empty => last.content = PromptContent::Text(context),
    }
}

fn partial_failure(error: ChatToolLoopError) -> (String, Vec<Value>, Vec<String>, bool) {
    let mut diagnostics = error.partial.diagnostics.clone();
    diagnostics.push(format!("AI provider stream: {}", error.source));
    (
        error.partial.text.clone(),
        error.partial.billing_segments.clone(),
        diagnostics,
        true,
    )
}

fn operation_id(input: &AiConversationInput) -> String {
    format!(
        "ai:{}:{}:{}",
        input.chat_id.0, input.message_id.0, input.sender_id.0
    )
}

fn group_chat_id(input: &AiConversationInput) -> Option<i64> {
    matches!(input.chat_type.as_str(), "group" | "supergroup").then_some(input.chat_id.0)
}

fn user_identity(input: &AiConversationInput) -> String {
    if input.sender_username.is_empty() {
        input.sender_first_name.clone()
    } else if input.sender_first_name.is_empty() {
        format!("({})", input.sender_username)
    } else {
        format!("{} ({})", input.sender_first_name, input.sender_username)
    }
}

fn estimate_reserve(messages: &[PromptMessage], model: &str) -> Result<i64, String> {
    let estimated = messages.iter().map(estimated_message).collect::<Vec<_>>();
    estimate_chat_reserve_credit_units(
        None,
        &estimated,
        Some(chat_output_token_limit(model)),
        SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE,
        model,
    )
    .map_err(|error| error.to_string())
}

fn estimated_message(message: &PromptMessage) -> EstimatedMessage {
    EstimatedMessage {
        role: TokenEstimateValue::Text(
            match message.role {
                PromptRole::System => "system",
                PromptRole::User => "user",
                PromptRole::Assistant => "assistant",
                PromptRole::Tool => "tool",
            }
            .to_owned(),
        ),
        content: match &message.content {
            PromptContent::Text(text) => TokenEstimateValue::Text(text.clone()),
            PromptContent::TextParts(parts) => TokenEstimateValue::Sequence(
                parts
                    .iter()
                    .cloned()
                    .map(TokenEstimateValue::Text)
                    .collect(),
            ),
            PromptContent::Empty => TokenEstimateValue::Empty,
        },
        name: TokenEstimateValue::Empty,
    }
}

fn price_segments(segments: &[Value]) -> Result<i64, String> {
    calculate_billing_for_segments(&Value::Array(segments.to_vec()))
        .map_err(|error| error.to_string())?
        .get("charged_credit_units")
        .and_then(Value::as_i64)
        .ok_or_else(|| "AI pricing output omitted charged_credit_units".to_owned())
}

#[allow(clippy::too_many_arguments)]
fn record_and_settle<Billing: ConversationBilling>(
    billing: &mut Billing,
    input: &AiConversationInput,
    operation_id: &str,
    segments: &[Value],
    actual_credit_units: i64,
    delivered: bool,
    reason: &str,
) -> Result<(), String> {
    for (index, segment) in segments.iter().enumerate() {
        billing.record_segment(ProviderSegmentRequest {
            user_id: input.sender_id.0,
            chat_id: group_chat_id(input),
            operation_id: operation_id.to_owned(),
            segment_id: format!("{operation_id}:{index}"),
            segment: segment.clone(),
        })?;
    }
    billing.settle(SettlementRequest {
        user_id: input.sender_id.0,
        chat_id: group_chat_id(input),
        operation_id: operation_id.to_owned(),
        actual_credit_units,
        delivered,
        reason: reason.to_owned(),
    })
}

fn formatted_date(timestamp: i64, timezone_offset_hours: i64) -> String {
    shifted_time(timestamp, timezone_offset_hours)
        .format("%A %d/%m/%Y")
        .to_string()
}

fn formatted_time(timestamp: i64, timezone_offset_hours: i64) -> String {
    shifted_time(timestamp, timezone_offset_hours)
        .format("%H:%M")
        .to_string()
}

fn shifted_time(timestamp: i64, timezone_offset_hours: i64) -> DateTime<FixedOffset> {
    let utc = DateTime::<Utc>::from_timestamp(timestamp, 0).unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
    let seconds = timezone_offset_hours.clamp(-23, 23).saturating_mul(3_600) as i32;
    let offset = FixedOffset::east_opt(seconds).unwrap_or_else(|| Utc.fix());
    utc.with_timezone(&offset)
}

fn format_credit_units(units: i64) -> String {
    let units = units.max(0);
    format!("{}.{:02}", units / 100, units % 100)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::convert::Infallible;

    use bot_adapters::openrouter_chat::OpenRouterChatError;
    use bot_core::provider_stream_policy::StreamToolCall;
    use bot_core::telegram_input::{ChatId, MessageId, UserId};

    use crate::chat_provider::{ChatRoundError, ChatRoundResult};
    use crate::chat_tool_loop::{ChatRoundStream, NativeToolRuntime, ToolExecutionResult};

    use super::*;

    struct Provider {
        rounds: RefCell<VecDeque<Result<ChatRoundResult, ChatRoundError>>>,
        prompts: RefCell<Vec<Vec<PromptMessage>>>,
    }

    impl ChatRoundStream for Provider {
        fn stream_round(
            &self,
            messages: &[PromptMessage],
            _tools: &[Value],
            on_text: &mut dyn FnMut(&str) -> Result<(), OpenRouterChatError>,
        ) -> Result<ChatRoundResult, ChatRoundError> {
            self.prompts.borrow_mut().push(messages.to_vec());
            let Some(round) = self.rounds.borrow_mut().pop_front() else {
                return Err(ChatRoundError {
                    source: OpenRouterChatError::IncompleteStream,
                    partial: Box::new(round("", None)),
                });
            };
            let round = round?;
            on_text(&round.text).map_err(|source| ChatRoundError {
                source,
                partial: Box::new(round.clone()),
            })?;
            Ok(round)
        }
    }

    fn round(text: &str, cost: Option<&str>) -> ChatRoundResult {
        ChatRoundResult {
            text: text.to_owned(),
            tool_calls: Vec::<StreamToolCall>::new(),
            finish_reason: Some("stop".to_owned()),
            billing_segment: Some(json!({
                "kind": "chat",
                "model": "synthetic/model",
                "usage": {"cost": cost.unwrap_or("0.0001")},
                "source": "openrouter",
                "metadata": {"provider": "openrouter", "provider_generation_id": "generation-1"}
            })),
        }
    }

    #[derive(Default)]
    struct NoTools;

    impl NativeToolRuntime for NoTools {
        fn schemas(&self, _task_mode: bool) -> Vec<Value> {
            Vec::new()
        }

        fn contains(&self, _name: &str, _task_mode: bool) -> bool {
            false
        }

        fn execute(
            &mut self,
            _name: &str,
            _arguments: &Value,
            _tool_call_id: &str,
        ) -> ToolExecutionResult {
            ToolExecutionResult::output("")
        }
    }

    struct Tools;

    impl ConversationToolFactory for Tools {
        type Tools = NoTools;

        fn create(&mut self, _input: &AiConversationInput) -> Result<Self::Tools, String> {
            Ok(NoTools)
        }
    }

    struct Media;

    impl MediaRuntime for Media {
        fn prepare(
            &mut self,
            kind: MediaKind,
            file_id: &str,
            duration_hint_seconds: Option<f64>,
        ) -> Result<crate::media::PreparedMedia, String> {
            assert_eq!(kind, MediaKind::Audio);
            assert_eq!(duration_hint_seconds, Some(4.5));
            Ok(crate::media::PreparedMedia::Audio {
                file_id: file_id.to_owned(),
                bytes: b"audio".to_vec(),
                duration_seconds: 4.5,
                reserve_credit_units: 7,
            })
        }

        fn execute(
            &mut self,
            prepared: crate::media::PreparedMedia,
            _prompt: &str,
        ) -> Result<MediaExecution, String> {
            let crate::media::PreparedMedia::Audio { file_id, .. } = prepared else {
                return Err("unexpected media".to_owned());
            };
            Ok(MediaExecution {
                kind: MediaKind::Audio,
                file_id,
                text: "synthetic transcript".to_owned(),
                billing_segment: Some(json!({
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "usage": {},
                    "audio_seconds": 4.5,
                    "source": "groq",
                    "metadata": {"provider": "groq"}
                })),
                cached: false,
            })
        }
    }

    #[derive(Default)]
    struct State {
        memory: ConversationMemory,
        incoming: Vec<AiConversationInput>,
        outgoing: Vec<(AiConversationInput, Option<i64>, String)>,
    }

    impl ConversationState for State {
        fn reply_metadata(
            &mut self,
            _chat_id: &str,
            _message_id: &str,
        ) -> Result<Option<AiReplyMetadata>, String> {
            Ok(None)
        }

        fn load_memory(
            &mut self,
            _chat_id: &str,
            _search_text: &str,
            _reply_to_message_id: Option<&str>,
            _max_history_messages: usize,
        ) -> Result<ConversationMemory, String> {
            Ok(self.memory.clone())
        }

        fn record_incoming(&mut self, input: &AiConversationInput) -> Result<(), String> {
            self.incoming.push(input.clone());
            Ok(())
        }

        fn record_outgoing(
            &mut self,
            input: &AiConversationInput,
            sent_message_id: Option<i64>,
            text: &str,
        ) -> Result<(), String> {
            self.outgoing
                .push((input.clone(), sent_message_id, text.to_owned()));
            Ok(())
        }
    }

    #[derive(Default)]
    struct Billing {
        decisions: VecDeque<ReserveDecision>,
        reserves: Vec<ReserveRequest>,
        segments: Vec<ProviderSegmentRequest>,
        settlements: Vec<SettlementRequest>,
    }

    impl ConversationBilling for Billing {
        fn reserve(&mut self, request: ReserveRequest) -> Result<ReserveDecision, String> {
            self.reserves.push(request);
            Ok(self.decisions.pop_front().unwrap_or(ReserveDecision {
                authorized: true,
                user_balance: 1_000,
                chat_balance: 0,
            }))
        }

        fn record_segment(&mut self, request: ProviderSegmentRequest) -> Result<(), String> {
            self.segments.push(request);
            Ok(())
        }

        fn settle(&mut self, request: SettlementRequest) -> Result<(), String> {
            self.settlements.push(request);
            Ok(())
        }
    }

    fn input() -> AiConversationInput {
        AiConversationInput {
            chat_id: ChatId(42),
            message_id: MessageId(7),
            chat_type: "private".to_owned(),
            chat_title: String::new(),
            sender_id: UserId(88),
            sender_first_name: "Synthetic".to_owned(),
            sender_username: "tester".to_owned(),
            message_text: "what happened?".to_owned(),
            command: "what".to_owned(),
            reply_to_message_id: None,
            reply_context: None,
            photo_file_id: None,
            audio_file_id: None,
            audio_duration_seconds: None,
            locale: Locale::En,
            timezone_offset_hours: -3,
            timestamp: 1_672_531_200,
            spontaneous: false,
        }
    }

    fn conversation(
        rounds: Vec<Result<ChatRoundResult, ChatRoundError>>,
        billing: Billing,
    ) -> NativeConversation<Provider, Tools, State, Billing> {
        NativeConversation::new(
            Provider {
                rounds: RefCell::new(rounds.into()),
                prompts: RefCell::new(Vec::new()),
            },
            Tools,
            State {
                memory: ConversationMemory {
                    summary: Some("prior summary".to_owned()),
                    history: vec![HistoryMessage {
                        role: PromptRole::Assistant,
                        text: "prior answer".to_owned(),
                    }],
                    retrieved: Vec::new(),
                },
                ..State::default()
            },
            billing,
            "synthetic persona",
            "synthetic/model",
            5,
        )
    }

    #[test]
    fn successful_turn_reserves_context_then_settles_only_after_delivery() {
        let mut service = conversation(
            vec![Ok(round("Gordo: **answer**", None))],
            Billing::default(),
        );
        let preparation = service.prepare(input());
        let Ok(AiPreparation::Reply {
            text,
            completion_id: Some(completion_id),
            ..
        }) = preparation
        else {
            return;
        };
        assert_eq!(text, "answer");
        assert!(!service.billing.reserves.is_empty());
        assert!(service.billing.settlements.is_empty());
        assert_eq!(service.state.incoming.len(), 1);
        assert!(service.state.outgoing.is_empty());
        assert_eq!(service.provider.prompts.borrow().len(), 1);

        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id: completion_id.clone(),
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.segments.len(), 1);
        assert_eq!(service.billing.settlements.len(), 1);
        assert!(service.billing.settlements[0].delivered);
        assert_eq!(service.billing.settlements[0].actual_credit_units, 2);
        assert_eq!(service.state.outgoing[0].1, Some(99));
        assert_eq!(service.state.outgoing[0].2, "answer");

        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.settlements.len(), 1);
    }

    #[test]
    fn successful_turn_forwards_provider_text_before_returning_cleaned_output() {
        let mut service = conversation(
            vec![Ok(round("Gordo: **answer**", None))],
            Billing::default(),
        );
        let mut streamed = String::new();
        let preparation = service.prepare_streaming(input(), &mut |token| {
            streamed.push_str(token);
            Ok(())
        });

        assert_eq!(streamed, "Gordo: **answer**");
        assert!(matches!(
            preparation,
            Ok(AiPreparation::Reply { text, .. }) if text == "answer"
        ));
        assert!(service.billing.settlements.is_empty());
    }

    #[test]
    fn audio_is_reserved_transcribed_added_to_context_and_settled_with_chat_usage() {
        let mut service = conversation(vec![Ok(round("answer", None))], Billing::default())
            .with_media(Box::new(Media));
        let mut request = input();
        request.audio_file_id = Some("audio-1".to_owned());
        request.audio_duration_seconds = Some(4.5);
        let preparation = service.prepare(request);
        let Ok(AiPreparation::Reply {
            completion_id: Some(completion_id),
            ..
        }) = preparation
        else {
            return;
        };
        assert!(
            service
                .billing
                .reserves
                .iter()
                .any(|reserve| reserve.metadata["usage_tag"] == "auto_audio_media")
        );
        let prompts = service.provider.prompts.borrow();
        let prompt_text = prompts[0]
            .iter()
            .filter_map(|message| match &message.content {
                PromptContent::Text(text) => Some(text.as_str()),
                PromptContent::TextParts(_) | PromptContent::Empty => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(prompt_text.contains("[Audio transcription: synthetic transcript]"));
        drop(prompts);

        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.segments.len(), 2);
        assert_eq!(service.billing.segments[0].segment["kind"], "transcribe");
        assert_eq!(service.billing.segments[1].segment["kind"], "chat");
    }

    #[test]
    fn denied_and_spontaneous_turns_never_call_the_provider() {
        let denied = ReserveDecision {
            authorized: false,
            user_balance: 25,
            chat_balance: 50,
        };
        let mut explicit = conversation(
            vec![Ok(round("must not run", None))],
            Billing {
                decisions: VecDeque::from([denied.clone()]),
                ..Billing::default()
            },
        );
        let Ok(AiPreparation::Reply {
            text,
            completion_id,
            ..
        }) = explicit.prepare(input())
        else {
            return;
        };
        assert!(text.contains("balance: 0.25"));
        assert_eq!(completion_id, None);
        assert!(explicit.provider.prompts.borrow().is_empty());

        let mut spontaneous_input = input();
        spontaneous_input.spontaneous = true;
        let mut spontaneous = conversation(
            vec![Ok(round("must not run", None))],
            Billing {
                decisions: VecDeque::from([denied]),
                ..Billing::default()
            },
        );
        assert_eq!(
            spontaneous.prepare(spontaneous_input),
            Ok(AiPreparation::silent())
        );
        assert!(spontaneous.provider.prompts.borrow().is_empty());
    }

    #[test]
    fn interrupted_usage_is_billed_on_failed_delivery_but_empty_calls_refund() {
        let partial = ChatRoundResult {
            text: "partial".to_owned(),
            tool_calls: Vec::new(),
            finish_reason: None,
            billing_segment: Some(json!({
                "kind": "chat",
                "model": "synthetic/model",
                "usage": {"cost": "0.00005"},
                "source": "openrouter",
                "metadata": {"provider": "openrouter", "provider_generation_id": "generation-2"}
            })),
        };
        let mut service = conversation(
            vec![Err(ChatRoundError {
                source: OpenRouterChatError::IncompleteStream,
                partial: Box::new(partial),
            })],
            Billing::default(),
        );
        let Ok(AiPreparation::Reply {
            completion_id: Some(completion_id),
            diagnostics,
            ..
        }) = service.prepare(input())
        else {
            return;
        };
        assert!(
            diagnostics
                .iter()
                .any(|value| value.contains("AI provider stream"))
        );
        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: false,
                sent_message_id: None,
            }),
            Ok(())
        );
        assert_eq!(service.billing.settlements[0].actual_credit_units, 1);
        assert_eq!(
            service.billing.settlements[0].reason,
            "ai_response_provider_usage_before_delivery_failure"
        );
        assert!(service.state.outgoing.is_empty());
    }

    #[test]
    fn type_signatures_remain_infallible_for_no_tool_execution() {
        let _: Result<(), Infallible> = Ok(());
    }
}
