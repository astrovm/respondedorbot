//! Native foreground AI conversation transaction.

use std::collections::HashMap;

use bot_core::ai_pricing::calculate_billing_for_segments;
use bot_core::ai_prompt::{
    ConversationPromptInput, HistoryMessage, PromptContent, PromptMessage, PromptRole,
    RetrievedMessage, build_conversation_prompt, build_system_prompt,
};
use bot_core::ai_reserve::{
    EstimatedMessage, TokenEstimateValue, VISION_OUTPUT_TOKEN_LIMIT, chat_output_token_limit,
    estimate_chat_reserve_credit_units, estimate_transcription_reserve_credit_units,
    estimate_vision_reserve_credit_units,
};
use bot_core::ai_response_cleanup::cleanup_response;
use bot_core::ai_usage::stable_provider_segment_id;
use bot_core::locale::{Locale, format_date};
use bot_core::text_cleanup::sanitize_summary_text;
use chrono::{DateTime, FixedOffset, Offset, Utc};
use serde_json::{Map, Value, json};

use crate::ai_dispatch::{
    AiConversationInput, AiConversationSource, AiDelivery, AiPreparation, AiReplyMetadata,
};
use crate::chat_tool_loop::{
    ChatRoundStream, ChatToolLoopError, NativeToolRuntime, run_chat_tool_loop,
};
use crate::compaction_scheduler::{
    CompactionScheduleContext, MemoryCompactionPlan, MemoryCompactionScheduler, PayerSource,
};
use crate::media::{MediaExecution, MediaKind, MediaPipelineError, MediaRuntime};

const MAX_HISTORY_MESSAGES: usize = 40;
const MAX_SUMMARY_MESSAGES: usize = 200;
const SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE: i64 = 4_000;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConversationMemory {
    pub summary: Option<String>,
    pub history: Vec<HistoryMessage>,
    pub retrieved: Vec<RetrievedMessage>,
    pub compaction_plan: Option<MemoryCompactionPlan>,
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

    fn load_summary_memory(
        &mut self,
        chat_id: &str,
        max_history_messages: usize,
    ) -> Result<ConversationMemory, String> {
        self.load_memory(chat_id, "", None, max_history_messages)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReserveRequest {
    pub user_id: i64,
    pub chat_id: Option<i64>,
    pub operation_id: String,
    pub reservation_id: String,
    pub amount: i64,
    pub creditless_user_hourly_limit: i64,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveDenial {
    CreditlessHourlyCap { limit: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReserveDecision {
    pub authorized: bool,
    pub user_balance: i64,
    pub chat_balance: i64,
    pub source: Option<PayerSource>,
    pub denial: Option<ReserveDenial>,
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
    pub billing_segments: Vec<Value>,
}

pub trait ConversationBilling {
    fn reserve(&mut self, request: ReserveRequest) -> Result<ReserveDecision, String>;

    fn record_segment(&mut self, request: ProviderSegmentRequest) -> Result<(), String>;

    fn settle(&mut self, request: SettlementRequest) -> Result<(), String>;

    fn abort_operation(&mut self, operation_id: &str) -> Result<(), String>;

    fn release_operation(&mut self, operation_id: &str);

    fn personal_balance(&mut self, _user_id: i64) -> Result<Option<i64>, String> {
        Ok(None)
    }
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
    kind: PendingKind,
    compaction_plan: Option<MemoryCompactionPlan>,
    compaction_payer: Option<PayerSource>,
}

#[derive(Debug, Clone, Copy)]
enum PendingKind {
    Conversation { provider_failed: bool },
    MediaCommand,
    SummaryCommand { provider_failed: bool },
}

#[derive(Debug, Default)]
struct PreparedConversationMedia {
    execution: Option<MediaExecution>,
    segments: Vec<Value>,
    reserve_decision: Option<ReserveDecision>,
    diagnostics: Vec<String>,
}

struct PreparedPrompt<Tools> {
    messages: Vec<PromptMessage>,
    tools: Tools,
    compaction_plan: Option<MemoryCompactionPlan>,
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
    compaction_scheduler: Option<Box<dyn MemoryCompactionScheduler>>,
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
            compaction_scheduler: None,
            pending: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_media(mut self, media: Box<dyn MediaRuntime>) -> Self {
        self.media = Some(media);
        self
    }

    #[must_use]
    pub fn with_compaction_scheduler(
        mut self,
        scheduler: Box<dyn MemoryCompactionScheduler>,
    ) -> Self {
        self.compaction_scheduler = Some(scheduler);
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
        if let Some(ReserveDenial::CreditlessHourlyCap { limit }) = decision.denial {
            return match locale {
                Locale::Es => format!(
                    "llegaste al limite de {limit} mensajes de ia pagados por el grupo por hora, boludo. cargá créditos con /topup si querés seguir"
                ),
                Locale::En => format!(
                    "you reached the limit of {limit} group-funded AI messages per hour. use /topup to continue"
                ),
            };
        }
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
            creditless_user_hourly_limit: input.creditless_user_hourly_limit,
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
            billing_segments: Vec::new(),
        })
    }

    fn finish_preparation<T>(
        &mut self,
        operation_id: &str,
        result: Result<T, String>,
    ) -> Result<T, String> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                self.pending.remove(operation_id);
                self.billing
                    .abort_operation(operation_id)
                    .map_err(|abort_error| {
                        format!("{error}; operation abort failed: {abort_error}")
                    })?;
                Err(error)
            }
        }
    }

    fn prompt(
        &mut self,
        input: &AiConversationInput,
        media_context: Option<&MediaExecution>,
    ) -> Result<PreparedPrompt<ToolFactory::Tools>, String> {
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
        let date = formatted_date(input.timestamp, input.timezone_offset_hours, input.locale);
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
        Ok(PreparedPrompt {
            messages,
            tools,
            compaction_plan: memory.compaction_plan,
        })
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

    fn prepare_media_command_transaction(
        &mut self,
        input: AiConversationInput,
    ) -> Result<AiPreparation, String> {
        let operation_id = operation_id(&input);
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
        let admission_kind = selected.map_or(MediaKind::Image, |(kind, _, _)| kind);
        let admission_amount = match admission_kind {
            MediaKind::Image => estimate_vision_reserve_credit_units(
                "Describe what you see in this image in detail.",
                0,
                1_200,
                VISION_OUTPUT_TOKEN_LIMIT,
                crate::native_ai::VISION_MODEL,
            ),
            MediaKind::Audio => estimate_transcription_reserve_credit_units(
                input.audio_duration_seconds.unwrap_or(1.0),
            ),
        }
        .map_err(|error| error.to_string())?;
        let admission = self.reserve(
            &input,
            &operation_id,
            "transcribe_command_media",
            admission_amount,
            1,
        )?;
        if !admission.authorized {
            return Ok(AiPreparation::reply(
                Self::insufficient(input.locale, &input, &admission),
                None,
            ));
        }

        let (text, segments, diagnostics) = if let Some((kind, file_id, duration)) = selected {
            let prepared = match self
                .media
                .as_mut()
                .ok_or_else(|| "native media runtime disappeared".to_owned())?
                .prepare(kind, file_id, duration)
            {
                Ok(prepared) => prepared,
                Err(error) => {
                    let text = media_command_prepare_error(
                        kind,
                        input.visual_media_kind.as_deref(),
                        input.locale,
                        &error,
                    );
                    self.pending.insert(
                        operation_id.clone(),
                        PendingConversation {
                            input,
                            text: text.clone(),
                            segments: Vec::new(),
                            kind: PendingKind::MediaCommand,
                            compaction_plan: None,
                            compaction_payer: None,
                        },
                    );
                    return Ok(AiPreparation::Reply {
                        text,
                        completion_id: Some(operation_id),
                        diagnostics: vec![format!("media command preparation: {error}")],
                    });
                }
            };
            let required_amount = prepared.reserve_credit_units();
            if required_amount > admission_amount {
                let extension = self.reserve(
                    &input,
                    &operation_id,
                    "transcribe_command_media_extension",
                    required_amount - admission_amount,
                    1,
                )?;
                if !extension.authorized {
                    self.settle_immediately(
                        &input,
                        &operation_id,
                        "transcribe_command_reserve_adjustment_failed",
                    )?;
                    return Ok(AiPreparation::reply(
                        Self::insufficient(input.locale, &input, &extension),
                        None,
                    ));
                }
            }
            let prompt = if input.visual_media_kind.as_deref() == Some("sticker") {
                sticker_prompt(input.locale)
            } else {
                media_prompt(kind, input.locale)
            };
            match self
                .media
                .as_mut()
                .ok_or_else(|| "native media runtime disappeared".to_owned())?
                .execute(prepared, prompt)
            {
                Ok(execution) => {
                    let text = media_command_success(
                        execution.kind,
                        input.visual_media_kind.as_deref(),
                        input.locale,
                        &execution.text,
                    );
                    (
                        text,
                        execution.billing_segment.into_iter().collect(),
                        Vec::new(),
                    )
                }
                Err(error) => (
                    media_command_provider_error(
                        kind,
                        input.visual_media_kind.as_deref(),
                        input.locale,
                    )
                    .to_owned(),
                    Vec::new(),
                    vec![format!("media command provider: {error}")],
                ),
            }
        } else {
            (
                if input.has_reply {
                    media_command_none(input.locale)
                } else {
                    media_command_reply_required(input.locale)
                }
                .to_owned(),
                Vec::new(),
                Vec::new(),
            )
        };
        self.pending.insert(
            operation_id.clone(),
            PendingConversation {
                input,
                text: text.clone(),
                segments,
                kind: PendingKind::MediaCommand,
                compaction_plan: None,
                compaction_payer: None,
            },
        );
        Ok(AiPreparation::Reply {
            text,
            completion_id: Some(operation_id),
            diagnostics,
        })
    }

    fn prepare_summary_command_transaction(
        &mut self,
        input: AiConversationInput,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<AiPreparation, String> {
        let operation_id = summary_operation_id(&input);
        let admission = vec![PromptMessage::text(PromptRole::User, "summary")];
        let base_amount = estimate_reserve(&admission, &self.model)?;
        let base = self.reserve(
            &input,
            &operation_id,
            "summary_command_base",
            base_amount,
            admission.len(),
        )?;
        if !base.authorized {
            return Ok(AiPreparation::reply(
                Self::insufficient(input.locale, &input, &base),
                None,
            ));
        }
        let memory = match self
            .state
            .load_summary_memory(&input.chat_id.0.to_string(), MAX_SUMMARY_MESSAGES)
        {
            Ok(memory) => memory,
            Err(error) => {
                self.settle_immediately(&input, &operation_id, "summary_preparation_failed")?;
                return Err(error);
            }
        };
        let ConversationMemory {
            summary,
            history,
            retrieved: _,
            compaction_plan: _,
        } = memory;
        if history.is_empty() {
            let text = summary.filter(|summary| !summary.is_empty()).map_or_else(
                || summary_empty(input.locale).to_owned(),
                |summary| sanitize_summary_text(&summary),
            );
            let segments = vec![internal_summary_cache_segment(&text)];
            self.pending.insert(
                operation_id.clone(),
                PendingConversation {
                    input,
                    text: text.clone(),
                    segments,
                    kind: PendingKind::SummaryCommand {
                        provider_failed: false,
                    },
                    compaction_plan: None,
                    compaction_payer: None,
                },
            );
            on_token(&text)?;
            return Ok(AiPreparation::reply(text, Some(operation_id)));
        }

        let prompt = summary_prompt(&input.message_text, input.locale);
        let mut messages = vec![PromptMessage::text(PromptRole::System, &self.persona)];
        if let Some(summary) = summary.filter(|summary| !summary.is_empty()) {
            messages.push(PromptMessage::text(PromptRole::Assistant, summary));
        }
        messages.extend(
            history
                .into_iter()
                .map(|message| PromptMessage::text(message.role, message.text)),
        );
        messages.push(PromptMessage::text(PromptRole::User, prompt));
        let full_amount = estimate_reserve(&messages, &self.model)?;
        if full_amount > base_amount {
            let extension = self.reserve(
                &input,
                &operation_id,
                "summary_command_context_extension",
                full_amount - base_amount,
                messages.len(),
            )?;
            if !extension.authorized {
                self.settle_immediately(
                    &input,
                    &operation_id,
                    "summary_reserve_adjustment_failed",
                )?;
                return Ok(AiPreparation::reply(
                    Self::insufficient(input.locale, &input, &extension),
                    None,
                ));
            }
        }

        let (raw_text, mut segments, diagnostics, provider_failed) =
            match self.provider.stream_round(&messages, &[], &mut |token| {
                on_token(token).map_err(bot_adapters::openrouter_chat::OpenRouterChatError::Stream)
            }) {
                Ok(result) => (
                    result.text,
                    result.billing_segment.into_iter().collect::<Vec<_>>(),
                    Vec::new(),
                    false,
                ),
                Err(error) => {
                    let partial = *error.partial;
                    (
                        partial.text,
                        partial.billing_segment.into_iter().collect::<Vec<_>>(),
                        vec![format!("summary provider stream: {}", error.source)],
                        true,
                    )
                }
            };
        for segment in &mut segments {
            if let Some(segment) = segment.as_object_mut() {
                segment.insert("kind".to_owned(), json!("summary"));
            }
        }
        let cleaned = sanitize_summary_text(&raw_text);
        let text = if provider_failed || cleaned.is_empty() {
            summary_error(input.locale).to_owned()
        } else {
            cleaned
        };
        self.pending.insert(
            operation_id.clone(),
            PendingConversation {
                input,
                text: text.clone(),
                segments,
                kind: PendingKind::SummaryCommand { provider_failed },
                compaction_plan: None,
                compaction_payer: None,
            },
        );
        Ok(AiPreparation::Reply {
            text,
            completion_id: Some(operation_id),
            diagnostics,
        })
    }

    fn prepare_transaction(
        &mut self,
        input: AiConversationInput,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<AiPreparation, String> {
        self.state.record_incoming(&input)?;
        let task_command = matches!(
            input.command.as_str(),
            "/tarea" | "/tareas" | "/task" | "/tasks"
        ) && !input.message_text.trim().is_empty();
        let mut provider_input = input.clone();
        if task_command {
            let required = match crate::native_ai::estimate_task_reserve_credit_units(
                &input.message_text,
                match input.locale {
                    Locale::Es => "es",
                    Locale::En => "en",
                },
            ) {
                Ok(required) => required.max(1),
                Err(_) => {
                    return Ok(AiPreparation::reply(
                        crate::task_tools::task_cost_error(input.locale),
                        None,
                    ));
                }
            };
            match self.billing.personal_balance(input.sender_id.0) {
                Ok(Some(balance)) if balance < required => {
                    return Ok(AiPreparation::reply(
                        crate::task_tools::task_credit_insufficient(
                            balance,
                            required,
                            input.locale,
                        ),
                        None,
                    ));
                }
                Ok(Some(_)) => {}
                Ok(None) => {
                    return Ok(AiPreparation::reply(
                        match input.locale {
                            Locale::Es => "el cobro de ia no está andando, avisale al admin",
                            Locale::En => "AI billing is unavailable, please tell the admin",
                        },
                        None,
                    ));
                }
                Err(_) => {
                    return Ok(AiPreparation::reply(
                        crate::task_tools::task_credit_check(input.locale),
                        None,
                    ));
                }
            }
            provider_input.message_text = match input.locale {
                Locale::Es => format!(
                    "creá una tarea programada para esta solicitud usando la herramienta task_set: {}",
                    input.message_text
                ),
                Locale::En => format!(
                    "create a scheduled task for this request using the task_set tool: {}",
                    input.message_text
                ),
            };
        }
        let operation_id = operation_id(&input);
        let admission = vec![PromptMessage::text(
            PromptRole::User,
            &provider_input.message_text,
        )];
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

        let PreparedPrompt {
            messages,
            mut tools,
            compaction_plan,
        } = match self.prompt(&provider_input, prepared_media.execution.as_ref()) {
            Ok(value) => value,
            Err(error) => {
                record_and_settle(
                    &mut self.billing,
                    &input,
                    &operation_id,
                    &segments,
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
                record_and_settle(
                    &mut self.billing,
                    &input,
                    &operation_id,
                    &segments,
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
            record_and_settle(
                &mut self.billing,
                &input,
                &operation_id,
                &segments,
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
                kind: PendingKind::Conversation { provider_failed },
                compaction_plan,
                compaction_payer: base.source,
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
        let operation_id = operation_id(&input);
        let result = self.prepare_transaction(input, &mut |_token| Ok(()));
        self.finish_preparation(&operation_id, result)
    }

    fn prepare_streaming(
        &mut self,
        input: AiConversationInput,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<AiPreparation, String> {
        let operation_id = operation_id(&input);
        let result = self.prepare_transaction(input, on_token);
        self.finish_preparation(&operation_id, result)
    }

    fn prepare_media_command(
        &mut self,
        input: AiConversationInput,
    ) -> Result<Option<AiPreparation>, String> {
        if self.media.is_none() {
            Ok(None)
        } else {
            let operation_id = operation_id(&input);
            let result = self.prepare_media_command_transaction(input).map(Some);
            self.finish_preparation(&operation_id, result)
        }
    }

    fn prepare_summary_command_streaming(
        &mut self,
        input: AiConversationInput,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<Option<AiPreparation>, String> {
        let operation_id = summary_operation_id(&input);
        let result = self
            .prepare_summary_command_transaction(input, on_token)
            .map(Some);
        self.finish_preparation(&operation_id, result)
    }

    fn record_ignored(&mut self, input: AiConversationInput) -> Result<(), String> {
        self.state.record_incoming(&input)
    }

    fn complete_delivery(&mut self, delivery: AiDelivery) -> Result<(), String> {
        let Some(pending) = self.pending.remove(&delivery.completion_id) else {
            return Ok(());
        };
        let reason = match (
            pending.kind,
            delivery.delivered,
            pending.segments.is_empty(),
        ) {
            (
                PendingKind::Conversation {
                    provider_failed: true,
                },
                true,
                _,
            ) => "ai_response_provider_usage_before_fallback",
            (
                PendingKind::Conversation {
                    provider_failed: false,
                },
                true,
                _,
            ) => "ai_response_success",
            (PendingKind::Conversation { .. }, false, true) => {
                "ai_response_delivery_failure_refund"
            }
            (PendingKind::Conversation { .. }, false, false) => {
                "ai_response_provider_usage_before_delivery_failure"
            }
            (PendingKind::MediaCommand, true, _) => "transcribe_command_success",
            (PendingKind::MediaCommand, false, true) => {
                "transcribe_command_delivery_failure_refund"
            }
            (PendingKind::MediaCommand, false, false) => {
                "transcribe_command_provider_usage_before_delivery_failure"
            }
            (
                PendingKind::SummaryCommand {
                    provider_failed: true,
                },
                true,
                false,
            ) => "summary_stream_provider_usage_before_fallback",
            (
                PendingKind::SummaryCommand {
                    provider_failed: true,
                },
                true,
                true,
            ) => "summary_stream_fallback",
            (PendingKind::SummaryCommand { .. }, true, _) => "summary_command_stream_success",
            (PendingKind::SummaryCommand { .. }, false, true) => "summary_stream_failed",
            (PendingKind::SummaryCommand { .. }, false, false) => {
                "summary_stream_provider_usage_before_delivery_failure"
            }
        };
        record_and_settle(
            &mut self.billing,
            &pending.input,
            &delivery.completion_id,
            &pending.segments,
            delivery.delivered,
            reason,
        )?;
        if delivery.delivered && matches!(pending.kind, PendingKind::Conversation { .. }) {
            self.state.record_outgoing(
                &pending.input,
                delivery.sent_message_id.map(|id| id.0),
                &pending.text,
            )?;
            if let (Some(plan), Some(scheduler)) =
                (pending.compaction_plan, self.compaction_scheduler.as_mut())
            {
                let _scheduled = scheduler.schedule(
                    plan,
                    CompactionScheduleContext {
                        user_id: pending.input.sender_id.0,
                        group_chat_id: group_chat_id(&pending.input),
                        origin_chat_id: pending.input.chat_id.0,
                        message_id: pending.input.message_id.0,
                        locale: pending.input.locale,
                        payer_source: pending.compaction_payer,
                    },
                );
            }
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

fn sticker_prompt(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => {
            "describí lo que ves en este sticker en detalle, en minúsculas, sin emojis, sin markdown, en lenguaje coloquial argentino"
        }
        Locale::En => "Describe this sticker in detail in English, without markdown or emojis.",
    }
}

fn media_command_reply_required(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "respondeme un audio, video, imagen o sticker y te digo qué carajo hay ahí",
        Locale::En => "reply to an audio, video, image, or sticker and I will process it",
    }
}

fn media_command_none(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "ese mensaje no tiene audio, video, imagen ni sticker para laburar",
        Locale::En => "that message has no audio, video, image, or sticker to process",
    }
}

fn media_command_prepare_error(
    kind: MediaKind,
    visual_kind: Option<&str>,
    locale: Locale,
    error: &str,
) -> String {
    if error == MediaPipelineError::Download.to_string() {
        return match (kind, visual_kind, locale) {
            (MediaKind::Audio, _, Locale::Es) => {
                "no pude bajar el audio, mandalo de nuevo".to_owned()
            }
            (MediaKind::Audio, _, Locale::En) => {
                "I could not download the audio, send it again".to_owned()
            }
            (MediaKind::Image, Some("sticker"), Locale::Es) => {
                "no pude bajar el sticker, mandalo de nuevo".to_owned()
            }
            (MediaKind::Image, Some("sticker"), Locale::En) => {
                "I could not download the sticker, send it again".to_owned()
            }
            (MediaKind::Image, _, Locale::Es) => {
                "no pude bajar la imagen, mandala de nuevo".to_owned()
            }
            (MediaKind::Image, _, Locale::En) => {
                "I could not download the image, send it again".to_owned()
            }
        };
    }
    if kind == MediaKind::Audio && error == MediaPipelineError::InvalidAudio.to_string() {
        return match locale {
            Locale::Es => "no pude medir la duración del audio".to_owned(),
            Locale::En => "I could not measure the audio duration".to_owned(),
        };
    }
    media_command_provider_error(kind, visual_kind, locale).to_owned()
}

fn media_command_provider_error(
    kind: MediaKind,
    visual_kind: Option<&str>,
    locale: Locale,
) -> &'static str {
    match (kind, visual_kind, locale) {
        (MediaKind::Audio, _, Locale::Es) => "no pude sacar nada de ese audio, probá más tarde",
        (MediaKind::Audio, _, Locale::En) => "I could not transcribe that audio, try again later",
        (MediaKind::Image, Some("sticker"), Locale::Es) => {
            "no pude sacar qué carajo tiene el sticker, probá más tarde"
        }
        (MediaKind::Image, Some("sticker"), Locale::En) => {
            "I could not describe the sticker, try again later"
        }
        (MediaKind::Image, _, Locale::Es) => {
            "no pude sacar qué mierda tiene la imagen, probá más tarde"
        }
        (MediaKind::Image, _, Locale::En) => "I could not describe the image, try again later",
    }
}

fn media_command_success(
    kind: MediaKind,
    visual_kind: Option<&str>,
    locale: Locale,
    text: &str,
) -> String {
    let text = sanitize_summary_text(text);
    match (kind, visual_kind, locale) {
        (MediaKind::Audio, _, Locale::Es) => format!("🎵 te saqué esto del audio: {text}"),
        (MediaKind::Audio, _, Locale::En) => format!("🎵 audio transcription: {text}"),
        (MediaKind::Image, Some("sticker"), Locale::Es) => {
            format!("🎨 en el sticker veo: {text}")
        }
        (MediaKind::Image, Some("sticker"), Locale::En) => format!("🎨 sticker: {text}"),
        (MediaKind::Image, _, Locale::Es) => format!("🖼️ en la imagen veo: {text}"),
        (MediaKind::Image, _, Locale::En) => format!("🖼️ image: {text}"),
    }
}

fn summary_prompt(custom: &str, locale: Locale) -> String {
    let custom = custom.trim();
    let custom = custom
        .split_once(char::is_whitespace)
        .filter(|(first, _)| first.chars().all(|character| character.is_ascii_digit()))
        .map_or_else(
            || {
                (!custom.is_empty() && !custom.chars().all(|character| character.is_ascii_digit()))
                    .then_some(custom)
            },
            |(_, remaining)| Some(remaining.trim()),
        )
        .filter(|value| !value.is_empty());
    let base = match locale {
        Locale::Es => {
            "actualizá el resumen anterior con los mensajes nuevos. entre 10 y 20 items cortos y concretos si hay material suficiente, uno por línea. incluí solo hechos relevantes: tema, decisiones, pendientes y datos clave. evitá relleno, repetición, contexto innecesario y frases largas. NUNCA uses markdown: no negritas, no headers, no tablas. usá solo guiones (-) al inicio de cada item."
        }
        Locale::En => {
            "update the previous summary with the new messages. use 10 to 20 short, concrete items when there is enough material, one per line. include only relevant facts: topic, decisions, pending work, and key data. avoid filler, repetition, unnecessary context, and long sentences. NEVER use markdown: no bold text, headings, or tables. use only hyphens (-) at the start of each item."
        }
    };
    custom.map_or_else(|| base.to_owned(), |custom| format!("{custom}. {base}"))
}

fn summary_empty(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "no hay mensajes para resumir",
        Locale::En => "there are no messages to summarize",
    }
}

fn summary_error(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "no pude generar el resumen",
        Locale::En => "I could not generate the summary",
    }
}

fn internal_summary_cache_segment(text: &str) -> Value {
    json!({
        "kind": "summary",
        "text": text,
        "model": crate::native_ai::PRIMARY_CHAT_MODEL,
        "source": "cache",
        "cached": true,
        "metadata": {"pricing_basis": "internal_cache"}
    })
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

fn summary_operation_id(input: &AiConversationInput) -> String {
    format!(
        "summary:{}:{}:{}",
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
    delivered: bool,
    reason: &str,
) -> Result<(), String> {
    let actual_credit_units = match price_segments(segments) {
        Ok(actual_credit_units) => actual_credit_units,
        Err(error) => {
            billing.release_operation(operation_id);
            return Err(error);
        }
    };
    let mut segment_error = None;
    for segment in segments {
        if let Err(error) = billing.record_segment(ProviderSegmentRequest {
            user_id: input.sender_id.0,
            chat_id: group_chat_id(input),
            operation_id: operation_id.to_owned(),
            segment_id: stable_provider_segment_id(segment),
            segment: segment.clone(),
        }) {
            segment_error.get_or_insert(error);
        }
    }
    let settlement = billing.settle(SettlementRequest {
        user_id: input.sender_id.0,
        chat_id: group_chat_id(input),
        operation_id: operation_id.to_owned(),
        actual_credit_units,
        delivered,
        reason: reason.to_owned(),
        billing_segments: segments.to_vec(),
    });
    match (segment_error, settlement) {
        (Some(segment_error), Err(settlement_error)) => Err(format!(
            "provider usage recording failed: {segment_error}; settlement failed: {settlement_error}"
        )),
        (Some(error), Ok(())) => Err(format!("provider usage recording failed: {error}")),
        (None, result) => result,
    }
}

fn formatted_date(timestamp: i64, timezone_offset_hours: i64, locale: Locale) -> String {
    format_date(
        shifted_time(timestamp, timezone_offset_hours).date_naive(),
        locale,
    )
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
    use std::rc::Rc;

    use bot_adapters::openrouter_chat::OpenRouterChatError;
    use bot_core::provider_pricing::DEEPSEEK_MODEL;
    use bot_core::provider_stream_policy::StreamToolCall;
    use bot_core::telegram_input::{ChatId, MessageId, UserId};

    use crate::chat_provider::{ChatRoundError, ChatRoundResult};
    use crate::chat_tool_loop::{ChatRoundStream, NativeToolRuntime, ToolExecutionResult};

    use super::*;

    struct Scheduler(Rc<RefCell<Vec<(MemoryCompactionPlan, CompactionScheduleContext)>>>);

    impl MemoryCompactionScheduler for Scheduler {
        fn schedule(
            &mut self,
            plan: MemoryCompactionPlan,
            context: CompactionScheduleContext,
        ) -> Result<bool, String> {
            self.0.borrow_mut().push((plan, context));
            Ok(true)
        }
    }

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

    struct StickerMedia;

    impl MediaRuntime for StickerMedia {
        fn prepare(
            &mut self,
            kind: MediaKind,
            file_id: &str,
            duration_hint_seconds: Option<f64>,
        ) -> Result<crate::media::PreparedMedia, String> {
            assert_eq!(kind, MediaKind::Image);
            assert_eq!(duration_hint_seconds, None);
            Ok(crate::media::PreparedMedia::Cached {
                kind,
                file_id: file_id.to_owned(),
                text: "ignored by fake execution".to_owned(),
            })
        }

        fn execute(
            &mut self,
            prepared: crate::media::PreparedMedia,
            prompt: &str,
        ) -> Result<MediaExecution, String> {
            assert!(prompt.starts_with("Describe this sticker"));
            Ok(MediaExecution {
                kind: prepared.kind(),
                file_id: "sticker-1".to_owned(),
                text: "**synthetic** [sticker](https://example.test)".to_owned(),
                billing_segment: None,
                cached: true,
            })
        }
    }

    struct DownloadFailureMedia;

    impl MediaRuntime for DownloadFailureMedia {
        fn prepare(
            &mut self,
            _kind: MediaKind,
            _file_id: &str,
            _duration_hint_seconds: Option<f64>,
        ) -> Result<crate::media::PreparedMedia, String> {
            Err(MediaPipelineError::Download.to_string())
        }

        fn execute(
            &mut self,
            _prepared: crate::media::PreparedMedia,
            _prompt: &str,
        ) -> Result<MediaExecution, String> {
            Err("must not execute".to_owned())
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
        released_operations: Vec<String>,
        personal_balance: Option<i64>,
        record_failure: bool,
        settlement_failure: bool,
    }

    impl ConversationBilling for Billing {
        fn reserve(&mut self, request: ReserveRequest) -> Result<ReserveDecision, String> {
            self.reserves.push(request);
            Ok(self.decisions.pop_front().unwrap_or(ReserveDecision {
                authorized: true,
                user_balance: 1_000,
                chat_balance: 0,
                source: Some(PayerSource::User),
                denial: None,
            }))
        }

        fn record_segment(&mut self, request: ProviderSegmentRequest) -> Result<(), String> {
            self.segments.push(request);
            if self.record_failure {
                Err("synthetic segment failure".to_owned())
            } else {
                Ok(())
            }
        }

        fn settle(&mut self, request: SettlementRequest) -> Result<(), String> {
            self.settlements.push(request);
            if self.settlement_failure {
                Err("synthetic settlement failure".to_owned())
            } else {
                Ok(())
            }
        }

        fn abort_operation(&mut self, operation_id: &str) -> Result<(), String> {
            self.released_operations.push(operation_id.to_owned());
            Ok(())
        }

        fn release_operation(&mut self, operation_id: &str) {
            self.released_operations.push(operation_id.to_owned());
        }

        fn personal_balance(&mut self, _user_id: i64) -> Result<Option<i64>, String> {
            Ok(self.personal_balance)
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
            has_reply: false,
            visual_media_kind: None,
            audio_media_kind: None,
            photo_file_id: None,
            audio_file_id: None,
            audio_duration_seconds: None,
            locale: Locale::En,
            timezone_offset_hours: -3,
            creditless_user_hourly_limit: 5,
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
                    compaction_plan: None,
                },
                ..State::default()
            },
            billing,
            "synthetic persona",
            DEEPSEEK_MODEL,
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
    fn delivery_finalization_releases_pending_state_after_attempting_failed_segments() {
        let mut service = conversation(
            vec![Ok(round("answer", None))],
            Billing {
                record_failure: true,
                ..Billing::default()
            },
        );
        let Ok(AiPreparation::Reply {
            completion_id: Some(completion_id),
            ..
        }) = service.prepare(input())
        else {
            return;
        };

        assert!(
            service
                .complete_delivery(AiDelivery {
                    completion_id: completion_id.clone(),
                    delivered: true,
                    sent_message_id: Some(MessageId(99)),
                })
                .is_err()
        );
        assert_eq!(service.billing.settlements.len(), 1);
        assert!(!service.pending.contains_key(&completion_id));

        service.billing.record_failure = false;
        assert!(
            service
                .complete_delivery(AiDelivery {
                    completion_id: completion_id.clone(),
                    delivered: true,
                    sent_message_id: Some(MessageId(99)),
                })
                .is_ok()
        );
        assert_eq!(service.billing.settlements.len(), 1);
        assert!(!service.pending.contains_key(&completion_id));
    }

    #[test]
    fn malformed_delivery_pricing_releases_the_active_operation() {
        let mut service = conversation(vec![Ok(round("answer", None))], Billing::default());
        let Ok(AiPreparation::Reply {
            completion_id: Some(completion_id),
            ..
        }) = service.prepare(input())
        else {
            return;
        };
        let Some(pending) = service.pending.get_mut(&completion_id) else {
            return;
        };
        pending.segments = vec![json!("invalid segment")];

        assert!(
            service
                .complete_delivery(AiDelivery {
                    completion_id: completion_id.clone(),
                    delivered: true,
                    sent_message_id: Some(MessageId(99)),
                })
                .is_err()
        );
        assert!(!service.pending.contains_key(&completion_id));
        assert_eq!(service.billing.released_operations, [completion_id]);
        assert!(service.billing.settlements.is_empty());
    }

    #[test]
    fn schedules_background_compaction_only_after_successful_delivery() -> Result<(), String> {
        let scheduled = Rc::new(RefCell::new(Vec::new()));
        let mut service = conversation(vec![Ok(round("answer", None))], Billing::default())
            .with_compaction_scheduler(Box::new(Scheduler(Rc::clone(&scheduled))));
        service.state.memory.compaction_plan = Some(MemoryCompactionPlan {
            chat_id: "42".to_owned(),
            messages: vec![json!({"id":"1","role":"user","text":"old"})],
            prior_summary: Some("prior summary".to_owned()),
            expected_marker: Some("0".to_owned()),
            target_marker: "1".to_owned(),
        });
        let prepared = service.prepare(input())?;
        let AiPreparation::Reply {
            completion_id: Some(completion_id),
            ..
        } = prepared
        else {
            return Err("expected a prepared reply".to_owned());
        };
        assert!(scheduled.borrow().is_empty());
        service.complete_delivery(AiDelivery {
            completion_id,
            delivered: true,
            sent_message_id: Some(MessageId(99)),
        })?;
        let scheduled = scheduled.borrow();
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].0.target_marker, "1");
        assert_eq!(scheduled[0].1.user_id, 88);
        assert_eq!(scheduled[0].1.origin_chat_id, 42);
        assert_eq!(scheduled[0].1.group_chat_id, None);
        assert_eq!(scheduled[0].1.message_id, 7);
        assert_eq!(scheduled[0].1.payer_source, Some(PayerSource::User));
        Ok(())
    }

    #[test]
    fn direct_task_command_checks_future_credits_and_prompts_the_task_tool() -> Result<(), String> {
        let mut task_input = input();
        task_input.command = "/task".to_owned();
        task_input.message_text = "remind me tomorrow at nine".to_owned();
        let mut service = conversation(
            vec![Ok(round("task scheduled", None))],
            Billing {
                personal_balance: Some(10_000),
                ..Billing::default()
            },
        );
        let prepared = service.prepare(task_input.clone())?;
        assert!(matches!(prepared, AiPreparation::Reply { .. }));
        let prompts = service.provider.prompts.borrow();
        let prompt_text = prompts[0]
            .iter()
            .filter_map(|message| match &message.content {
                PromptContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(prompt_text.iter().any(|text| {
            text.contains(
                "create a scheduled task for this request using the task_set tool: remind me tomorrow at nine",
            )
        }));
        drop(prompts);
        assert_eq!(
            service.state.incoming[0].message_text,
            task_input.message_text
        );

        let mut denied = conversation(
            vec![Ok(round("must not run", None))],
            Billing {
                personal_balance: Some(0),
                ..Billing::default()
            },
        );
        let denied_reply = denied.prepare(task_input)?;
        assert!(matches!(
            denied_reply,
            AiPreparation::Reply { ref text, completion_id: None, .. }
                if text.starts_with("you do not have enough personal credits")
        ));
        assert!(denied.provider.prompts.borrow().is_empty());
        assert!(denied.billing.reserves.is_empty());
        Ok(())
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
    fn explicit_media_command_transcribes_and_settles_only_after_delivery() {
        let mut service = conversation(Vec::new(), Billing::default()).with_media(Box::new(Media));
        let mut request = input();
        request.command = "/transcribe".to_owned();
        request.has_reply = true;
        request.audio_media_kind = Some("voice".to_owned());
        request.audio_file_id = Some("audio-1".to_owned());
        request.audio_duration_seconds = Some(4.5);
        let preparation = service.prepare_media_command(request);
        let Ok(Some(AiPreparation::Reply {
            text,
            completion_id: Some(completion_id),
            ..
        })) = preparation
        else {
            return;
        };
        assert_eq!(text, "🎵 audio transcription: synthetic transcript");
        assert_eq!(
            service.billing.reserves[0].metadata["usage_tag"],
            "transcribe_command_media"
        );
        assert!(service.billing.settlements.is_empty());

        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.segments.len(), 1);
        assert_eq!(
            service.billing.settlements[0].reason,
            "transcribe_command_success"
        );
        assert!(service.state.incoming.is_empty());
        assert!(service.state.outgoing.is_empty());
    }

    #[test]
    fn explicit_media_command_accepts_media_attached_to_the_command_message() {
        let mut service = conversation(Vec::new(), Billing::default()).with_media(Box::new(Media));
        let mut request = input();
        request.command = "/transcribe".to_owned();
        request.has_reply = false;
        request.audio_media_kind = Some("voice".to_owned());
        request.audio_file_id = Some("attached-audio".to_owned());
        request.audio_duration_seconds = Some(4.5);

        assert!(matches!(
            service.prepare_media_command(request),
            Ok(Some(AiPreparation::Reply { ref text, .. }))
                if text == "🎵 audio transcription: synthetic transcript"
        ));
    }

    #[test]
    fn explicit_media_command_preserves_reply_help_and_refunds_after_delivery() {
        let mut service = conversation(Vec::new(), Billing::default()).with_media(Box::new(Media));
        let preparation = service.prepare_media_command(input());
        let Ok(Some(AiPreparation::Reply {
            text,
            completion_id: Some(completion_id),
            ..
        })) = preparation
        else {
            return;
        };
        assert_eq!(
            text,
            "reply to an audio, video, image, or sticker and I will process it"
        );
        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.settlements[0].actual_credit_units, 0);
    }

    #[test]
    fn explicit_sticker_command_uses_sticker_copy_and_sanitizes_cached_text() {
        let mut service =
            conversation(Vec::new(), Billing::default()).with_media(Box::new(StickerMedia));
        let mut request = input();
        request.command = "/describe".to_owned();
        request.has_reply = true;
        request.visual_media_kind = Some("sticker".to_owned());
        request.photo_file_id = Some("sticker-1".to_owned());
        let preparation = service.prepare_media_command(request);
        assert!(matches!(
            preparation,
            Ok(Some(AiPreparation::Reply { ref text, .. }))
                if text == "🎨 sticker: synthetic sticker"
        ));
    }

    #[test]
    fn explicit_media_download_failure_is_localized_and_refundable() {
        let mut service =
            conversation(Vec::new(), Billing::default()).with_media(Box::new(DownloadFailureMedia));
        let mut request = input();
        request.command = "/transcribe".to_owned();
        request.has_reply = true;
        request.audio_file_id = Some("voice-1".to_owned());
        request.audio_duration_seconds = Some(2.0);
        let preparation = service.prepare_media_command(request);
        let Ok(Some(AiPreparation::Reply {
            text,
            completion_id: Some(completion_id),
            diagnostics,
        })) = preparation
        else {
            return;
        };
        assert_eq!(text, "I could not download the audio, send it again");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.settlements[0].actual_credit_units, 0);
    }

    #[test]
    fn summary_command_streams_custom_prompt_and_settles_summary_usage() {
        let mut service = conversation(
            vec![Ok(round("**synthetic summary**", None))],
            Billing::default(),
        );
        let mut request = input();
        request.command = "/summary".to_owned();
        request.message_text = "25 focus on decisions".to_owned();
        let mut streamed = String::new();
        let preparation = service.prepare_summary_command_streaming(request, &mut |token| {
            streamed.push_str(token);
            Ok(())
        });
        let Ok(Some(AiPreparation::Reply {
            text,
            completion_id: Some(completion_id),
            ..
        })) = preparation
        else {
            return;
        };
        assert_eq!(streamed, "**synthetic summary**");
        assert_eq!(text, "synthetic summary");
        let prompts = service.provider.prompts.borrow();
        let PromptContent::Text(prompt) =
            &prompts[0].last().unwrap_or_else(|| unreachable!()).content
        else {
            return;
        };
        assert!(prompt.starts_with("focus on decisions. update the previous summary"));
        drop(prompts);

        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.segments[0].segment["kind"], "summary");
        assert_eq!(
            service.billing.settlements[0].reason,
            "summary_command_stream_success"
        );
        assert!(service.state.outgoing.is_empty());
    }

    #[test]
    fn summary_command_returns_cached_or_empty_text_without_provider_io() {
        let mut cached = conversation(Vec::new(), Billing::default());
        cached.state.memory.history.clear();
        cached.state.memory.summary = Some("**cached summary**".to_owned());
        let mut cached_tokens = String::new();
        let cached_result = cached.prepare_summary_command_streaming(input(), &mut |token| {
            cached_tokens.push_str(token);
            Ok(())
        });
        let Ok(Some(AiPreparation::Reply {
            text,
            completion_id: Some(completion_id),
            ..
        })) = cached_result
        else {
            return;
        };
        assert_eq!(text, "cached summary");
        assert_eq!(cached_tokens, "cached summary");
        assert!(cached.provider.prompts.borrow().is_empty());
        assert_eq!(
            cached.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(cached.billing.settlements[0].actual_credit_units, 0);

        let mut empty = conversation(Vec::new(), Billing::default());
        empty.state.memory.history.clear();
        empty.state.memory.summary = None;
        let empty_result = empty.prepare_summary_command_streaming(input(), &mut |_token| Ok(()));
        assert!(matches!(
            empty_result,
            Ok(Some(AiPreparation::Reply { ref text, .. }))
                if text == "there are no messages to summarize"
        ));
        assert!(empty.provider.prompts.borrow().is_empty());
    }

    #[test]
    fn summary_provider_failure_refunds_empty_usage_after_delivery() {
        let mut service = conversation(
            vec![Err(ChatRoundError {
                source: OpenRouterChatError::IncompleteStream,
                partial: Box::new(ChatRoundResult {
                    text: String::new(),
                    tool_calls: Vec::new(),
                    finish_reason: None,
                    billing_segment: None,
                }),
            })],
            Billing::default(),
        );
        let preparation = service.prepare_summary_command_streaming(input(), &mut |_token| Ok(()));
        let Ok(Some(AiPreparation::Reply {
            text,
            completion_id: Some(completion_id),
            ..
        })) = preparation
        else {
            return;
        };
        assert_eq!(text, "I could not generate the summary");
        assert_eq!(
            service.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(MessageId(99)),
            }),
            Ok(())
        );
        assert_eq!(service.billing.settlements[0].actual_credit_units, 0);
        assert_eq!(
            service.billing.settlements[0].reason,
            "summary_stream_fallback"
        );
    }

    #[test]
    fn denied_and_spontaneous_turns_never_call_the_provider() {
        let denied = ReserveDecision {
            authorized: false,
            user_balance: 25,
            chat_balance: 50,
            source: None,
            denial: None,
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
    fn creditless_hourly_cap_denial_uses_the_specific_localized_reply() {
        let denial = ReserveDecision {
            authorized: false,
            user_balance: 0,
            chat_balance: 5_000,
            source: Some(PayerSource::Chat),
            denial: Some(ReserveDenial::CreditlessHourlyCap { limit: 3 }),
        };
        let mut service = conversation(
            vec![Ok(round("must not run", None))],
            Billing {
                decisions: VecDeque::from([denial]),
                ..Billing::default()
            },
        );
        let mut request = input();
        request.chat_type = "supergroup".to_owned();
        assert_eq!(
            service.prepare(request),
            Ok(AiPreparation::reply(
                "you reached the limit of 3 group-funded AI messages per hour. use /topup to continue",
                None,
            ))
        );
        assert!(service.provider.prompts.borrow().is_empty());
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
