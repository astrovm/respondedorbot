//! Concrete native AI and billing adapters for scheduled task execution.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::billing_read::{AiChargeResult, BillingRepository};
use bot_adapters::openrouter_chat::{
    ChatCompletion, ChatCompletionRequest, ChatMessage, ChatRole, OpenRouterChatError,
    OpenRouterPricingCache, OpenRouterTransport, complete_with,
};
use bot_core::ai_pricing::calculate_billing_for_segments;
use bot_core::ai_prompt::build_system_prompt;
use bot_core::ai_reserve::{
    EstimatedMessage, ReserveEstimateError, TokenEstimateValue, chat_output_token_limit,
    estimate_chat_reserve_credit_units_with_pricing, estimate_firecrawl_reserve_credit_units,
};
use bot_core::ai_usage::stable_provider_segment_id;
use bot_core::credit_units::{CreditUnits, display_credit_units};
use bot_core::locale::{Locale, format_date};
#[cfg(test)]
use bot_core::provider_pricing::DEEPSEEK_FLASH_MODEL;
use bot_core::provider_pricing::{DEEPSEEK_MODEL, TokenPricing};
use bot_core::scheduled_tasks::ScheduledTask;
use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_input::ChatId;
use chrono::{DateTime, FixedOffset, Offset, Utc};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::dispatcher::ActionSink;
use crate::firecrawl_tool::ScheduledWebSearch;
use crate::native_tools::{NativeTool, tool_schema};
use crate::task_executor::{
    TaskAiProvider, TaskBilling, TaskMessenger, TaskPromptMessage, TaskProviderFailure,
    TaskProviderReply, TaskReserveOutcome, build_task_messages,
};
use crate::tool_requests::validate_request;

pub const PRIMARY_CHAT_MODEL: &str = DEEPSEEK_MODEL;
pub const VISION_MODEL: &str = DEEPSEEK_MODEL;
pub const OPENROUTER_TRANSCRIPTION_MODEL: &str =
    bot_core::provider_pricing::OPENROUTER_TRANSCRIPTION_MODEL;
const SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE: i64 = 4_000;
const TASK_WEB_SEARCH_MAX_USES: usize = 3;

pub fn reservation_pricing_for_model(
    model: &str,
    pricing: Option<&OpenRouterPricingCache>,
) -> Result<TokenPricing, String> {
    if let Some(pricing) = pricing {
        return pricing
            .pricing(model)
            .map_err(|error| format!("OpenRouter pricing lookup failed: {error}"))?
            .ok_or_else(|| format!("OpenRouter catalog has no pricing for {model}"));
    }
    #[cfg(test)]
    if model.split(':').next() == Some(DEEPSEEK_MODEL) {
        return Ok(TokenPricing {
            input_per_million: 300_000,
            cached_input_per_million: Some(6_000),
            cache_write_per_million: None,
            audio_input_per_million: None,
            output_per_million: 1_200_000,
        });
    }
    #[cfg(test)]
    if model.split(':').next() == Some(DEEPSEEK_FLASH_MODEL) {
        return Ok(TokenPricing {
            input_per_million: 67_900,
            cached_input_per_million: Some(16_800),
            cache_write_per_million: None,
            audio_input_per_million: None,
            output_per_million: 168_000,
        });
    }
    Err(format!("OpenRouter pricing is unavailable for {model}"))
}

pub fn estimate_task_reserve_credit_units(
    text: &str,
    locale: &str,
    pricing: Option<&OpenRouterPricingCache>,
) -> Result<i64, String> {
    let estimated_messages = build_task_messages(text, locale)
        .into_iter()
        .map(|message| EstimatedMessage {
            role: TokenEstimateValue::Text(message.role.to_owned()),
            content: TokenEstimateValue::Text(message.content),
            name: TokenEstimateValue::Empty,
        })
        .collect::<Vec<_>>();
    let pricing = reservation_pricing_for_model(PRIMARY_CHAT_MODEL, pricing)?;
    estimate_chat_reserve_credit_units_with_pricing(
        None,
        &estimated_messages,
        Some(chat_output_token_limit(PRIMARY_CHAT_MODEL)),
        SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE,
        PRIMARY_CHAT_MODEL,
        &pricing,
    )
    .map_err(|error| error.to_string())
}

fn add_task_web_search_reserve(chat: i64) -> Result<i64, ReserveEstimateError> {
    let search = estimate_firecrawl_reserve_credit_units()?;
    let searches = search
        .checked_mul(TASK_WEB_SEARCH_MAX_USES as i64)
        .ok_or(ReserveEstimateError::Overflow)?;
    chat.checked_add(searches)
        .ok_or(ReserveEstimateError::Overflow)
}

pub struct OpenRouterTaskProvider<Transport> {
    transport: Transport,
    api_key: String,
    base_url: String,
    model: String,
    persona: String,
    pricing: Option<Arc<OpenRouterPricingCache>>,
    web_search: Option<Box<dyn ScheduledWebSearch>>,
}

impl<Transport> OpenRouterTaskProvider<Transport> {
    #[must_use]
    pub fn new(
        transport: Transport,
        api_key: &str,
        base_url: &str,
        model: &str,
        persona: &str,
    ) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            base_url: base_url.to_owned(),
            model: model.to_owned(),
            persona: persona.to_owned(),
            pricing: None,
            web_search: None,
        }
    }

    #[must_use]
    pub fn with_openrouter_pricing(mut self, pricing: Arc<OpenRouterPricingCache>) -> Self {
        self.pricing = Some(pricing);
        self
    }

    #[must_use]
    pub fn with_web_search(mut self, web_search: Box<dyn ScheduledWebSearch>) -> Self {
        self.web_search = Some(web_search);
        self
    }

    fn request(
        &self,
        messages: &[TaskPromptMessage],
        task: &ScheduledTask,
    ) -> Result<ChatCompletionRequest, OpenRouterChatError> {
        let mut request_messages = Vec::with_capacity(messages.len() + 1);
        request_messages.push(ChatMessage::text(
            ChatRole::System,
            task_system_prompt(&self.persona, task),
        ));
        request_messages.extend(messages.iter().map(|message| {
            ChatMessage::text(task_message_role(message.role), message.content.clone())
        }));
        let mut request = ChatCompletionRequest::new(&self.model, request_messages);
        if let Some(pricing) = self.pricing.as_ref() {
            pricing.apply_to_request(&mut request)?;
        }
        request.max_tokens = u64::try_from(chat_output_token_limit(&self.model)).ok();
        if self.web_search.is_some() {
            request.tools = vec![tool_schema(NativeTool::WebSearch)];
        }
        Ok(request)
    }
}

impl<Transport: OpenRouterTransport> TaskAiProvider for OpenRouterTaskProvider<Transport> {
    type Error = OpenRouterChatError;

    fn complete(
        &mut self,
        messages: &[TaskPromptMessage],
        task: &ScheduledTask,
        _execution_id: &str,
    ) -> Result<TaskProviderReply, TaskProviderFailure<Self::Error>> {
        let mut request = self
            .request(messages, task)
            .map_err(|source| TaskProviderFailure::new(source, Vec::new()))?;
        let mut billing_segments = Vec::new();
        let mut text = String::new();
        let mut search_uses = 0_usize;
        let locale = if task.locale == "en" {
            Locale::En
        } else {
            Locale::Es
        };
        for _ in 0..crate::chat_tool_loop::DEFAULT_MAX_TOOL_ROUNDS {
            let completion =
                complete_with(&self.transport, &self.api_key, &self.base_url, &request)
                    .map_err(|source| TaskProviderFailure::new(source, billing_segments.clone()))?;
            billing_segments.push(task_completion_segment(&completion));
            text = completion.text.clone();
            let calls = completion
                .tool_calls
                .into_iter()
                .filter(|call| call.function.name == NativeTool::WebSearch.name())
                .collect::<Vec<_>>();
            if calls.is_empty() {
                break;
            }
            request
                .messages
                .push(ChatMessage::assistant_tool_calls(calls.clone()));
            for call in calls {
                let arguments = serde_json::from_str(&call.function.arguments)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| json!({}));
                let result = if search_uses >= TASK_WEB_SEARCH_MAX_USES {
                    crate::chat_tool_loop::ToolExecutionResult::output(
                        "web_search usage limit reached",
                    )
                } else {
                    search_uses += 1;
                    validate_request(NativeTool::WebSearch, &arguments, locale).map_or_else(
                        crate::chat_tool_loop::ToolExecutionResult::output,
                        |tool_request| {
                            self.web_search.as_mut().map_or_else(
                                || {
                                    crate::chat_tool_loop::ToolExecutionResult::output(
                                        "web_search is unavailable",
                                    )
                                },
                                |search| search.execute(tool_request, &call.id, locale),
                            )
                        },
                    )
                };
                if let Some(segment) = result.billing_segment {
                    billing_segments.push(segment);
                }
                request
                    .messages
                    .push(ChatMessage::tool_result(call.id, result.output));
            }
        }
        Ok(TaskProviderReply {
            text,
            fallback: false,
            billing_segments,
        })
    }
}

fn task_completion_segment(completion: &ChatCompletion) -> Value {
    let annotation_types = completion
        .annotations
        .iter()
        .filter_map(|annotation| annotation.get("type").and_then(Value::as_str))
        .collect::<Vec<_>>();
    let mut metadata = Map::from_iter([("provider".to_owned(), json!("openrouter"))]);
    optional_string(
        &mut metadata,
        "upstream_provider",
        completion.upstream_provider.clone(),
    );
    optional_string(
        &mut metadata,
        "service_tier",
        completion.service_tier.clone(),
    );
    optional_string(
        &mut metadata,
        "provider_generation_id",
        completion.generation_id.clone(),
    );
    if !annotation_types.is_empty() {
        metadata.insert(
            "web_search_citation_count".to_owned(),
            json!(
                annotation_types
                    .iter()
                    .filter(|kind| **kind == "url_citation")
                    .count()
            ),
        );
    }
    json!({
        "kind": "chat",
        "model": completion.model,
        "usage": completion.usage,
        "source": "openrouter",
        "metadata": metadata,
    })
}

fn optional_string(metadata: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        metadata.insert(key.to_owned(), json!(value));
    }
}

fn task_message_role(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn task_system_prompt(persona: &str, task: &ScheduledTask) -> String {
    let locale = if task.locale == "en" {
        Locale::En
    } else {
        Locale::Es
    };
    build_system_prompt(
        persona,
        locale,
        &formatted_date(i64::from(task.timezone_offset), locale),
        true,
        true,
    )
}

fn formatted_date(timezone_offset_hours: i64, locale: Locale) -> String {
    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64);
    let utc =
        DateTime::<Utc>::from_timestamp(unix_seconds, 0).unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
    let seconds = timezone_offset_hours.clamp(-23, 23).saturating_mul(3_600) as i32;
    let offset = FixedOffset::east_opt(seconds).unwrap_or(Utc.fix());
    format_date(utc.with_timezone(&offset).date_naive(), locale)
}

pub trait TaskCreditStore {
    type Error: std::fmt::Display;

    fn charge(
        &self,
        user_id: i64,
        amount: i32,
        metadata: &Map<String, Value>,
        idempotency_key: &str,
        operation_id: &str,
    ) -> Result<AiChargeResult, Self::Error>;

    fn record_segment(&self, user_id: i64, metadata: &Value) -> Result<bool, Self::Error>;

    fn list_segments(&self, user_id: i64, operation_id: &str) -> Result<Vec<Value>, Self::Error>;

    fn settle_once(
        &self,
        user_id: i64,
        operation_id: &str,
        actual_credit_units: i64,
        metadata: &Map<String, Value>,
    ) -> Result<bool, Self::Error>;
}

impl TaskCreditStore for BillingRepository {
    type Error = bot_adapters::billing_read::BillingError;

    fn charge(
        &self,
        user_id: i64,
        amount: i32,
        metadata: &Map<String, Value>,
        idempotency_key: &str,
        operation_id: &str,
    ) -> Result<AiChargeResult, Self::Error> {
        self.charge_ai_credits(
            user_id,
            None,
            amount,
            "ai_reserve",
            metadata,
            Some("user"),
            Some(idempotency_key),
            operation_id,
        )
    }

    fn record_segment(&self, user_id: i64, metadata: &Value) -> Result<bool, Self::Error> {
        self.record_ai_provider_usage(user_id, None, metadata)
    }

    fn list_segments(&self, user_id: i64, operation_id: &str) -> Result<Vec<Value>, Self::Error> {
        self.list_ai_provider_segments(user_id, operation_id)
    }

    fn settle_once(
        &self,
        user_id: i64,
        operation_id: &str,
        actual_credit_units: i64,
        metadata: &Map<String, Value>,
    ) -> Result<bool, Self::Error> {
        self.settle_ai_operation_once(user_id, None, operation_id, actual_credit_units, metadata)
            .map(|result| result.applied)
    }
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum NativeTaskBillingError {
    #[error("scheduled-task credit estimate failed: {0}")]
    Estimate(String),
    #[error("scheduled-task credit amount exceeds the database range")]
    AmountRange,
    #[error("scheduled-task credit storage failed: {0}")]
    Store(String),
    #[error("scheduled-task usage pricing failed: {0}")]
    Pricing(String),
    #[error("scheduled task has no billable user")]
    MissingUser,
}

impl NativeTaskBillingError {
    fn store(error: impl std::fmt::Display) -> Self {
        Self::Store(error.to_string())
    }

    fn estimate(error: ReserveEstimateError) -> Self {
        Self::Estimate(error.to_string())
    }
}

pub struct PostgresTaskBilling<Store> {
    store: Store,
    model: String,
    pricing: Option<Arc<OpenRouterPricingCache>>,
    web_search_enabled: bool,
}

impl<Store> PostgresTaskBilling<Store> {
    #[must_use]
    pub fn new(store: Store, model: &str) -> Self {
        Self {
            store,
            model: model.to_owned(),
            pricing: None,
            web_search_enabled: false,
        }
    }

    #[must_use]
    pub fn with_openrouter_pricing(mut self, pricing: Arc<OpenRouterPricingCache>) -> Self {
        self.pricing = Some(pricing);
        self
    }

    #[must_use]
    pub const fn with_web_search(mut self, enabled: bool) -> Self {
        self.web_search_enabled = enabled;
        self
    }

    fn settle_amount(
        &self,
        task: &ScheduledTask,
        execution_id: &str,
        segments: &[Value],
        reason: &'static str,
        amount: i64,
        pricing: Option<&Value>,
    ) -> Result<(), NativeTaskBillingError>
    where
        Store: TaskCreditStore,
    {
        let user_id = task.user_id.ok_or(NativeTaskBillingError::MissingUser)?;
        let operation_id = operation_id(execution_id);
        let mut settlement = Map::new();
        settlement.insert("reason".to_owned(), json!(reason));
        settlement.insert("task_id".to_owned(), json!(task.id.as_str()));
        settlement.insert("execution_id".to_owned(), json!(execution_id));
        settlement.insert("operation_id".to_owned(), json!(operation_id));
        if !segments.is_empty() {
            settlement.insert("billing_segments".to_owned(), json!(segments));
        }
        if let Some(pricing) = pricing {
            copy_task_pricing_metadata(&mut settlement, pricing);
        }
        self.store
            .settle_once(user_id, &operation_id, amount, &settlement)
            .map_err(NativeTaskBillingError::store)?;
        Ok(())
    }

    fn finalize(
        &self,
        task: &ScheduledTask,
        execution_id: &str,
        new_segments: &[Value],
        reason: &'static str,
    ) -> Result<(), NativeTaskBillingError>
    where
        Store: TaskCreditStore,
    {
        let user_id = task.user_id.ok_or(NativeTaskBillingError::MissingUser)?;
        let operation_id = operation_id(execution_id);
        for segment in new_segments {
            let metadata = json!({
                "operation_id": operation_id,
                "segment_id": stable_provider_segment_id(segment),
                "task_id": task.id.as_str(),
                "execution_id": execution_id,
                "segment": segment,
            });
            self.store
                .record_segment(user_id, &metadata)
                .map_err(NativeTaskBillingError::store)?;
        }
        let durable_segments = self
            .store
            .list_segments(user_id, &operation_id)
            .map_err(NativeTaskBillingError::store)?;
        if durable_segments.is_empty() {
            return self.settle_amount(task, execution_id, &[], reason, 0, None);
        }
        let pricing = calculate_billing_for_segments(&json!(durable_segments))
            .map_err(|error| NativeTaskBillingError::Pricing(error.to_string()))?;
        if pricing.get("pricing_complete").and_then(Value::as_bool) != Some(true) {
            return Ok(());
        }
        let amount = pricing
            .get("charged_credit_units")
            .and_then(Value::as_i64)
            .ok_or(NativeTaskBillingError::Pricing(String::from(
                "missing charge total",
            )))?;
        self.settle_amount(
            task,
            execution_id,
            &durable_segments,
            reason,
            amount,
            Some(&pricing),
        )
    }
}

fn copy_task_pricing_metadata(metadata: &mut Map<String, Value>, pricing: &Value) {
    for key in [
        "pricing_version",
        "raw_usd_micros",
        "markup_multiplier",
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

impl<Store: TaskCreditStore> TaskBilling for PostgresTaskBilling<Store> {
    type Error = NativeTaskBillingError;

    fn reserve(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        prompt: &[TaskPromptMessage],
    ) -> Result<TaskReserveOutcome, Self::Error> {
        let user_id = task.user_id.ok_or(NativeTaskBillingError::MissingUser)?;
        let estimated_messages = prompt
            .iter()
            .map(|message| EstimatedMessage {
                role: TokenEstimateValue::Text(message.role.to_owned()),
                content: TokenEstimateValue::Text(message.content.clone()),
                name: TokenEstimateValue::Empty,
            })
            .collect::<Vec<_>>();
        let pricing = reservation_pricing_for_model(&self.model, self.pricing.as_deref())
            .map_err(NativeTaskBillingError::Estimate)?;
        let chat_amount = estimate_chat_reserve_credit_units_with_pricing(
            None,
            &estimated_messages,
            Some(chat_output_token_limit(&self.model)),
            SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE,
            &self.model,
            &pricing,
        )
        .map_err(NativeTaskBillingError::estimate)?;
        let amount = if self.web_search_enabled {
            add_task_web_search_reserve(chat_amount).map_err(NativeTaskBillingError::estimate)?
        } else {
            chat_amount
        };
        let amount_i32 = i32::try_from(amount).map_err(|_| NativeTaskBillingError::AmountRange)?;
        let operation_id = operation_id(execution_id);
        let idempotency_key = format!("{operation_id}:reserve");
        let metadata = Map::from_iter([
            ("command".to_owned(), json!("task")),
            ("usage_tag".to_owned(), json!("task_ai")),
            ("settlement_id".to_owned(), json!(idempotency_key)),
            ("operation_id".to_owned(), json!(operation_id)),
            ("task_id".to_owned(), json!(task.id.as_str())),
            ("execution_id".to_owned(), json!(execution_id)),
            ("chat_id".to_owned(), json!(task.chat_id)),
            ("reserved_credit_units".to_owned(), json!(amount)),
            ("model".to_owned(), json!(self.model)),
        ]);
        let result = self
            .store
            .charge(
                user_id,
                amount_i32,
                &metadata,
                &idempotency_key,
                &operation_id,
            )
            .map_err(NativeTaskBillingError::store)?;
        if result.ok {
            return Ok(TaskReserveOutcome::Authorized);
        }
        if matches!(
            result.reason.as_deref(),
            Some("operation_settled" | "reservation_refunded")
        ) {
            return Ok(TaskReserveOutcome::AlreadySettled);
        }
        Ok(TaskReserveOutcome::Denied {
            message: insufficient_credits_message(task.locale.as_str(), result.user_balance),
        })
    }

    fn settle(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        segments: &[Value],
        reason: &'static str,
    ) -> Result<(), Self::Error> {
        self.finalize(task, execution_id, segments, reason)
    }

    fn refund(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        reason: &'static str,
    ) -> Result<(), Self::Error> {
        self.finalize(task, execution_id, &[], reason)
    }
}

fn operation_id(execution_id: &str) -> String {
    format!("task:{execution_id}")
}

fn insufficient_credits_message(locale: &str, balance: i64) -> String {
    let balance = display_credit_units(CreditUnits::new(balance));
    if locale == "en" {
        format!("You are out of AI credits. Balance: {balance}\nUse /topup to add more")
    } else {
        format!(
            "Te quedaste seco de créditos de IA, boludo. Saldo: {balance}\nMetele /topup si querés que siga laburando"
        )
    }
}

pub struct ActionTaskMessenger<Sink> {
    sink: Sink,
}

impl<Sink> ActionTaskMessenger<Sink> {
    #[must_use]
    pub const fn new(sink: Sink) -> Self {
        Self { sink }
    }
}

impl<Sink> TaskMessenger for ActionTaskMessenger<Sink>
where
    Sink: ActionSink,
    Sink::Error: std::fmt::Display,
{
    type Error = String;

    fn send(&mut self, chat_id: &str, text: &str) -> Result<(), Self::Error> {
        let chat_id = chat_id
            .parse::<i64>()
            .map_err(|_| "scheduled-task Telegram chat ID is invalid".to_owned())?;
        self.sink
            .execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(chat_id),
                text,
            )))
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Arc;

    use bot_adapters::billing_read::AiChargeResult;
    use bot_adapters::openrouter_chat::{
        HttpRequest, HttpResponse, OpenRouterChatError, OpenRouterPricingCache, OpenRouterTransport,
    };
    use bot_core::locale::Locale;
    use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};
    use bot_core::telegram_actions::TelegramAction;
    use serde_json::{Map, Value, json};

    use super::{
        ActionTaskMessenger, NativeTaskBillingError, NativeTool, OpenRouterTaskProvider,
        PostgresTaskBilling, TaskAiProvider, TaskBilling, TaskCreditStore, TaskMessenger,
        TaskPromptMessage, TaskReserveOutcome, reservation_pricing_for_model, validate_request,
    };
    use crate::chat_tool_loop::ToolExecutionResult;
    use crate::dispatcher::{ActionReceipt, ActionSink};
    use crate::firecrawl_tool::ScheduledWebSearch;
    use crate::tool_requests::ExternalToolRequest;

    struct Search;

    impl ScheduledWebSearch for Search {
        fn execute(
            &mut self,
            _request: ExternalToolRequest,
            _tool_call_id: &str,
            _locale: Locale,
        ) -> ToolExecutionResult {
            ToolExecutionResult {
                output: "synthetic search".to_owned(),
                failure_fallback: None,
                billing_segment: Some(json!({
                    "kind": "web_search",
                    "model": "",
                    "usage": {},
                    "source": "firecrawl",
                    "metadata": {
                        "tool_call_id": "call-1",
                        "firecrawl_credits_used": 2
                    }
                })),
                diagnostics: Vec::new(),
            }
        }
    }

    struct Transport {
        requests: RefCell<Vec<HttpRequest>>,
        responses: RefCell<VecDeque<HttpResponse>>,
    }

    impl OpenRouterTransport for Transport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            self.responses
                .borrow_mut()
                .pop_front()
                .ok_or(OpenRouterChatError::Transport(
                    "missing response".to_owned(),
                ))
        }
    }

    fn task(locale: &str) -> ScheduledTask {
        let id = TaskId::new("task123");
        let Ok(id) = id else { panic!() };
        ScheduledTask {
            id,
            chat_id: "-100123".to_owned(),
            text: "synthetic task".to_owned(),
            user_name: "synthetic-user".to_owned(),
            user_id: Some(42),
            schedule: TaskSchedule::Once,
            timezone_offset: -3,
            locale: locale.to_owned(),
            schedule_anchor_at: Some(1_000),
            next_run_at: Some(1_000),
            last_execution_id: None,
        }
    }

    #[test]
    fn openrouter_task_request_preserves_system_tool_usage_and_billing_contract() {
        let transport = Transport {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from([HttpResponse {
                status_code: 200,
                body: json!({
                    "id": "generation-1",
                    "model": "resolved/model",
                    "provider": "Synthetic Provider",
                    "service_tier": "paid",
                    "choices": [{
                        "message": {
                            "content": "synthetic answer",
                            "annotations": [{"type": "url_citation"}]
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7}
                })
                .to_string(),
                headers: BTreeMap::new(),
            }])),
        };
        let mut provider = OpenRouterTaskProvider::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "deepseek/deepseek-v4.1-flash",
            "synthetic persona",
        )
        .with_web_search(Box::new(Search));
        let reply = provider.complete(
            &[TaskPromptMessage {
                role: "user",
                content: "do it".to_owned(),
            }],
            &task("en"),
            "task123:1000",
        );
        let Ok(reply) = reply else { panic!() };
        assert_eq!(reply.text, "synthetic answer");
        assert!(!reply.fallback);
        assert_eq!(reply.billing_segments[0]["model"], "resolved/model");
        assert_eq!(
            reply.billing_segments[0]["metadata"]["provider_generation_id"],
            "generation-1"
        );
        assert_eq!(
            reply.billing_segments[0]["metadata"]["web_search_citation_count"],
            1
        );
        let request = &provider.transport.requests.borrow()[0];
        assert_eq!(request.bearer_token, "synthetic-key");
        let body = serde_json::from_str::<Value>(&request.body);
        let Ok(body) = body else { panic!() };
        assert_eq!(body["max_tokens"], 8_192);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "web_search");
        assert!(
            body["messages"][0]["content"]
                .as_str()
                .is_some_and(|content| content.contains("RUNNING SCHEDULED TASK:"))
        );
        assert!(
            body["messages"][0]["content"]
                .as_str()
                .is_some_and(|content| content.contains("synthetic persona"))
        );
        assert_eq!(body["messages"][1]["content"], "do it");
    }

    #[test]
    fn reservation_pricing_reports_catalog_lookup_failures() {
        let pricing = OpenRouterPricingCache::new("synthetic-key", "not-a-url");
        let Ok(pricing) = pricing else { panic!() };
        let result = reservation_pricing_for_model("synthetic/model", Some(&pricing));
        assert!(result.is_err());
    }

    #[test]
    fn task_pricing_lookup_failure_stops_provider_before_transport_io() {
        let transport = Transport {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::new()),
        };
        let pricing = OpenRouterPricingCache::new("synthetic-key", "not-a-url");
        let Ok(pricing) = pricing else { panic!() };
        let pricing = Arc::new(pricing);
        let mut provider = OpenRouterTaskProvider::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "deepseek/deepseek-v4.1-flash",
            "synthetic persona",
        )
        .with_openrouter_pricing(pricing);

        let failure = provider
            .complete(
                &[TaskPromptMessage {
                    role: "user",
                    content: "do it".to_owned(),
                }],
                &task("en"),
                "task123:1000",
            )
            .err();
        let Some(failure) = failure else { panic!() };
        assert_eq!(failure.source, OpenRouterChatError::InvalidBaseUrl);
        assert!(failure.billing_segments.is_empty());
        assert!(provider.transport.requests.borrow().is_empty());
    }

    #[test]
    fn scheduled_web_search_records_exact_firecrawl_usage_and_continues_the_tool_loop() {
        let transport = Transport {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from([
                HttpResponse {
                    status_code: 200,
                    body: json!({
                        "id": "generation-tool",
                        "model": "resolved/model",
                        "choices": [{
                            "message": {
                                "content": null,
                                "tool_calls": [{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": "{\"query\":\"synthetic query\"}"
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {"cost": 0.0001}
                    })
                    .to_string(),
                    headers: BTreeMap::new(),
                },
                HttpResponse {
                    status_code: 200,
                    body: json!({
                        "id": "generation-final",
                        "model": "resolved/model",
                        "choices": [{
                            "message": {"content": "final answer"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"cost": 0.0002}
                    })
                    .to_string(),
                    headers: BTreeMap::new(),
                },
            ])),
        };
        let mut provider = OpenRouterTaskProvider::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "deepseek/deepseek-v4.1-flash",
            "synthetic persona",
        )
        .with_web_search(Box::new(Search));

        let reply = provider.complete(
            &[TaskPromptMessage {
                role: "user",
                content: "search".to_owned(),
            }],
            &task("en"),
            "task123:1000",
        );
        let Ok(reply) = reply else { panic!() };

        assert_eq!(reply.text, "final answer");
        assert_eq!(reply.billing_segments.len(), 3);
        assert_eq!(reply.billing_segments[1]["source"], "firecrawl");
        assert_eq!(
            reply.billing_segments[1]["metadata"]["firecrawl_credits_used"],
            2
        );
        let requests = provider.transport.requests.borrow();
        assert_eq!(requests.len(), 2);
        let followup = serde_json::from_str::<Value>(&requests[1].body);
        let Ok(followup) = followup else { panic!() };
        assert_eq!(followup["messages"][2]["tool_calls"][0]["id"], "call-1");
        assert_eq!(followup["messages"][3]["tool_call_id"], "call-1");
    }

    #[test]
    fn scheduled_tool_loop_returns_incurred_usage_when_a_later_round_fails() {
        let transport = Transport {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from([
                HttpResponse {
                    status_code: 200,
                    body: json!({
                        "id": "generation-tool",
                        "model": "resolved/model",
                        "choices": [{
                            "message": {
                                "content": null,
                                "tool_calls": [{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": "{\"query\":\"synthetic query\"}"
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {"cost": 0.0001}
                    })
                    .to_string(),
                    headers: BTreeMap::new(),
                },
                HttpResponse {
                    status_code: 503,
                    body: json!({"error": {"message": "synthetic unavailable"}}).to_string(),
                    headers: BTreeMap::new(),
                },
            ])),
        };
        let mut provider = OpenRouterTaskProvider::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "deepseek/deepseek-v4.1-flash",
            "synthetic persona",
        )
        .with_web_search(Box::new(Search));

        let failure = provider
            .complete(
                &[TaskPromptMessage {
                    role: "user",
                    content: "search".to_owned(),
                }],
                &task("en"),
                "task123:1000",
            )
            .err();
        let Some(failure) = failure else { panic!() };

        assert_eq!(failure.billing_segments.len(), 2);
        assert_eq!(failure.billing_segments[0]["source"], "openrouter");
        assert_eq!(failure.billing_segments[1]["source"], "firecrawl");
    }

    type CapturedCharge = (i64, i32, String, String, Map<String, Value>);
    type CapturedSettlement = (i64, String, i64, Map<String, Value>);

    #[derive(Default)]
    struct Store {
        /// Name of the store operation that fails, if any.
        fail_on: Option<&'static str>,
        charge_result: RefCell<Option<AiChargeResult>>,
        charges: RefCell<Vec<CapturedCharge>>,
        segments: RefCell<Vec<Value>>,
        settlements: RefCell<Vec<CapturedSettlement>>,
    }

    impl TaskCreditStore for Store {
        type Error = &'static str;

        fn charge(
            &self,
            user_id: i64,
            amount: i32,
            metadata: &Map<String, Value>,
            idempotency_key: &str,
            operation_id: &str,
        ) -> Result<AiChargeResult, Self::Error> {
            if self.fail_on == Some("charge") {
                return Err("synthetic charge failure");
            }
            self.charges.borrow_mut().push((
                user_id,
                amount,
                idempotency_key.to_owned(),
                operation_id.to_owned(),
                metadata.clone(),
            ));
            Ok(self
                .charge_result
                .borrow_mut()
                .take()
                .unwrap_or(AiChargeResult {
                    ok: true,
                    applied: true,
                    reason: None,
                    source: Some("user".to_owned()),
                    amount: i64::from(amount),
                    user_balance: 900,
                    chat_balance: 0,
                }))
        }

        fn record_segment(&self, _user_id: i64, metadata: &Value) -> Result<bool, Self::Error> {
            if self.fail_on == Some("record") {
                return Err("synthetic record failure");
            }
            self.segments.borrow_mut().push(metadata.clone());
            Ok(true)
        }

        fn list_segments(
            &self,
            _user_id: i64,
            operation_id: &str,
        ) -> Result<Vec<Value>, Self::Error> {
            if self.fail_on == Some("list") {
                return Err("synthetic list failure");
            }
            Ok(self
                .segments
                .borrow()
                .iter()
                .filter(|metadata| metadata["operation_id"] == operation_id)
                .filter_map(|metadata| metadata.get("segment").cloned())
                .collect())
        }

        fn settle_once(
            &self,
            user_id: i64,
            operation_id: &str,
            actual_credit_units: i64,
            metadata: &Map<String, Value>,
        ) -> Result<bool, Self::Error> {
            if self.fail_on == Some("settle") {
                return Err("synthetic settle failure");
            }
            self.settlements.borrow_mut().push((
                user_id,
                operation_id.to_owned(),
                actual_credit_units,
                metadata.clone(),
            ));
            Ok(true)
        }
    }

    #[test]
    fn billing_uses_stable_personal_reservation_and_persists_segments_before_settlement() {
        let mut billing =
            PostgresTaskBilling::new(Store::default(), "deepseek/deepseek-v4.1-flash");
        let prompt = [TaskPromptMessage {
            role: "user",
            content: "synthetic prompt".to_owned(),
        }];
        assert_eq!(
            billing.reserve(&task("es"), "task123:1000", &prompt),
            Ok(TaskReserveOutcome::Authorized)
        );
        let charge = billing.store.charges.borrow()[0].clone();
        assert_eq!(charge.0, 42);
        assert!(charge.1 > 0);
        assert_eq!(charge.2, "task:task123:1000:reserve");
        assert_eq!(charge.3, "task:task123:1000");
        assert_eq!(charge.4["task_id"], "task123");

        let segment = json!({
            "kind": "chat",
            "model": "deepseek/deepseek-v4.1-flash",
            "usage": {
                "prompt_tokens": 10_000,
                "completion_tokens": 5_000,
                "cost": "0.001"
            },
            "source": "openrouter",
            "metadata": {
                "provider": "openrouter",
                "provider_generation_id": "generation-1"
            }
        });
        let settled = billing.settle(&task("es"), "task123:1000", &[segment], "task_success");
        assert_eq!(settled, Ok(()));
        assert_eq!(
            billing.store.segments.borrow()[0]["segment_id"],
            "openrouter:generation-1"
        );
        let settlement = &billing.store.settlements.borrow()[0];
        assert_eq!(settlement.0, 42);
        assert_eq!(settlement.1, "task:task123:1000");
        assert!(settlement.2 > 0);
        assert_eq!(settlement.3["reason"], "task_success");
        assert_eq!(settlement.3["pricing_complete"], true);
        assert_eq!(settlement.3["billing_segments"][0]["kind"], "chat");
    }

    #[test]
    fn task_settlement_includes_usage_persisted_by_an_earlier_attempt() {
        let earlier_segment = json!({
            "kind": "chat",
            "model": "deepseek/deepseek-v4.1-flash",
            "usage": {"cost": "0.001"},
            "source": "openrouter",
            "metadata": {
                "provider": "openrouter",
                "provider_generation_id": "generation-earlier"
            }
        });
        let current_segment = json!({
            "kind": "chat",
            "model": "deepseek/deepseek-v4.1-flash",
            "usage": {"cost": "0.002"},
            "source": "openrouter",
            "metadata": {
                "provider": "openrouter",
                "provider_generation_id": "generation-current"
            }
        });
        let store = Store {
            segments: RefCell::new(vec![json!({
                "operation_id": "task:task123:1000",
                "segment_id": "openrouter:generation-earlier",
                "segment": earlier_segment,
            })]),
            ..Store::default()
        };
        let mut billing = PostgresTaskBilling::new(store, "deepseek/deepseek-v4.1-flash");

        let settled = billing.settle(
            &task("en"),
            "task123:1000",
            &[current_segment],
            "task_success",
        );
        assert_eq!(settled, Ok(()));

        let settlements = billing.store.settlements.borrow();
        let settlement = &settlements[0];
        assert_eq!(
            settlement.3["billing_segments"].as_array().map(Vec::len),
            Some(2)
        );
        assert_eq!(settlement.2, 60);
    }

    #[test]
    fn task_refund_charges_usage_persisted_by_an_earlier_attempt() {
        let earlier_segment = json!({
            "kind": "chat",
            "model": "deepseek/deepseek-v4.1-flash",
            "usage": {"cost": "0.001"},
            "source": "openrouter",
            "metadata": {
                "provider": "openrouter",
                "provider_generation_id": "generation-earlier"
            }
        });
        let store = Store {
            segments: RefCell::new(vec![json!({
                "operation_id": "task:task123:1000",
                "segment_id": "openrouter:generation-earlier",
                "segment": earlier_segment,
            })]),
            ..Store::default()
        };
        let mut billing = PostgresTaskBilling::new(store, "deepseek/deepseek-v4.1-flash");

        let refunded = billing.refund(&task("en"), "task123:1000", "task_error");
        assert_eq!(refunded, Ok(()));

        let settlements = billing.store.settlements.borrow();
        let settlement = &settlements[0];
        assert_eq!(settlement.2, 20);
        assert_eq!(
            settlement.3["billing_segments"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(settlement.3["pricing_complete"], true);
    }

    #[test]
    fn task_keeps_reservation_open_until_provider_cost_is_reconciled() {
        let segment = json!({
            "kind": "chat",
            "model": "deepseek/deepseek-v4.1-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "source": "openrouter",
            "metadata": {
                "provider": "openrouter",
                "provider_generation_id": "generation-pending",
                "provider_usage_pending": true
            }
        });
        let mut billing =
            PostgresTaskBilling::new(Store::default(), "deepseek/deepseek-v4.1-flash");

        let settled = billing.settle(&task("en"), "task123:1000", &[segment], "task_success");
        assert_eq!(settled, Ok(()));

        assert_eq!(billing.store.segments.borrow().len(), 1);
        assert!(billing.store.settlements.borrow().is_empty());
    }

    #[test]
    fn task_reserve_adds_firecrawl_capacity_only_when_search_is_enabled() {
        let prompt = [TaskPromptMessage {
            role: "user",
            content: "synthetic prompt".to_owned(),
        }];
        let mut without_search =
            PostgresTaskBilling::new(Store::default(), "deepseek/deepseek-v4.1-flash");
        let reserved = without_search.reserve(&task("en"), "without-search", &prompt);
        assert_eq!(reserved, Ok(TaskReserveOutcome::Authorized));
        let without_search_amount = without_search.store.charges.borrow()[0].1;

        let mut with_search =
            PostgresTaskBilling::new(Store::default(), "deepseek/deepseek-v4.1-flash")
                .with_web_search(true);
        let reserved = with_search.reserve(&task("en"), "with-search", &prompt);
        assert_eq!(reserved, Ok(TaskReserveOutcome::Authorized));
        let with_search_amount = with_search.store.charges.borrow()[0].1;

        assert!(with_search_amount > without_search_amount);
    }

    #[test]
    fn billing_distinguishes_denial_from_a_durably_finished_execution() {
        let denied_store = Store {
            charge_result: RefCell::new(Some(AiChargeResult {
                ok: false,
                applied: false,
                reason: None,
                source: None,
                amount: 0,
                user_balance: 123,
                chat_balance: 0,
            })),
            ..Store::default()
        };
        let mut denied = PostgresTaskBilling::new(denied_store, "deepseek/deepseek-v4.1-flash");
        let outcome = denied.reserve(&task("en"), "task123:1000", &[]);
        let Ok(outcome) = outcome else { panic!() };
        assert!(
            matches!(outcome, TaskReserveOutcome::Denied { message } if message.contains("Balance: 1.23"))
        );

        let settled_store = Store {
            charge_result: RefCell::new(Some(AiChargeResult {
                ok: false,
                applied: false,
                reason: Some("operation_settled".to_owned()),
                source: None,
                amount: 0,
                user_balance: 123,
                chat_balance: 0,
            })),
            ..Store::default()
        };
        let mut settled = PostgresTaskBilling::new(settled_store, "deepseek/deepseek-v4.1-flash");
        assert_eq!(
            settled.reserve(&task("es"), "task123:1000", &[]),
            Ok(TaskReserveOutcome::AlreadySettled)
        );
    }

    #[test]
    fn refund_is_an_exactly_once_zero_cost_settlement() {
        let mut billing =
            PostgresTaskBilling::new(Store::default(), "deepseek/deepseek-v4.1-flash");
        let refunded = billing.refund(&task("es"), "task123:1000", "task_error");
        assert_eq!(refunded, Ok(()));
        let settlement = &billing.store.settlements.borrow()[0];
        assert_eq!(settlement.2, 0);
        assert_eq!(settlement.3["reason"], "task_error");
    }

    /// Records actions; the flag makes every delivery fail.
    #[derive(Default)]
    struct Sink(Vec<TelegramAction>, bool);

    impl ActionSink for Sink {
        type Error = &'static str;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            if self.1 {
                return Err("synthetic delivery failure");
            }
            self.0.push(action);
            Ok(ActionReceipt { message_id: None })
        }
    }

    #[test]
    fn task_messenger_uses_the_typed_telegram_action_boundary() {
        let mut messenger = ActionTaskMessenger::new(Sink::default());
        assert_eq!(messenger.send("-100123", "synthetic message"), Ok(()));
        assert!(matches!(
            &messenger.sink.0[0],
            TelegramAction::SendMessage(message)
                if message.chat_id.0 == -100123 && message.text == "synthetic message"
        ));
        assert!(messenger.send("not-a-chat", "ignored").is_err());
        assert_eq!(messenger.sink.0.len(), 1);
    }

    #[test]
    fn reservation_pricing_without_a_catalog_rejects_unknown_models() {
        assert_eq!(
            reservation_pricing_for_model("synthetic/unpriced", None),
            Err("OpenRouter pricing is unavailable for synthetic/unpriced".to_owned())
        );
    }

    #[test]
    fn spanish_task_tool_calls_are_validated_limited_and_answered_without_search() {
        let call = |id: &str, name: &str, arguments: Value| {
            json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments.to_string()}
            })
        };
        let transport = Transport {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from([
                HttpResponse {
                    status_code: 200,
                    body: json!({
                        "id": "generation-tools",
                        "model": "resolved/model",
                        "choices": [{
                            "message": {
                                "content": null,
                                "tool_calls": [
                                    call("call-invalid", "web_search", json!("not an object")),
                                    call("call-1", "web_search", json!({"query": "uno"})),
                                    call("call-2", "web_search", json!({"query": "dos"})),
                                    call("call-3", "web_search", json!({"query": "tres"})),
                                    call("call-other", "synthetic_tool", json!({})),
                                ]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {"cost": 0.0001}
                    })
                    .to_string(),
                    headers: BTreeMap::new(),
                },
                HttpResponse {
                    status_code: 200,
                    body: json!({
                        "id": "generation-final",
                        "model": "resolved/model",
                        "choices": [{
                            "message": {"content": "respuesta final"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"cost": 0.0002}
                    })
                    .to_string(),
                    headers: BTreeMap::new(),
                },
            ])),
        };
        let mut provider = OpenRouterTaskProvider::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "deepseek/deepseek-v4.1-flash",
            "synthetic persona",
        );

        let reply = provider.complete(
            &[TaskPromptMessage {
                role: "user",
                content: "buscá".to_owned(),
            }],
            &task("es"),
            "task123:1000",
        );
        let Ok(reply) = reply else { panic!() };

        assert_eq!(reply.text, "respuesta final");
        // Only the two model rounds are billable: no search ever ran.
        assert_eq!(reply.billing_segments.len(), 2);
        let requests = provider.transport.requests.borrow();
        assert_eq!(requests.len(), 2);
        let followup = serde_json::from_str::<Value>(&requests[1].body);
        let Ok(followup) = followup else { panic!() };
        assert!(
            followup["messages"][0]["content"]
                .as_str()
                .is_some_and(|content| content.contains("EJECUTANDO TAREA PROGRAMADA:"))
        );
        let assistant_calls = followup["messages"][2]["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| call["id"].as_str().unwrap_or_default().to_owned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        assert_eq!(
            assistant_calls,
            ["call-invalid", "call-1", "call-2", "call-3"]
        );
        let validation = validate_request(NativeTool::WebSearch, &json!({}), Locale::Es)
            .err()
            .unwrap_or_default();
        assert!(!validation.is_empty());
        let tool_results = (3..7)
            .map(|index| {
                (
                    followup["messages"][index]["tool_call_id"]
                        .as_str()
                        .unwrap_or_default(),
                    followup["messages"][index]["content"]
                        .as_str()
                        .unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tool_results,
            [
                ("call-invalid", validation.as_str()),
                ("call-1", "web_search is unavailable"),
                ("call-2", "web_search is unavailable"),
                ("call-3", "web_search usage limit reached"),
            ]
        );
        assert!(followup["messages"].get(7).is_none());
    }

    #[test]
    fn spanish_credit_denials_use_the_localized_top_up_message() {
        let store = Store {
            charge_result: RefCell::new(Some(AiChargeResult {
                ok: false,
                applied: false,
                reason: None,
                source: None,
                amount: 0,
                user_balance: 123,
                chat_balance: 0,
            })),
            ..Store::default()
        };
        let mut billing = PostgresTaskBilling::new(store, "deepseek/deepseek-v4.1-flash");
        assert_eq!(
            billing.reserve(&task("es"), "task123:1000", &[]),
            Ok(TaskReserveOutcome::Denied {
                message: "Te quedaste seco de créditos de IA, boludo. Saldo: 1.23\n\
                          Metele /topup si querés que siga laburando"
                    .to_owned()
            })
        );
    }

    #[test]
    fn task_messenger_reports_sink_delivery_failures() {
        let mut messenger = ActionTaskMessenger::new(Sink(Vec::new(), true));
        assert_eq!(
            messenger.send("-100123", "synthetic message"),
            Err("synthetic delivery failure".to_owned())
        );
        assert!(messenger.sink.0.is_empty());
    }

    #[test]
    fn task_billing_surfaces_store_and_pricing_failures() {
        let billing_for = |fail_on| {
            PostgresTaskBilling::new(
                Store {
                    fail_on: Some(fail_on),
                    ..Store::default()
                },
                "deepseek/deepseek-v4.1-flash",
            )
        };
        let segment = json!({
            "kind": "chat",
            "model": "deepseek/deepseek-v4.1-flash",
            "usage": {"cost": "0.001"},
            "source": "openrouter",
            "metadata": {"provider": "openrouter", "provider_generation_id": "generation-1"}
        });
        assert_eq!(
            billing_for("charge").reserve(&task("en"), "task123:1000", &[]),
            Err(NativeTaskBillingError::Store(
                "synthetic charge failure".to_owned()
            ))
        );
        assert_eq!(
            billing_for("record").settle(
                &task("en"),
                "task123:1000",
                std::slice::from_ref(&segment),
                "task_success"
            ),
            Err(NativeTaskBillingError::Store(
                "synthetic record failure".to_owned()
            ))
        );
        assert_eq!(
            billing_for("list").refund(&task("en"), "task123:1000", "task_error"),
            Err(NativeTaskBillingError::Store(
                "synthetic list failure".to_owned()
            ))
        );
        let mut settle_failure = billing_for("settle");
        assert_eq!(
            settle_failure.refund(&task("en"), "task123:1000", "task_error"),
            Err(NativeTaskBillingError::Store(
                "synthetic settle failure".to_owned()
            ))
        );
        assert!(settle_failure.store.settlements.borrow().is_empty());

        // A persisted segment that cannot be priced blocks the settlement.
        let unpriceable = Store {
            segments: RefCell::new(vec![json!({
                "operation_id": "task:task123:1000",
                "segment": "not a billing segment",
            })]),
            ..Store::default()
        };
        let mut billing = PostgresTaskBilling::new(unpriceable, "deepseek/deepseek-v4.1-flash");
        assert!(matches!(
            billing.refund(&task("en"), "task123:1000", "task_error"),
            Err(NativeTaskBillingError::Pricing(_))
        ));
        assert!(billing.store.settlements.borrow().is_empty());
    }

    type CatalogServer = std::thread::JoinHandle<std::io::Result<()>>;

    /// Serves one catalog response to the pricing cache's first refresh.
    fn catalog_pricing(
        catalog: Value,
    ) -> std::io::Result<(Arc<OpenRouterPricingCache>, CatalogServer)> {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 8_192];
            let _ = stream.read(&mut request);
            let body = catalog.to_string();
            stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
        });
        let cache = OpenRouterPricingCache::new("synthetic-key", &format!("http://{address}"));
        let Ok(cache) = cache else { panic!() };
        Ok((Arc::new(cache), server))
    }

    #[test]
    fn catalog_pricing_rejects_absent_models_and_unrepresentable_prices()
    -> Result<(), Box<dyn std::error::Error>> {
        let served = catalog_pricing(json!({
            "data": [{
                "id": super::PRIMARY_CHAT_MODEL,
                "pricing": {"prompt": "1e20", "completion": "1e20"}
            }]
        }));
        let (pricing, server) = served?;

        assert_eq!(
            reservation_pricing_for_model("synthetic/absent", Some(&pricing)),
            Err("OpenRouter catalog has no pricing for synthetic/absent".to_owned())
        );
        let estimate = super::estimate_task_reserve_credit_units("synthetic", "en", Some(&pricing));
        assert!(
            matches!(&estimate, Err(error) if !error.is_empty()),
            "{estimate:?}"
        );
        let mut billing = PostgresTaskBilling::new(Store::default(), super::PRIMARY_CHAT_MODEL)
            .with_openrouter_pricing(Arc::clone(&pricing));
        assert!(matches!(
            billing.reserve(&task("en"), "task123:1000", &[]),
            Err(NativeTaskBillingError::Estimate(_))
        ));
        assert!(billing.store.charges.borrow().is_empty());
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }
}
