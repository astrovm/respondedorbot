//! Concrete native AI and billing adapters for scheduled task execution.

use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::billing_read::{AiChargeResult, BillingRepository};
use bot_adapters::openrouter_chat::{
    ChatCompletionRequest, ChatMessage, ChatRole, OpenRouterChatError, OpenRouterTransport,
    complete_with,
};
use bot_core::ai_pricing::calculate_billing_for_segments;
use bot_core::ai_prompt::build_system_prompt;
use bot_core::ai_reserve::{
    EstimatedMessage, TokenEstimateValue, chat_output_token_limit,
    estimate_chat_reserve_credit_units,
};
use bot_core::credit_units::{CreditUnits, format_credit_units};
use bot_core::locale::Locale;
use bot_core::provider_config::web_search_tool;
use bot_core::scheduled_tasks::ScheduledTask;
use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_input::ChatId;
use chrono::{DateTime, FixedOffset, Utc};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::dispatcher::ActionSink;
use crate::task_executor::{
    TaskAiProvider, TaskBilling, TaskMessenger, TaskPromptMessage, TaskProviderReply,
    TaskReserveOutcome, build_task_messages,
};

pub const PRIMARY_CHAT_MODEL: &str = "deepseek/deepseek-v4-flash-0731";
const SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE: i64 = 4_000;
const WEB_SEARCH_MAX_RESULTS: i64 = 5;
const WEB_SEARCH_MAX_USES: i64 = 3;

pub fn estimate_task_reserve_credit_units(
    text: &str,
    locale: &str,
) -> Result<i64, bot_core::ai_reserve::ReserveEstimateError> {
    let estimated_messages = build_task_messages(text, locale)
        .into_iter()
        .map(|message| EstimatedMessage {
            role: TokenEstimateValue::Text(message.role.to_owned()),
            content: TokenEstimateValue::Text(message.content),
            name: TokenEstimateValue::Empty,
        })
        .collect::<Vec<_>>();
    estimate_chat_reserve_credit_units(
        None,
        &estimated_messages,
        Some(chat_output_token_limit(PRIMARY_CHAT_MODEL)),
        SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE,
        PRIMARY_CHAT_MODEL,
    )
}

pub struct OpenRouterTaskProvider<Transport> {
    transport: Transport,
    api_key: String,
    base_url: String,
    model: String,
    persona: String,
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
        }
    }

    fn request(
        &self,
        messages: &[TaskPromptMessage],
        task: &ScheduledTask,
    ) -> ChatCompletionRequest {
        let mut request_messages = Vec::with_capacity(messages.len() + 1);
        request_messages.push(ChatMessage::text(
            ChatRole::System,
            task_system_prompt(&self.persona, task),
        ));
        request_messages.extend(messages.iter().map(|message| {
            ChatMessage::text(task_message_role(message.role), message.content.clone())
        }));
        let mut request = ChatCompletionRequest::new(&self.model, request_messages);
        request.max_tokens = u64::try_from(chat_output_token_limit(&self.model)).ok();
        request.tools = vec![
            serde_json::to_value(web_search_tool(WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_MAX_USES))
                .unwrap_or_else(|_| json!({})),
        ];
        request
    }
}

impl<Transport: OpenRouterTransport> TaskAiProvider for OpenRouterTaskProvider<Transport> {
    type Error = OpenRouterChatError;

    fn complete(
        &mut self,
        messages: &[TaskPromptMessage],
        task: &ScheduledTask,
        _execution_id: &str,
    ) -> Result<TaskProviderReply, Self::Error> {
        let completion = complete_with(
            &self.transport,
            &self.api_key,
            &self.base_url,
            &self.request(messages, task),
        )?;
        let annotation_types = completion
            .annotations
            .iter()
            .filter_map(|annotation| annotation.get("type").and_then(Value::as_str))
            .collect::<Vec<_>>();
        let mut metadata = Map::from_iter([("provider".to_owned(), json!("openrouter"))]);
        optional_string(
            &mut metadata,
            "upstream_provider",
            completion.upstream_provider,
        );
        optional_string(&mut metadata, "service_tier", completion.service_tier);
        optional_string(
            &mut metadata,
            "provider_generation_id",
            completion.generation_id,
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
        Ok(TaskProviderReply {
            text: completion.text,
            fallback: false,
            billing_segments: vec![json!({
                "kind": "chat",
                "model": completion.model,
                "usage": completion.usage,
                "source": "openrouter",
                "metadata": metadata,
            })],
        })
    }
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
    build_system_prompt(
        persona,
        if task.locale == "en" {
            Locale::En
        } else {
            Locale::Es
        },
        &formatted_date(i64::from(task.timezone_offset)),
        true,
        true,
    )
}

fn formatted_date(timezone_offset_hours: i64) -> String {
    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64);
    let utc =
        DateTime::<Utc>::from_timestamp(unix_seconds, 0).unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
    let seconds = timezone_offset_hours.clamp(-23, 23).saturating_mul(3_600) as i32;
    FixedOffset::east_opt(seconds).map_or_else(
        || utc.format("%A %d/%m/%Y").to_string(),
        |offset| utc.with_timezone(&offset).format("%A %d/%m/%Y").to_string(),
    )
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

pub struct PostgresTaskBilling<Store> {
    store: Store,
    model: String,
}

impl<Store> PostgresTaskBilling<Store> {
    #[must_use]
    pub fn new(store: Store, model: &str) -> Self {
        Self {
            store,
            model: model.to_owned(),
        }
    }

    fn settle_amount(
        &self,
        task: &ScheduledTask,
        execution_id: &str,
        segments: &[Value],
        reason: &'static str,
        amount: i64,
    ) -> Result<(), NativeTaskBillingError>
    where
        Store: TaskCreditStore,
    {
        let user_id = task.user_id.ok_or(NativeTaskBillingError::MissingUser)?;
        let operation_id = operation_id(execution_id);
        for (index, segment) in segments.iter().enumerate() {
            let metadata = json!({
                "operation_id": operation_id,
                "segment_id": format!("{execution_id}:{index}"),
                "task_id": task.id.as_str(),
                "execution_id": execution_id,
                "segment": segment,
            });
            self.store
                .record_segment(user_id, &metadata)
                .map_err(|error| NativeTaskBillingError::Store(error.to_string()))?;
        }
        let mut settlement = Map::new();
        settlement.insert("reason".to_owned(), json!(reason));
        settlement.insert("task_id".to_owned(), json!(task.id.as_str()));
        settlement.insert("execution_id".to_owned(), json!(execution_id));
        settlement.insert("operation_id".to_owned(), json!(operation_id));
        self.store
            .settle_once(user_id, &operation_id, amount, &settlement)
            .map_err(|error| NativeTaskBillingError::Store(error.to_string()))?;
        Ok(())
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
        let amount = estimate_chat_reserve_credit_units(
            None,
            &estimated_messages,
            Some(chat_output_token_limit(&self.model)),
            SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE,
            &self.model,
        )
        .map_err(|error| NativeTaskBillingError::Estimate(error.to_string()))?;
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
            .map_err(|error| NativeTaskBillingError::Store(error.to_string()))?;
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
        let pricing = calculate_billing_for_segments(&json!(segments))
            .map_err(|error| NativeTaskBillingError::Pricing(error.to_string()))?;
        let amount = pricing
            .get("charged_credit_units")
            .and_then(Value::as_i64)
            .ok_or_else(|| NativeTaskBillingError::Pricing("missing charge total".to_owned()))?;
        self.settle_amount(task, execution_id, segments, reason, amount)
    }

    fn refund(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        reason: &'static str,
    ) -> Result<(), Self::Error> {
        self.settle_amount(task, execution_id, &[], reason, 0)
    }
}

fn operation_id(execution_id: &str) -> String {
    format!("task:{execution_id}")
}

fn insufficient_credits_message(locale: &str, balance: i64) -> String {
    let balance = format_credit_units(CreditUnits::new(balance));
    if locale == "en" {
        format!("you are out of AI credits\nbalance: {balance}\nuse /topup to add more")
    } else {
        format!(
            "te quedaste seco de créditos ia, boludo.\nsaldo: {balance}\nmetele /topup si querés que siga laburando"
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
    use std::collections::BTreeMap;

    use bot_adapters::billing_read::AiChargeResult;
    use bot_adapters::openrouter_chat::{
        HttpRequest, HttpResponse, OpenRouterChatError, OpenRouterTransport,
    };
    use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};
    use bot_core::telegram_actions::TelegramAction;
    use serde_json::{Map, Value, json};

    use super::{
        ActionTaskMessenger, OpenRouterTaskProvider, PostgresTaskBilling, TaskAiProvider,
        TaskBilling, TaskCreditStore, TaskMessenger, TaskPromptMessage, TaskReserveOutcome,
    };
    use crate::dispatcher::{ActionReceipt, ActionSink};

    struct Transport {
        requests: RefCell<Vec<HttpRequest>>,
        response: RefCell<Option<HttpResponse>>,
    }

    impl OpenRouterTransport for Transport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .ok_or_else(|| OpenRouterChatError::Transport("missing response".to_owned()))
        }
    }

    fn task(locale: &str) -> ScheduledTask {
        ScheduledTask {
            id: TaskId::new("task123").unwrap_or_else(|error| panic!("task id: {error}")),
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
            response: RefCell::new(Some(HttpResponse {
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
            })),
        };
        let mut provider = OpenRouterTaskProvider::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "deepseek/deepseek-v4-flash-0731",
            "synthetic persona",
        );
        let reply = provider
            .complete(
                &[TaskPromptMessage {
                    role: "user",
                    content: "do it".to_owned(),
                }],
                &task("en"),
                "task123:1000",
            )
            .unwrap_or_else(|error| panic!("provider response: {error}"));
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
        let body: Value = serde_json::from_str(&request.body)
            .unwrap_or_else(|error| panic!("request json: {error}"));
        assert_eq!(body["max_tokens"], 8_192);
        assert_eq!(body["tools"][0]["type"], "openrouter:web_search");
        assert_eq!(body["tools"][0]["parameters"]["max_uses"], 3);
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

    type CapturedCharge = (i64, i32, String, String, Map<String, Value>);
    type CapturedSettlement = (i64, String, i64, Map<String, Value>);

    #[derive(Default)]
    struct Store {
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
            self.segments.borrow_mut().push(metadata.clone());
            Ok(true)
        }

        fn settle_once(
            &self,
            user_id: i64,
            operation_id: &str,
            actual_credit_units: i64,
            metadata: &Map<String, Value>,
        ) -> Result<bool, Self::Error> {
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
            PostgresTaskBilling::new(Store::default(), "deepseek/deepseek-v4-flash-0731");
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
            "model": "deepseek/deepseek-v4-flash-0731",
            "usage": {
                "prompt_tokens": 10_000,
                "completion_tokens": 5_000,
                "cost": "0.001"
            },
            "source": "openrouter",
            "metadata": {"provider": "openrouter"}
        });
        billing
            .settle(&task("es"), "task123:1000", &[segment], "task_success")
            .unwrap_or_else(|error| panic!("settlement: {error}"));
        assert_eq!(
            billing.store.segments.borrow()[0]["segment_id"],
            "task123:1000:0"
        );
        let settlement = &billing.store.settlements.borrow()[0];
        assert_eq!(settlement.0, 42);
        assert_eq!(settlement.1, "task:task123:1000");
        assert!(settlement.2 > 0);
        assert_eq!(settlement.3["reason"], "task_success");
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
        let mut denied = PostgresTaskBilling::new(denied_store, "deepseek/deepseek-v4-flash-0731");
        let outcome = denied
            .reserve(&task("en"), "task123:1000", &[])
            .unwrap_or_else(|error| panic!("reserve: {error}"));
        assert!(
            matches!(outcome, TaskReserveOutcome::Denied { message } if message.contains("balance: 1.23"))
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
        let mut settled =
            PostgresTaskBilling::new(settled_store, "deepseek/deepseek-v4-flash-0731");
        assert_eq!(
            settled.reserve(&task("es"), "task123:1000", &[]),
            Ok(TaskReserveOutcome::AlreadySettled)
        );
    }

    #[test]
    fn refund_is_an_exactly_once_zero_cost_settlement() {
        let mut billing =
            PostgresTaskBilling::new(Store::default(), "deepseek/deepseek-v4-flash-0731");
        billing
            .refund(&task("es"), "task123:1000", "task_error")
            .unwrap_or_else(|error| panic!("refund: {error}"));
        let settlement = &billing.store.settlements.borrow()[0];
        assert_eq!(settlement.2, 0);
        assert_eq!(settlement.3["reason"], "task_error");
    }

    #[derive(Default)]
    struct Sink(Vec<TelegramAction>);

    impl ActionSink for Sink {
        type Error = &'static str;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
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
}
