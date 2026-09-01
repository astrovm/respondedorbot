//! Production Redis, PostgreSQL, and OpenRouter ports for memory compaction.

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::openrouter_chat::{
    ChatCompletionRequest, ChatMessage, ChatRole, OpenRouterTransport, ReqwestOpenRouterTransport,
    complete_with,
};
use bot_adapters::redis_compaction_queue::RedisCompactionQueue;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_message_state::RedisMessageState;
use bot_core::ai_pricing::calculate_billing_for_segments;
use bot_core::ai_reserve::{chat_output_token_limit, credit_units_from_usd_micros};
use bot_core::ai_usage::{ProviderSegmentIdentity, provider_segment_id};
use bot_core::command_state::CHAT_STATE_TTL_SECONDS;
use bot_core::credit_units::{CREDIT_SCALE, rescale_credit_units};
use bot_core::message_state::{chat_compacted_until_key, chat_summary_key};
use bot_core::text_cleanup::sanitize_summary_text;
use serde_json::{Map, Value, json};

use crate::compaction_worker::{
    CompactionBilling, CompactionProvider, CompactionProviderResult, CompactionState,
    CompactionWorker, SettlementRequest,
};

pub const COMPACTION_MODEL: &str = "deepseek/deepseek-v4-flash-0731";
const MAX_SUMMARY_MESSAGES: usize = 200;
const PRODUCTION_LOCK_TTL_SECONDS: i64 = 3_600;

pub type ProductionCompactionWorker = CompactionWorker<
    RedisCompactionQueue,
    RedisCompactionState,
    OpenRouterCompactionProvider<ReqwestOpenRouterTransport>,
    PostgresCompactionBilling,
    Box<dyn FnMut() -> String + Send>,
>;

pub fn production_compaction_worker(
    endpoint: &RedisEndpoint,
    database_url: &str,
    openrouter_api_key: &str,
    openrouter_base_url: &str,
    system_prompt: &str,
    owner_prefix: &str,
) -> Result<ProductionCompactionWorker, String> {
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let owner_prefix = owner_prefix.to_owned();
    let token: Box<dyn FnMut() -> String + Send> = Box::new(move || {
        let sequence = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("{owner_prefix}:compaction:{sequence}")
    });
    Ok(CompactionWorker::new(
        RedisCompactionQueue::new(endpoint).map_err(|error| error.to_string())?,
        RedisCompactionState::new(endpoint)?,
        OpenRouterCompactionProvider::new(
            ReqwestOpenRouterTransport::new().map_err(|error| error.to_string())?,
            openrouter_api_key,
            openrouter_base_url,
            COMPACTION_MODEL,
            system_prompt,
        ),
        PostgresCompactionBilling::new(database_url),
        token,
    )
    .with_lock_ttl_seconds(PRODUCTION_LOCK_TTL_SECONDS))
}

pub struct RedisCompactionState {
    state: RedisMessageState,
    ttl_seconds: i64,
}

impl RedisCompactionState {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, String> {
        RedisMessageState::new(endpoint)
            .map(|state| Self {
                state,
                ttl_seconds: CHAT_STATE_TTL_SECONDS,
            })
            .map_err(|error| error.to_string())
    }
}

impl CompactionState for RedisCompactionState {
    type Error = String;

    fn load(&mut self, chat_id: &str) -> Result<(Option<String>, Option<String>), Self::Error> {
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
        Ok((summary, marker))
    }

    fn save(
        &mut self,
        chat_id: &str,
        summary: &str,
        target_marker: &str,
    ) -> Result<(), Self::Error> {
        self.state
            .save_compaction_result(
                &chat_summary_key(chat_id),
                &chat_compacted_until_key(chat_id),
                summary,
                target_marker,
                self.ttl_seconds,
            )
            .map_err(|error| error.to_string())
    }
}

pub struct OpenRouterCompactionProvider<Transport> {
    transport: Transport,
    api_key: String,
    base_url: String,
    model: String,
    system_prompt: String,
}

impl<Transport> OpenRouterCompactionProvider<Transport> {
    #[must_use]
    pub fn new(
        transport: Transport,
        api_key: &str,
        base_url: &str,
        model: &str,
        system_prompt: &str,
    ) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            base_url: base_url.to_owned(),
            model: model.to_owned(),
            system_prompt: system_prompt.to_owned(),
        }
    }

    fn request(
        &self,
        messages: &[Value],
        prior_summary: Option<&str>,
        locale: &str,
    ) -> ChatCompletionRequest {
        let mut prompt = vec![ChatMessage::text(ChatRole::System, &self.system_prompt)];
        if let Some(prior_summary) = prior_summary.filter(|value| !value.is_empty()) {
            prompt.push(ChatMessage::text(ChatRole::Assistant, prior_summary));
        }
        prompt.extend(
            messages
                .iter()
                .rev()
                .take(MAX_SUMMARY_MESSAGES)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .filter_map(compaction_message),
        );
        prompt.push(ChatMessage::text(
            ChatRole::User,
            if locale.starts_with("es") {
                "actualizá el resumen previo con los mensajes nuevos. usá formato denso: temas, hechos clave, decisiones y pendientes. omití saludos y chat casual. mantené el idioma original. NUNCA uses markdown: no negritas, no headers, no tablas."
            } else {
                "update the previous summary with the new messages. use a dense format with topics, key facts, decisions, and pending work. omit greetings and casual chat. preserve the original language. NEVER use markdown: no bold, headings, or tables."
            },
        ));
        let mut request = ChatCompletionRequest::new(&self.model, prompt);
        request.max_tokens = u64::try_from(chat_output_token_limit(&self.model)).ok();
        request
    }
}

impl<Transport: OpenRouterTransport> CompactionProvider
    for OpenRouterCompactionProvider<Transport>
{
    type Error = String;

    fn compact(
        &mut self,
        messages: &[Value],
        prior_summary: Option<&str>,
        locale: &str,
    ) -> Result<CompactionProviderResult, Self::Error> {
        let completion = complete_with(
            &self.transport,
            &self.api_key,
            &self.base_url,
            &self.request(messages, prior_summary, locale),
        )
        .map_err(|error| error.to_string())?;
        let cleaned = sanitize_summary_text(&completion.text);
        if cleaned.is_empty() {
            return Err("summary provider returned empty text".to_owned());
        }
        let summary = if locale.starts_with("es") {
            format!("[contexto anterior: {cleaned}]")
        } else {
            format!("[previous context: {cleaned}]")
        };
        let mut metadata = Map::from_iter([("provider".to_owned(), json!("openrouter"))]);
        if completion.model != self.model {
            metadata.insert("requested_model".to_owned(), json!(&self.model));
        }
        insert_optional(
            &mut metadata,
            "provider_generation_id",
            completion.generation_id,
        );
        insert_optional(
            &mut metadata,
            "upstream_provider",
            completion.upstream_provider,
        );
        insert_optional(&mut metadata, "service_tier", completion.service_tier);
        metadata.insert("compaction_result".to_owned(), json!(&summary));
        let segment = json!({
            "kind": "summary",
            "text": completion.text,
            "model": completion.model,
            "usage": completion.usage,
            "source": "openrouter",
            "metadata": metadata,
        });
        let cost_usd_micros = calculate_billing_for_segments(&json!([segment.clone()]))
            .ok()
            .and_then(|billing| {
                billing
                    .get("raw_usd_micros_exact")
                    .and_then(Value::as_str)
                    .map(ceil_decimal)
            })
            .unwrap_or_default();
        Ok(CompactionProviderResult {
            summary,
            cost_usd_micros,
            billing_segment: Some(segment),
        })
    }
}

pub struct PostgresCompactionBilling {
    repository: BillingRepository,
}

impl PostgresCompactionBilling {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            repository: BillingRepository::new(database_url),
        }
    }
}

impl CompactionBilling for PostgresCompactionBilling {
    type Error = String;

    fn list_provider_segments(
        &mut self,
        user_id: i64,
        operation_id: &str,
    ) -> Result<Vec<Value>, Self::Error> {
        self.repository
            .list_ai_provider_segments(user_id, operation_id)
            .map_err(|error| error.to_string())
    }

    fn record_provider_segment(
        &mut self,
        job: &bot_adapters::compaction_job::CompactionJobRecord,
        operation_id: &str,
        segment: &Value,
    ) -> Result<(), Self::Error> {
        let metadata = json!({
            "operation_id": operation_id,
            "segment_id": stable_segment_id(segment),
            "segment": segment,
        });
        self.repository
            .record_ai_provider_usage(
                job.user_id,
                reservation_chat_id(&job.reservation),
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn settle(&mut self, request: SettlementRequest<'_>) -> Result<(), Self::Error> {
        let reservation = &request.job.reservation;
        let reserved = reservation_i64(reservation, "reserved_credit_units")
            .and_then(|units| {
                rescale_credit_units(units, reservation_credit_scale(reservation))
                    .ok()
                    .map(|units| units.value())
            })
            .unwrap_or_default();
        let pricing = (!request.billing_segments.is_empty())
            .then(|| {
                calculate_billing_for_segments(&Value::Array(request.billing_segments.to_vec()))
            })
            .transpose()
            .map_err(|error| error.to_string())?;
        let pricing_complete = pricing
            .as_ref()
            .is_none_or(|pricing| pricing["pricing_complete"] == json!(true));
        let billed = pricing
            .as_ref()
            .and_then(|pricing| pricing.get("charged_credit_units"))
            .and_then(Value::as_i64)
            .unwrap_or_else(|| {
                credit_units_from_usd_micros(i128::from(request.job.result_cost_usd_micros))
                    .unwrap_or_default()
            });
        let actual = request.actual_credit_units.map_or_else(
            || {
                if pricing_complete {
                    billed
                } else {
                    reserved.max(billed)
                }
            },
            |value| value.max(0),
        );
        let operation_id = reservation_nested_string(reservation, "operation_id");
        let settlement_id = reservation_nested_string(reservation, "settlement_id");
        let mut metadata = Map::from_iter([
            ("reason".to_owned(), json!(request.reason)),
            ("message_id".to_owned(), json!(request.job.message_id)),
            ("settlement_id".to_owned(), json!(settlement_id)),
            ("operation_id".to_owned(), json!(operation_id)),
            ("credit_scale".to_owned(), json!(CREDIT_SCALE)),
            (
                "billing_segments".to_owned(),
                json!(request.billing_segments),
            ),
            ("pricing_complete".to_owned(), json!(pricing_complete)),
        ]);
        if let Some(pricing) = pricing.and_then(|value| value.as_object().cloned()) {
            for key in [
                "pricing_version",
                "raw_usd_micros",
                "markup_multiplier",
                "model_breakdown",
                "tool_breakdown",
                "segment_breakdown",
            ] {
                if let Some(value) = pricing.get(key) {
                    metadata.insert(key.to_owned(), value.clone());
                }
            }
        }
        self.repository
            .settle_legacy_ai_reservation_once(
                request.job.user_id,
                reservation_chat_id(reservation),
                reservation
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or("user"),
                reserved,
                actual,
                reservation
                    .get("usage_tag")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn settle_incompatible(&mut self, chat_id: &str, decoded: &Value) -> Result<bool, Self::Error> {
        let Some(user_id) = decoded.get("user_id").and_then(Value::as_i64) else {
            return Ok(false);
        };
        let Some(reservation) = decoded.get("reservation").filter(|value| value.is_object()) else {
            return Ok(false);
        };
        let usage_tag = reservation
            .get("usage_tag")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if usage_tag.is_empty() {
            return Ok(false);
        }
        let reserved = reservation_i64(reservation, "reserved_credit_units")
            .and_then(|units| {
                rescale_credit_units(units, reservation_credit_scale(reservation))
                    .ok()
                    .map(|units| units.value())
            })
            .unwrap_or_default();
        let metadata = Map::from_iter([
            (
                "reason".to_owned(),
                json!("memory_compaction_incompatible_job"),
            ),
            ("chat_id".to_owned(), json!(chat_id)),
            (
                "settlement_id".to_owned(),
                json!(reservation_nested_string(reservation, "settlement_id")),
            ),
            (
                "operation_id".to_owned(),
                json!(reservation_nested_string(reservation, "operation_id")),
            ),
            ("credit_scale".to_owned(), json!(CREDIT_SCALE)),
        ]);
        self.repository
            .settle_legacy_ai_reservation_once(
                user_id,
                reservation_chat_id(reservation),
                reservation
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or("user"),
                reserved,
                0,
                usage_tag,
                &metadata,
            )
            .map(|_| true)
            .map_err(|error| error.to_string())
    }
}

fn compaction_message(message: &Value) -> Option<ChatMessage> {
    let content = message
        .get("content")
        .or_else(|| message.get("text"))
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())?;
    let role = match message.get("role").and_then(Value::as_str) {
        Some("assistant") => ChatRole::Assistant,
        Some("system") => ChatRole::System,
        Some("tool") => ChatRole::Tool,
        _ => ChatRole::User,
    };
    Some(ChatMessage::text(role, content))
}

fn insert_optional(metadata: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        metadata.insert(key.to_owned(), json!(value));
    }
}

fn ceil_decimal(value: &str) -> i64 {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    whole
        .parse::<i64>()
        .ok()
        .and_then(|whole| whole.checked_add(i64::from(fraction.bytes().any(|byte| byte != b'0'))))
        .unwrap_or_default()
}

fn reservation_i64(reservation: &Value, key: &str) -> Option<i64> {
    reservation
        .get(key)
        .and_then(|value| value.as_i64().or_else(|| value.as_str()?.parse().ok()))
}

fn reservation_credit_scale(reservation: &Value) -> Option<i64> {
    reservation_i64(reservation, "credit_scale").or_else(|| {
        reservation
            .get("metadata")
            .and_then(|metadata| reservation_i64(metadata, "credit_scale"))
    })
}

fn reservation_chat_id(reservation: &Value) -> Option<i64> {
    reservation_i64(reservation, "chat_scope_id")
}

fn reservation_nested_string(reservation: &Value, key: &str) -> String {
    reservation
        .get(key)
        .and_then(Value::as_str)
        .or_else(|| reservation.get("metadata")?.get(key)?.as_str())
        .unwrap_or_default()
        .to_owned()
}

fn stable_segment_id(segment: &Value) -> String {
    let metadata = segment.get("metadata").and_then(Value::as_object);
    let source = segment
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("provider");
    let kind = segment
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model = segment
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let canonical = serde_json::to_string(segment).unwrap_or_default();
    provider_segment_id(
        &ProviderSegmentIdentity {
            source_with_default: source,
            source_or_provider: source,
            kind_or_unknown: kind,
            model_or_unknown: model,
            provider_generation_id: metadata
                .and_then(|value| value.get("provider_generation_id"))
                .and_then(Value::as_str),
            provider_request_id: metadata
                .and_then(|value| value.get("provider_request_id"))
                .and_then(Value::as_str),
            tool_rounds: metadata
                .and_then(|value| value.get("tool_rounds"))
                .and_then(Value::as_str),
        },
        &canonical,
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::error::Error;

    use bot_adapters::openrouter_chat::{
        HttpRequest, HttpResponse, OpenRouterChatError, OpenRouterTransport,
    };
    use bot_adapters::redis_connection::RedisEndpoint;
    use serde_json::{Value, json};

    use super::{
        OpenRouterCompactionProvider, ceil_decimal, production_compaction_worker, stable_segment_id,
    };
    use crate::compaction_worker::CompactionProvider;

    struct Transport {
        request: RefCell<Option<HttpRequest>>,
    }

    #[test]
    fn production_worker_composition_is_side_effect_free() {
        let result = production_compaction_worker(
            &RedisEndpoint {
                host: "synthetic.invalid".to_owned(),
                port: 6379,
                password: Some("synthetic-password".to_owned()),
            },
            "postgresql://synthetic.invalid/database",
            "synthetic-openrouter-key",
            "https://openrouter.example.test/api/v1",
            "synthetic persona",
            "synthetic-owner",
        );
        assert!(result.is_ok());
    }

    impl OpenRouterTransport for Transport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.request.replace(Some(request.clone()));
            Ok(HttpResponse {
                status_code: 200,
                body: json!({
                    "id":"generation-1",
                    "model":"resolved/model",
                    "provider":"Synthetic",
                    "choices":[{"message":{"content":"**fact**\n\n# pending"},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":10,"completion_tokens":5,"cost":"0.0012345"}
                }).to_string(),
                headers: Default::default(),
            })
        }
    }

    #[test]
    fn builds_legacy_compatible_summary_request_and_billing_segment() -> Result<(), Box<dyn Error>>
    {
        let transport = Transport {
            request: RefCell::new(None),
        };
        let mut provider = OpenRouterCompactionProvider::new(
            transport,
            "key",
            "https://synthetic.invalid/api/v1",
            "requested/model",
            "persona",
        );
        let result = provider.compact(
            &[json!({"role":"user","text":"hello"})],
            Some("old context"),
            "en",
        )?;
        assert_eq!(result.summary, "[previous context: fact\n\npending]");
        let segment = result.billing_segment.unwrap_or(Value::Null);
        assert_eq!(segment["kind"], "summary");
        assert_eq!(segment["text"], "**fact**\n\n# pending");
        assert_eq!(segment["metadata"]["compaction_result"], result.summary);
        assert_eq!(
            segment["metadata"]["provider_generation_id"],
            "generation-1"
        );
        assert_eq!(stable_segment_id(&segment), "openrouter:generation-1");
        let body: Value = serde_json::from_str(
            &provider
                .transport
                .request
                .borrow()
                .as_ref()
                .map(|request| request.body.clone())
                .unwrap_or_default(),
        )?;
        assert_eq!(body["messages"][0]["content"], "persona");
        assert_eq!(body["messages"][1]["content"], "old context");
        assert_eq!(body["messages"][2]["content"], "hello");
        assert!(
            body["messages"][3]["content"]
                .as_str()
                .unwrap_or_default()
                .contains("NEVER use markdown")
        );
        Ok(())
    }

    #[test]
    fn decimal_ceiling_matches_python_cost_checkpoint() {
        assert_eq!(ceil_decimal("1234.00000000"), 1234);
        assert_eq!(ceil_decimal("1234.00000001"), 1235);
        assert_eq!(ceil_decimal("0"), 0);
    }
}
