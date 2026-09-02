//! Foreground planning handoff for durable background memory compaction.

use std::fmt::Display;

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::compaction_job::{COMPACTION_JOB_SCHEMA_VERSION, CompactionJobRecord};
use bot_adapters::redis_compaction_queue::RedisCompactionQueue;
use bot_core::ai_reserve::{
    EstimatedMessage, TokenEstimateValue, chat_output_token_limit,
    estimate_chat_reserve_credit_units,
};
use bot_core::credit_units::CREDIT_SCALE;
use bot_core::locale::Locale;
use serde_json::{Map, Value, json};

use crate::compaction_adapters::COMPACTION_MODEL;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryCompactionPlan {
    pub chat_id: String,
    pub messages: Vec<Value>,
    pub prior_summary: Option<String>,
    pub expected_marker: Option<String>,
    pub target_marker: String,
}

#[derive(Debug, Clone, Copy)]
pub struct CompactionScheduleContext {
    pub user_id: i64,
    pub group_chat_id: Option<i64>,
    pub origin_chat_id: i64,
    pub message_id: i64,
    pub locale: Locale,
    pub payer_source: Option<PayerSource>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayerSource {
    User,
    Chat,
}

impl PayerSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Chat => "chat",
        }
    }
}

pub trait MemoryCompactionScheduler {
    fn schedule(
        &mut self,
        plan: MemoryCompactionPlan,
        context: CompactionScheduleContext,
    ) -> Result<bool, String>;
}

pub trait CompactionEnqueueStore {
    type Error: Display;

    fn job_exists(&mut self, chat_id: &str) -> Result<bool, Self::Error>;
    fn insert_job(&mut self, chat_id: &str, payload: &str) -> Result<bool, Self::Error>;
}

impl CompactionEnqueueStore for RedisCompactionQueue {
    type Error = bot_adapters::redis_compaction_queue::RedisCompactionQueueError;

    fn job_exists(&mut self, chat_id: &str) -> Result<bool, Self::Error> {
        RedisCompactionQueue::job_exists(self, chat_id)
    }

    fn insert_job(&mut self, chat_id: &str, payload: &str) -> Result<bool, Self::Error> {
        RedisCompactionQueue::insert_job(self, chat_id, payload)
    }
}

pub trait CompactionReservationStore {
    type Error: Display;

    fn reserve(
        &mut self,
        context: CompactionScheduleContext,
        usage_tag: &str,
        reserve_credit_units: i64,
        target_marker: &str,
        message_count: usize,
    ) -> Result<Option<Value>, Self::Error>;

    fn refund_enqueue_failure(
        &mut self,
        user_id: i64,
        reservation: &Value,
    ) -> Result<(), Self::Error>;
}

pub struct NativeCompactionScheduler<Queue, Billing, Token> {
    queue: Queue,
    billing: Billing,
    token: Token,
    model: String,
    system_prompt: String,
}

impl<Queue, Billing, Token> NativeCompactionScheduler<Queue, Billing, Token> {
    #[must_use]
    pub fn new(
        queue: Queue,
        billing: Billing,
        token: Token,
        model: &str,
        system_prompt: &str,
    ) -> Self {
        Self {
            queue,
            billing,
            token,
            model: model.to_owned(),
            system_prompt: system_prompt.to_owned(),
        }
    }

    pub fn into_parts(self) -> (Queue, Billing, Token) {
        (self.queue, self.billing, self.token)
    }

    fn estimate(&self, plan: &MemoryCompactionPlan) -> Result<i64, String> {
        let system = estimated_message("system", &self.system_prompt);
        let mut messages = Vec::new();
        if let Some(prior) = plan
            .prior_summary
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            messages.push(estimated_message("assistant", prior));
        }
        messages.extend(plan.messages.iter().filter_map(estimated_stored_message));
        messages.push(estimated_message(
            "user",
            "update the previous summary with the new messages",
        ));
        estimate_chat_reserve_credit_units(
            Some(&system),
            &messages,
            Some(chat_output_token_limit(&self.model)),
            0,
            &self.model,
        )
        .map(|units| units.max(1))
        .map_err(|error| error.to_string())
    }
}

impl<Queue, Billing, Token> MemoryCompactionScheduler
    for NativeCompactionScheduler<Queue, Billing, Token>
where
    Queue: CompactionEnqueueStore,
    Billing: CompactionReservationStore,
    Token: FnMut() -> String,
{
    fn schedule(
        &mut self,
        plan: MemoryCompactionPlan,
        context: CompactionScheduleContext,
    ) -> Result<bool, String> {
        if self
            .queue
            .job_exists(&plan.chat_id)
            .map_err(|error| error.to_string())?
        {
            return Ok(false);
        }
        let reserve_credit_units = self.estimate(&plan)?;
        let usage_tag = format!(
            "memory_compaction:{}:{}:{}",
            plan.chat_id,
            plan.target_marker,
            (self.token)()
        );
        let Some(mut reservation) = self
            .billing
            .reserve(
                context,
                &usage_tag,
                reserve_credit_units,
                &plan.target_marker,
                plan.messages.len(),
            )
            .map_err(|error| error.to_string())?
        else {
            return Ok(false);
        };
        if let Some(reservation) = reservation.as_object_mut() {
            reservation.insert("credit_scale".to_owned(), json!(CREDIT_SCALE));
        }
        let job = CompactionJobRecord {
            schema_version: COMPACTION_JOB_SCHEMA_VERSION,
            chat_id: plan.chat_id.clone(),
            messages: plan.messages,
            prior_summary: plan.prior_summary,
            expected_marker: plan.expected_marker,
            target_marker: plan.target_marker,
            reservation,
            user_id: context.user_id,
            message_id: Some(context.message_id.to_string()),
            locale: match context.locale {
                Locale::Es => "es",
                Locale::En => "en",
            }
            .to_owned(),
            attempts: 0,
            next_attempt_at: 0.0,
            result_summary: None,
            result_cost_usd_micros: 0,
            result_billing_segment: None,
        };
        let payload = serde_json::to_string(&job).map_err(|error| error.to_string())?;
        let stored = match self.queue.insert_job(&plan.chat_id, &payload) {
            Ok(stored) => stored,
            Err(error) => {
                self.billing
                    .refund_enqueue_failure(context.user_id, &job.reservation)
                    .map_err(|refund_error| {
                        format!("{error}; compaction reservation refund failed: {refund_error}")
                    })?;
                return Err(error.to_string());
            }
        };
        if !stored {
            self.billing
                .refund_enqueue_failure(context.user_id, &job.reservation)
                .map_err(|error| error.to_string())?;
        }
        Ok(stored)
    }
}

pub struct PostgresCompactionReservations {
    repository: BillingRepository,
}

impl PostgresCompactionReservations {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            repository: BillingRepository::new(database_url),
        }
    }
}

impl CompactionReservationStore for PostgresCompactionReservations {
    type Error = String;

    fn reserve(
        &mut self,
        context: CompactionScheduleContext,
        usage_tag: &str,
        reserve_credit_units: i64,
        target_marker: &str,
        message_count: usize,
    ) -> Result<Option<Value>, Self::Error> {
        let settlement_id = format!(
            "{}:{}:{}:{usage_tag}",
            context.user_id, context.origin_chat_id, context.message_id
        );
        let operation_id = settlement_id.clone();
        let metadata = Map::from_iter([
            ("usage_tag".to_owned(), json!(usage_tag)),
            ("settlement_id".to_owned(), json!(&settlement_id)),
            ("idempotency_key".to_owned(), json!(&settlement_id)),
            ("operation_id".to_owned(), json!(&operation_id)),
            ("message_id".to_owned(), json!(context.message_id)),
            ("origin_chat_id".to_owned(), json!(context.origin_chat_id)),
            ("credit_scale".to_owned(), json!(CREDIT_SCALE)),
            (
                "reserved_credit_units".to_owned(),
                json!(reserve_credit_units),
            ),
            ("target_marker".to_owned(), json!(target_marker)),
            ("message_count".to_owned(), json!(message_count)),
            ("background".to_owned(), json!(true)),
        ]);
        let amount = i32::try_from(reserve_credit_units)
            .map_err(|_| "compaction reservation exceeds the database range".to_owned())?;
        let result = self
            .repository
            .charge_ai_credits(
                context.user_id,
                context.group_chat_id,
                amount,
                "ai_reserve",
                &metadata,
                context.payer_source.map(PayerSource::as_str),
                Some(&settlement_id),
                &operation_id,
            )
            .map_err(|error| error.to_string())?;
        if !result.ok {
            return Ok(None);
        }
        Ok(Some(json!({
            "reserved_credit_units": result.amount,
            "chat_scope_id": context.group_chat_id,
            "source": result.source.unwrap_or_else(|| "user".to_owned()),
            "usage_tag": usage_tag,
            "metadata": metadata,
            "credit_scale": CREDIT_SCALE,
        })))
    }

    fn refund_enqueue_failure(
        &mut self,
        user_id: i64,
        reservation: &Value,
    ) -> Result<(), Self::Error> {
        let reserved = reservation
            .get("reserved_credit_units")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let metadata = reservation.get("metadata").and_then(Value::as_object);
        let operation_id = metadata
            .and_then(|metadata| metadata.get("operation_id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let settlement_id = metadata
            .and_then(|metadata| metadata.get("settlement_id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let settlement = Map::from_iter([
            (
                "reason".to_owned(),
                json!("memory_compaction_enqueue_failed"),
            ),
            ("operation_id".to_owned(), json!(operation_id)),
            ("settlement_id".to_owned(), json!(settlement_id)),
            ("credit_scale".to_owned(), json!(CREDIT_SCALE)),
        ]);
        self.repository
            .settle_legacy_ai_reservation_once(
                user_id,
                reservation.get("chat_scope_id").and_then(Value::as_i64),
                reservation
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or("user"),
                reserved,
                0,
                reservation
                    .get("usage_tag")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                &settlement,
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

#[must_use]
pub fn production_compaction_scheduler(
    queue: RedisCompactionQueue,
    database_url: &str,
    system_prompt: &str,
) -> NativeCompactionScheduler<
    RedisCompactionQueue,
    PostgresCompactionReservations,
    impl FnMut() -> String + use<>,
> {
    NativeCompactionScheduler::new(
        queue,
        PostgresCompactionReservations::new(database_url),
        random_token,
        COMPACTION_MODEL,
        system_prompt,
    )
}

fn estimated_message(role: &str, content: &str) -> EstimatedMessage {
    EstimatedMessage {
        role: TokenEstimateValue::Text(role.to_owned()),
        content: TokenEstimateValue::Text(content.to_owned()),
        name: TokenEstimateValue::Empty,
    }
}

fn estimated_stored_message(message: &Value) -> Option<EstimatedMessage> {
    let content = message
        .get("content")
        .or_else(|| message.get("text"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    Some(estimated_message(
        message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user"),
        content,
    ))
}

fn random_token() -> String {
    use rand::Rng;
    let mut bytes = [0_u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::time::{SystemTime, UNIX_EPOCH};

    use bot_adapters::billing_read::BillingRepository;
    use bot_adapters::billing_schema::BillingSchemaRepository;
    use bot_adapters::redis_compaction_queue::RedisCompactionQueue;
    use bot_adapters::redis_connection::RedisEndpoint;
    use serde_json::{Value, json};

    use super::{
        COMPACTION_MODEL, CompactionEnqueueStore, CompactionReservationStore,
        CompactionScheduleContext, MemoryCompactionPlan, MemoryCompactionScheduler,
        NativeCompactionScheduler, PostgresCompactionReservations,
    };
    use bot_core::locale::Locale;

    #[derive(Default)]
    struct Queue {
        exists: bool,
        insert: bool,
        insert_error: bool,
        payloads: Vec<Value>,
    }

    impl CompactionEnqueueStore for Queue {
        type Error = &'static str;

        fn job_exists(&mut self, _chat_id: &str) -> Result<bool, Self::Error> {
            Ok(self.exists)
        }
        fn insert_job(&mut self, _chat_id: &str, payload: &str) -> Result<bool, Self::Error> {
            if self.insert_error {
                return Err("synthetic Redis failure");
            }
            self.payloads
                .push(serde_json::from_str(payload).unwrap_or(Value::Null));
            Ok(self.insert)
        }
    }

    #[derive(Default)]
    struct Billing {
        reserves: usize,
        refunds: usize,
    }

    impl CompactionReservationStore for Billing {
        type Error = Infallible;

        fn reserve(
            &mut self,
            _context: CompactionScheduleContext,
            usage_tag: &str,
            reserve_credit_units: i64,
            _target_marker: &str,
            _message_count: usize,
        ) -> Result<Option<Value>, Self::Error> {
            self.reserves += 1;
            Ok(Some(json!({
                "reserved_credit_units": reserve_credit_units,
                "source":"user",
                "usage_tag": usage_tag,
            })))
        }
        fn refund_enqueue_failure(
            &mut self,
            _user_id: i64,
            _reservation: &Value,
        ) -> Result<(), Self::Error> {
            self.refunds += 1;
            Ok(())
        }
    }

    fn plan() -> MemoryCompactionPlan {
        MemoryCompactionPlan {
            chat_id: "123".to_owned(),
            messages: vec![json!({"id":"1","role":"user","text":"hello"})],
            prior_summary: None,
            expected_marker: None,
            target_marker: "1".to_owned(),
        }
    }

    fn context() -> CompactionScheduleContext {
        CompactionScheduleContext {
            user_id: 42,
            group_chat_id: Some(-100),
            origin_chat_id: -100,
            message_id: 9,
            locale: Locale::En,
            payer_source: Some(super::PayerSource::Chat),
        }
    }

    #[test]
    fn production_reservation_and_queue_ports_round_trip_against_local_stores() -> Result<(), String>
    {
        let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
            return Ok(());
        };
        let Some(port) = std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
        else {
            return Ok(());
        };
        BillingSchemaRepository::new(&database_url)
            .ensure_schema()
            .map_err(|error| error.to_string())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let suffix = i64::try_from(nonce % 100_000_000).map_err(|error| error.to_string())?;
        let user_id = 7_100_000_000_000_i64 + suffix;
        let repository = BillingRepository::new(&database_url);
        repository
            .mint_user_credits(user_id, 100, None)
            .map_err(|error| error.to_string())?;
        let context = CompactionScheduleContext {
            user_id,
            group_chat_id: None,
            origin_chat_id: user_id,
            message_id: 7,
            locale: Locale::En,
            payer_source: Some(super::PayerSource::User),
        };
        let usage_tag = format!("synthetic-compaction-{nonce}");
        let mut billing = PostgresCompactionReservations::new(&database_url);
        let reservation = billing
            .reserve(context, &usage_tag, 10, "message-7", 3)?
            .ok_or_else(|| "synthetic reservation was denied".to_owned())?;
        assert_eq!(reservation["source"], "user");
        assert_eq!(
            repository
                .get_balance("user", user_id)
                .map_err(|error| error.to_string())?,
            90
        );
        billing.refund_enqueue_failure(user_id, &reservation)?;
        assert_eq!(
            repository
                .get_balance("user", user_id)
                .map_err(|error| error.to_string())?,
            100
        );

        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
            port,
            password: std::env::var("TEST_REDIS_PASSWORD")
                .ok()
                .filter(|value| !value.is_empty()),
        };
        let mut queue = RedisCompactionQueue::new(&endpoint).map_err(|error| error.to_string())?;
        let chat_id = format!("synthetic-scheduler-{nonce}");
        assert!(
            !CompactionEnqueueStore::job_exists(&mut queue, &chat_id)
                .map_err(|error| error.to_string())?
        );
        assert!(
            CompactionEnqueueStore::insert_job(&mut queue, &chat_id, r#"{"value":1}"#)
                .map_err(|error| error.to_string())?
        );
        assert!(
            CompactionEnqueueStore::job_exists(&mut queue, &chat_id)
                .map_err(|error| error.to_string())?
        );
        Ok(())
    }

    #[test]
    fn reserves_and_persists_a_python_readable_job() {
        let mut scheduler = NativeCompactionScheduler::new(
            Queue {
                insert: true,
                ..Queue::default()
            },
            Billing::default(),
            || "nonce".to_owned(),
            "deepseek/deepseek-v4-flash-0731",
            "persona",
        );
        assert_eq!(scheduler.schedule(plan(), context()), Ok(true));
        let (queue, billing, _) = scheduler.into_parts();
        assert_eq!(billing.reserves, 1);
        assert_eq!(billing.refunds, 0);
        assert_eq!(queue.payloads[0]["schema_version"], 1);
        assert_eq!(queue.payloads[0]["chat_id"], "123");
        assert_eq!(queue.payloads[0]["locale"], "en");
        assert_eq!(queue.payloads[0]["reservation"]["credit_scale"], 100);
        assert_eq!(
            queue.payloads[0]["reservation"]["usage_tag"],
            "memory_compaction:123:1:nonce"
        );
    }

    #[test]
    fn existing_job_skips_reservation_and_lost_insert_refunds() {
        let mut existing = NativeCompactionScheduler::new(
            Queue {
                exists: true,
                insert: true,
                ..Queue::default()
            },
            Billing::default(),
            || "nonce".to_owned(),
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(existing.schedule(plan(), context()), Ok(false));
        assert_eq!(existing.into_parts().1.reserves, 0);

        let mut lost = NativeCompactionScheduler::new(
            Queue::default(),
            Billing::default(),
            || "nonce".to_owned(),
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(lost.schedule(plan(), context()), Ok(false));
        let (_, billing, _) = lost.into_parts();
        assert_eq!(billing.reserves, 1);
        assert_eq!(billing.refunds, 1);

        let mut failed = NativeCompactionScheduler::new(
            Queue {
                insert_error: true,
                ..Queue::default()
            },
            Billing::default(),
            || "nonce".to_owned(),
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(
            failed.schedule(plan(), context()),
            Err("synthetic Redis failure".to_owned())
        );
        assert_eq!(failed.into_parts().1.refunds, 1);
    }
}
