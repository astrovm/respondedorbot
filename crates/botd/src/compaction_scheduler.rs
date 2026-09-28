//! Foreground planning handoff for durable background memory compaction.

use std::fmt::Display;
use std::sync::Arc;

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::compaction_job::{COMPACTION_JOB_SCHEMA_VERSION, CompactionJobRecord};
use bot_adapters::openrouter_chat::OpenRouterPricingCache;
use bot_adapters::redis_compaction_queue::RedisCompactionQueue;
use bot_core::ai_reserve::{
    EstimatedMessage, TokenEstimateValue, chat_output_token_limit,
    estimate_chat_reserve_credit_units_with_pricing,
};
use bot_core::credit_units::CREDIT_SCALE;
use bot_core::locale::Locale;
use serde_json::{Map, Value, json};

use crate::compaction_adapters::COMPACTION_MODEL;
use crate::native_ai::reservation_pricing_for_model;

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
    pricing: Option<Arc<OpenRouterPricingCache>>,
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
            pricing: None,
        }
    }

    #[must_use]
    pub fn with_openrouter_pricing(mut self, pricing: Arc<OpenRouterPricingCache>) -> Self {
        self.pricing = Some(pricing);
        self
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
        let pricing = reservation_pricing_for_model(&self.model, self.pricing.as_deref())?;
        estimate_chat_reserve_credit_units_with_pricing(
            Some(&system),
            &messages,
            Some(chat_output_token_limit(&self.model)),
            0,
            &self.model,
            &pricing,
        )
        .map(|units| units.max(1))
        .map_err(crate::error_text)
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
            .map_err(crate::error_text)?
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
            .map_err(crate::error_text)?
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
        let payload = serde_json::to_string(&job).map_err(crate::error_text)?;
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
                .map_err(crate::error_text)?;
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
            .or(Err("compaction reservation exceeds the database range"))?;
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
            .map_err(crate::error_text)?;
        if !result.ok {
            return Ok(None);
        }
        Ok(Some(json!({
            "reserved_credit_units": result.amount,
            "chat_scope_id": context.group_chat_id,
            "source": result.source.unwrap_or(String::from("user")),
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
            .map_err(crate::error_text)
    }
}

#[must_use]
pub fn production_compaction_scheduler(
    queue: RedisCompactionQueue,
    database_url: &str,
    system_prompt: &str,
    pricing: Option<Arc<OpenRouterPricingCache>>,
) -> NativeCompactionScheduler<
    RedisCompactionQueue,
    PostgresCompactionReservations,
    impl FnMut() -> String + use<>,
> {
    let scheduler = NativeCompactionScheduler::new(
        queue,
        PostgresCompactionReservations::new(database_url),
        random_token,
        COMPACTION_MODEL,
        system_prompt,
    );
    if let Some(pricing) = pricing {
        scheduler.with_openrouter_pricing(pricing)
    } else {
        scheduler
    }
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
        deny: bool,
        refund_error: bool,
    }

    impl CompactionReservationStore for Billing {
        type Error = &'static str;

        fn reserve(
            &mut self,
            _context: CompactionScheduleContext,
            usage_tag: &str,
            reserve_credit_units: i64,
            _target_marker: &str,
            _message_count: usize,
        ) -> Result<Option<Value>, Self::Error> {
            self.reserves += 1;
            if self.deny {
                return Ok(None);
            }
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
            if self.refund_error {
                return Err("synthetic refund failure");
            }
            Ok(())
        }
    }

    fn token() -> String {
        "nonce".to_owned()
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
        local_stores().map_or(Ok(()), |(database_url, endpoint)| {
            assert_ports_round_trip(&database_url, &endpoint)
        })
    }

    fn local_stores() -> Option<(String, RedisEndpoint)> {
        crate::test_env::database_url().zip(crate::test_env::redis_endpoint())
    }

    fn synthetic_user(database_url: &str, base: i64) -> Result<(u128, i64), String> {
        BillingSchemaRepository::new(database_url)
            .ensure_schema()
            .map_err(crate::error_text)?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(crate::error_text)?
            .as_nanos();
        let suffix = i64::try_from(nonce % 100_000_000).map_err(crate::error_text)?;
        let user_id = base + suffix;
        BillingRepository::new(database_url)
            .mint_user_credits(user_id, 100, None)
            .map_err(crate::error_text)?;
        Ok((nonce, user_id))
    }

    fn assert_ports_round_trip(database_url: &str, endpoint: &RedisEndpoint) -> Result<(), String> {
        let (nonce, user_id) = synthetic_user(database_url, 7_100_000_000_000)?;
        let repository = BillingRepository::new(database_url);
        let context = CompactionScheduleContext {
            user_id,
            group_chat_id: None,
            origin_chat_id: user_id,
            message_id: 7,
            locale: Locale::En,
            payer_source: Some(super::PayerSource::User),
        };
        let usage_tag = format!("synthetic-compaction-{nonce}");
        let mut billing = PostgresCompactionReservations::new(database_url);
        let reservation = billing
            .reserve(context, &usage_tag, 10, "message-7", 3)?
            .ok_or(String::from("synthetic reservation was denied"))?;
        assert_eq!(reservation["source"], "user");
        assert_eq!(
            repository
                .get_balance("user", user_id)
                .map_err(crate::error_text)?,
            90
        );
        billing.refund_enqueue_failure(user_id, &reservation)?;
        assert_eq!(
            repository
                .get_balance("user", user_id)
                .map_err(crate::error_text)?,
            100
        );

        let chat_payer = CompactionScheduleContext {
            group_chat_id: Some(-user_id),
            payer_source: Some(super::PayerSource::Chat),
            ..context
        };
        let chat_tag = format!("synthetic-compaction-chat-{nonce}");
        assert_eq!(
            billing.reserve(chat_payer, &chat_tag, 10, "message-8", 3),
            Ok(None)
        );
        assert_eq!(
            repository
                .get_balance("user", user_id)
                .map_err(crate::error_text)?,
            100
        );

        let mut queue = RedisCompactionQueue::new(endpoint).map_err(crate::error_text)?;
        let chat_id = format!("synthetic-scheduler-{nonce}");
        assert!(
            !CompactionEnqueueStore::job_exists(&mut queue, &chat_id).map_err(crate::error_text)?
        );
        assert!(
            CompactionEnqueueStore::insert_job(&mut queue, &chat_id, r#"{"value":1}"#)
                .map_err(crate::error_text)?
        );
        assert!(
            CompactionEnqueueStore::job_exists(&mut queue, &chat_id).map_err(crate::error_text)?
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
            token,
            COMPACTION_MODEL,
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
            token,
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(existing.schedule(plan(), context()), Ok(false));
        assert_eq!(existing.into_parts().1.reserves, 0);

        let mut lost = NativeCompactionScheduler::new(
            Queue::default(),
            Billing::default(),
            token,
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
            token,
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(
            failed.schedule(plan(), context()),
            Err("synthetic Redis failure".to_owned())
        );
        assert_eq!(failed.into_parts().1.refunds, 1);
    }

    #[test]
    fn spanish_context_persists_the_spanish_summary_locale() {
        let mut scheduler = NativeCompactionScheduler::new(
            Queue {
                insert: true,
                ..Queue::default()
            },
            Billing::default(),
            token,
            COMPACTION_MODEL,
            "persona",
        );
        let spanish = CompactionScheduleContext {
            locale: Locale::Es,
            ..context()
        };
        assert_eq!(scheduler.schedule(plan(), spanish), Ok(true));
        let (queue, _, _) = scheduler.into_parts();
        assert_eq!(queue.payloads[0]["locale"], "es");
        assert_eq!(queue.payloads[0]["message_id"], "9");
    }

    #[test]
    fn production_scheduler_reserves_with_a_random_usage_tag_and_enqueues() -> Result<(), String> {
        local_stores().map_or(Ok(()), |(database_url, endpoint)| {
            assert_production_scheduler_enqueues(&database_url, &endpoint)
        })
    }

    fn assert_production_scheduler_enqueues(
        database_url: &str,
        endpoint: &RedisEndpoint,
    ) -> Result<(), String> {
        let (nonce, user_id) = synthetic_user(database_url, 7_150_000_000_000)?;
        let queue = RedisCompactionQueue::new(endpoint).map_err(crate::error_text)?;
        let mut scheduler =
            super::production_compaction_scheduler(queue, database_url, "persona", None);
        let chat_id = format!("synthetic-production-scheduler-{nonce}");
        let production_plan = MemoryCompactionPlan {
            chat_id: chat_id.clone(),
            ..plan()
        };
        let context = CompactionScheduleContext {
            user_id,
            group_chat_id: None,
            origin_chat_id: user_id,
            payer_source: None,
            ..context()
        };
        assert_eq!(
            scheduler.schedule(production_plan.clone(), context),
            Ok(true)
        );
        assert_eq!(scheduler.schedule(production_plan, context), Ok(false));
        let (mut queue, _, mut next_token) = scheduler.into_parts();
        assert!(
            CompactionEnqueueStore::job_exists(&mut queue, &chat_id).map_err(crate::error_text)?
        );
        let (first, second) = (next_token(), next_token());
        assert_eq!(first.len(), 32);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first, second);
        let balance = BillingRepository::new(database_url)
            .get_balance("user", user_id)
            .map_err(crate::error_text)?;
        assert!(balance < 100, "reservation was not charged: {balance}");
        Ok(())
    }

    #[test]
    fn denied_reservations_and_failed_refunds_are_reported_without_enqueueing() {
        let mut denied = NativeCompactionScheduler::new(
            Queue {
                insert: true,
                ..Queue::default()
            },
            Billing {
                deny: true,
                ..Billing::default()
            },
            token,
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(denied.schedule(plan(), context()), Ok(false));
        let (queue, billing, _) = denied.into_parts();
        assert!(queue.payloads.is_empty());
        assert_eq!((billing.reserves, billing.refunds), (1, 0));

        let mut lost_refund = NativeCompactionScheduler::new(
            Queue {
                insert_error: true,
                ..Queue::default()
            },
            Billing {
                refund_error: true,
                ..Billing::default()
            },
            token,
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(
            lost_refund.schedule(plan(), context()),
            Err(
                "synthetic Redis failure; compaction reservation refund failed: synthetic refund failure"
                    .to_owned()
            )
        );

        let mut unstored_refund = NativeCompactionScheduler::new(
            Queue::default(),
            Billing {
                refund_error: true,
                ..Billing::default()
            },
            token,
            COMPACTION_MODEL,
            "persona",
        );
        assert_eq!(
            unstored_refund.schedule(plan(), context()),
            Err("synthetic refund failure".to_owned())
        );
        assert_eq!(unstored_refund.into_parts().1.refunds, 1);
    }

    #[test]
    fn oversized_reservations_are_rejected_before_database_io() {
        let mut billing = PostgresCompactionReservations::new("postgresql://synthetic.invalid/db");
        assert_eq!(
            billing.reserve(context(), "synthetic-tag", i64::from(i32::MAX) + 1, "1", 1),
            Err("compaction reservation exceeds the database range".to_owned())
        );
    }
}
