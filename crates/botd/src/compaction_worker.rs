//! Durable, crash-safe memory-compaction orchestration.

use std::fmt::Display;

use bot_adapters::compaction_job::{COMPACTION_JOB_SCHEMA_VERSION, CompactionJobRecord};
use bot_adapters::redis_compaction_queue::{QueueJob, RedisCompactionQueue};
use bot_core::compaction_policy::{
    CompactionDisposition, evaluate_compaction, is_due, retry_after_failure,
};
use serde_json::{Value, json};
use thiserror::Error;

const DEFAULT_LOCK_TTL_SECONDS: i64 = 300;

pub trait CompactionQueue {
    type Error: Display;

    fn list_jobs(&mut self) -> Result<Vec<QueueJob>, Self::Error>;
    fn replace_job(&mut self, chat_id: &str, payload: &str) -> Result<(), Self::Error>;
    fn delete_job(&mut self, chat_id: &str) -> Result<bool, Self::Error>;
    fn acquire_lock(
        &mut self,
        chat_id: &str,
        token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, Self::Error>;
    fn release_lock(&mut self, chat_id: &str, token: &str) -> Result<bool, Self::Error>;
    fn quarantine_job(
        &mut self,
        chat_id: &str,
        dead_job_id: &str,
        dead_payload: &str,
    ) -> Result<bool, Self::Error>;
}

impl CompactionQueue for RedisCompactionQueue {
    type Error = bot_adapters::redis_compaction_queue::RedisCompactionQueueError;

    fn list_jobs(&mut self) -> Result<Vec<QueueJob>, Self::Error> {
        RedisCompactionQueue::list_jobs(self)
    }

    fn replace_job(&mut self, chat_id: &str, payload: &str) -> Result<(), Self::Error> {
        RedisCompactionQueue::replace_job(self, chat_id, payload)
    }

    fn delete_job(&mut self, chat_id: &str) -> Result<bool, Self::Error> {
        RedisCompactionQueue::delete_job(self, chat_id)
    }

    fn acquire_lock(
        &mut self,
        chat_id: &str,
        token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, Self::Error> {
        RedisCompactionQueue::acquire_lock(self, chat_id, token, ttl_seconds)
    }

    fn release_lock(&mut self, chat_id: &str, token: &str) -> Result<bool, Self::Error> {
        RedisCompactionQueue::release_lock(self, chat_id, token)
    }

    fn quarantine_job(
        &mut self,
        chat_id: &str,
        dead_job_id: &str,
        dead_payload: &str,
    ) -> Result<bool, Self::Error> {
        RedisCompactionQueue::quarantine_job(self, chat_id, dead_job_id, dead_payload)
    }
}

pub trait CompactionState {
    type Error: Display;

    fn load(&mut self, chat_id: &str) -> Result<(Option<String>, Option<String>), Self::Error>;
    fn save(
        &mut self,
        chat_id: &str,
        summary: &str,
        target_marker: &str,
    ) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionProviderResult {
    pub summary: String,
    pub cost_usd_micros: i64,
    pub billing_segment: Option<Value>,
}

pub trait CompactionProvider {
    type Error: Display;

    fn compact(
        &mut self,
        messages: &[Value],
        prior_summary: Option<&str>,
        locale: &str,
    ) -> Result<CompactionProviderResult, Self::Error>;
}

pub struct SettlementRequest<'a> {
    pub job: &'a CompactionJobRecord,
    pub billing_segments: &'a [Value],
    pub actual_credit_units: Option<i64>,
    pub reason: &'a str,
}

pub trait CompactionBilling {
    type Error: Display;

    fn is_settled(&mut self, job: &CompactionJobRecord) -> Result<bool, Self::Error>;
    fn list_provider_segments(
        &mut self,
        user_id: i64,
        operation_id: &str,
    ) -> Result<Vec<Value>, Self::Error>;
    fn record_provider_segment(
        &mut self,
        job: &CompactionJobRecord,
        operation_id: &str,
        segment: &Value,
    ) -> Result<(), Self::Error>;
    fn settle(&mut self, request: SettlementRequest<'_>) -> Result<(), Self::Error>;
    fn settle_incompatible(&mut self, chat_id: &str, decoded: &Value) -> Result<bool, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionFailure {
    pub chat_id: String,
    pub stage: &'static str,
    pub error: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionRunReport {
    pub completed: usize,
    pub retried: usize,
    pub quarantined: usize,
    pub skipped_not_due: usize,
    pub skipped_locked: usize,
    pub failures: Vec<CompactionFailure>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompactionWorkerError {
    #[error("compaction lock TTL must be positive")]
    InvalidLockTtl,
    #[error("compaction queue failed: {0}")]
    Queue(String),
}

pub struct CompactionWorker<Queue, State, Provider, Billing, Token> {
    queue: Queue,
    state: State,
    provider: Provider,
    billing: Billing,
    token: Token,
    lock_ttl_seconds: i64,
}

impl<Queue, State, Provider, Billing, Token>
    CompactionWorker<Queue, State, Provider, Billing, Token>
{
    pub fn new(
        queue: Queue,
        state: State,
        provider: Provider,
        billing: Billing,
        token: Token,
    ) -> Self {
        Self {
            queue,
            state,
            provider,
            billing,
            token,
            lock_ttl_seconds: DEFAULT_LOCK_TTL_SECONDS,
        }
    }

    pub fn with_lock_ttl_seconds(mut self, lock_ttl_seconds: i64) -> Self {
        self.lock_ttl_seconds = lock_ttl_seconds;
        self
    }

    pub fn into_parts(self) -> (Queue, State, Provider, Billing, Token) {
        (
            self.queue,
            self.state,
            self.provider,
            self.billing,
            self.token,
        )
    }
}

impl<Queue, State, Provider, Billing, Token>
    CompactionWorker<Queue, State, Provider, Billing, Token>
where
    Queue: CompactionQueue,
    State: CompactionState,
    Provider: CompactionProvider,
    Billing: CompactionBilling,
    Token: FnMut() -> String,
{
    pub fn run_once(&mut self, now: f64) -> Result<CompactionRunReport, CompactionWorkerError> {
        if self.lock_ttl_seconds <= 0 {
            return Err(CompactionWorkerError::InvalidLockTtl);
        }
        let jobs = self
            .queue
            .list_jobs()
            .map_err(|error| CompactionWorkerError::Queue(error.to_string()))?;
        let mut report = CompactionRunReport::default();
        for raw in jobs {
            self.handle_raw_job(raw, now, &mut report);
        }
        Ok(report)
    }

    fn handle_raw_job(&mut self, raw: QueueJob, now: f64, report: &mut CompactionRunReport) {
        let decoded = match serde_json::from_str::<Value>(&raw.payload) {
            Ok(decoded) => decoded,
            Err(error) => {
                let token = (self.token)();
                let dead_payload = json!({
                    "chat_id": raw.chat_id,
                    "payload": raw.payload,
                    "reason": "undecodable",
                    "quarantined_at": now,
                })
                .to_string();
                match self.queue.quarantine_job(
                    &raw.chat_id,
                    &format!("{}:{token}", raw.chat_id),
                    &dead_payload,
                ) {
                    Ok(true) => report.quarantined += 1,
                    Ok(false) => report.failures.push(failure(
                        &raw.chat_id,
                        "quarantine",
                        "queue did not quarantine the unreadable job",
                    )),
                    Err(queue_error) => report.failures.push(failure(
                        &raw.chat_id,
                        "quarantine",
                        format!("{error}; {queue_error}"),
                    )),
                }
                return;
            }
        };
        let job = match serde_json::from_value::<CompactionJobRecord>(decoded.clone()) {
            Ok(job) if job.schema_version == COMPACTION_JOB_SCHEMA_VERSION => job,
            Ok(job) => {
                self.handle_incompatible(
                    &raw.chat_id,
                    &decoded,
                    format!("unsupported schema version {}", job.schema_version),
                    report,
                );
                return;
            }
            Err(error) => {
                self.handle_incompatible(&raw.chat_id, &decoded, error.to_string(), report);
                return;
            }
        };
        if !is_due(job.next_attempt_at, now) {
            report.skipped_not_due += 1;
            return;
        }
        let token = (self.token)();
        match self
            .queue
            .acquire_lock(&raw.chat_id, &token, self.lock_ttl_seconds)
        {
            Ok(true) => {}
            Ok(false) => {
                report.skipped_locked += 1;
                return;
            }
            Err(error) => {
                report
                    .failures
                    .push(failure(&raw.chat_id, "acquire_lock", error));
                return;
            }
        }

        let outcome = self.process_job(job, now);
        match outcome {
            Ok(()) => match self.queue.delete_job(&raw.chat_id) {
                Ok(_) => report.completed += 1,
                Err(error) => {
                    report
                        .failures
                        .push(failure(&raw.chat_id, "delete_completed", error))
                }
            },
            Err(failed) => {
                let (mut job, stage, error) = *failed;
                let transition =
                    retry_after_failure(job.attempts, now, job.result_billing_segment.is_some());
                job.attempts = transition.attempts;
                if transition.terminal {
                    let settlement = self.settle(
                        &job,
                        transition.actual_credit_units,
                        "memory_compaction_failed",
                    );
                    match settlement.and_then(|()| {
                        self.queue
                            .delete_job(&raw.chat_id)
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    }) {
                        Ok(()) => report.completed += 1,
                        Err(settlement_error) => report.failures.push(failure(
                            &raw.chat_id,
                            "terminal_settlement",
                            format!("{stage}: {error}; {settlement_error}"),
                        )),
                    }
                } else if let Some(next_attempt_at) = transition.next_attempt_at {
                    job.next_attempt_at = next_attempt_at;
                    match serde_json::to_string(&job)
                        .map_err(|error| error.to_string())
                        .and_then(|payload| {
                            self.queue
                                .replace_job(&raw.chat_id, &payload)
                                .map_err(|error| error.to_string())
                        }) {
                        Ok(()) => report.retried += 1,
                        Err(retry_error) => report.failures.push(failure(
                            &raw.chat_id,
                            "schedule_retry",
                            format!("{stage}: {error}; {retry_error}"),
                        )),
                    }
                }
            }
        }
        if let Err(error) = self.queue.release_lock(&raw.chat_id, &token) {
            report
                .failures
                .push(failure(&raw.chat_id, "release_lock", error));
        }
    }

    fn handle_incompatible(
        &mut self,
        chat_id: &str,
        decoded: &Value,
        decode_error: String,
        report: &mut CompactionRunReport,
    ) {
        match self.billing.settle_incompatible(chat_id, decoded) {
            Ok(true) => match self.queue.delete_job(chat_id) {
                Ok(_) => report.completed += 1,
                Err(error) => report.failures.push(failure(
                    chat_id,
                    "delete_incompatible",
                    format!("{decode_error}; {error}"),
                )),
            },
            Ok(false) => report.failures.push(failure(
                chat_id,
                "incompatible",
                format!("{decode_error}; job could not be safely settled"),
            )),
            Err(error) => report.failures.push(failure(
                chat_id,
                "settle_incompatible",
                format!("{decode_error}; {error}"),
            )),
        }
    }

    fn process_job(
        &mut self,
        mut job: CompactionJobRecord,
        _now: f64,
    ) -> Result<(), Box<(CompactionJobRecord, &'static str, String)>> {
        if self
            .billing
            .is_settled(&job)
            .map_err(|error| Box::new((job.clone(), "check_settlement", error.to_string())))?
        {
            return Ok(());
        }
        let (current_summary, current_marker) = self
            .state
            .load(&job.chat_id)
            .map_err(|error| Box::new((job.clone(), "load_state", error.to_string())))?;
        if job.result_summary.as_deref().is_none_or(str::is_empty) {
            let operation_id = reservation_string(&job.reservation, "operation_id");
            if !operation_id.is_empty() {
                let segments = self
                    .billing
                    .list_provider_segments(job.user_id, &operation_id)
                    .map_err(|error| {
                        Box::new((job.clone(), "restore_provider_usage", error.to_string()))
                    })?;
                if let Some((summary, segment)) = restored_summary(&segments) {
                    job.result_summary = Some(summary);
                    job.result_billing_segment = Some(segment);
                    job.result_cost_usd_micros = raw_cost_usd_micros(&segments);
                    let payload = serde_json::to_string(&job).map_err(|error| {
                        Box::new((job.clone(), "encode_recovered_job", error.to_string()))
                    })?;
                    self.queue
                        .replace_job(&job.chat_id, &payload)
                        .map_err(|error| {
                            Box::new((job.clone(), "checkpoint_recovered_job", error.to_string()))
                        })?;
                }
            }
        }

        match evaluate_compaction(
            current_summary.as_deref(),
            current_marker.as_deref(),
            job.prior_summary.as_deref(),
            job.expected_marker.as_deref(),
            job.result_summary.as_deref(),
            &job.target_marker,
        ) {
            CompactionDisposition::SettleRecoveredSuccess => self
                .settle(&job, None, "memory_compaction_success")
                .map_err(|error| Box::new((job, "settle_recovered", error))),
            CompactionDisposition::SettleObsolete => {
                let actual = job.result_billing_segment.is_none().then_some(0);
                self.settle(&job, actual, "memory_compaction_obsolete")
                    .map_err(|error| Box::new((job, "settle_obsolete", error)))
            }
            CompactionDisposition::GenerateSummary => {
                let result = self
                    .provider
                    .compact(&job.messages, job.prior_summary.as_deref(), &job.locale)
                    .map_err(|error| Box::new((job.clone(), "provider", error.to_string())))?;
                if result.summary.trim().is_empty()
                    || (result.cost_usd_micros <= 0 && result.billing_segment.is_none())
                {
                    return Err(Box::new((
                        job,
                        "provider",
                        "summary provider did not produce billable output".to_owned(),
                    )));
                }
                job.result_summary = Some(result.summary);
                job.result_cost_usd_micros = result.cost_usd_micros;
                job.result_billing_segment = result.billing_segment;
                self.persist_provider_usage(&job)
                    .map_err(|error| Box::new((job.clone(), "record_provider_usage", error)))?;
                let payload = serde_json::to_string(&job).map_err(|error| {
                    Box::new((job.clone(), "encode_provider_result", error.to_string()))
                })?;
                self.queue
                    .replace_job(&job.chat_id, &payload)
                    .map_err(|error| {
                        Box::new((job.clone(), "checkpoint_provider_result", error.to_string()))
                    })?;
                self.save_and_settle(job)
            }
            CompactionDisposition::SaveAndSettle => {
                self.persist_provider_usage(&job)
                    .map_err(|error| Box::new((job.clone(), "record_provider_usage", error)))?;
                self.save_and_settle(job)
            }
        }
    }

    fn save_and_settle(
        &mut self,
        job: CompactionJobRecord,
    ) -> Result<(), Box<(CompactionJobRecord, &'static str, String)>> {
        let Some(summary) = job
            .result_summary
            .as_deref()
            .filter(|value| !value.is_empty())
        else {
            return Err(Box::new((
                job,
                "save_state",
                "compaction result is empty".to_owned(),
            )));
        };
        self.state
            .save(&job.chat_id, summary, &job.target_marker)
            .map_err(|error| Box::new((job.clone(), "save_state", error.to_string())))?;
        self.settle(&job, None, "memory_compaction_success")
            .map_err(|error| Box::new((job, "settle_success", error)))
    }

    fn persist_provider_usage(&mut self, job: &CompactionJobRecord) -> Result<(), String> {
        let operation_id = reservation_string(&job.reservation, "operation_id");
        if let Some(segment) = job.result_billing_segment.as_ref()
            && !operation_id.is_empty()
        {
            self.billing
                .record_provider_segment(job, &operation_id, segment)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn settle(
        &mut self,
        job: &CompactionJobRecord,
        actual_credit_units: Option<i64>,
        reason: &str,
    ) -> Result<(), String> {
        self.persist_provider_usage(job)?;
        let operation_id = reservation_string(&job.reservation, "operation_id");
        let mut segments = if operation_id.is_empty() {
            Vec::new()
        } else {
            self.billing
                .list_provider_segments(job.user_id, &operation_id)
                .map_err(|error| error.to_string())?
        };
        if segments.is_empty()
            && let Some(segment) = job.result_billing_segment.clone()
        {
            segments.push(segment);
        }
        self.billing
            .settle(SettlementRequest {
                job,
                billing_segments: &segments,
                actual_credit_units,
                reason,
            })
            .map_err(|error| error.to_string())
    }
}

fn failure(chat_id: &str, stage: &'static str, error: impl Display) -> CompactionFailure {
    CompactionFailure {
        chat_id: chat_id.to_owned(),
        stage,
        error: error.to_string(),
    }
}

fn reservation_string(reservation: &Value, key: &str) -> String {
    reservation
        .get(key)
        .and_then(Value::as_str)
        .or_else(|| reservation.get("metadata")?.get(key)?.as_str())
        .unwrap_or_default()
        .to_owned()
}

fn restored_summary(segments: &[Value]) -> Option<(String, Value)> {
    segments.iter().rev().find_map(|segment| {
        let summary = segment
            .get("metadata")
            .and_then(|metadata| metadata.get("compaction_result"))
            .and_then(Value::as_str)
            .or_else(|| segment.get("text").and_then(Value::as_str))?
            .trim();
        (segment.get("kind").and_then(Value::as_str) == Some("summary") && !summary.is_empty())
            .then(|| (summary.to_owned(), segment.clone()))
    })
}

fn raw_cost_usd_micros(segments: &[Value]) -> i64 {
    bot_core::ai_pricing::calculate_billing_for_segments(&Value::Array(segments.to_vec()))
        .ok()
        .and_then(|billing| billing.get("raw_usd_micros").and_then(Value::as_i64))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::convert::Infallible;

    use serde_json::{Value, json};

    use super::{
        CompactionBilling, CompactionProvider, CompactionProviderResult, CompactionQueue,
        CompactionState, CompactionWorker, SettlementRequest,
    };
    use bot_adapters::compaction_job::CompactionJobRecord;
    use bot_adapters::redis_compaction_queue::QueueJob;

    #[derive(Default)]
    struct Queue {
        jobs: HashMap<String, String>,
        lock_available: bool,
        replaced: Vec<CompactionJobRecord>,
        deleted: Vec<String>,
        quarantined: Vec<Value>,
        releases: usize,
    }

    impl CompactionQueue for Queue {
        type Error = Infallible;

        fn list_jobs(&mut self) -> Result<Vec<QueueJob>, Self::Error> {
            Ok(self
                .jobs
                .iter()
                .map(|(chat_id, payload)| QueueJob {
                    chat_id: chat_id.clone(),
                    payload: payload.clone(),
                })
                .collect())
        }
        fn replace_job(&mut self, chat_id: &str, payload: &str) -> Result<(), Self::Error> {
            self.jobs.insert(chat_id.to_owned(), payload.to_owned());
            if let Ok(job) = serde_json::from_str(payload) {
                self.replaced.push(job);
            }
            Ok(())
        }
        fn delete_job(&mut self, chat_id: &str) -> Result<bool, Self::Error> {
            self.deleted.push(chat_id.to_owned());
            Ok(self.jobs.remove(chat_id).is_some())
        }
        fn acquire_lock(
            &mut self,
            _chat_id: &str,
            _token: &str,
            _ttl_seconds: i64,
        ) -> Result<bool, Self::Error> {
            Ok(self.lock_available)
        }
        fn release_lock(&mut self, _chat_id: &str, _token: &str) -> Result<bool, Self::Error> {
            self.releases += 1;
            Ok(true)
        }
        fn quarantine_job(
            &mut self,
            _chat_id: &str,
            _dead_job_id: &str,
            dead_payload: &str,
        ) -> Result<bool, Self::Error> {
            self.quarantined
                .push(serde_json::from_str(dead_payload).unwrap_or(Value::Null));
            Ok(true)
        }
    }

    #[derive(Default)]
    struct State {
        current: (Option<String>, Option<String>),
        saved: Vec<(String, String, String)>,
    }

    impl CompactionState for State {
        type Error = Infallible;

        fn load(
            &mut self,
            _chat_id: &str,
        ) -> Result<(Option<String>, Option<String>), Self::Error> {
            Ok(self.current.clone())
        }
        fn save(&mut self, chat_id: &str, summary: &str, marker: &str) -> Result<(), Self::Error> {
            self.saved
                .push((chat_id.to_owned(), summary.to_owned(), marker.to_owned()));
            self.current = (Some(summary.to_owned()), Some(marker.to_owned()));
            Ok(())
        }
    }

    struct Provider {
        replies: VecDeque<Result<CompactionProviderResult, &'static str>>,
        calls: usize,
    }

    impl CompactionProvider for Provider {
        type Error = &'static str;

        fn compact(
            &mut self,
            _messages: &[Value],
            _prior_summary: Option<&str>,
            _locale: &str,
        ) -> Result<CompactionProviderResult, Self::Error> {
            self.calls += 1;
            self.replies.pop_front().unwrap_or(Err("no reply"))
        }
    }

    #[derive(Default)]
    struct Billing {
        settled: bool,
        durable: Vec<Value>,
        recorded: Vec<Value>,
        settlements: Vec<(String, Option<i64>, Vec<Value>)>,
        incompatible: usize,
    }

    impl CompactionBilling for Billing {
        type Error = Infallible;

        fn is_settled(&mut self, _job: &CompactionJobRecord) -> Result<bool, Self::Error> {
            Ok(self.settled)
        }
        fn list_provider_segments(
            &mut self,
            _user_id: i64,
            _operation_id: &str,
        ) -> Result<Vec<Value>, Self::Error> {
            Ok(self.durable.clone())
        }
        fn record_provider_segment(
            &mut self,
            _job: &CompactionJobRecord,
            _operation_id: &str,
            segment: &Value,
        ) -> Result<(), Self::Error> {
            self.recorded.push(segment.clone());
            if !self.durable.contains(segment) {
                self.durable.push(segment.clone());
            }
            Ok(())
        }
        fn settle(&mut self, request: SettlementRequest<'_>) -> Result<(), Self::Error> {
            self.settlements.push((
                request.reason.to_owned(),
                request.actual_credit_units,
                request.billing_segments.to_vec(),
            ));
            Ok(())
        }
        fn settle_incompatible(
            &mut self,
            _chat_id: &str,
            _decoded: &Value,
        ) -> Result<bool, Self::Error> {
            self.incompatible += 1;
            Ok(true)
        }
    }

    fn job() -> CompactionJobRecord {
        CompactionJobRecord {
            schema_version: 1,
            chat_id: "chat-1".to_owned(),
            messages: vec![json!({"role":"user","content":"hello"})],
            prior_summary: None,
            expected_marker: None,
            target_marker: "m1".to_owned(),
            reservation: json!({
                "operation_id":"operation-1",
                "usage_tag":"memory_compaction:chat-1:m1:test",
            }),
            user_id: 7,
            message_id: Some("9".to_owned()),
            locale: "en".to_owned(),
            attempts: 0,
            next_attempt_at: 0.0,
            result_summary: None,
            result_cost_usd_micros: 0,
            result_billing_segment: None,
        }
    }

    fn worker(
        job: CompactionJobRecord,
        provider: Provider,
        billing: Billing,
    ) -> CompactionWorker<Queue, State, Provider, Billing, impl FnMut() -> String> {
        let mut queue = Queue {
            lock_available: true,
            ..Queue::default()
        };
        queue.jobs.insert(
            job.chat_id.clone(),
            serde_json::to_string(&job).unwrap_or_default(),
        );
        CompactionWorker::new(queue, State::default(), provider, billing, || {
            "token".to_owned()
        })
    }

    #[test]
    fn checkpoints_usage_and_result_before_saving_and_settling() {
        let segment = json!({
            "kind":"summary", "source":"openrouter", "model":"model",
            "text":"dense summary", "usage":{"cost":"0.001"},
        });
        let provider = Provider {
            replies: VecDeque::from([Ok(CompactionProviderResult {
                summary: "dense summary".to_owned(),
                cost_usd_micros: 1_000,
                billing_segment: Some(segment.clone()),
            })]),
            calls: 0,
        };
        let mut worker = worker(job(), provider, Billing::default());
        let report = worker.run_once(100.0).unwrap_or_default();
        assert_eq!(report.completed, 1);
        assert!(report.failures.is_empty());
        let (queue, state, provider, billing, _) = worker.into_parts();
        assert_eq!(provider.calls, 1);
        assert_eq!(billing.recorded, [segment.clone(), segment]);
        assert_eq!(billing.settlements[0].0, "memory_compaction_success");
        assert_eq!(
            state.saved[0],
            (
                "chat-1".to_owned(),
                "dense summary".to_owned(),
                "m1".to_owned()
            )
        );
        assert_eq!(
            queue.replaced[0].result_summary.as_deref(),
            Some("dense summary")
        );
        assert_eq!(queue.releases, 1);
    }

    #[test]
    fn deletes_already_settled_jobs_without_repeating_provider_work() {
        let provider = Provider {
            replies: VecDeque::new(),
            calls: 0,
        };
        let mut worker = worker(
            job(),
            provider,
            Billing {
                settled: true,
                ..Billing::default()
            },
        );

        assert_eq!(worker.run_once(100.0).unwrap_or_default().completed, 1);

        let (queue, state, provider, billing, _) = worker.into_parts();
        assert_eq!(provider.calls, 0);
        assert!(state.saved.is_empty());
        assert!(billing.settlements.is_empty());
        assert!(queue.jobs.is_empty());
    }

    #[test]
    fn restores_durable_provider_result_without_calling_provider() {
        let segment = json!({
            "kind":"summary", "source":"openrouter", "model":"model",
            "text":"recovered", "usage":{"cost":"0.002"},
        });
        let provider = Provider {
            replies: VecDeque::new(),
            calls: 0,
        };
        let mut worker = worker(
            job(),
            provider,
            Billing {
                durable: vec![segment],
                ..Billing::default()
            },
        );
        assert_eq!(worker.run_once(100.0).unwrap_or_default().completed, 1);
        let (queue, state, provider, billing, _) = worker.into_parts();
        assert_eq!(provider.calls, 0);
        assert_eq!(
            queue.replaced[0].result_summary.as_deref(),
            Some("recovered")
        );
        assert_eq!(state.saved[0].1, "recovered");
        assert_eq!(billing.settlements[0].0, "memory_compaction_success");
    }

    #[test]
    fn restores_usage_for_legacy_reservations_with_nested_operation_identity() {
        let mut nested = job();
        nested.reservation = json!({
            "metadata": {"operation_id":"operation-1"},
            "usage_tag":"memory_compaction:chat-1:m1:test",
        });
        let segment = json!({
            "kind":"summary", "source":"openrouter", "model":"model",
            "text":"nested recovery", "usage":{"cost":"0.002"},
        });
        let provider = Provider {
            replies: VecDeque::new(),
            calls: 0,
        };
        let mut worker = worker(
            nested,
            provider,
            Billing {
                durable: vec![segment],
                ..Billing::default()
            },
        );
        assert_eq!(worker.run_once(100.0).unwrap_or_default().completed, 1);
        let (_, state, provider, _, _) = worker.into_parts();
        assert_eq!(provider.calls, 0);
        assert_eq!(state.saved[0].1, "nested recovery");
    }

    #[test]
    fn settles_recovered_and_obsolete_jobs_without_regeneration() {
        let mut recovered = job();
        recovered.result_summary = Some("done".to_owned());
        recovered.result_billing_segment = Some(json!({"kind":"summary","text":"done"}));
        let provider = Provider {
            replies: VecDeque::new(),
            calls: 0,
        };
        let mut recovered_worker = worker(recovered, provider, Billing::default());
        recovered_worker.state.current = (Some("done".to_owned()), Some("m1".to_owned()));
        assert_eq!(
            recovered_worker.run_once(1.0).unwrap_or_default().completed,
            1
        );
        let (_, _, provider, billing, _) = recovered_worker.into_parts();
        assert_eq!(provider.calls, 0);
        assert_eq!(billing.settlements[0].0, "memory_compaction_success");

        let provider = Provider {
            replies: VecDeque::new(),
            calls: 0,
        };
        let mut worker = worker(job(), provider, Billing::default());
        worker.state.current = (Some("newer".to_owned()), Some("m2".to_owned()));
        assert_eq!(worker.run_once(1.0).unwrap_or_default().completed, 1);
        let (_, _, _, billing, _) = worker.into_parts();
        assert_eq!(billing.settlements[0].0, "memory_compaction_obsolete");
        assert_eq!(billing.settlements[0].1, Some(0));
    }

    #[test]
    fn retries_with_backoff_then_refunds_terminal_failure() {
        let provider = Provider {
            replies: VecDeque::from([Err("one"), Err("two"), Err("three")]),
            calls: 0,
        };
        let mut worker = worker(job(), provider, Billing::default());
        let first = worker.run_once(100.0).unwrap_or_default();
        assert_eq!(first.retried, 1);
        assert_eq!(
            worker.queue.replaced.last().map(|job| job.attempts),
            Some(1)
        );
        assert_eq!(
            worker.run_once(129.0).unwrap_or_default().skipped_not_due,
            1
        );
        assert_eq!(worker.run_once(130.0).unwrap_or_default().retried, 1);
        assert_eq!(worker.run_once(190.0).unwrap_or_default().completed, 1);
        let (queue, _, provider, billing, _) = worker.into_parts();
        assert_eq!(provider.calls, 3);
        assert_eq!(billing.settlements[0].0, "memory_compaction_failed");
        assert_eq!(billing.settlements[0].1, Some(0));
        assert_eq!(queue.releases, 3);
    }

    #[test]
    fn skips_contended_locks_and_quarantines_only_undecodable_json() {
        let provider = Provider {
            replies: VecDeque::new(),
            calls: 0,
        };
        let mut worker = worker(job(), provider, Billing::default());
        worker.queue.lock_available = false;
        assert_eq!(worker.run_once(1.0).unwrap_or_default().skipped_locked, 1);
        assert_eq!(worker.queue.releases, 0);

        worker.queue.jobs.clear();
        worker
            .queue
            .jobs
            .insert("bad".to_owned(), "not-json".to_owned());
        assert_eq!(worker.run_once(2.0).unwrap_or_default().quarantined, 1);
        assert_eq!(worker.queue.quarantined[0]["reason"], "undecodable");

        worker.queue.jobs.clear();
        worker
            .queue
            .jobs
            .insert("old".to_owned(), json!({"schema_version":9}).to_string());
        assert_eq!(worker.run_once(3.0).unwrap_or_default().completed, 1);
        assert_eq!(worker.billing.incompatible, 1);
    }
}
