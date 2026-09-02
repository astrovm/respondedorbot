#![allow(clippy::panic)]

use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use bot_adapters::compaction_job::{COMPACTION_JOB_SCHEMA_VERSION, CompactionJobRecord};
use bot_adapters::redis_compaction_queue::QueueJob;
use bot_adapters::redis_update_queue::QueuedUpdate;
use bot_adapters::telegram_polling::{IncomingEvent, IncomingUpdate};
use bot_core::locale::Locale;
use botd::background::{
    BackgroundError, BackgroundSupervisor, BackgroundWorker, BackgroundWorkerSpec,
};
use botd::compaction_scheduler::{
    CompactionEnqueueStore, CompactionReservationStore, CompactionScheduleContext,
    MemoryCompactionPlan, MemoryCompactionScheduler, NativeCompactionScheduler, PayerSource,
};
use botd::compaction_worker::{
    CompactionBilling, CompactionProvider, CompactionProviderResult, CompactionQueue,
    CompactionState, CompactionWorker, SettlementRequest,
};
use botd::operational_reporting::{OperationalReport, OperationalReporter};
use botd::runtime::{
    DurableParallelUpdateHandler, DurableUpdateQueue, ParallelHandlerBuildError,
    UpdateConfirmation, UpdateHandler,
};
use serde_json::{Value, json};

const TEST_MODEL: &str = "deepseek/deepseek-v4-flash-0731";

#[derive(Default)]
struct Reporter {
    reports: Mutex<Vec<OperationalReport>>,
    fail: bool,
}

impl OperationalReporter for Reporter {
    fn report(&self, report: &OperationalReport) -> Result<(), String> {
        self.reports
            .lock()
            .map_err(|_| "synthetic reporter lock failure".to_owned())?
            .push(report.clone());
        if self.fail {
            Err("synthetic reporter failure".to_owned())
        } else {
            Ok(())
        }
    }
}

struct ShutdownFailureWorker;

impl BackgroundWorker for ShutdownFailureWorker {
    fn run_once(&mut self, _now_epoch_seconds: i64) -> Result<(), String> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), String> {
        Err("synthetic shutdown failure".to_owned())
    }
}

struct PanicWorker;

impl BackgroundWorker for PanicWorker {
    fn run_once(&mut self, _now_epoch_seconds: i64) -> Result<(), String> {
        panic!("synthetic background panic")
    }
}

#[test]
fn background_failures_are_reported_even_when_reporting_or_shutdown_fails() {
    let reporter = Arc::new(Reporter {
        fail: true,
        ..Reporter::default()
    });
    let invalid = BackgroundSupervisor::start(
        vec![BackgroundWorkerSpec::new(
            "zero-interval",
            Duration::ZERO,
            Box::new(ShutdownFailureWorker),
        )],
        reporter.clone(),
    );
    assert!(matches!(
        invalid,
        Err(BackgroundError::InvalidInterval { .. })
    ));

    let mut supervisor = BackgroundSupervisor::start(
        vec![BackgroundWorkerSpec::new(
            "shutdown-failure",
            Duration::from_secs(60),
            Box::new(ShutdownFailureWorker),
        )],
        reporter.clone(),
    )
    .unwrap_or_else(|_| unreachable!());
    thread::sleep(Duration::from_millis(20));
    assert!(supervisor.stop().is_ok());

    let mut panicking = BackgroundSupervisor::start(
        vec![BackgroundWorkerSpec::new(
            "panic-worker",
            Duration::from_secs(60),
            Box::new(PanicWorker),
        )],
        reporter.clone(),
    )
    .unwrap_or_else(|_| unreachable!());
    for _ in 0..100 {
        if panicking.has_failed() {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert!(panicking.has_failed());
    assert!(matches!(
        panicking.stop(),
        Err(BackgroundError::Panicked { .. })
    ));
    assert!(
        reporter
            .reports
            .lock()
            .is_ok_and(|reports| reports.len() >= 3)
    );
}

#[derive(Default)]
struct Queue {
    jobs: Vec<QueueJob>,
    replaced: Vec<String>,
    deleted: Vec<String>,
    quarantined: Vec<String>,
}

impl CompactionQueue for Queue {
    type Error = String;

    fn list_jobs(&mut self) -> Result<Vec<QueueJob>, Self::Error> {
        if self.jobs.iter().any(|job| job.chat_id == "list-error") {
            return Err("synthetic list failure".to_owned());
        }
        Ok(std::mem::take(&mut self.jobs))
    }

    fn replace_job(&mut self, chat_id: &str, _payload: &str) -> Result<(), Self::Error> {
        if chat_id.contains("replace-error") {
            return Err("synthetic replace failure".to_owned());
        }
        self.replaced.push(chat_id.to_owned());
        Ok(())
    }

    fn delete_job(&mut self, chat_id: &str) -> Result<bool, Self::Error> {
        if chat_id.contains("delete-error") {
            return Err("synthetic delete failure".to_owned());
        }
        self.deleted.push(chat_id.to_owned());
        Ok(true)
    }

    fn acquire_lock(
        &mut self,
        chat_id: &str,
        _token: &str,
        _ttl_seconds: i64,
    ) -> Result<bool, Self::Error> {
        if chat_id.contains("lock-error") {
            Err("synthetic lock failure".to_owned())
        } else {
            Ok(!chat_id.contains("lock-busy"))
        }
    }

    fn release_lock(&mut self, chat_id: &str, _token: &str) -> Result<bool, Self::Error> {
        if chat_id.contains("release-error") {
            Err("synthetic release failure".to_owned())
        } else {
            Ok(true)
        }
    }

    fn quarantine_job(
        &mut self,
        chat_id: &str,
        _dead_job_id: &str,
        _dead_payload: &str,
    ) -> Result<bool, Self::Error> {
        if chat_id.contains("quarantine-error") {
            Err("synthetic quarantine failure".to_owned())
        } else if chat_id.contains("quarantine-false") {
            Ok(false)
        } else {
            self.quarantined.push(chat_id.to_owned());
            Ok(true)
        }
    }
}

#[derive(Default)]
struct State {
    saved: Vec<String>,
}

impl CompactionState for State {
    type Error = String;

    fn load(&mut self, chat_id: &str) -> Result<(Option<String>, Option<String>), Self::Error> {
        if chat_id.contains("load-error") {
            Err("synthetic state load failure".to_owned())
        } else if chat_id.contains("recovered") {
            Ok((
                Some("synthetic summary".to_owned()),
                Some("target".to_owned()),
            ))
        } else if chat_id.contains("obsolete") {
            Ok((Some("new state".to_owned()), Some("other".to_owned())))
        } else {
            Ok((None, None))
        }
    }

    fn save(
        &mut self,
        chat_id: &str,
        _summary: &str,
        _target_marker: &str,
    ) -> Result<(), Self::Error> {
        if chat_id.contains("save-error") {
            Err("synthetic state save failure".to_owned())
        } else {
            self.saved.push(chat_id.to_owned());
            Ok(())
        }
    }
}

#[derive(Default)]
struct Provider;

impl CompactionProvider for Provider {
    type Error = String;

    fn compact(
        &mut self,
        messages: &[Value],
        _prior_summary: Option<&str>,
        _locale: &str,
    ) -> Result<CompactionProviderResult, Self::Error> {
        let behavior = messages
            .first()
            .and_then(|message| message.get("behavior"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if behavior == "provider-error" {
            return Err("synthetic provider failure".to_owned());
        }
        let (summary, cost, segment) = match behavior {
            "empty" => (String::new(), 0, None),
            "unbillable" => ("synthetic summary".to_owned(), 0, None),
            _ => (
                "synthetic summary".to_owned(),
                10,
                Some(json!({
                    "kind": "summary",
                    "model": "synthetic/model",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "source": "synthetic",
                    "metadata": {"compaction_result": "synthetic summary"}
                })),
            ),
        };
        Ok(CompactionProviderResult {
            summary,
            cost_usd_micros: cost,
            billing_segment: segment,
        })
    }
}

#[derive(Default)]
struct Billing {
    settled: Vec<String>,
    recorded: Vec<String>,
    incompatible: Vec<String>,
}

impl CompactionBilling for Billing {
    type Error = String;

    fn is_settled(&mut self, job: &CompactionJobRecord) -> Result<bool, Self::Error> {
        if job.chat_id.contains("check-error") {
            Err("synthetic settlement check failure".to_owned())
        } else {
            Ok(job.chat_id.contains("already-settled"))
        }
    }

    fn list_provider_segments(
        &mut self,
        _user_id: i64,
        operation_id: &str,
    ) -> Result<Vec<Value>, Self::Error> {
        if operation_id.contains("segments-error") {
            Err("synthetic segment read failure".to_owned())
        } else if operation_id.contains("restore") {
            Ok(vec![json!({
                "kind": "summary",
                "model": "synthetic/model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "source": "synthetic",
                "metadata": {"compaction_result": "restored summary"}
            })])
        } else {
            Ok(Vec::new())
        }
    }

    fn record_provider_segment(
        &mut self,
        job: &CompactionJobRecord,
        _operation_id: &str,
        _segment: &Value,
    ) -> Result<(), Self::Error> {
        if job.chat_id.contains("record-error") {
            Err("synthetic segment write failure".to_owned())
        } else {
            self.recorded.push(job.chat_id.clone());
            Ok(())
        }
    }

    fn settle(&mut self, request: SettlementRequest<'_>) -> Result<(), Self::Error> {
        if request.job.chat_id.contains("settle-error") {
            Err("synthetic settle failure".to_owned())
        } else {
            self.settled.push(request.job.chat_id.clone());
            Ok(())
        }
    }

    fn settle_incompatible(
        &mut self,
        chat_id: &str,
        _decoded: &Value,
    ) -> Result<bool, Self::Error> {
        self.incompatible.push(chat_id.to_owned());
        if chat_id.contains("incompatible-error") {
            Err("synthetic incompatible settlement failure".to_owned())
        } else {
            Ok(!chat_id.contains("incompatible-false"))
        }
    }
}

fn compaction_job(chat_id: &str) -> CompactionJobRecord {
    let behavior = chat_id
        .split_once(':')
        .map_or("success", |(_, behavior)| behavior);
    CompactionJobRecord {
        schema_version: COMPACTION_JOB_SCHEMA_VERSION,
        chat_id: chat_id.to_owned(),
        messages: vec![json!({"behavior": behavior, "text": "synthetic message"})],
        prior_summary: None,
        expected_marker: None,
        target_marker: "target".to_owned(),
        reservation: json!({"operation_id": format!("operation-{chat_id}")}),
        user_id: 7,
        message_id: Some("8".to_owned()),
        locale: "en".to_owned(),
        attempts: 0,
        next_attempt_at: 0.0,
        result_summary: None,
        result_cost_usd_micros: 0,
        result_billing_segment: None,
    }
}

fn queue_job(job: CompactionJobRecord) -> QueueJob {
    QueueJob {
        chat_id: job.chat_id.clone(),
        payload: serde_json::to_string(&job).unwrap_or_else(|_| unreachable!()),
    }
}

#[test]
fn compaction_worker_exercises_durable_failure_and_recovery_boundaries() {
    let invalid = CompactionWorker::new(
        Queue::default(),
        State::default(),
        Provider,
        Billing::default(),
        || "token".to_owned(),
    )
    .with_lock_ttl_seconds(0)
    .run_once(100.0);
    assert!(invalid.is_err());

    let list_error = CompactionWorker::new(
        Queue {
            jobs: vec![QueueJob {
                chat_id: "list-error".to_owned(),
                payload: String::new(),
            }],
            ..Queue::default()
        },
        State::default(),
        Provider,
        Billing::default(),
        || "token".to_owned(),
    )
    .run_once(100.0);
    assert!(list_error.is_err());

    let mut jobs = vec![
        QueueJob {
            chat_id: "bad-quarantine-ok".to_owned(),
            payload: "not-json".to_owned(),
        },
        QueueJob {
            chat_id: "bad-quarantine-false".to_owned(),
            payload: "not-json".to_owned(),
        },
        QueueJob {
            chat_id: "bad-quarantine-error".to_owned(),
            payload: "not-json".to_owned(),
        },
        QueueJob {
            chat_id: "incompatible-false".to_owned(),
            payload: json!({"unexpected": true}).to_string(),
        },
        QueueJob {
            chat_id: "incompatible-error".to_owned(),
            payload: json!({"unexpected": true}).to_string(),
        },
        QueueJob {
            chat_id: "incompatible-delete-error".to_owned(),
            payload: json!({"unexpected": true}).to_string(),
        },
    ];
    let mut future = compaction_job("future");
    future.next_attempt_at = 200.0;
    jobs.push(queue_job(future));
    for id in [
        "lock-busy",
        "lock-error",
        "already-settled-delete-error",
        "check-error",
        "load-error",
        "provider-error:provider-error",
        "provider-empty:empty",
        "provider-unbillable:unbillable",
        "record-error",
        "replace-error",
        "save-error",
        "settle-error",
        "release-error",
        "success",
        "recovered",
        "obsolete",
    ] {
        jobs.push(queue_job(compaction_job(id)));
    }
    let mut terminal = compaction_job("terminal-provider-error:provider-error");
    terminal.attempts = 2;
    jobs.push(queue_job(terminal));

    let mut worker = CompactionWorker::new(
        Queue {
            jobs,
            ..Queue::default()
        },
        State::default(),
        Provider,
        Billing::default(),
        || "synthetic-token".to_owned(),
    );
    let report = worker.run_once(100.0).unwrap_or_else(|_| unreachable!());
    assert!(report.completed >= 4);
    assert!(report.retried >= 4);
    assert_eq!(report.quarantined, 1);
    assert_eq!(report.skipped_not_due, 1);
    assert_eq!(report.skipped_locked, 1);
    let stages = report
        .failures
        .iter()
        .map(|failure| failure.stage)
        .collect::<HashSet<_>>();
    for stage in [
        "quarantine",
        "incompatible",
        "settle_incompatible",
        "delete_incompatible",
        "acquire_lock",
        "delete_completed",
        "schedule_retry",
        "release_lock",
    ] {
        assert!(stages.contains(stage), "missing stage {stage}: {stages:?}");
    }
    let (queue, state, _provider, billing, _token) = worker.into_parts();
    assert!(queue.quarantined.contains(&"bad-quarantine-ok".to_owned()));
    assert!(state.saved.contains(&"success".to_owned()));
    assert!(billing.settled.iter().any(|chat| chat == "success"));
}

struct Enqueue {
    exists: bool,
    inserted: Result<bool, &'static str>,
    payloads: Vec<String>,
}

impl Default for Enqueue {
    fn default() -> Self {
        Self {
            exists: false,
            inserted: Ok(true),
            payloads: Vec::new(),
        }
    }
}

impl CompactionEnqueueStore for Enqueue {
    type Error = &'static str;

    fn job_exists(&mut self, _chat_id: &str) -> Result<bool, Self::Error> {
        Ok(self.exists)
    }

    fn insert_job(&mut self, _chat_id: &str, payload: &str) -> Result<bool, Self::Error> {
        self.payloads.push(payload.to_owned());
        self.inserted
    }
}

#[derive(Default)]
struct Reservation {
    value: Option<Value>,
    reserve_error: bool,
    refund_error: bool,
    refunds: usize,
}

impl CompactionReservationStore for Reservation {
    type Error = &'static str;

    fn reserve(
        &mut self,
        _context: CompactionScheduleContext,
        _usage_tag: &str,
        _reserve_credit_units: i64,
        _target_marker: &str,
        _message_count: usize,
    ) -> Result<Option<Value>, Self::Error> {
        if self.reserve_error {
            Err("synthetic reserve failure")
        } else {
            Ok(self.value.clone())
        }
    }

    fn refund_enqueue_failure(
        &mut self,
        _user_id: i64,
        _reservation: &Value,
    ) -> Result<(), Self::Error> {
        self.refunds += 1;
        if self.refund_error {
            Err("synthetic refund failure")
        } else {
            Ok(())
        }
    }
}

fn plan() -> MemoryCompactionPlan {
    MemoryCompactionPlan {
        chat_id: "synthetic-chat".to_owned(),
        messages: vec![json!({"role": "user", "text": "synthetic message"})],
        prior_summary: Some("synthetic prior summary".to_owned()),
        expected_marker: Some("before".to_owned()),
        target_marker: "after".to_owned(),
    }
}

fn schedule_context() -> CompactionScheduleContext {
    CompactionScheduleContext {
        user_id: 7,
        group_chat_id: Some(-100),
        origin_chat_id: -100,
        message_id: 8,
        locale: Locale::En,
        payer_source: Some(PayerSource::Chat),
    }
}

#[test]
fn compaction_scheduler_refunds_every_failed_enqueue_shape() {
    let cases = [
        (false, Ok(true), false, true),
        (false, Ok(false), false, false),
        (false, Err("synthetic enqueue failure"), false, false),
        (false, Err("synthetic enqueue failure"), true, false),
    ];
    for (exists, inserted, refund_error, expected_success) in cases {
        let queue = Enqueue {
            exists,
            inserted,
            ..Enqueue::default()
        };
        let billing = Reservation {
            value: Some(json!({"operation_id": "synthetic-operation"})),
            refund_error,
            ..Reservation::default()
        };
        let mut scheduler = NativeCompactionScheduler::new(
            queue,
            billing,
            || "synthetic-token".to_owned(),
            TEST_MODEL,
            "synthetic system prompt",
        );
        let result = scheduler.schedule(plan(), schedule_context());
        if expected_success {
            assert_eq!(result, Ok(true));
        } else if inserted == Ok(false) {
            assert_eq!(result, Ok(false));
        } else {
            assert!(result.is_err());
        }
        let (queue, billing, _token) = scheduler.into_parts();
        if inserted != Ok(true) {
            assert_eq!(billing.refunds, 1);
        }
        if let Some(payload) = queue.payloads.first() {
            let decoded = serde_json::from_str::<Value>(payload).unwrap_or(Value::Null);
            assert_eq!(decoded["locale"], "en");
            assert_eq!(decoded["reservation"]["credit_scale"], 100);
        }
    }

    let mut existing = NativeCompactionScheduler::new(
        Enqueue {
            exists: true,
            ..Enqueue::default()
        },
        Reservation::default(),
        || "unused".to_owned(),
        TEST_MODEL,
        "synthetic system prompt",
    );
    assert_eq!(existing.schedule(plan(), schedule_context()), Ok(false));

    let mut denied = NativeCompactionScheduler::new(
        Enqueue::default(),
        Reservation::default(),
        || "unused".to_owned(),
        TEST_MODEL,
        "synthetic system prompt",
    );
    assert_eq!(denied.schedule(plan(), schedule_context()), Ok(false));
}

#[derive(Clone, Default)]
struct DurableQueue {
    state: Arc<Mutex<DurableQueueState>>,
}

#[derive(Default)]
struct DurableQueueState {
    queued: Vec<QueuedUpdate>,
    inserted: Vec<i64>,
    replaced: Vec<i64>,
    deleted: Vec<i64>,
    quarantined: Vec<i64>,
    fail: Option<&'static str>,
}

impl DurableUpdateQueue for DurableQueue {
    type Error = &'static str;

    fn insert_update(&self, update_id: i64, _payload: &str) -> Result<bool, Self::Error> {
        let mut state = self.state.lock().map_err(|_| "synthetic lock failure")?;
        if state.fail == Some("insert") {
            return Err("synthetic insert failure");
        }
        state.inserted.push(update_id);
        Ok(true)
    }

    fn list_updates(&self) -> Result<Vec<QueuedUpdate>, Self::Error> {
        let state = self.state.lock().map_err(|_| "synthetic lock failure")?;
        if state.fail == Some("list") {
            return Err("synthetic list failure");
        }
        Ok(state.queued.clone())
    }

    fn replace_update(&self, update_id: i64, _payload: &str) -> Result<(), Self::Error> {
        let mut state = self.state.lock().map_err(|_| "synthetic lock failure")?;
        if state.fail == Some("replace") {
            return Err("synthetic replace failure");
        }
        state.replaced.push(update_id);
        Ok(())
    }

    fn delete_update(&self, update_id: i64) -> Result<bool, Self::Error> {
        let mut state = self.state.lock().map_err(|_| "synthetic lock failure")?;
        if state.fail == Some("delete") {
            return Err("synthetic delete failure");
        }
        state.deleted.push(update_id);
        Ok(true)
    }

    fn quarantine_update(&self, update_id: i64, _payload: &str) -> Result<(), Self::Error> {
        let mut state = self.state.lock().map_err(|_| "synthetic lock failure")?;
        if state.fail == Some("quarantine") {
            return Err("synthetic quarantine failure");
        }
        state.quarantined.push(update_id);
        Ok(())
    }
}

struct Handler {
    fail: bool,
}

impl UpdateHandler for Handler {
    type Error = &'static str;

    fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
        if self.fail {
            Err("synthetic handler failure")
        } else {
            Ok(())
        }
    }
}

fn incoming(update_id: i64) -> IncomingUpdate {
    IncomingUpdate {
        update_id,
        event: IncomingEvent::Unsupported,
    }
}

#[test]
fn durable_parallel_handler_surfaces_queue_and_recovery_failures() {
    assert!(matches!(
        DurableParallelUpdateHandler::<Handler, DurableQueue>::start(
            0,
            1,
            DurableQueue::default(),
            || Ok::<_, Infallible>(Handler { fail: false }),
        ),
        Err(ParallelHandlerBuildError::NoWorkers)
    ));
    assert!(matches!(
        DurableParallelUpdateHandler::<Handler, DurableQueue>::start(
            1,
            0,
            DurableQueue::default(),
            || Ok::<_, Infallible>(Handler { fail: false }),
        ),
        Err(ParallelHandlerBuildError::EmptyQueue)
    ));
    assert!(matches!(
        DurableParallelUpdateHandler::<Handler, DurableQueue>::start(
            1,
            1,
            DurableQueue::default(),
            || Err::<Handler, _>("synthetic startup failure"),
        ),
        Err(ParallelHandlerBuildError::WorkerStartup { .. })
    ));

    let malformed = DurableQueue::default();
    malformed
        .state
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .queued = vec![
        QueuedUpdate {
            update_id: 41,
            payload: "not-json".to_owned(),
        },
        QueuedUpdate {
            update_id: 42,
            payload: json!({
                "schema_version": 99,
                "update": incoming(42),
                "attempts": 0,
                "completed": false
            })
            .to_string(),
        },
        QueuedUpdate {
            update_id: 43,
            payload: json!({
                "schema_version": 1,
                "update": incoming(999),
                "attempts": 0,
                "completed": false
            })
            .to_string(),
        },
    ];
    let mut handler = DurableParallelUpdateHandler::start(1, 4, malformed.clone(), || {
        Ok::<_, Infallible>(Handler { fail: false })
    })
    .unwrap_or_else(|_| unreachable!());
    assert!(handler.prepare().is_ok());
    let failures = handler.take_background_failures();
    assert_eq!(failures.quarantined.len(), 3);
    assert_eq!(
        malformed
            .state
            .lock()
            .unwrap_or_else(|_| unreachable!())
            .quarantined
            .len(),
        3
    );

    let queue = DurableQueue::default();
    let mut handler = DurableParallelUpdateHandler::start(1, 4, queue.clone(), || {
        Ok::<_, Infallible>(Handler { fail: false })
    })
    .unwrap_or_else(|_| unreachable!());
    assert!(handler.handle(incoming(50)).is_ok());
    for _ in 0..100 {
        let failures = handler.take_background_failures();
        if failures.fatal.is_some()
            || queue
                .state
                .lock()
                .is_ok_and(|state| !state.replaced.is_empty())
        {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert!(handler.confirm_updates(UpdateConfirmation::All).is_ok());
    assert!(
        queue
            .state
            .lock()
            .is_ok_and(|state| state.deleted.contains(&50))
    );

    for failure in ["list", "insert"] {
        let queue = DurableQueue::default();
        queue.state.lock().unwrap_or_else(|_| unreachable!()).fail = Some(failure);
        let mut handler = DurableParallelUpdateHandler::start(1, 2, queue, || {
            Ok::<_, Infallible>(Handler { fail: false })
        })
        .unwrap_or_else(|_| unreachable!());
        if failure == "list" {
            assert!(handler.prepare().is_err());
        } else {
            assert!(handler.handle(incoming(60)).is_err());
        }
    }
}
