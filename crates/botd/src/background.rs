//! Interruptible lifecycle for native background services.

use std::fmt::Display;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use thiserror::Error;

use crate::compaction_adapters::{ProductionCompactionWorker, production_compaction_worker};
use crate::compaction_worker::{
    CompactionBilling, CompactionProvider, CompactionQueue, CompactionState, CompactionWorker,
};
use crate::composition::TelegramActionSink;
use crate::composition::TelegramDeliveryCoordinator;
use crate::lightning_payments::{LightningPaymentWorker, OpenNodeOptions, opennode_provider};
use crate::operational_reporting::{OperationalReport, OperationalReporter};
use crate::price_refresh::production_price_refresh_worker;
use crate::reconciliation::{
    ActiveOperationRegistry, AiBillingReconciler, GenerationSource, ReconciliationSettings,
    ReconciliationStore, production_reconciler,
};
use crate::scheduler::SchedulerMode;
use crate::scheduler::{ScheduledTaskExecutor, SchedulerStep, SchedulerStore, TaskScheduler};
use crate::task_service::{TaskServiceOptions, build_task_scheduler};
use bot_adapters::billing_read::BillingRepository;
use bot_adapters::openrouter_chat::OpenRouterPricingCache;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::telegram_http::ReqwestTelegramTransport;

const TASK_INTERVAL: Duration = Duration::from_secs(1);
const TASK_WORKER_COUNT: usize = 4;
const COMPACTION_INTERVAL: Duration = Duration::from_secs(2);
const LIGHTNING_POLL_INTERVAL: Duration = Duration::from_secs(5);
const PRICE_REFRESH_INTERVAL: Duration = Duration::from_secs(30 * 60);
const REPEATED_FAILURE_REPORT_INTERVAL: Duration = Duration::from_secs(15 * 60);

pub struct ProductionBackgroundOptions<'a> {
    pub redis_endpoint: &'a RedisEndpoint,
    pub database_url: &'a str,
    pub telegram_token: &'a str,
    pub openrouter_api_key: &'a str,
    pub openrouter_base_url: &'a str,
    pub openrouter_pricing: Arc<OpenRouterPricingCache>,
    pub firecrawl_api_key: Option<&'a str>,
    pub system_prompt: &'a str,
    pub owner_token: &'a str,
    pub scheduler_mode: SchedulerMode,
    pub reconciliation_interval: Duration,
    pub reconciliation_settings: ReconciliationSettings,
    pub active_operations: ActiveOperationRegistry,
    pub coinmarketcap_key: Option<&'a str>,
    pub opennode: Option<OpenNodeOptions>,
    pub telegram_delivery: TelegramDeliveryCoordinator,
}

pub fn build_production_background_specs(
    options: ProductionBackgroundOptions<'_>,
) -> Result<Vec<BackgroundWorkerSpec>, String> {
    let mut task_workers = Vec::with_capacity(TASK_WORKER_COUNT);
    for worker in 0..TASK_WORKER_COUNT {
        let scheduler = build_task_scheduler(TaskServiceOptions {
            redis_endpoint: options.redis_endpoint,
            database_url: options.database_url,
            telegram_token: options.telegram_token,
            openrouter_api_key: options.openrouter_api_key,
            openrouter_base_url: options.openrouter_base_url,
            openrouter_pricing: Arc::clone(&options.openrouter_pricing),
            firecrawl_api_key: options.firecrawl_api_key,
            system_prompt: options.system_prompt,
            owner_token: options.owner_token,
            mode: options.scheduler_mode,
            telegram_delivery: options.telegram_delivery.clone(),
        })
        .map_err(|error| error.to_string())?
        .with_claim_token(format!("{}:worker-{worker}", options.owner_token));
        task_workers.push(BackgroundWorkerSpec::new(
            format!("task-scheduler-{}", worker + 1),
            TASK_INTERVAL,
            Box::new(scheduler),
        ));
    }
    let compaction = compaction_worker(&options)?;
    let reconciliation = production_reconciler(
        options.database_url,
        options.openrouter_api_key,
        options.openrouter_base_url,
        options.active_operations,
        options.reconciliation_settings,
    )?;
    let price_refresh =
        production_price_refresh_worker(options.redis_endpoint, options.coinmarketcap_key)?;
    task_workers.extend([
        BackgroundWorkerSpec::new(
            "memory-compaction",
            COMPACTION_INTERVAL,
            Box::new(compaction),
        ),
        BackgroundWorkerSpec::new(
            "ai-billing-reconciliation",
            options.reconciliation_interval,
            Box::new(reconciliation),
        ),
        BackgroundWorkerSpec::new(
            "price-cache-refresh",
            PRICE_REFRESH_INTERVAL,
            Box::new(price_refresh),
        ),
    ]);
    if let Some(opennode) = &options.opennode {
        let transport = ReqwestTelegramTransport::new()
            .map_err(|error| format!("could not construct Telegram transport: {error:?}"))?;
        task_workers.push(BackgroundWorkerSpec::new(
            "lightning-payments",
            LIGHTNING_POLL_INTERVAL,
            Box::new(LightningPaymentWorker::new(
                opennode_provider(&opennode.api_url, &opennode.api_key)?,
                BillingRepository::new(options.database_url),
                TelegramActionSink::new(transport, options.telegram_token)
                    .with_delivery_coordinator(options.telegram_delivery.clone()),
            )),
        ));
    }
    Ok(task_workers)
}

fn compaction_worker(
    options: &ProductionBackgroundOptions<'_>,
) -> Result<ProductionCompactionWorker, String> {
    production_compaction_worker(
        options.redis_endpoint,
        options.database_url,
        options.openrouter_api_key,
        options.openrouter_base_url,
        Arc::clone(&options.openrouter_pricing),
        options.system_prompt,
        options.owner_token,
    )
}

pub trait BackgroundWorker: Send + 'static {
    fn run_once(&mut self, now_epoch_seconds: i64) -> Result<(), String>;

    fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
}

pub struct BackgroundWorkerSpec {
    name: String,
    interval: Duration,
    worker: Box<dyn BackgroundWorker>,
}

impl BackgroundWorkerSpec {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        interval: Duration,
        worker: Box<dyn BackgroundWorker>,
    ) -> Self {
        Self {
            name: name.into(),
            interval,
            worker,
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum BackgroundError {
    #[error("background worker {name} has a zero interval")]
    InvalidInterval { name: String },
    #[error("could not start background worker {name}: {error}")]
    Spawn { name: String, error: String },
    #[error("background worker {name} panicked")]
    Panicked { name: String },
}

impl BackgroundError {
    fn operational_report(&self) -> OperationalReport {
        match self {
            Self::InvalidInterval { name } => OperationalReport::new(
                format!("el proceso en segundo plano {name} tiene un intervalo de cero"),
                self.to_string(),
            ),
            Self::Spawn { name, error } => OperationalReport::new(
                format!("no se pudo iniciar el proceso en segundo plano {name}: {error}"),
                self.to_string(),
            ),
            Self::Panicked { name } => OperationalReport::new(
                format!("el proceso en segundo plano {name} entró en pánico"),
                self.to_string(),
            ),
        }
    }
}

type WorkerBody = Box<dyn FnOnce() + Send + 'static>;

fn spawn_named_thread(name: String, body: WorkerBody) -> std::io::Result<JoinHandle<()>> {
    thread::Builder::new().name(name).spawn(body)
}

struct WorkerHandle {
    name: String,
    handle: JoinHandle<()>,
}

pub struct BackgroundSupervisor {
    stopping: Arc<AtomicBool>,
    wake: Arc<(Mutex<()>, Condvar)>,
    handles: Vec<WorkerHandle>,
    reporter: Arc<dyn OperationalReporter>,
    failure: Arc<Mutex<Option<BackgroundError>>>,
}

impl BackgroundSupervisor {
    pub fn start(
        specs: Vec<BackgroundWorkerSpec>,
        reporter: Arc<dyn OperationalReporter>,
    ) -> Result<Self, BackgroundError> {
        Self::start_with_spawner(specs, reporter, &mut spawn_named_thread)
    }

    fn start_with_spawner(
        specs: Vec<BackgroundWorkerSpec>,
        reporter: Arc<dyn OperationalReporter>,
        spawn: &mut dyn FnMut(String, WorkerBody) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Self, BackgroundError> {
        let stopping = Arc::new(AtomicBool::new(false));
        let wake = Arc::new((Mutex::new(()), Condvar::new()));
        let mut supervisor = Self {
            stopping,
            wake,
            handles: Vec::with_capacity(specs.len()),
            reporter,
            failure: Arc::new(Mutex::new(None)),
        };
        for spec in specs {
            if spec.interval.is_zero() {
                supervisor.stop_best_effort();
                let failure = BackgroundError::InvalidInterval { name: spec.name };
                supervisor.report_failure(&failure);
                return Err(failure);
            }
            let name = spec.name.clone();
            let thread_name = name.clone();
            let stopping = supervisor.stopping.clone();
            let wake = supervisor.wake.clone();
            let reporter = supervisor.reporter.clone();
            let failure = supervisor.failure.clone();
            let mut worker = spec.worker;
            let interval = spec.interval;
            let handle = spawn(
                thread_name,
                Box::new(move || {
                    let mut last_reported_failure: Option<(String, Instant)> = None;
                    while !stopping.load(Ordering::Acquire) {
                        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            worker.run_once(now_epoch_seconds())
                        }));
                        let run = match run {
                            Ok(run) => run,
                            Err(_) => {
                                let fatal = BackgroundError::Panicked { name: name.clone() };
                                if let Ok(mut recorded) = failure.lock() {
                                    *recorded = Some(fatal.clone());
                                }
                                let report = fatal.operational_report();
                                eprintln!("{fatal}");
                                if let Err(report_error) = reporter.report(&report) {
                                    eprintln!(
                                        "could not deliver background panic report: {report_error}"
                                    );
                                }
                                stopping.store(true, Ordering::Release);
                                wake.1.notify_all();
                                break;
                            }
                        };
                        match run {
                            Ok(()) => last_reported_failure = None,
                            Err(error) => {
                                let report = OperationalReport::new(
                                    format!("falló el proceso en segundo plano {name}: {error}"),
                                    format!("background worker {name} failed: {error}"),
                                );
                                let message = report.english().to_owned();
                                eprintln!("{message}");
                                let now = Instant::now();
                                let should_report = last_reported_failure.as_ref().is_none_or(
                                    |(previous, reported_at)| {
                                        previous != &message
                                            || now.duration_since(*reported_at)
                                                >= REPEATED_FAILURE_REPORT_INTERVAL
                                    },
                                );
                                if should_report {
                                    if let Err(report_error) = reporter.report(&report) {
                                        eprintln!(
                                            "could not deliver background failure report: {report_error}"
                                        );
                                    }
                                    last_reported_failure = Some((message, now));
                                }
                            }
                        }
                        // The wake mutex guards no data, so a poisoned lock
                        // carries no inconsistent state to protect.
                        let (lock, changed) = &*wake;
                        let guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
                        if stopping.load(Ordering::Acquire) {
                            break;
                        }
                        let _woken = changed.wait_timeout(guard, interval);
                    }
                    if let Err(error) = worker.shutdown() {
                        let report = OperationalReport::new(
                            format!(
                                "falló el apagado del proceso en segundo plano {name}: {error}"
                            ),
                            format!("background worker {name} shutdown failed: {error}"),
                        );
                        eprintln!("{}", report.english());
                        if let Err(report_error) = reporter.report(&report) {
                            eprintln!(
                                "could not deliver background shutdown failure report: {report_error}"
                            );
                        }
                    }
                }),
            );
            let handle = match handle {
                Ok(handle) => handle,
                Err(error) => {
                    supervisor.stop_best_effort();
                    let failure = BackgroundError::Spawn {
                        name: spec.name.clone(),
                        error: error.to_string(),
                    };
                    supervisor.report_failure(&failure);
                    return Err(failure);
                }
            };
            supervisor.handles.push(WorkerHandle {
                name: spec.name,
                handle,
            });
        }
        Ok(supervisor)
    }

    pub fn stop(&mut self) -> Result<(), BackgroundError> {
        self.stopping.store(true, Ordering::Release);
        self.wake.1.notify_all();
        while let Some(worker) = self.handles.pop() {
            if worker.handle.join().is_err() {
                let failure = BackgroundError::Panicked { name: worker.name };
                self.report_failure(&failure);
                return Err(failure);
            }
        }
        if let Ok(mut failure) = self.failure.lock()
            && let Some(failure) = failure.take()
        {
            return Err(failure);
        }
        Ok(())
    }

    #[must_use]
    pub fn has_failed(&self) -> bool {
        self.failure
            .lock()
            .map(|failure| failure.is_some())
            .unwrap_or(true)
    }

    fn stop_best_effort(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.wake.1.notify_all();
        while let Some(worker) = self.handles.pop() {
            if worker.handle.join().is_err() {
                self.report_failure(&BackgroundError::Panicked { name: worker.name });
            }
        }
    }

    fn report_failure(&self, failure: &BackgroundError) {
        eprintln!("{failure}");
        if let Err(report_error) = self.reporter.report(&failure.operational_report()) {
            eprintln!("could not deliver operational failure report: {report_error}");
        }
    }
}

impl Drop for BackgroundSupervisor {
    fn drop(&mut self) {
        self.stop_best_effort();
    }
}

impl<Store, Executor> BackgroundWorker for TaskScheduler<Store, Executor>
where
    Store: SchedulerStore + Send + 'static,
    Executor: ScheduledTaskExecutor + Send + 'static,
    Store::Error: Display,
    Executor::Error: Display,
{
    fn run_once(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        match self
            .step(now_epoch_seconds)
            .map_err(|error| error.to_string())?
        {
            SchedulerStep::Observed { failures, .. } if !failures.is_empty() => Err(failures
                .into_iter()
                .map(|failure| {
                    format!(
                        "task {} failed at {}: {}",
                        failure.task_id, failure.stage, failure.error
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")),
            SchedulerStep::NotOwner | SchedulerStep::Observed { .. } => Ok(()),
        }
    }

    fn shutdown(&mut self) -> Result<(), String> {
        TaskScheduler::shutdown(self)
            .map(|_released| ())
            .map_err(|error| error.to_string())
    }
}

impl<Queue, State, Provider, Billing, Token> BackgroundWorker
    for CompactionWorker<Queue, State, Provider, Billing, Token>
where
    Queue: CompactionQueue + Send + 'static,
    State: CompactionState + Send + 'static,
    Provider: CompactionProvider + Send + 'static,
    Billing: CompactionBilling + Send + 'static,
    Token: FnMut() -> String + Send + 'static,
    Queue::Error: Display,
    State::Error: Display,
    Provider::Error: Display,
    Billing::Error: Display,
{
    fn run_once(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        let report = CompactionWorker::run_once(self, now_epoch_seconds as f64)
            .map_err(|error| error.to_string())?;
        if report.failures.is_empty() {
            Ok(())
        } else {
            Err(report
                .failures
                .into_iter()
                .map(|failure| {
                    format!(
                        "chat {} failed at {}: {}",
                        failure.chat_id, failure.stage, failure.error
                    )
                })
                .collect::<Vec<_>>()
                .join("; "))
        }
    }
}

impl<Store, Generations> BackgroundWorker for AiBillingReconciler<Store, Generations>
where
    Store: ReconciliationStore + Send + 'static,
    Generations: GenerationSource + Send + 'static,
    Store::Error: Display,
    Generations::Error: Display,
{
    fn run_once(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
        let report = AiBillingReconciler::run_once(self, now_epoch_seconds)?;
        if report.failures.is_empty() {
            Ok(())
        } else {
            Err(report
                .failures
                .into_iter()
                .map(|failure| format!("operation {}: {}", failure.operation_id, failure.error))
                .collect::<Vec<_>>()
                .join("; "))
        }
    }
}

fn now_epoch_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64)
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::sync::{Arc, Mutex, PoisonError};

    use bot_adapters::openrouter_chat::OpenRouterPricingCache;
    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_core::locale::Locale;

    use super::{
        BackgroundError, BackgroundSupervisor, BackgroundWorker, BackgroundWorkerSpec,
        ProductionBackgroundOptions, build_production_background_specs, spawn_named_thread,
    };
    use crate::composition::TelegramDeliveryCoordinator;
    use crate::lightning_payments::OpenNodeOptions;
    use crate::operational_reporting::{OperationalReport, OperationalReporter};
    use crate::reconciliation::{ActiveOperationRegistry, ReconciliationSettings};
    use crate::scheduler::SchedulerMode;
    use std::time::Duration;

    struct Worker {
        ran: mpsc::Sender<i64>,
        stopped: mpsc::Sender<()>,
        fail: bool,
    }

    #[derive(Default)]
    struct Reporter {
        messages: Mutex<Vec<OperationalReport>>,
    }

    impl OperationalReporter for Reporter {
        fn report(&self, report: &OperationalReport) -> Result<(), String> {
            self.messages
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(report.clone());
            Ok(())
        }
    }

    impl BackgroundWorker for Worker {
        fn run_once(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
            let _ = self.ran.send(now_epoch_seconds);
            if self.fail {
                Err("synthetic run failure".to_owned())
            } else {
                Ok(())
            }
        }

        fn shutdown(&mut self) -> Result<(), String> {
            let _ = self.stopped.send(());
            Ok(())
        }
    }

    #[test]
    fn starts_immediately_repeats_failures_and_stops_interruptibly() -> TestResult {
        let (ran_tx, ran_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let reporter = Arc::new(Reporter::default());
        let started = BackgroundSupervisor::start(
            vec![BackgroundWorkerSpec::new(
                "synthetic-worker",
                Duration::from_millis(20),
                Box::new(Worker {
                    ran: ran_tx,
                    stopped: stopped_tx,
                    fail: true,
                }),
            )],
            reporter.clone(),
        );
        let mut supervisor = started?;
        assert!(ran_rx.recv_timeout(Duration::from_secs(1)).is_ok());
        assert!(ran_rx.recv_timeout(Duration::from_secs(1)).is_ok());
        assert!(supervisor.stop().is_ok());
        assert!(stopped_rx.recv_timeout(Duration::from_secs(1)).is_ok());
        assert_eq!(
            ran_rx.recv_timeout(Duration::from_millis(50)),
            Err(RecvTimeoutError::Disconnected)
        );
        let messages = reporter.messages.lock();
        assert!(messages.is_ok_and(|messages| {
            messages.len() == 1
                && messages[0]
                    .for_locale(Locale::Es)
                    .starts_with("falló el proceso en segundo plano synthetic-worker")
        }));
        assert!(supervisor.stop().is_ok());
        Ok(())
    }

    #[test]
    fn worker_panics_fail_the_live_supervisor_immediately() -> TestResult {
        struct PanicWorker;
        impl BackgroundWorker for PanicWorker {
            fn run_once(&mut self, _now_epoch_seconds: i64) -> Result<(), String> {
                panic!("synthetic worker panic")
            }
        }

        let reporter = Arc::new(Reporter::default());
        let started = BackgroundSupervisor::start(
            vec![BackgroundWorkerSpec::new(
                "panic-worker",
                Duration::from_secs(60),
                Box::new(PanicWorker),
            )],
            reporter,
        );
        let mut supervisor = started?;
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let mut failed = false;
        while !failed {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
            failed = supervisor.has_failed();
        }
        assert!(supervisor.has_failed());
        assert!(supervisor.stop().is_err());
        Ok(())
    }

    #[test]
    fn rejects_zero_intervals_before_starting_a_worker() {
        let (ran_tx, ran_rx) = mpsc::channel();
        let (stopped_tx, _stopped_rx) = mpsc::channel();
        let reporter = Arc::new(Reporter::default());
        let result = BackgroundSupervisor::start(
            vec![BackgroundWorkerSpec::new(
                "invalid-worker",
                Duration::ZERO,
                Box::new(Worker {
                    ran: ran_tx,
                    stopped: stopped_tx,
                    fail: false,
                }),
            )],
            reporter.clone(),
        );
        assert!(result.is_err());
        assert_eq!(
            ran_rx.recv_timeout(Duration::from_millis(20)),
            Err(RecvTimeoutError::Disconnected)
        );
        assert!(
            reporter
                .messages
                .lock()
                .is_ok_and(|messages| messages.len() == 1)
        );
    }

    #[test]
    fn spawn_failures_have_a_localized_operational_report() {
        let failure = BackgroundError::Spawn {
            name: "synthetic-worker".to_owned(),
            error: "synthetic spawn failure".to_owned(),
        };
        let report = failure.operational_report();
        assert!(report.for_locale(Locale::Es).contains("no se pudo iniciar"));
        assert!(report.english().contains("synthetic spawn failure"));
    }

    #[test]
    fn production_background_composition_does_not_start_or_contact_services() -> TestResult {
        let pricing = OpenRouterPricingCache::new(
            "synthetic-openrouter-key",
            "https://openrouter.example.test/api/v1",
        );
        let pricing = Arc::new(pricing?);
        let build = |openrouter_base_url| {
            build_production_background_specs(ProductionBackgroundOptions {
                redis_endpoint: &RedisEndpoint {
                    host: "synthetic.invalid".to_owned(),
                    port: 6379,
                    password: Some("synthetic-password".to_owned()),
                },
                database_url: "postgresql://synthetic.invalid/database",
                telegram_token: "synthetic-telegram-token",
                openrouter_api_key: "synthetic-openrouter-key",
                openrouter_base_url,
                openrouter_pricing: Arc::clone(&pricing),
                firecrawl_api_key: None,
                system_prompt: "synthetic persona",
                owner_token: "synthetic-owner",
                scheduler_mode: SchedulerMode::Authoritative,
                reconciliation_interval: Duration::from_secs(60),
                reconciliation_settings: ReconciliationSettings::default(),
                active_operations: ActiveOperationRegistry::default(),
                coinmarketcap_key: Some("synthetic-coinmarketcap-key"),
                opennode: Some(OpenNodeOptions {
                    api_key: "synthetic-opennode-key".to_owned(),
                    api_url: "https://opennode.example.test".to_owned(),
                }),
                telegram_delivery: TelegramDeliveryCoordinator::default(),
            })
        };
        assert!(build("not-a-url").is_err());
        let result = build("https://openrouter.example.test/api/v1");
        assert!(result.is_ok());
        assert_eq!(result.map(|specs| specs.len()), Ok(8));
        Ok(())
    }

    #[test]
    fn task_verifier_runs_through_the_background_worker_boundary() -> TestResult {
        std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .map_or(Ok(()), run_task_verifier)
    }

    fn run_task_verifier(port: u16) -> TestResult {
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or(String::from("127.0.0.1")),
            port,
            // Empty passwords are ignored by the Redis client.
            password: std::env::var("TEST_REDIS_PASSWORD").ok(),
        };
        let mut verifier =
            crate::task_service::build_task_verifier(&endpoint, "synthetic-background-verifier")?;
        BackgroundWorker::run_once(&mut verifier, 1_700_000_000)?;
        BackgroundWorker::shutdown(&mut verifier)?;
        Ok(())
    }

    struct FailingReporter {
        attempts: Mutex<usize>,
    }

    impl OperationalReporter for FailingReporter {
        fn report(&self, _report: &OperationalReport) -> Result<(), String> {
            if let Ok(mut attempts) = self.attempts.lock() {
                *attempts += 1;
            }
            Err("synthetic admin chat outage".to_owned())
        }
    }

    #[test]
    fn undeliverable_failure_reports_do_not_stop_the_worker() -> TestResult {
        let (ran_tx, ran_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let reporter = Arc::new(FailingReporter {
            attempts: Mutex::new(0),
        });
        let started = BackgroundSupervisor::start(
            vec![BackgroundWorkerSpec::new(
                "unreported-worker",
                Duration::from_millis(5),
                Box::new(Worker {
                    ran: ran_tx,
                    stopped: stopped_tx,
                    fail: true,
                }),
            )],
            reporter.clone(),
        );
        let mut supervisor = started?;
        for _ in 0..3 {
            assert!(ran_rx.recv_timeout(Duration::from_secs(1)).is_ok());
        }
        assert!(!supervisor.has_failed());
        assert!(supervisor.stop().is_ok());
        assert!(stopped_rx.recv_timeout(Duration::from_secs(1)).is_ok());
        // Identical failures are only reported once per repeat interval.
        assert!(
            reporter
                .attempts
                .lock()
                .is_ok_and(|attempts| *attempts == 1)
        );
        Ok(())
    }

    #[test]
    fn spawn_failures_stop_started_workers_and_report_the_failed_worker() {
        let (ran_tx, ran_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let (unused_ran_tx, _unused_ran_rx) = mpsc::channel();
        let (unused_stopped_tx, _unused_stopped_rx) = mpsc::channel();
        let reporter = Arc::new(Reporter::default());
        let mut spawned = Vec::new();
        let result = BackgroundSupervisor::start_with_spawner(
            vec![
                BackgroundWorkerSpec::new(
                    "healthy-worker",
                    Duration::from_secs(60),
                    Box::new(Worker {
                        ran: ran_tx,
                        stopped: stopped_tx,
                        fail: false,
                    }),
                ),
                BackgroundWorkerSpec::new(
                    "unspawnable-worker",
                    Duration::from_secs(60),
                    Box::new(Worker {
                        ran: unused_ran_tx,
                        stopped: unused_stopped_tx,
                        fail: false,
                    }),
                ),
            ],
            reporter.clone(),
            &mut |name, body| {
                spawned.push(name.clone());
                if spawned.len() == 1 {
                    return spawn_named_thread(name, body);
                }
                // Let the first worker complete a run before the second spawn fails.
                assert!(ran_rx.recv_timeout(Duration::from_secs(1)).is_ok());
                Err(std::io::Error::other("synthetic thread exhaustion"))
            },
        );
        assert!(matches!(
            &result,
            Err(BackgroundError::Spawn { name, error })
                if name == "unspawnable-worker" && error == "synthetic thread exhaustion"
        ));
        assert_eq!(spawned, ["healthy-worker", "unspawnable-worker"]);
        // The healthy worker was shut down before start returned.
        assert_eq!(stopped_rx.try_recv(), Ok(()));
        let messages = reporter
            .messages
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].for_locale(Locale::Es),
            "no se pudo iniciar el proceso en segundo plano unspawnable-worker: synthetic thread exhaustion"
        );
    }

    struct PanickingShutdownWorker {
        ran: mpsc::Sender<()>,
    }

    impl BackgroundWorker for PanickingShutdownWorker {
        fn run_once(&mut self, _now_epoch_seconds: i64) -> Result<(), String> {
            let _ = self.ran.send(());
            Ok(())
        }

        fn shutdown(&mut self) -> Result<(), String> {
            panic!("synthetic shutdown panic")
        }
    }

    #[test]
    fn worker_threads_that_panic_during_shutdown_fail_the_stop() -> TestResult {
        let reporter = Arc::new(Reporter::default());
        let (ran_tx, ran_rx) = mpsc::channel();
        let started = BackgroundSupervisor::start(
            vec![BackgroundWorkerSpec::new(
                "shutdown-panic-worker",
                Duration::from_secs(60),
                Box::new(PanickingShutdownWorker { ran: ran_tx }),
            )],
            reporter.clone(),
        );
        let mut supervisor = started?;
        assert_eq!(ran_rx.recv_timeout(Duration::from_secs(1)), Ok(()));
        assert!(!supervisor.has_failed());
        assert!(matches!(
            supervisor.stop(),
            Err(BackgroundError::Panicked { name }) if name == "shutdown-panic-worker"
        ));
        let messages = reporter
            .messages
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].english(),
            "background worker shutdown-panic-worker panicked"
        );
        assert_eq!(
            messages[0].for_locale(Locale::Es),
            "el proceso en segundo plano shutdown-panic-worker entró en pánico"
        );
        Ok(())
    }

    #[test]
    fn aborted_startup_reports_workers_that_panic_while_being_stopped() {
        let reporter = Arc::new(Reporter::default());
        let (ran_tx, _ran_rx) = mpsc::channel();
        let (stopped_tx, _stopped_rx) = mpsc::channel();
        let (panicking_ran_tx, _panicking_ran_rx) = mpsc::channel();
        let result = BackgroundSupervisor::start(
            vec![
                BackgroundWorkerSpec::new(
                    "shutdown-panic-worker",
                    Duration::from_secs(60),
                    Box::new(PanickingShutdownWorker {
                        ran: panicking_ran_tx,
                    }),
                ),
                BackgroundWorkerSpec::new(
                    "zero-interval-worker",
                    Duration::ZERO,
                    Box::new(Worker {
                        ran: ran_tx,
                        stopped: stopped_tx,
                        fail: false,
                    }),
                ),
            ],
            reporter.clone(),
        );
        assert!(matches!(
            result,
            Err(BackgroundError::InvalidInterval { name }) if name == "zero-interval-worker"
        ));
        let messages = reporter
            .messages
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let english = messages
            .iter()
            .map(|message| message.english().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            english,
            [
                "background worker shutdown-panic-worker panicked",
                "background worker zero-interval-worker has a zero interval",
            ]
        );
    }

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct FailingPanicReporter {
        attempts: Mutex<usize>,
    }

    impl OperationalReporter for FailingPanicReporter {
        fn report(&self, _report: &OperationalReport) -> Result<(), String> {
            *self.attempts.lock().unwrap_or_else(PoisonError::into_inner) += 1;
            Err("synthetic admin chat outage".to_owned())
        }
    }

    #[test]
    fn worker_panics_fail_the_supervisor_even_when_the_report_is_lost() -> TestResult {
        struct PanicWorker;
        impl BackgroundWorker for PanicWorker {
            fn run_once(&mut self, _now_epoch_seconds: i64) -> Result<(), String> {
                panic!("synthetic worker panic")
            }
        }

        let reporter = Arc::new(FailingPanicReporter {
            attempts: Mutex::new(0),
        });
        let started = BackgroundSupervisor::start(
            vec![BackgroundWorkerSpec::new(
                "unreported-panic-worker",
                Duration::from_secs(60),
                Box::new(PanicWorker),
            )],
            reporter.clone(),
        );
        let mut supervisor = started?;
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let mut failed = false;
        while !failed {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
            failed = supervisor.has_failed();
        }
        assert!(matches!(
            supervisor.stop(),
            Err(BackgroundError::Panicked { name }) if name == "unreported-panic-worker"
        ));
        assert_eq!(
            *reporter
                .attempts
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
            1
        );
        Ok(())
    }

    #[test]
    fn production_composition_rejects_an_empty_owner_token() -> TestResult {
        let pricing = OpenRouterPricingCache::new(
            "synthetic-openrouter-key",
            "https://openrouter.example.test/api/v1",
        );
        let pricing = Arc::new(pricing?);
        let result = build_production_background_specs(ProductionBackgroundOptions {
            redis_endpoint: &RedisEndpoint {
                host: "synthetic.invalid".to_owned(),
                port: 6379,
                password: None,
            },
            database_url: "postgresql://synthetic.invalid/database",
            telegram_token: "synthetic-telegram-token",
            openrouter_api_key: "synthetic-openrouter-key",
            openrouter_base_url: "https://openrouter.example.test/api/v1",
            openrouter_pricing: pricing,
            firecrawl_api_key: None,
            system_prompt: "synthetic persona",
            owner_token: "",
            scheduler_mode: SchedulerMode::Authoritative,
            reconciliation_interval: Duration::from_secs(60),
            reconciliation_settings: ReconciliationSettings::default(),
            active_operations: ActiveOperationRegistry::default(),
            coinmarketcap_key: None,
            opennode: None,
            telegram_delivery: TelegramDeliveryCoordinator::default(),
        });
        assert!(matches!(&result, Err(error) if error.contains("owner token")));
        Ok(())
    }
}
