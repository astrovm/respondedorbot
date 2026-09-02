//! Interruptible lifecycle for native background services.

use std::fmt::Display;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use thiserror::Error;

use crate::compaction_adapters::production_compaction_worker;
use crate::compaction_worker::{
    CompactionBilling, CompactionProvider, CompactionQueue, CompactionState, CompactionWorker,
};
use crate::composition::TelegramDeliveryCoordinator;
use crate::operational_reporting::{OperationalReport, OperationalReporter};
use crate::price_refresh::production_price_refresh_worker;
use crate::reconciliation::{
    ActiveOperationRegistry, AiBillingReconciler, GenerationSource, ReconciliationSettings,
    ReconciliationStore, production_reconciler,
};
use crate::scheduler::SchedulerMode;
use crate::scheduler::{ScheduledTaskExecutor, SchedulerStep, SchedulerStore, TaskScheduler};
use crate::task_service::{TaskServiceOptions, build_task_scheduler};
use bot_adapters::redis_connection::RedisEndpoint;

const TASK_INTERVAL: Duration = Duration::from_secs(1);
const TASK_WORKER_COUNT: usize = 4;
const COMPACTION_INTERVAL: Duration = Duration::from_secs(2);
const PRICE_REFRESH_INTERVAL: Duration = Duration::from_secs(30 * 60);
const REPEATED_FAILURE_REPORT_INTERVAL: Duration = Duration::from_secs(15 * 60);

pub struct ProductionBackgroundOptions<'a> {
    pub redis_endpoint: &'a RedisEndpoint,
    pub database_url: &'a str,
    pub telegram_token: &'a str,
    pub openrouter_api_key: &'a str,
    pub openrouter_base_url: &'a str,
    pub firecrawl_api_key: Option<&'a str>,
    pub system_prompt: &'a str,
    pub owner_token: &'a str,
    pub scheduler_mode: SchedulerMode,
    pub reconciliation_interval: Duration,
    pub reconciliation_settings: ReconciliationSettings,
    pub active_operations: ActiveOperationRegistry,
    pub coinmarketcap_key: Option<&'a str>,
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
    let compaction = production_compaction_worker(
        options.redis_endpoint,
        options.database_url,
        options.openrouter_api_key,
        options.openrouter_base_url,
        options.system_prompt,
        options.owner_token,
    )?;
    let reconciliation = production_reconciler(
        options.database_url,
        options.openrouter_api_key,
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
    Ok(task_workers)
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
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
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
                                    format!(
                                        "falló el proceso en segundo plano {name}: {error}"
                                    ),
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
                        let (lock, changed) = &*wake;
                        let Ok(guard) = lock.lock() else {
                            break;
                        };
                        if stopping.load(Ordering::Acquire) {
                            break;
                        }
                        if changed.wait_timeout(guard, interval).is_err() {
                            break;
                        }
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
                });
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
    use std::sync::{Arc, Mutex};

    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_core::locale::Locale;

    use super::{
        BackgroundError, BackgroundSupervisor, BackgroundWorker, BackgroundWorkerSpec,
        ProductionBackgroundOptions, build_production_background_specs,
    };
    use crate::composition::TelegramDeliveryCoordinator;
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
                .map_err(|_| "report lock poisoned".to_owned())?
                .push(report.clone());
            Ok(())
        }
    }

    impl BackgroundWorker for Worker {
        fn run_once(&mut self, now_epoch_seconds: i64) -> Result<(), String> {
            self.ran
                .send(now_epoch_seconds)
                .map_err(|error| error.to_string())?;
            if self.fail {
                Err("synthetic run failure".to_owned())
            } else {
                Ok(())
            }
        }

        fn shutdown(&mut self) -> Result<(), String> {
            self.stopped.send(()).map_err(|error| error.to_string())
        }
    }

    #[test]
    fn starts_immediately_repeats_failures_and_stops_interruptibly() {
        let (ran_tx, ran_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let reporter = Arc::new(Reporter::default());
        let mut supervisor = BackgroundSupervisor::start(
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
        assert!(supervisor.is_ok());
        let Some(supervisor) = supervisor.as_mut().ok() else {
            return;
        };
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
    }

    #[test]
    fn worker_panics_fail_the_live_supervisor_immediately() {
        struct PanicWorker;
        impl BackgroundWorker for PanicWorker {
            fn run_once(&mut self, _now_epoch_seconds: i64) -> Result<(), String> {
                panic!("synthetic worker panic")
            }
        }

        let reporter = Arc::new(Reporter::default());
        let mut supervisor = BackgroundSupervisor::start(
            vec![BackgroundWorkerSpec::new(
                "panic-worker",
                Duration::from_secs(60),
                Box::new(PanicWorker),
            )],
            reporter,
        )
        .unwrap_or_else(|error| panic!("supervisor startup: {error}"));
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !supervisor.has_failed() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(supervisor.has_failed());
        assert!(supervisor.stop().is_err());
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
    fn production_background_composition_does_not_start_or_contact_services() {
        let result = build_production_background_specs(ProductionBackgroundOptions {
            redis_endpoint: &RedisEndpoint {
                host: "synthetic.invalid".to_owned(),
                port: 6379,
                password: Some("synthetic-password".to_owned()),
            },
            database_url: "postgresql://synthetic.invalid/database",
            telegram_token: "synthetic-telegram-token",
            openrouter_api_key: "synthetic-openrouter-key",
            openrouter_base_url: "https://openrouter.example.test/api/v1",
            firecrawl_api_key: None,
            system_prompt: "synthetic persona",
            owner_token: "synthetic-owner",
            scheduler_mode: SchedulerMode::Authoritative,
            reconciliation_interval: Duration::from_secs(60),
            reconciliation_settings: ReconciliationSettings::default(),
            active_operations: ActiveOperationRegistry::default(),
            coinmarketcap_key: Some("synthetic-coinmarketcap-key"),
            telegram_delivery: TelegramDeliveryCoordinator::default(),
        });
        assert!(result.is_ok());
        assert_eq!(result.map(|specs| specs.len()), Ok(7));
    }

    #[test]
    fn task_verifier_runs_through_the_background_worker_boundary() -> Result<(), String> {
        let Some(port) = std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
        else {
            return Ok(());
        };
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
            port,
            password: std::env::var("TEST_REDIS_PASSWORD")
                .ok()
                .filter(|value| !value.is_empty()),
        };
        let mut verifier =
            crate::task_service::build_task_verifier(&endpoint, "synthetic-background-verifier")
                .map_err(|error| error.to_string())?;
        BackgroundWorker::run_once(&mut verifier, 1_700_000_000)?;
        BackgroundWorker::shutdown(&mut verifier)?;
        Ok(())
    }
}
