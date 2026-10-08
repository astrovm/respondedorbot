//! Long-poll state ownership and update dispatch.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use bot_adapters::redis_update_queue::{QueuedUpdate, RedisUpdateQueue};
use bot_adapters::telegram_polling::{
    IncomingUpdate, PollFailure, PollOutcome, PollingError, next_offset,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_UPDATE_ATTEMPTS: usize = 3;
/// Wait before each retry of a failed update, by the attempt that failed.
const UPDATE_RETRY_DELAYS: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(5)];

fn retry_delay(failed_attempt: usize) -> Option<Duration> {
    UPDATE_RETRY_DELAYS.get(failed_attempt).copied()
}
const DURABLE_UPDATE_SCHEMA_VERSION: u32 = 1;
const WORKER_STOPPED_DURING_STARTUP: &str = "worker stopped during startup";

pub trait UpdateSource {
    fn poll(&mut self, offset: Option<i64>) -> Result<PollOutcome, PollingError>;
}

pub trait UpdateHandler {
    type Error;

    fn prepare(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error>;

    fn error_disposition(&self, _error: &Self::Error) -> HandlerErrorDisposition {
        HandlerErrorDisposition::RetryUpdate
    }

    fn confirm_updates(&mut self, _confirmation: UpdateConfirmation) -> Result<(), Self::Error> {
        Ok(())
    }

    fn finish_batch(&mut self) -> Vec<UpdateFailure> {
        Vec::new()
    }

    fn take_background_failures(&mut self) -> BackgroundUpdateFailures {
        BackgroundUpdateFailures::default()
    }

    fn shutdown(&mut self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandlerErrorDisposition {
    RetryUpdate,
    /// The failure would repeat on every attempt, so quarantine the update
    /// right away instead of retrying it.
    DiscardUpdate,
    StopRuntime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateConfirmation {
    Before(i64),
    All,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateFailure {
    pub update_id: i64,
    pub error: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackgroundUpdateFailures {
    pub retrying: Vec<UpdateFailure>,
    pub quarantined: Vec<UpdateFailure>,
    pub fatal: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParallelHandlerError {
    #[error("parallel update queue is not available")]
    QueueUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParallelHandlerBuildError {
    #[error("parallel update processing requires at least one worker")]
    NoWorkers,
    #[error("parallel update processing requires a non-empty queue")]
    EmptyQueue,
    #[error("worker {worker} could not start: {error}")]
    WorkerStartup { worker: usize, error: String },
}

pub struct ParallelUpdateHandler<Handler> {
    updates: Option<SyncSender<IncomingUpdate>>,
    workers: Vec<JoinHandle<()>>,
    completions: Receiver<UpdateCompletion>,
    pending: Vec<i64>,
    handler: PhantomData<fn() -> Handler>,
}

struct UpdateCompletion {
    update_id: i64,
    error: Option<String>,
}

impl<Handler> ParallelUpdateHandler<Handler> {
    pub fn start<Factory, FactoryError>(
        worker_count: usize,
        queue_capacity: usize,
        factory: Factory,
    ) -> Result<Self, ParallelHandlerBuildError>
    where
        Handler: UpdateHandler + 'static,
        Handler::Error: Display,
        Factory: Fn() -> Result<Handler, FactoryError> + Send + Sync + 'static,
        FactoryError: Display,
    {
        if worker_count == 0 {
            return Err(ParallelHandlerBuildError::NoWorkers);
        }
        if queue_capacity == 0 {
            return Err(ParallelHandlerBuildError::EmptyQueue);
        }

        let (update_sender, update_receiver) = mpsc::sync_channel::<IncomingUpdate>(queue_capacity);
        let update_receiver = Arc::new(Mutex::new(update_receiver));
        let (startup_sender, startup_receiver) = mpsc::channel();
        let factory = Arc::new(factory);
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut workers = Vec::with_capacity(worker_count);

        for worker in 0..worker_count {
            let update_receiver = update_receiver.clone();
            let startup_sender = startup_sender.clone();
            let factory = factory.clone();
            let completion_sender = completion_sender.clone();
            workers.push(thread::spawn(move || {
                let mut handler = match panic::catch_unwind(AssertUnwindSafe(|| factory())) {
                    Ok(Ok(handler)) => {
                        let _ = startup_sender.send((worker, None));
                        handler
                    }
                    Ok(Err(error)) => {
                        let _ = startup_sender.send((worker, Some(error.to_string())));
                        return;
                    }
                    Err(_) => {
                        let _ = startup_sender
                            .send((worker, Some("worker factory panicked".to_owned())));
                        return;
                    }
                };
                loop {
                    let Ok(Ok(update)) = update_receiver.lock().map(|receiver| receiver.recv())
                    else {
                        return;
                    };
                    let update_id = update.update_id;
                    let (error, panicked) =
                        match panic::catch_unwind(AssertUnwindSafe(|| handler.handle(update))) {
                            Ok(result) => (result.err().map(|error| error.to_string()), false),
                            Err(_) => (Some("update handler panicked".to_owned()), true),
                        };
                    let _ = completion_sender.send(UpdateCompletion { update_id, error });
                    if panicked {
                        return;
                    }
                }
            }));
        }
        drop(startup_sender);
        drop(completion_sender);

        for _ in 0..worker_count {
            // Every worker reports once, so a closed channel is only defensive.
            let (worker, error) = startup_receiver.recv().unwrap_or((
                workers.len(),
                Some(WORKER_STOPPED_DURING_STARTUP.to_owned()),
            ));
            if let Some(error) = error {
                drop(update_sender);
                join_workers(&mut workers);
                return Err(ParallelHandlerBuildError::WorkerStartup { worker, error });
            }
        }

        Ok(Self {
            updates: Some(update_sender),
            workers,
            completions: completion_receiver,
            pending: Vec::new(),
            handler: PhantomData,
        })
    }

    fn stop(&mut self) {
        // Closing the queue lets workers drain every accepted update before they
        // exit. Telegram updates must never disappear during a graceful deploy.
        self.updates.take();
        join_workers(&mut self.workers);
    }
}

impl<Handler> UpdateHandler for ParallelUpdateHandler<Handler> {
    type Error = ParallelHandlerError;

    fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
        let update_id = update.update_id;
        self.updates
            .as_ref()
            .ok_or(ParallelHandlerError::QueueUnavailable)?
            .send(update)
            .map_err(|_| ParallelHandlerError::QueueUnavailable)?;
        self.pending.push(update_id);
        Ok(())
    }

    fn finish_batch(&mut self) -> Vec<UpdateFailure> {
        let expected = self.pending.len();
        let mut failures = Vec::new();
        for _ in 0..expected {
            match self.completions.recv() {
                Ok(completion) => {
                    if let Some(position) = self
                        .pending
                        .iter()
                        .position(|update_id| *update_id == completion.update_id)
                    {
                        self.pending.swap_remove(position);
                    }
                    if let Some(error) = completion.error {
                        failures.push(UpdateFailure {
                            update_id: completion.update_id,
                            error,
                        });
                    }
                }
                Err(_) => break,
            }
        }
        failures.extend(self.pending.drain(..).map(|update_id| UpdateFailure {
            update_id,
            error: "parallel update worker stopped before completing its batch".to_owned(),
        }));
        failures
    }

    fn shutdown(&mut self) {
        self.stop();
    }
}

impl<Handler> Drop for ParallelUpdateHandler<Handler> {
    fn drop(&mut self) {
        self.stop();
    }
}

pub trait DurableUpdateQueue: Clone + Send + 'static {
    type Error: Display;

    fn insert_update(&self, update_id: i64, payload: &str) -> Result<bool, Self::Error>;
    fn list_updates(&self) -> Result<Vec<QueuedUpdate>, Self::Error>;
    fn replace_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error>;
    fn delete_updates(&self, update_ids: &[i64]) -> Result<usize, Self::Error>;
    fn quarantine_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error>;
}

impl DurableUpdateQueue for RedisUpdateQueue {
    type Error = bot_adapters::redis_update_queue::RedisUpdateQueueError;

    fn insert_update(&self, update_id: i64, payload: &str) -> Result<bool, Self::Error> {
        Self::insert_update(self, update_id, payload)
    }

    fn list_updates(&self) -> Result<Vec<QueuedUpdate>, Self::Error> {
        Self::list_updates(self)
    }

    fn replace_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error> {
        Self::replace_update(self, update_id, payload)
    }

    fn delete_updates(&self, update_ids: &[i64]) -> Result<usize, Self::Error> {
        Self::delete_updates(self, update_ids)
    }

    fn quarantine_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error> {
        Self::quarantine_update(self, update_id, payload)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DurableUpdateRecord {
    schema_version: u32,
    update: IncomingUpdate,
    attempts: usize,
    #[serde(default)]
    completed: bool,
}

#[derive(Serialize)]
struct DeadUpdateRecord<'a> {
    schema_version: u32,
    update: &'a IncomingUpdate,
    attempts: usize,
    error: &'a str,
}

struct DurableUpdateCompletion {
    record: DurableUpdateRecord,
    error: Option<String>,
    // Retrying cannot help, so the update is quarantined on this attempt.
    permanent: bool,
    // The worker already put the retry back on the update queue, so the
    // polling thread only persists the attempt instead of resubmitting it.
    resubmitted: bool,
}

type SharedUpdateSender = Arc<Mutex<Option<SyncSender<DurableUpdateRecord>>>>;

/// Reports a worker's result and, for a retryable failure, resubmits the
/// update right away instead of waiting for the next long poll to return.
///
/// Every report is sent while holding the shared sender lock, so a failure's
/// report always reaches the polling thread before the report of its retry.
fn report_durable_completion(
    updates: &Mutex<Option<SyncSender<DurableUpdateRecord>>>,
    completions: &Sender<DurableUpdateCompletion>,
    mut completion: DurableUpdateCompletion,
) {
    let sender = updates.lock().unwrap_or_else(PoisonError::into_inner);
    let next_attempts = completion.record.attempts.saturating_add(1);
    if completion.error.is_some()
        && !completion.permanent
        && next_attempts < MAX_UPDATE_ATTEMPTS
        && let Some(sender) = sender.as_ref()
    {
        let mut retry = completion.record.clone();
        retry.attempts = next_attempts;
        // Never block a worker on a full queue: the polling thread resubmits
        // whatever could not be queued here.
        completion.resubmitted = sender.try_send(retry).is_ok();
    }
    let _ = completions.send(completion);
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DurableParallelHandlerError {
    #[error("parallel update queue is not available")]
    QueueUnavailable,
    #[error("durable update queue failed: {0}")]
    DurableQueue(String),
    /// Carries the full "could not encode durable update: ..." message.
    #[error("{0}")]
    Serialization(String),
}

pub struct DurableParallelUpdateHandler<Handler, Queue> {
    updates: SharedUpdateSender,
    workers: Vec<JoinHandle<()>>,
    completions: Receiver<DurableUpdateCompletion>,
    queue: Queue,
    active: HashSet<i64>,
    completed: HashSet<i64>,
    failures: BackgroundUpdateFailures,
    recovered: bool,
    handler: PhantomData<fn() -> Handler>,
}

impl<Handler, Queue> DurableParallelUpdateHandler<Handler, Queue>
where
    Handler: UpdateHandler + 'static,
    Handler::Error: Display,
    Queue: DurableUpdateQueue,
{
    pub fn start<Factory, FactoryError>(
        worker_count: usize,
        queue_capacity: usize,
        queue: Queue,
        factory: Factory,
    ) -> Result<Self, ParallelHandlerBuildError>
    where
        Factory: Fn() -> Result<Handler, FactoryError> + Send + Sync + 'static,
        FactoryError: Display,
    {
        if worker_count == 0 {
            return Err(ParallelHandlerBuildError::NoWorkers);
        }
        if queue_capacity == 0 {
            return Err(ParallelHandlerBuildError::EmptyQueue);
        }

        let (update_sender, update_receiver) =
            mpsc::sync_channel::<DurableUpdateRecord>(queue_capacity);
        let update_receiver = Arc::new(Mutex::new(update_receiver));
        let updates: SharedUpdateSender = Arc::new(Mutex::new(Some(update_sender)));
        let (startup_sender, startup_receiver) = mpsc::channel();
        let (completion_sender, completion_receiver) = mpsc::channel();
        let factory = Arc::new(factory);
        let mut workers = Vec::with_capacity(worker_count);

        for worker in 0..worker_count {
            let update_receiver = update_receiver.clone();
            let startup_sender = startup_sender.clone();
            let completion_sender = completion_sender.clone();
            let updates = Arc::clone(&updates);
            let factory = factory.clone();
            workers.push(thread::spawn(move || {
                let mut handler = match panic::catch_unwind(AssertUnwindSafe(|| factory())) {
                    Ok(Ok(handler)) => {
                        let _ = startup_sender.send((worker, None));
                        handler
                    }
                    Ok(Err(error)) => {
                        let _ = startup_sender.send((worker, Some(error.to_string())));
                        return;
                    }
                    Err(_) => {
                        let _ = startup_sender
                            .send((worker, Some("worker factory panicked".to_owned())));
                        return;
                    }
                };
                loop {
                    let Ok(Ok(record)) = update_receiver.lock().map(|receiver| receiver.recv())
                    else {
                        return;
                    };
                    let result = panic::catch_unwind(AssertUnwindSafe(|| {
                        handler.handle(record.update.clone())
                    }));
                    let (error, permanent, panicked) = match result {
                        Ok(Ok(())) => (None, false, false),
                        Ok(Err(error)) => (
                            Some(error.to_string()),
                            handler.error_disposition(&error)
                                == HandlerErrorDisposition::DiscardUpdate,
                            false,
                        ),
                        // A panic repeats on every retry, so the update is
                        // quarantined at once instead of taking more workers.
                        Err(_) => (Some("update handler panicked".to_owned()), true, true),
                    };
                    if error.is_some()
                        && !permanent
                        && let Some(delay) = retry_delay(record.attempts)
                    {
                        // Give a brief outage (Redis, Postgres, Telegram)
                        // time to pass before the update runs again.
                        // Unit tests exercise retries without sleeping through them.
                        thread::sleep(if cfg!(test) { Duration::ZERO } else { delay });
                    }
                    report_durable_completion(
                        &updates,
                        &completion_sender,
                        DurableUpdateCompletion {
                            record,
                            error,
                            permanent,
                            resubmitted: false,
                        },
                    );
                    if panicked {
                        // The panic may have left the handler half-updated,
                        // so the worker continues with a fresh one.
                        match panic::catch_unwind(AssertUnwindSafe(|| factory())) {
                            Ok(Ok(fresh)) => handler = fresh,
                            Ok(Err(_)) | Err(_) => return,
                        }
                    }
                }
            }));
        }
        drop(startup_sender);
        drop(completion_sender);

        for _ in 0..worker_count {
            // Every worker reports once, so a closed channel is only defensive.
            let (worker, error) = startup_receiver.recv().unwrap_or((
                workers.len(),
                Some(WORKER_STOPPED_DURING_STARTUP.to_owned()),
            ));
            if let Some(error) = error {
                close_update_sender(&updates);
                join_workers(&mut workers);
                return Err(ParallelHandlerBuildError::WorkerStartup { worker, error });
            }
        }

        Ok(Self {
            updates,
            workers,
            completions: completion_receiver,
            queue,
            active: HashSet::new(),
            completed: HashSet::new(),
            failures: BackgroundUpdateFailures::default(),
            recovered: false,
            handler: PhantomData,
        })
    }

    fn update_sender(&self) -> Option<SyncSender<DurableUpdateRecord>> {
        self.updates
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn drain_completions(&mut self, retry: bool) {
        while let Ok(completion) = self.completions.try_recv() {
            self.handle_completion(completion, retry);
        }
    }

    fn recover(&mut self) -> Result<(), DurableParallelHandlerError> {
        if self.recovered {
            return Ok(());
        }
        let queued = self
            .queue
            .list_updates()
            .map_err(|error| DurableParallelHandlerError::DurableQueue(error.to_string()))?;
        for queued in queued {
            let update_id = queued.update_id;
            if self.active.contains(&update_id) {
                continue;
            }
            let record = match decode_queued_update(&queued) {
                Ok(record) => record,
                Err(error) => {
                    self.queue
                        .quarantine_update(update_id, &queued.payload)
                        .map_err(|queue_error| {
                            DurableParallelHandlerError::DurableQueue(queue_error.to_string())
                        })?;
                    self.failures
                        .quarantined
                        .push(UpdateFailure { update_id, error });
                    continue;
                }
            };
            if record.completed {
                self.active.insert(update_id);
                self.completed.insert(update_id);
                continue;
            }
            self.update_sender()
                .ok_or(DurableParallelHandlerError::QueueUnavailable)?
                .send(record)
                .map_err(|_| DurableParallelHandlerError::QueueUnavailable)?;
            self.active.insert(update_id);
        }
        self.recovered = true;
        Ok(())
    }

    fn handle_completion(&mut self, completion: DurableUpdateCompletion, retry: bool) {
        let mut record = completion.record;
        let update_id = record.update.update_id;
        let Some(error) = completion.error else {
            record.completed = true;
            let persisted =
                encode_record(&record, "completed durable update").and_then(|payload| {
                    self.queue
                        .replace_update(update_id, &payload)
                        .map_err(|queue_error| {
                            format!("could not persist completed durable update: {queue_error}")
                        })
                });
            match persisted {
                Ok(()) => {
                    self.completed.insert(update_id);
                }
                Err(fatal) => self.set_fatal(fatal),
            }
            return;
        };

        if self.failures.fatal.is_some() {
            // The runtime is already stopping for recovery. Workers may have
            // resubmitted this update meanwhile, so leave its durable record
            // as last persisted and let the restart retry it.
            return;
        }
        if !completion.resubmitted {
            self.active.remove(&update_id);
        }
        record.attempts = record.attempts.saturating_add(1);
        if completion.permanent || record.attempts >= MAX_UPDATE_ATTEMPTS {
            let payload = encode_record(
                &DeadUpdateRecord {
                    schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
                    update: &record.update,
                    attempts: record.attempts,
                    error: &error,
                },
                &format!("dead durable update {update_id}"),
            );
            let quarantined = payload.and_then(|payload| {
                self.queue
                    .quarantine_update(update_id, &payload)
                    .map_err(|queue_error| {
                        format!("could not quarantine durable update {update_id}: {queue_error}")
                    })
            });
            match quarantined {
                Ok(()) => self
                    .failures
                    .quarantined
                    .push(UpdateFailure { update_id, error }),
                Err(fatal) => self.set_fatal(fatal),
            }
            return;
        }

        let persisted = encode_record(&record, &format!("durable update {update_id} retry"))
            .and_then(|payload| {
                self.queue
                    .replace_update(update_id, &payload)
                    .map_err(|queue_error| {
                        format!("could not persist durable update {update_id} retry: {queue_error}")
                    })
            });
        match persisted {
            Ok(()) => {
                self.failures
                    .retrying
                    .push(UpdateFailure { update_id, error });
                if completion.resubmitted {
                    return;
                }
                if retry && let Some(sender) = self.update_sender() {
                    match sender.send(record) {
                        Ok(()) => {
                            self.active.insert(update_id);
                        }
                        Err(_) => {
                            self.set_fatal(format!(
                                "could not resubmit durable update {update_id}"
                            ));
                        }
                    }
                }
            }
            Err(fatal) => self.set_fatal(fatal),
        }
    }

    fn set_fatal(&mut self, error: String) {
        if self.failures.fatal.is_none() {
            self.failures.fatal = Some(error);
        }
    }

    fn stop(&mut self) {
        close_update_sender(&self.updates);
        join_workers(&mut self.workers);
        self.drain_completions(false);
    }
}

impl<Handler, Queue> UpdateHandler for DurableParallelUpdateHandler<Handler, Queue>
where
    Handler: UpdateHandler + 'static,
    Handler::Error: Display,
    Queue: DurableUpdateQueue,
{
    type Error = DurableParallelHandlerError;

    fn prepare(&mut self) -> Result<(), Self::Error> {
        self.recover()
    }

    fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
        self.recover()?;
        let update_id = update.update_id;
        if self.active.contains(&update_id) {
            return Ok(());
        }
        let record = DurableUpdateRecord {
            schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
            update,
            attempts: 0,
            completed: false,
        };
        let payload = encode_record(&record, "durable update")
            .map_err(DurableParallelHandlerError::Serialization)?;
        self.queue
            .insert_update(update_id, &payload)
            .map_err(|error| DurableParallelHandlerError::DurableQueue(error.to_string()))?;
        self.update_sender()
            .ok_or(DurableParallelHandlerError::QueueUnavailable)?
            .send(record)
            .map_err(|_| DurableParallelHandlerError::QueueUnavailable)?;
        self.active.insert(update_id);
        Ok(())
    }

    fn error_disposition(&self, _error: &Self::Error) -> HandlerErrorDisposition {
        HandlerErrorDisposition::StopRuntime
    }

    fn confirm_updates(&mut self, confirmation: UpdateConfirmation) -> Result<(), Self::Error> {
        let mut confirmed = self
            .completed
            .iter()
            .copied()
            .filter(|update_id| match confirmation {
                UpdateConfirmation::Before(offset) => *update_id < offset,
                UpdateConfirmation::All => true,
            })
            .collect::<Vec<_>>();
        if confirmed.is_empty() {
            return Ok(());
        }
        confirmed.sort_unstable();
        // One round trip for the whole confirmed batch.
        self.queue
            .delete_updates(&confirmed)
            .map_err(|error| DurableParallelHandlerError::DurableQueue(error.to_string()))?;
        for update_id in confirmed {
            self.completed.remove(&update_id);
            self.active.remove(&update_id);
        }
        Ok(())
    }

    fn take_background_failures(&mut self) -> BackgroundUpdateFailures {
        self.drain_completions(true);
        std::mem::take(&mut self.failures)
    }

    fn shutdown(&mut self) {
        self.stop();
    }
}

impl<Handler, Queue> Drop for DurableParallelUpdateHandler<Handler, Queue> {
    fn drop(&mut self) {
        close_update_sender(&self.updates);
        join_workers(&mut self.workers);
    }
}

fn encode_record(record: &impl Serialize, description: &str) -> Result<String, String> {
    serde_json::to_string(record)
        .map_err(|error| format!("could not encode {description}: {error}"))
}

fn close_update_sender(updates: &Mutex<Option<SyncSender<DurableUpdateRecord>>>) {
    updates
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
}

fn decode_queued_update(queued: &QueuedUpdate) -> Result<DurableUpdateRecord, String> {
    let record = serde_json::from_str::<DurableUpdateRecord>(&queued.payload)
        .map_err(|error| error.to_string())?;
    if record.schema_version != DURABLE_UPDATE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema version {}",
            record.schema_version
        ));
    }
    if record.update.update_id != queued.update_id {
        return Err(format!(
            "payload contains update id {}",
            record.update.update_id
        ));
    }
    Ok(record)
}

fn join_workers(workers: &mut Vec<JoinHandle<()>>) {
    for worker in workers.drain(..) {
        let _ = worker.join();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    Idle,
    Dispatched {
        count: usize,
    },
    HandlerFailures {
        retrying: Vec<UpdateFailure>,
        quarantined: Vec<UpdateFailure>,
    },
    Retry(PollFailure),
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Poll(#[from] PollingError),
    #[error("update handler preparation failed: {0}")]
    Handler(String),
}

pub struct PollingRuntime<Source, Handler> {
    source: Source,
    handler: Handler,
    offset: Option<i64>,
    pending_offset: Option<i64>,
    completed_updates: HashSet<i64>,
    failure_attempts: HashMap<i64, usize>,
}

impl<Source, Handler> PollingRuntime<Source, Handler>
where
    Source: UpdateSource,
    Handler: UpdateHandler,
    Handler::Error: Display,
{
    #[must_use]
    pub fn new(source: Source, handler: Handler) -> Self {
        Self {
            source,
            handler,
            offset: None,
            pending_offset: None,
            completed_updates: HashSet::new(),
            failure_attempts: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn offset(&self) -> Option<i64> {
        self.offset
    }

    pub fn step(&mut self) -> Result<StepOutcome, RuntimeError> {
        self.handler
            .prepare()
            .map_err(|error| RuntimeError::Handler(error.to_string()))?;
        let background_failures = self.handler.take_background_failures();
        if let Some(error) = background_failures.fatal {
            return Err(RuntimeError::Handler(error));
        }
        if !background_failures.retrying.is_empty() || !background_failures.quarantined.is_empty() {
            return Ok(StepOutcome::HandlerFailures {
                retrying: background_failures.retrying,
                quarantined: background_failures.quarantined,
            });
        }
        let requested_offset = self.offset;
        match self.source.poll(requested_offset)? {
            PollOutcome::Retry(failure) => Ok(StepOutcome::Retry(failure)),
            PollOutcome::Updates(updates) => {
                let confirmation = requested_offset.map_or_else(
                    || {
                        updates
                            .iter()
                            .map(|update| update.update_id)
                            .min()
                            .map_or(UpdateConfirmation::All, UpdateConfirmation::Before)
                    },
                    UpdateConfirmation::Before,
                );
                if updates.is_empty() {
                    self.handler
                        .confirm_updates(confirmation)
                        .map_err(|error| RuntimeError::Handler(error.to_string()))?;
                    return Ok(StepOutcome::Idle);
                }
                if let Some(next) = next_offset(&updates, self.offset) {
                    self.pending_offset = Some(
                        self.pending_offset
                            .map_or(next, |pending| pending.max(next)),
                    );
                }
                let mut count = 0;
                let mut attempted = Vec::new();
                let mut failures = Vec::new();
                let mut discarded = HashSet::new();
                for update in updates {
                    let update_id = update.update_id;
                    if self.completed_updates.contains(&update_id) {
                        continue;
                    }
                    attempted.push(update_id);
                    match self.handler.handle(update) {
                        Ok(()) => count += 1,
                        Err(handler_error) => {
                            match self.handler.error_disposition(&handler_error) {
                                HandlerErrorDisposition::StopRuntime => {
                                    return Err(RuntimeError::Handler(handler_error.to_string()));
                                }
                                HandlerErrorDisposition::DiscardUpdate => {
                                    discarded.insert(update_id);
                                }
                                HandlerErrorDisposition::RetryUpdate => {}
                            }
                            failures.push(UpdateFailure {
                                update_id,
                                error: handler_error.to_string(),
                            });
                        }
                    }
                }
                failures.extend(self.handler.finish_batch());
                // Clean up the previous batch only once the new one is on its
                // way, so durable-queue bookkeeping never delays fresh updates.
                self.handler
                    .confirm_updates(confirmation)
                    .map_err(|error| RuntimeError::Handler(error.to_string()))?;
                let failed_ids = failures
                    .iter()
                    .map(|failure| failure.update_id)
                    .collect::<HashSet<_>>();
                for update_id in attempted {
                    if !failed_ids.contains(&update_id) {
                        self.completed_updates.insert(update_id);
                        self.failure_attempts.remove(&update_id);
                    }
                }
                let mut retrying = Vec::new();
                let mut quarantined = Vec::new();
                for failure in failures {
                    let attempts = self.failure_attempts.entry(failure.update_id).or_default();
                    *attempts = attempts.saturating_add(1);
                    if discarded.contains(&failure.update_id) || *attempts >= MAX_UPDATE_ATTEMPTS {
                        self.failure_attempts.remove(&failure.update_id);
                        self.completed_updates.insert(failure.update_id);
                        quarantined.push(failure);
                    } else {
                        retrying.push(failure);
                    }
                }
                if self.failure_attempts.is_empty() {
                    self.offset = self.pending_offset;
                    self.pending_offset = None;
                    self.completed_updates.clear();
                }
                if retrying.is_empty() && quarantined.is_empty() {
                    Ok(StepOutcome::Dispatched { count })
                } else {
                    Ok(StepOutcome::HandlerFailures {
                        retrying,
                        quarantined,
                    })
                }
            }
        }
    }

    pub fn shutdown(&mut self) {
        self.handler.shutdown();
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex, PoisonError};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_adapters::redis_update_queue::QueuedUpdate;
    use bot_adapters::redis_update_queue::RedisUpdateQueue;
    use bot_adapters::telegram_http::TransportFailureKind;
    use bot_adapters::telegram_polling::{
        IncomingEvent, IncomingMessage, IncomingUpdate, PollFailure, PollOutcome,
    };
    use bot_core::telegram_input::ChatId;

    use super::{
        BackgroundUpdateFailures, DURABLE_UPDATE_SCHEMA_VERSION, DurableParallelHandlerError,
        DurableParallelUpdateHandler, DurableUpdateCompletion, DurableUpdateQueue,
        DurableUpdateRecord, HandlerErrorDisposition, MAX_UPDATE_ATTEMPTS,
        ParallelHandlerBuildError, ParallelHandlerError, ParallelUpdateHandler, PollingError,
        PollingRuntime, RuntimeError, StepOutcome, UpdateConfirmation, UpdateFailure,
        UpdateHandler, UpdateSource, retry_delay,
    };

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Takes one unit from a failure budget, reporting whether it was spent.
    fn spend(budget: &AtomicUsize) -> bool {
        budget
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
    }

    #[derive(Clone, Default)]
    struct MemoryDurableQueue {
        updates: Arc<Mutex<HashMap<i64, String>>>,
        dead: Arc<Mutex<HashMap<i64, String>>>,
        insert_failures: Arc<AtomicUsize>,
        list_failures: Arc<AtomicUsize>,
        replace_failures: Arc<AtomicUsize>,
        delete_failures: Arc<AtomicUsize>,
        quarantine_failures: Arc<AtomicUsize>,
        delete_calls: Arc<AtomicUsize>,
    }

    impl MemoryDurableQueue {
        fn stored(&self) -> std::sync::MutexGuard<'_, HashMap<i64, String>> {
            self.updates.lock().unwrap_or_else(PoisonError::into_inner)
        }

        fn record(&self, update_id: i64) -> Option<DurableUpdateRecord> {
            self.stored()
                .get(&update_id)
                .and_then(|payload| serde_json::from_str(payload).ok())
        }
    }

    impl DurableUpdateQueue for MemoryDurableQueue {
        type Error = &'static str;

        fn insert_update(&self, update_id: i64, payload: &str) -> Result<bool, Self::Error> {
            if spend(&self.insert_failures) {
                return Err("synthetic insert failure");
            }
            let mut updates = self.stored();
            if updates.contains_key(&update_id) {
                return Ok(false);
            }
            updates.insert(update_id, payload.to_owned());
            Ok(true)
        }

        fn list_updates(&self) -> Result<Vec<QueuedUpdate>, Self::Error> {
            if spend(&self.list_failures) {
                return Err("synthetic list failure");
            }
            Ok(self
                .stored()
                .iter()
                .map(|(update_id, payload)| QueuedUpdate {
                    update_id: *update_id,
                    payload: payload.clone(),
                })
                .collect())
        }

        fn replace_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error> {
            if spend(&self.replace_failures) {
                return Err("synthetic replace failure");
            }
            self.stored().insert(update_id, payload.to_owned());
            Ok(())
        }

        fn delete_updates(&self, update_ids: &[i64]) -> Result<usize, Self::Error> {
            if spend(&self.delete_failures) {
                return Err("synthetic delete failure");
            }
            self.delete_calls.fetch_add(1, Ordering::SeqCst);
            let mut updates = self.stored();
            Ok(update_ids
                .iter()
                .filter(|update_id| updates.remove(update_id).is_some())
                .count())
        }

        fn quarantine_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error> {
            if spend(&self.quarantine_failures) {
                return Err("synthetic quarantine failure");
            }
            self.dead
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(update_id, payload.to_owned());
            self.stored().remove(&update_id);
            Ok(())
        }
    }

    /// Polls `done` until it holds, failing the test after one second.
    fn wait_until(mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut reached = false;
        while !reached {
            assert!(
                Instant::now() < deadline,
                "condition was not reached in time"
            );
            thread::yield_now();
            reached = done();
        }
    }

    fn persisted_completed(queue: &MemoryDurableQueue, update_id: i64) -> bool {
        queue
            .record(update_id)
            .is_some_and(|record| record.completed)
    }

    /// Behavior shared by every worker a factory creates. Budgets count the
    /// calls, across all workers, that fail or panic before calls succeed.
    #[derive(Default)]
    struct Script {
        started: Mutex<Option<mpsc::Sender<(i64, i64)>>>,
        failing: AtomicUsize,
        failing_permanently: AtomicUsize,
        panicking: AtomicUsize,
        gate_first: Option<Mutex<mpsc::Receiver<()>>>,
        release: Option<(Mutex<bool>, Condvar)>,
        active: AtomicUsize,
        maximum: AtomicUsize,
        pause: Option<Duration>,
        calls: AtomicUsize,
    }

    impl Script {
        fn released(&self) {
            if let Some((released, wake)) = &self.release {
                *released.lock().unwrap_or_else(PoisonError::into_inner) = true;
                wake.notify_all();
            }
        }
    }

    struct Worker(Arc<Script>);

    impl UpdateHandler for Worker {
        type Error = &'static str;

        fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
            let script = &self.0;
            let call = script.calls.fetch_add(1, Ordering::SeqCst);
            let chat_id = match update.event {
                IncomingEvent::Message(message) => message.chat_id.map_or(0, |id| id.0),
                _ => 0,
            };
            if let Some(started) = script
                .started
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_ref()
            {
                let _ = started.send((update.update_id, chat_id));
            }
            let active = script.active.fetch_add(1, Ordering::SeqCst) + 1;
            script.maximum.fetch_max(active, Ordering::SeqCst);
            if let Some(pause) = script.pause {
                thread::sleep(pause);
            }
            if let (0, Some(gate)) = (call, &script.gate_first) {
                let _ = gate.lock().map(|gate| gate.recv());
            }
            if let Some((released, wake)) = &script.release {
                let mut guard = released.lock().unwrap_or_else(PoisonError::into_inner);
                while !*guard {
                    guard = wake.wait(guard).unwrap_or_else(PoisonError::into_inner);
                }
            }
            script.active.fetch_sub(1, Ordering::SeqCst);
            assert!(!spend(&script.panicking), "synthetic handler panic");
            if spend(&script.failing) {
                return Err("synthetic handler failure");
            }
            if spend(&script.failing_permanently) {
                return Err("synthetic permanent failure");
            }
            Ok(())
        }

        fn error_disposition(&self, error: &Self::Error) -> HandlerErrorDisposition {
            if *error == "synthetic permanent failure" {
                HandlerErrorDisposition::DiscardUpdate
            } else {
                HandlerErrorDisposition::RetryUpdate
            }
        }
    }

    type Factory = Box<dyn Fn() -> Result<Worker, String> + Send + Sync>;

    fn factory(script: &Arc<Script>) -> Factory {
        let script = Arc::clone(script);
        Box::new(move || Ok(Worker(Arc::clone(&script))))
    }

    /// A script whose workers report the updates they start.
    fn reporting_script(script: Script) -> (Arc<Script>, mpsc::Receiver<(i64, i64)>) {
        let (sender, receiver) = mpsc::channel();
        *script
            .started
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(sender);
        (Arc::new(script), receiver)
    }

    fn started_id(receiver: &mpsc::Receiver<(i64, i64)>) -> Option<i64> {
        receiver
            .recv_timeout(Duration::from_secs(1))
            .ok()
            .map(|(update_id, _)| update_id)
    }

    fn released_script() -> Script {
        Script {
            release: Some((Mutex::new(false), Condvar::new())),
            ..Script::default()
        }
    }

    fn gated_script(gate: mpsc::Receiver<()>) -> Script {
        Script {
            gate_first: Some(Mutex::new(gate)),
            ..Script::default()
        }
    }

    type Durable = DurableParallelUpdateHandler<Worker, MemoryDurableQueue>;

    fn durable(
        workers: usize,
        capacity: usize,
        queue: &MemoryDurableQueue,
        script: &Arc<Script>,
    ) -> Result<Durable, ParallelHandlerBuildError> {
        DurableParallelUpdateHandler::start(workers, capacity, queue.clone(), factory(script))
    }

    fn update(update_id: i64) -> IncomingUpdate {
        IncomingUpdate {
            update_id,
            event: IncomingEvent::Unsupported,
        }
    }

    fn chat_update(update_id: i64, chat_id: i64) -> IncomingUpdate {
        IncomingUpdate {
            update_id,
            event: IncomingEvent::Message(Box::new(IncomingMessage {
                message_id: None,
                chat_id: Some(ChatId(chat_id)),
                chat_type: Some("group".to_owned()),
                chat_title: None,
                sender_id: None,
                sender_first_name: None,
                sender_last_name: None,
                sender_username: None,
                sender_language_code: None,
                has_reply: false,
                replied_message_id: None,
                replied_sender_first_name: None,
                replied_sender_username: None,
                replied_text: None,
                visual_media_kind: None,
                audio_media_kind: None,
                audio_duration_seconds: None,
                content: None,
            })),
        }
    }

    fn record(update_id: i64, attempts: usize) -> DurableUpdateRecord {
        DurableUpdateRecord {
            schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
            update: update(update_id),
            attempts,
            completed: false,
        }
    }

    fn completion(update_id: i64, attempts: usize, error: Option<&str>) -> DurableUpdateCompletion {
        DurableUpdateCompletion {
            record: record(update_id, attempts),
            error: error.map(str::to_owned),
            permanent: false,
            resubmitted: false,
        }
    }

    struct Source {
        outcomes: VecDeque<Result<PollOutcome, PollingError>>,
        offsets: Vec<Option<i64>>,
    }

    impl UpdateSource for Source {
        fn poll(&mut self, offset: Option<i64>) -> Result<PollOutcome, PollingError> {
            self.offsets.push(offset);
            self.outcomes
                .pop_front()
                .unwrap_or(Ok(PollOutcome::Updates(Vec::new())))
        }
    }

    fn source(outcomes: Vec<Result<PollOutcome, PollingError>>) -> Source {
        Source {
            outcomes: VecDeque::from(outcomes),
            offsets: Vec::new(),
        }
    }

    fn updates(ids: &[i64]) -> Result<PollOutcome, PollingError> {
        Ok(PollOutcome::Updates(
            ids.iter().copied().map(update).collect(),
        ))
    }

    /// A polling-side handler with scripted results for every runtime hook.
    #[derive(Default)]
    struct Handler {
        handled: Vec<i64>,
        events: Vec<String>,
        fail_on: Option<i64>,
        discard_on: Option<i64>,
        stop_on: Option<i64>,
        prepare_error: Option<&'static str>,
        confirm_error: Option<&'static str>,
        background: VecDeque<BackgroundUpdateFailures>,
    }

    impl UpdateHandler for Handler {
        type Error = &'static str;

        fn prepare(&mut self) -> Result<(), Self::Error> {
            self.prepare_error.map_or(Ok(()), Err)
        }

        fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
            self.events.push(format!("handle {}", update.update_id));
            if self.stop_on == Some(update.update_id) {
                return Err("synthetic fatal failure");
            }
            if self.fail_on == Some(update.update_id) {
                return Err("synthetic handler failure");
            }
            if self.discard_on == Some(update.update_id) {
                return Err("synthetic permanent failure");
            }
            self.handled.push(update.update_id);
            Ok(())
        }

        fn error_disposition(&self, error: &Self::Error) -> HandlerErrorDisposition {
            match *error {
                "synthetic fatal failure" => HandlerErrorDisposition::StopRuntime,
                "synthetic permanent failure" => HandlerErrorDisposition::DiscardUpdate,
                _ => HandlerErrorDisposition::RetryUpdate,
            }
        }

        fn confirm_updates(&mut self, confirmation: UpdateConfirmation) -> Result<(), Self::Error> {
            self.events.push(format!("confirm {confirmation:?}"));
            self.confirm_error.map_or(Ok(()), Err)
        }

        fn take_background_failures(&mut self) -> BackgroundUpdateFailures {
            self.background.pop_front().unwrap_or_default()
        }
    }

    type ScriptedRuntime = PollingRuntime<Source, Handler>;

    fn runtime(
        outcomes: Vec<Result<PollOutcome, PollingError>>,
        handler: Handler,
    ) -> ScriptedRuntime {
        PollingRuntime::new(source(outcomes), handler)
    }

    #[test]
    fn first_poll_processes_pending_updates_before_advancing_the_offset() {
        let mut runtime = runtime(
            vec![updates(&[7, 8]), updates(&[9]), updates(&[])],
            Handler::default(),
        );
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 2 }));
        assert_eq!(runtime.offset(), Some(9));
        assert_eq!(runtime.handler.handled, vec![7, 8]);
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(runtime.offset(), Some(10));
        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
        assert_eq!(runtime.source.offsets, vec![None, Some(9), Some(10)]);
        assert_eq!(runtime.handler.handled, vec![7, 8, 9]);
    }

    #[test]
    fn failed_update_is_retried_without_repeating_successes_then_quarantined() {
        let mut runtime = runtime(
            vec![
                updates(&[10, 11, 12]),
                // Telegram resends the whole batch until the offset advances.
                updates(&[10, 11, 12]),
                updates(&[11]),
                updates(&[13]),
            ],
            Handler {
                fail_on: Some(11),
                ..Handler::default()
            },
        );
        assert!(matches!(
            runtime.step(),
            Ok(StepOutcome::HandlerFailures { retrying, quarantined })
                if retrying.len() == 1 && quarantined.is_empty()
        ));
        assert_eq!(runtime.offset(), None);
        assert_eq!(runtime.handler.handled, vec![10, 12]);
        assert!(matches!(
            runtime.step(),
            Ok(StepOutcome::HandlerFailures { retrying, quarantined })
                if retrying.len() == 1 && quarantined.is_empty()
        ));
        assert!(matches!(
            runtime.step(),
            Ok(StepOutcome::HandlerFailures { retrying, quarantined })
                if retrying.is_empty() && quarantined.len() == 1
        ));
        assert_eq!(runtime.offset(), Some(13));
        assert_eq!(runtime.handler.handled, vec![10, 12]);
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(runtime.handler.handled, vec![10, 12, 13]);
        assert_eq!(runtime.source.offsets, [None, None, None, Some(13)]);
    }

    #[test]
    fn permanently_failing_update_is_quarantined_without_retries() {
        let mut runtime = runtime(
            vec![updates(&[20, 21]), updates(&[22])],
            Handler {
                discard_on: Some(20),
                ..Handler::default()
            },
        );
        assert_eq!(
            runtime.step(),
            Ok(StepOutcome::HandlerFailures {
                retrying: Vec::new(),
                quarantined: vec![UpdateFailure {
                    update_id: 20,
                    error: "synthetic permanent failure".to_owned(),
                }],
            })
        );
        assert_eq!(runtime.offset(), Some(22));
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(runtime.handler.handled, vec![21, 22]);
        assert_eq!(runtime.source.offsets, [None, Some(22)]);
    }

    #[test]
    fn retries_and_poll_errors_leave_offset_unchanged() {
        let retry = PollFailure::Transport {
            failure: TransportFailureKind::Timeout,
        };
        let mut runtime = runtime(
            vec![
                Ok(PollOutcome::Retry(retry.clone())),
                Err(PollingError::InvalidResponse),
            ],
            Handler::default(),
        );
        assert_eq!(runtime.step(), Ok(StepOutcome::Retry(retry)));
        assert!(matches!(runtime.step(), Err(RuntimeError::Poll(_))));
        assert_eq!(runtime.offset(), None);
        assert_eq!(runtime.source.offsets, vec![None, None]);
    }

    #[test]
    fn handler_hook_failures_stop_the_runtime_without_advancing() {
        let mut preparing = runtime(
            vec![updates(&[1])],
            Handler {
                prepare_error: Some("synthetic prepare failure"),
                ..Handler::default()
            },
        );
        assert_eq!(
            preparing.step(),
            Err(RuntimeError::Handler(
                "synthetic prepare failure".to_owned()
            ))
        );
        assert!(preparing.source.offsets.is_empty());

        let mut idle_confirmation = runtime(
            vec![updates(&[])],
            Handler {
                confirm_error: Some("synthetic confirmation failure"),
                ..Handler::default()
            },
        );
        assert_eq!(
            idle_confirmation.step(),
            Err(RuntimeError::Handler(
                "synthetic confirmation failure".to_owned()
            ))
        );

        let mut batch_confirmation = runtime(
            vec![updates(&[4])],
            Handler {
                confirm_error: Some("synthetic confirmation failure"),
                ..Handler::default()
            },
        );
        assert_eq!(
            batch_confirmation.step(),
            Err(RuntimeError::Handler(
                "synthetic confirmation failure".to_owned()
            ))
        );
        assert_eq!(batch_confirmation.handler.handled, [4]);
        assert_eq!(batch_confirmation.offset(), None);

        let mut stopping = runtime(
            vec![updates(&[5, 6])],
            Handler {
                stop_on: Some(5),
                ..Handler::default()
            },
        );
        assert_eq!(
            stopping.step(),
            Err(RuntimeError::Handler("synthetic fatal failure".to_owned()))
        );
        assert!(stopping.handler.handled.is_empty());
        assert_eq!(stopping.offset(), None);
    }

    #[test]
    fn supports_infallible_handlers() {
        struct InfallibleHandler;
        impl UpdateHandler for InfallibleHandler {
            type Error = Infallible;
            fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
                Ok(())
            }
        }
        let mut runtime = PollingRuntime::new(source(vec![updates(&[1])]), InfallibleHandler);
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
    }

    #[test]
    fn background_failures_are_reported_before_polling() {
        let retrying = vec![UpdateFailure {
            update_id: 91,
            error: "synthetic retry".to_owned(),
        }];
        let quarantined = vec![UpdateFailure {
            update_id: 92,
            error: "synthetic quarantine".to_owned(),
        }];
        let mut runtime = runtime(
            Vec::new(),
            Handler {
                background: VecDeque::from([
                    BackgroundUpdateFailures {
                        retrying: retrying.clone(),
                        quarantined: quarantined.clone(),
                        fatal: None,
                    },
                    BackgroundUpdateFailures {
                        fatal: Some("synthetic fatal failure".to_owned()),
                        ..BackgroundUpdateFailures::default()
                    },
                ]),
                ..Handler::default()
            },
        );
        assert_eq!(
            runtime.step(),
            Ok(StepOutcome::HandlerFailures {
                retrying,
                quarantined,
            })
        );
        assert_eq!(
            runtime.step(),
            Err(RuntimeError::Handler("synthetic fatal failure".to_owned()))
        );
        assert!(runtime.source.offsets.is_empty());
    }

    #[test]
    fn new_updates_are_dispatched_before_the_previous_batch_is_confirmed() {
        let mut runtime = runtime(vec![updates(&[10]), updates(&[11])], Handler::default());
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
        assert_eq!(
            runtime.handler.events,
            [
                "handle 10",
                "confirm Before(10)",
                "handle 11",
                "confirm Before(11)",
                "confirm Before(12)",
            ]
        );
    }

    #[test]
    fn an_update_at_the_maximum_id_is_handled_without_advancing_the_offset() {
        let mut runtime = runtime(vec![updates(&[i64::MAX])], Handler::default());
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(runtime.handler.handled, [i64::MAX]);
        assert_eq!(runtime.offset(), None);
        assert_eq!(runtime.pending_offset, None);
    }

    #[test]
    fn handler_defaults_and_parallel_validation_are_safe() -> TestResult {
        let script = Arc::new(Script::default());
        let mut worker = Worker(Arc::clone(&script));
        assert_eq!(worker.prepare(), Ok(()));
        assert_eq!(
            worker.error_disposition(&"synthetic"),
            HandlerErrorDisposition::RetryUpdate
        );
        assert_eq!(worker.confirm_updates(UpdateConfirmation::All), Ok(()));
        assert!(worker.finish_batch().is_empty());
        assert_eq!(worker.take_background_failures(), Default::default());
        worker.shutdown();

        assert!(matches!(
            ParallelUpdateHandler::start(0, 1, factory(&script)),
            Err(ParallelHandlerBuildError::NoWorkers)
        ));
        assert!(matches!(
            ParallelUpdateHandler::start(1, 0, factory(&script)),
            Err(ParallelHandlerBuildError::EmptyQueue)
        ));
        let failing: Factory =
            Box::new(|| Err::<Worker, _>("synthetic startup failure".to_owned()));
        assert!(matches!(
            ParallelUpdateHandler::start(1, 1, failing),
            Err(ParallelHandlerBuildError::WorkerStartup { error, .. })
                if error == "synthetic startup failure"
        ));
        let queue = MemoryDurableQueue::default();
        assert!(matches!(
            durable(0, 1, &queue, &script),
            Err(ParallelHandlerBuildError::NoWorkers)
        ));
        assert!(matches!(
            durable(1, 0, &queue, &script),
            Err(ParallelHandlerBuildError::EmptyQueue)
        ));
        let failing: Factory =
            Box::new(|| Err::<Worker, _>("synthetic durable startup failure".to_owned()));
        assert!(matches!(
            DurableParallelUpdateHandler::<Worker, MemoryDurableQueue>::start(1, 1, queue, failing),
            Err(ParallelHandlerBuildError::WorkerStartup { error, .. })
                if error == "synthetic durable startup failure"
        ));

        let mut stopped = ParallelUpdateHandler::start(1, 1, factory(&script))?;
        stopped.shutdown();
        assert_eq!(
            stopped.handle(update(1)),
            Err(ParallelHandlerError::QueueUnavailable)
        );
        Ok(())
    }

    #[test]
    fn factory_panics_are_reported_without_hanging_startup() {
        let panicking = || -> Factory {
            Box::new(|| -> Result<Worker, String> { panic!("synthetic factory panic") })
        };
        assert!(matches!(
            ParallelUpdateHandler::start(2, 2, panicking()),
            Err(ParallelHandlerBuildError::WorkerStartup { error, .. })
                if error == "worker factory panicked"
        ));
        assert!(matches!(
            DurableParallelUpdateHandler::<Worker, MemoryDurableQueue>::start(
                2,
                2,
                MemoryDurableQueue::default(),
                panicking(),
            ),
            Err(ParallelHandlerBuildError::WorkerStartup { error, .. })
                if error == "worker factory panicked"
        ));
    }

    #[test]
    fn parallel_handler_runs_same_chat_and_different_chat_updates_together() -> TestResult {
        let (script, started) = reporting_script(released_script());
        let mut handler = ParallelUpdateHandler::start(5, 5, factory(&script))?;
        for update in [
            chat_update(1, 10),
            chat_update(2, 10),
            chat_update(3, 20),
            chat_update(4, 30),
            update(5),
        ] {
            assert!(handler.handle(update).is_ok());
        }
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.extend(started.recv_timeout(Duration::from_secs(1)));
        }
        assert_eq!(seen.len(), 5);
        assert_eq!(seen.iter().filter(|(_, chat)| *chat == 10).count(), 2);
        assert!(seen.iter().any(|(_, chat)| *chat == 20));
        assert!(seen.iter().any(|(_, chat)| *chat == 30));
        // An update without a chat gets its own worker as well.
        assert!(seen.contains(&(5, 0)));
        script.released();
        handler.shutdown();
        Ok(())
    }

    #[test]
    fn parallel_handler_never_exceeds_its_worker_limit() -> TestResult {
        let script = Arc::new(Script {
            pause: Some(Duration::from_millis(10)),
            ..Script::default()
        });
        let mut handler = ParallelUpdateHandler::start(3, 12, factory(&script))?;
        for update_id in 1..=12 {
            assert!(handler.handle(update(update_id)).is_ok());
        }
        wait_until(|| script.maximum.load(Ordering::SeqCst) >= 3);
        handler.shutdown();
        assert_eq!(script.maximum.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[test]
    fn shutdown_finishes_active_work_and_drains_the_backlog() -> TestResult {
        let (script, started) = reporting_script(released_script());
        let mut handler = ParallelUpdateHandler::start(1, 3, factory(&script))?;
        assert!(handler.handle(update(1)).is_ok());
        assert_eq!(started_id(&started), Some(1));
        assert!(handler.handle(update(2)).is_ok());
        assert!(handler.handle(update(3)).is_ok());

        let shutdown = thread::spawn(move || handler.shutdown());
        script.released();
        assert!(shutdown.join().is_ok());
        assert_eq!(started_id(&started), Some(2));
        assert_eq!(started_id(&started), Some(3));
        Ok(())
    }

    #[test]
    fn parallel_handler_reports_worker_failures() -> TestResult {
        let script = Arc::new(Script {
            failing: AtomicUsize::new(1),
            ..Script::default()
        });
        let mut handler = ParallelUpdateHandler::start(1, 1, factory(&script))?;
        assert!(handler.handle(update(42)).is_ok());
        let failures = handler.finish_batch();
        handler.shutdown();
        assert_eq!(
            failures,
            [UpdateFailure {
                update_id: 42,
                error: "synthetic handler failure".to_owned(),
            }]
        );
        Ok(())
    }

    #[test]
    fn parallel_handler_converts_panics_into_completions() -> TestResult {
        let script = Arc::new(Script {
            panicking: AtomicUsize::new(1),
            ..Script::default()
        });
        let mut handler = ParallelUpdateHandler::start(1, 1, factory(&script))?;
        assert!(handler.handle(update(42)).is_ok());
        assert_eq!(
            handler.finish_batch(),
            [UpdateFailure {
                update_id: 42,
                error: "update handler panicked".to_owned(),
            }]
        );
        handler.shutdown();
        Ok(())
    }

    #[test]
    fn parallel_batch_reports_updates_stranded_by_a_dead_worker() -> TestResult {
        let (gate, gate_receiver) = mpsc::channel();
        let (script, started) = reporting_script(Script {
            panicking: AtomicUsize::new(1),
            ..gated_script(gate_receiver)
        });
        let mut handler = ParallelUpdateHandler::start(1, 2, factory(&script))?;
        assert_eq!(handler.handle(update(1)), Ok(()));
        assert_eq!(started_id(&started), Some(1));
        // Queued behind the blocked update on the only worker.
        assert_eq!(handler.handle(update(2)), Ok(()));
        assert_eq!(gate.send(()), Ok(()));

        assert_eq!(
            handler.finish_batch(),
            [
                UpdateFailure {
                    update_id: 1,
                    error: "update handler panicked".to_owned(),
                },
                UpdateFailure {
                    update_id: 2,
                    error: "parallel update worker stopped before completing its batch".to_owned(),
                },
            ]
        );
        assert!(handler.pending.is_empty());
        handler.shutdown();
        Ok(())
    }

    #[test]
    fn durable_completion_retries_quarantines_and_preserves_queue_failures() -> TestResult {
        let queue = MemoryDurableQueue::default();
        let mut handler = durable(1, 2, &queue, &Arc::new(Script::default()))?;
        handler.handle_completion(completion(801, 0, Some("synthetic retry")), false);
        assert_eq!(handler.failures.retrying.len(), 1);
        assert!(queue.stored().contains_key(&801));

        handler.handle_completion(
            completion(
                802,
                MAX_UPDATE_ATTEMPTS - 1,
                Some("synthetic terminal failure"),
            ),
            false,
        );
        assert_eq!(handler.failures.quarantined.len(), 1);
        assert!(
            queue
                .dead
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .contains_key(&802)
        );

        queue.replace_failures.store(1, Ordering::SeqCst);
        handler.handle_completion(
            completion(803, 0, Some("synthetic persistence failure")),
            false,
        );
        assert_eq!(
            handler.failures.fatal.as_deref(),
            Some("could not persist durable update 803 retry: synthetic replace failure")
        );
        // Once stopping for recovery, later failures leave their records alone.
        handler.handle_completion(completion(804, 0, Some("synthetic late failure")), false);
        assert!(!queue.stored().contains_key(&804));
        handler.stop();
        Ok(())
    }

    #[test]
    fn durable_completion_persistence_failures_are_fatal() -> TestResult {
        let queue = MemoryDurableQueue::default();
        let mut handler = durable(1, 2, &queue, &Arc::new(Script::default()))?;
        queue.replace_failures.store(1, Ordering::SeqCst);
        handler.handle_completion(completion(811, 0, None), false);
        assert_eq!(
            handler.failures.fatal.as_deref(),
            Some("could not persist completed durable update: synthetic replace failure")
        );
        assert!(!handler.completed.contains(&811));
        handler.stop();

        let queue = MemoryDurableQueue::default();
        let mut handler = durable(1, 2, &queue, &Arc::new(Script::default()))?;
        queue.quarantine_failures.store(1, Ordering::SeqCst);
        handler.handle_completion(
            completion(
                812,
                MAX_UPDATE_ATTEMPTS - 1,
                Some("synthetic terminal failure"),
            ),
            false,
        );
        assert_eq!(
            handler.failures.fatal.as_deref(),
            Some("could not quarantine durable update 812: synthetic quarantine failure")
        );
        assert!(handler.failures.quarantined.is_empty());
        handler.stop();
        Ok(())
    }

    #[test]
    fn recovery_quarantines_unreadable_records_and_resubmits_pending_ones() -> TestResult {
        let queue = MemoryDurableQueue::default();
        queue
            .stored()
            .insert(821, "not a durable record".to_owned());
        queue
            .stored()
            .insert(822, serde_json::to_string(&record(999, 0))?);
        queue.stored().insert(
            823,
            serde_json::to_string(&DurableUpdateRecord {
                schema_version: DURABLE_UPDATE_SCHEMA_VERSION + 1,
                ..record(823, 0)
            })?,
        );
        queue
            .stored()
            .insert(824, serde_json::to_string(&record(824, 1))?);
        let (script, started) = reporting_script(Script::default());
        let mut handler = durable(1, 4, &queue, &script)?;
        assert_eq!(handler.prepare(), Ok(()));
        // Recovery runs once; later preparations are no-ops.
        assert_eq!(handler.prepare(), Ok(()));
        assert_eq!(started_id(&started), Some(824));
        let quarantined = handler
            .failures
            .quarantined
            .iter()
            .map(|failure| failure.update_id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(quarantined, [821, 822, 823].into());
        assert!(handler.active.contains(&824));
        handler.stop();
        Ok(())
    }

    #[test]
    fn recovery_failures_stop_the_runtime_before_polling() -> TestResult {
        let queue = MemoryDurableQueue::default();
        queue.list_failures.store(1, Ordering::SeqCst);
        let mut handler = durable(1, 1, &queue, &Arc::new(Script::default()))?;
        assert_eq!(
            handler.prepare(),
            Err(DurableParallelHandlerError::DurableQueue(
                "synthetic list failure".to_owned()
            ))
        );
        // Admission retries recovery, so the failure repeats until it heals.
        queue.list_failures.store(1, Ordering::SeqCst);
        assert_eq!(
            handler.handle(update(1)),
            Err(DurableParallelHandlerError::DurableQueue(
                "synthetic list failure".to_owned()
            ))
        );
        handler.stop();

        let queue = MemoryDurableQueue::default();
        queue.stored().insert(831, "unreadable".to_owned());
        queue.quarantine_failures.store(1, Ordering::SeqCst);
        let mut handler = durable(1, 1, &queue, &Arc::new(Script::default()))?;
        assert_eq!(
            handler.prepare(),
            Err(DurableParallelHandlerError::DurableQueue(
                "synthetic quarantine failure".to_owned()
            ))
        );
        assert!(queue.stored().contains_key(&831));
        handler.stop();
        Ok(())
    }

    #[test]
    fn durable_handler_polls_and_starts_later_updates_while_earlier_work_is_running() -> TestResult
    {
        let (script, started) = reporting_script(released_script());
        let handler = durable(2, 4, &MemoryDurableQueue::default(), &script)?;
        let mut runtime = PollingRuntime::new(source(vec![updates(&[1]), updates(&[2])]), handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(started_id(&started), Some(1));
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(started_id(&started), Some(2));
        assert_eq!(runtime.source.offsets, [None, Some(2)]);
        script.released();
        runtime.shutdown();
        Ok(())
    }

    #[test]
    fn durable_admission_failure_stops_without_advancing_the_telegram_offset() -> TestResult {
        let queue = MemoryDurableQueue::default();
        queue.insert_failures.store(1, Ordering::SeqCst);
        let handler = durable(1, 2, &queue, &Arc::new(Script::default()))?;
        let mut runtime = PollingRuntime::new(source(vec![updates(&[41])]), handler);

        assert!(matches!(runtime.step(), Err(RuntimeError::Handler(_))));
        assert_eq!(runtime.offset(), None);
        assert_eq!(runtime.source.offsets, [None]);
        assert!(queue.stored().is_empty());
        runtime.shutdown();
        // A stopped handler no longer admits updates.
        assert_eq!(
            runtime.handler.handle(update(42)),
            Err(DurableParallelHandlerError::QueueUnavailable)
        );
        assert_eq!(
            runtime
                .handler
                .error_disposition(&DurableParallelHandlerError::QueueUnavailable),
            HandlerErrorDisposition::StopRuntime
        );
        Ok(())
    }

    #[test]
    fn completed_update_is_retained_until_telegram_confirms_it() -> TestResult {
        let retry = PollFailure::Transport {
            failure: TransportFailureKind::Timeout,
        };
        let queue = MemoryDurableQueue::default();
        let (script, started) = reporting_script(Script::default());
        let handler = durable(1, 2, &queue, &script)?;
        let mut runtime = PollingRuntime::new(
            source(vec![
                updates(&[51]),
                Ok(PollOutcome::Retry(retry.clone())),
                updates(&[]),
            ]),
            handler,
        );

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(started_id(&started), Some(51));
        wait_until(|| {
            let failures = runtime.handler.take_background_failures();
            assert!(failures.fatal.is_none());
            persisted_completed(&queue, 51)
        });
        assert_eq!(runtime.step(), Ok(StepOutcome::Retry(retry)));
        assert!(queue.stored().contains_key(&51));

        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
        assert!(queue.stored().is_empty());
        assert_eq!(started.try_recv(), Err(mpsc::TryRecvError::Empty));
        runtime.shutdown();
        Ok(())
    }

    #[test]
    fn recovered_completed_update_is_not_replayed_before_confirmation() -> TestResult {
        let queue = MemoryDurableQueue::default();
        let persisted = serde_json::to_string(&DurableUpdateRecord {
            completed: true,
            ..record(61, 0)
        })?;
        assert_eq!(queue.insert_update(61, &persisted), Ok(true));
        let (script, started) = reporting_script(Script::default());
        let handler = durable(1, 2, &queue, &script)?;
        let mut runtime = PollingRuntime::new(source(vec![updates(&[61]), updates(&[])]), handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert!(queue.stored().contains_key(&61));
        assert_eq!(started.try_recv(), Err(mpsc::TryRecvError::Empty));

        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
        assert!(queue.stored().is_empty());
        assert_eq!(runtime.source.offsets, [None, Some(62)]);
        assert_eq!(started.try_recv(), Err(mpsc::TryRecvError::Empty));
        runtime.shutdown();
        Ok(())
    }

    #[test]
    fn confirmation_failure_stops_runtime_with_completed_update_intact() -> TestResult {
        let queue = MemoryDurableQueue::default();
        let (script, started) = reporting_script(Script::default());
        let handler = durable(1, 2, &queue, &script)?;
        let mut runtime = PollingRuntime::new(source(vec![updates(&[66]), updates(&[])]), handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(started_id(&started), Some(66));
        wait_until(|| {
            let failures = runtime.handler.take_background_failures();
            assert!(failures.fatal.is_none());
            persisted_completed(&queue, 66)
        });
        queue.delete_failures.store(1, Ordering::SeqCst);

        assert!(matches!(runtime.step(), Err(RuntimeError::Handler(_))));
        assert!(queue.stored().contains_key(&66));
        runtime.shutdown();
        Ok(())
    }

    #[test]
    fn failed_retry_persistence_stops_runtime_for_recovery() -> TestResult {
        let queue = MemoryDurableQueue::default();
        queue.replace_failures.store(1, Ordering::SeqCst);
        let (script, started) = reporting_script(Script {
            failing: AtomicUsize::new(usize::MAX),
            ..Script::default()
        });
        let handler = durable(1, 2, &queue, &script)?;
        let mut runtime = PollingRuntime::new(source(vec![updates(&[71])]), handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(started_id(&started), Some(71));
        let mut outcome = Ok(StepOutcome::Idle);
        wait_until(|| {
            outcome = runtime.step();
            outcome.is_err()
        });
        assert!(
            matches!(
                &outcome,
                Err(RuntimeError::Handler(error))
                    if error.contains("could not persist durable update 71 retry")
            ),
            "durable transition failure did not stop runtime: {outcome:?}"
        );
        assert!(queue.stored().contains_key(&71));
        runtime.shutdown();
        Ok(())
    }

    #[test]
    fn duplicate_durable_updates_are_recovered_once() -> TestResult {
        let queue = MemoryDurableQueue::default();
        assert_eq!(queue.insert_update(91, "synthetic payload"), Ok(true));
        assert_eq!(queue.insert_update(91, "duplicate payload"), Ok(false));
        let mut durable = durable(1, 1, &queue, &Arc::new(Script::default()))?;
        durable.active.insert(91);
        assert_eq!(durable.recover(), Ok(()));
        // An update already in flight is not admitted twice.
        assert_eq!(durable.handle(update(91)), Ok(()));
        assert_eq!(
            queue.stored().get(&91).map(String::as_str),
            Some("synthetic payload")
        );
        durable.stop();
        Ok(())
    }

    #[test]
    fn failed_durable_update_is_retried_without_waiting_for_the_next_poll() -> TestResult {
        let queue = MemoryDurableQueue::default();
        let (script, started) = reporting_script(Script {
            failing: AtomicUsize::new(1),
            ..Script::default()
        });
        let mut handler = durable(2, 4, &queue, &script)?;

        let begun = Instant::now();
        assert_eq!(handler.handle(update(501)), Ok(()));
        // Nothing drains completions here, standing in for a polling thread
        // blocked in a long poll: the worker must resubmit on its own.
        for _ in 0..2 {
            assert_eq!(started_id(&started), Some(501));
        }
        assert!(begun.elapsed() < Duration::from_secs(1));

        let mut retrying = Vec::new();
        wait_until(|| {
            let failures = handler.take_background_failures();
            assert!(failures.quarantined.is_empty() && failures.fatal.is_none());
            retrying.extend(failures.retrying);
            handler.completed.contains(&501)
        });
        assert_eq!(retrying.len(), 1);
        assert!(handler.completed.contains(&501) && handler.active.contains(&501));
        let record = queue.record(501);
        assert!(matches!(record, Some(record) if record.completed && record.attempts == 1));
        assert_eq!(script.calls.load(Ordering::SeqCst), 2);
        handler.stop();
        Ok(())
    }

    #[test]
    fn permanently_failing_durable_update_is_quarantined_on_its_first_attempt() -> TestResult {
        let queue = MemoryDurableQueue::default();
        let script = Arc::new(Script {
            failing_permanently: AtomicUsize::new(1),
            ..Script::default()
        });
        let mut handler = durable(2, 4, &queue, &script)?;
        assert_eq!(handler.handle(update(511)), Ok(()));

        let mut quarantined = Vec::new();
        wait_until(|| {
            let failures = handler.take_background_failures();
            assert!(failures.retrying.is_empty() && failures.fatal.is_none());
            quarantined.extend(failures.quarantined);
            !quarantined.is_empty()
        });
        handler.stop();
        assert_eq!(
            quarantined,
            [UpdateFailure {
                update_id: 511,
                error: "synthetic permanent failure".to_owned(),
            }]
        );
        assert_eq!(script.calls.load(Ordering::SeqCst), 1);
        assert!(!queue.stored().contains_key(&511));
        let dead = queue
            .dead
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&511)
            .cloned()
            .unwrap_or_default();
        assert!(dead.contains(r#""attempts":1"#), "{dead}");
        Ok(())
    }

    #[test]
    fn durable_confirmation_deletes_the_whole_batch_in_one_call() -> TestResult {
        let queue = MemoryDurableQueue::default();
        let mut handler = durable(1, 1, &queue, &Arc::new(Script::default()))?;
        assert_eq!(handler.confirm_updates(UpdateConfirmation::All), Ok(()));
        assert_eq!(queue.delete_calls.load(Ordering::SeqCst), 0);

        for update_id in [601, 602, 603] {
            assert_eq!(queue.insert_update(update_id, "synthetic"), Ok(true));
            handler.completed.insert(update_id);
            handler.active.insert(update_id);
        }
        assert_eq!(
            handler.confirm_updates(UpdateConfirmation::Before(603)),
            Ok(())
        );
        assert_eq!(queue.delete_calls.load(Ordering::SeqCst), 1);
        let stored = queue.stored();
        assert!(
            !stored.contains_key(&601) && !stored.contains_key(&602) && stored.contains_key(&603)
        );
        drop(stored);
        assert_eq!(handler.completed.iter().copied().collect::<Vec<_>>(), [603]);
        assert_eq!(handler.confirm_updates(UpdateConfirmation::All), Ok(()));
        assert!(queue.stored().is_empty() && handler.completed.is_empty());
        handler.stop();
        Ok(())
    }

    #[test]
    fn polling_thread_resubmits_a_retry_that_did_not_fit_the_full_queue() -> TestResult {
        let (gate, gate_receiver) = mpsc::channel();
        let (script, started) = reporting_script(Script {
            failing: AtomicUsize::new(1),
            ..gated_script(gate_receiver)
        });
        let queue = MemoryDurableQueue::default();
        let mut handler = durable(1, 1, &queue, &script)?;
        assert_eq!(handler.handle(update(701)), Ok(()));
        assert_eq!(started_id(&started), Some(701));
        // Fills the one-slot queue, so the worker cannot resubmit 701 itself.
        assert_eq!(handler.handle(update(702)), Ok(()));
        assert_eq!(gate.send(()), Ok(()));

        let mut retrying = Vec::new();
        wait_until(|| {
            let failures = handler.take_background_failures();
            assert_eq!(failures.fatal, None);
            assert!(failures.quarantined.is_empty());
            retrying.extend(failures.retrying);
            handler.completed.contains(&701) && handler.completed.contains(&702)
        });
        assert_eq!(
            retrying,
            [UpdateFailure {
                update_id: 701,
                error: "synthetic handler failure".to_owned(),
            }]
        );
        assert_eq!(
            started.try_iter().map(|(id, _)| id).collect::<Vec<_>>(),
            [702, 701],
            "the retry runs after the update that blocked the queue"
        );
        let record = queue.record(701);
        assert!(matches!(record, Some(record) if record.completed && record.attempts == 1));
        handler.stop();
        Ok(())
    }

    #[test]
    fn a_panicking_update_is_quarantined_and_its_worker_keeps_serving() -> TestResult {
        let script = Arc::new(Script {
            panicking: AtomicUsize::new(1),
            ..Script::default()
        });
        let queue = MemoryDurableQueue::default();
        let mut handler = durable(1, 2, &queue, &script)?;
        assert_eq!(handler.handle(update(711)), Ok(()));
        assert_eq!(handler.handle(update(712)), Ok(()));

        let mut failures = BackgroundUpdateFailures::default();
        wait_until(|| {
            let next = handler.take_background_failures();
            failures.quarantined.extend(next.quarantined);
            failures.retrying.extend(next.retrying);
            persisted_completed(&queue, 712)
        });
        // The panic would repeat on every retry, so it is not retried.
        assert_eq!(
            failures.quarantined,
            [UpdateFailure {
                update_id: 711,
                error: "update handler panicked".to_owned(),
            }]
        );
        assert!(failures.retrying.is_empty());
        assert_eq!(script.calls.load(Ordering::SeqCst), 2);
        handler.stop();
        Ok(())
    }

    #[test]
    fn failed_updates_wait_longer_before_each_retry() {
        assert_eq!(retry_delay(0), Some(Duration::from_secs(1)));
        assert_eq!(retry_delay(1), Some(Duration::from_secs(5)));
        assert_eq!(retry_delay(MAX_UPDATE_ATTEMPTS - 1), None);
    }

    #[test]
    fn a_worker_that_cannot_be_rebuilt_after_a_panic_stops() -> TestResult {
        let script = Arc::new(Script {
            panicking: AtomicUsize::new(1),
            ..Script::default()
        });
        let built = Arc::new(AtomicUsize::new(0));
        let factory: Factory = {
            let script = Arc::clone(&script);
            let built = Arc::clone(&built);
            Box::new(move || {
                if built.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(Worker(Arc::clone(&script)))
                } else {
                    Err("synthetic rebuild failure".to_owned())
                }
            })
        };
        let queue = MemoryDurableQueue::default();
        let mut handler = DurableParallelUpdateHandler::start(1, 2, queue.clone(), factory)?;
        assert_eq!(handler.handle(update(731)), Ok(()));
        wait_until(|| built.load(Ordering::SeqCst) == 2);
        handler.stop();
        assert_eq!(script.calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn redis_durable_queue_implements_the_runtime_port() -> TestResult {
        std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .map_or(Ok(()), exercise_redis_update_queue)
    }

    fn exercise_redis_update_queue(port: u16) -> TestResult {
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or(String::from("127.0.0.1")),
            port,
            // Empty passwords are ignored by the Redis client.
            password: std::env::var("TEST_REDIS_PASSWORD").ok(),
        };
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let update_id = i64::try_from(suffix % 1_000_000_000)?;
        let queue = RedisUpdateQueue::new(&endpoint)?;
        let inserted = DurableUpdateQueue::insert_update(&queue, update_id, "synthetic update");
        assert!(inserted?);
        let duplicate = DurableUpdateQueue::insert_update(&queue, update_id, "synthetic duplicate");
        assert!(!duplicate?);
        assert!(
            DurableUpdateQueue::list_updates(&queue)?
                .iter()
                .any(|queued| queued.update_id == update_id)
        );
        DurableUpdateQueue::replace_update(&queue, update_id, "synthetic replacement")?;
        DurableUpdateQueue::quarantine_update(&queue, update_id, "synthetic dead update")?;
        assert_eq!(DurableUpdateQueue::delete_updates(&queue, &[update_id])?, 0);
        Ok(())
    }

    #[test]
    fn unencodable_records_name_what_could_not_be_encoded() {
        // JSON objects need string keys, so a tuple-keyed map cannot be encoded.
        let unencodable = std::collections::BTreeMap::from([((1, 2), "synthetic")]);
        let encoded = super::encode_record(&unencodable, "synthetic record");
        assert!(
            matches!(&encoded, Err(error) if error.starts_with("could not encode synthetic record: ")),
            "{encoded:?}"
        );
        assert_eq!(
            DurableParallelHandlerError::Serialization(
                "could not encode durable update: synthetic".to_owned()
            )
            .to_string(),
            "could not encode durable update: synthetic"
        );
    }
}
