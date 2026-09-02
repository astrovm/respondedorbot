//! Long-poll state ownership and update dispatch.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use bot_adapters::redis_update_queue::{QueuedUpdate, RedisUpdateQueue};
use bot_adapters::telegram_polling::{
    IncomingUpdate, PollFailure, PollOutcome, PollingError, next_offset,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_UPDATE_ATTEMPTS: usize = 3;
const DURABLE_UPDATE_SCHEMA_VERSION: u32 = 1;

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
                    let update = match update_receiver.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => return,
                    };
                    let Ok(update) = update else {
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
            match startup_receiver.recv() {
                Ok((_worker, None)) => {}
                Ok((worker, Some(error))) => {
                    drop(update_sender);
                    join_workers(&mut workers);
                    return Err(ParallelHandlerBuildError::WorkerStartup { worker, error });
                }
                Err(_) => {
                    drop(update_sender);
                    join_workers(&mut workers);
                    return Err(ParallelHandlerBuildError::WorkerStartup {
                        worker: workers.len(),
                        error: "worker stopped during startup".to_owned(),
                    });
                }
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
    fn delete_update(&self, update_id: i64) -> Result<bool, Self::Error>;
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

    fn delete_update(&self, update_id: i64) -> Result<bool, Self::Error> {
        Self::delete_update(self, update_id)
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
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DurableParallelHandlerError {
    #[error("parallel update queue is not available")]
    QueueUnavailable,
    #[error("durable update queue failed: {0}")]
    DurableQueue(String),
    #[error("could not encode durable update: {0}")]
    Serialization(String),
}

pub struct DurableParallelUpdateHandler<Handler, Queue> {
    updates: Option<SyncSender<DurableUpdateRecord>>,
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
        let (startup_sender, startup_receiver) = mpsc::channel();
        let (completion_sender, completion_receiver) = mpsc::channel();
        let factory = Arc::new(factory);
        let mut workers = Vec::with_capacity(worker_count);

        for worker in 0..worker_count {
            let update_receiver = update_receiver.clone();
            let startup_sender = startup_sender.clone();
            let completion_sender = completion_sender.clone();
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
                    let record = match update_receiver.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => return,
                    };
                    let Ok(record) = record else { return };
                    let result = panic::catch_unwind(AssertUnwindSafe(|| {
                        handler.handle(record.update.clone())
                    }));
                    let (error, panicked) = match result {
                        Ok(result) => (result.err().map(|error| error.to_string()), false),
                        Err(_) => (Some("update handler panicked".to_owned()), true),
                    };
                    let _ = completion_sender.send(DurableUpdateCompletion { record, error });
                    if panicked {
                        return;
                    }
                }
            }));
        }
        drop(startup_sender);
        drop(completion_sender);

        for _ in 0..worker_count {
            match startup_receiver.recv() {
                Ok((_worker, None)) => {}
                Ok((worker, Some(error))) => {
                    drop(update_sender);
                    join_workers(&mut workers);
                    return Err(ParallelHandlerBuildError::WorkerStartup { worker, error });
                }
                Err(_) => {
                    drop(update_sender);
                    join_workers(&mut workers);
                    return Err(ParallelHandlerBuildError::WorkerStartup {
                        worker: workers.len(),
                        error: "worker stopped during startup".to_owned(),
                    });
                }
            }
        }

        Ok(Self {
            updates: Some(update_sender),
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
            self.updates
                .as_ref()
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
            match serde_json::to_string(&record) {
                Ok(payload) => match self.queue.replace_update(update_id, &payload) {
                    Ok(()) => {
                        self.completed.insert(update_id);
                    }
                    Err(queue_error) => self.set_fatal(format!(
                        "could not persist completed durable update: {queue_error}"
                    )),
                },
                Err(serialization_error) => self.set_fatal(format!(
                    "could not encode completed durable update: {serialization_error}"
                )),
            }
            return;
        };

        self.active.remove(&update_id);
        record.attempts = record.attempts.saturating_add(1);
        if record.attempts >= MAX_UPDATE_ATTEMPTS {
            let payload = serde_json::to_string(&DeadUpdateRecord {
                schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
                update: &record.update,
                attempts: record.attempts,
                error: &error,
            });
            match payload {
                Ok(payload) => match self.queue.quarantine_update(update_id, &payload) {
                    Ok(()) => self
                        .failures
                        .quarantined
                        .push(UpdateFailure { update_id, error }),
                    Err(queue_error) => self.set_fatal(format!(
                        "could not quarantine durable update {update_id}: {queue_error}"
                    )),
                },
                Err(serialization_error) => self.set_fatal(format!(
                    "could not encode dead durable update {update_id}: {serialization_error}"
                )),
            }
            return;
        }

        let payload = serde_json::to_string(&record);
        match payload {
            Ok(payload) => match self.queue.replace_update(update_id, &payload) {
                Ok(()) => {
                    self.failures
                        .retrying
                        .push(UpdateFailure { update_id, error });
                    if retry && let Some(sender) = &self.updates {
                        match sender.send(record) {
                            Ok(()) => {
                                self.active.insert(update_id);
                            }
                            Err(_) => self.set_fatal(format!(
                                "could not resubmit durable update {update_id}"
                            )),
                        }
                    }
                }
                Err(queue_error) => self.set_fatal(format!(
                    "could not persist durable update {update_id} retry: {queue_error}"
                )),
            },
            Err(serialization_error) => self.set_fatal(format!(
                "could not encode durable update {update_id} retry: {serialization_error}"
            )),
        }
    }

    fn set_fatal(&mut self, error: String) {
        if self.failures.fatal.is_none() {
            self.failures.fatal = Some(error);
        }
    }

    fn stop(&mut self) {
        self.updates.take();
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
        let payload = serde_json::to_string(&record)
            .map_err(|error| DurableParallelHandlerError::Serialization(error.to_string()))?;
        self.queue
            .insert_update(update_id, &payload)
            .map_err(|error| DurableParallelHandlerError::DurableQueue(error.to_string()))?;
        self.updates
            .as_ref()
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
        let confirmed = self
            .completed
            .iter()
            .copied()
            .filter(|update_id| match confirmation {
                UpdateConfirmation::Before(offset) => *update_id < offset,
                UpdateConfirmation::All => true,
            })
            .collect::<Vec<_>>();
        for update_id in confirmed {
            self.queue
                .delete_update(update_id)
                .map_err(|error| DurableParallelHandlerError::DurableQueue(error.to_string()))?;
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
        self.updates.take();
        join_workers(&mut self.workers);
    }
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
                self.handler
                    .confirm_updates(confirmation)
                    .map_err(|error| RuntimeError::Handler(error.to_string()))?;
                if updates.is_empty() {
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
                for update in updates {
                    let update_id = update.update_id;
                    if self.completed_updates.contains(&update_id) {
                        continue;
                    }
                    attempted.push(update_id);
                    match self.handler.handle(update) {
                        Ok(()) => count += 1,
                        Err(handler_error) => {
                            if self.handler.error_disposition(&handler_error)
                                == HandlerErrorDisposition::StopRuntime
                            {
                                return Err(RuntimeError::Handler(handler_error.to_string()));
                            }
                            failures.push(UpdateFailure {
                                update_id,
                                error: handler_error.to_string(),
                            });
                        }
                    }
                }
                failures.extend(self.handler.finish_batch());
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
                    if *attempts >= MAX_UPDATE_ATTEMPTS {
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
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_adapters::redis_update_queue::QueuedUpdate;
    use bot_adapters::redis_update_queue::RedisUpdateQueue;
    use bot_adapters::telegram_http::TransportFailureKind;
    use bot_adapters::telegram_polling::{
        IncomingEvent, IncomingMessage, IncomingUpdate, PollFailure,
    };
    use bot_core::telegram_input::ChatId;

    use super::{
        DURABLE_UPDATE_SCHEMA_VERSION, DurableParallelUpdateHandler, DurableUpdateCompletion,
        DurableUpdateQueue, DurableUpdateRecord, HandlerErrorDisposition,
        ParallelHandlerBuildError, ParallelHandlerError, ParallelUpdateHandler, PollingRuntime,
        RuntimeError, StepOutcome, UpdateConfirmation, UpdateFailure, UpdateHandler, UpdateSource,
    };

    #[derive(Clone, Default)]
    struct MemoryDurableQueue {
        updates: Arc<Mutex<HashMap<i64, String>>>,
        dead: Arc<Mutex<HashMap<i64, String>>>,
        insert_failures: Arc<AtomicUsize>,
        replace_failures: Arc<AtomicUsize>,
        delete_failures: Arc<AtomicUsize>,
    }

    impl DurableUpdateQueue for MemoryDurableQueue {
        type Error = &'static str;

        fn insert_update(&self, update_id: i64, payload: &str) -> Result<bool, Self::Error> {
            if self
                .insert_failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Err("synthetic insert failure");
            }
            let mut updates = self.updates.lock().map_err(|_| "queue lock poisoned")?;
            if updates.contains_key(&update_id) {
                return Ok(false);
            }
            updates.insert(update_id, payload.to_owned());
            Ok(true)
        }

        fn list_updates(&self) -> Result<Vec<QueuedUpdate>, Self::Error> {
            let updates = self.updates.lock().map_err(|_| "queue lock poisoned")?;
            Ok(updates
                .iter()
                .map(|(update_id, payload)| QueuedUpdate {
                    update_id: *update_id,
                    payload: payload.clone(),
                })
                .collect())
        }

        fn replace_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error> {
            if self
                .replace_failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Err("synthetic replace failure");
            }
            self.updates
                .lock()
                .map_err(|_| "queue lock poisoned")?
                .insert(update_id, payload.to_owned());
            Ok(())
        }

        fn delete_update(&self, update_id: i64) -> Result<bool, Self::Error> {
            if self
                .delete_failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Err("synthetic delete failure");
            }
            Ok(self
                .updates
                .lock()
                .map_err(|_| "queue lock poisoned")?
                .remove(&update_id)
                .is_some())
        }

        fn quarantine_update(&self, update_id: i64, payload: &str) -> Result<(), Self::Error> {
            self.dead
                .lock()
                .map_err(|_| "queue lock poisoned")?
                .insert(update_id, payload.to_owned());
            self.updates
                .lock()
                .map_err(|_| "queue lock poisoned")?
                .remove(&update_id);
            Ok(())
        }
    }

    struct Source {
        outcomes:
            VecDeque<Result<bot_adapters::telegram_polling::PollOutcome, super::PollingError>>,
        offsets: Vec<Option<i64>>,
    }

    #[test]
    fn durable_completion_retries_quarantines_and_preserves_queue_failures() {
        struct Handler;
        impl UpdateHandler for Handler {
            type Error = Infallible;

            fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
                Ok(())
            }
        }

        let queue = MemoryDurableQueue::default();
        let mut handler = DurableParallelUpdateHandler::start(1, 2, queue.clone(), || {
            Ok::<_, Infallible>(Handler)
        })
        .unwrap_or_else(|_| unreachable!());
        handler.handle_completion(
            DurableUpdateCompletion {
                record: DurableUpdateRecord {
                    schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
                    update: update(801),
                    attempts: 0,
                    completed: false,
                },
                error: Some("synthetic retry".to_owned()),
            },
            false,
        );
        assert_eq!(handler.failures.retrying.len(), 1);
        assert!(
            queue
                .updates
                .lock()
                .is_ok_and(|updates| updates.contains_key(&801))
        );

        handler.handle_completion(
            DurableUpdateCompletion {
                record: DurableUpdateRecord {
                    schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
                    update: update(802),
                    attempts: super::MAX_UPDATE_ATTEMPTS - 1,
                    completed: false,
                },
                error: Some("synthetic terminal failure".to_owned()),
            },
            false,
        );
        assert_eq!(handler.failures.quarantined.len(), 1);
        assert!(queue.dead.lock().is_ok_and(|dead| dead.contains_key(&802)));

        queue.replace_failures.store(1, Ordering::SeqCst);
        handler.handle_completion(
            DurableUpdateCompletion {
                record: DurableUpdateRecord {
                    schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
                    update: update(803),
                    attempts: 0,
                    completed: false,
                },
                error: Some("synthetic persistence failure".to_owned()),
            },
            false,
        );
        assert!(handler.failures.fatal.is_some());
        handler.stop();
    }

    impl UpdateSource for Source {
        fn poll(
            &mut self,
            offset: Option<i64>,
        ) -> Result<bot_adapters::telegram_polling::PollOutcome, super::PollingError> {
            self.offsets.push(offset);
            self.outcomes.pop_front().unwrap_or(Ok(
                bot_adapters::telegram_polling::PollOutcome::Updates(Vec::new()),
            ))
        }
    }

    #[derive(Default)]
    struct Handler {
        handled: Vec<i64>,
        fail_on: Option<i64>,
    }

    impl UpdateHandler for Handler {
        type Error = &'static str;

        fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
            if self.fail_on == Some(update.update_id) {
                return Err("synthetic handler failure");
            }
            self.handled.push(update.update_id);
            Ok(())
        }
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

    #[test]
    fn first_poll_processes_pending_updates_before_advancing_the_offset() {
        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(7),
                    update(8),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(9),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(
                    Vec::new(),
                )),
            ]),
            offsets: Vec::new(),
        };
        let mut runtime = PollingRuntime::new(source, Handler::default());
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
        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(10),
                    update(11),
                    update(12),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(11),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(11),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(13),
                ])),
            ]),
            offsets: Vec::new(),
        };
        let handler = Handler {
            fail_on: Some(11),
            ..Handler::default()
        };
        let mut runtime = PollingRuntime::new(source, handler);
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
    fn retries_and_poll_errors_leave_offset_unchanged() {
        let retry = PollFailure::Transport {
            failure: TransportFailureKind::Timeout,
        };
        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Retry(
                    retry.clone(),
                )),
                Err(super::PollingError::InvalidResponse),
            ]),
            offsets: Vec::new(),
        };
        let mut runtime = PollingRuntime::new(source, Handler::default());
        assert_eq!(runtime.step(), Ok(StepOutcome::Retry(retry)));
        assert!(matches!(runtime.step(), Err(RuntimeError::Poll(_))));
        assert_eq!(runtime.offset(), None);
        assert_eq!(runtime.source.offsets, vec![None, None]);
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
        let source = Source {
            outcomes: VecDeque::new(),
            offsets: Vec::new(),
        };
        let mut runtime = PollingRuntime::new(source, InfallibleHandler);
        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
    }

    #[test]
    fn durable_handler_polls_and_starts_later_updates_while_earlier_work_is_running() {
        struct BlockingHandler {
            started: mpsc::Sender<i64>,
            release: Arc<(Mutex<bool>, Condvar)>,
        }

        impl UpdateHandler for BlockingHandler {
            type Error = Infallible;

            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                let _ = self.started.send(update.update_id);
                let (released, wake) = &*self.release;
                if let Ok(guard) = released.lock() {
                    let _guard = wake.wait_while(guard, |released| !*released);
                }
                Ok(())
            }
        }

        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(1),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(2),
                ])),
            ]),
            offsets: Vec::new(),
        };
        let (started_sender, started_receiver) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let factory_release = release.clone();
        let handler =
            DurableParallelUpdateHandler::start(2, 4, MemoryDurableQueue::default(), move || {
                Ok::<_, Infallible>(BlockingHandler {
                    started: started_sender.clone(),
                    release: factory_release.clone(),
                })
            });
        assert!(handler.is_ok());
        let Ok(handler) = handler else { return };
        let mut runtime = PollingRuntime::new(source, handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(started_receiver.recv_timeout(Duration::from_secs(1)), Ok(1));
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(started_receiver.recv_timeout(Duration::from_secs(1)), Ok(2));
        assert_eq!(runtime.source.offsets, [None, Some(2)]);

        if let Ok(mut released) = release.0.lock() {
            *released = true;
            release.1.notify_all();
        }
        runtime.shutdown();
    }

    #[test]
    fn durable_admission_failure_stops_without_advancing_the_telegram_offset() {
        struct InfallibleHandler;
        impl UpdateHandler for InfallibleHandler {
            type Error = Infallible;

            fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
                Ok(())
            }
        }

        let source = Source {
            outcomes: VecDeque::from([Ok(bot_adapters::telegram_polling::PollOutcome::Updates(
                vec![update(41)],
            ))]),
            offsets: Vec::new(),
        };
        let queue = MemoryDurableQueue::default();
        queue.insert_failures.store(1, Ordering::SeqCst);
        let handler = DurableParallelUpdateHandler::start(1, 2, queue.clone(), || {
            Ok::<_, Infallible>(InfallibleHandler)
        });
        assert!(handler.is_ok());
        let Ok(handler) = handler else { return };
        let mut runtime = PollingRuntime::new(source, handler);

        assert!(matches!(runtime.step(), Err(RuntimeError::Handler(_))));
        assert_eq!(runtime.offset(), None);
        assert_eq!(runtime.source.offsets, [None]);
        assert!(queue.updates.lock().is_ok_and(|updates| updates.is_empty()));
        runtime.shutdown();
    }

    #[test]
    fn completed_update_is_retained_until_telegram_confirms_it() {
        struct RecordingHandler {
            processed: mpsc::Sender<i64>,
        }
        impl UpdateHandler for RecordingHandler {
            type Error = Infallible;

            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                let _ = self.processed.send(update.update_id);
                Ok(())
            }
        }

        let retry = PollFailure::Transport {
            failure: TransportFailureKind::Timeout,
        };
        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(51),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Retry(
                    retry.clone(),
                )),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(
                    Vec::new(),
                )),
            ]),
            offsets: Vec::new(),
        };
        let queue = MemoryDurableQueue::default();
        let (processed_sender, processed_receiver) = mpsc::channel();
        let handler = DurableParallelUpdateHandler::start(1, 2, queue.clone(), move || {
            Ok::<_, Infallible>(RecordingHandler {
                processed: processed_sender.clone(),
            })
        });
        assert!(handler.is_ok());
        let Ok(handler) = handler else { return };
        let mut runtime = PollingRuntime::new(source, handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(
            processed_receiver.recv_timeout(Duration::from_secs(1)),
            Ok(51)
        );
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let failures = runtime.handler.take_background_failures();
            assert!(failures.fatal.is_none());
            let completed = queue
                .updates
                .lock()
                .ok()
                .and_then(|updates| updates.get(&51).cloned())
                .and_then(|payload| serde_json::from_str::<DurableUpdateRecord>(&payload).ok())
                .is_some_and(|record| record.completed);
            if completed {
                break;
            }
            if Instant::now() >= deadline {
                panic!("completed durable update was not persisted");
            }
            thread::yield_now();
        }
        assert_eq!(runtime.step(), Ok(StepOutcome::Retry(retry)));
        assert!(
            queue
                .updates
                .lock()
                .is_ok_and(|updates| updates.contains_key(&51))
        );

        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
        assert!(queue.updates.lock().is_ok_and(|updates| updates.is_empty()));
        assert_eq!(
            processed_receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        );
        runtime.shutdown();
    }

    #[test]
    fn recovered_completed_update_is_not_replayed_before_confirmation() {
        struct RecordingHandler {
            processed: mpsc::Sender<i64>,
        }
        impl UpdateHandler for RecordingHandler {
            type Error = Infallible;

            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                let _ = self.processed.send(update.update_id);
                Ok(())
            }
        }

        let queue = MemoryDurableQueue::default();
        let persisted = serde_json::to_string(&DurableUpdateRecord {
            schema_version: DURABLE_UPDATE_SCHEMA_VERSION,
            update: update(61),
            attempts: 0,
            completed: true,
        });
        assert!(persisted.is_ok());
        let Ok(persisted) = persisted else { return };
        assert!(queue.insert_update(61, &persisted).is_ok());
        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(61),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(
                    Vec::new(),
                )),
            ]),
            offsets: Vec::new(),
        };
        let (processed_sender, processed_receiver) = mpsc::channel();
        let handler = DurableParallelUpdateHandler::start(1, 2, queue.clone(), move || {
            Ok::<_, Infallible>(RecordingHandler {
                processed: processed_sender.clone(),
            })
        });
        assert!(handler.is_ok());
        let Ok(handler) = handler else { return };
        let mut runtime = PollingRuntime::new(source, handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert!(
            queue
                .updates
                .lock()
                .is_ok_and(|updates| updates.contains_key(&61))
        );
        assert_eq!(
            processed_receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        );

        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
        assert!(queue.updates.lock().is_ok_and(|updates| updates.is_empty()));
        assert_eq!(runtime.source.offsets, [None, Some(62)]);
        assert_eq!(
            processed_receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        );
        runtime.shutdown();
    }

    #[test]
    fn confirmation_failure_stops_runtime_with_completed_update_intact() {
        struct RecordingHandler {
            processed: mpsc::Sender<i64>,
        }
        impl UpdateHandler for RecordingHandler {
            type Error = Infallible;

            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                let _ = self.processed.send(update.update_id);
                Ok(())
            }
        }

        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(66),
                ])),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(
                    Vec::new(),
                )),
            ]),
            offsets: Vec::new(),
        };
        let queue = MemoryDurableQueue::default();
        let (processed_sender, processed_receiver) = mpsc::channel();
        let handler = DurableParallelUpdateHandler::start(1, 2, queue.clone(), move || {
            Ok::<_, Infallible>(RecordingHandler {
                processed: processed_sender.clone(),
            })
        });
        assert!(handler.is_ok());
        let Ok(handler) = handler else { return };
        let mut runtime = PollingRuntime::new(source, handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(
            processed_receiver.recv_timeout(Duration::from_secs(1)),
            Ok(66)
        );
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let failures = runtime.handler.take_background_failures();
            assert!(failures.fatal.is_none());
            let completed = queue
                .updates
                .lock()
                .ok()
                .and_then(|updates| updates.get(&66).cloned())
                .and_then(|payload| serde_json::from_str::<DurableUpdateRecord>(&payload).ok())
                .is_some_and(|record| record.completed);
            if completed {
                break;
            }
            if Instant::now() >= deadline {
                panic!("completed durable update was not persisted");
            }
            thread::yield_now();
        }
        queue.delete_failures.store(1, Ordering::SeqCst);

        assert!(matches!(runtime.step(), Err(RuntimeError::Handler(_))));
        assert!(
            queue
                .updates
                .lock()
                .is_ok_and(|updates| updates.contains_key(&66))
        );
        runtime.shutdown();
    }

    #[test]
    fn failed_retry_persistence_stops_runtime_for_recovery() {
        struct FailingHandler {
            processed: mpsc::Sender<i64>,
        }
        impl UpdateHandler for FailingHandler {
            type Error = &'static str;

            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                let _ = self.processed.send(update.update_id);
                Err("synthetic handler failure")
            }
        }

        let source = Source {
            outcomes: VecDeque::from([Ok(bot_adapters::telegram_polling::PollOutcome::Updates(
                vec![update(71)],
            ))]),
            offsets: Vec::new(),
        };
        let queue = MemoryDurableQueue::default();
        queue.replace_failures.store(1, Ordering::SeqCst);
        let (processed_sender, processed_receiver) = mpsc::channel();
        let handler = DurableParallelUpdateHandler::start(1, 2, queue.clone(), move || {
            Ok::<_, Infallible>(FailingHandler {
                processed: processed_sender.clone(),
            })
        });
        assert!(handler.is_ok());
        let Ok(handler) = handler else { return };
        let mut runtime = PollingRuntime::new(source, handler);

        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(
            processed_receiver.recv_timeout(Duration::from_secs(1)),
            Ok(71)
        );
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            match runtime.step() {
                Err(RuntimeError::Handler(error)) => {
                    assert!(error.contains("could not persist durable update 71 retry"));
                    break;
                }
                Ok(_) if Instant::now() < deadline => thread::yield_now(),
                outcome => panic!("durable transition failure did not stop runtime: {outcome:?}"),
            }
        }
        assert!(
            queue
                .updates
                .lock()
                .is_ok_and(|updates| updates.contains_key(&71))
        );
        runtime.shutdown();
    }

    #[test]
    fn parallel_handler_runs_same_chat_and_different_chat_updates_together() {
        struct BlockingHandler {
            started: mpsc::Sender<(i64, i64)>,
            release: Arc<(Mutex<bool>, Condvar)>,
        }
        impl UpdateHandler for BlockingHandler {
            type Error = Infallible;

            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                let chat_id = match update.event {
                    IncomingEvent::Message(message) => message.chat_id.map_or(0, |id| id.0),
                    _ => 0,
                };
                let _ = self.started.send((update.update_id, chat_id));
                let (released, wake) = &*self.release;
                if let Ok(guard) = released.lock() {
                    let _guard = wake.wait_while(guard, |released| !*released);
                }
                Ok(())
            }
        }

        let (started_sender, started_receiver) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let factory_sender = started_sender.clone();
        let factory_release = release.clone();
        let handler = ParallelUpdateHandler::start(4, 4, move || {
            Ok::<_, Infallible>(BlockingHandler {
                started: factory_sender.clone(),
                release: factory_release.clone(),
            })
        });
        assert!(handler.is_ok());
        let Ok(mut handler) = handler else { return };

        for update in [
            chat_update(1, 10),
            chat_update(2, 10),
            chat_update(3, 20),
            chat_update(4, 30),
        ] {
            assert!(handler.handle(update).is_ok());
        }

        let mut started = Vec::new();
        for _ in 0..4 {
            if let Ok(update) = started_receiver.recv_timeout(Duration::from_secs(1)) {
                started.push(update);
            }
        }
        assert_eq!(started.len(), 4);
        assert_eq!(started.iter().filter(|(_, chat)| *chat == 10).count(), 2);
        assert!(started.iter().any(|(_, chat)| *chat == 20));
        assert!(started.iter().any(|(_, chat)| *chat == 30));

        if let Ok(mut released) = release.0.lock() {
            *released = true;
            release.1.notify_all();
        }
        handler.shutdown();
    }

    #[test]
    fn parallel_handler_never_exceeds_its_worker_limit() {
        struct MeasuringHandler {
            active: Arc<AtomicUsize>,
            maximum: Arc<AtomicUsize>,
        }
        impl UpdateHandler for MeasuringHandler {
            type Error = Infallible;

            fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.maximum.fetch_max(active, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(10));
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let factory_active = active.clone();
        let factory_maximum = maximum.clone();
        let handler = ParallelUpdateHandler::start(3, 12, move || {
            Ok::<_, Infallible>(MeasuringHandler {
                active: factory_active.clone(),
                maximum: factory_maximum.clone(),
            })
        });
        assert!(handler.is_ok());
        let Ok(mut handler) = handler else { return };

        for update_id in 1..=12 {
            assert!(handler.handle(update(update_id)).is_ok());
        }
        let deadline = Instant::now() + Duration::from_secs(1);
        while maximum.load(Ordering::SeqCst) < 3 && Instant::now() < deadline {
            thread::yield_now();
        }
        handler.shutdown();
        assert_eq!(maximum.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn shutdown_finishes_active_work_and_drains_the_backlog() {
        struct BlockingHandler {
            started: mpsc::Sender<i64>,
            release: Arc<(Mutex<bool>, Condvar)>,
        }
        impl UpdateHandler for BlockingHandler {
            type Error = Infallible;

            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                let _ = self.started.send(update.update_id);
                let (released, wake) = &*self.release;
                if let Ok(guard) = released.lock() {
                    let _guard = wake.wait_while(guard, |released| !*released);
                }
                Ok(())
            }
        }

        let (started_sender, started_receiver) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let factory_release = release.clone();
        let handler = ParallelUpdateHandler::start(1, 3, move || {
            Ok::<_, Infallible>(BlockingHandler {
                started: started_sender.clone(),
                release: factory_release.clone(),
            })
        });
        assert!(handler.is_ok());
        let Ok(mut handler) = handler else { return };
        assert!(handler.handle(update(1)).is_ok());
        assert_eq!(started_receiver.recv_timeout(Duration::from_secs(1)), Ok(1));
        assert!(handler.handle(update(2)).is_ok());
        assert!(handler.handle(update(3)).is_ok());

        let shutdown = thread::spawn(move || handler.shutdown());
        if let Ok(mut released) = release.0.lock() {
            *released = true;
            release.1.notify_all();
        }
        assert!(shutdown.join().is_ok());
        assert_eq!(started_receiver.recv_timeout(Duration::from_secs(1)), Ok(2));
        assert_eq!(started_receiver.recv_timeout(Duration::from_secs(1)), Ok(3));
    }

    #[test]
    fn parallel_handler_reports_worker_failures() {
        struct FailingHandler;
        impl UpdateHandler for FailingHandler {
            type Error = &'static str;

            fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
                Err("synthetic parallel failure")
            }
        }

        let handler = ParallelUpdateHandler::start(1, 1, || Ok::<_, Infallible>(FailingHandler));
        assert!(handler.is_ok());
        let Ok(mut handler) = handler else { return };
        assert!(handler.handle(update(42)).is_ok());
        let failures = handler.finish_batch();
        handler.shutdown();
        assert!(failures.first().is_some_and(|failure| {
            failure.update_id == 42 && failure.error == "synthetic parallel failure"
        }));
    }

    #[test]
    fn parallel_handler_converts_panics_into_completions() {
        struct PanickingHandler;
        impl UpdateHandler for PanickingHandler {
            type Error = Infallible;

            fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
                panic!("synthetic handler panic")
            }
        }

        let handler = ParallelUpdateHandler::start(1, 1, || Ok::<_, Infallible>(PanickingHandler));
        assert!(handler.is_ok());
        let Ok(mut handler) = handler else { return };
        assert!(handler.handle(update(42)).is_ok());
        assert_eq!(
            handler.finish_batch(),
            [UpdateFailure {
                update_id: 42,
                error: "update handler panicked".to_owned(),
            }]
        );
        handler.shutdown();
    }

    #[test]
    fn parallel_handler_reports_factory_panics_without_hanging_startup() {
        struct FactoryHandler;
        impl UpdateHandler for FactoryHandler {
            type Error = Infallible;

            fn handle(&mut self, _update: IncomingUpdate) -> Result<(), Self::Error> {
                Ok(())
            }
        }

        let handler = ParallelUpdateHandler::<FactoryHandler>::start(2, 2, || {
            panic!("synthetic factory panic");
            #[allow(unreachable_code)]
            Ok::<_, Infallible>(FactoryHandler)
        });
        assert!(matches!(
            handler,
            Err(ParallelHandlerBuildError::WorkerStartup { error, .. })
                if error == "worker factory panicked"
        ));
    }

    #[test]
    fn handler_defaults_and_parallel_validation_are_safe() {
        let mut handler = Handler::default();
        assert_eq!(handler.prepare(), Ok(()));
        assert_eq!(
            handler.error_disposition(&"synthetic"),
            HandlerErrorDisposition::RetryUpdate
        );
        assert_eq!(handler.confirm_updates(UpdateConfirmation::All), Ok(()));
        assert!(handler.finish_batch().is_empty());
        assert_eq!(handler.take_background_failures(), Default::default());
        handler.shutdown();

        assert!(matches!(
            ParallelUpdateHandler::<Handler>::start(0, 1, || Ok::<_, Infallible>(
                Handler::default()
            )),
            Err(ParallelHandlerBuildError::NoWorkers)
        ));
        assert!(matches!(
            ParallelUpdateHandler::<Handler>::start(1, 0, || Ok::<_, Infallible>(
                Handler::default()
            )),
            Err(ParallelHandlerBuildError::EmptyQueue)
        ));
        assert!(matches!(
            ParallelUpdateHandler::<Handler>::start(1, 1, || {
                Err::<Handler, _>("synthetic startup failure")
            }),
            Err(ParallelHandlerBuildError::WorkerStartup { error, .. })
                if error == "synthetic startup failure"
        ));
        assert!(matches!(
            DurableParallelUpdateHandler::<Handler, MemoryDurableQueue>::start(
                0,
                1,
                MemoryDurableQueue::default(),
                || Ok::<_, Infallible>(Handler::default()),
            ),
            Err(ParallelHandlerBuildError::NoWorkers)
        ));
        assert!(matches!(
            DurableParallelUpdateHandler::<Handler, MemoryDurableQueue>::start(
                1,
                0,
                MemoryDurableQueue::default(),
                || Ok::<_, Infallible>(Handler::default()),
            ),
            Err(ParallelHandlerBuildError::EmptyQueue)
        ));
        assert!(matches!(
            DurableParallelUpdateHandler::<Handler, MemoryDurableQueue>::start(
                1,
                1,
                MemoryDurableQueue::default(),
                || Err::<Handler, _>("synthetic durable startup failure"),
            ),
            Err(ParallelHandlerBuildError::WorkerStartup { error, .. })
                if error == "synthetic durable startup failure"
        ));

        let mut stopped =
            ParallelUpdateHandler::start(1, 1, || Ok::<_, Infallible>(Handler::default()))
                .unwrap_or_else(|_| unreachable!());
        stopped.shutdown();
        assert_eq!(
            stopped.handle(update(1)),
            Err(ParallelHandlerError::QueueUnavailable)
        );
    }

    #[test]
    fn redis_durable_queue_implements_the_runtime_port() -> Result<(), String> {
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
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let update_id = i64::try_from(suffix % 1_000_000_000).map_err(|error| error.to_string())?;
        let queue = RedisUpdateQueue::new(&endpoint).map_err(|error| error.to_string())?;
        assert!(
            DurableUpdateQueue::insert_update(&queue, update_id, "synthetic queued update")
                .map_err(|error| error.to_string())?
        );
        assert!(
            !DurableUpdateQueue::insert_update(&queue, update_id, "synthetic duplicate")
                .map_err(|error| error.to_string())?
        );
        assert!(
            DurableUpdateQueue::list_updates(&queue)
                .map_err(|error| error.to_string())?
                .iter()
                .any(|queued| queued.update_id == update_id)
        );
        DurableUpdateQueue::replace_update(&queue, update_id, "synthetic replacement")
            .map_err(|error| error.to_string())?;
        DurableUpdateQueue::quarantine_update(&queue, update_id, "synthetic dead update")
            .map_err(|error| error.to_string())?;
        assert!(
            !DurableUpdateQueue::delete_update(&queue, update_id)
                .map_err(|error| error.to_string())?
        );
        Ok(())
    }
}
