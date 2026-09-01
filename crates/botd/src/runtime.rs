//! Long-poll state ownership and update dispatch.

use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use bot_adapters::telegram_polling::{
    IncomingUpdate, PollFailure, PollOutcome, PollingError, next_offset,
};
use thiserror::Error;

pub trait UpdateSource {
    fn poll(&mut self, offset: Option<i64>) -> Result<PollOutcome, PollingError>;
}

pub trait UpdateHandler {
    type Error;

    fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error>;

    fn take_failure(&mut self) -> Option<UpdateFailure> {
        None
    }

    fn shutdown(&mut self) {}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateFailure {
    pub update_id: i64,
    pub error: String,
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
    failures: Receiver<UpdateFailure>,
    workers: Vec<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
    handler: PhantomData<fn() -> Handler>,
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
        let (failure_sender, failure_receiver) = mpsc::channel();
        let (startup_sender, startup_receiver) = mpsc::channel();
        let factory = Arc::new(factory);
        let stopping = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(worker_count);

        for worker in 0..worker_count {
            let update_receiver = update_receiver.clone();
            let failure_sender = failure_sender.clone();
            let startup_sender = startup_sender.clone();
            let factory = factory.clone();
            let stopping = stopping.clone();
            workers.push(thread::spawn(move || {
                let mut handler = match factory() {
                    Ok(handler) => {
                        let _ = startup_sender.send((worker, None));
                        handler
                    }
                    Err(error) => {
                        let _ = startup_sender.send((worker, Some(error.to_string())));
                        return;
                    }
                };
                loop {
                    if stopping.load(Ordering::Acquire) {
                        return;
                    }
                    let update = match update_receiver.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => return,
                    };
                    let Ok(update) = update else {
                        return;
                    };
                    if stopping.load(Ordering::Acquire) {
                        return;
                    }
                    let update_id = update.update_id;
                    if let Err(error) = handler.handle(update) {
                        let _ = failure_sender.send(UpdateFailure {
                            update_id,
                            error: error.to_string(),
                        });
                    }
                }
            }));
        }
        drop(startup_sender);
        drop(failure_sender);

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
            failures: failure_receiver,
            workers,
            stopping,
            handler: PhantomData,
        })
    }

    fn stop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.updates.take();
        join_workers(&mut self.workers);
    }
}

impl<Handler> UpdateHandler for ParallelUpdateHandler<Handler> {
    type Error = ParallelHandlerError;

    fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
        self.updates
            .as_ref()
            .ok_or(ParallelHandlerError::QueueUnavailable)?
            .send(update)
            .map_err(|_| ParallelHandlerError::QueueUnavailable)
    }

    fn take_failure(&mut self) -> Option<UpdateFailure> {
        self.failures.try_recv().ok()
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

fn join_workers(workers: &mut Vec<JoinHandle<()>>) {
    for worker in workers.drain(..) {
        let _ = worker.join();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    Idle,
    Synchronized { dropped: usize },
    Dispatched { count: usize },
    Retry(PollFailure),
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum RuntimeError<HandlerError> {
    #[error(transparent)]
    Poll(#[from] PollingError),
    #[error("update handler failed for update {update_id}")]
    Handler {
        update_id: i64,
        handler_error: HandlerError,
    },
    #[error("update handler failed for update {update_id}: {handler_error}")]
    BackgroundHandler {
        update_id: i64,
        handler_error: String,
    },
}

pub struct PollingRuntime<Source, Handler> {
    source: Source,
    handler: Handler,
    offset: Option<i64>,
    synchronized: bool,
}

impl<Source, Handler> PollingRuntime<Source, Handler>
where
    Source: UpdateSource,
    Handler: UpdateHandler,
{
    #[must_use]
    pub const fn new(source: Source, handler: Handler) -> Self {
        Self {
            source,
            handler,
            offset: None,
            synchronized: false,
        }
    }

    #[must_use]
    pub const fn offset(&self) -> Option<i64> {
        self.offset
    }

    pub fn step(&mut self) -> Result<StepOutcome, RuntimeError<Handler::Error>> {
        if let Some(failure) = self.handler.take_failure() {
            return Err(RuntimeError::BackgroundHandler {
                update_id: failure.update_id,
                handler_error: failure.error,
            });
        }
        let poll_offset = if self.synchronized {
            self.offset
        } else {
            // Telegram interprets -1 as "return only the newest queued update and
            // forget everything before it". The returned update is discarded too,
            // preserving the retired runtime's drop_pending_updates behavior.
            Some(-1)
        };
        match self.source.poll(poll_offset)? {
            PollOutcome::Retry(failure) => Ok(StepOutcome::Retry(failure)),
            PollOutcome::Updates(updates) if !self.synchronized => {
                self.offset = next_offset(&updates, self.offset);
                self.synchronized = true;
                Ok(StepOutcome::Synchronized {
                    dropped: updates.len(),
                })
            }
            PollOutcome::Updates(updates) if updates.is_empty() => Ok(StepOutcome::Idle),
            PollOutcome::Updates(updates) => {
                let mut count = 0;
                for update in updates {
                    let update_id = update.update_id;
                    let handled = self.handler.handle(update);
                    self.offset = update_id.checked_add(1).or(self.offset);
                    handled.map_err(|handler_error| RuntimeError::Handler {
                        update_id,
                        handler_error,
                    })?;
                    count += 1;
                }
                Ok(StepOutcome::Dispatched { count })
            }
        }
    }

    pub fn shutdown(&mut self) {
        self.handler.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    use bot_adapters::telegram_http::TransportFailureKind;
    use bot_adapters::telegram_polling::{
        IncomingEvent, IncomingMessage, IncomingUpdate, PollFailure,
    };
    use bot_core::telegram_input::ChatId;

    use super::{
        ParallelUpdateHandler, PollingRuntime, RuntimeError, StepOutcome, UpdateHandler,
        UpdateSource,
    };

    struct Source {
        outcomes:
            VecDeque<Result<bot_adapters::telegram_polling::PollOutcome, super::PollingError>>,
        offsets: Vec<Option<i64>>,
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
    fn first_poll_discards_pending_updates_and_synchronizes_the_offset() {
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
        assert_eq!(runtime.step(), Ok(StepOutcome::Synchronized { dropped: 2 }));
        assert_eq!(runtime.offset(), Some(9));
        assert!(runtime.handler.handled.is_empty());
        assert_eq!(runtime.step(), Ok(StepOutcome::Dispatched { count: 1 }));
        assert_eq!(runtime.offset(), Some(10));
        assert_eq!(runtime.step(), Ok(StepOutcome::Idle));
        assert_eq!(runtime.source.offsets, vec![Some(-1), Some(9), Some(10)]);
        assert_eq!(runtime.handler.handled, vec![9]);
    }

    #[test]
    fn failed_update_is_acknowledged_so_later_updates_can_continue() {
        let source = Source {
            outcomes: VecDeque::from([
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(
                    Vec::new(),
                )),
                Ok(bot_adapters::telegram_polling::PollOutcome::Updates(vec![
                    update(10),
                    update(11),
                ])),
            ]),
            offsets: Vec::new(),
        };
        let handler = Handler {
            fail_on: Some(11),
            ..Handler::default()
        };
        let mut runtime = PollingRuntime::new(source, handler);
        assert_eq!(runtime.step(), Ok(StepOutcome::Synchronized { dropped: 0 }));
        assert!(matches!(
            runtime.step(),
            Err(RuntimeError::Handler { update_id: 11, .. })
        ));
        assert_eq!(runtime.offset(), Some(12));
        assert_eq!(runtime.handler.handled, vec![10]);
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
        assert_eq!(runtime.source.offsets, vec![Some(-1), Some(-1)]);
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
        assert_eq!(runtime.step(), Ok(StepOutcome::Synchronized { dropped: 0 }));
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
    fn shutdown_finishes_active_work_without_draining_the_backlog() {
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

        let stopping = handler.stopping.clone();
        let shutdown = thread::spawn(move || handler.shutdown());
        let deadline = Instant::now() + Duration::from_secs(1);
        while !stopping.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(stopping.load(Ordering::Acquire));
        if let Ok(mut released) = release.0.lock() {
            *released = true;
            release.1.notify_all();
        }
        assert!(shutdown.join().is_ok());
        assert!(started_receiver.try_recv().is_err());
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
        let deadline = Instant::now() + Duration::from_secs(1);
        let failure = loop {
            if let Some(failure) = handler.take_failure() {
                break Some(failure);
            }
            if Instant::now() >= deadline {
                break None;
            }
            thread::yield_now();
        };
        handler.shutdown();
        assert!(failure.is_some_and(|failure| {
            failure.update_id == 42 && failure.error == "synthetic parallel failure"
        }));
    }
}
