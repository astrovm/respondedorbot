//! Long-poll state ownership and ordered update dispatch.

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
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;

    use bot_adapters::telegram_http::TransportFailureKind;
    use bot_adapters::telegram_polling::{IncomingEvent, IncomingUpdate, PollFailure};

    use super::{PollingRuntime, RuntimeError, StepOutcome, UpdateHandler, UpdateSource};

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
}
