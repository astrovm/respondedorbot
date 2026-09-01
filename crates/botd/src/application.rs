//! Process composition and graceful lifecycle.

use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use bot_adapters::openrouter_chat::DEFAULT_OPENROUTER_BASE_URL;
use bot_adapters::telegram_http::ReqwestTelegramTransport;
use bot_adapters::telegram_polling::PollFailure;
use bot_core::locale::Locale;
use bot_core::telegram_commands::command_publication_actions;

use crate::background::{
    BackgroundSupervisor, ProductionBackgroundOptions, build_production_background_specs,
};
use crate::composition::{NativeRuntimeOptions, TelegramActionSink, build_native_runtime};
use crate::config::ProductionConfig;
use crate::dispatcher::ActionSink;
use crate::operational_reporting::{
    NoopOperationalReporter, OperationalReport, OperationalReporter, TelegramOperationalReporter,
};
use crate::reconciliation::ActiveOperationRegistry;
use crate::runtime::{PollingRuntime, RuntimeError, StepOutcome, UpdateHandler, UpdateSource};
use crate::scheduler::SchedulerMode;

#[must_use]
pub fn retry_delay(failure: &PollFailure) -> Duration {
    match failure {
        PollFailure::RateLimited {
            retry_after_seconds: Some(seconds),
        } => Duration::from_secs((*seconds).max(1)),
        PollFailure::Transport { .. }
        | PollFailure::Http { .. }
        | PollFailure::Conflict
        | PollFailure::RateLimited { .. }
        | PollFailure::Api { .. } => Duration::from_secs(1),
    }
}

pub fn publish_commands<S>(sink: &mut S) -> Vec<OperationalReport>
where
    S: ActionSink,
    S::Error: Display,
{
    let mut diagnostics = Vec::new();
    for action in command_publication_actions() {
        if let Err(error) = sink.execute(action) {
            diagnostics.push(OperationalReport::new(
                format!("falló la publicación de comandos de Telegram: {error}"),
                format!("Telegram command publication failed: {error}"),
            ));
            break;
        }
    }
    diagnostics
}

pub fn run_polling_until<Source, Handler, Stop, Wait, ReportRetry, ReportHandler>(
    runtime: &mut PollingRuntime<Source, Handler>,
    mut should_stop: Stop,
    mut wait: Wait,
    mut report_poll_retry: ReportRetry,
    mut report_handler_failure: ReportHandler,
) -> Result<(), String>
where
    Source: UpdateSource,
    Handler: UpdateHandler,
    Handler::Error: Display,
    Stop: FnMut() -> bool,
    Wait: FnMut(Duration),
    ReportRetry: FnMut(&PollFailure),
    ReportHandler: FnMut(i64, &str),
{
    let mut last_poll_failure = None;
    while !should_stop() {
        match runtime.step() {
            Ok(StepOutcome::Retry(failure)) => {
                if last_poll_failure.as_ref() != Some(&failure) {
                    report_poll_retry(&failure);
                }
                last_poll_failure = Some(failure.clone());
                wait(retry_delay(&failure));
            }
            Ok(
                StepOutcome::Idle
                | StepOutcome::Synchronized { .. }
                | StepOutcome::Dispatched { .. },
            ) => last_poll_failure = None,
            Err(RuntimeError::Handler {
                update_id,
                handler_error,
            }) => {
                last_poll_failure = None;
                report_handler_failure(update_id, &handler_error.to_string());
            }
            Err(error @ RuntimeError::Poll(_)) => return Err(error.to_string()),
        }
    }
    Ok(())
}

fn interruptible_wait(stopping: &AtomicBool, duration: Duration) {
    let deadline = Instant::now() + duration;
    while !stopping.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        thread::sleep(remaining.min(Duration::from_millis(100)));
    }
}

fn build_operational_reporter(
    config: &ProductionConfig,
) -> Result<Arc<dyn OperationalReporter>, String> {
    let Some(admin_chat_id) = config.admin_user_id else {
        return Ok(Arc::new(NoopOperationalReporter));
    };
    let transport = ReqwestTelegramTransport::new()
        .map_err(|error| format!("could not construct admin reporting transport: {error:?}"))?;
    let secrets = [
        Some(config.runtime.telegram_token()),
        Some(config.database_url()),
        config.redis_endpoint.password.as_deref(),
        Some(config.coinmarketcap_key()),
        config.giphy_api_key(),
        Some(config.openrouter_api_key()),
        config.groq_free_api_key(),
        config.groq_api_key(),
        config.firecrawl_api_key(),
        Some(config.system_prompt.as_str()),
    ]
    .into_iter()
    .flatten()
    .map(str::to_owned);
    Ok(Arc::new(TelegramOperationalReporter::new(
        transport,
        config.runtime.telegram_token(),
        admin_chat_id,
        config.instance_name.as_deref(),
        secrets,
        Locale::Es,
    )))
}

fn report_best_effort(reporter: &dyn OperationalReporter, report: &OperationalReport) {
    if let Err(error) = reporter.report(report) {
        eprintln!("could not deliver operational report: {error}");
    }
}

pub fn run_production(config: &ProductionConfig) -> Result<(), String> {
    let reporter = build_operational_reporter(config)?;
    let active_operations = ActiveOperationRegistry::default();
    let openrouter_base_url = config
        .openrouter_base_url
        .as_deref()
        .unwrap_or(DEFAULT_OPENROUTER_BASE_URL);
    let mut runtime = build_native_runtime(NativeRuntimeOptions {
        token: config.runtime.telegram_token(),
        database_url: config.database_url(),
        bot_name: &config.bot_name,
        instance_name: config.instance_name.clone(),
        redis_endpoint: &config.redis_endpoint,
        long_poll_timeout: config.runtime.long_poll_timeout,
        admin_user_id: config.admin_user_id,
        coinmarketcap_key: Some(config.coinmarketcap_key().to_owned()),
        giphy_api_key: config.giphy_api_key().map(str::to_owned),
        openrouter_api_key: Some(config.openrouter_api_key().to_owned()),
        openrouter_base_url: config.openrouter_base_url.clone(),
        groq_free_api_key: config.groq_free_api_key().map(str::to_owned),
        groq_api_key: config.groq_api_key().map(str::to_owned),
        firecrawl_api_key: config.firecrawl_api_key().map(str::to_owned),
        system_prompt: Some(config.system_prompt.clone()),
        trigger_words: Some(config.trigger_words.clone()),
        active_operations: active_operations.clone(),
    })
    .map_err(|error| error.to_string())?;
    let specs = build_production_background_specs(ProductionBackgroundOptions {
        redis_endpoint: &config.redis_endpoint,
        database_url: config.database_url(),
        telegram_token: config.runtime.telegram_token(),
        openrouter_api_key: config.openrouter_api_key(),
        openrouter_base_url,
        system_prompt: &config.system_prompt,
        owner_token: &config.owner_token(),
        scheduler_mode: SchedulerMode::Authoritative,
        reconciliation_interval: config.reconciliation_interval,
        reconciliation_settings: config.reconciliation_settings,
        active_operations,
        coinmarketcap_key: Some(config.coinmarketcap_key()),
    })?;
    let mut supervisor =
        BackgroundSupervisor::start(specs, reporter.clone()).map_err(|error| error.to_string())?;

    let command_transport = ReqwestTelegramTransport::new()
        .map_err(|error| format!("could not construct command publication transport: {error:?}"))?;
    let mut command_sink =
        TelegramActionSink::new(command_transport, config.runtime.telegram_token());
    for diagnostic in publish_commands(&mut command_sink) {
        eprintln!("{}", diagnostic.english());
        report_best_effort(reporter.as_ref(), &diagnostic);
    }

    let stopping = Arc::new(AtomicBool::new(false));
    let signal_stopping = stopping.clone();
    ctrlc::set_handler(move || signal_stopping.store(true, Ordering::Release))
        .map_err(|error| format!("could not install shutdown signal handler: {error}"))?;
    let polling_result = run_polling_until(
        &mut runtime,
        || stopping.load(Ordering::Acquire),
        |duration| interruptible_wait(&stopping, duration),
        |failure| {
            let report = OperationalReport::new(
                format!("reintento del sondeo de Telegram: {failure:?}"),
                format!("Telegram polling retry: {failure:?}"),
            );
            eprintln!("{}", report.english());
            report_best_effort(reporter.as_ref(), &report);
        },
        |update_id, error| {
            let report = OperationalReport::new(
                format!("falló la actualización {update_id} de Telegram: {error}"),
                format!("Telegram update {update_id} failed: {error}"),
            );
            eprintln!("{}", report.english());
            report_best_effort(reporter.as_ref(), &report);
        },
    );
    if let Err(error) = &polling_result {
        let report = OperationalReport::new(
            format!("falló el proceso de sondeo de Telegram: {error}"),
            format!("Telegram polling runtime failed: {error}"),
        );
        report_best_effort(reporter.as_ref(), &report);
    }
    let shutdown_result = supervisor.stop().map_err(|error| error.to_string());
    polling_result.and(shutdown_result)
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::time::Duration;

    use bot_adapters::telegram_http::TransportFailureKind;
    use bot_adapters::telegram_polling::{
        IncomingEvent, IncomingUpdate, PollFailure, PollOutcome, PollingError,
    };
    use bot_core::locale::Locale;
    use bot_core::telegram_actions::TelegramAction;

    use super::{publish_commands, retry_delay, run_polling_until};
    use crate::dispatcher::{ActionReceipt, ActionSink};
    use crate::runtime::{PollingRuntime, UpdateHandler, UpdateSource};

    #[derive(Default)]
    struct Sink {
        actions: Vec<TelegramAction>,
        fail_after: Option<usize>,
    }

    impl ActionSink for Sink {
        type Error = &'static str;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            if self.fail_after == Some(self.actions.len()) {
                return Err("synthetic publication failure");
            }
            self.actions.push(action);
            Ok(ActionReceipt { message_id: None })
        }
    }

    #[test]
    fn publishes_default_spanish_and_english_command_menus_in_order() {
        let mut sink = Sink::default();
        assert!(publish_commands(&mut sink).is_empty());
        assert_eq!(sink.actions.len(), 3);
        assert!(matches!(
            &sink.actions[0],
            TelegramAction::SetCommands {
                language_code: None,
                ..
            }
        ));
        assert!(matches!(
            &sink.actions[1],
            TelegramAction::SetCommands {
                language_code: Some(language),
                ..
            } if language == "es"
        ));
        assert!(matches!(
            &sink.actions[2],
            TelegramAction::SetCommands {
                language_code: Some(language),
                ..
            } if language == "en"
        ));
    }

    #[test]
    fn command_publication_stops_after_failure_without_failing_startup() {
        let mut sink = Sink {
            fail_after: Some(1),
            ..Sink::default()
        };
        let diagnostics = publish_commands(&mut sink);
        assert_eq!(sink.actions.len(), 1);
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0]
                .english()
                .contains("synthetic publication failure")
        );
        assert!(
            diagnostics[0]
                .for_locale(Locale::Es)
                .contains("falló la publicación")
        );
    }

    #[test]
    fn polling_retry_delays_match_rate_limit_and_transient_failure_policy() {
        assert_eq!(
            retry_delay(&PollFailure::RateLimited {
                retry_after_seconds: Some(12),
            }),
            Duration::from_secs(12)
        );
        assert_eq!(
            retry_delay(&PollFailure::Transport {
                failure: TransportFailureKind::Timeout,
            }),
            Duration::from_secs(1)
        );
    }

    #[test]
    fn handler_failures_are_reported_acknowledged_and_do_not_stop_polling() {
        struct Source {
            outcomes: VecDeque<Result<PollOutcome, PollingError>>,
            offsets: Rc<RefCell<Vec<Option<i64>>>>,
        }
        impl UpdateSource for Source {
            fn poll(&mut self, offset: Option<i64>) -> Result<PollOutcome, PollingError> {
                self.offsets.borrow_mut().push(offset);
                self.outcomes
                    .pop_front()
                    .unwrap_or(Ok(PollOutcome::Updates(Vec::new())))
            }
        }
        struct Handler {
            handled: Rc<RefCell<Vec<i64>>>,
        }
        impl UpdateHandler for Handler {
            type Error = &'static str;
            fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
                if update.update_id == 11 {
                    return Err("synthetic action failure");
                }
                self.handled.borrow_mut().push(update.update_id);
                Ok(())
            }
        }
        let update = |update_id| IncomingUpdate {
            update_id,
            event: IncomingEvent::Unsupported,
        };
        let offsets = Rc::new(RefCell::new(Vec::new()));
        let handled = Rc::new(RefCell::new(Vec::new()));
        let source = Source {
            outcomes: VecDeque::from([
                Ok(PollOutcome::Updates(Vec::new())),
                Ok(PollOutcome::Updates(vec![update(10), update(11)])),
                Ok(PollOutcome::Updates(vec![update(12)])),
            ]),
            offsets: offsets.clone(),
        };
        let mut runtime = PollingRuntime::new(
            source,
            Handler {
                handled: handled.clone(),
            },
        );
        let iterations = Cell::new(0);
        let failures = RefCell::new(Vec::new());
        let result = run_polling_until(
            &mut runtime,
            || {
                let current = iterations.get();
                iterations.set(current + 1);
                current >= 3
            },
            |_| {},
            |_| {},
            |update_id, error| failures.borrow_mut().push((update_id, error.to_owned())),
        );
        assert_eq!(result, Ok(()));
        assert_eq!(*handled.borrow(), [10, 12]);
        assert_eq!(
            *failures.borrow(),
            [(11, "synthetic action failure".to_owned())]
        );
        assert_eq!(*offsets.borrow(), [Some(-1), None, Some(12)]);
        assert_eq!(runtime.offset(), Some(13));
    }

    #[test]
    fn polling_retries_are_reported_once_until_a_success_resets_the_failure() {
        struct Source {
            outcomes: VecDeque<Result<PollOutcome, PollingError>>,
        }
        impl UpdateSource for Source {
            fn poll(&mut self, _: Option<i64>) -> Result<PollOutcome, PollingError> {
                self.outcomes
                    .pop_front()
                    .unwrap_or(Ok(PollOutcome::Updates(Vec::new())))
            }
        }
        struct Handler;
        impl UpdateHandler for Handler {
            type Error = &'static str;
            fn handle(&mut self, _: IncomingUpdate) -> Result<(), Self::Error> {
                Ok(())
            }
        }

        let failure = PollFailure::Transport {
            failure: TransportFailureKind::Request,
        };
        let source = Source {
            outcomes: VecDeque::from([
                Ok(PollOutcome::Updates(Vec::new())),
                Ok(PollOutcome::Retry(failure.clone())),
                Ok(PollOutcome::Retry(failure.clone())),
                Ok(PollOutcome::Updates(Vec::new())),
                Ok(PollOutcome::Retry(failure.clone())),
            ]),
        };
        let mut runtime = PollingRuntime::new(source, Handler);
        let iterations = Cell::new(0);
        let reports = RefCell::new(Vec::new());
        let waits = Cell::new(0);
        assert_eq!(
            run_polling_until(
                &mut runtime,
                || {
                    let current = iterations.get();
                    iterations.set(current + 1);
                    current >= 5
                },
                |_| waits.set(waits.get() + 1),
                |failure| reports.borrow_mut().push(failure.clone()),
                |_, _| {},
            ),
            Ok(())
        );
        assert_eq!(*reports.borrow(), [failure.clone(), failure]);
        assert_eq!(waits.get(), 3);
    }
}
