//! Process composition and graceful lifecycle.

use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use bot_adapters::billing_schema::BillingSchemaRepository;
use bot_adapters::chat_config::ChatConfigRepository;
use bot_adapters::openrouter_chat::{DEFAULT_OPENROUTER_BASE_URL, OpenRouterPricingCache};
use bot_adapters::telegram_http::{ReqwestTelegramTransport, TransportFailureKind};
use bot_adapters::telegram_polling::PollFailure;
use bot_core::locale::Locale;
use bot_core::telegram_commands::{chat_command_menu_action, command_publication_actions};
use bot_core::telegram_input::ChatId;

use crate::background::{
    BackgroundSupervisor, ProductionBackgroundOptions, build_production_background_specs,
};
use crate::composition::{
    NativeRuntimeOptions, TelegramActionSink, TelegramDeliveryCoordinator, build_native_runtime,
};
use crate::config::ProductionConfig;
use crate::dispatcher::ActionSink;
use crate::error_text;
use crate::lightning_payments::OpenNodeOptions;
use crate::operational_reporting::{
    NoopOperationalReporter, OperationalReport, OperationalReporter, TelegramOperationalReporter,
};
use crate::reconciliation::ActiveOperationRegistry;
use crate::runtime::{PollingRuntime, StepOutcome, UpdateHandler, UpdateSource};
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
    // Each menu is independent, so one failure must not keep the others
    // (like the all-groups menu) from being published.
    let mut diagnostics = Vec::new();
    for action in command_publication_actions() {
        if let Err(error) = sink.execute(action) {
            diagnostics.push(OperationalReport::new(
                format!("falló la publicación de comandos de Telegram: {error}"),
                format!("Telegram command publication failed: {error}"),
            ));
        }
    }
    diagnostics
}

/// Rewrites the menu of each chat with a fixed language, so chats that got
/// their own menu pick up catalog changes without anyone touching `/config`.
/// One report covers every failed chat.
pub fn refresh_chat_menus<S, E>(
    sink: &mut S,
    chats: Result<Vec<(i64, String)>, E>,
) -> Vec<OperationalReport>
where
    S: ActionSink,
    S::Error: Display,
    E: Display,
{
    let chats = match chats {
        Ok(chats) => chats,
        Err(error) => {
            return vec![OperationalReport::new(
                format!("no pude cargar los chats para actualizar sus menús: {error}"),
                format!("could not load chats to refresh their menus: {error}"),
            )];
        }
    };
    let mut failed = 0_usize;
    let mut last_error = String::new();
    for (chat_id, language) in &chats {
        let locale = bot_core::locale::normalize_locale(language, Locale::Es);
        if let Err(error) = sink.execute(chat_command_menu_action(ChatId(*chat_id), locale)) {
            failed += 1;
            last_error = error.to_string();
        }
    }
    if failed == 0 {
        return Vec::new();
    }
    vec![OperationalReport::new(
        format!(
            "falló la actualización del menú de {failed} de {} chats: {last_error}",
            chats.len()
        ),
        format!(
            "chat command menu refresh failed for {failed} of {} chats: {last_error}",
            chats.len()
        ),
    )]
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
            Ok(StepOutcome::HandlerFailures {
                retrying,
                quarantined,
            }) => {
                last_poll_failure = None;
                for failure in retrying {
                    report_handler_failure(failure.update_id, &failure.error);
                }
                for failure in quarantined {
                    report_handler_failure(
                        failure.update_id,
                        &format!("quarantined without further retries: {}", failure.error),
                    );
                }
            }
            Ok(StepOutcome::Idle | StepOutcome::Dispatched { .. }) => last_poll_failure = None,
            Err(error) => return Err(error_text(error)),
        }
    }
    Ok(())
}

/// Polls until a shutdown signal or `failed` asks to stop, reporting poll
/// retries and update failures to the admin in the background.
fn poll_until_stopped<Source, Handler>(
    runtime: &mut PollingRuntime<Source, Handler>,
    stopping: &AtomicBool,
    failed: impl Fn() -> bool,
    reports: &BackgroundReports,
) -> Result<(), String>
where
    Source: UpdateSource,
    Handler: UpdateHandler,
    Handler::Error: Display,
{
    run_polling_until(
        runtime,
        || stopping.load(Ordering::Acquire) || failed(),
        |duration| interruptible_wait(stopping, duration),
        |failure| log_and_queue(reports, poll_retry_report(failure)),
        |update_id, error| log_and_queue(reports, update_failure_report(update_id, error)),
    )
}

fn telegram_transport(
    built: Result<ReqwestTelegramTransport, TransportFailureKind>,
    purpose: &str,
) -> Result<ReqwestTelegramTransport, String> {
    match built {
        Ok(transport) => Ok(transport),
        Err(error) => Err(format!(
            "could not construct {purpose} transport: {error:?}"
        )),
    }
}

type ShutdownHandler = Box<dyn FnMut() + Send>;
type ShutdownInstaller = fn(ShutdownHandler) -> Result<(), ctrlc::Error>;

fn install_shutdown_handler(
    stopping: &Arc<AtomicBool>,
    install: ShutdownInstaller,
) -> Result<(), String> {
    let signal_stopping = Arc::clone(stopping);
    let handler: ShutdownHandler = Box::new(move || signal_stopping.store(true, Ordering::Release));
    match install(handler) {
        Ok(()) => Ok(()),
        Err(error) => Err(format!(
            "could not install shutdown signal handler: {error}"
        )),
    }
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

fn opennode_options(config: &ProductionConfig) -> Option<OpenNodeOptions> {
    config.opennode_api_key().map(|api_key| OpenNodeOptions {
        api_key: api_key.to_owned(),
        api_url: config.opennode_api_url.clone(),
    })
}

fn build_operational_reporter(
    config: &ProductionConfig,
) -> Result<Arc<dyn OperationalReporter>, String> {
    let Some(admin_chat_id) = config.admin_user_id else {
        return Ok(Arc::new(NoopOperationalReporter));
    };
    let transport = telegram_transport(ReqwestTelegramTransport::new(), "admin reporting")?;
    let secrets = [
        Some(config.runtime.telegram_token()),
        Some(config.database_url()),
        config.redis_endpoint.password.as_deref(),
        Some(config.coinmarketcap_key()),
        config.giphy_api_key(),
        Some(config.openrouter_api_key()),
        config.firecrawl_api_key(),
        config.supadata_api_key(),
        config.apify_api_key(),
        config.opennode_api_key(),
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

fn poll_retry_report(failure: &PollFailure) -> OperationalReport {
    OperationalReport::new(
        format!("reintento del sondeo de Telegram: {failure:?}"),
        format!("Telegram polling retry: {failure:?}"),
    )
}

fn update_failure_report(update_id: i64, error: &str) -> OperationalReport {
    OperationalReport::new(
        format!("falló la actualización {update_id} de Telegram: {error}"),
        format!("Telegram update {update_id} failed: {error}"),
    )
}

fn log_and_queue(reports: &BackgroundReports, report: OperationalReport) {
    eprintln!("{}", report.english());
    reports.report(report);
}

fn report_best_effort(reporter: &dyn OperationalReporter, report: &OperationalReport) {
    if let Err(error) = reporter.report(report) {
        eprintln!("could not deliver operational report: {error}");
    }
}

const OPERATIONAL_REPORT_QUEUE_CAPACITY: usize = 32;

/// Delivers operational reports from a dedicated thread, so a slow or
/// rate-limited admin chat never stalls the polling loop. Reports beyond the
/// bounded queue are dropped (and logged) rather than applying backpressure.
struct BackgroundReports {
    sender: Option<SyncSender<OperationalReport>>,
    worker: Option<JoinHandle<()>>,
}

impl BackgroundReports {
    fn start(reporter: Arc<dyn OperationalReporter>, capacity: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<OperationalReport>(capacity);
        let worker = thread::spawn(move || {
            for report in receiver {
                report_best_effort(reporter.as_ref(), &report);
            }
        });
        Self {
            sender: Some(sender),
            worker: Some(worker),
        }
    }

    fn report(&self, report: OperationalReport) {
        let Some(sender) = &self.sender else {
            return;
        };
        if let Err(TrySendError::Full(report) | TrySendError::Disconnected(report)) =
            sender.try_send(report)
        {
            eprintln!(
                "dropped operational report, the report queue is unavailable: {}",
                report.english()
            );
        }
    }

    /// Delivers every queued report before returning.
    fn shutdown(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for BackgroundReports {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn run_production(config: &ProductionConfig) -> Result<(), String> {
    BillingSchemaRepository::new(config.database_url())
        .ensure_schema()
        .map_err(|error| format!("could not initialize billing schema: {error}"))?;
    let reporter = build_operational_reporter(config)?;
    let active_operations = ActiveOperationRegistry::default();
    let telegram_delivery = TelegramDeliveryCoordinator::default();
    let openrouter_base_url = config
        .openrouter_base_url
        .as_deref()
        .unwrap_or(DEFAULT_OPENROUTER_BASE_URL);
    let openrouter_pricing = Arc::new(
        OpenRouterPricingCache::new(config.openrouter_api_key(), openrouter_base_url)
            .map_err(error_text)?,
    );
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
        openrouter_pricing: Some(Arc::clone(&openrouter_pricing)),
        firecrawl_api_key: config.firecrawl_api_key().map(str::to_owned),
        supadata_api_key: config.supadata_api_key().map(str::to_owned),
        apify_api_key: config.apify_api_key().map(str::to_owned),
        opennode: opennode_options(config),
        system_prompt: Some(config.system_prompt.clone()),
        trigger_words: Some(config.trigger_words.clone()),
        active_operations: active_operations.clone(),
        telegram_delivery: telegram_delivery.clone(),
    })
    .map_err(error_text)?;
    let specs = build_production_background_specs(ProductionBackgroundOptions {
        redis_endpoint: &config.redis_endpoint,
        database_url: config.database_url(),
        telegram_token: config.runtime.telegram_token(),
        openrouter_api_key: config.openrouter_api_key(),
        openrouter_base_url,
        openrouter_pricing,
        firecrawl_api_key: config.firecrawl_api_key(),
        system_prompt: &config.system_prompt,
        owner_token: &config.owner_token(),
        scheduler_mode: SchedulerMode::Authoritative,
        reconciliation_interval: config.reconciliation_interval,
        reconciliation_settings: config.reconciliation_settings,
        active_operations,
        coinmarketcap_key: Some(config.coinmarketcap_key()),
        opennode: opennode_options(config),
        telegram_delivery: telegram_delivery.clone(),
    })?;
    let mut supervisor =
        BackgroundSupervisor::start(specs, reporter.clone()).map_err(error_text)?;
    let mut reports = BackgroundReports::start(reporter, OPERATIONAL_REPORT_QUEUE_CAPACITY);

    let command_transport =
        telegram_transport(ReqwestTelegramTransport::new(), "command publication")?;
    let mut command_sink =
        TelegramActionSink::new(command_transport, config.runtime.telegram_token())
            .with_delivery_coordinator(telegram_delivery);
    let mut diagnostics = publish_commands(&mut command_sink);
    let chats = ChatConfigRepository::new(config.database_url()).chats_with_language();
    diagnostics.extend(refresh_chat_menus(&mut command_sink, chats));
    for diagnostic in diagnostics {
        log_and_queue(&reports, diagnostic);
    }

    let stopping = Arc::new(AtomicBool::new(false));
    install_shutdown_handler(&stopping, ctrlc::set_handler::<ShutdownHandler>)?;
    let polling_result = poll_until_stopped(
        &mut runtime,
        &stopping,
        || supervisor.has_failed(),
        &reports,
    );
    if let Err(error) = &polling_result {
        let report = OperationalReport::new(
            format!("falló el proceso de sondeo de Telegram: {error}"),
            format!("Telegram polling runtime failed: {error}"),
        );
        reports.report(report);
    }
    runtime.shutdown();
    let shutdown_result = supervisor.stop().map_err(error_text);
    reports.shutdown();
    polling_result.and(shutdown_result)
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use bot_adapters::telegram_http::{ReqwestTelegramTransport, TransportFailureKind};
    use bot_adapters::telegram_polling::{
        IncomingEvent, IncomingUpdate, PollFailure, PollOutcome, PollingError,
    };
    use bot_core::locale::Locale;
    use bot_core::telegram_actions::TelegramAction;

    use super::{
        BackgroundReports, ShutdownHandler, build_operational_reporter, install_shutdown_handler,
        interruptible_wait, log_and_queue, opennode_options, poll_retry_report, poll_until_stopped,
        publish_commands, refresh_chat_menus, report_best_effort, retry_delay, run_polling_until,
        telegram_transport, update_failure_report,
    };
    use crate::config::ProductionConfig;
    use crate::dispatcher::{ActionReceipt, ActionSink};
    use crate::operational_reporting::{OperationalReport, OperationalReporter};
    use crate::runtime::{PollingRuntime, UpdateHandler, UpdateSource};

    #[derive(Default)]
    struct Sink {
        actions: Vec<TelegramAction>,
        calls: usize,
        fail_call: Option<usize>,
    }

    impl ActionSink for Sink {
        type Error = &'static str;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            let call = self.calls;
            self.calls += 1;
            if self.fail_call == Some(call) {
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
        assert_eq!(sink.actions.len(), 5);
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
        assert!(matches!(
            &sink.actions[3],
            TelegramAction::SetCommands {
                language_code: None,
                scope: bot_core::telegram_actions::CommandScope::AllGroupChats,
                ..
            }
        ));
        assert!(matches!(
            &sink.actions[4],
            TelegramAction::SetCommands {
                language_code: Some(language),
                scope: bot_core::telegram_actions::CommandScope::AllGroupChats,
                ..
            } if language == "en"
        ));
    }

    #[test]
    fn chat_menus_are_refreshed_in_each_chat_language() {
        use bot_core::telegram_actions::CommandScope;
        use bot_core::telegram_input::ChatId;
        let mut sink = Sink::default();
        let chats: Result<_, &str> = Ok(vec![(-42, "en".to_owned()), (7, "es".to_owned())]);
        assert!(refresh_chat_menus(&mut sink, chats).is_empty());
        // Unrelated actions are ignored by the filter below.
        let unrelated = TelegramAction::DeleteMessage {
            chat_id: ChatId(1),
            message_id: bot_core::telegram_input::MessageId(1),
        };
        let menus = sink
            .actions
            .iter()
            .chain(std::iter::once(&unrelated))
            .filter_map(|action| match action {
                TelegramAction::SetCommands {
                    commands,
                    language_code: None,
                    scope: CommandScope::Chat(chat_id),
                } => Some((*chat_id, commands[0].command, commands[0].description)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            menus,
            [
                (ChatId(-42), "ask", "ask me anything"),
                (ChatId(7), "ask", "preguntame lo que quieras"),
            ]
        );
    }

    #[test]
    fn chat_menu_refresh_reports_failures_once_and_keeps_going() {
        let mut sink = Sink {
            fail_call: Some(0),
            ..Sink::default()
        };
        let chats: Result<_, &str> = Ok(vec![(1, "es".to_owned()), (2, "en".to_owned())]);
        let reports = refresh_chat_menus(&mut sink, chats);
        assert_eq!(sink.calls, 2);
        assert_eq!(sink.actions.len(), 1);
        assert_eq!(reports.len(), 1);
        assert_eq!(
            reports[0].english(),
            "chat command menu refresh failed for 1 of 2 chats: synthetic publication failure"
        );
        assert!(
            reports[0]
                .for_locale(Locale::Es)
                .contains("falló la actualización del menú de 1 de 2 chats")
        );

        let mut untouched = Sink::default();
        let reports = refresh_chat_menus(&mut untouched, Err::<Vec<(i64, String)>, _>("db down"));
        assert_eq!(untouched.calls, 0);
        assert_eq!(
            reports
                .iter()
                .map(OperationalReport::english)
                .collect::<Vec<_>>(),
            ["could not load chats to refresh their menus: db down"]
        );
        assert!(reports[0].for_locale(Locale::Es).contains("db down"));
    }

    #[test]
    fn command_publication_failure_still_publishes_the_other_menus() {
        let mut sink = Sink {
            fail_call: Some(1),
            ..Sink::default()
        };
        let diagnostics = publish_commands(&mut sink);
        assert_eq!(sink.calls, 5);
        assert_eq!(sink.actions.len(), 4);
        assert!(matches!(
            sink.actions.last(),
            Some(TelegramAction::SetCommands {
                scope: bot_core::telegram_actions::CommandScope::AllGroupChats,
                ..
            })
        ));
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

    struct ScriptedSource {
        outcomes: VecDeque<Result<PollOutcome, PollingError>>,
        offsets: Rc<RefCell<Vec<Option<i64>>>>,
    }

    impl UpdateSource for ScriptedSource {
        fn poll(&mut self, offset: Option<i64>) -> Result<PollOutcome, PollingError> {
            self.offsets.borrow_mut().push(offset);
            self.outcomes
                .pop_front()
                .unwrap_or(Ok(PollOutcome::Updates(Vec::new())))
        }
    }

    struct ScriptedHandler {
        failing: Vec<i64>,
        handled: Rc<RefCell<Vec<i64>>>,
    }

    impl UpdateHandler for ScriptedHandler {
        type Error = &'static str;
        fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
            if self.failing.contains(&update.update_id) {
                return Err("synthetic handler failure");
            }
            self.handled.borrow_mut().push(update.update_id);
            Ok(())
        }
    }

    type ScriptedRuntime = PollingRuntime<ScriptedSource, ScriptedHandler>;
    type SharedOffsets = Rc<RefCell<Vec<Option<i64>>>>;
    type SharedHandled = Rc<RefCell<Vec<i64>>>;

    fn scripted_runtime(
        outcomes: Vec<Result<PollOutcome, PollingError>>,
        failing: Vec<i64>,
    ) -> (ScriptedRuntime, SharedOffsets, SharedHandled) {
        let offsets = Rc::new(RefCell::new(Vec::new()));
        let handled = Rc::new(RefCell::new(Vec::new()));
        let runtime = PollingRuntime::new(
            ScriptedSource {
                outcomes: VecDeque::from(outcomes),
                offsets: Rc::clone(&offsets),
            },
            ScriptedHandler {
                failing,
                handled: Rc::clone(&handled),
            },
        );
        (runtime, offsets, handled)
    }

    #[derive(Debug, Default, PartialEq)]
    struct PollingLog {
        waits: Vec<Duration>,
        retries: Vec<PollFailure>,
        failures: Vec<(i64, String)>,
    }

    /// Runs `steps` polling iterations, recording every callback.
    fn poll_steps(runtime: &mut ScriptedRuntime, steps: usize) -> (Result<(), String>, PollingLog) {
        let log = RefCell::new(PollingLog::default());
        let iterations = Cell::new(0);
        let result = run_polling_until(
            runtime,
            || {
                let current = iterations.get();
                iterations.set(current + 1);
                current >= steps
            },
            |duration| log.borrow_mut().waits.push(duration),
            |failure| log.borrow_mut().retries.push(failure.clone()),
            |update_id, error| {
                log.borrow_mut()
                    .failures
                    .push((update_id, error.to_owned()))
            },
        );
        (result, log.into_inner())
    }

    fn update(update_id: i64) -> IncomingUpdate {
        IncomingUpdate {
            update_id,
            event: IncomingEvent::Unsupported,
        }
    }

    #[test]
    fn handler_failures_are_reported_unacknowledged_and_do_not_stop_polling() {
        let (mut runtime, offsets, handled) = scripted_runtime(
            vec![
                Ok(PollOutcome::Updates(vec![update(10), update(11)])),
                Ok(PollOutcome::Retry(PollFailure::Conflict)),
                Ok(PollOutcome::Updates(vec![update(10), update(11)])),
                Ok(PollOutcome::Updates(vec![update(10), update(11)])),
                Ok(PollOutcome::Updates(vec![update(12)])),
            ],
            vec![11],
        );
        let (result, log) = poll_steps(&mut runtime, 5);
        assert_eq!(result, Ok(()));
        // A conflicting poller between attempts is waited out and reported.
        assert_eq!(log.waits, [Duration::from_secs(1)]);
        assert_eq!(log.retries, [PollFailure::Conflict]);
        assert_eq!(*handled.borrow(), [10, 12]);
        assert_eq!(
            log.failures,
            [
                (11, "synthetic handler failure".to_owned()),
                (11, "synthetic handler failure".to_owned()),
                (
                    11,
                    "quarantined without further retries: synthetic handler failure".to_owned()
                ),
            ]
        );
        assert_eq!(*offsets.borrow(), [None, None, None, None, Some(12)]);
        assert_eq!(runtime.offset(), Some(13));
    }

    #[test]
    fn polling_retries_are_reported_once_until_a_success_resets_the_failure() {
        let failure = PollFailure::Transport {
            failure: TransportFailureKind::Request,
        };
        let (mut runtime, _, _) = scripted_runtime(
            vec![
                Ok(PollOutcome::Updates(Vec::new())),
                Ok(PollOutcome::Retry(failure.clone())),
                Ok(PollOutcome::Retry(failure.clone())),
                Ok(PollOutcome::Updates(vec![update(5)])),
                Ok(PollOutcome::Retry(failure.clone())),
            ],
            vec![5],
        );
        let (result, log) = poll_steps(&mut runtime, 5);
        assert_eq!(result, Ok(()));
        // A handler-failure step also resets the retry deduplication.
        assert_eq!(log.retries, [failure.clone(), failure]);
        assert_eq!(log.waits.len(), 3);
        assert_eq!(log.failures, [(5, "synthetic handler failure".to_owned())]);
    }

    #[test]
    fn polling_errors_stop_the_loop_with_their_message() {
        let (mut runtime, offsets, _) =
            scripted_runtime(vec![Err(PollingError::InvalidResponse)], Vec::new());
        let (result, log) = poll_steps(&mut runtime, 5);
        assert_eq!(result, Err(PollingError::InvalidResponse.to_string()));
        assert_eq!(log, PollingLog::default());
        assert_eq!(*offsets.borrow(), [None]);
    }

    #[test]
    fn operational_reporting_composes_with_and_without_an_admin() {
        fn config(admin: Option<&str>) -> ProductionConfig {
            let lookup = |name: &str| match name {
                "TELEGRAM_TOKEN" => Some("synthetic-telegram-token".to_owned()),
                "SUPABASE_POSTGRES_URL" => Some(
                    "postgresql://synthetic:synthetic@db.example.test/database?sslmode=require"
                        .to_owned(),
                ),
                "TELEGRAM_USERNAME" => Some("synthetic_test_bot".to_owned()),
                "COINMARKETCAP_KEY" => Some("synthetic-market-key".to_owned()),
                "OPENROUTER_API_KEY" => Some("synthetic-ai-key".to_owned()),
                "ADMIN_CHAT_ID" => admin.map(str::to_owned),
                // The admin config also exercises the optional Lightning keys.
                "OPENNODE_API_KEY" => admin.map(|_| "synthetic-opennode-key".to_owned()),
                _ => None,
            };
            let config = ProductionConfig::from_lookup_and_prompt(lookup, || {
                Ok(Some("synthetic system prompt".to_owned()))
            });
            let Ok(config) = config else { unreachable!() };
            config
        }

        assert!(build_operational_reporter(&config(None)).is_ok());
        assert!(telegram_transport(ReqwestTelegramTransport::new(), "admin reporting").is_ok());
        let refused = telegram_transport(Err(TransportFailureKind::Request), "admin reporting");
        assert_eq!(
            refused.err().as_deref(),
            Some("could not construct admin reporting transport: Request")
        );
        assert!(build_operational_reporter(&config(Some("42"))).is_ok());
        assert!(opennode_options(&config(None)).is_none());
        assert_eq!(
            opennode_options(&config(Some("42"))).map(|options| (options.api_key, options.api_url)),
            Some((
                "synthetic-opennode-key".to_owned(),
                "https://api.opennode.com".to_owned()
            ))
        );

        struct FailingReporter;
        impl OperationalReporter for FailingReporter {
            fn report(&self, _report: &OperationalReport) -> Result<(), String> {
                Err("synthetic report failure".to_owned())
            }
        }
        report_best_effort(
            &FailingReporter,
            &OperationalReport::new("informe sintético", "synthetic report"),
        );

        let stopping = AtomicBool::new(true);
        interruptible_wait(&stopping, Duration::from_secs(1));
        stopping.store(false, Ordering::Release);
        interruptible_wait(&stopping, Duration::ZERO);
    }

    #[test]
    fn background_reports_do_not_block_the_caller_and_drop_on_overflow() {
        struct GatedReporter {
            entered: std::sync::Mutex<std::sync::mpsc::Sender<std::thread::ThreadId>>,
            release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
            delivered: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }
        impl OperationalReporter for GatedReporter {
            fn report(&self, report: &OperationalReport) -> Result<(), String> {
                if let Ok(entered) = self.entered.lock() {
                    let _ = entered.send(std::thread::current().id());
                }
                if let Ok(release) = self.release.lock() {
                    let _ = release.recv_timeout(Duration::from_secs(5));
                }
                if let Ok(mut delivered) = self.delivered.lock() {
                    delivered.push(report.english().to_owned());
                }
                Ok(())
            }
        }

        let (entered, entered_receiver) = std::sync::mpsc::channel();
        let (release, release_receiver) = std::sync::mpsc::channel();
        let delivered = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut reports = BackgroundReports::start(
            std::sync::Arc::new(GatedReporter {
                entered: std::sync::Mutex::new(entered),
                release: std::sync::Mutex::new(release_receiver),
                delivered: std::sync::Arc::clone(&delivered),
            }),
            1,
        );
        let report = |text: &str| OperationalReport::new(text, text);

        let started = std::time::Instant::now();
        reports.report(report("first"));
        let reporter_thread = entered_receiver.recv_timeout(Duration::from_secs(1));
        assert!(matches!(reporter_thread, Ok(id) if id != std::thread::current().id()));
        // The reporter is blocked on Telegram: one report queues, the next drops.
        reports.report(report("queued"));
        reports.report(report("overflow"));
        assert!(started.elapsed() < Duration::from_secs(1));

        for _ in 0..2 {
            assert!(release.send(()).is_ok());
        }
        reports.shutdown();
        assert!(
            delivered
                .lock()
                .is_ok_and(|delivered| *delivered == ["first", "queued"])
        );
        reports.report(report("after shutdown"));
        assert!(delivered.lock().is_ok_and(|delivered| delivered.len() == 2));
    }

    #[test]
    fn polling_wait_is_interrupted_before_its_deadline() {
        let stopping = std::sync::Arc::new(AtomicBool::new(false));
        let signal = stopping.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            signal.store(true, Ordering::Release);
        });
        let started = std::time::Instant::now();
        interruptible_wait(&stopping, Duration::from_secs(1));
        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(thread.join().is_ok());
    }

    #[test]
    fn polling_diagnostics_are_localized_and_queued_for_the_admin() {
        #[derive(Default)]
        struct Recorder(std::sync::Mutex<Vec<OperationalReport>>);
        impl OperationalReporter for Recorder {
            fn report(&self, report: &OperationalReport) -> Result<(), String> {
                self.0
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(report.clone());
                Ok(())
            }
        }

        let retry = poll_retry_report(&PollFailure::Conflict);
        assert_eq!(retry.english(), "Telegram polling retry: Conflict");
        assert_eq!(
            retry.for_locale(Locale::Es),
            "reintento del sondeo de Telegram: Conflict"
        );
        let update = update_failure_report(42, "synthetic failure");
        assert_eq!(
            update.english(),
            "Telegram update 42 failed: synthetic failure"
        );
        assert_eq!(
            update.for_locale(Locale::Es),
            "falló la actualización 42 de Telegram: synthetic failure"
        );

        let recorder = std::sync::Arc::new(Recorder::default());
        let mut reports = BackgroundReports::start(recorder.clone(), 4);
        log_and_queue(&reports, retry.clone());
        log_and_queue(&reports, update.clone());
        reports.shutdown();
        assert!(
            recorder
                .0
                .lock()
                .is_ok_and(|delivered| *delivered == [retry, update])
        );
    }

    #[test]
    fn production_polling_reports_update_failures_and_retries_until_stopped() {
        #[derive(Default)]
        struct Recorder(std::sync::Mutex<Vec<String>>);
        impl OperationalReporter for Recorder {
            fn report(&self, report: &OperationalReport) -> Result<(), String> {
                self.0
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(report.english().to_owned());
                Ok(())
            }
        }
        struct Source {
            outcomes: VecDeque<PollOutcome>,
            stopping: std::sync::Arc<AtomicBool>,
        }
        impl UpdateSource for Source {
            fn poll(&mut self, _: Option<i64>) -> Result<PollOutcome, PollingError> {
                let outcome = self
                    .outcomes
                    .pop_front()
                    .unwrap_or(PollOutcome::Updates(Vec::new()));
                // A shutdown signal arrives while the retry is pending, so
                // the retry wait returns at once.
                self.stopping
                    .store(self.outcomes.is_empty(), Ordering::Release);
                Ok(outcome)
            }
        }
        struct Handler;
        impl UpdateHandler for Handler {
            type Error = &'static str;
            fn handle(&mut self, _: IncomingUpdate) -> Result<(), Self::Error> {
                Err("synthetic delivery failure")
            }
        }

        let stopping = std::sync::Arc::new(AtomicBool::new(false));
        let mut runtime = PollingRuntime::new(
            Source {
                outcomes: VecDeque::from([
                    PollOutcome::Updates(vec![IncomingUpdate {
                        update_id: 77,
                        event: IncomingEvent::Unsupported,
                    }]),
                    PollOutcome::Retry(PollFailure::Conflict),
                ]),
                stopping: stopping.clone(),
            },
            Handler,
        );
        let recorder = std::sync::Arc::new(Recorder::default());
        let mut reports = BackgroundReports::start(recorder.clone(), 4);
        let failure_checks = Cell::new(0);
        let started = std::time::Instant::now();

        let result = poll_until_stopped(
            &mut runtime,
            &stopping,
            || {
                failure_checks.set(failure_checks.get() + 1);
                false
            },
            &reports,
        );
        reports.shutdown();

        assert_eq!(result, Ok(()));
        assert!(started.elapsed() < Duration::from_millis(900));
        assert_eq!(failure_checks.get(), 2);
        assert!(recorder.0.lock().is_ok_and(|delivered| {
            *delivered
                == [
                    "Telegram update 77 failed: synthetic delivery failure",
                    "Telegram polling retry: Conflict",
                ]
        }));
    }

    fn signal_now(mut handler: ShutdownHandler) -> Result<(), ctrlc::Error> {
        handler();
        Ok(())
    }

    fn already_registered(_handler: ShutdownHandler) -> Result<(), ctrlc::Error> {
        Err(ctrlc::Error::MultipleHandlers)
    }

    #[test]
    fn shutdown_signal_requests_a_stop_and_install_failures_are_reported() {
        let stopping = std::sync::Arc::new(AtomicBool::new(false));
        assert_eq!(install_shutdown_handler(&stopping, signal_now), Ok(()));
        assert!(stopping.load(Ordering::Acquire));

        let untouched = std::sync::Arc::new(AtomicBool::new(false));
        let refused = install_shutdown_handler(&untouched, already_registered);
        assert_eq!(
            refused.err().as_deref(),
            Some(
                "could not install shutdown signal handler: \
                 Ctrl-C error: Ctrl-C signal handler already registered"
            )
        );
        assert!(!untouched.load(Ordering::Acquire));
    }
}
