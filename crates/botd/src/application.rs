//! Native process composition and graceful lifecycle.

use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use bot_adapters::openrouter_chat::DEFAULT_OPENROUTER_BASE_URL;
use bot_adapters::telegram_http::ReqwestTelegramTransport;
use bot_adapters::telegram_polling::PollFailure;
use bot_core::telegram_commands::command_publication_actions;

use crate::background::{
    BackgroundSupervisor, ProductionBackgroundOptions, build_production_background_specs,
};
use crate::composition::{NativeRuntimeOptions, TelegramActionSink, build_native_runtime};
use crate::config::ProductionConfig;
use crate::dispatcher::ActionSink;
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

pub fn publish_commands<S>(sink: &mut S) -> Vec<String>
where
    S: ActionSink,
    S::Error: Display,
{
    let mut diagnostics = Vec::new();
    for action in command_publication_actions() {
        if let Err(error) = sink.execute(action) {
            diagnostics.push(format!("Telegram command publication failed: {error}"));
            break;
        }
    }
    diagnostics
}

pub fn run_polling_until<Source, Handler, Stop, Wait>(
    runtime: &mut PollingRuntime<Source, Handler>,
    mut should_stop: Stop,
    mut wait: Wait,
) -> Result<(), String>
where
    Source: UpdateSource,
    Handler: UpdateHandler,
    Handler::Error: Display,
    Stop: FnMut() -> bool,
    Wait: FnMut(Duration),
{
    while !should_stop() {
        match runtime.step().map_err(|error| error.to_string())? {
            StepOutcome::Retry(failure) => wait(retry_delay(&failure)),
            StepOutcome::Idle | StepOutcome::Dispatched { .. } => {}
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

pub fn run_production(config: &ProductionConfig) -> Result<(), String> {
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
    let mut supervisor = BackgroundSupervisor::start(specs).map_err(|error| error.to_string())?;

    let command_transport = ReqwestTelegramTransport::new()
        .map_err(|error| format!("could not construct command publication transport: {error:?}"))?;
    let mut command_sink =
        TelegramActionSink::new(command_transport, config.runtime.telegram_token());
    for diagnostic in publish_commands(&mut command_sink) {
        eprintln!("{diagnostic}");
    }

    let stopping = Arc::new(AtomicBool::new(false));
    let signal_stopping = stopping.clone();
    ctrlc::set_handler(move || signal_stopping.store(true, Ordering::Release))
        .map_err(|error| format!("could not install shutdown signal handler: {error}"))?;
    let polling_result = run_polling_until(
        &mut runtime,
        || stopping.load(Ordering::Acquire),
        |duration| interruptible_wait(&stopping, duration),
    );
    let shutdown_result = supervisor.stop().map_err(|error| error.to_string());
    polling_result.and(shutdown_result)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bot_adapters::telegram_http::TransportFailureKind;
    use bot_adapters::telegram_polling::PollFailure;
    use bot_core::telegram_actions::TelegramAction;

    use super::{publish_commands, retry_delay};
    use crate::dispatcher::{ActionReceipt, ActionSink};

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
        assert!(diagnostics[0].contains("synthetic publication failure"));
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
}
