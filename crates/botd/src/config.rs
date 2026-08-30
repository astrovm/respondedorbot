//! Environment configuration for the native runtime.

use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use thiserror::Error;

const DEFAULT_HANDLER_WORKERS: usize = 16;
const DEFAULT_LONG_POLL_SECONDS: u64 = 30;

#[derive(Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    telegram_token: String,
    pub handler_workers: NonZeroUsize,
    pub long_poll_timeout: Duration,
}

impl fmt::Debug for RuntimeConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeConfig")
            .field("telegram_token", &"[REDACTED]")
            .field("handler_workers", &self.handler_workers)
            .field("long_poll_timeout", &self.long_poll_timeout)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigError {
    #[error("TELEGRAM_TOKEN not set")]
    MissingTelegramToken,
    #[error("TELEGRAM_LONG_POLL_SECONDS must be a positive integer")]
    InvalidLongPollTimeout,
}

impl RuntimeConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    pub fn from_lookup<F>(lookup: F) -> Result<Self, ConfigError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let telegram_token = lookup("TELEGRAM_TOKEN")
            .filter(|value| !value.is_empty())
            .ok_or(ConfigError::MissingTelegramToken)?;
        let handler_workers = lookup("BOT_HANDLER_WORKERS")
            .and_then(|value| value.parse::<i64>().ok())
            .map_or(DEFAULT_HANDLER_WORKERS, |value| value.max(1) as usize);
        let handler_workers = NonZeroUsize::new(handler_workers).unwrap_or(NonZeroUsize::MIN);
        let long_poll_seconds = lookup("TELEGRAM_LONG_POLL_SECONDS").map_or(
            Ok(DEFAULT_LONG_POLL_SECONDS),
            |value| {
                value
                    .parse::<u64>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or(ConfigError::InvalidLongPollTimeout)
            },
        )?;
        Ok(Self {
            telegram_token,
            handler_workers,
            long_poll_timeout: Duration::from_secs(long_poll_seconds),
        })
    }

    #[must_use]
    pub fn telegram_token(&self) -> &str {
        &self.telegram_token
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{ConfigError, DEFAULT_HANDLER_WORKERS, RuntimeConfig};

    fn config(values: &[(&str, &str)]) -> Result<RuntimeConfig, ConfigError> {
        let values = values
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect::<HashMap<_, _>>();
        RuntimeConfig::from_lookup(|name| values.get(name).cloned())
    }

    #[test]
    fn requires_a_nonempty_token_without_exposing_it_in_debug_output() {
        assert_eq!(config(&[]), Err(ConfigError::MissingTelegramToken));
        assert_eq!(
            config(&[("TELEGRAM_TOKEN", "")]),
            Err(ConfigError::MissingTelegramToken)
        );
        let actual = config(&[("TELEGRAM_TOKEN", "synthetic-secret")]);
        assert!(actual.is_ok());
        let rendered = format!("{actual:?}");
        assert!(rendered.contains("[REDACTED]"));
        assert!(!rendered.contains("synthetic-secret"));
    }

    #[test]
    fn preserves_python_worker_defaults_and_minimum() {
        for (raw, expected) in [
            (None, DEFAULT_HANDLER_WORKERS),
            (Some("invalid"), DEFAULT_HANDLER_WORKERS),
            (Some("0"), 1),
            (Some("-9"), 1),
            (Some("24"), 24),
        ] {
            let mut values = vec![("TELEGRAM_TOKEN", "token")];
            if let Some(raw) = raw {
                values.push(("BOT_HANDLER_WORKERS", raw));
            }
            let actual = config(&values);
            assert_eq!(
                actual.map(|value| value.handler_workers.get()),
                Ok(expected)
            );
        }
    }

    #[test]
    fn long_poll_timeout_is_explicit_and_positive() {
        assert_eq!(
            config(&[("TELEGRAM_TOKEN", "token")]).map(|value| value.long_poll_timeout.as_secs()),
            Ok(30)
        );
        assert_eq!(
            config(&[
                ("TELEGRAM_TOKEN", "token"),
                ("TELEGRAM_LONG_POLL_SECONDS", "45"),
            ])
            .map(|value| value.long_poll_timeout.as_secs()),
            Ok(45)
        );
        for invalid in ["", "0", "-1", "1.5", "many"] {
            assert_eq!(
                config(&[
                    ("TELEGRAM_TOKEN", "token"),
                    ("TELEGRAM_LONG_POLL_SECONDS", invalid),
                ]),
                Err(ConfigError::InvalidLongPollTimeout)
            );
        }
    }

    #[test]
    fn token_accessor_returns_the_original_secret_only_when_requested() {
        let actual = config(&[("TELEGRAM_TOKEN", " synthetic-token ")]);
        assert_eq!(
            actual.map(|value| value.telegram_token().to_owned()),
            Ok(" synthetic-token ".to_owned())
        );
    }
}
