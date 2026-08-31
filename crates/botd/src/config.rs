//! Environment configuration for the native runtime.

use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use bot_adapters::redis_connection::RedisEndpoint;
use thiserror::Error;

const DEFAULT_HANDLER_WORKERS: usize = 16;
const DEFAULT_LONG_POLL_SECONDS: u64 = 30;
const DEFAULT_AI_LEDGER_RETENTION_DAYS: i64 = 30;

#[derive(Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    telegram_token: String,
    pub handler_workers: NonZeroUsize,
    pub long_poll_timeout: Duration,
}

#[derive(Clone)]
pub struct TaskVerificationConfig {
    pub redis_endpoint: RedisEndpoint,
    pub owner_token: String,
}

#[derive(Clone)]
pub struct MaintenanceConfig {
    pub redis_endpoint: RedisEndpoint,
    database_url: Option<String>,
    pub redis_maxmemory: String,
    pub redis_maxmemory_policy: String,
    pub ai_ledger_retention_days: i64,
}

impl fmt::Debug for TaskVerificationConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TaskVerificationConfig")
            .field("redis_host", &self.redis_endpoint.host)
            .field("redis_port", &self.redis_endpoint.port)
            .field(
                "redis_password",
                &self.redis_endpoint.password.as_ref().map(|_| "[REDACTED]"),
            )
            .field("owner_token", &self.owner_token)
            .finish()
    }
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

impl fmt::Debug for MaintenanceConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MaintenanceConfig")
            .field("redis_host", &self.redis_endpoint.host)
            .field("redis_port", &self.redis_endpoint.port)
            .field(
                "redis_password",
                &self.redis_endpoint.password.as_ref().map(|_| "[REDACTED]"),
            )
            .field(
                "database_url",
                &self.database_url.as_ref().map(|_| "[REDACTED]"),
            )
            .field("redis_maxmemory", &self.redis_maxmemory)
            .field("redis_maxmemory_policy", &self.redis_maxmemory_policy)
            .field("ai_ledger_retention_days", &self.ai_ledger_retention_days)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigError {
    #[error("TELEGRAM_TOKEN not set")]
    MissingTelegramToken,
    #[error("TELEGRAM_LONG_POLL_SECONDS must be a positive integer")]
    InvalidLongPollTimeout,
    #[error("REDIS_PORT must be an integer from 1 through 65535")]
    InvalidRedisPort,
    #[error("AI_LEDGER_RETENTION_DAYS must be a positive integer")]
    InvalidLedgerRetention,
}

fn redis_endpoint_from_lookup<F>(lookup: &F) -> Result<RedisEndpoint, ConfigError>
where
    F: Fn(&str) -> Option<String>,
{
    let host = lookup("REDIS_HOST")
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "localhost".to_owned());
    let port = lookup("REDIS_PORT").map_or(Ok(6379), |value| {
        value
            .parse::<u16>()
            .ok()
            .filter(|port| *port > 0)
            .ok_or(ConfigError::InvalidRedisPort)
    })?;
    let password = lookup("REDIS_PASSWORD").filter(|value| !value.is_empty());
    Ok(RedisEndpoint {
        host,
        port,
        password,
    })
}

impl TaskVerificationConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    pub fn from_lookup<F>(lookup: F) -> Result<Self, ConfigError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let redis_endpoint = redis_endpoint_from_lookup(&lookup)?;
        let owner_token = lookup("BOT_INSTANCE_NAME")
            .filter(|value| !value.trim().is_empty())
            .map_or_else(
                || "botd-verifier".to_owned(),
                |value| format!("{value}:verify"),
            );
        Ok(Self {
            redis_endpoint,
            owner_token,
        })
    }
}

impl MaintenanceConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    pub fn from_lookup<F>(lookup: F) -> Result<Self, ConfigError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let redis_endpoint = redis_endpoint_from_lookup(&lookup)?;
        let database_url = lookup("SUPABASE_POSTGRES_URL").filter(|value| !value.trim().is_empty());
        let redis_maxmemory = lookup("REDIS_MAXMEMORY")
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "256mb".to_owned());
        let redis_maxmemory_policy = lookup("REDIS_MAXMEMORY_POLICY")
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "allkeys-lru".to_owned());
        let ai_ledger_retention_days = lookup("AI_LEDGER_RETENTION_DAYS").map_or(
            Ok(DEFAULT_AI_LEDGER_RETENTION_DAYS),
            |value| {
                value
                    .parse::<i64>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or(ConfigError::InvalidLedgerRetention)
            },
        )?;
        Ok(Self {
            redis_endpoint,
            database_url,
            redis_maxmemory,
            redis_maxmemory_policy,
            ai_ledger_retention_days,
        })
    }

    #[must_use]
    pub fn database_url(&self) -> Option<&str> {
        self.database_url.as_deref()
    }
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

    use super::{
        ConfigError, DEFAULT_HANDLER_WORKERS, MaintenanceConfig, RuntimeConfig,
        TaskVerificationConfig,
    };

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

    #[test]
    fn task_verification_configuration_is_independent_of_telegram_credentials() {
        let values = HashMap::from([
            ("REDIS_HOST".to_owned(), "redis.internal".to_owned()),
            ("REDIS_PORT".to_owned(), "6380".to_owned()),
            ("REDIS_PASSWORD".to_owned(), "synthetic-secret".to_owned()),
            ("BOT_INSTANCE_NAME".to_owned(), "worker-a".to_owned()),
        ]);
        let config = TaskVerificationConfig::from_lookup(|name| values.get(name).cloned())
            .map_err(|error| error.to_string());
        assert!(config.is_ok());
        let Some(config) = config.ok() else {
            return;
        };
        assert_eq!(config.redis_endpoint.host, "redis.internal");
        assert_eq!(config.redis_endpoint.port, 6380);
        assert_eq!(config.owner_token, "worker-a:verify");
        let debug = format!("{config:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("synthetic-secret"));
    }

    #[test]
    fn task_verification_redis_port_is_bounded() {
        for invalid in ["", "0", "65536", "many"] {
            assert!(matches!(
                TaskVerificationConfig::from_lookup(|name| {
                    (name == "REDIS_PORT").then(|| invalid.to_owned())
                }),
                Err(ConfigError::InvalidRedisPort)
            ));
        }
    }

    #[test]
    fn maintenance_configuration_is_token_independent_and_redacts_credentials() {
        let values = HashMap::from([
            ("REDIS_HOST".to_owned(), "redis.internal".to_owned()),
            ("REDIS_PORT".to_owned(), "6380".to_owned()),
            ("REDIS_PASSWORD".to_owned(), "redis-secret".to_owned()),
            (
                "SUPABASE_POSTGRES_URL".to_owned(),
                "postgresql://database-secret".to_owned(),
            ),
            ("REDIS_MAXMEMORY".to_owned(), "512mb".to_owned()),
            (
                "REDIS_MAXMEMORY_POLICY".to_owned(),
                "volatile-lru".to_owned(),
            ),
            ("AI_LEDGER_RETENTION_DAYS".to_owned(), "45".to_owned()),
        ]);
        let config = MaintenanceConfig::from_lookup(|name| values.get(name).cloned());
        assert!(config.is_ok());
        let Some(config) = config.ok() else {
            return;
        };
        assert_eq!(config.redis_endpoint.host, "redis.internal");
        assert_eq!(config.redis_endpoint.port, 6380);
        assert_eq!(config.database_url(), Some("postgresql://database-secret"));
        assert_eq!(config.redis_maxmemory, "512mb");
        assert_eq!(config.redis_maxmemory_policy, "volatile-lru");
        assert_eq!(config.ai_ledger_retention_days, 45);
        let debug = format!("{config:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("redis-secret"));
        assert!(!debug.contains("database-secret"));
    }

    #[test]
    fn maintenance_defaults_and_retention_validation_match_python() {
        let config = MaintenanceConfig::from_lookup(|_| None);
        assert!(config.is_ok());
        let Some(config) = config.ok() else {
            return;
        };
        assert_eq!(config.redis_endpoint.host, "localhost");
        assert_eq!(config.redis_endpoint.port, 6379);
        assert_eq!(config.database_url(), None);
        assert_eq!(config.redis_maxmemory, "256mb");
        assert_eq!(config.redis_maxmemory_policy, "allkeys-lru");
        assert_eq!(config.ai_ledger_retention_days, 30);
        for invalid in ["", "0", "-1", "many"] {
            assert_eq!(
                MaintenanceConfig::from_lookup(|name| {
                    (name == "AI_LEDGER_RETENTION_DAYS").then(|| invalid.to_owned())
                })
                .map(|_| ()),
                Err(ConfigError::InvalidLedgerRetention)
            );
        }
    }
}
