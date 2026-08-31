//! Environment configuration for the native runtime.

use std::fmt;
use std::fs;
use std::path::Path;
use std::time::Duration;

use bot_adapters::redis_connection::RedisEndpoint;
use thiserror::Error;

use crate::reconciliation::ReconciliationSettings;

const DEFAULT_LONG_POLL_SECONDS: u64 = 30;
const DEFAULT_AI_LEDGER_RETENTION_DAYS: i64 = 30;
const DEFAULT_RECONCILIATION_INTERVAL_SECONDS: i64 = 60;
const DEFAULT_RECONCILIATION_RETRY_SECONDS: i64 = 3_600;
const DEFAULT_RECONCILIATION_SAFETY_CREDIT_UNITS: i64 = 10;
const DEFAULT_RECONCILIATION_STALE_SECONDS: i64 = 300;
const DEFAULT_TRIGGER_WORDS: [&str; 6] = [
    "gordo",
    "respondedor",
    "atendedor",
    "gordito",
    "dogor",
    "bot",
];

#[derive(Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    telegram_token: String,
    pub long_poll_timeout: Duration,
}

#[derive(Clone)]
pub struct ProductionConfig {
    pub runtime: RuntimeConfig,
    database_url: String,
    pub bot_name: String,
    pub instance_name: Option<String>,
    pub redis_endpoint: RedisEndpoint,
    pub admin_user_id: Option<i64>,
    coinmarketcap_key: String,
    giphy_api_key: Option<String>,
    openrouter_api_key: String,
    pub openrouter_base_url: Option<String>,
    groq_free_api_key: Option<String>,
    groq_api_key: Option<String>,
    firecrawl_api_key: Option<String>,
    pub system_prompt: String,
    pub trigger_words: Vec<String>,
    pub reconciliation_interval: Duration,
    pub reconciliation_settings: ReconciliationSettings,
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
            .field("long_poll_timeout", &self.long_poll_timeout)
            .finish()
    }
}

impl fmt::Debug for ProductionConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProductionConfig")
            .field("runtime", &self.runtime)
            .field("database_url", &"[REDACTED]")
            .field("bot_name", &self.bot_name)
            .field("instance_name", &self.instance_name)
            .field("redis_host", &self.redis_endpoint.host)
            .field("redis_port", &self.redis_endpoint.port)
            .field(
                "redis_password",
                &self.redis_endpoint.password.as_ref().map(|_| "[REDACTED]"),
            )
            .field("admin_user_id", &self.admin_user_id)
            .field("coinmarketcap_configured", &true)
            .field("giphy_configured", &self.giphy_api_key.is_some())
            .field("openrouter_configured", &true)
            .field("groq_free_configured", &self.groq_free_api_key.is_some())
            .field("groq_configured", &self.groq_api_key.is_some())
            .field("firecrawl_configured", &self.firecrawl_api_key.is_some())
            .field("system_prompt", &"[REDACTED]")
            .field("trigger_words", &self.trigger_words)
            .field("reconciliation_interval", &self.reconciliation_interval)
            .field("reconciliation_settings", &self.reconciliation_settings)
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
    #[error("SUPABASE_POSTGRES_URL not set")]
    MissingDatabaseUrl,
    #[error("TELEGRAM_USERNAME not set")]
    MissingTelegramUsername,
    #[error("COINMARKETCAP_KEY not set")]
    MissingCoinMarketCapKey,
    #[error("OPENROUTER_API_KEY not set")]
    MissingOpenRouterApiKey,
    #[error("ADMIN_CHAT_ID must be an integer")]
    InvalidAdminUserId,
    #[error("BOT_SYSTEM_PROMPT not set and workspace/SOUL.md or workspace/RULES.md is missing")]
    MissingSystemPrompt,
    #[error("could not read the workspace prompt: {0}")]
    WorkspacePrompt(String),
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
            long_poll_timeout: Duration::from_secs(long_poll_seconds),
        })
    }

    #[must_use]
    pub fn telegram_token(&self) -> &str {
        &self.telegram_token
    }
}

fn optional_trimmed<F>(lookup: &F, name: &str) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    lookup(name)
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn configured_integer<F>(lookup: &F, name: &str, default: i64) -> i64
where
    F: Fn(&str) -> Option<String>,
{
    lookup(name)
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(default)
}

fn read_workspace_prompt() -> Result<Option<String>, ConfigError> {
    let mut parts = Vec::new();
    for path in [
        Path::new("workspace/SOUL.md"),
        Path::new("workspace/RULES.md"),
    ] {
        match fs::read_to_string(path) {
            Ok(value) if !value.trim().is_empty() => parts.push(value.trim().to_owned()),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(ConfigError::WorkspacePrompt(error.to_string())),
        }
    }
    Ok((!parts.is_empty()).then(|| parts.join("\n\n")))
}

impl ProductionConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup_and_prompt(|name| std::env::var(name).ok(), read_workspace_prompt)
    }

    pub fn from_lookup_and_prompt<F, P>(lookup: F, prompt: P) -> Result<Self, ConfigError>
    where
        F: Fn(&str) -> Option<String>,
        P: FnOnce() -> Result<Option<String>, ConfigError>,
    {
        let runtime = RuntimeConfig::from_lookup(&lookup)?;
        let redis_endpoint = redis_endpoint_from_lookup(&lookup)?;
        let database_url = optional_trimmed(&lookup, "SUPABASE_POSTGRES_URL")
            .ok_or(ConfigError::MissingDatabaseUrl)?;
        let bot_name = optional_trimmed(&lookup, "TELEGRAM_USERNAME")
            .map(|value| value.trim_start_matches('@').to_owned())
            .filter(|value| !value.is_empty())
            .ok_or(ConfigError::MissingTelegramUsername)?;
        let admin_user_id = optional_trimmed(&lookup, "ADMIN_CHAT_ID")
            .map(|value| {
                value
                    .parse::<i64>()
                    .map_err(|_| ConfigError::InvalidAdminUserId)
            })
            .transpose()?;
        let coinmarketcap_key = optional_trimmed(&lookup, "COINMARKETCAP_KEY")
            .ok_or(ConfigError::MissingCoinMarketCapKey)?;
        let openrouter_api_key = optional_trimmed(&lookup, "OPENROUTER_API_KEY")
            .ok_or(ConfigError::MissingOpenRouterApiKey)?;
        let system_prompt = match optional_trimmed(&lookup, "BOT_SYSTEM_PROMPT") {
            Some(value) => value,
            None => prompt()?.ok_or(ConfigError::MissingSystemPrompt)?,
        };
        let trigger_words = optional_trimmed(&lookup, "BOT_TRIGGER_WORDS")
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .filter(|values| !values.is_empty())
            .unwrap_or_else(|| DEFAULT_TRIGGER_WORDS.map(str::to_owned).to_vec());
        let reconciliation_interval = Duration::from_secs(
            configured_integer(
                &lookup,
                "AI_RECONCILIATION_INTERVAL_SECONDS",
                DEFAULT_RECONCILIATION_INTERVAL_SECONDS,
            )
            .max(5) as u64,
        );
        let reconciliation_settings = ReconciliationSettings {
            batch_limit: 500,
            retry_window_seconds: configured_integer(
                &lookup,
                "AI_RECONCILIATION_RETRY_SECONDS",
                DEFAULT_RECONCILIATION_RETRY_SECONDS,
            ),
            safety_credit_units: configured_integer(
                &lookup,
                "AI_RECONCILIATION_SAFETY_CREDIT_UNITS",
                DEFAULT_RECONCILIATION_SAFETY_CREDIT_UNITS,
            ),
            stale_seconds: configured_integer(
                &lookup,
                "AI_RECONCILIATION_STALE_SECONDS",
                DEFAULT_RECONCILIATION_STALE_SECONDS,
            ),
        }
        .normalized();
        Ok(Self {
            runtime,
            database_url,
            bot_name,
            instance_name: optional_trimmed(&lookup, "FRIENDLY_INSTANCE_NAME"),
            redis_endpoint,
            admin_user_id,
            coinmarketcap_key,
            giphy_api_key: optional_trimmed(&lookup, "GIPHY_API_KEY"),
            openrouter_api_key,
            openrouter_base_url: optional_trimmed(&lookup, "OPENROUTER_BASE_URL"),
            groq_free_api_key: optional_trimmed(&lookup, "GROQ_FREE_API_KEY"),
            groq_api_key: optional_trimmed(&lookup, "GROQ_API_KEY"),
            firecrawl_api_key: optional_trimmed(&lookup, "FIRECRAWL_API_KEY"),
            system_prompt,
            trigger_words,
            reconciliation_interval,
            reconciliation_settings,
        })
    }

    #[must_use]
    pub fn database_url(&self) -> &str {
        &self.database_url
    }

    #[must_use]
    pub fn coinmarketcap_key(&self) -> &str {
        &self.coinmarketcap_key
    }

    #[must_use]
    pub fn giphy_api_key(&self) -> Option<&str> {
        self.giphy_api_key.as_deref()
    }

    #[must_use]
    pub fn openrouter_api_key(&self) -> &str {
        &self.openrouter_api_key
    }

    #[must_use]
    pub fn groq_free_api_key(&self) -> Option<&str> {
        self.groq_free_api_key.as_deref()
    }

    #[must_use]
    pub fn groq_api_key(&self) -> Option<&str> {
        self.groq_api_key.as_deref()
    }

    #[must_use]
    pub fn firecrawl_api_key(&self) -> Option<&str> {
        self.firecrawl_api_key.as_deref()
    }

    #[must_use]
    pub fn owner_token(&self) -> String {
        self.instance_name
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("botd")
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        ConfigError, MaintenanceConfig, ProductionConfig, RuntimeConfig, TaskVerificationConfig,
    };

    fn config(values: &[(&str, &str)]) -> Result<RuntimeConfig, ConfigError> {
        let values = values
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect::<HashMap<_, _>>();
        RuntimeConfig::from_lookup(|name| values.get(name).cloned())
    }

    fn production(
        values: &[(&str, &str)],
        workspace_prompt: Option<&str>,
    ) -> Result<ProductionConfig, ConfigError> {
        let values = values
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect::<HashMap<_, _>>();
        ProductionConfig::from_lookup_and_prompt(
            |name| values.get(name).cloned(),
            || Ok(workspace_prompt.map(str::to_owned)),
        )
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

    #[test]
    fn production_configuration_validates_required_boundaries_and_workspace_prompt() {
        let base = [
            ("TELEGRAM_TOKEN", "synthetic-telegram-secret"),
            ("TELEGRAM_USERNAME", " @test_bot "),
            (
                "SUPABASE_POSTGRES_URL",
                "postgresql://synthetic-database-secret",
            ),
            ("COINMARKETCAP_KEY", "synthetic-cmc-secret"),
            ("OPENROUTER_API_KEY", "synthetic-openrouter-secret"),
        ];
        assert_eq!(
            production(&[], Some("prompt")).map(|_| ()),
            Err(ConfigError::MissingTelegramToken)
        );
        assert_eq!(
            production(&[("TELEGRAM_TOKEN", "token")], Some("prompt")).map(|_| ()),
            Err(ConfigError::MissingDatabaseUrl)
        );
        assert_eq!(
            production(
                &[
                    ("TELEGRAM_TOKEN", "token"),
                    ("SUPABASE_POSTGRES_URL", "postgresql://database"),
                ],
                Some("prompt"),
            )
            .map(|_| ()),
            Err(ConfigError::MissingTelegramUsername)
        );
        assert_eq!(
            production(&base, None).map(|_| ()),
            Err(ConfigError::MissingSystemPrompt)
        );
        let config = production(&base, Some("soul\n\nrules"));
        assert!(config.is_ok());
        let Some(config) = config.ok() else {
            return;
        };
        assert_eq!(config.bot_name, "test_bot");
        assert_eq!(config.system_prompt, "soul\n\nrules");
        assert_eq!(config.trigger_words.len(), 6);
        assert_eq!(config.reconciliation_interval.as_secs(), 60);
    }

    #[test]
    fn production_configuration_parses_optional_services_and_redacts_all_secrets() {
        let config = production(
            &[
                ("TELEGRAM_TOKEN", "telegram-secret"),
                ("TELEGRAM_USERNAME", "test_bot"),
                ("SUPABASE_POSTGRES_URL", "database-secret"),
                ("REDIS_PASSWORD", "redis-secret"),
                ("ADMIN_CHAT_ID", "99"),
                ("COINMARKETCAP_KEY", "cmc-secret"),
                ("GIPHY_API_KEY", "giphy-secret"),
                ("OPENROUTER_API_KEY", "openrouter-secret"),
                ("GROQ_FREE_API_KEY", "groq-free-secret"),
                ("GROQ_API_KEY", "groq-secret"),
                ("FIRECRAWL_API_KEY", "firecrawl-secret"),
                ("BOT_SYSTEM_PROMPT", "prompt-secret"),
                ("BOT_TRIGGER_WORDS", " gordo, test, ,bot "),
                ("FRIENDLY_INSTANCE_NAME", "VPS"),
                ("AI_RECONCILIATION_INTERVAL_SECONDS", "1"),
                ("AI_RECONCILIATION_RETRY_SECONDS", "20"),
                ("AI_RECONCILIATION_SAFETY_CREDIT_UNITS", "-5"),
                ("AI_RECONCILIATION_STALE_SECONDS", "4"),
            ],
            Some("unused"),
        );
        assert!(config.is_ok());
        let Some(config) = config.ok() else {
            return;
        };
        assert_eq!(config.admin_user_id, Some(99));
        assert_eq!(config.trigger_words, ["gordo", "test", "bot"]);
        assert_eq!(config.owner_token(), "VPS");
        assert_eq!(config.reconciliation_interval.as_secs(), 5);
        assert_eq!(config.reconciliation_settings.retry_window_seconds, 60);
        assert_eq!(config.reconciliation_settings.safety_credit_units, 0);
        assert_eq!(config.reconciliation_settings.stale_seconds, 30);
        let debug = format!("{config:?}");
        for secret in [
            "telegram-secret",
            "database-secret",
            "redis-secret",
            "cmc-secret",
            "giphy-secret",
            "openrouter-secret",
            "groq-free-secret",
            "groq-secret",
            "firecrawl-secret",
            "prompt-secret",
        ] {
            assert!(!debug.contains(secret));
        }
    }

    #[test]
    fn production_configuration_rejects_invalid_admin_id_and_uses_numeric_defaults() {
        let base = [
            ("TELEGRAM_TOKEN", "token"),
            ("TELEGRAM_USERNAME", "bot"),
            ("SUPABASE_POSTGRES_URL", "database"),
            ("BOT_SYSTEM_PROMPT", "prompt"),
            ("COINMARKETCAP_KEY", "cmc"),
            ("OPENROUTER_API_KEY", "openrouter"),
            ("ADMIN_CHAT_ID", "not-a-number"),
        ];
        assert_eq!(
            production(&base, None).map(|_| ()),
            Err(ConfigError::InvalidAdminUserId)
        );
        let config = production(
            &[
                ("TELEGRAM_TOKEN", "token"),
                ("TELEGRAM_USERNAME", "bot"),
                ("SUPABASE_POSTGRES_URL", "database"),
                ("BOT_SYSTEM_PROMPT", "prompt"),
                ("COINMARKETCAP_KEY", "cmc"),
                ("OPENROUTER_API_KEY", "openrouter"),
                ("AI_RECONCILIATION_INTERVAL_SECONDS", "invalid"),
                ("AI_RECONCILIATION_RETRY_SECONDS", "invalid"),
            ],
            None,
        );
        assert_eq!(
            config.map(|config| (
                config.reconciliation_interval.as_secs(),
                config.reconciliation_settings.retry_window_seconds,
            )),
            Ok((60, 3_600))
        );
    }
}
