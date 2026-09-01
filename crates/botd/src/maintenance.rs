//! Periodic Redis and PostgreSQL maintenance entrypoint.

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::billing_schema::BillingSchemaRepository;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_maintenance::{RedisMaintenanceResult, run_redis_maintenance};
use serde::Serialize;
use serde_json::{Value, json};
use thiserror::Error;

#[derive(Clone)]
pub struct MaintenanceOptions<'a> {
    pub redis_endpoint: &'a RedisEndpoint,
    pub database_url: Option<&'a str>,
    pub redis_maxmemory: &'a str,
    pub redis_maxmemory_policy: &'a str,
    pub ai_ledger_retention_days: i64,
}

#[derive(Debug, Serialize)]
pub struct MaintenanceReport {
    pub redis: RedisMaintenanceResult,
    pub ledger: Value,
}

#[derive(Debug, Error)]
pub enum MaintenanceError {
    #[error("Redis maintenance failed: {0}")]
    Redis(#[from] bot_adapters::redis_maintenance::RedisMaintenanceError),
    #[error("AI ledger retention days must be positive")]
    InvalidRetention,
    #[error("AI ledger maintenance failed: {0}")]
    Ledger(#[from] bot_adapters::billing_read::BillingError),
    #[error("billing schema maintenance failed: {0}")]
    Schema(#[from] bot_adapters::billing_schema::BillingSchemaError),
}

pub fn run_maintenance(
    options: MaintenanceOptions<'_>,
) -> Result<MaintenanceReport, MaintenanceError> {
    if options.ai_ledger_retention_days <= 0 {
        return Err(MaintenanceError::InvalidRetention);
    }
    let redis = run_redis_maintenance(
        options.redis_endpoint,
        options.redis_maxmemory,
        options.redis_maxmemory_policy,
    )?;
    let ledger = if let Some(database_url) = options.database_url {
        BillingSchemaRepository::new(database_url).ensure_schema()?;
        serde_json::to_value(
            BillingRepository::new(database_url)
                .purge_expired_ai_ledger_events(options.ai_ledger_retention_days)?,
        )
        .unwrap_or_else(|_| json!({"skipped": true, "reason": "serialization failed"}))
    } else {
        json!({"skipped": true, "reason": "postgres not configured"})
    };
    Ok(MaintenanceReport { redis, ledger })
}

#[cfg(test)]
mod tests {
    use bot_adapters::redis_connection::RedisEndpoint;

    use super::{MaintenanceError, MaintenanceOptions, run_maintenance};

    #[test]
    fn invalid_retention_fails_before_redis_io() {
        let result = run_maintenance(MaintenanceOptions {
            redis_endpoint: &RedisEndpoint {
                host: "synthetic.invalid".to_owned(),
                port: 1,
                password: None,
            },
            database_url: None,
            redis_maxmemory: "256mb",
            redis_maxmemory_policy: "allkeys-lru",
            ai_ledger_retention_days: 0,
        });
        assert!(matches!(result, Err(MaintenanceError::InvalidRetention)));
    }
}
