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

    fn test_redis_endpoint() -> Option<RedisEndpoint> {
        let port = std::env::var("TEST_REDIS_PORT").ok()?.parse().ok()?;
        Some(RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
            port,
            password: None,
        })
    }

    fn options<'a>(
        endpoint: &'a RedisEndpoint,
        database_url: Option<&'a str>,
    ) -> MaintenanceOptions<'a> {
        MaintenanceOptions {
            redis_endpoint: endpoint,
            database_url,
            redis_maxmemory: "256mb",
            redis_maxmemory_policy: "allkeys-lru",
            ai_ledger_retention_days: 1,
        }
    }

    #[test]
    fn maintenance_can_run_without_postgres() -> Result<(), String> {
        test_redis_endpoint().map_or(Ok(()), |endpoint| {
            let report = run_maintenance(options(&endpoint, None)).map_err(|e| e.to_string())?;
            assert_eq!(report.ledger["reason"], "postgres not configured");
            assert!(report.redis.maxmemory.is_some());
            assert!(report.redis.maxmemory_policy.is_some());
            Ok(())
        })
    }

    #[test]
    fn maintenance_purges_the_postgres_ledger_when_configured() -> Result<(), String> {
        let stores = test_redis_endpoint().zip(std::env::var("TEST_DATABASE_URL").ok());
        stores.map_or(Ok(()), |(endpoint, database_url)| {
            let report = run_maintenance(options(&endpoint, Some(&database_url)))
                .map_err(|error| error.to_string())?;
            assert!(report.ledger.is_object());
            assert!(report.ledger.get("skipped").is_none());
            Ok(())
        })
    }

    #[test]
    fn unreachable_postgres_is_reported_after_redis_maintenance() -> Result<(), String> {
        test_redis_endpoint().map_or(Ok(()), |endpoint| {
            let result = run_maintenance(options(
                &endpoint,
                Some("postgresql://synthetic:synthetic@127.0.0.1:1/database?sslmode=disable"),
            ));
            assert!(matches!(result, Err(MaintenanceError::Schema(_))));
            Ok(())
        })
    }
}
