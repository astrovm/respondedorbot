//! Composition for the explicit persisted-data migration command.

use bot_adapters::legacy_migration::{
    LegacyMigrationError, MigrationMode, PostgresMigrationReport, RedisMigrationReport,
    migrate_postgres, migrate_redis,
};
use bot_adapters::redis_connection::RedisEndpoint;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct LegacyMigrationReport {
    pub mode: &'static str,
    pub redis: RedisMigrationReport,
    pub postgres: PostgresMigrationReport,
}

pub fn run_legacy_migration(
    redis_endpoint: &RedisEndpoint,
    database_url: &str,
    mode: MigrationMode,
    now: i64,
) -> Result<LegacyMigrationReport, LegacyMigrationError> {
    let redis = migrate_redis(redis_endpoint, mode, now)?;
    let postgres = migrate_postgres(database_url, mode)?;
    Ok(LegacyMigrationReport {
        mode: match mode {
            MigrationMode::DryRun => "dry-run",
            MigrationMode::Apply => "apply",
        },
        redis,
        postgres,
    })
}
