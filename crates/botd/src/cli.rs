//! Command-line mode selection and process entry behavior.

use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::legacy_migration::MigrationMode;

use crate::application::run_production;
use crate::config::{
    LegacyMigrationConfig, MaintenanceConfig, ProductionConfig, TaskVerificationConfig,
};
use crate::legacy_migration::run_legacy_migration;
use crate::maintenance::{MaintenanceOptions, run_maintenance};
use crate::task_service::verify_tasks_once;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Runtime,
    MigrateLegacy { apply: bool },
    Maintenance,
    VerifyTasks,
    CheckConfig,
}

fn mode(arguments: impl IntoIterator<Item = impl AsRef<str>>) -> Mode {
    let arguments = arguments
        .into_iter()
        .map(|argument| argument.as_ref().to_owned())
        .collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == "--migrate-legacy")
    {
        Mode::MigrateLegacy {
            apply: arguments.iter().any(|argument| argument == "--apply"),
        }
    } else if arguments.iter().any(|argument| argument == "--maintenance") {
        Mode::Maintenance
    } else if arguments
        .iter()
        .any(|argument| argument == "--verify-tasks")
    {
        Mode::VerifyTasks
    } else if arguments
        .iter()
        .any(|argument| argument == "--check-config")
    {
        Mode::CheckConfig
    } else {
        Mode::Runtime
    }
}

pub fn run(arguments: impl IntoIterator<Item = impl AsRef<str>>) -> ExitCode {
    match mode(arguments) {
        Mode::MigrateLegacy { apply } => migrate_legacy(apply),
        Mode::Maintenance => maintenance(),
        Mode::VerifyTasks => verify_tasks(),
        Mode::CheckConfig => check_config(),
        Mode::Runtime => native_runtime(),
    }
}

fn migrate_legacy(apply: bool) -> ExitCode {
    let config = match LegacyMigrationConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("FATAL: {error}");
            return ExitCode::FAILURE;
        }
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64);
    let migration_mode = if apply {
        MigrationMode::Apply
    } else {
        MigrationMode::DryRun
    };
    match run_legacy_migration(
        &config.redis_endpoint,
        config.database_url(),
        migration_mode,
        now,
    ) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(encoded) => {
                println!("{encoded}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("migration report encoding failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("legacy migration failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn check_config() -> ExitCode {
    match ProductionConfig::from_env() {
        Ok(config) => {
            println!(
                "configuration valid: bot={} long_poll_seconds={} trigger_words={}",
                config.bot_name,
                config.runtime.long_poll_timeout.as_secs(),
                config.trigger_words.len(),
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("FATAL: {error}");
            ExitCode::FAILURE
        }
    }
}

fn native_runtime() -> ExitCode {
    let config = match ProductionConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("FATAL: {error}");
            return ExitCode::FAILURE;
        }
    };
    match run_production(&config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FATAL: native runtime failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn maintenance() -> ExitCode {
    let config = match MaintenanceConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("FATAL: {error}");
            return ExitCode::FAILURE;
        }
    };
    match run_maintenance(MaintenanceOptions {
        redis_endpoint: &config.redis_endpoint,
        database_url: config.database_url(),
        redis_maxmemory: &config.redis_maxmemory,
        redis_maxmemory_policy: &config.redis_maxmemory_policy,
        ai_ledger_retention_days: config.ai_ledger_retention_days,
    }) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(encoded) => {
                println!("{encoded}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("maintenance report encoding failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("maintenance failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn verify_tasks() -> ExitCode {
    let config = match TaskVerificationConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("FATAL: {error}");
            return ExitCode::FAILURE;
        }
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64);
    match verify_tasks_once(&config.redis_endpoint, &config.owner_token, now) {
        Ok(step) => {
            println!("scheduled-task verification: {step:?}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("scheduled-task verification failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, mode};

    #[test]
    fn mode_selection_has_explicit_precedence_and_runtime_fallback() {
        assert_eq!(mode(["botd"]), Mode::Runtime);
        assert_eq!(mode(["botd", "--check-config"]), Mode::CheckConfig);
        assert_eq!(mode(["botd", "--verify-tasks"]), Mode::VerifyTasks);
        assert_eq!(mode(["botd", "--maintenance"]), Mode::Maintenance);
        assert_eq!(
            mode(["botd", "--migrate-legacy"]),
            Mode::MigrateLegacy { apply: false }
        );
        assert_eq!(
            mode(["botd", "--migrate-legacy", "--apply"]),
            Mode::MigrateLegacy { apply: true }
        );
        assert_eq!(
            mode(["botd", "--check-config", "--maintenance"]),
            Mode::Maintenance
        );
        assert_eq!(
            mode(["botd", "--maintenance", "--migrate-legacy"]),
            Mode::MigrateLegacy { apply: false }
        );
    }
}
