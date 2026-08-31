//! Native process entrypoint.

use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use botd::config::{MaintenanceConfig, RuntimeConfig, TaskVerificationConfig};
use botd::maintenance::{MaintenanceOptions, run_maintenance};
use botd::task_service::verify_tasks_once;

fn main() -> ExitCode {
    if std::env::args().any(|argument| argument == "--maintenance") {
        return maintenance();
    }
    if std::env::args().any(|argument| argument == "--verify-tasks") {
        return verify_tasks();
    }
    match RuntimeConfig::from_env() {
        Ok(config) if std::env::args().any(|argument| argument == "--check-config") => {
            println!(
                "configuration valid: handler_workers={} long_poll_seconds={}",
                config.handler_workers,
                config.long_poll_timeout.as_secs()
            );
            ExitCode::SUCCESS
        }
        Ok(_) => {
            eprintln!("native dispatcher is not authoritative; use --check-config");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("FATAL: {error}");
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
