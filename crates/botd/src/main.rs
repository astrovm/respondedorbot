//! Native process entrypoint.

use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use botd::application::run_production;
use botd::config::{MaintenanceConfig, ProductionConfig, TaskVerificationConfig};
use botd::maintenance::{MaintenanceOptions, run_maintenance};
use botd::task_service::verify_tasks_once;

fn main() -> ExitCode {
    if std::env::args().any(|argument| argument == "--maintenance") {
        return maintenance();
    }
    if std::env::args().any(|argument| argument == "--verify-tasks") {
        return verify_tasks();
    }
    if std::env::args().any(|argument| argument == "--check-config") {
        return check_config();
    }
    native_runtime()
}

fn check_config() -> ExitCode {
    match ProductionConfig::from_env() {
        Ok(config) => {
            println!(
                "configuration valid: bot={} handler_workers={} long_poll_seconds={} trigger_words={}",
                config.bot_name,
                config.runtime.handler_workers,
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
