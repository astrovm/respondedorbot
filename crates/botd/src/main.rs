//! Native process entrypoint.

use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use botd::config::{RuntimeConfig, TaskVerificationConfig};
use botd::task_service::verify_tasks_once;

fn main() -> ExitCode {
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
