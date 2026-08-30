//! Native process entrypoint.

use std::process::ExitCode;

use botd::config::RuntimeConfig;

fn main() -> ExitCode {
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
