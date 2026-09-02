use std::error::Error;
use std::process::Command;

fn botd(arguments: &[&str]) -> Result<std::process::Output, std::io::Error> {
    Command::new(env!("CARGO_BIN_EXE_botd"))
        .args(arguments)
        .env_clear()
        .env("REDIS_PORT", "invalid")
        .output()
}

#[test]
fn every_cli_mode_reports_missing_configuration_without_starting_services()
-> Result<(), Box<dyn Error>> {
    for arguments in [
        Vec::new(),
        vec!["--check-config"],
        vec!["--maintenance"],
        vec!["--verify-tasks"],
    ] {
        let output = botd(&arguments)?;
        assert!(!output.status.success());
        assert!(String::from_utf8(output.stderr)?.contains("FATAL:"));
    }
    Ok(())
}

#[test]
fn valid_configuration_check_exits_successfully() -> Result<(), Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_botd"))
        .arg("--check-config")
        .env_clear()
        .env("TELEGRAM_TOKEN", "synthetic-telegram-token")
        .env("TELEGRAM_USERNAME", "synthetic_test_bot")
        .env(
            "SUPABASE_POSTGRES_URL",
            "postgresql://synthetic:synthetic@db.example.test/database?sslmode=require",
        )
        .env("COINMARKETCAP_KEY", "synthetic-market-key")
        .env("OPENROUTER_API_KEY", "synthetic-ai-key")
        .env("BOT_SYSTEM_PROMPT", "synthetic system prompt")
        .output()?;
    assert!(output.status.success());
    assert!(String::from_utf8(output.stdout)?.contains("configuration valid"));
    Ok(())
}

#[test]
fn operational_cli_modes_succeed_against_disposable_local_services() -> Result<(), Box<dyn Error>> {
    let Some(redis_port) = std::env::var("TEST_REDIS_PORT").ok() else {
        return Ok(());
    };
    let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
        return Ok(());
    };
    let redis_host = std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned());

    let verify = Command::new(env!("CARGO_BIN_EXE_botd"))
        .arg("--verify-tasks")
        .env_clear()
        .env("REDIS_HOST", &redis_host)
        .env("REDIS_PORT", &redis_port)
        .env("BOT_INSTANCE_NAME", "synthetic-cli")
        .output()?;
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8(verify.stderr)?
    );

    let maintenance = Command::new(env!("CARGO_BIN_EXE_botd"))
        .arg("--maintenance")
        .env_clear()
        .env("REDIS_HOST", redis_host)
        .env("REDIS_PORT", redis_port)
        .env("REDIS_MAXMEMORY", "64mb")
        .env("REDIS_MAXMEMORY_POLICY", "volatile-lru")
        .env("AI_LEDGER_RETENTION_DAYS", "30")
        .env("SUPABASE_POSTGRES_URL", database_url)
        .output()?;
    assert!(
        maintenance.status.success(),
        "{}",
        String::from_utf8(maintenance.stderr)?
    );
    let report: serde_json::Value = serde_json::from_slice(&maintenance.stdout)?;
    assert!(report.get("redis").is_some());
    assert!(report.get("ledger").is_some());
    Ok(())
}
