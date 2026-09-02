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
