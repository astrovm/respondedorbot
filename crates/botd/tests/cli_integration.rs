use std::error::Error;
use std::process::Command;

fn botd(arguments: &[&str]) -> Result<std::process::Output, std::io::Error> {
    let mut command = isolated_botd();
    command
        .args(arguments)
        .env("REDIS_PORT", "invalid")
        .output()
}

fn isolated_botd() -> Command {
    let coverage_profile = std::env::var_os("LLVM_PROFILE_FILE");
    let mut command = Command::new(env!("CARGO_BIN_EXE_botd"));
    command.env_clear();
    if let Some(coverage_profile) = coverage_profile {
        command.env("LLVM_PROFILE_FILE", coverage_profile);
    }
    command
}

struct Workspace(std::path::PathBuf);

impl Workspace {
    fn new() -> Result<Self, std::io::Error> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "botd-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(path.join("workspace"))?;
        std::fs::write(path.join("workspace/SOUL.md"), "synthetic personality")?;
        std::fs::write(path.join("workspace/RULES.md"), "synthetic rules")?;
        Ok(Self(path))
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _result = std::fs::remove_dir_all(&self.0);
    }
}

fn configured_botd(workspace: &Workspace) -> Command {
    let mut command = isolated_botd();
    command
        .current_dir(&workspace.0)
        .env("TELEGRAM_TOKEN", "synthetic-telegram-token")
        .env("TELEGRAM_USERNAME", "synthetic_test_bot")
        .env(
            "SUPABASE_POSTGRES_URL",
            "postgresql://synthetic:synthetic@db.example.test/database?sslmode=require",
        )
        .env("COINMARKETCAP_KEY", "synthetic-market-key")
        .env("OPENROUTER_API_KEY", "synthetic-ai-key");
    command
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
    let workspace = Workspace::new()?;
    let output = configured_botd(&workspace).arg("--check-config").output()?;
    assert!(output.status.success());
    assert!(String::from_utf8(output.stdout)?.contains("configuration valid"));
    Ok(())
}

#[test]
fn operational_cli_modes_report_storage_connection_failures() -> Result<(), Box<dyn Error>> {
    let cases = [
        (vec!["--maintenance"], "maintenance failed:"),
        (
            vec!["--verify-tasks"],
            "scheduled-task verification failed:",
        ),
    ];
    for (arguments, expected_error) in cases {
        let output = isolated_botd()
            .args(arguments)
            .env("REDIS_HOST", "127.0.0.1")
            .env("REDIS_PORT", "1")
            .env(
                "SUPABASE_POSTGRES_URL",
                "postgresql://synthetic:synthetic@db.example.test/database?sslmode=require",
            )
            .output()?;
        assert!(!output.status.success());
        assert!(String::from_utf8(output.stderr)?.contains(expected_error));
    }
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

    let verify = isolated_botd()
        .arg("--verify-tasks")
        .env("REDIS_HOST", &redis_host)
        .env("REDIS_PORT", &redis_port)
        .env("BOT_INSTANCE_NAME", "synthetic-cli")
        .output()?;
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8(verify.stderr)?
    );

    let maintenance = isolated_botd()
        .arg("--maintenance")
        .env("REDIS_HOST", redis_host)
        .env("REDIS_PORT", &redis_port)
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

#[test]
fn production_startup_composes_real_services_before_reporting_an_unavailable_cache()
-> Result<(), Box<dyn Error>> {
    let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
        return Ok(());
    };
    let workspace = Workspace::new()?;
    let output = configured_botd(&workspace)
        .env("TELEGRAM_TOKEN", "synthetic-telegram-token")
        .env("TELEGRAM_USERNAME", "synthetic_test_bot")
        .env("SUPABASE_POSTGRES_URL", database_url)
        .env("REDIS_HOST", "127.0.0.1")
        .env("REDIS_PORT", "1")
        .env("COINMARKETCAP_KEY", "synthetic-market-key")
        .env("OPENROUTER_API_KEY", "synthetic-ai-key")
        .env("BOT_INSTANCE_NAME", "synthetic-startup")
        .output()?;

    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr)?;
    assert!(error.contains("FATAL:"), "{error}");
    assert!(
        error.contains("Redis") || error.contains("redis") || error.contains("Connection"),
        "{error}"
    );
    Ok(())
}

#[test]
fn configuration_requires_both_nonempty_personality_files() -> Result<(), Box<dyn Error>> {
    for name in ["SOUL.md", "RULES.md"] {
        let workspace = Workspace::new()?;
        let path = workspace.0.join("workspace").join(name);
        for contents in [Some(" \n"), None] {
            if let Some(contents) = contents {
                std::fs::write(&path, contents)?;
            } else {
                std::fs::remove_file(&path)?;
            }
            let output = configured_botd(&workspace)
                .arg("--check-config")
                .env("BOT_SYSTEM_PROMPT", "obsolete synthetic override")
                .output()?;
            assert!(!output.status.success());
            assert!(String::from_utf8(output.stderr)?.contains("must both exist and contain text"));
        }
    }
    Ok(())
}
