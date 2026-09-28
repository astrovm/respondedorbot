//! Native runtime startup failures that happen before any network service.

use std::error::Error;
use std::process::Command;

struct Workspace(std::path::PathBuf);

impl Workspace {
    fn new() -> Result<Self, std::io::Error> {
        let path = std::env::temp_dir().join(format!(
            "botd-startup-failure-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos())
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

#[test]
fn unreachable_billing_database_stops_startup_before_any_service() -> Result<(), Box<dyn Error>> {
    let workspace = Workspace::new()?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_botd"));
    command.env_clear();
    if let Some(profile) = std::env::var_os("LLVM_PROFILE_FILE") {
        command.env("LLVM_PROFILE_FILE", profile);
    }
    let output = command
        .current_dir(&workspace.0)
        .env("TELEGRAM_TOKEN", "synthetic-telegram-token")
        .env("TELEGRAM_USERNAME", "synthetic_test_bot")
        .env(
            "SUPABASE_POSTGRES_URL",
            "postgresql://synthetic:synthetic@127.0.0.1:1/synthetic?sslmode=disable",
        )
        .env("REDIS_HOST", "127.0.0.1")
        .env("REDIS_PORT", "1")
        .env("COINMARKETCAP_KEY", "synthetic-market-key")
        .env("OPENROUTER_API_KEY", "synthetic-ai-key")
        .output()?;

    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr)?;
    assert!(
        error.contains("FATAL: native runtime failed: could not initialize billing schema"),
        "{error}"
    );
    Ok(())
}
