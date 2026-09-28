use std::error::Error;
use std::process::Command;

/// Every outbound HTTP request is sent to a closed local port, so the process
/// never reaches a real provider.
const CLOSED_PROXY: &str = "http://127.0.0.1:1";

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
    fn new(soul: &[u8]) -> Result<Self, std::io::Error> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "botd-cli-lifecycle-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(path.join("workspace"))?;
        std::fs::write(path.join("workspace/SOUL.md"), soul)?;
        std::fs::write(path.join("workspace/RULES.md"), "synthetic rules")?;
        Ok(Self(path))
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _result = std::fs::remove_dir_all(&self.0);
    }
}

fn configured_botd(workspace: &Workspace, database_url: &str) -> Command {
    let mut command = isolated_botd();
    command
        .current_dir(&workspace.0)
        .env("TELEGRAM_TOKEN", "synthetic-telegram-token")
        .env("TELEGRAM_USERNAME", "synthetic_test_bot")
        .env("TELEGRAM_LONG_POLL_SECONDS", "1")
        .env("SUPABASE_POSTGRES_URL", database_url)
        .env("COINMARKETCAP_KEY", "synthetic-market-key")
        .env("OPENROUTER_API_KEY", "synthetic-ai-key")
        .env("REDIS_HOST", "127.0.0.1")
        .env("HTTPS_PROXY", CLOSED_PROXY)
        .env("HTTP_PROXY", CLOSED_PROXY)
        .env("ALL_PROXY", CLOSED_PROXY);
    command
}

#[test]
fn unreadable_workspace_prompt_is_reported_as_an_io_failure() -> Result<(), Box<dyn Error>> {
    let workspace = Workspace::new(&[0xff, 0xfe, 0xfd])?;
    let output = configured_botd(
        &workspace,
        "postgresql://synthetic:synthetic@db.example.test/database?sslmode=require",
    )
    .env("REDIS_PORT", "1")
    .arg("--check-config")
    .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("FATAL: could not read the workspace prompt:"),
        "{stderr}"
    );
    Ok(())
}
