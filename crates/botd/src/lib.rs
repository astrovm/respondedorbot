//! Application composition and lifecycle for the native bot process.

pub mod ai_dispatch;
pub mod application;
pub mod background;
pub mod chat_members_tool;
pub mod chat_provider;
pub mod chat_tool_loop;
pub mod cli;
pub mod compaction_adapters;
pub mod compaction_scheduler;
pub mod compaction_worker;
pub mod composition;
pub mod config;
pub mod conversation;
pub mod conversation_adapters;
pub mod dispatcher;
pub mod firecrawl_tool;
pub mod hacker_news_tool;
pub mod maintenance;
pub mod market_tools;
pub mod media;
pub mod media_adapters;
pub mod native_ai;
pub mod native_tools;
pub mod operational_reporting;
pub mod poll_tools;
pub mod price_refresh;
pub mod random_tool;
pub mod reconciliation;
pub mod runtime;
pub mod scheduler;
pub mod task_executor;
pub mod task_service;
pub mod task_tools;
pub mod telegram_stream;
pub mod tool_output;
pub mod tool_requests;
pub mod web_fetch_tool;
pub mod youtube;

/// Renders an adapter or store error as the plain-text message the runtime
/// ports report. Shared so every `map_err` site uses one formatting path.
pub(crate) fn error_text(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// Environment for tests that use the disposable local PostgreSQL and Redis
/// services (`TEST_DATABASE_URL`, `TEST_REDIS_HOST`, `TEST_REDIS_PORT`,
/// `TEST_REDIS_PASSWORD`); each test skips its storage checks when unset.
#[cfg(test)]
pub(crate) mod test_env {
    use bot_adapters::redis_connection::RedisEndpoint;

    pub(crate) type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn non_empty(value: String) -> Option<String> {
        (!value.is_empty()).then_some(value)
    }

    pub(crate) fn database_url() -> Option<String> {
        std::env::var("TEST_DATABASE_URL").ok().and_then(non_empty)
    }

    pub(crate) fn redis_endpoint() -> Option<RedisEndpoint> {
        let port = std::env::var("TEST_REDIS_PORT").ok()?.parse().ok()?;
        let host = std::env::var("TEST_REDIS_HOST").ok().and_then(non_empty);
        Some(RedisEndpoint {
            host: host.unwrap_or(String::from("127.0.0.1")),
            port,
            password: std::env::var("TEST_REDIS_PASSWORD")
                .ok()
                .and_then(non_empty),
        })
    }

    #[test]
    fn error_text_and_environment_helpers_normalize_values() {
        assert_eq!(
            super::error_text(std::fmt::Error),
            "an error occurred when formatting an argument"
        );
        assert_eq!(non_empty(String::new()), None);
        assert_eq!(non_empty("value".to_owned()), Some("value".to_owned()));
    }
}
