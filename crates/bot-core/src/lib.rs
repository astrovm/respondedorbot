//! Typed domain and application logic for respondedorbot.

pub mod admin_reports;
pub mod ai_pricing;
pub mod ai_request;
pub mod ai_reserve;
pub mod ai_response_cleanup;
pub mod ai_usage;
pub mod base_conversion;
pub mod cache_policy;
pub mod command_normalization;
pub mod command_parsing;
pub mod compaction_policy;
pub mod config_callbacks;
pub mod credit_units;
pub mod devo;
pub mod hacker_news;
pub mod links;
pub mod market_context;
pub mod market_models;
pub mod message_state;
pub mod polymarket;
pub mod price_queries;
pub mod provider_chain;
pub mod provider_errors;
pub mod provider_retry;
pub mod provider_runtime_policy;
pub mod provider_stream_policy;
pub mod provider_tools;
pub mod provider_web_search;
pub mod random_reply;
pub mod random_selection;
pub mod routing;
pub mod rulo;
pub mod satoshi;
pub mod task_triggers;
pub mod telegram_streaming;
pub mod tool_registry;
pub mod weather;

/// Version of the temporary Python/Rust compatibility protocol.
pub const MIGRATION_PROTOCOL_VERSION: u16 = 1;

/// Return the temporary compatibility protocol version.
#[must_use]
pub const fn migration_protocol_version() -> u16 {
    MIGRATION_PROTOCOL_VERSION
}

#[cfg(test)]
mod tests {
    use super::{MIGRATION_PROTOCOL_VERSION, migration_protocol_version};

    #[test]
    fn exposes_the_declared_migration_protocol_version() {
        assert_eq!(migration_protocol_version(), MIGRATION_PROTOCOL_VERSION);
    }
}
