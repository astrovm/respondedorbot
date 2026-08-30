//! External service adapters for respondedorbot.

pub mod bcra;
pub mod billing_read;
pub mod billing_schema;
pub mod chat_config;
pub mod coinmarketcap;
pub mod compaction_job;
pub mod criptoya;
pub mod dollar;
pub mod finviz;
pub mod firecrawl;
pub mod giphy;
pub mod giphy_pool;
pub mod link_preview;
pub mod openrouter_generation;
pub mod polymarket;
pub mod redis_chat_admin;
pub mod redis_compaction_queue;
pub mod redis_connection;
pub mod redis_json_cache;
pub mod redis_maintenance;
pub mod redis_media_cache;
pub mod redis_message_state;
pub mod redis_task_store;
pub mod request_cache;
pub mod stock_pool;
pub mod telegram_actions;
pub mod telegram_chat_admin;
pub mod telegram_http;
pub mod telegram_polling;
pub mod weather;
pub mod yahoo_finance;

/// Return the core protocol version used by this adapter build.
#[must_use]
pub const fn migration_protocol_version() -> u16 {
    bot_core::migration_protocol_version()
}

#[cfg(test)]
mod tests {
    use super::migration_protocol_version;

    #[test]
    fn uses_a_nonzero_core_protocol_version() {
        assert!(migration_protocol_version() > 0);
    }
}
