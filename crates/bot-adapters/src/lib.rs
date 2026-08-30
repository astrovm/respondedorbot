//! External service adapters for respondedorbot.

pub mod compaction_job;
pub mod redis_media_cache;

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
