//! Typed domain and application logic for respondedorbot.

pub mod base_conversion;
pub mod command_parsing;
pub mod credit_units;
pub mod market_context;
pub mod price_queries;
pub mod random_selection;
pub mod routing;
pub mod task_triggers;

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
