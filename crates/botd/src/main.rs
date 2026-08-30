//! Final Rust process entrypoint.
//!
//! The Python runtime remains authoritative until the Telegram cutover phase.

fn main() {
    let _protocol_version = bot_adapters::migration_protocol_version();
}
