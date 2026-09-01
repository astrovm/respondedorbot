//! Shared blocking HTTP client construction.

use std::sync::OnceLock;

use reqwest::blocking::Client;

pub(crate) fn shared_client(
    slot: &'static OnceLock<Client>,
    build: impl FnOnce() -> Result<Client, reqwest::Error>,
) -> Result<Client, reqwest::Error> {
    if let Some(client) = slot.get() {
        return Ok(client.clone());
    }
    let client = build()?;
    Ok(slot.get_or_init(|| client).clone())
}
