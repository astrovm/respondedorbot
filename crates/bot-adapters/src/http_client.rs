//! Shared blocking HTTP client construction.

use std::sync::{Mutex, OnceLock, PoisonError};

use reqwest::blocking::Client;

/// Serializes first builds so dispatchers composed in parallel wait for one
/// client instead of each loading the certificate store.
static BUILD: Mutex<()> = Mutex::new(());

pub(crate) fn shared_client(
    slot: &'static OnceLock<Client>,
    build: impl FnOnce() -> Result<Client, reqwest::Error>,
) -> Result<Client, reqwest::Error> {
    if let Some(client) = slot.get() {
        return Ok(client.clone());
    }
    let _guard = BUILD.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(client) = slot.get() {
        return Ok(client.clone());
    }
    let client = build()?;
    Ok(slot.get_or_init(|| client).clone())
}
