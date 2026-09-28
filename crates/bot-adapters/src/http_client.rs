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
    // Another caller may have built the client while this one waited.
    let client = slot.get().cloned().map_or_else(build, Ok)?;
    Ok(slot.get_or_init(|| client).clone())
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use reqwest::blocking::Client;

    use super::shared_client;

    static SLOT: OnceLock<Client> = OnceLock::new();
    static BUILDS: AtomicUsize = AtomicUsize::new(0);

    fn counted_build() -> Result<Client, reqwest::Error> {
        BUILDS.fetch_add(1, Ordering::SeqCst);
        Client::builder().build()
    }

    #[test]
    fn builds_the_shared_client_once_and_reuses_it() {
        assert!(shared_client(&SLOT, counted_build).is_ok());
        assert!(shared_client(&SLOT, counted_build).is_ok());
        assert!(SLOT.get().is_some());
        assert_eq!(BUILDS.load(Ordering::SeqCst), 1);
    }
}
