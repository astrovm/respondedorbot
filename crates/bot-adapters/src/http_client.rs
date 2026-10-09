//! Shared blocking HTTP client construction.

use std::sync::{Mutex, OnceLock, PoisonError};

use reqwest::blocking::Client;
use serde::Serialize;

/// A complete desktop browser user agent. Cloudflare-fronted sites such as
/// Finviz reject truncated ones with a 403 challenge page.
pub(crate) const BROWSER_USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";

/// Why a request to a market or data provider failed before it got a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportFailureKind {
    Timeout,
    Connection,
    Request,
}

pub(crate) fn classify_error(error: reqwest::Error) -> TransportFailureKind {
    if error.is_timeout() {
        TransportFailureKind::Timeout
    } else if error.is_connect() {
        TransportFailureKind::Connection
    } else {
        TransportFailureKind::Request
    }
}

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

    use super::{TransportFailureKind, classify_error, shared_client};

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

    #[test]
    fn reqwest_failures_are_classified_by_cause() {
        use crate::web_fetch::reqwest_error_fixtures as fixtures;
        assert_eq!(
            fixtures::timeout().map(classify_error),
            Some(TransportFailureKind::Timeout)
        );
        assert_eq!(
            fixtures::connection().map(classify_error),
            Some(TransportFailureKind::Connection)
        );
        assert_eq!(
            fixtures::request().map(classify_error),
            Some(TransportFailureKind::Request)
        );
    }
}
