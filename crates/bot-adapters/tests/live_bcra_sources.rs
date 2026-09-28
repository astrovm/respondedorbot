//! Opt-in check against the official BCRA sources (`BCRA_LIVE_TEST=1`).
//!
//! Kept out of the unit suite so default runs never touch the network.

use std::collections::HashMap;

use bot_adapters::bcra::{ReqwestBcraTransport, load_bcra};
use bot_adapters::request_cache::RequestCache;
use bot_core::locale::Locale;

#[derive(Default)]
struct MemoryCache(HashMap<String, String>);

impl RequestCache for MemoryCache {
    type Error = std::convert::Infallible;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        Ok(self.0.get(key).cloned())
    }

    fn set(&mut self, key: &str, value: &str, _ttl_seconds: i64) -> Result<(), Self::Error> {
        self.0.insert(key.to_owned(), value.to_owned());
        Ok(())
    }

    fn take(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        Ok(self.0.remove(key))
    }

    fn claim(&mut self, key: &str, value: &str, _ttl_seconds: i64) -> Result<bool, Self::Error> {
        Ok(self.0.insert(key.to_owned(), value.to_owned()).is_none())
    }
}

#[test]
fn live_official_sources_parse_when_explicitly_enabled() {
    if std::env::var("BCRA_LIVE_TEST").as_deref() != Ok("1") {
        return;
    }
    let transport = ReqwestBcraTransport::new();
    assert!(transport.is_ok());
    let Ok(transport) = transport else { return };
    let load = load_bcra(
        &transport,
        &mut MemoryCache::default(),
        Locale::Es,
        1_788_043_200,
    );
    assert!(load.text.is_some(), "diagnostics={:?}", load.diagnostics);
}
