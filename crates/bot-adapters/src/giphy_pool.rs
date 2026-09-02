//! Redis-compatible Giphy greeting pool loading and stale fallback.

use bot_core::greeting_commands::GreetingCategory;

use crate::giphy::{GiphyTransport, SearchOutcome, search_with};
use crate::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};

const FRESH_TTL_SECONDS: i64 = 86_400;
const STALE_TTL_SECONDS: i64 = 7 * 86_400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolLoad {
    pub urls: Vec<String>,
    pub diagnostics: Vec<String>,
}

pub trait GiphyPoolCache {
    type Error: std::fmt::Display;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error>;

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error>;
}

impl GiphyPoolCache for RedisJsonCache {
    type Error = RedisJsonCacheError;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        RedisJsonCache::get(self, key)
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
        RedisJsonCache::set(self, key, value, Some(ttl_seconds)).map(|_stored| ())
    }
}

fn decode_pool(raw: &str) -> Option<Vec<String>> {
    serde_json::from_str(raw).ok()
}

fn search_diagnostic(category: GreetingCategory, term: &str, outcome: &SearchOutcome) -> String {
    let category = category.cache_name();
    match outcome {
        SearchOutcome::Success { .. } => {
            format!("Giphy {category} search for {term:?} unexpectedly succeeded")
        }
        SearchOutcome::HttpError { status_code } => {
            format!("Giphy {category} search for {term:?} returned HTTP {status_code}")
        }
        SearchOutcome::InvalidJson => {
            format!("Giphy {category} search for {term:?} returned invalid JSON")
        }
        SearchOutcome::InvalidPayload => {
            format!("Giphy {category} search for {term:?} returned an invalid payload")
        }
        SearchOutcome::TransportError { kind } => {
            format!("Giphy {category} search for {term:?} transport failed: {kind:?}")
        }
    }
}

#[must_use]
pub fn load_giphy_pool<T, C>(
    transport: &T,
    cache: &mut C,
    api_key: Option<&str>,
    category: GreetingCategory,
    mut next_offset: impl FnMut() -> u16,
) -> PoolLoad
where
    T: GiphyTransport,
    C: GiphyPoolCache,
{
    let fresh_key = format!("giphy_pool:{}", category.cache_name());
    let stale_key = format!("giphy_pool_stale:{}", category.cache_name());
    let mut diagnostics = Vec::new();
    match cache.get(&fresh_key) {
        Ok(Some(raw)) => {
            if let Some(urls) = decode_pool(&raw) {
                return PoolLoad { urls, diagnostics };
            }
            diagnostics.push(format!("invalid JSON list in Redis key {fresh_key}"));
        }
        Ok(None) => {}
        Err(error) => diagnostics.push(format!("could not read Redis key {fresh_key}: {error}")),
    }

    let mut urls = Vec::new();
    if let Some(api_key) = api_key.filter(|value| !value.is_empty()) {
        for term in category.search_terms() {
            let offset = next_offset();
            let outcome = search_with(transport, api_key, term, offset);
            match outcome {
                SearchOutcome::Success { urls: found } => urls.extend(found),
                failure => diagnostics.push(search_diagnostic(category, term, &failure)),
            }
        }
    }

    if !urls.is_empty() {
        let Ok(encoded) = serde_json::to_string(&urls) else {
            diagnostics.push("could not encode Giphy pool for Redis".to_owned());
            return PoolLoad { urls, diagnostics };
        };
        for (key, ttl) in [
            (&fresh_key, FRESH_TTL_SECONDS),
            (&stale_key, STALE_TTL_SECONDS),
        ] {
            if let Err(error) = cache.set(key, &encoded, ttl) {
                diagnostics.push(format!("could not write Redis key {key}: {error}"));
            }
        }
        return PoolLoad { urls, diagnostics };
    }

    match cache.get(&stale_key) {
        Ok(Some(raw)) => {
            if let Some(stale) = decode_pool(&raw) {
                urls = stale;
            } else {
                diagnostics.push(format!("invalid JSON list in Redis key {stale_key}"));
            }
        }
        Ok(None) => {}
        Err(error) => diagnostics.push(format!("could not read Redis key {stale_key}: {error}")),
    }
    PoolLoad { urls, diagnostics }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use bot_core::greeting_commands::GreetingCategory;

    use super::{GiphyPoolCache, load_giphy_pool};
    use crate::giphy::{GiphyTransport, HttpResponse, SearchRequest, TransportFailureKind};

    #[derive(Default)]
    struct Cache {
        gets: VecDeque<Result<Option<String>, &'static str>>,
        sets: Vec<(String, String, i64)>,
        fail_sets: bool,
    }

    impl GiphyPoolCache for Cache {
        type Error = &'static str;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            self.gets.pop_front().unwrap_or(Ok(None))
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            if self.fail_sets {
                return Err("synthetic write failure");
            }
            self.sets
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            Ok(())
        }
    }

    struct Transport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<SearchRequest>>,
    }

    impl GiphyTransport for Transport {
        fn search(&self, request: &SearchRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(responses: Vec<Result<HttpResponse, TransportFailureKind>>) -> Transport {
        Transport {
            responses: RefCell::new(responses.into()),
            requests: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn fresh_pool_is_authoritative_even_when_empty() {
        for raw in [r#"["https://example.test/cached.gif"]"#, "[]"] {
            let mut cache = Cache {
                gets: VecDeque::from([Ok(Some(raw.to_owned()))]),
                ..Cache::default()
            };
            let transport = transport(Vec::new());
            let load = load_giphy_pool(
                &transport,
                &mut cache,
                Some("synthetic-key"),
                GreetingCategory::Morning,
                || unreachable!(),
            );
            assert_eq!(load.urls.is_empty(), raw == "[]");
            assert!(transport.requests.borrow().is_empty());
            assert!(cache.sets.is_empty());
        }
    }

    #[test]
    fn fetches_all_terms_and_writes_fresh_and_stale_compatible_keys() {
        let success = |url: &str| {
            Ok(HttpResponse {
                status_code: 200,
                body: format!(r#"{{"data":[{{"images":{{"original":{{"url":"{url}"}}}}}}]}}"#),
            })
        };
        let search_transport = transport(vec![
            success("https://example.test/1.gif"),
            Err(TransportFailureKind::Timeout),
            success("https://example.test/3.gif"),
            success("https://example.test/4.gif"),
        ]);
        let mut cache = Cache::default();
        let mut offsets = [3, 5, 8, 13].into_iter();
        let load = load_giphy_pool(
            &search_transport,
            &mut cache,
            Some("synthetic-key"),
            GreetingCategory::Morning,
            || offsets.next().unwrap_or(0),
        );
        assert_eq!(load.urls.len(), 3);
        assert_eq!(load.diagnostics.len(), 1);
        assert_eq!(search_transport.requests.borrow()[1].term, "buenos dias");
        assert_eq!(search_transport.requests.borrow()[3].offset, 13);
        assert_eq!(cache.sets[0].0, "giphy_pool:gm");
        assert_eq!(cache.sets[0].2, 86_400);
        assert_eq!(cache.sets[1].0, "giphy_pool_stale:gm");
        assert_eq!(cache.sets[1].2, 604_800);
    }

    #[test]
    fn missing_key_or_failed_search_uses_stale_pool_with_diagnostics() {
        let failed_transport = transport(vec![
            Err(TransportFailureKind::Connection),
            Err(TransportFailureKind::Connection),
            Err(TransportFailureKind::Connection),
            Err(TransportFailureKind::Connection),
        ]);
        let mut cache = Cache {
            gets: VecDeque::from([
                Err("fresh unavailable"),
                Ok(Some(r#"["https://example.test/stale.gif"]"#.to_owned())),
            ]),
            ..Cache::default()
        };
        let mut offsets = [0; 4].into_iter();
        let load = load_giphy_pool(
            &failed_transport,
            &mut cache,
            Some("synthetic-key"),
            GreetingCategory::Night,
            || offsets.next().unwrap_or(0),
        );
        assert_eq!(load.urls, vec!["https://example.test/stale.gif"]);
        assert_eq!(load.diagnostics.len(), 5);

        let mut cache = Cache {
            gets: VecDeque::from([Ok(None), Ok(Some("bad".to_owned()))]),
            ..Cache::default()
        };
        let empty_transport = transport(Vec::new());
        let load = load_giphy_pool(
            &empty_transport,
            &mut cache,
            None,
            GreetingCategory::Night,
            || unreachable!(),
        );
        assert!(load.urls.is_empty());
        assert_eq!(load.diagnostics.len(), 1);
    }

    #[test]
    fn malformed_fresh_pool_and_provider_failures_keep_useful_diagnostics() {
        let search_transport = transport(vec![
            Ok(HttpResponse {
                status_code: 503,
                body: String::new(),
            }),
            Ok(HttpResponse {
                status_code: 200,
                body: "not json".to_owned(),
            }),
            Ok(HttpResponse {
                status_code: 200,
                body: r#"{"data":{}}"#.to_owned(),
            }),
            Ok(HttpResponse {
                status_code: 200,
                body: r#"{"data":[{"images":{"original":{"url":"https://example.test/result.gif"}}}]}"#
                    .to_owned(),
            }),
        ]);
        let mut cache = Cache {
            gets: VecDeque::from([Ok(Some("not json".to_owned()))]),
            fail_sets: true,
            ..Cache::default()
        };
        let load = load_giphy_pool(
            &search_transport,
            &mut cache,
            Some("synthetic-key"),
            GreetingCategory::Morning,
            || 0,
        );
        assert_eq!(load.urls, vec!["https://example.test/result.gif"]);
        assert_eq!(load.diagnostics.len(), 6);
    }
}
