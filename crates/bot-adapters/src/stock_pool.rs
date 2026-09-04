//! Redis-cached Finviz mega-cap stock pool.

use serde_json::Value;

use crate::finviz::{FinvizTransport, ScreenerOutcome, fetch_with};
use crate::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};

const CACHE_KEY: &str = "market:stock_screener:mega_cap";
const CACHE_TTL_SECONDS: i64 = 3_600;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StockPoolLoad {
    pub symbols: Vec<String>,
    pub diagnostics: Vec<String>,
}

pub trait StockPoolCache {
    type Error: std::fmt::Display;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error>;

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error>;
}

impl StockPoolCache for RedisJsonCache {
    type Error = RedisJsonCacheError;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        RedisJsonCache::get(self, key)
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
        RedisJsonCache::set(self, key, value, Some(ttl_seconds)).map(|_stored| ())
    }
}

fn python_text(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        value => value.to_string(),
    }
}

fn decode_symbols(raw: &str) -> Option<Vec<String>> {
    serde_json::from_str::<Value>(raw)
        .ok()?
        .as_array()
        .map(|values| values.iter().map(python_text).collect())
}

#[must_use]
pub fn load_stock_pool<T: FinvizTransport, C: StockPoolCache>(
    transport: &T,
    cache: &mut C,
) -> StockPoolLoad {
    let mut diagnostics = Vec::new();
    match cache.get(CACHE_KEY) {
        Ok(Some(raw)) => {
            if let Some(symbols) = decode_symbols(&raw) {
                return StockPoolLoad {
                    symbols,
                    diagnostics,
                };
            }
            diagnostics.push(format!("invalid JSON list in Redis key {CACHE_KEY}"));
        }
        Ok(None) => {}
        Err(error) => diagnostics.push(format!("could not read Redis key {CACHE_KEY}: {error}")),
    }

    let symbols = match fetch_with(transport) {
        ScreenerOutcome::Success { symbols } => symbols,
        ScreenerOutcome::HttpError { status_code } => {
            diagnostics.push(format!("Finviz returned HTTP {status_code}"));
            Vec::new()
        }
        ScreenerOutcome::TransportError { kind } => {
            diagnostics.push(format!("Finviz transport failed: {kind:?}"));
            Vec::new()
        }
    };
    if !symbols.is_empty() {
        match serde_json::to_string(&symbols) {
            Ok(encoded) => {
                if let Err(error) = cache.set(CACHE_KEY, &encoded, CACHE_TTL_SECONDS) {
                    diagnostics.push(format!("could not write Redis key {CACHE_KEY}: {error}"));
                }
            }
            Err(error) => diagnostics.push(format!("could not encode Finviz pool: {error}")),
        }
    }
    StockPoolLoad {
        symbols,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use super::{StockPoolCache, load_stock_pool};
    use crate::finviz::{FinvizTransport, HttpResponse, TransportFailureKind};

    #[derive(Default)]
    struct Cache {
        gets: VecDeque<Result<Option<String>, &'static str>>,
        set_result: Option<Result<(), &'static str>>,
        writes: Vec<(String, String, i64)>,
    }

    impl StockPoolCache for Cache {
        type Error = &'static str;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            self.gets.pop_front().unwrap_or(Ok(None))
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            self.writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            self.set_result.take().unwrap_or(Ok(()))
        }
    }

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        calls: RefCell<usize>,
    }

    impl FinvizTransport for Transport {
        fn fetch(&self) -> Result<HttpResponse, TransportFailureKind> {
            *self.calls.borrow_mut() += 1;
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(response: Result<HttpResponse, TransportFailureKind>) -> Transport {
        Transport {
            response: RefCell::new(Some(response)),
            calls: RefCell::new(0),
        }
    }

    #[test]
    fn cached_lists_are_authoritative_even_when_empty_and_coerce_values() {
        for (raw, expected) in [
            (
                r#"["AAPL",1,true,false,null]"#,
                vec!["AAPL", "1", "True", "False", "None"],
            ),
            ("[]", Vec::new()),
        ] {
            let mut cache = Cache {
                gets: VecDeque::from([Ok(Some(raw.to_owned()))]),
                ..Cache::default()
            };
            let transport = transport(Err(TransportFailureKind::Request));
            let load = load_stock_pool(&transport, &mut cache);
            assert_eq!(load.symbols, expected);
            assert_eq!(*transport.calls.borrow(), 0);
        }
    }

    #[test]
    fn cache_miss_fetches_and_writes_python_compatible_pool() {
        let transport = transport(Ok(HttpResponse {
            status_code: 200,
            body: r#"data-boxover-ticker="AAPL" data-boxover-company="Apple""#.to_owned(),
        }));
        let mut cache = Cache::default();
        let load = load_stock_pool(&transport, &mut cache);
        assert_eq!(load.symbols, ["AAPL"]);
        assert!(load.diagnostics.is_empty());
        assert_eq!(cache.writes[0].0, "market:stock_screener:mega_cap");
        assert_eq!(cache.writes[0].1, r#"["AAPL"]"#);
        assert_eq!(cache.writes[0].2, 3_600);
    }

    #[test]
    fn invalid_or_failed_cache_continues_and_failures_are_diagnostic() {
        let failed_transport = transport(Err(TransportFailureKind::Timeout));
        let mut cache = Cache {
            gets: VecDeque::from([Err("offline")]),
            ..Cache::default()
        };
        let load = load_stock_pool(&failed_transport, &mut cache);
        assert!(load.symbols.is_empty());
        assert_eq!(load.diagnostics.len(), 2);

        let transport = transport(Ok(HttpResponse {
            status_code: 200,
            body: r#"data-boxover-ticker="MSFT" data-boxover-company="Microsoft""#.to_owned(),
        }));
        let mut cache = Cache {
            gets: VecDeque::from([Ok(Some("not-json".to_owned()))]),
            set_result: Some(Err("readonly")),
            ..Cache::default()
        };
        let load = load_stock_pool(&transport, &mut cache);
        assert_eq!(load.symbols, ["MSFT"]);
        assert_eq!(load.diagnostics.len(), 2);
    }

    #[test]
    fn empty_and_http_error_results_are_not_cached() {
        for response in [
            Ok(HttpResponse {
                status_code: 200,
                body: String::new(),
            }),
            Ok(HttpResponse {
                status_code: 503,
                body: String::new(),
            }),
        ] {
            let mut cache = Cache::default();
            let load = load_stock_pool(&transport(response), &mut cache);
            assert!(load.symbols.is_empty());
            assert!(cache.writes.is_empty());
        }
    }
}
