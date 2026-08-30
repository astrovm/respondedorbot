//! Typed Yahoo Finance chart adapter using the shared request cache.

use std::thread;
use std::time::Duration;

use bot_core::stocks::{StockQuote, parse_yahoo_quote, select_yahoo_symbol};
use reqwest::blocking::Client;
use serde_json::json;

use crate::request_cache::{
    JsonHttpResponse, RequestCache, load_cached_json, python_json_string, python_request_cache_key,
};

const CHART_URL: &str = "https://query1.finance.yahoo.com/v8/finance/chart";
const SEARCH_URL: &str = "https://query1.finance.yahoo.com/v1/finance/search";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const CACHE_TTL_SECONDS: i64 = 300;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YahooChartRequest {
    pub symbol: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YahooSearchRequest {
    pub query: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportFailureKind {
    Timeout,
    Connection,
    Request,
}

pub trait YahooFinanceTransport {
    fn chart(&self, request: &YahooChartRequest) -> Result<HttpResponse, TransportFailureKind>;

    fn search(&self, request: &YahooSearchRequest) -> Result<HttpResponse, TransportFailureKind>;

    fn before_retry(&self) {}
}

pub struct ReqwestYahooFinanceTransport {
    client: Client,
}

impl ReqwestYahooFinanceTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl YahooFinanceTransport for ReqwestYahooFinanceTransport {
    fn chart(&self, request: &YahooChartRequest) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(format!("{CHART_URL}/{}", request.symbol))
            .query(&[("range", "5d"), ("interval", "1d")])
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_error)
    }

    fn search(&self, request: &YahooSearchRequest) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(SEARCH_URL)
            .query(&[
                ("q", request.query.as_str()),
                ("quotesCount", "5"),
                ("newsCount", "0"),
            ])
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_error)
    }

    fn before_retry(&self) {
        thread::sleep(Duration::from_millis(500));
    }
}

fn classify_error(error: reqwest::Error) -> TransportFailureKind {
    if error.is_timeout() {
        TransportFailureKind::Timeout
    } else if error.is_connect() {
        TransportFailureKind::Connection
    } else {
        TransportFailureKind::Request
    }
}

fn cache_key(symbol: &str) -> String {
    let arguments = format!(
        "{{\"api_url\": \"{CHART_URL}/{symbol}\", \"headers\": {{\"User-Agent\": \"Mozilla/5.0\"}}, \"parameters\": {{\"interval\": \"1d\", \"range\": \"5d\"}}}}"
    );
    python_request_cache_key(&arguments)
}

fn search_cache_key(query: &str) -> String {
    let arguments = format!(
        "{{\"api_url\": \"{SEARCH_URL}\", \"headers\": {{\"User-Agent\": \"Mozilla/5.0\"}}, \"parameters\": {{\"newsCount\": 0, \"q\": {}, \"quotesCount\": 5}}}}",
        python_json_string(query)
    );
    python_request_cache_key(&arguments)
}

#[derive(Debug, Clone, PartialEq)]
pub struct YahooQuoteLoad {
    pub quote: Option<StockQuote>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YahooSymbolLoad {
    pub symbol: Option<String>,
    pub diagnostics: Vec<String>,
}

#[must_use]
pub fn load_quote<T: YahooFinanceTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    symbol: &str,
    now_unix: i64,
) -> YahooQuoteLoad {
    let request = YahooChartRequest {
        symbol: symbol.to_owned(),
    };
    let load = load_cached_json(
        cache,
        &cache_key(symbol),
        CACHE_TTL_SECONDS,
        now_unix,
        &format!("Yahoo chart request symbol={symbol}"),
        || {
            transport
                .chart(&request)
                .map(|response| JsonHttpResponse {
                    status_code: response.status_code,
                    body: response.body,
                })
                .map_err(|error| format!("transport {error:?}"))
        },
        || transport.before_retry(),
    );
    let quote = load
        .data
        .as_ref()
        .and_then(|data| parse_yahoo_quote(&json!({"data": data}), symbol));
    let mut diagnostics = load.diagnostics;
    if quote.is_none() {
        diagnostics.push(format!("Yahoo chart had no usable quote for {symbol}"));
    }
    YahooQuoteLoad { quote, diagnostics }
}

#[must_use]
pub fn load_symbol<T: YahooFinanceTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    query: &str,
    now_unix: i64,
) -> YahooSymbolLoad {
    let normalized = query.trim().trim_start_matches('$');
    let compact = normalized.replace(' ', "");
    let mut diagnostics = Vec::new();
    for (index, search_query) in [normalized, compact.as_str()].into_iter().enumerate() {
        if index == 1 && compact == normalized {
            continue;
        }
        let request = YahooSearchRequest {
            query: search_query.to_owned(),
        };
        let load = load_cached_json(
            cache,
            &search_cache_key(search_query),
            CACHE_TTL_SECONDS,
            now_unix,
            &format!("Yahoo search request query={search_query}"),
            || {
                transport
                    .search(&request)
                    .map(|response| JsonHttpResponse {
                        status_code: response.status_code,
                        body: response.body,
                    })
                    .map_err(|error| format!("transport {error:?}"))
            },
            || transport.before_retry(),
        );
        diagnostics.extend(load.diagnostics);
        if let Some(symbol) = load
            .data
            .as_ref()
            .and_then(|data| select_yahoo_symbol(&json!({"data": data})))
        {
            return YahooSymbolLoad {
                symbol: Some(symbol),
                diagnostics,
            };
        }
    }
    diagnostics.push(format!("Yahoo search had no usable symbol for {query}"));
    YahooSymbolLoad {
        symbol: None,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use crate::request_cache::RequestCache;

    use super::{
        HttpResponse, TransportFailureKind, YahooChartRequest, YahooFinanceTransport,
        YahooSearchRequest, cache_key, load_quote, load_symbol, search_cache_key,
    };

    struct Transport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<YahooChartRequest>>,
        searches: RefCell<Vec<YahooSearchRequest>>,
    }

    impl YahooFinanceTransport for Transport {
        fn chart(&self, request: &YahooChartRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }

        fn search(
            &self,
            request: &YahooSearchRequest,
        ) -> Result<HttpResponse, TransportFailureKind> {
            self.searches.borrow_mut().push(request.clone());
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[derive(Default)]
    struct Cache {
        values: VecDeque<Result<Option<String>, &'static str>>,
        writes: Vec<(String, String, i64)>,
    }

    impl RequestCache for Cache {
        type Error = &'static str;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            self.values.pop_front().unwrap_or(Ok(None))
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            self.writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            Ok(())
        }
    }

    fn response(body: &str) -> Result<HttpResponse, TransportFailureKind> {
        Ok(HttpResponse {
            status_code: 200,
            body: body.to_owned(),
        })
    }

    fn chart() -> &'static str {
        r#"{"chart":{"result":[{"meta":{"symbol":"BZ=F","regularMarketPrice":98.15,"chartPreviousClose":107.6,"currency":"USD"},"indicators":{"quote":[{"close":[107.6,98.15]}]}}]}}"#
    }

    #[test]
    fn cache_key_exactly_matches_python_sorted_json() {
        assert_eq!(
            cache_key("BZ=F"),
            "request_cache:e707742a639323bff2cb566da569aae5da343c2ad818cfe72f9858d4b3f95e5b"
        );
    }

    #[test]
    fn loads_and_caches_a_typed_chart_quote() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([response(chart())])),
            requests: RefCell::default(),
            searches: RefCell::default(),
        };
        let mut cache = Cache::default();
        let load = load_quote(&transport, &mut cache, "BZ=F", 100);
        assert_eq!(load.quote.as_ref().map(|quote| quote.price), Some(98.15));
        assert_eq!(
            load.quote.as_ref().map(|quote| quote.symbol.as_str()),
            Some("BZ=F")
        );
        assert!(load.diagnostics.is_empty());
        assert_eq!(transport.requests.borrow()[0].symbol, "BZ=F");
        assert_eq!(cache.writes[0].2, 300);
    }

    #[test]
    fn stale_cache_and_failures_are_diagnostic_without_panics() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                Err(TransportFailureKind::Timeout),
                Err(TransportFailureKind::Connection),
            ])),
            requests: RefCell::default(),
            searches: RefCell::default(),
        };
        let mut cache = Cache {
            values: VecDeque::from([Ok(Some(format!(r#"{{"timestamp":1,"data":{}}}"#, chart())))]),
            ..Cache::default()
        };
        let load = load_quote(&transport, &mut cache, "BZ=F", 1_000);
        assert!(load.quote.is_some());
        assert_eq!(load.diagnostics.len(), 2);

        let transport = Transport {
            responses: RefCell::new(VecDeque::from([response(r#"{"chart":{"result":[]}}"#)])),
            requests: RefCell::default(),
            searches: RefCell::default(),
        };
        let load = load_quote(&transport, &mut Cache::default(), "CL=F", 100);
        assert!(load.quote.is_none());
        assert!(load.diagnostics[0].contains("no usable quote"));
    }

    #[test]
    fn search_keys_match_python_and_resolution_tries_spaced_then_compact_queries() {
        assert_eq!(
            search_cache_key("Apple Inc"),
            "request_cache:ff75d1736468f2fe6c897e078a9a9ec6c2a82ca29ae38fc0b55621f87e5375f0"
        );
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                response(r#"{"quotes":[]}"#),
                response(r#"{"quotes":[{"quoteType":"EQUITY","symbol":"AAPL"}]}"#),
            ])),
            requests: RefCell::default(),
            searches: RefCell::default(),
        };
        let load = load_symbol(&transport, &mut Cache::default(), " $Apple Inc ", 100);
        assert_eq!(load.symbol.as_deref(), Some("AAPL"));
        assert_eq!(
            transport
                .searches
                .borrow()
                .iter()
                .map(|request| request.query.as_str())
                .collect::<Vec<_>>(),
            ["Apple Inc", "AppleInc"]
        );
    }

    #[test]
    fn search_deduplicates_single_word_query_and_reports_failures() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                Err(TransportFailureKind::Timeout),
                Err(TransportFailureKind::Connection),
            ])),
            requests: RefCell::default(),
            searches: RefCell::default(),
        };
        let load = load_symbol(&transport, &mut Cache::default(), "$EXM", 100);
        assert!(load.symbol.is_none());
        assert_eq!(transport.searches.borrow().len(), 2);
        assert!(
            load.diagnostics
                .last()
                .is_some_and(|diagnostic| diagnostic.contains("no usable symbol"))
        );
    }
}
