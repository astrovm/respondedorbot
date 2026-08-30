//! Typed Yahoo Finance chart adapter using the shared request cache.

use std::thread;
use std::time::Duration;

use bot_core::cache_policy::request_cache_key;
use bot_core::stocks::{StockQuote, parse_yahoo_quote};
use reqwest::blocking::Client;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::request_cache::{JsonHttpResponse, RequestCache, load_cached_json};

const CHART_URL: &str = "https://query1.finance.yahoo.com/v8/finance/chart";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const CACHE_TTL_SECONDS: i64 = 300;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YahooChartRequest {
    pub symbol: String,
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
    let hash = Sha256::digest(arguments.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    request_cache_key(&hash)
}

#[derive(Debug, Clone, PartialEq)]
pub struct YahooQuoteLoad {
    pub quote: Option<StockQuote>,
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use crate::request_cache::RequestCache;

    use super::{
        HttpResponse, TransportFailureKind, YahooChartRequest, YahooFinanceTransport, cache_key,
        load_quote,
    };

    struct Transport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<YahooChartRequest>>,
    }

    impl YahooFinanceTransport for Transport {
        fn chart(&self, request: &YahooChartRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
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
        };
        let load = load_quote(&transport, &mut Cache::default(), "CL=F", 100);
        assert!(load.quote.is_none());
        assert!(load.diagnostics[0].contains("no usable quote"));
    }
}
