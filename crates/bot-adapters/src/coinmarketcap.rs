//! Typed CoinMarketCap Bitcoin-price adapter.

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use bot_core::cache_policy::request_cache_history_key;
use bot_core::market_prices::{CryptoAsset, CryptoQuote};
use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde_json::Value;

use crate::request_cache::{
    JsonHttpResponse, RequestCache, load_cached_json, python_json_string, python_request_cache_key,
};

const LISTINGS_URL: &str = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest";
const QUOTES_URL: &str = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MARKET_CACHE_TTL_SECONDS: i64 = 300;
const MARKET_HISTORY_TTL_SECONDS: i64 = 3 * 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitcoinPriceRequest {
    pub api_key: String,
    pub currency: String,
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

#[derive(Debug, Clone, PartialEq)]
pub enum BitcoinPriceOutcome {
    Price(f64),
    Missing,
    HttpError { status_code: u16 },
    InvalidJson,
    TransportError(TransportFailureKind),
}

pub trait CoinMarketCapTransport {
    fn get(&self, request: &BitcoinPriceRequest) -> Result<HttpResponse, TransportFailureKind>;
}

pub struct ReqwestCoinMarketCapTransport {
    client: Client,
}

impl ReqwestCoinMarketCapTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl CoinMarketCapTransport for ReqwestCoinMarketCapTransport {
    fn get(&self, request: &BitcoinPriceRequest) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(LISTINGS_URL)
            .query(&[
                ("start", "1"),
                ("limit", "100"),
                ("convert", request.currency.as_str()),
            ])
            .header("Accepts", "application/json")
            .header("X-CMC_PRO_API_KEY", &request.api_key)
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarketRequestKind {
    Listings,
    Quotes {
        identifiers: Vec<String>,
        by_slug: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketRequest {
    pub api_key: String,
    pub currency: String,
    pub kind: MarketRequestKind,
}

pub trait CoinMarketCapMarketTransport {
    fn get_market(&self, request: &MarketRequest) -> Result<HttpResponse, TransportFailureKind>;

    fn before_retry(&self) {}
}

impl CoinMarketCapMarketTransport for ReqwestCoinMarketCapTransport {
    fn get_market(&self, request: &MarketRequest) -> Result<HttpResponse, TransportFailureKind> {
        let builder = match &request.kind {
            MarketRequestKind::Listings => self.client.get(LISTINGS_URL).query(&[
                ("start", "1"),
                ("limit", "100"),
                ("convert", request.currency.as_str()),
            ]),
            MarketRequestKind::Quotes {
                identifiers,
                by_slug,
            } => {
                let parameter = if *by_slug { "slug" } else { "symbol" };
                self.client.get(QUOTES_URL).query(&[
                    (parameter, identifiers.join(",")),
                    ("convert", request.currency.clone()),
                ])
            }
        };
        let response = builder
            .header("Accepts", "application/json")
            .header("X-CMC_PRO_API_KEY", &request.api_key)
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

#[derive(Debug, Clone, PartialEq)]
pub struct MarketAssetsLoad {
    pub assets: Option<Vec<CryptoAsset>>,
    pub diagnostics: Vec<String>,
}

fn market_cache_arguments(request: &MarketRequest) -> String {
    let (url, parameters) = match &request.kind {
        MarketRequestKind::Listings => (
            LISTINGS_URL,
            format!(
                "{{\"convert\": {}, \"limit\": \"100\", \"start\": \"1\"}}",
                python_json_string(&request.currency)
            ),
        ),
        MarketRequestKind::Quotes {
            identifiers,
            by_slug,
        } => {
            let key = if *by_slug { "slug" } else { "symbol" };
            (
                QUOTES_URL,
                format!(
                    "{{\"convert\": {}, \"{key}\": {}}}",
                    python_json_string(&request.currency),
                    python_json_string(&identifiers.join(","))
                ),
            )
        }
    };
    format!(
        "{{\"api_url\": \"{url}\", \"headers\": {{\"Accepts\": \"application/json\", \"X-CMC_PRO_API_KEY\": {}}}, \"parameters\": {parameters}}}",
        python_json_string(&request.api_key)
    )
}

fn market_cache_hash(arguments: &str) -> String {
    python_request_cache_key(arguments)
        .strip_prefix("request_cache:")
        .unwrap_or_default()
        .to_owned()
}

fn hour_key(timestamp: i64) -> Option<String> {
    DateTime::<Utc>::from_timestamp(timestamp, 0)
        .map(|value| value.format("%Y-%m-%d-%H").to_string())
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(value)) => value.as_f64(),
        Some(Value::String(value)) => value.parse().ok(),
        _ => None,
    }
    .filter(|value| value.is_finite())
}

fn text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        _ => String::new(),
    }
}

fn parse_asset(value: &Value) -> Option<CryptoAsset> {
    let object = value.as_object()?;
    let quotes = object
        .get("quote")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(currency, quote)| {
            let quote = quote.as_object()?;
            Some((
                currency.clone(),
                CryptoQuote {
                    price: number(quote.get("price"))?,
                    percent_change_1h: number(quote.get("percent_change_1h")),
                    percent_change_24h: number(quote.get("percent_change_24h")),
                    percent_change_7d: number(quote.get("percent_change_7d")),
                    percent_change_30d: number(quote.get("percent_change_30d")),
                },
            ))
        })
        .collect::<HashMap<_, _>>();
    Some(CryptoAsset {
        id: text(object.get("id")),
        symbol: text(object.get("symbol")),
        name: text(object.get("name")),
        slug: text(object.get("slug")),
        quotes,
    })
}

fn parse_market_assets(payload: &Value, kind: &MarketRequestKind) -> Option<Vec<CryptoAsset>> {
    let data = payload.get("data")?;
    let values = match kind {
        MarketRequestKind::Listings => data.as_array()?.iter().collect::<Vec<_>>(),
        MarketRequestKind::Quotes { .. } => data
            .as_object()?
            .values()
            .flat_map(|value| {
                value
                    .as_array()
                    .map_or_else(|| vec![value], |values| values.iter().collect())
            })
            .collect(),
    };
    Some(values.into_iter().filter_map(parse_asset).collect())
}

pub fn load_market_assets<T: CoinMarketCapMarketTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    request: &MarketRequest,
    now_unix: i64,
) -> MarketAssetsLoad {
    if matches!(&request.kind, MarketRequestKind::Quotes { identifiers, .. } if identifiers.is_empty())
    {
        return MarketAssetsLoad {
            assets: Some(Vec::new()),
            diagnostics: Vec::new(),
        };
    }
    let key = python_request_cache_key(&market_cache_arguments(request));
    let load = load_cached_json(
        cache,
        &key,
        MARKET_CACHE_TTL_SECONDS,
        now_unix,
        "CoinMarketCap market request",
        || {
            transport
                .get_market(request)
                .map(|response| JsonHttpResponse {
                    status_code: response.status_code,
                    body: response.body,
                })
                .map_err(|error| format!("{error:?}"))
        },
        || transport.before_retry(),
    );
    let assets = load
        .data
        .as_ref()
        .and_then(|payload| parse_market_assets(payload, &request.kind));
    let mut diagnostics = load.diagnostics;
    if load.data.is_some() && assets.is_none() {
        diagnostics.push("CoinMarketCap payload has invalid data".to_owned());
    }
    MarketAssetsLoad {
        assets,
        diagnostics,
    }
}

/// Refresh one listings cache and, only after a provider fetch, persist the
/// language-neutral hourly snapshot used by timeframe comparisons.
pub fn refresh_market_snapshot<T: CoinMarketCapMarketTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    api_key: &str,
    currency: &str,
    now_unix: i64,
) -> Vec<String> {
    let request = MarketRequest {
        api_key: api_key.to_owned(),
        currency: currency.to_owned(),
        kind: MarketRequestKind::Listings,
    };
    let arguments = market_cache_arguments(&request);
    let key = python_request_cache_key(&arguments);
    let load = load_cached_json(
        cache,
        &key,
        MARKET_CACHE_TTL_SECONDS,
        now_unix,
        "CoinMarketCap market refresh",
        || {
            transport
                .get_market(&request)
                .map(|response| JsonHttpResponse {
                    status_code: response.status_code,
                    body: response.body,
                })
                .map_err(|error| format!("{error:?}"))
        },
        || transport.before_retry(),
    );
    let mut diagnostics = load.diagnostics;
    if !load.refreshed {
        return diagnostics;
    }
    let Some(data) = load.data else {
        return diagnostics;
    };
    let Some(hour) = hour_key(now_unix) else {
        diagnostics
            .push("CoinMarketCap refresh timestamp is outside the supported range".to_owned());
        return diagnostics;
    };
    let history_key = request_cache_history_key(&hour, &market_cache_hash(&arguments));
    match cache.get(&history_key) {
        Ok(Some(_)) => {}
        Ok(None) => {
            let value = serde_json::json!({"timestamp": now_unix, "data": data}).to_string();
            if let Err(error) = cache.set(&history_key, &value, MARKET_HISTORY_TTL_SECONDS) {
                diagnostics.push(format!(
                    "could not write CoinMarketCap history key {history_key}: {error}"
                ));
            }
        }
        Err(error) => diagnostics.push(format!(
            "could not read CoinMarketCap history key {history_key}: {error}"
        )),
    }
    diagnostics
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

#[must_use]
pub fn parse_bitcoin_price(response: HttpResponse, currency: &str) -> BitcoinPriceOutcome {
    if response.status_code >= 400 {
        return BitcoinPriceOutcome::HttpError {
            status_code: response.status_code,
        };
    }
    let Ok(payload) = serde_json::from_str::<Value>(&response.body) else {
        return BitcoinPriceOutcome::InvalidJson;
    };
    let price = payload
        .get("data")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("quote"))
        .and_then(|quote| quote.get(currency))
        .and_then(|quote| quote.get("price"));
    let price = match price {
        Some(Value::Number(value)) => value.as_f64(),
        Some(Value::String(value)) => value.parse().ok(),
        Some(_) | None => None,
    };
    price
        .filter(|price| price.is_finite())
        .map_or(BitcoinPriceOutcome::Missing, BitcoinPriceOutcome::Price)
}

#[must_use]
pub fn fetch_bitcoin_price<T: CoinMarketCapTransport>(
    transport: &T,
    api_key: &str,
    currency: &str,
) -> BitcoinPriceOutcome {
    let request = BitcoinPriceRequest {
        api_key: api_key.to_owned(),
        currency: currency.to_owned(),
    };
    match transport.get(&request) {
        Ok(response) => parse_bitcoin_price(response, currency),
        Err(error) => BitcoinPriceOutcome::TransportError(error),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use crate::request_cache::RequestCache;

    use super::{
        BitcoinPriceOutcome, BitcoinPriceRequest, CoinMarketCapMarketTransport,
        CoinMarketCapTransport, HttpResponse, MarketRequest, MarketRequestKind,
        TransportFailureKind, fetch_bitcoin_price, load_market_assets, market_cache_arguments,
        parse_bitcoin_price, refresh_market_snapshot,
    };

    struct Transport {
        result: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<BitcoinPriceRequest>>,
    }

    struct MarketTransport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<MarketRequest>>,
    }

    impl CoinMarketCapMarketTransport for MarketTransport {
        fn get_market(
            &self,
            request: &MarketRequest,
        ) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[derive(Default)]
    struct Cache {
        reads: VecDeque<Result<Option<String>, &'static str>>,
        writes: Vec<(String, String, i64)>,
    }

    impl RequestCache for Cache {
        type Error = &'static str;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            self.reads.pop_front().unwrap_or(Ok(None))
        }

        fn set(&mut self, key: &str, value: &str, ttl: i64) -> Result<(), Self::Error> {
            self.writes.push((key.to_owned(), value.to_owned(), ttl));
            Ok(())
        }
    }

    impl CoinMarketCapTransport for Transport {
        fn get(&self, request: &BitcoinPriceRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.result
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[test]
    fn parses_numeric_and_string_prices_and_missing_payloads() {
        for (body, expected) in [
            (
                r#"{"data":[{"quote":{"USD":{"price":50000.5}}}]}"#,
                BitcoinPriceOutcome::Price(50_000.5),
            ),
            (
                r#"{"data":[{"quote":{"USD":{"price":"42.25"}}}]}"#,
                BitcoinPriceOutcome::Price(42.25),
            ),
            (r#"{"data":[]}"#, BitcoinPriceOutcome::Missing),
            (r#"{"data":[{"quote":{}}]}"#, BitcoinPriceOutcome::Missing),
        ] {
            assert_eq!(
                parse_bitcoin_price(
                    HttpResponse {
                        status_code: 200,
                        body: body.to_owned(),
                    },
                    "USD",
                ),
                expected
            );
        }
    }

    #[test]
    fn classifies_http_json_and_transport_failures_without_exposing_credentials() {
        assert_eq!(
            parse_bitcoin_price(
                HttpResponse {
                    status_code: 429,
                    body: "ignored".to_owned(),
                },
                "USD",
            ),
            BitcoinPriceOutcome::HttpError { status_code: 429 }
        );
        assert_eq!(
            parse_bitcoin_price(
                HttpResponse {
                    status_code: 200,
                    body: "bad".to_owned(),
                },
                "USD",
            ),
            BitcoinPriceOutcome::InvalidJson
        );
        let transport = Transport {
            result: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            fetch_bitcoin_price(&transport, "synthetic-secret", "ARS"),
            BitcoinPriceOutcome::TransportError(TransportFailureKind::Timeout)
        );
        assert_eq!(
            transport.requests.borrow().as_slice(),
            &[BitcoinPriceRequest {
                api_key: "synthetic-secret".to_owned(),
                currency: "ARS".to_owned(),
            }]
        );
    }

    #[test]
    fn market_requests_match_python_cache_identity_and_parse_typed_rows() {
        let request = MarketRequest {
            api_key: "synthetic-secret".to_owned(),
            currency: "USD".to_owned(),
            kind: MarketRequestKind::Quotes {
                identifiers: vec!["BTC".to_owned(), "ETH".to_owned()],
                by_slug: false,
            },
        };
        assert_eq!(
            market_cache_arguments(&request),
            "{\"api_url\": \"https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest\", \"headers\": {\"Accepts\": \"application/json\", \"X-CMC_PRO_API_KEY\": \"synthetic-secret\"}, \"parameters\": {\"convert\": \"USD\", \"symbol\": \"BTC,ETH\"}}"
        );
        let transport = MarketTransport {
            responses: RefCell::new(VecDeque::from([Ok(HttpResponse {
                status_code: 200,
                body: r#"{"data":{"BTC":[{"id":1,"symbol":"BTC","name":"Bitcoin","slug":"bitcoin","quote":{"USD":{"price":"50000.5","percent_change_24h":2.5}}}]}}"#.to_owned(),
            })])),
            requests: RefCell::new(Vec::new()),
        };
        let mut cache = Cache::default();
        let load = load_market_assets(&transport, &mut cache, &request, 100);
        let Some(assets) = load.assets else {
            return;
        };
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0].id, "1");
        assert_eq!(assets[0].quotes["USD"].price, 50_000.5);
        assert!(load.diagnostics.is_empty());
        assert_eq!(transport.requests.borrow().as_slice(), &[request]);
        assert_eq!(cache.writes.len(), 1);
        assert_eq!(cache.writes[0].2, 300);
    }

    #[test]
    fn market_load_uses_stale_cache_after_provider_failures() {
        let request = MarketRequest {
            api_key: "key".to_owned(),
            currency: "USD".to_owned(),
            kind: MarketRequestKind::Listings,
        };
        let transport = MarketTransport {
            responses: RefCell::new(VecDeque::from([
                Err(TransportFailureKind::Timeout),
                Err(TransportFailureKind::Connection),
            ])),
            requests: RefCell::new(Vec::new()),
        };
        let mut cache = Cache {
            reads: VecDeque::from([Ok(Some(
                r#"{"timestamp":1,"data":{"data":[{"id":1,"symbol":"BTC","name":"Bitcoin","slug":"bitcoin","quote":{"USD":{"price":42}}}]}}"#.to_owned(),
            ))]),
            ..Cache::default()
        };
        let load = load_market_assets(&transport, &mut cache, &request, 1_000);
        let Some(assets) = load.assets else {
            return;
        };
        assert_eq!(assets[0].symbol, "BTC");
        assert_eq!(load.diagnostics.len(), 2);
    }

    #[test]
    fn refresh_writes_python_compatible_hourly_snapshot_only_after_fetch() {
        let response = Ok(HttpResponse {
            status_code: 200,
            body: r#"{"data":[{"id":1,"symbol":"BTC","name":"Bitcoin","slug":"bitcoin","quote":{"ARS":{"price":42}}}]}"#
                .to_owned(),
        });
        let transport = MarketTransport {
            responses: RefCell::new(VecDeque::from([response])),
            requests: RefCell::new(Vec::new()),
        };
        let mut cache = Cache {
            reads: VecDeque::from([Ok(None), Ok(None)]),
            ..Cache::default()
        };
        let diagnostics = refresh_market_snapshot(
            &transport,
            &mut cache,
            "synthetic-secret",
            "ARS",
            1_725_000_000,
        );
        assert!(diagnostics.is_empty());
        assert_eq!(cache.writes.len(), 2);
        assert_eq!(cache.writes[0].2, 300);
        assert!(
            cache.writes[1]
                .0
                .starts_with("request_cache_history:2024-08-30-06:")
        );
        assert_eq!(cache.writes[1].2, 3 * 24 * 60 * 60);

        let transport = MarketTransport {
            responses: RefCell::new(VecDeque::new()),
            requests: RefCell::new(Vec::new()),
        };
        let mut cache = Cache {
            reads: VecDeque::from([Ok(Some(
                r#"{"timestamp":1725000000,"data":{"data":[]}}"#.to_owned(),
            ))]),
            ..Cache::default()
        };
        let diagnostics = refresh_market_snapshot(
            &transport,
            &mut cache,
            "synthetic-secret",
            "ARS",
            1_725_000_001,
        );
        assert!(diagnostics.is_empty());
        assert!(cache.writes.is_empty());
    }
}
