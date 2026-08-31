//! Typed CriptoYa dollar-market adapter with Python-compatible Redis caches.

use std::thread;
use std::time::Duration;

use bot_core::cache_policy::{CacheDecision, evaluate_cache, request_cache_history_key};
use bot_core::dollar::{CurrencyBands, DollarRate, render_dollar_rates};
use bot_core::locale::Locale;
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};
use crate::request_cache::{
    JsonHttpResponse, RequestCache, load_cached_json, python_request_cache_key,
};

const DOLLAR_URL: &str = "https://criptoya.com/api/dolar";
const DOLLAR_REQUEST_ARGUMENTS: &str = concat!(
    "{\"api_url\": \"https://criptoya.com/api/dolar\", ",
    "\"headers\": null, \"parameters\": null}"
);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TTL_SECONDS: i64 = 300;
const HISTORY_TTL_SECONDS: i64 = 3 * 24 * 60 * 60;
const FORMATTED_TTL_SECONDS: i64 = 300;
const FORMATTED_STALE_GRACE_SECONDS: i64 = 30 * 60;

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

pub trait DollarTransport {
    fn get(&self) -> Result<HttpResponse, TransportFailureKind>;

    fn before_retry(&self) {}
}

pub trait DollarCache: RequestCache {
    fn set_if_absent(
        &mut self,
        key: &str,
        value: &str,
        ttl_seconds: i64,
    ) -> Result<bool, Self::Error>;
}

impl DollarCache for RedisJsonCache {
    fn set_if_absent(
        &mut self,
        key: &str,
        value: &str,
        ttl_seconds: i64,
    ) -> Result<bool, RedisJsonCacheError> {
        RedisJsonCache::set_if_absent(self, key, value, ttl_seconds)
    }
}

pub struct ReqwestDollarTransport {
    client: Client,
}

impl ReqwestDollarTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl DollarTransport for ReqwestDollarTransport {
    fn get(&self) -> Result<HttpResponse, TransportFailureKind> {
        let response = self.client.get(DOLLAR_URL).send().map_err(classify_error)?;
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

#[derive(Debug, Clone, PartialEq)]
pub struct DollarMarketLoad {
    pub text: Option<String>,
    pub diagnostics: Vec<String>,
}

/// Refresh the raw dollar response and persist the same hourly snapshot used
/// by the Python cache service. Provider failures remain diagnostic-only so
/// the other independent market refreshes can still run.
pub fn refresh_dollar_snapshot<T: DollarTransport, C: DollarCache>(
    transport: &T,
    cache: &mut C,
    now_unix: i64,
) -> Vec<String> {
    let load = load_cached_json(
        cache,
        &request_key(),
        REQUEST_TTL_SECONDS,
        now_unix,
        "CriptoYa dollar refresh",
        || {
            transport
                .get()
                .map(|response| JsonHttpResponse {
                    status_code: response.status_code,
                    body: response.body,
                })
                .map_err(|error| format!("transport {error:?}"))
        },
        || transport.before_retry(),
    );
    let mut diagnostics = load.diagnostics;
    if let (true, Some(current)) = (load.refreshed, load.data) {
        let hourly_value = json!({"timestamp": now_unix, "data": current}).to_string();
        let current_history_key = request_cache_history_key(&hour_key(now_unix), request_hash());
        if let Err(error) =
            cache.set_if_absent(&current_history_key, &hourly_value, HISTORY_TTL_SECONDS)
        {
            diagnostics.push(format!(
                "could not write dollar history key {current_history_key}: {error}"
            ));
        }
    }
    diagnostics
}

fn request_hash() -> &'static str {
    "e566a0454c06bb8e67fb679a5d285cecc3edac67b888df87021dcfa94971b49b"
}

fn request_key() -> String {
    python_request_cache_key(DOLLAR_REQUEST_ARGUMENTS)
}

fn div_floor(value: i64, divisor: i64) -> i64 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder < 0 {
        quotient - 1
    } else {
        quotient
    }
}

fn civil_date(days_since_epoch: i64) -> (i64, i64, i64) {
    let days = days_since_epoch + 719_468;
    let era = div_floor(days, 146_097);
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn hour_key(timestamp: i64) -> String {
    let hours = div_floor(timestamp, 3_600);
    let hour = hours.rem_euclid(24);
    let (year, month, day) = civil_date(div_floor(timestamp, 86_400));
    format!("{year:04}-{month:02}-{day:02}-{hour:02}")
}

fn cached_value<C: DollarCache>(
    cache: &mut C,
    key: &str,
    diagnostics: &mut Vec<String>,
) -> Option<Value> {
    match cache.get(key) {
        Ok(Some(raw)) => match serde_json::from_str(&raw) {
            Ok(value) => Some(value),
            Err(error) => {
                diagnostics.push(format!("invalid dollar cache key {key}: {error}"));
                None
            }
        },
        Ok(None) => None,
        Err(error) => {
            diagnostics.push(format!("could not read dollar cache key {key}: {error}"));
            None
        }
    }
}

fn number(value: Option<&Value>) -> Option<f64> {
    let value = match value? {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }?;
    value.is_finite().then_some(value)
}

fn path_number(value: &Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    number(Some(current))
}

fn change(current: f64, daily: Option<f64>, history: Option<f64>, hours_ago: i64) -> Option<f64> {
    if hours_ago == 24 {
        return daily;
    }
    history
        .filter(|historical| *historical != 0.0)
        .map(|historical| ((current - historical) / historical) * 100.0)
}

fn parse_rates(current: &Value, history: Option<&Value>, hours_ago: i64) -> Vec<DollarRate> {
    const SPECS: [(&str, &[&str], &[&str]); 9] = [
        (
            "Mayorista",
            &["mayorista", "price"],
            &["mayorista", "variation"],
        ),
        ("Oficial", &["oficial", "price"], &["oficial", "variation"]),
        ("Tarjeta", &["tarjeta", "price"], &["tarjeta", "variation"]),
        (
            "MEP",
            &["mep", "al30", "ci", "price"],
            &["mep", "al30", "ci", "variation"],
        ),
        (
            "CCL",
            &["ccl", "al30", "ci", "price"],
            &["ccl", "al30", "ci", "variation"],
        ),
        ("Blue", &["blue", "ask"], &["blue", "variation"]),
        (
            "Bitcoin",
            &["cripto", "ccb", "ask"],
            &["cripto", "ccb", "variation"],
        ),
        (
            "USDC",
            &["cripto", "usdc", "ask"],
            &["cripto", "usdc", "variation"],
        ),
        (
            "USDT",
            &["cripto", "usdt", "ask"],
            &["cripto", "usdt", "variation"],
        ),
    ];
    SPECS
        .iter()
        .filter_map(|(name, price_path, variation_path)| {
            let price = path_number(current, price_path)?;
            Some(DollarRate {
                name,
                price,
                change: change(
                    price,
                    path_number(current, variation_path),
                    history.and_then(|history| path_number(history, price_path)),
                    hours_ago,
                ),
            })
        })
        .collect()
}

fn unwrap_data(value: &Value) -> &Value {
    value.get("data").unwrap_or(value)
}

fn tcrm<C: DollarCache>(
    cache: &mut C,
    hours_ago: i64,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Option<DollarRate> {
    let current = cached_value(cache, "tcrm_100", diagnostics)
        .as_ref()
        .and_then(|value| number(value.get("data")))?;
    let history_key = request_cache_history_key(
        &hour_key(now_unix.saturating_sub(hours_ago.saturating_mul(3_600))),
        "tcrm_100",
    );
    let historical = cached_value(cache, &history_key, diagnostics)
        .as_ref()
        .and_then(|value| number(value.get("data")));
    Some(DollarRate {
        name: "TCRM 100",
        price: current,
        change: historical
            .filter(|value| *value != 0.0)
            .map(|value| ((current / value) - 1.0) * 100.0),
    })
}

fn currency_bands<C: DollarCache>(
    cache: &mut C,
    diagnostics: &mut Vec<String>,
) -> Option<CurrencyBands> {
    let cached = cached_value(cache, "bcra_currency_band_limits", diagnostics)?;
    let data = unwrap_data(&cached);
    Some(CurrencyBands {
        lower: number(data.get("lower"))?,
        upper: number(data.get("upper"))?,
        lower_change: number(data.get("lower_change_pct")),
        upper_change: number(data.get("upper_change_pct")),
    })
}

fn formatted_snapshot<C: DollarCache>(
    cache: &mut C,
    hours_ago: i64,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Option<String> {
    let key = format!("market:dolar:formatted:{hours_ago}");
    let cached = cached_value(cache, &key, diagnostics)?;
    let timestamp = cached.get("timestamp").and_then(Value::as_i64)?;
    if matches!(
        evaluate_cache(
            Some(timestamp),
            now_unix,
            FORMATTED_TTL_SECONDS,
            FORMATTED_STALE_GRACE_SECONDS,
        ),
        CacheDecision::Fresh | CacheDecision::ServeStale
    ) {
        cached
            .get("value")
            .and_then(Value::as_str)
            .map(str::to_owned)
    } else {
        None
    }
}

fn store_formatted<C: DollarCache>(
    cache: &mut C,
    hours_ago: i64,
    now_unix: i64,
    text: &str,
    diagnostics: &mut Vec<String>,
) {
    let key = format!("market:dolar:formatted:{hours_ago}");
    let value = json!({"timestamp": now_unix, "value": text}).to_string();
    if let Err(error) = cache.set(
        &key,
        &value,
        FORMATTED_TTL_SECONDS + FORMATTED_STALE_GRACE_SECONDS,
    ) {
        diagnostics.push(format!(
            "could not write formatted dollar cache key {key}: {error}"
        ));
    }
}

#[must_use]
pub fn load_dollar_market<T: DollarTransport, C: DollarCache>(
    transport: &T,
    cache: &mut C,
    hours_ago: i64,
    locale: Locale,
    now_unix: i64,
) -> DollarMarketLoad {
    let mut diagnostics = Vec::new();
    if let Some(text) = formatted_snapshot(cache, hours_ago, now_unix, &mut diagnostics) {
        return DollarMarketLoad {
            text: Some(text),
            diagnostics,
        };
    }
    let history_key = request_cache_history_key(
        &hour_key(now_unix.saturating_sub(hours_ago.saturating_mul(3_600))),
        request_hash(),
    );
    let history = (hours_ago != 24)
        .then(|| cached_value(cache, &history_key, &mut diagnostics))
        .flatten();
    let load = load_cached_json(
        cache,
        &request_key(),
        REQUEST_TTL_SECONDS,
        now_unix,
        "CriptoYa dollar request",
        || {
            transport
                .get()
                .map(|response| JsonHttpResponse {
                    status_code: response.status_code,
                    body: response.body,
                })
                .map_err(|error| format!("transport {error:?}"))
        },
        || transport.before_retry(),
    );
    diagnostics.extend(load.diagnostics);
    let refreshed = load.refreshed;
    let Some(current) = load.data else {
        return DollarMarketLoad {
            text: None,
            diagnostics,
        };
    };
    if refreshed {
        let hourly_value = json!({"timestamp": now_unix, "data": current}).to_string();
        let current_history_key = request_cache_history_key(&hour_key(now_unix), request_hash());
        if let Err(error) =
            cache.set_if_absent(&current_history_key, &hourly_value, HISTORY_TTL_SECONDS)
        {
            diagnostics.push(format!(
                "could not write dollar history key {current_history_key}: {error}"
            ));
        }
    }
    let mut rates = parse_rates(&current, history.as_ref().map(unwrap_data), hours_ago);
    if let Some(tcrm) = tcrm(cache, hours_ago, now_unix, &mut diagnostics) {
        rates.push(tcrm);
    }
    let bands = currency_bands(cache, &mut diagnostics);
    let text = render_dollar_rates(&rates, bands.as_ref(), hours_ago, locale);
    if let Some(text) = &text {
        store_formatted(cache, hours_ago, now_unix, text, &mut diagnostics);
    }
    DollarMarketLoad { text, diagnostics }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};

    use super::{
        DollarCache, DollarTransport, HISTORY_TTL_SECONDS, HttpResponse, TransportFailureKind,
        hour_key, load_dollar_market, refresh_dollar_snapshot, request_key,
    };
    use crate::request_cache::RequestCache;
    use bot_core::locale::Locale;

    #[derive(Default)]
    struct Cache {
        values: HashMap<String, String>,
        writes: Vec<(String, String, i64, bool)>,
    }

    impl RequestCache for Cache {
        type Error = &'static str;

        fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
            Ok(self.values.get(key).cloned())
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            self.values.insert(key.to_owned(), value.to_owned());
            self.writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds, false));
            Ok(())
        }
    }

    impl DollarCache for Cache {
        fn set_if_absent(
            &mut self,
            key: &str,
            value: &str,
            ttl_seconds: i64,
        ) -> Result<bool, Self::Error> {
            let inserted = !self.values.contains_key(key);
            if inserted {
                self.values.insert(key.to_owned(), value.to_owned());
            }
            self.writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds, true));
            Ok(inserted)
        }
    }

    struct Transport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        calls: RefCell<usize>,
    }

    impl DollarTransport for Transport {
        fn get(&self) -> Result<HttpResponse, TransportFailureKind> {
            *self.calls.borrow_mut() += 1;
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn body() -> String {
        serde_json::json!({
            "mayorista":{"price":1400,"variation":1},
            "oficial":{"price":1420,"variation":2},
            "tarjeta":{"price":1988,"variation":3},
            "mep":{"al30":{"ci":{"price":1450,"variation":4}}},
            "ccl":{"al30":{"ci":{"price":1460,"variation":5}}},
            "blue":{"ask":1430,"variation":6},
            "cripto":{
                "ccb":{"ask":1470,"variation":7},
                "usdc":{"ask":1480,"variation":8},
                "usdt":{"ask":1490,"variation":9}
            }
        })
        .to_string()
    }

    #[test]
    fn uses_exact_cache_key_and_renders_all_daily_sources() {
        assert_eq!(
            request_key(),
            "request_cache:e566a0454c06bb8e67fb679a5d285cecc3edac67b888df87021dcfa94971b49b"
        );
        assert_eq!(hour_key(1_725_000_000), "2024-08-30-06");
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([Ok(HttpResponse {
                status_code: 200,
                body: body(),
            })])),
            calls: RefCell::new(0),
        };
        let mut cache = Cache::default();
        cache.values.insert(
            "tcrm_100".to_owned(),
            r#"{"timestamp":1725000000,"data":1410}"#.to_owned(),
        );
        cache.values.insert(
            "bcra_currency_band_limits".to_owned(),
            r#"{"data":{"lower":950,"upper":1550,"lower_change_pct":0.1,"upper_change_pct":0.2}}"#
                .to_owned(),
        );
        let load = load_dollar_market(&transport, &mut cache, 24, Locale::Es, 1_725_000_000);
        let text = load.text.unwrap_or_default();
        for expected in [
            "Mayorista: 1400 (+1% 24hs)",
            "Oficial: 1420 (+2% 24hs)",
            "Tarjeta: 1988 (+3% 24hs)",
            "MEP: 1450 (+4% 24hs)",
            "CCL: 1460 (+5% 24hs)",
            "Blue: 1430 (+6% 24hs)",
            "Bitcoin: 1470 (+7% 24hs)",
            "USDC: 1480 (+8% 24hs)",
            "USDT: 1490 (+9% 24hs)",
            "TCRM 100: 1410",
            "Banda piso: 950 (+0.1% 24hs)",
        ] {
            assert!(text.contains(expected), "missing {expected} in {text}");
        }
        assert!(
            cache
                .writes
                .iter()
                .any(|write| write.0.starts_with("request_cache_history:"))
        );
        assert!(
            cache
                .writes
                .iter()
                .any(|write| write.0 == "market:dolar:formatted:24")
        );
    }

    #[test]
    fn hourly_history_computes_changes_and_stale_formatted_cache_skips_http() {
        let now = 1_725_000_000;
        let mut cache = Cache::default();
        let history_key = format!(
            "request_cache_history:{}:e566a0454c06bb8e67fb679a5d285cecc3edac67b888df87021dcfa94971b49b",
            hour_key(now - 6 * 3_600)
        );
        cache.values.insert(
            history_key,
            format!(r#"{{"timestamp":{},"data":{}}}"#, now - 6 * 3_600, body()),
        );
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([Ok(HttpResponse {
                status_code: 200,
                body: body().replace("1400", "1500"),
            })])),
            calls: RefCell::new(0),
        };
        let first = load_dollar_market(&transport, &mut cache, 6, Locale::En, now);
        assert!(
            first
                .text
                .unwrap_or_default()
                .contains("Mayorista: 1500 (+7.14% 6hs)")
        );
        let second = load_dollar_market(&transport, &mut cache, 6, Locale::Es, now + 301);
        assert_eq!(*transport.calls.borrow(), 1);
        assert_eq!(
            second.text,
            cache
                .values
                .get("market:dolar:formatted:6")
                .and_then(|raw| {
                    serde_json::from_str::<serde_json::Value>(raw)
                        .ok()?
                        .get("value")?
                        .as_str()
                        .map(str::to_owned)
                })
        );
    }

    #[test]
    fn provider_failure_returns_no_text_with_diagnostics() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                Err(TransportFailureKind::Timeout),
                Err(TransportFailureKind::Connection),
            ])),
            calls: RefCell::new(0),
        };
        let load = load_dollar_market(&transport, &mut Cache::default(), 24, Locale::Es, 100);
        assert!(load.text.is_none());
        assert_eq!(*transport.calls.borrow(), 2);
        assert!(load.diagnostics.iter().any(|item| item.contains("Timeout")));
        assert!(
            load.diagnostics
                .iter()
                .any(|item| item.contains("Connection"))
        );
    }

    #[test]
    fn dedicated_refresh_writes_history_only_for_a_new_provider_response() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([Ok(HttpResponse {
                status_code: 200,
                body: body(),
            })])),
            calls: RefCell::new(0),
        };
        let mut cache = Cache::default();
        let diagnostics = refresh_dollar_snapshot(&transport, &mut cache, 1_725_000_000);
        assert!(diagnostics.is_empty());
        assert_eq!(*transport.calls.borrow(), 1);
        assert!(cache.writes.iter().any(|write| {
            write.0.starts_with("request_cache_history:2024-08-30-06:")
                && write.2 == HISTORY_TTL_SECONDS
                && write.3
        }));

        let transport = Transport {
            responses: RefCell::new(VecDeque::new()),
            calls: RefCell::new(0),
        };
        let mut cache = Cache::default();
        cache.values.insert(
            request_key(),
            format!(r#"{{"timestamp":1725000000,"data":{}}}"#, body()),
        );
        let diagnostics = refresh_dollar_snapshot(&transport, &mut cache, 1_725_000_001);
        assert!(diagnostics.is_empty());
        assert_eq!(*transport.calls.borrow(), 0);
        assert!(cache.writes.is_empty());
    }
}
