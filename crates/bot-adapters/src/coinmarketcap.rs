//! Typed CoinMarketCap Bitcoin-price adapter.

use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::Value;

const LISTINGS_URL: &str = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

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

    use super::{
        BitcoinPriceOutcome, BitcoinPriceRequest, CoinMarketCapTransport, HttpResponse,
        TransportFailureKind, fetch_bitcoin_price, parse_bitcoin_price,
    };

    struct Transport {
        result: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<BitcoinPriceRequest>>,
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
}
