//! Typed CriptoYa dollar-quote adapter.

use std::time::Duration;

use bot_core::devo::DevoQuotes;
use reqwest::blocking::Client;
use serde::Deserialize;

const DOLLAR_URL: &str = "https://criptoya.com/api/dolar";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

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
pub enum DollarQuotesOutcome {
    Quotes(DevoQuotes),
    Missing,
    HttpError { status_code: u16 },
    InvalidJson,
    TransportError(TransportFailureKind),
}

pub trait CriptoYaTransport {
    fn get_dollar_quotes(&self) -> Result<HttpResponse, TransportFailureKind>;
}

pub struct ReqwestCriptoYaTransport {
    client: Client,
}

impl ReqwestCriptoYaTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl CriptoYaTransport for ReqwestCriptoYaTransport {
    fn get_dollar_quotes(&self) -> Result<HttpResponse, TransportFailureKind> {
        let response = self.client.get(DOLLAR_URL).send().map_err(classify_error)?;
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Number {
    Float(f64),
    Text(String),
}

impl Number {
    fn finite(self) -> Option<f64> {
        let value = match self {
            Self::Float(value) => value,
            Self::Text(value) => value.parse().ok()?,
        };
        value.is_finite().then_some(value)
    }
}

#[derive(Debug, Deserialize)]
struct Price {
    price: Number,
}

#[derive(Debug, Deserialize)]
struct Usdt {
    ask: Number,
    bid: Number,
}

#[derive(Debug, Deserialize)]
struct Crypto {
    usdt: Usdt,
}

#[derive(Debug, Deserialize)]
struct DollarPayload {
    oficial: Price,
    tarjeta: Price,
    cripto: Crypto,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponsePayload {
    Direct(DollarPayload),
    Cached { data: DollarPayload },
}

impl ResponsePayload {
    fn into_payload(self) -> DollarPayload {
        match self {
            Self::Direct(payload) | Self::Cached { data: payload } => payload,
        }
    }
}

#[must_use]
pub fn parse_dollar_quotes(response: HttpResponse) -> DollarQuotesOutcome {
    if response.status_code >= 400 {
        return DollarQuotesOutcome::HttpError {
            status_code: response.status_code,
        };
    }
    let Ok(payload) = serde_json::from_str::<ResponsePayload>(&response.body) else {
        return DollarQuotesOutcome::InvalidJson;
    };
    let payload = payload.into_payload();
    let quotes = (
        payload.oficial.price.finite(),
        payload.tarjeta.price.finite(),
        payload.cripto.usdt.ask.finite(),
        payload.cripto.usdt.bid.finite(),
    );
    match quotes {
        (Some(official), Some(card), Some(usdt_ask), Some(usdt_bid)) => {
            DollarQuotesOutcome::Quotes(DevoQuotes {
                official,
                card,
                usdt_ask,
                usdt_bid,
            })
        }
        _ => DollarQuotesOutcome::Missing,
    }
}

#[must_use]
pub fn fetch_dollar_quotes<T: CriptoYaTransport>(transport: &T) -> DollarQuotesOutcome {
    match transport.get_dollar_quotes() {
        Ok(response) => parse_dollar_quotes(response),
        Err(error) => DollarQuotesOutcome::TransportError(error),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use bot_core::devo::DevoQuotes;

    use super::{
        CriptoYaTransport, DollarQuotesOutcome, HttpResponse, TransportFailureKind,
        fetch_dollar_quotes, parse_dollar_quotes,
    };

    struct Transport(RefCell<Option<Result<HttpResponse, TransportFailureKind>>>);

    impl CriptoYaTransport for Transport {
        fn get_dollar_quotes(&self) -> Result<HttpResponse, TransportFailureKind> {
            self.0
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[test]
    fn parses_direct_and_cached_numeric_quotes() {
        for body in [
            r#"{"oficial":{"price":100},"tarjeta":{"price":"150"},"cripto":{"usdt":{"ask":200,"bid":"190"}}}"#,
            r#"{"data":{"oficial":{"price":100},"tarjeta":{"price":"150"},"cripto":{"usdt":{"ask":200,"bid":"190"}}}}"#,
        ] {
            assert_eq!(
                parse_dollar_quotes(HttpResponse {
                    status_code: 200,
                    body: body.to_owned(),
                }),
                DollarQuotesOutcome::Quotes(DevoQuotes {
                    official: 100.0,
                    card: 150.0,
                    usdt_ask: 200.0,
                    usdt_bid: 190.0,
                })
            );
        }
    }

    #[test]
    fn classifies_http_json_missing_and_transport_failures() {
        assert_eq!(
            parse_dollar_quotes(HttpResponse {
                status_code: 503,
                body: String::new(),
            }),
            DollarQuotesOutcome::HttpError { status_code: 503 }
        );
        assert_eq!(
            parse_dollar_quotes(HttpResponse {
                status_code: 200,
                body: "bad".to_owned(),
            }),
            DollarQuotesOutcome::InvalidJson
        );
        assert_eq!(
            parse_dollar_quotes(HttpResponse {
                status_code: 200,
                body: r#"{"oficial":{"price":"NaN"},"tarjeta":{"price":150},"cripto":{"usdt":{"ask":200,"bid":190}}}"#.to_owned(),
            }),
            DollarQuotesOutcome::Missing
        );
        let transport = Transport(RefCell::new(Some(Err(TransportFailureKind::Timeout))));
        assert_eq!(
            fetch_dollar_quotes(&transport),
            DollarQuotesOutcome::TransportError(TransportFailureKind::Timeout)
        );
    }
}
