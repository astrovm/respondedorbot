//! Typed CriptoYa dollar-quote adapter.

use std::time::Duration;

use bot_core::devo::DevoQuotes;
use bot_core::rulo::{ExchangeQuote, RuloInput};
use reqwest::blocking::Client;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

const DOLLAR_URL: &str = "https://criptoya.com/api/dolar";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const RULO_USD_AMOUNT: f64 = 1000.0;

#[derive(Debug, Clone, PartialEq)]
pub enum CriptoYaRequest {
    Dollar,
    Exchange {
        asset: String,
        fiat: String,
        amount: f64,
    },
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
pub enum DollarQuotesOutcome {
    Quotes(DevoQuotes),
    Missing,
    HttpError { status_code: u16 },
    InvalidJson,
    TransportError(TransportFailureKind),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuloMarketOutcome {
    Input(RuloInput),
    InvalidJson,
    HttpError { status_code: u16 },
    TransportError(TransportFailureKind),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExchangeQuotesOutcome {
    Quotes(Vec<ExchangeQuote>),
    InvalidJson,
    HttpError { status_code: u16 },
    TransportError(TransportFailureKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeSide {
    Ask,
    Bid,
}

pub trait CriptoYaTransport {
    fn get(&self, request: &CriptoYaRequest) -> Result<HttpResponse, TransportFailureKind>;
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
    fn get(&self, request: &CriptoYaRequest) -> Result<HttpResponse, TransportFailureKind> {
        let url = match request {
            CriptoYaRequest::Dollar => DOLLAR_URL.to_owned(),
            CriptoYaRequest::Exchange {
                asset,
                fiat,
                amount,
            } => format!("https://criptoya.com/api/{asset}/{fiat}/{amount:.0}"),
        };
        let response = self.client.get(url).send().map_err(classify_error)?;
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
    match transport.get(&CriptoYaRequest::Dollar) {
        Ok(response) => parse_dollar_quotes(response),
        Err(error) => DollarQuotesOutcome::TransportError(error),
    }
}

#[derive(Debug, Deserialize)]
struct OptionalPrice {
    price: Option<Number>,
}

#[derive(Debug, Deserialize)]
struct BluePrice {
    bid: Option<Number>,
    price: Option<Number>,
}

#[derive(Debug, Deserialize)]
struct MepSettlement {
    price: Option<Number>,
}

#[derive(Debug, Deserialize)]
struct MepBond {
    ci: Option<MepSettlement>,
}

#[derive(Debug, Deserialize)]
struct MepMarket {
    al30: Option<MepBond>,
}

#[derive(Debug, Deserialize)]
struct RuloDollarPayload {
    oficial: Option<OptionalPrice>,
    mep: Option<MepMarket>,
    blue: Option<BluePrice>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RuloResponsePayload {
    Cached { data: RuloDollarPayload },
    Direct(RuloDollarPayload),
}

fn optional_number(value: Option<Number>) -> Option<f64> {
    value.and_then(Number::finite)
}

#[must_use]
pub fn parse_rulo_market(response: HttpResponse) -> RuloMarketOutcome {
    if response.status_code >= 400 {
        return RuloMarketOutcome::HttpError {
            status_code: response.status_code,
        };
    }
    let Ok(payload) = serde_json::from_str::<RuloResponsePayload>(&response.body) else {
        return RuloMarketOutcome::InvalidJson;
    };
    let payload = match payload {
        RuloResponsePayload::Direct(payload) | RuloResponsePayload::Cached { data: payload } => {
            payload
        }
    };
    let official = payload
        .oficial
        .and_then(|value| optional_number(value.price));
    let mep = payload
        .mep
        .and_then(|value| value.al30)
        .and_then(|value| value.ci)
        .and_then(|value| optional_number(value.price));
    let blue = payload
        .blue
        .and_then(|value| preferred_number(value.bid, value.price));
    RuloMarketOutcome::Input(RuloInput {
        official,
        mep,
        blue,
        usd_to_usdt: Vec::new(),
        usdt_to_ars: Vec::new(),
        usd_amount: RULO_USD_AMOUNT,
    })
}

#[must_use]
pub fn fetch_rulo_market<T: CriptoYaTransport>(transport: &T) -> RuloMarketOutcome {
    match transport.get(&CriptoYaRequest::Dollar) {
        Ok(response) => parse_rulo_market(response),
        Err(error) => RuloMarketOutcome::TransportError(error),
    }
}

#[derive(Debug, Deserialize)]
struct RawExchangeQuote {
    #[serde(rename = "totalAsk")]
    total_ask: Option<Number>,
    ask: Option<Number>,
    #[serde(rename = "totalBid")]
    total_bid: Option<Number>,
    bid: Option<Number>,
}

struct ExchangeBook(Vec<(String, RawExchangeQuote)>);

impl<'de> Deserialize<'de> for ExchangeBook {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BookVisitor;

        impl<'de> Visitor<'de> for BookVisitor {
            type Value = ExchangeBook;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an exchange quote object")
            }

            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut quotes = Vec::new();
                while let Some((exchange, quote)) = access.next_entry::<String, ExchangeEntry>()? {
                    match quote {
                        ExchangeEntry::Quote(quote) => quotes.push((exchange, quote)),
                        ExchangeEntry::Ignore(_ignored) => {}
                    }
                }
                Ok(ExchangeBook(quotes))
            }
        }

        deserializer.deserialize_map(BookVisitor)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ExchangeEntry {
    Quote(RawExchangeQuote),
    Ignore(serde::de::IgnoredAny),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ExchangeResponsePayload {
    Cached { data: ExchangeBook },
    Direct(ExchangeBook),
}

fn preferred_number(primary: Option<Number>, fallback: Option<Number>) -> Option<f64> {
    let primary = optional_number(primary);
    if primary.is_some_and(|value| value != 0.0) {
        primary
    } else {
        optional_number(fallback)
    }
}

fn excluded(exchange: &str, side: ExchangeSide) -> bool {
    match side {
        ExchangeSide::Ask => matches!(
            exchange.to_ascii_lowercase().as_str(),
            "banexcoin" | "xapo" | "x4t"
        ),
        ExchangeSide::Bid => exchange.eq_ignore_ascii_case("okexp2p"),
    }
}

#[must_use]
pub fn parse_exchange_quotes(response: HttpResponse, side: ExchangeSide) -> ExchangeQuotesOutcome {
    if response.status_code >= 400 {
        return ExchangeQuotesOutcome::HttpError {
            status_code: response.status_code,
        };
    }
    let Ok(payload) = serde_json::from_str::<ExchangeResponsePayload>(&response.body) else {
        return ExchangeQuotesOutcome::InvalidJson;
    };
    let book = match payload {
        ExchangeResponsePayload::Direct(book) | ExchangeResponsePayload::Cached { data: book } => {
            book
        }
    };
    let quotes = book
        .0
        .into_iter()
        .filter(|(exchange, _quote)| !excluded(exchange, side))
        .map(|(exchange, quote)| {
            let price = match side {
                ExchangeSide::Ask => preferred_number(quote.total_ask, quote.ask),
                ExchangeSide::Bid => preferred_number(quote.total_bid, quote.bid),
            };
            ExchangeQuote { exchange, price }
        })
        .collect();
    ExchangeQuotesOutcome::Quotes(quotes)
}

#[must_use]
pub fn fetch_exchange_quotes<T: CriptoYaTransport>(
    transport: &T,
    fiat: &str,
    side: ExchangeSide,
) -> ExchangeQuotesOutcome {
    let request = CriptoYaRequest::Exchange {
        asset: "USDT".to_owned(),
        fiat: fiat.to_owned(),
        amount: RULO_USD_AMOUNT,
    };
    match transport.get(&request) {
        Ok(response) => parse_exchange_quotes(response, side),
        Err(error) => ExchangeQuotesOutcome::TransportError(error),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use bot_core::devo::DevoQuotes;
    use bot_core::rulo::{ExchangeQuote, RuloInput};

    use super::{
        CriptoYaRequest, CriptoYaTransport, DollarQuotesOutcome, ExchangeQuotesOutcome,
        ExchangeSide, HttpResponse, RuloMarketOutcome, TransportFailureKind, fetch_dollar_quotes,
        fetch_exchange_quotes, fetch_rulo_market, parse_dollar_quotes, parse_exchange_quotes,
        parse_rulo_market,
    };

    struct Transport {
        results: RefCell<Vec<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<CriptoYaRequest>>,
    }

    impl CriptoYaTransport for Transport {
        fn get(&self, request: &CriptoYaRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            if self.results.borrow().is_empty() {
                return Err(TransportFailureKind::Request);
            }
            self.results.borrow_mut().remove(0)
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
        let transport = Transport {
            results: RefCell::new(vec![Err(TransportFailureKind::Timeout)]),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            fetch_dollar_quotes(&transport),
            DollarQuotesOutcome::TransportError(TransportFailureKind::Timeout)
        );
        assert_eq!(
            transport.requests.borrow().as_slice(),
            &[CriptoYaRequest::Dollar]
        );
    }

    #[test]
    fn parses_rulo_market_fields_without_requiring_devo_quotes() {
        let expected = RuloMarketOutcome::Input(RuloInput {
            official: Some(1440.0),
            mep: Some(1459.73),
            blue: Some(1430.0),
            usd_to_usdt: Vec::new(),
            usdt_to_ars: Vec::new(),
            usd_amount: 1000.0,
        });
        for body in [
            r#"{"oficial":{"price":1440},"blue":{"bid":"1430","price":1400},"mep":{"al30":{"ci":{"price":1459.73}}}}"#,
            r#"{"data":{"oficial":{"price":1440},"blue":{"bid":"1430"},"mep":{"al30":{"ci":{"price":1459.73}}}}}"#,
        ] {
            assert_eq!(
                parse_rulo_market(HttpResponse {
                    status_code: 200,
                    body: body.to_owned(),
                }),
                expected
            );
        }
        assert_eq!(
            parse_rulo_market(HttpResponse {
                status_code: 200,
                body: "{}".to_owned(),
            }),
            RuloMarketOutcome::Input(RuloInput {
                official: None,
                mep: None,
                blue: None,
                usd_to_usdt: Vec::new(),
                usdt_to_ars: Vec::new(),
                usd_amount: 1000.0,
            })
        );
        assert_eq!(
            parse_rulo_market(HttpResponse {
                status_code: 200,
                body: r#"{"oficial":{"price":1440},"blue":{"bid":0,"price":1425}}"#.to_owned(),
            }),
            RuloMarketOutcome::Input(RuloInput {
                official: Some(1440.0),
                mep: None,
                blue: Some(1425.0),
                usd_to_usdt: Vec::new(),
                usdt_to_ars: Vec::new(),
                usd_amount: 1000.0,
            })
        );
    }

    #[test]
    fn parses_exchange_precedence_exclusions_and_provider_order() {
        let body = r#"{"buenbit":{"totalAsk":"1.031","ask":1.1},"xapo":{"totalAsk":1.001},"ripio":{"totalAsk":0,"ask":1.04},"broken":{"totalAsk":"bad"},"metadata":"ignored"}"#;
        assert_eq!(
            parse_exchange_quotes(
                HttpResponse {
                    status_code: 200,
                    body: body.to_owned(),
                },
                ExchangeSide::Ask,
            ),
            ExchangeQuotesOutcome::Quotes(vec![
                ExchangeQuote {
                    exchange: "buenbit".to_owned(),
                    price: Some(1.031),
                },
                ExchangeQuote {
                    exchange: "ripio".to_owned(),
                    price: Some(1.04),
                },
                ExchangeQuote {
                    exchange: "broken".to_owned(),
                    price: None,
                },
            ])
        );
        assert_eq!(
            parse_exchange_quotes(
                HttpResponse {
                    status_code: 200,
                    body: r#"{"data":{"okexp2p":{"totalBid":9999},"buenbit":{"totalBid":1458.44,"bid":1400}}}"#.to_owned(),
                },
                ExchangeSide::Bid,
            ),
            ExchangeQuotesOutcome::Quotes(vec![ExchangeQuote {
                exchange: "buenbit".to_owned(),
                price: Some(1458.44),
            }])
        );
    }

    #[test]
    fn rulo_fetches_exact_requests_and_classifies_failures() {
        let transport = Transport {
            results: RefCell::new(vec![
                Ok(HttpResponse {
                    status_code: 200,
                    body: "{}".to_owned(),
                }),
                Err(TransportFailureKind::Connection),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        assert!(matches!(
            fetch_rulo_market(&transport),
            RuloMarketOutcome::Input(_)
        ));
        assert_eq!(
            fetch_exchange_quotes(&transport, "USD", ExchangeSide::Ask),
            ExchangeQuotesOutcome::TransportError(TransportFailureKind::Connection)
        );
        assert_eq!(
            transport.requests.borrow().as_slice(),
            &[
                CriptoYaRequest::Dollar,
                CriptoYaRequest::Exchange {
                    asset: "USDT".to_owned(),
                    fiat: "USD".to_owned(),
                    amount: 1000.0,
                }
            ]
        );
        assert_eq!(
            parse_rulo_market(HttpResponse {
                status_code: 429,
                body: String::new(),
            }),
            RuloMarketOutcome::HttpError { status_code: 429 }
        );
        assert_eq!(
            parse_exchange_quotes(
                HttpResponse {
                    status_code: 200,
                    body: "bad".to_owned(),
                },
                ExchangeSide::Bid,
            ),
            ExchangeQuotesOutcome::InvalidJson
        );
    }
}
