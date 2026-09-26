//! Typed CriptoYa dollar-quote adapter.

use std::sync::OnceLock;
use std::time::Duration;

use bot_core::devo::DevoQuotes;
use bot_core::rulo::{ExchangeQuote, RuloInput};
use reqwest::blocking::Client;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

use crate::request_cache::{RequestCache, python_request_cache_key};

const DOLLAR_URL: &str = "https://criptoya.com/api/dolar";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const RULO_USD_AMOUNT: f64 = 1000.0;
/// CriptoYa quotes move slowly enough that repeated /devo and /rulo commands
/// within this window can share one provider response.
pub const CACHE_TTL_SECONDS: i64 = 45;

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
    api_base: String,
}

impl ReqwestCriptoYaTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder().timeout(REQUEST_TIMEOUT).build()
        })
        .map(|client| Self {
            client,
            api_base: DOLLAR_URL.trim_end_matches("/dolar").to_owned(),
        })
        .map_err(|_| TransportFailureKind::Request)
    }

    #[cfg(test)]
    fn with_api_base(api_base: &str) -> Result<Self, TransportFailureKind> {
        Self::new().map(|mut transport| {
            transport.api_base = api_base.trim_end_matches('/').to_owned();
            transport
        })
    }
}

impl CriptoYaTransport for ReqwestCriptoYaTransport {
    fn get(&self, request: &CriptoYaRequest) -> Result<HttpResponse, TransportFailureKind> {
        let url = match request {
            CriptoYaRequest::Dollar => format!("{}/dolar", self.api_base),
            CriptoYaRequest::Exchange {
                asset,
                fiat,
                amount,
            } => format!("{}/{asset}/{fiat}/{amount:.0}", self.api_base),
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
    match transport.get(&exchange_request(fiat)) {
        Ok(response) => parse_exchange_quotes(response, side),
        Err(error) => ExchangeQuotesOutcome::TransportError(error),
    }
}

/// The USDT order-book request used by `/rulo` for one fiat currency.
#[must_use]
pub fn exchange_request(fiat: &str) -> CriptoYaRequest {
    CriptoYaRequest::Exchange {
        asset: "USDT".to_owned(),
        fiat: fiat.to_owned(),
        amount: RULO_USD_AMOUNT,
    }
}

/// Provider responses for [`cached_get_all`], in request order.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedResponses {
    pub results: Vec<Result<HttpResponse, TransportFailureKind>>,
    pub diagnostics: Vec<String>,
}

fn cache_key(request: &CriptoYaRequest) -> String {
    let path = match request {
        CriptoYaRequest::Dollar => "dolar".to_owned(),
        CriptoYaRequest::Exchange {
            asset,
            fiat,
            amount,
        } => format!("{asset}/{fiat}/{amount:.0}"),
    };
    python_request_cache_key(&format!("criptoya:{path}"))
}

#[derive(Deserialize)]
struct CachedEntry {
    timestamp: i64,
    data: Value,
}

fn cached_response<C: RequestCache>(
    cache: &mut C,
    key: &str,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Option<HttpResponse> {
    let raw = match cache.get(key) {
        Ok(raw) => raw?,
        Err(error) => {
            diagnostics.push(format!("could not read CriptoYa cache key {key}: {error}"));
            return None;
        }
    };
    match serde_json::from_str::<CachedEntry>(&raw) {
        Ok(entry) if now_unix.saturating_sub(entry.timestamp) <= CACHE_TTL_SECONDS => {
            Some(HttpResponse {
                status_code: 200,
                body: entry.data.to_string(),
            })
        }
        Ok(_) => None,
        Err(error) => {
            diagnostics.push(format!("invalid CriptoYa cache key {key}: {error}"));
            None
        }
    }
}

fn store_response<C: RequestCache>(
    cache: &mut C,
    key: &str,
    response: &HttpResponse,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) {
    if response.status_code >= 400 {
        return;
    }
    let Ok(data) = serde_json::from_str::<Value>(&response.body) else {
        return;
    };
    let value = json!({"timestamp": now_unix, "data": data}).to_string();
    if let Err(error) = cache.set(key, &value, CACHE_TTL_SECONDS) {
        diagnostics.push(format!("could not write CriptoYa cache key {key}: {error}"));
    }
}

/// Answer each request from the shared request cache when a response younger
/// than [`CACHE_TTL_SECONDS`] exists; fetch the rest from CriptoYa at the same
/// time instead of one after another, and cache every successful JSON
/// response. Failures are returned typed and never cached.
pub fn cached_get_all<T, C>(
    transport: &T,
    cache: &mut C,
    requests: &[CriptoYaRequest],
    now_unix: i64,
) -> CachedResponses
where
    T: CriptoYaTransport + Sync,
    C: RequestCache,
{
    let mut diagnostics = Vec::new();
    let keys = requests.iter().map(cache_key).collect::<Vec<_>>();
    let cached = keys
        .iter()
        .map(|key| cached_response(cache, key, now_unix, &mut diagnostics))
        .collect::<Vec<_>>();
    let missing = requests
        .iter()
        .zip(&cached)
        .filter(|(_request, cached)| cached.is_none())
        .map(|(request, _cached)| request)
        .collect::<Vec<_>>();
    let mut fetched = if missing.len() > 1 {
        std::thread::scope(|scope| {
            let handles = missing
                .iter()
                .map(|request| scope.spawn(move || transport.get(request)))
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or(Err(TransportFailureKind::Request)))
                .collect::<Vec<_>>()
        })
    } else {
        missing
            .iter()
            .map(|request| transport.get(request))
            .collect()
    }
    .into_iter();
    let mut results = Vec::with_capacity(requests.len());
    for (key, cached) in keys.iter().zip(cached) {
        let result = match cached {
            Some(response) => Ok(response),
            None => {
                let result = fetched.next().unwrap_or(Err(TransportFailureKind::Request));
                if let Ok(response) = &result {
                    store_response(cache, key, response, now_unix, &mut diagnostics);
                }
                result
            }
        };
        results.push(result);
    }
    CachedResponses {
        results,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use bot_core::devo::DevoQuotes;
    use bot_core::rulo::{ExchangeQuote, RuloInput};

    use super::{
        CriptoYaRequest, CriptoYaTransport, DollarQuotesOutcome, ExchangeQuotesOutcome,
        ExchangeSide, HttpResponse, ReqwestCriptoYaTransport, RuloMarketOutcome,
        TransportFailureKind, fetch_dollar_quotes, fetch_exchange_quotes, fetch_rulo_market,
        parse_dollar_quotes, parse_exchange_quotes, parse_rulo_market,
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

    #[test]
    fn reqwest_transport_builds_dollar_and_exchange_paths() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            for path in ["/api/dolar", "/api/USDT/ARS/1000"] {
                let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
                let mut request = [0_u8; 1_024];
                let bytes = stream.read(&mut request).unwrap_or_default();
                assert!(
                    String::from_utf8_lossy(&request[..bytes])
                        .starts_with(&format!("GET {path} HTTP/1.1"))
                );
                let body = r#"{"synthetic":true}"#;
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap_or_else(|_| unreachable!());
            }
        });
        let transport = ReqwestCriptoYaTransport::with_api_base(&format!("http://{address}/api/"))
            .unwrap_or_else(|_| unreachable!());
        assert!(transport.get(&CriptoYaRequest::Dollar).is_ok());
        assert!(
            transport
                .get(&CriptoYaRequest::Exchange {
                    asset: "USDT".to_owned(),
                    fiat: "ARS".to_owned(),
                    amount: 1_000.0,
                })
                .is_ok()
        );
        assert!(server.join().is_ok());
        let unavailable = ReqwestCriptoYaTransport::with_api_base("http://127.0.0.1:1/api/")
            .unwrap_or_else(|_| unreachable!());
        assert!(unavailable.get(&CriptoYaRequest::Dollar).is_err());
    }

    struct KeyedTransport {
        requests: std::sync::Mutex<Vec<CriptoYaRequest>>,
        barrier: Option<std::sync::Barrier>,
    }

    impl CriptoYaTransport for KeyedTransport {
        fn get(&self, request: &CriptoYaRequest) -> Result<HttpResponse, TransportFailureKind> {
            if let Ok(mut requests) = self.requests.lock() {
                requests.push(request.clone());
            }
            // Only concurrent calls get past the barrier; a sequential
            // implementation would deadlock here.
            if let Some(barrier) = &self.barrier {
                barrier.wait();
            }
            match request {
                CriptoYaRequest::Dollar => Ok(HttpResponse {
                    status_code: 200,
                    body: r#"{"oficial":{"price":100}}"#.to_owned(),
                }),
                CriptoYaRequest::Exchange { fiat, .. } if fiat == "USD" => Ok(HttpResponse {
                    status_code: 503,
                    body: String::new(),
                }),
                CriptoYaRequest::Exchange { fiat, .. } if fiat == "ARS" => Ok(HttpResponse {
                    status_code: 200,
                    body: "not-json".to_owned(),
                }),
                CriptoYaRequest::Exchange { .. } => Err(TransportFailureKind::Timeout),
            }
        }
    }

    #[derive(Default)]
    struct MemoryCache {
        values: std::collections::HashMap<String, String>,
        fail: bool,
    }

    impl crate::request_cache::RequestCache for MemoryCache {
        type Error = String;

        fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
            if self.fail {
                return Err("synthetic read failure".to_owned());
            }
            Ok(self.values.get(key).cloned())
        }

        fn set(&mut self, key: &str, value: &str, _ttl_seconds: i64) -> Result<(), Self::Error> {
            if self.fail {
                return Err("synthetic write failure".to_owned());
            }
            self.values.insert(key.to_owned(), value.to_owned());
            Ok(())
        }

        fn take(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
            Ok(self.values.remove(key))
        }

        fn claim(&mut self, _key: &str, _value: &str, _ttl: i64) -> Result<bool, Self::Error> {
            Ok(true)
        }
    }

    fn rulo_requests() -> Vec<CriptoYaRequest> {
        vec![
            CriptoYaRequest::Dollar,
            super::exchange_request("USD"),
            super::exchange_request("ARS"),
        ]
    }

    #[test]
    fn cached_get_all_fetches_misses_concurrently_and_caches_only_json_successes() {
        let transport = KeyedTransport {
            requests: std::sync::Mutex::new(Vec::new()),
            barrier: Some(std::sync::Barrier::new(3)),
        };
        let mut cache = MemoryCache::default();
        let now = 1_700_000_000;
        let load = super::cached_get_all(&transport, &mut cache, &rulo_requests(), now);
        assert!(load.diagnostics.is_empty());
        assert_eq!(load.results.len(), 3);
        assert!(load.results[0].as_ref().is_ok_and(|r| r.status_code == 200));
        assert!(load.results[1].as_ref().is_ok_and(|r| r.status_code == 503));
        assert!(load.results[2].as_ref().is_ok_and(|r| r.body == "not-json"));
        assert_eq!(cache.values.len(), 1);

        // A fresh cached dollar response is served without contacting CriptoYa;
        // the two uncached requests still run (concurrently, or the barrier
        // would block).
        let transport = KeyedTransport {
            requests: std::sync::Mutex::new(Vec::new()),
            barrier: Some(std::sync::Barrier::new(2)),
        };
        let load = super::cached_get_all(
            &transport,
            &mut cache,
            &rulo_requests(),
            now + super::CACHE_TTL_SECONDS,
        );
        assert_eq!(
            load.results[0],
            Ok(HttpResponse {
                status_code: 200,
                body: r#"{"oficial":{"price":100}}"#.to_owned(),
            })
        );
        assert_eq!(transport.requests.lock().map(|r| r.len()).ok(), Some(2));

        // Once the entry is older than the TTL it is fetched again, alone.
        let transport = KeyedTransport {
            requests: std::sync::Mutex::new(Vec::new()),
            barrier: None,
        };
        let load = super::cached_get_all(
            &transport,
            &mut cache,
            &[CriptoYaRequest::Dollar],
            now + super::CACHE_TTL_SECONDS + 1,
        );
        assert!(load.results[0].is_ok());
        assert_eq!(transport.requests.lock().map(|r| r.len()).ok(), Some(1));
    }

    #[test]
    fn cached_get_all_reports_cache_failures_and_still_fetches() {
        let transport = KeyedTransport {
            requests: std::sync::Mutex::new(Vec::new()),
            barrier: None,
        };
        let mut cache = MemoryCache {
            fail: true,
            ..MemoryCache::default()
        };
        let load = super::cached_get_all(
            &transport,
            &mut cache,
            &[CriptoYaRequest::Dollar, super::exchange_request("EUR")],
            1_700_000_000,
        );
        assert_eq!(load.results[1], Err(TransportFailureKind::Timeout));
        assert!(load.results[0].is_ok());
        assert_eq!(load.diagnostics.len(), 3);
        assert!(load.diagnostics[0].contains("could not read"));
        assert!(load.diagnostics[2].contains("could not write"));

        let mut cache = MemoryCache::default();
        cache
            .values
            .insert(super::cache_key(&CriptoYaRequest::Dollar), "bad".to_owned());
        let load = super::cached_get_all(
            &transport,
            &mut cache,
            &[CriptoYaRequest::Dollar],
            1_700_000_000,
        );
        assert!(load.results[0].is_ok());
        assert!(load.diagnostics[0].contains("invalid CriptoYa cache key"));
    }
}
