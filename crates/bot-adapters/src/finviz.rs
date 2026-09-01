//! Blocking Finviz mega-cap screener adapter.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Duration;

use regex::Regex;
use reqwest::blocking::Client;
use serde::Serialize;

const SCREENER_URL: &str = "https://finviz.com/screener.ashx";
const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36";
const MAX_SYMBOLS: usize = 10;

static COMPANY_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportFailureKind {
    Timeout,
    Connection,
    Request,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ScreenerOutcome {
    Success { symbols: Vec<String> },
    HttpError { status_code: u16 },
    TransportError { kind: TransportFailureKind },
}

pub trait FinvizTransport {
    fn fetch(&self) -> Result<HttpResponse, TransportFailureKind>;
}

pub struct ReqwestFinvizTransport {
    client: Client,
}

impl ReqwestFinvizTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder().timeout(Duration::from_secs(10)).build()
        })
        .map(|client| Self { client })
        .map_err(|_| TransportFailureKind::Request)
    }
}

impl FinvizTransport for ReqwestFinvizTransport {
    fn fetch(&self) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(SCREENER_URL)
            .query(&[("v", "152"), ("f", "cap_mega"), ("o", "-marketcap")])
            .header("User-Agent", USER_AGENT)
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
pub fn parse_symbols(html: &str) -> Vec<String> {
    let Some(regex) = COMPANY_REGEX
        .get_or_init(|| {
            Regex::new(r#"data-boxover-ticker="([A-Z.]+)"\s+data-boxover-company="([^"]+)""#)
        })
        .as_ref()
        .ok()
    else {
        return Vec::new();
    };
    let mut companies = HashSet::new();
    let mut symbols = Vec::new();
    for captures in regex.captures_iter(html) {
        let (Some(symbol), Some(company)) = (captures.get(1), captures.get(2)) else {
            continue;
        };
        if companies.insert(company.as_str().to_owned()) {
            symbols.push(symbol.as_str().to_owned());
            if symbols.len() == MAX_SYMBOLS {
                break;
            }
        }
    }
    symbols
}

#[must_use]
pub fn fetch_with<T: FinvizTransport>(transport: &T) -> ScreenerOutcome {
    match transport.fetch() {
        Ok(response) if (200..400).contains(&response.status_code) => ScreenerOutcome::Success {
            symbols: parse_symbols(&response.body),
        },
        Ok(response) => ScreenerOutcome::HttpError {
            status_code: response.status_code,
        },
        Err(kind) => ScreenerOutcome::TransportError { kind },
    }
}

#[must_use]
pub fn fetch() -> ScreenerOutcome {
    match ReqwestFinvizTransport::new() {
        Ok(transport) => fetch_with(&transport),
        Err(kind) => ScreenerOutcome::TransportError { kind },
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{
        FinvizTransport, HttpResponse, ScreenerOutcome, TransportFailureKind, fetch_with,
        parse_symbols,
    };

    struct FakeTransport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
    }

    impl FinvizTransport for FakeTransport {
        fn fetch(&self) -> Result<HttpResponse, TransportFailureKind> {
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[test]
    fn parser_deduplicates_companies_preserves_order_and_limits_results() {
        let html = (0..12)
            .map(|index| {
                let symbol = format!("S{}", char::from(b'A' + index));
                let company = if index == 2 {
                    "Company 1".to_owned()
                } else {
                    format!("Company {index}")
                };
                format!(r#"<a data-boxover-ticker="{symbol}" data-boxover-company="{company}">"#)
            })
            .collect::<String>();
        assert_eq!(
            parse_symbols(&html),
            ["SA", "SB", "SD", "SE", "SF", "SG", "SH", "SI", "SJ", "SK"].map(str::to_owned)
        );
    }

    #[test]
    fn fetch_classifies_success_http_and_transport_results() {
        let success = FakeTransport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code: 200,
                body: r#"data-boxover-ticker="EXM" data-boxover-company="Example""#.to_owned(),
            }))),
        };
        assert_eq!(
            fetch_with(&success),
            ScreenerOutcome::Success {
                symbols: vec!["EXM".to_owned()],
            }
        );
        let http = FakeTransport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code: 503,
                body: String::new(),
            }))),
        };
        assert_eq!(
            fetch_with(&http),
            ScreenerOutcome::HttpError { status_code: 503 }
        );
        let transport = FakeTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
        };
        assert_eq!(
            fetch_with(&transport),
            ScreenerOutcome::TransportError {
                kind: TransportFailureKind::Timeout,
            }
        );
    }
}
