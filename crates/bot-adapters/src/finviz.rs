//! Blocking Finviz mega-cap screener adapter.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::Serialize;

const SCREENER_URL: &str = "https://finviz.com/screener/stocks";
const MAX_SYMBOLS: usize = 10;

/// Each screener row starts with this attribute, followed by the symbol.
const ROW_KEY: &str = "data-row-key=\"";
const COMPANY_CELL: &str = "data-column-id=\"company\"";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

pub use crate::http_client::TransportFailureKind;
use crate::http_client::{BROWSER_USER_AGENT, classify_error};

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
    screener_url: String,
}

impl ReqwestFinvizTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder().timeout(Duration::from_secs(10)).build()
        })
        .map(|client| Self {
            client,
            screener_url: SCREENER_URL.to_owned(),
        })
        .map_err(|_| TransportFailureKind::Request)
    }

    #[cfg(test)]
    fn with_screener_url(screener_url: &str) -> Result<Self, TransportFailureKind> {
        Self::new().map(|mut transport| {
            transport.screener_url = screener_url.to_owned();
            transport
        })
    }
}

impl FinvizTransport for ReqwestFinvizTransport {
    fn fetch(&self) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(&self.screener_url)
            .query(&[("v", "152"), ("f", "cap_mega"), ("o", "-marketcap")])
            .header("User-Agent", BROWSER_USER_AGENT)
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_error)
    }
}

/// Reads the screener table's rows in order, one symbol per company, so
/// share classes such as GOOGL and GOOG count once.
#[must_use]
pub fn parse_symbols(html: &str) -> Vec<String> {
    let mut companies = HashSet::new();
    let mut symbols = Vec::new();
    for row in html.split(ROW_KEY).skip(1) {
        let Some((symbol, row)) = row.split_once('"') else {
            continue;
        };
        if symbol.is_empty()
            || !symbol.bytes().all(|byte| {
                byte.is_ascii_uppercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-')
            })
        {
            continue;
        }
        let company = row
            .split_once(COMPANY_CELL)
            .and_then(|(_, cell)| cell.split_once('>'))
            .and_then(|(_, cell)| cell.split_once('<'))
            .map_or(symbol, |(company, _)| company.trim());
        if companies.insert(company.to_owned()) {
            symbols.push(symbol.to_owned());
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

#[cfg(test)]
pub(crate) mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::{
        FinvizTransport, HttpResponse, ReqwestFinvizTransport, ScreenerOutcome,
        TransportFailureKind, fetch_with, parse_symbols,
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

    /// One screener row in Finviz's table markup.
    pub(crate) fn row(symbol: &str, company: &str) -> String {
        format!(
            concat!(
                r#"<div class="fv-table_row" data-row-key="{symbol}" role="row">"#,
                r#"<div data-column-id="ticker" role="gridcell" data-ticker="{symbol}">"#,
                r#"<a href="/quote?t={symbol}"><span>{symbol}</span></a></div>"#,
                r#"<div aria-colindex="3" class="fv-table_cell" data-column-id="company" role="gridcell">{company}</div>"#,
                r#"<div data-column-id="sector" role="gridcell">Technology</div></div>"#
            ),
            symbol = symbol,
            company = company,
        )
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
                row(&symbol, &company)
            })
            .collect::<String>();
        assert_eq!(
            parse_symbols(&html),
            ["SA", "SB", "SD", "SE", "SF", "SG", "SH", "SI", "SJ", "SK"].map(str::to_owned)
        );
    }

    #[test]
    fn parser_reads_class_shares_and_rows_without_a_company() {
        let html = [
            row("BRK-B", "Berkshire Hathaway Inc"),
            row("BRK-A", "Berkshire Hathaway Inc"),
            row("BF.B", "Brown-Forman Corp"),
            r#"<div data-row-key="NOCO" role="row"></div>"#.to_owned(),
            r#"<div data-row-key="lower" role="row"></div><div data-row-key="OPEN"#.to_owned(),
        ]
        .concat();
        assert_eq!(
            parse_symbols(&html),
            ["BRK-B", "BF.B", "NOCO"].map(str::to_owned)
        );
    }

    #[test]
    fn fetch_classifies_success_http_and_transport_results() {
        let success = FakeTransport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code: 200,
                body: row("EXM", "Example"),
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

    #[test]
    fn reqwest_transport_sends_the_mega_cap_screener_contract() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 2_048];
            let bytes = stream.read(&mut request).unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..bytes]);
            assert!(
                request.starts_with("GET /screener/stocks?v=152&f=cap_mega&o=-marketcap HTTP/1.1")
            );
            assert!(request.to_ascii_lowercase().contains(
                &format!("user-agent: {}", super::BROWSER_USER_AGENT).to_ascii_lowercase()
            ));
            let body = "synthetic screener";
            let written = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            written?;
            Ok(())
        });
        let transport =
            ReqwestFinvizTransport::with_screener_url(&format!("http://{address}/screener/stocks"))
                .ok()
                .ok_or("unexpected error")?;
        let response = transport.fetch().ok().ok_or("unexpected error")?;
        assert_eq!(response.status_code, 200);
        assert_eq!(response.body, "synthetic screener");
        assert!(matches!(server.join(), Ok(Ok(()))));
        let unavailable = ReqwestFinvizTransport::with_screener_url("http://127.0.0.1:1/screener")
            .ok()
            .ok_or("unexpected error")?;
        assert!(unavailable.fetch().is_err());
        Ok(())
    }
}
