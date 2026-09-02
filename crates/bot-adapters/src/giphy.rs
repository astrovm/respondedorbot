//! Blocking Giphy search adapter for greeting GIF pools.

use std::time::Duration;

use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::Value;

const GIPHY_SEARCH_URL: &str = "https://api.giphy.com/v1/gifs/search";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchRequest {
    pub api_key: String,
    pub term: String,
    pub offset: u16,
}

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
pub enum SearchOutcome {
    Success { urls: Vec<String> },
    HttpError { status_code: u16 },
    InvalidJson,
    InvalidPayload,
    TransportError { kind: TransportFailureKind },
}

pub trait GiphyTransport {
    fn search(&self, request: &SearchRequest) -> Result<HttpResponse, TransportFailureKind>;
}

pub struct ReqwestGiphyTransport {
    client: Client,
    search_url: String,
}

impl ReqwestGiphyTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map(|client| Self {
                client,
                search_url: GIPHY_SEARCH_URL.to_owned(),
            })
            .map_err(|_| TransportFailureKind::Request)
    }

    #[cfg(test)]
    fn with_search_url(search_url: &str) -> Result<Self, TransportFailureKind> {
        Self::new().map(|mut transport| {
            transport.search_url = search_url.to_owned();
            transport
        })
    }
}

impl GiphyTransport for ReqwestGiphyTransport {
    fn search(&self, request: &SearchRequest) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(&self.search_url)
            .query(&[
                ("api_key", request.api_key.clone()),
                ("q", request.term.clone()),
                ("limit", "25".to_owned()),
                ("offset", request.offset.to_string()),
                ("rating", "g".to_owned()),
            ])
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
pub fn parse_response(response: HttpResponse) -> SearchOutcome {
    if response.status_code >= 400 {
        return SearchOutcome::HttpError {
            status_code: response.status_code,
        };
    }
    let Ok(payload) = serde_json::from_str::<Value>(&response.body) else {
        return SearchOutcome::InvalidJson;
    };
    let Some(items) = payload.get("data").and_then(Value::as_array) else {
        return SearchOutcome::InvalidPayload;
    };
    let mut urls = Vec::new();
    for item in items {
        let Some(item) = item.as_object() else {
            return SearchOutcome::InvalidPayload;
        };
        let Some(images) = item.get("images") else {
            continue;
        };
        let Some(images) = images.as_object() else {
            return SearchOutcome::InvalidPayload;
        };
        let Some(original) = images.get("original") else {
            continue;
        };
        let Some(original) = original.as_object() else {
            return SearchOutcome::InvalidPayload;
        };
        match original.get("url") {
            Some(Value::String(url)) if !url.is_empty() => urls.push(url.clone()),
            Some(Value::Null) | Some(Value::String(_)) | None => {}
            Some(_) => return SearchOutcome::InvalidPayload,
        }
    }
    SearchOutcome::Success { urls }
}

pub fn search_with<T: GiphyTransport>(
    transport: &T,
    api_key: &str,
    term: &str,
    offset: u16,
) -> SearchOutcome {
    let request = SearchRequest {
        api_key: api_key.to_owned(),
        term: term.to_owned(),
        offset,
    };
    match transport.search(&request) {
        Ok(response) => parse_response(response),
        Err(kind) => SearchOutcome::TransportError { kind },
    }
}

#[must_use]
pub fn search(api_key: &str, term: &str, offset: u16) -> SearchOutcome {
    match ReqwestGiphyTransport::new() {
        Ok(transport) => search_with(&transport, api_key, term, offset),
        Err(kind) => SearchOutcome::TransportError { kind },
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::{
        GiphyTransport, HttpResponse, ReqwestGiphyTransport, SearchOutcome, SearchRequest,
        TransportFailureKind, parse_response, search_with,
    };

    struct FakeTransport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<SearchRequest>>,
    }

    impl GiphyTransport for FakeTransport {
        fn search(&self, request: &SearchRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[test]
    fn extracts_original_urls_and_ignores_missing_or_empty_urls() {
        assert_eq!(
            parse_response(HttpResponse {
                status_code: 200,
                body: r#"{"data":[{"images":{"original":{"url":"https://example.test/a.gif"}}},{"images":{"original":{"url":""}}},{"images":{}},{}]}"#.to_owned(),
            }),
            SearchOutcome::Success {
                urls: vec!["https://example.test/a.gif".to_owned()],
            }
        );
    }

    #[test]
    fn classifies_http_json_and_payload_failures() {
        assert_eq!(
            parse_response(HttpResponse {
                status_code: 429,
                body: "ignored".to_owned(),
            }),
            SearchOutcome::HttpError { status_code: 429 }
        );
        assert_eq!(
            parse_response(HttpResponse {
                status_code: 200,
                body: "not-json".to_owned(),
            }),
            SearchOutcome::InvalidJson
        );
        for body in [
            r#"[]"#,
            r#"{"data":{}}"#,
            r#"{"data":[1]}"#,
            r#"{"data":[{"images":[] }]}"#,
            r#"{"data":[{"images":{"original":[]}}]}"#,
            r#"{"data":[{"images":{"original":{"url":1}}}]}"#,
        ] {
            assert_eq!(
                parse_response(HttpResponse {
                    status_code: 200,
                    body: body.to_owned(),
                }),
                SearchOutcome::InvalidPayload
            );
        }
    }

    #[test]
    fn request_preserves_credentials_term_offset_and_transport_failure() {
        let transport = FakeTransport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            search_with(&transport, "synthetic-key", "buenos dias", 73),
            SearchOutcome::TransportError {
                kind: TransportFailureKind::Timeout,
            }
        );
        let requests = transport.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].api_key, "synthetic-key");
        assert_eq!(requests[0].term, "buenos dias");
        assert_eq!(requests[0].offset, 73);
    }

    #[test]
    fn reqwest_transport_sends_the_complete_search_contract() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
            let mut request = [0_u8; 2_048];
            let bytes = stream.read(&mut request).unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..bytes]);
            assert!(request.starts_with("GET /search?"));
            for expected in [
                "api_key=synthetic-key",
                "q=synthetic+term",
                "limit=25",
                "offset=9",
                "rating=g",
            ] {
                assert!(request.contains(expected), "{request}");
            }
            let body = r#"{"data":[]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap_or_else(|_| unreachable!());
        });
        let transport = ReqwestGiphyTransport::with_search_url(&format!("http://{address}/search"))
            .unwrap_or_else(|_| unreachable!());
        let response = transport
            .search(&SearchRequest {
                api_key: "synthetic-key".to_owned(),
                term: "synthetic term".to_owned(),
                offset: 9,
            })
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(response.status_code, 200);
        assert_eq!(response.body, r#"{"data":[]}"#);
        assert!(server.join().is_ok());
        let unavailable = ReqwestGiphyTransport::with_search_url("http://127.0.0.1:1/search")
            .unwrap_or_else(|_| unreachable!());
        assert!(
            unavailable
                .search(&SearchRequest {
                    api_key: "synthetic-key".to_owned(),
                    term: "synthetic term".to_owned(),
                    offset: 0,
                })
                .is_err()
        );
    }
}
