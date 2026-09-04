//! Blocking Firecrawl search adapter.

use std::io::Read;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::OnceLock;
use thiserror::Error;

use bot_core::provider_pricing::FIRECRAWL_AUDIO_CREDITS;

const SEARCH_URL: &str = "https://api.firecrawl.dev/v2/search";
const SCRAPE_URL: &str = "https://api.firecrawl.dev/v2/scrape";
const MAX_RESULTS: usize = 5;
const MAX_ATTEMPTS: usize = 3;
const API_TIMEOUT_MS: u64 = 60_000;
const MAX_DESCRIPTION_CHARS: usize = 1_200;
pub const FIRECRAWL_AUDIO_MAX_BYTES: u64 = 50_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchRequest {
    pub query: String,
    pub api_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryHttpResponse {
    pub status_code: u16,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransportError {
    #[error("Firecrawl request timed out")]
    Timeout,
    #[error("Firecrawl connection failed")]
    Connection,
    #[error("Firecrawl transport failed: {0}")]
    Other(String),
}

pub trait FirecrawlTransport {
    fn post(&self, request: &SearchRequest) -> Result<HttpResponse, TransportError>;
}

pub trait FirecrawlAudioTransport {
    fn scrape_audio(&self, request: &SearchRequest) -> Result<HttpResponse, TransportError>;

    fn download_audio(&self, url: &str) -> Result<BinaryHttpResponse, TransportError>;
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AudioScrapeOutcome {
    Success {
        audio_url: String,
        title: String,
        credits_used: Value,
        request_id: Value,
    },
    Timeout,
    Connection,
    HttpError {
        status_code: u16,
        detail: String,
    },
    InvalidJson,
    ApiError {
        detail: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SearchOutcome {
    Success {
        query: String,
        results: Vec<SearchResult>,
        credits_used: Value,
        request_id: Value,
    },
    Timeout,
    Connection,
    HttpError {
        status_code: u16,
        detail: String,
    },
    InvalidJson,
    ApiError {
        detail: String,
    },
}

pub struct ReqwestFirecrawlTransport {
    client: Client,
    search_url: String,
    scrape_url: String,
}

impl ReqwestFirecrawlTransport {
    pub fn new() -> Result<Self, TransportError> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(75))
                .build()
        })
        .map(|client| Self {
            client,
            search_url: SEARCH_URL.to_owned(),
            scrape_url: SCRAPE_URL.to_owned(),
        })
        .map_err(|error| TransportError::Other(error.to_string()))
    }

    #[cfg(test)]
    fn with_search_url(search_url: &str) -> Result<Self, TransportError> {
        Self::new().map(|mut transport| {
            transport.search_url = search_url.to_owned();
            transport
        })
    }

    #[cfg(test)]
    fn with_scrape_url(scrape_url: &str) -> Result<Self, TransportError> {
        Self::new().map(|mut transport| {
            transport.scrape_url = scrape_url.to_owned();
            transport
        })
    }
}

impl FirecrawlTransport for ReqwestFirecrawlTransport {
    fn post(&self, request: &SearchRequest) -> Result<HttpResponse, TransportError> {
        let response = self
            .client
            .post(&self.search_url)
            .bearer_auth(&request.api_key)
            .json(&json!({
                "query": request.query,
                "limit": MAX_RESULTS,
                "sources": ["web"],
                "timeout": API_TIMEOUT_MS,
            }))
            .send()
            .map_err(classify_reqwest_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_reqwest_error)
    }
}

impl FirecrawlAudioTransport for ReqwestFirecrawlTransport {
    fn scrape_audio(&self, request: &SearchRequest) -> Result<HttpResponse, TransportError> {
        let response = self
            .client
            .post(&self.scrape_url)
            .bearer_auth(&request.api_key)
            .json(&json!({
                "url": request.query,
                "formats": ["audio"],
            }))
            .send()
            .map_err(classify_reqwest_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_reqwest_error)
    }

    fn download_audio(&self, url: &str) -> Result<BinaryHttpResponse, TransportError> {
        let parsed = url::Url::parse(url).map_err(|_| {
            TransportError::Other("Firecrawl returned an invalid audio URL".to_owned())
        })?;
        if parsed.scheme() != "https" {
            return Err(TransportError::Other(
                "Firecrawl returned a non-HTTPS audio URL".to_owned(),
            ));
        }
        let mut response = self
            .client
            .get(parsed)
            .send()
            .map_err(classify_reqwest_error)?;
        let status_code = response.status().as_u16();
        let content_length = response.content_length();
        read_audio_response(
            status_code,
            content_length,
            &mut response,
            FIRECRAWL_AUDIO_MAX_BYTES,
        )
    }
}

fn read_audio_response(
    status_code: u16,
    content_length: Option<u64>,
    reader: &mut impl Read,
    max_bytes: u64,
) -> Result<BinaryHttpResponse, TransportError> {
    if content_length.is_some_and(|length| length > max_bytes) {
        return Err(TransportError::Other(
            "Firecrawl audio exceeds the size limit".to_owned(),
        ));
    }
    let capacity = content_length
        .and_then(|length| usize::try_from(length.min(max_bytes)).ok())
        .unwrap_or_default();
    let mut body = Vec::with_capacity(capacity);
    reader
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut body)
        .map_err(|error| TransportError::Other(error.to_string()))?;
    if body.len() as u64 > max_bytes {
        return Err(TransportError::Other(
            "Firecrawl audio exceeds the size limit".to_owned(),
        ));
    }
    Ok(BinaryHttpResponse { status_code, body })
}

fn classify_reqwest_error(error: reqwest::Error) -> TransportError {
    if error.is_timeout() {
        TransportError::Timeout
    } else if error.is_connect() {
        TransportError::Connection
    } else {
        TransportError::Other(error.to_string())
    }
}

#[must_use]
pub fn clean_text(value: &Value, max_chars: usize) -> String {
    let text = match value {
        Value::Null | Value::Bool(false) => String::new(),
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Number(value) if value.as_f64() == Some(0.0) => String::new(),
        Value::Array(value) if value.is_empty() => String::new(),
        Value::Object(value) if value.is_empty() => String::new(),
        other => other.to_string(),
    };
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let prefix: String = collapsed
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect();
    format!("{}…", prefix.trim_end())
}

fn response_error(body: &str) -> String {
    let Ok(payload) = serde_json::from_str::<Value>(body) else {
        let detail = clean_text(&Value::String(body.to_owned()), 500);
        return if detail.is_empty() {
            "respuesta sin detalles".to_owned()
        } else {
            detail
        };
    };
    let detail = payload
        .as_object()
        .and_then(|object| object.get("error").or_else(|| object.get("message")))
        .unwrap_or(&payload);
    clean_text(detail, 500)
}

fn extract_results(payload: &Value) -> Vec<SearchResult> {
    let Some(data) = payload.get("data") else {
        return Vec::new();
    };
    let raw_results = data
        .as_object()
        .and_then(|object| object.get("web"))
        .unwrap_or(data);
    let Some(raw_results) = raw_results.as_array() else {
        return Vec::new();
    };
    raw_results
        .iter()
        .take(MAX_RESULTS)
        .filter_map(|raw_result| {
            let object = raw_result.as_object()?;
            let url = clean_text(object.get("url").unwrap_or(&Value::Null), 2_000);
            if url.is_empty() {
                return None;
            }
            Some(SearchResult {
                title: clean_text(object.get("title").unwrap_or(&Value::Null), 300),
                url,
                description: clean_text(
                    object.get("description").unwrap_or(&Value::Null),
                    MAX_DESCRIPTION_CHARS,
                ),
            })
        })
        .collect()
}

#[must_use]
pub fn parse_response(response: HttpResponse, query: &str) -> SearchOutcome {
    if response.status_code >= 400 {
        return SearchOutcome::HttpError {
            status_code: response.status_code,
            detail: response_error(&response.body),
        };
    }
    let Ok(payload) = serde_json::from_str::<Value>(&response.body) else {
        return SearchOutcome::InvalidJson;
    };
    if payload.get("success") != Some(&Value::Bool(true)) {
        return SearchOutcome::ApiError {
            detail: response_error(&response.body),
        };
    }
    SearchOutcome::Success {
        query: query.to_owned(),
        results: extract_results(&payload),
        credits_used: payload.get("creditsUsed").cloned().unwrap_or(Value::Null),
        request_id: payload.get("id").cloned().unwrap_or(Value::Null),
    }
}

#[must_use]
pub fn parse_audio_response(response: HttpResponse) -> AudioScrapeOutcome {
    if response.status_code >= 400 {
        return AudioScrapeOutcome::HttpError {
            status_code: response.status_code,
            detail: response_error(&response.body),
        };
    }
    let Ok(payload) = serde_json::from_str::<Value>(&response.body) else {
        return AudioScrapeOutcome::InvalidJson;
    };
    if payload.get("success") != Some(&Value::Bool(true)) {
        return AudioScrapeOutcome::ApiError {
            detail: response_error(&response.body),
        };
    }
    let data = payload.get("data").unwrap_or(&Value::Null);
    let audio_url = clean_text(data.get("audio").unwrap_or(&Value::Null), 4_000);
    if audio_url.is_empty() {
        return AudioScrapeOutcome::ApiError {
            detail: "Firecrawl returned no YouTube audio".to_owned(),
        };
    }
    AudioScrapeOutcome::Success {
        audio_url,
        title: clean_text(
            data.get("metadata")
                .and_then(|metadata| metadata.get("title"))
                .unwrap_or(&Value::Null),
            500,
        ),
        credits_used: payload
            .get("creditsUsed")
            .cloned()
            .unwrap_or_else(|| json!(FIRECRAWL_AUDIO_CREDITS)),
        request_id: payload.get("id").cloned().unwrap_or(Value::Null),
    }
}

pub fn scrape_audio_with<T, S>(
    transport: &T,
    api_key: &str,
    url: &str,
    sleep: S,
) -> Result<AudioScrapeOutcome, TransportError>
where
    T: FirecrawlAudioTransport,
    S: Fn(Duration),
{
    let request = SearchRequest {
        query: url.to_owned(),
        api_key: api_key.to_owned(),
    };
    let mut attempt = 0;
    loop {
        match transport.scrape_audio(&request) {
            Ok(response) => {
                let retryable =
                    matches!(response.status_code, 408 | 409 | 429) || response.status_code >= 500;
                if retryable && attempt + 1 < MAX_ATTEMPTS {
                    sleep(Duration::from_secs(1 << attempt));
                    attempt += 1;
                    continue;
                }
                return Ok(parse_audio_response(response));
            }
            Err(TransportError::Timeout) => {
                if attempt + 1 == MAX_ATTEMPTS {
                    return Ok(AudioScrapeOutcome::Timeout);
                }
                sleep(Duration::from_secs(1 << attempt));
                attempt += 1;
            }
            Err(TransportError::Connection) => {
                if attempt + 1 == MAX_ATTEMPTS {
                    return Ok(AudioScrapeOutcome::Connection);
                }
                sleep(Duration::from_secs(1 << attempt));
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

pub fn search_with<T, S>(
    transport: &T,
    api_key: &str,
    query: &str,
    sleep: S,
) -> Result<SearchOutcome, TransportError>
where
    T: FirecrawlTransport,
    S: Fn(Duration),
{
    let request = SearchRequest {
        query: query.to_owned(),
        api_key: api_key.to_owned(),
    };
    let mut attempt = 0;
    loop {
        match transport.post(&request) {
            Ok(response) => {
                let retryable =
                    matches!(response.status_code, 408 | 409 | 429) || response.status_code >= 500;
                if retryable && attempt + 1 < MAX_ATTEMPTS {
                    sleep(Duration::from_secs(1 << attempt));
                    attempt += 1;
                    continue;
                }
                return Ok(parse_response(response, query));
            }
            Err(TransportError::Timeout) => {
                if attempt + 1 == MAX_ATTEMPTS {
                    return Ok(SearchOutcome::Timeout);
                }
                sleep(Duration::from_secs(1 << attempt));
                attempt += 1;
            }
            Err(TransportError::Connection) => {
                if attempt + 1 == MAX_ATTEMPTS {
                    return Ok(SearchOutcome::Connection);
                }
                sleep(Duration::from_secs(1 << attempt));
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{Cursor, Error, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::{
        AudioScrapeOutcome, FirecrawlAudioTransport, FirecrawlTransport, HttpResponse,
        ReqwestFirecrawlTransport, SearchOutcome, SearchRequest, TransportError, clean_text,
        extract_results, parse_audio_response, read_audio_response, response_error,
        scrape_audio_with, search_with,
    };
    use serde_json::{Value, json};

    struct FakeTransport {
        responses: RefCell<Vec<Result<HttpResponse, TransportError>>>,
        requests: RefCell<Vec<SearchRequest>>,
    }

    impl FirecrawlTransport for FakeTransport {
        fn post(&self, request: &SearchRequest) -> Result<HttpResponse, TransportError> {
            self.requests.borrow_mut().push(request.clone());
            self.responses.borrow_mut().remove(0)
        }
    }

    struct FakeAudioTransport {
        responses: RefCell<Vec<Result<HttpResponse, TransportError>>>,
        requests: RefCell<Vec<SearchRequest>>,
    }

    impl FirecrawlAudioTransport for FakeAudioTransport {
        fn scrape_audio(&self, request: &SearchRequest) -> Result<HttpResponse, TransportError> {
            self.requests.borrow_mut().push(request.clone());
            self.responses.borrow_mut().remove(0)
        }

        fn download_audio(&self, _: &str) -> Result<super::BinaryHttpResponse, TransportError> {
            Err(TransportError::Other("unused download".to_owned()))
        }
    }

    fn response(status_code: u16, body: Value) -> Result<HttpResponse, TransportError> {
        Ok(HttpResponse {
            status_code,
            body: body.to_string(),
        })
    }

    #[test]
    fn success_normalizes_bounded_results_and_preserves_accounting_metadata() {
        let transport = FakeTransport {
            responses: RefCell::new(vec![response(
                200,
                json!({
                    "success": true,
                    "id": "request-1",
                    "creditsUsed": 2,
                    "data": {"web": [
                        {"title": "  Example\nTitle ", "url": "https://example.com", "description": " summary "},
                        {"title": "missing URL"}
                    ]}
                }),
            )]),
            requests: RefCell::new(Vec::new()),
        };
        let outcome = search_with(&transport, "synthetic-key", "query", |_| {});
        assert!(outcome.is_ok());
        let Ok(outcome) = outcome else {
            return;
        };
        assert!(matches!(outcome, SearchOutcome::Success { .. }));
        if let SearchOutcome::Success {
            results,
            credits_used,
            request_id,
            ..
        } = outcome
        {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].title, "Example Title");
            assert_eq!(credits_used, json!(2));
            assert_eq!(request_id, json!("request-1"));
        }
        assert_eq!(transport.requests.borrow()[0].api_key, "synthetic-key");
    }

    #[test]
    fn transient_statuses_and_timeouts_retry_with_exponential_delays() {
        let transport = FakeTransport {
            responses: RefCell::new(vec![
                response(500, json!({"error": "temporary"})),
                Err(TransportError::Timeout),
                response(200, json!({"success": true, "data": []})),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        let delays = RefCell::new(Vec::new());
        let outcome = search_with(&transport, "key", "query", |delay| {
            delays.borrow_mut().push(delay.as_secs());
        });
        assert!(outcome.is_ok());
        let Ok(outcome) = outcome else {
            return;
        };
        assert!(matches!(outcome, SearchOutcome::Success { .. }));
        assert_eq!(*delays.borrow(), vec![1, 2]);
    }

    #[test]
    fn final_transport_and_response_failures_are_typed() {
        for (error, expected) in [
            (TransportError::Timeout, SearchOutcome::Timeout),
            (TransportError::Connection, SearchOutcome::Connection),
        ] {
            let transport = FakeTransport {
                responses: RefCell::new(vec![Err(error.clone()), Err(error.clone()), Err(error)]),
                requests: RefCell::new(Vec::new()),
            };
            assert_eq!(
                search_with(&transport, "key", "query", |_| {}).ok(),
                Some(expected),
            );
        }
        let http = FakeTransport {
            responses: RefCell::new(vec![response(400, json!({"error": "bad request"}))]),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            search_with(&http, "key", "query", |_| {}).ok(),
            Some(SearchOutcome::HttpError {
                status_code: 400,
                detail: "bad request".to_owned(),
            }),
        );
    }

    #[test]
    fn malformed_and_unsuccessful_success_responses_remain_distinct() {
        let invalid = FakeTransport {
            responses: RefCell::new(vec![Ok(HttpResponse {
                status_code: 200,
                body: "not-json".to_owned(),
            })]),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            search_with(&invalid, "key", "query", |_| {}).ok(),
            Some(SearchOutcome::InvalidJson),
        );
        let rejected = FakeTransport {
            responses: RefCell::new(vec![response(
                200,
                json!({"success": false, "message": "denied"}),
            )]),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            search_with(&rejected, "key", "query", |_| {}).ok(),
            Some(SearchOutcome::ApiError {
                detail: "denied".to_owned(),
            }),
        );
    }

    #[test]
    fn audio_response_preserves_provider_metadata_and_fixed_credit_fallback() {
        assert_eq!(
            parse_audio_response(HttpResponse {
                status_code: 200,
                body: json!({
                    "success": true,
                    "id": "synthetic-request",
                    "data": {
                        "audio": "https://media.example.test/audio.mp3",
                        "metadata": {"title": "Synthetic title"}
                    }
                })
                .to_string(),
            }),
            AudioScrapeOutcome::Success {
                audio_url: "https://media.example.test/audio.mp3".to_owned(),
                title: "Synthetic title".to_owned(),
                credits_used: json!(5),
                request_id: json!("synthetic-request"),
            }
        );
        assert!(matches!(
            parse_audio_response(HttpResponse {
                status_code: 200,
                body: json!({"success": true, "data": {}}).to_string(),
            }),
            AudioScrapeOutcome::ApiError { .. }
        ));
        assert!(matches!(
            parse_audio_response(HttpResponse {
                status_code: 200,
                body: json!({"success": false, "message": "synthetic rejection"}).to_string(),
            }),
            AudioScrapeOutcome::ApiError { detail } if detail == "synthetic rejection"
        ));
        assert_eq!(
            parse_audio_response(HttpResponse {
                status_code: 502,
                body: json!({"error": "temporary"}).to_string(),
            }),
            AudioScrapeOutcome::HttpError {
                status_code: 502,
                detail: "temporary".to_owned(),
            }
        );
        assert_eq!(
            parse_audio_response(HttpResponse {
                status_code: 200,
                body: "invalid".to_owned(),
            }),
            AudioScrapeOutcome::InvalidJson
        );
    }

    #[test]
    fn audio_scrape_retries_transient_failures_and_keeps_request_identity() {
        let transport = FakeAudioTransport {
            responses: RefCell::new(vec![
                Err(TransportError::Connection),
                response(503, json!({"error": "temporary"})),
                response(
                    200,
                    json!({
                        "success": true,
                        "creditsUsed": 4,
                        "data": {"audio": "https://media.example.test/audio.mp3"}
                    }),
                ),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        let delays = RefCell::new(Vec::new());
        let outcome = scrape_audio_with(
            &transport,
            "synthetic-key",
            "https://youtube.test/watch?v=synthetic",
            |delay| delays.borrow_mut().push(delay.as_secs()),
        );
        assert!(matches!(outcome, Ok(AudioScrapeOutcome::Success { .. })));
        assert_eq!(*delays.borrow(), vec![1, 2]);
        assert_eq!(transport.requests.borrow()[0].api_key, "synthetic-key");
        assert_eq!(
            transport.requests.borrow()[0].query,
            "https://youtube.test/watch?v=synthetic"
        );
    }

    #[test]
    fn audio_scrape_terminal_transport_failures_are_typed() {
        for (error, expected) in [
            (TransportError::Timeout, AudioScrapeOutcome::Timeout),
            (TransportError::Connection, AudioScrapeOutcome::Connection),
        ] {
            let transport = FakeAudioTransport {
                responses: RefCell::new(vec![Err(error.clone()), Err(error.clone()), Err(error)]),
                requests: RefCell::new(Vec::new()),
            };
            assert_eq!(
                scrape_audio_with(&transport, "synthetic-key", "https://example.test", |_| {}).ok(),
                Some(expected)
            );
        }
        let transport = FakeAudioTransport {
            responses: RefCell::new(vec![Err(TransportError::Other(
                "synthetic failure".to_owned(),
            ))]),
            requests: RefCell::new(Vec::new()),
        };
        assert!(matches!(
            scrape_audio_with(&transport, "synthetic-key", "https://example.test", |_| {}),
            Err(TransportError::Other(detail)) if detail == "synthetic failure"
        ));
    }

    #[test]
    fn bounded_audio_reader_handles_success_size_limits_and_read_errors() {
        let fake = FakeAudioTransport {
            responses: RefCell::new(Vec::new()),
            requests: RefCell::new(Vec::new()),
        };
        assert!(fake.download_audio("https://media.example.test").is_err());
        assert_eq!(
            read_audio_response(206, Some(3), &mut Cursor::new(vec![1, 2, 3]), 3),
            Ok(super::BinaryHttpResponse {
                status_code: 206,
                body: vec![1, 2, 3],
            })
        );
        assert!(matches!(
            read_audio_response(200, Some(4), &mut Cursor::new(vec![1, 2, 3, 4]), 3),
            Err(TransportError::Other(detail)) if detail.contains("size limit")
        ));
        assert!(matches!(
            read_audio_response(200, None, &mut Cursor::new(vec![1, 2, 3, 4]), 3),
            Err(TransportError::Other(detail)) if detail.contains("size limit")
        ));
        struct BrokenReader;
        impl Read for BrokenReader {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(Error::other("synthetic read failure"))
            }
        }
        assert!(matches!(
            read_audio_response(200, None, &mut BrokenReader, 3),
            Err(TransportError::Other(detail)) if detail == "synthetic read failure"
        ));
    }

    #[test]
    fn text_cleanup_is_unicode_safe_at_zero_and_positive_limits() {
        assert_eq!(clean_text(&json!("  hola\n mundo  "), 20), "hola mundo");
        assert_eq!(clean_text(&json!("áéí"), 3), "áéí");
        assert_eq!(clean_text(&json!("áéí"), 2), "á…");
        assert_eq!(clean_text(&json!("text"), 0), "…");
        assert_eq!(clean_text(&json!(true), 20), "True");
        assert_eq!(clean_text(&json!(0), 20), "");
        assert_eq!(clean_text(&json!([]), 20), "");
        assert_eq!(clean_text(&json!({}), 20), "");
        assert_eq!(clean_text(&json!([1, 2]), 20), "[1,2]");
        assert_eq!(clean_text(&json!({"value": 1}), 20), r#"{"value":1}"#);
        assert_eq!(response_error("   "), "respuesta sin detalles");
        assert_eq!(response_error("plain failure"), "plain failure");
        assert!(extract_results(&json!({})).is_empty());
        assert!(extract_results(&json!({"data": {"web": {}}})).is_empty());
    }

    #[test]
    fn reqwest_transport_sends_authenticated_bounded_search_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
            let mut request = [0_u8; 8_192];
            let bytes = stream.read(&mut request).unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..bytes]);
            assert!(request.starts_with("POST /search HTTP/1.1"));
            assert!(request.contains("authorization: Bearer synthetic-key"));
            assert!(request.contains(
                r#"{"limit":5,"query":"synthetic query","sources":["web"],"timeout":60000}"#
            ));
            let body = r#"{"success":true,"data":[]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap_or_else(|_| unreachable!());
        });
        let transport =
            ReqwestFirecrawlTransport::with_search_url(&format!("http://{address}/search"))
                .unwrap_or_else(|_| unreachable!());
        let response = transport
            .post(&SearchRequest {
                query: "synthetic query".to_owned(),
                api_key: "synthetic-key".to_owned(),
            })
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(response.status_code, 200);
        assert_eq!(response.body, r#"{"success":true,"data":[]}"#);
        assert!(server.join().is_ok());
        let unavailable = ReqwestFirecrawlTransport::with_search_url("http://127.0.0.1:1/search")
            .unwrap_or_else(|_| unreachable!());
        assert!(
            unavailable
                .post(&SearchRequest {
                    query: "synthetic query".to_owned(),
                    api_key: "synthetic-key".to_owned(),
                })
                .is_err()
        );
        let malformed = ReqwestFirecrawlTransport::with_search_url("://invalid")
            .unwrap_or_else(|_| unreachable!());
        assert!(matches!(
            malformed.post(&SearchRequest {
                query: "synthetic query".to_owned(),
                api_key: "synthetic-key".to_owned(),
            }),
            Err(TransportError::Other(_))
        ));
    }

    #[test]
    fn reqwest_transport_sends_audio_scrape_payload_and_rejects_unsafe_download_urls() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
            let mut request = [0_u8; 8_192];
            let bytes = stream.read(&mut request).unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..bytes]);
            assert!(request.starts_with("POST /scrape HTTP/1.1"));
            assert!(request.contains("authorization: Bearer synthetic-key"));
            assert!(request.contains(
                r#"{"formats":["audio"],"url":"https://youtube.test/watch?v=synthetic"}"#
            ));
            let body =
                r#"{"success":true,"data":{"audio":"https://media.example.test/audio.mp3"}}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap_or_else(|_| unreachable!());
        });
        let transport =
            ReqwestFirecrawlTransport::with_scrape_url(&format!("http://{address}/scrape"))
                .unwrap_or_else(|_| unreachable!());
        let response = transport
            .scrape_audio(&SearchRequest {
                query: "https://youtube.test/watch?v=synthetic".to_owned(),
                api_key: "synthetic-key".to_owned(),
            })
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(response.status_code, 200);
        assert!(server.join().is_ok());
        assert!(
            transport
                .download_audio("http://example.test/audio.mp3")
                .is_err()
        );
        assert!(transport.download_audio("not a URL").is_err());
    }
}
