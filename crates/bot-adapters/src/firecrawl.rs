//! Blocking Firecrawl search adapter for the synchronous migration runtime.

use std::thread;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};
use thiserror::Error;

const SEARCH_URL: &str = "https://api.firecrawl.dev/v2/search";
const MAX_RESULTS: usize = 5;
const MAX_ATTEMPTS: usize = 3;
const API_TIMEOUT_MS: u64 = 60_000;
const MAX_DESCRIPTION_CHARS: usize = 1_200;

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
}

impl ReqwestFirecrawlTransport {
    pub fn new() -> Result<Self, TransportError> {
        Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(75))
            .build()
            .map(|client| Self { client })
            .map_err(|error| TransportError::Other(error.to_string()))
    }
}

impl FirecrawlTransport for ReqwestFirecrawlTransport {
    fn post(&self, request: &SearchRequest) -> Result<HttpResponse, TransportError> {
        let response = self
            .client
            .post(SEARCH_URL)
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
    for attempt in 0..MAX_ATTEMPTS {
        match transport.post(&request) {
            Ok(response) => {
                let retryable =
                    matches!(response.status_code, 408 | 409 | 429) || response.status_code >= 500;
                if retryable && attempt + 1 < MAX_ATTEMPTS {
                    sleep(Duration::from_secs(1 << attempt));
                    continue;
                }
                return Ok(parse_response(response, query));
            }
            Err(error @ (TransportError::Timeout | TransportError::Connection)) => {
                if attempt + 1 == MAX_ATTEMPTS {
                    let outcome = match error {
                        TransportError::Timeout => SearchOutcome::Timeout,
                        TransportError::Connection => SearchOutcome::Connection,
                        TransportError::Other(detail) => {
                            return Err(TransportError::Other(detail));
                        }
                    };
                    return Ok(outcome);
                }
                sleep(Duration::from_secs(1 << attempt));
            }
            Err(error) => return Err(error),
        }
    }
    Err(TransportError::Other(
        "Firecrawl returned no response".to_owned(),
    ))
}

pub fn search(api_key: &str, query: &str) -> Result<SearchOutcome, TransportError> {
    let transport = ReqwestFirecrawlTransport::new()?;
    search_with(&transport, api_key, query, thread::sleep)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{
        FirecrawlTransport, HttpResponse, SearchOutcome, SearchRequest, TransportError, clean_text,
        search_with,
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
    fn text_cleanup_is_unicode_safe_at_zero_and_positive_limits() {
        assert_eq!(clean_text(&json!("  hola\n mundo  "), 20), "hola mundo");
        assert_eq!(clean_text(&json!("áéí"), 3), "áéí");
        assert_eq!(clean_text(&json!("áéí"), 2), "á…");
        assert_eq!(clean_text(&json!("text"), 0), "…");
    }
}
