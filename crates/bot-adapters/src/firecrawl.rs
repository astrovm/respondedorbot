//! Blocking Firecrawl search adapter.

use std::time::Duration;

use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::OnceLock;
use thiserror::Error;

use crate::http_client::error_chain;

const SEARCH_URL: &str = "https://api.firecrawl.dev/v2/search";
/// Firecrawl applies the limit to each source, so a search returns up to this
/// many news results plus this many web results.
const MAX_RESULTS: usize = 5;
const MAX_ATTEMPTS: usize = 3;
const API_TIMEOUT_MS: u64 = 60_000;
const MAX_DESCRIPTION_CHARS: usize = 1_200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchRequest {
    pub query: String,
    pub api_key: String,
    pub sources: &'static [&'static str],
}

/// Web results rank by relevance, so a story from the last few hours loses to
/// older pages; news results bring it in with a date.
const NEWS_AND_WEB: &[&str] = &["news", "web"];
const NEWS: &[&str] = &["news"];

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
    /// Publication age of a news result, such as "2 hours ago".
    #[serde(skip_serializing_if = "String::is_empty")]
    pub date: String,
    #[serde(skip)]
    pub news: bool,
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
        })
        .map_err(classify_reqwest_error)
    }

    #[cfg(test)]
    fn with_search_url(search_url: &str) -> Result<Self, TransportError> {
        Self::new().map(|mut transport| {
            transport.search_url = search_url.to_owned();
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
                "sources": request.sources,
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
        TransportError::Other(error_chain(&error))
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
    let Some(sources) = data.as_object() else {
        return source_results(data);
    };
    ["news", "web"]
        .into_iter()
        .filter_map(|source| Some((source, sources.get(source)?)))
        .flat_map(|(source, raw_results)| {
            let mut results = source_results(raw_results);
            for result in &mut results {
                result.news = source == "news";
            }
            results
        })
        .collect()
}

fn source_results(raw_results: &Value) -> Vec<SearchResult> {
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
                // News results name their summary `snippet`.
                description: clean_text(
                    object
                        .get("description")
                        .or_else(|| object.get("snippet"))
                        .unwrap_or(&Value::Null),
                    MAX_DESCRIPTION_CHARS,
                ),
                date: clean_text(object.get("date").unwrap_or(&Value::Null), 100),
                news: false,
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
    let outcome = post_with_retries(
        transport,
        &SearchRequest {
            query: query.to_owned(),
            api_key: api_key.to_owned(),
            sources: NEWS_AND_WEB,
        },
        &sleep,
    )?;
    let SearchOutcome::Success {
        query: outcome_query,
        results,
        credits_used,
        request_id,
    } = outcome
    else {
        return Ok(outcome);
    };
    let news_query = news_query(query);
    if results.iter().any(|result| result.news) || news_query.is_empty() || news_query == query {
        return Ok(SearchOutcome::Success {
            query: outcome_query,
            results,
            credits_used,
            request_id,
        });
    }
    // News search needs every word to match, so dates the model adds, such
    // as "October 2026", leave it empty. Ask again for news without them.
    let retry = post_with_retries(
        transport,
        &SearchRequest {
            query: news_query,
            api_key: api_key.to_owned(),
            sources: NEWS,
        },
        &sleep,
    );
    let Ok(SearchOutcome::Success {
        results: news,
        credits_used: news_credits,
        ..
    }) = retry
    else {
        return Ok(SearchOutcome::Success {
            query: outcome_query,
            results,
            credits_used,
            request_id,
        });
    };
    Ok(SearchOutcome::Success {
        query: outcome_query,
        results: news.into_iter().chain(results).collect(),
        credits_used: add_credits(&credits_used, &news_credits),
        request_id,
    })
}

fn post_with_retries<T, S>(
    transport: &T,
    request: &SearchRequest,
    sleep: &S,
) -> Result<SearchOutcome, TransportError>
where
    T: FirecrawlTransport,
    S: Fn(Duration),
{
    let mut attempt = 0;
    loop {
        match transport.post(request) {
            Ok(response) => {
                let retryable =
                    matches!(response.status_code, 408 | 409 | 429) || response.status_code >= 500;
                if retryable && attempt + 1 < MAX_ATTEMPTS {
                    sleep(Duration::from_secs(1 << attempt));
                    attempt += 1;
                    continue;
                }
                return Ok(parse_response(response, &request.query));
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

/// Drops years, month names, day numbers next to a month, and the words that
/// only frame a date ("news", "today", "de"), in English and Spanish.
#[must_use]
pub fn news_query(query: &str) -> String {
    let words = query.split_whitespace().collect::<Vec<_>>();
    let bare_words = words
        .iter()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .collect::<Vec<_>>();
    let numeric = |index: usize| {
        bare_words
            .get(index)
            .is_some_and(|word| is_day(word) || is_year(word))
    };
    // "may" is only a month next to a number, so "binance may be hacked" keeps it.
    let months = bare_words
        .iter()
        .enumerate()
        .map(|(index, word)| {
            is_month(word)
                || (word == "may" && ((index > 0 && numeric(index - 1)) || numeric(index + 1)))
        })
        .collect::<Vec<_>>();
    let has_month = months.contains(&true);
    let dated = bare_words
        .iter()
        .zip(&months)
        .map(|(word, month)| *month || is_year(word) || (has_month && is_day(word)))
        .collect::<Vec<_>>();
    words
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            let word = &bare_words[*index];
            let next_to_date =
                (*index > 0 && dated[index - 1]) || dated.get(index + 1) == Some(&true);
            !dated[*index]
                && !matches!(
                    word.as_str(),
                    "news" | "noticias" | "today" | "hoy" | "latest" | "últimas" | "ultimas"
                )
                && !(next_to_date && matches!(word.as_str(), "de" | "del" | "of" | "in" | "en"))
        })
        .map(|(_, word)| *word)
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_year(word: &str) -> bool {
    word.len() == 4
        && (word.starts_with("19") || word.starts_with("20"))
        && word.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_day(word: &str) -> bool {
    let digits = word.trim_end_matches(|character: char| character.is_ascii_alphabetic());
    let suffix = &word[digits.len()..];
    matches!(suffix, "" | "st" | "nd" | "rd" | "th")
        && digits
            .parse::<u8>()
            .is_ok_and(|day| (1..=31).contains(&day))
}

fn is_month(word: &str) -> bool {
    matches!(
        word,
        "january"
            | "february"
            | "march"
            | "april"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
            | "enero"
            | "febrero"
            | "marzo"
            | "abril"
            | "mayo"
            | "junio"
            | "julio"
            | "agosto"
            | "septiembre"
            | "setiembre"
            | "octubre"
            | "noviembre"
            | "diciembre"
    )
}

fn add_credits(first: &Value, second: &Value) -> Value {
    let credits = |value: &Value| value.as_u64().or_else(|| value.as_str()?.parse().ok());
    match (credits(first), credits(second)) {
        (Some(first), Some(second)) => json!(first + second),
        (None, Some(_)) => second.clone(),
        _ => first.clone(),
    }
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::{
        FirecrawlTransport, HttpResponse, ReqwestFirecrawlTransport, SearchOutcome, SearchRequest,
        TransportError, clean_text, extract_results, response_error, search_with,
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

    /// Every scenario shares one `search_with` instantiation; retry delays
    /// are recorded instead of slept.
    fn search(
        transport: &FakeTransport,
        api_key: &str,
        delays: &RefCell<Vec<u64>>,
    ) -> Result<SearchOutcome, TransportError> {
        let record: &dyn Fn(std::time::Duration) =
            &|delay| delays.borrow_mut().push(delay.as_secs());
        search_with(transport, api_key, "query", record)
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
        let outcome = search(&transport, "synthetic-key", &RefCell::default());
        assert_eq!(
            outcome,
            Ok(SearchOutcome::Success {
                query: "query".to_owned(),
                results: vec![super::SearchResult {
                    title: "Example Title".to_owned(),
                    url: "https://example.com".to_owned(),
                    description: "summary".to_owned(),
                    date: String::new(),
                    news: false,
                }],
                credits_used: json!(2),
                request_id: json!("request-1"),
            })
        );
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
        let outcome = search(&transport, "key", &delays);
        assert!(matches!(outcome, Ok(SearchOutcome::Success { .. })));
        assert_eq!(*delays.borrow(), vec![1, 2]);
        assert_eq!(transport.requests.borrow().len(), 3);
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
                search(&transport, "key", &RefCell::default()).ok(),
                Some(expected),
            );
        }
        let http = FakeTransport {
            responses: RefCell::new(vec![response(400, json!({"error": "bad request"}))]),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            search(&http, "key", &RefCell::default()).ok(),
            Some(SearchOutcome::HttpError {
                status_code: 400,
                detail: "bad request".to_owned(),
            }),
        );
        let unexpected = FakeTransport {
            responses: RefCell::new(vec![Err(TransportError::Other("tls".to_owned()))]),
            requests: RefCell::new(Vec::new()),
        };
        let delays = RefCell::new(Vec::new());
        assert_eq!(
            search(&unexpected, "key", &delays),
            Err(TransportError::Other("tls".to_owned()))
        );
        assert!(delays.borrow().is_empty());
        assert_eq!(unexpected.requests.borrow().len(), 1);
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
            search(&invalid, "key", &RefCell::default()).ok(),
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
            search(&rejected, "key", &RefCell::default()).ok(),
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
    fn news_results_come_first_with_their_date_and_snippet() {
        let results = extract_results(&json!({"data": {
            "web": [{"title": "Older page", "url": "https://example.com/old", "description": "web summary"}],
            "news": [{"title": "Fresh story", "url": "https://example.com/new", "snippet": "news summary", "date": "2 hours ago", "position": 1}]
        }}));
        assert_eq!(
            results,
            vec![
                super::SearchResult {
                    title: "Fresh story".to_owned(),
                    url: "https://example.com/new".to_owned(),
                    description: "news summary".to_owned(),
                    date: "2 hours ago".to_owned(),
                    news: true,
                },
                super::SearchResult {
                    title: "Older page".to_owned(),
                    url: "https://example.com/old".to_owned(),
                    description: "web summary".to_owned(),
                    date: String::new(),
                    news: false,
                },
            ]
        );
        assert_eq!(
            serde_json::to_value(&results[1]).ok(),
            Some(
                json!({"title": "Older page", "url": "https://example.com/old", "description": "web summary"})
            )
        );
        let legacy =
            extract_results(&json!({"data": [{"url": "https://example.com", "title": "Flat"}]}));
        assert_eq!(legacy.len(), 1);
    }

    #[test]
    fn news_queries_drop_dates_and_the_words_that_frame_them() {
        for (query, expected) in [
            ("binance hack news October 2026", "binance hack"),
            ("binance hack octubre 2026", "binance hack"),
            (
                "Binance BNB drop news today October 29 2026 hack",
                "Binance BNB drop hack",
            ),
            (
                "noticias del 29 de octubre de 2026 sobre binance",
                "sobre binance",
            ),
            ("binance hack 29th, november", "binance hack"),
            ("top 10 crypto hacks 2026", "top 10 crypto hacks"),
            ("binance hack", "binance hack"),
            ("may 5 binance", "binance"),
            ("binance may 2026", "binance"),
            ("binance may be hacked", "binance may be hacked"),
            ("may", "may"),
            ("news today", ""),
            ("binance 32 october 0 2099 1899", "binance 32 0 1899"),
        ] {
            assert_eq!(super::news_query(query), expected, "{query}");
        }
    }

    fn news_payload(
        credits: Value,
        news: &[&str],
        web: &[&str],
    ) -> Result<HttpResponse, TransportError> {
        let entries = |urls: &[&str]| {
            urls.iter()
                .map(|url| json!({"title": "Synthetic", "url": url, "snippet": "s", "date": "1 hour ago"}))
                .collect::<Vec<_>>()
        };
        response(
            200,
            json!({"success": true, "id": "request-1", "creditsUsed": credits,
                "data": {"news": entries(news), "web": entries(web)}}),
        )
    }

    fn search_query(
        transport: &FakeTransport,
        query: &str,
    ) -> Result<SearchOutcome, TransportError> {
        search_with(transport, "key", query, |_| {})
    }

    fn urls(outcome: &Result<SearchOutcome, TransportError>) -> Vec<String> {
        match outcome {
            Ok(SearchOutcome::Success { results, .. }) => {
                results.iter().map(|result| result.url.clone()).collect()
            }
            _ => Vec::new(),
        }
    }

    #[test]
    fn dated_queries_without_news_ask_again_for_news_without_the_dates() {
        let transport = FakeTransport {
            responses: RefCell::new(vec![
                news_payload(json!(2), &[], &["https://example.com/old"]),
                news_payload(json!("2"), &["https://example.com/new"], &[]),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        let outcome = search_query(&transport, "binance hack October 2026");
        assert_eq!(
            urls(&outcome),
            ["https://example.com/new", "https://example.com/old"]
        );
        assert!(matches!(
            &outcome,
            Ok(SearchOutcome::Success { query, credits_used, request_id, .. })
                if query == "binance hack October 2026"
                    && *credits_used == json!(4)
                    && *request_id == json!("request-1")
        ));
        let requests = transport.requests.borrow();
        assert_eq!(requests[0].query, "binance hack October 2026");
        assert_eq!(requests[0].sources, ["news", "web"]);
        assert_eq!(requests[1].query, "binance hack");
        assert_eq!(requests[1].sources, ["news"]);
    }

    #[test]
    fn news_retry_only_runs_when_it_can_help_and_never_loses_the_first_results() {
        for (query, first, expected_requests) in [
            (
                "binance hack 2026",
                news_payload(json!(2), &["https://example.com/new"], &[]),
                1,
            ),
            (
                "binance hack",
                news_payload(json!(2), &[], &["https://example.com/old"]),
                1,
            ),
            (
                "October 2026",
                news_payload(json!(2), &[], &["https://example.com/old"]),
                1,
            ),
            ("binance 2026", response(500, json!({"error": "down"})), 3),
        ] {
            let transport = FakeTransport {
                responses: RefCell::new(vec![first.clone(), first.clone(), first]),
                requests: RefCell::new(Vec::new()),
            };
            let outcome = search_query(&transport, query);
            assert!(outcome.is_ok(), "{query}");
            assert_eq!(
                transport.requests.borrow().len(),
                expected_requests,
                "{query}"
            );
        }

        for retry in [
            response(400, json!({"error": "bad"})),
            Err(TransportError::Other("tls".to_owned())),
        ] {
            let transport = FakeTransport {
                responses: RefCell::new(vec![
                    news_payload(json!(2), &[], &["https://example.com/old"]),
                    retry,
                ]),
                requests: RefCell::new(Vec::new()),
            };
            let outcome = search_query(&transport, "binance hack 2026");
            assert_eq!(urls(&outcome), ["https://example.com/old"]);
            assert!(matches!(
                outcome,
                Ok(SearchOutcome::Success { credits_used, .. }) if credits_used == json!(2)
            ));
        }
    }

    #[test]
    fn credits_add_up_when_both_are_known() {
        use super::add_credits;
        assert_eq!(add_credits(&json!(2), &json!("2")), json!(4));
        assert_eq!(add_credits(&Value::Null, &json!(2)), json!(2));
        assert_eq!(add_credits(&json!(2), &Value::Null), json!(2));
        assert_eq!(add_credits(&Value::Null, &Value::Null), Value::Null);
    }

    #[test]
    fn reqwest_transport_sends_authenticated_bounded_search_payload() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 8_192];
            let bytes = stream.read(&mut request).unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..bytes]);
            assert!(request.starts_with("POST /search HTTP/1.1"));
            assert!(request.contains("authorization: Bearer synthetic-key"));
            assert!(request.contains(
                r#"{"limit":5,"query":"synthetic query","sources":["news","web"],"timeout":60000}"#
            ));
            let body = r#"{"success":true,"data":[]}"#;
            let written = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            written?;
            Ok(())
        });
        let transport =
            ReqwestFirecrawlTransport::with_search_url(&format!("http://{address}/search"))?;
        let response = transport.post(&SearchRequest {
            query: "synthetic query".to_owned(),
            api_key: "synthetic-key".to_owned(),
            sources: super::NEWS_AND_WEB,
        })?;
        assert_eq!(response.status_code, 200);
        assert_eq!(response.body, r#"{"success":true,"data":[]}"#);
        assert!(matches!(server.join(), Ok(Ok(()))));
        let unavailable = ReqwestFirecrawlTransport::with_search_url("http://127.0.0.1:1/search")?;
        assert!(
            unavailable
                .post(&SearchRequest {
                    query: "synthetic query".to_owned(),
                    api_key: "synthetic-key".to_owned(),
                    sources: super::NEWS_AND_WEB,
                })
                .is_err()
        );
        let malformed = ReqwestFirecrawlTransport::with_search_url("://invalid")?;
        assert!(matches!(
            malformed.post(&SearchRequest {
                query: "synthetic query".to_owned(),
                api_key: "synthetic-key".to_owned(),
                sources: super::NEWS,
            }),
            Err(TransportError::Other(_))
        ));
        Ok(())
    }

    #[test]
    fn reqwest_failures_are_classified_by_cause() {
        use crate::web_fetch::reqwest_error_fixtures as fixtures;
        assert_eq!(
            fixtures::timeout().map(super::classify_reqwest_error),
            Some(super::TransportError::Timeout)
        );
        assert_eq!(
            fixtures::connection().map(super::classify_reqwest_error),
            Some(super::TransportError::Connection)
        );
        assert!(matches!(
            fixtures::request().map(super::classify_reqwest_error),
            Some(super::TransportError::Other(detail)) if detail.contains("builder error")
        ));
    }
}
