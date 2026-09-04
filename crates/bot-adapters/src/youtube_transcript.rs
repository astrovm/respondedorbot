//! YouTube caption providers used without downloading video or audio.

use std::io::Read;
use std::sync::OnceLock;
use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::{Value, json};
use thiserror::Error;

const SUPADATA_URL: &str = "https://api.supadata.ai/v1/transcript";
const APIFY_URL: &str =
    "https://api.apify.com/v2/acts/apihq~youtube-transcript-scraper/run-sync-get-dataset-items";
const RESPONSE_MAX_BYTES: u64 = 2_000_000;
const SUPADATA_JOB_POLL_ATTEMPTS: usize = 30;
const SUPADATA_JOB_POLL_INTERVAL: Duration = Duration::from_secs(1);
const SUPADATA_JOB_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TranscriptTransportError {
    #[error("transcript request timed out")]
    Timeout,
    #[error("transcript provider connection failed")]
    Connection,
    #[error("transcript transport failed: {0}")]
    Other(String),
}

pub trait YoutubeTranscriptTransport {
    fn supadata(
        &self,
        api_key: &str,
        video_url: &str,
    ) -> Result<HttpResponse, TranscriptTransportError>;

    fn supadata_job(
        &self,
        api_key: &str,
        job_id: &str,
    ) -> Result<HttpResponse, TranscriptTransportError>;

    fn apify(
        &self,
        api_key: &str,
        video_id: &str,
    ) -> Result<HttpResponse, TranscriptTransportError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptOutcome {
    Success { text: String, language: String },
    Unavailable { detail: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SupadataOutcome {
    Terminal(TranscriptOutcome),
    Pending { job_id: String },
}

pub struct ReqwestYoutubeTranscriptTransport {
    client: Client,
    supadata_url: String,
    apify_url: String,
}

impl ReqwestYoutubeTranscriptTransport {
    pub fn new() -> Result<Self, TranscriptTransportError> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(60))
                .build()
        })
        .map(|client| Self {
            client,
            supadata_url: SUPADATA_URL.to_owned(),
            apify_url: APIFY_URL.to_owned(),
        })
        .map_err(|error| TranscriptTransportError::Other(error.to_string()))
    }

    #[cfg(test)]
    fn with_urls(supadata_url: &str, apify_url: &str) -> Result<Self, TranscriptTransportError> {
        Self::new().map(|mut transport| {
            transport.supadata_url = supadata_url.to_owned();
            transport.apify_url = apify_url.to_owned();
            transport
        })
    }
}

impl YoutubeTranscriptTransport for ReqwestYoutubeTranscriptTransport {
    fn supadata(
        &self,
        api_key: &str,
        video_url: &str,
    ) -> Result<HttpResponse, TranscriptTransportError> {
        let response = self
            .client
            .get(&self.supadata_url)
            .header("x-api-key", api_key)
            .query(&[("url", video_url), ("mode", "native"), ("text", "true")])
            .send()
            .map_err(classify_reqwest_error)?;
        response_from(response)
    }

    fn apify(
        &self,
        api_key: &str,
        video_id: &str,
    ) -> Result<HttpResponse, TranscriptTransportError> {
        let response = self
            .client
            .post(&self.apify_url)
            .bearer_auth(api_key)
            .query(&[("clean", "true"), ("format", "json")])
            .json(&json!({"videoId": video_id, "metadata": true}))
            .send()
            .map_err(classify_reqwest_error)?;
        response_from(response)
    }

    fn supadata_job(
        &self,
        api_key: &str,
        job_id: &str,
    ) -> Result<HttpResponse, TranscriptTransportError> {
        let mut url = reqwest::Url::parse(&self.supadata_url)
            .map_err(|error| TranscriptTransportError::Other(error.to_string()))?;
        url.path_segments_mut()
            .map_err(|_| {
                TranscriptTransportError::Other(
                    "Supadata transcript URL cannot contain a job path".to_owned(),
                )
            })?
            .push(job_id);
        let response = self
            .client
            .get(url)
            .header("x-api-key", api_key)
            .timeout(SUPADATA_JOB_REQUEST_TIMEOUT)
            .send()
            .map_err(classify_reqwest_error)?;
        response_from(response)
    }
}

fn response_from(
    response: reqwest::blocking::Response,
) -> Result<HttpResponse, TranscriptTransportError> {
    let status_code = response.status().as_u16();
    if response
        .content_length()
        .is_some_and(|length| length > RESPONSE_MAX_BYTES)
    {
        return Err(TranscriptTransportError::Other(
            "transcript provider response exceeded the size limit".to_owned(),
        ));
    }
    let mut body = Vec::new();
    response
        .take(RESPONSE_MAX_BYTES + 1)
        .read_to_end(&mut body)
        .map_err(|error| TranscriptTransportError::Other(error.to_string()))?;
    if body.len() as u64 > RESPONSE_MAX_BYTES {
        return Err(TranscriptTransportError::Other(
            "transcript provider response exceeded the size limit".to_owned(),
        ));
    }
    String::from_utf8(body)
        .map(|body| HttpResponse { status_code, body })
        .map_err(|_| {
            TranscriptTransportError::Other("transcript provider returned invalid UTF-8".to_owned())
        })
}

fn classify_reqwest_error(error: reqwest::Error) -> TranscriptTransportError {
    if error.is_timeout() {
        TranscriptTransportError::Timeout
    } else if error.is_connect() {
        TranscriptTransportError::Connection
    } else {
        TranscriptTransportError::Other(error.to_string())
    }
}

fn parse_supadata(response: HttpResponse) -> SupadataOutcome {
    let payload = match provider_payload(&response) {
        Ok(payload) => payload,
        Err(detail) => {
            return SupadataOutcome::Terminal(TranscriptOutcome::Unavailable { detail });
        }
    };
    if response.status_code == 202 {
        return payload
            .get("jobId")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|job_id| !job_id.is_empty())
            .map_or_else(
                || {
                    SupadataOutcome::Terminal(TranscriptOutcome::Unavailable {
                        detail: "Supadata accepted a transcript job without an ID".to_owned(),
                    })
                },
                |job_id| SupadataOutcome::Pending {
                    job_id: job_id.to_owned(),
                },
            );
    }
    SupadataOutcome::Terminal(supadata_success(&payload))
}

fn parse_supadata_job(response: HttpResponse, job_id: &str) -> SupadataOutcome {
    let payload = match provider_payload(&response) {
        Ok(payload) => payload,
        Err(detail) => {
            return SupadataOutcome::Terminal(TranscriptOutcome::Unavailable { detail });
        }
    };
    match payload.get("status").and_then(Value::as_str) {
        Some("queued" | "active") => SupadataOutcome::Pending {
            job_id: job_id.to_owned(),
        },
        Some("completed") => SupadataOutcome::Terminal(supadata_success(&payload)),
        Some("failed") => SupadataOutcome::Terminal(TranscriptOutcome::Unavailable {
            detail: response_detail(&payload),
        }),
        Some(status) => SupadataOutcome::Terminal(TranscriptOutcome::Unavailable {
            detail: format!("Supadata returned unknown transcript job status: {status}"),
        }),
        None => SupadataOutcome::Terminal(TranscriptOutcome::Unavailable {
            detail: "Supadata returned no transcript job status".to_owned(),
        }),
    }
}

pub fn fetch_supadata_with<T, S>(
    transport: &T,
    api_key: &str,
    video_url: &str,
    mut sleep: S,
) -> Result<TranscriptOutcome, TranscriptTransportError>
where
    T: YoutubeTranscriptTransport,
    S: FnMut(Duration),
{
    let mut outcome = parse_supadata(transport.supadata(api_key, video_url)?);
    for _ in 0..SUPADATA_JOB_POLL_ATTEMPTS {
        let job_id = match outcome {
            SupadataOutcome::Terminal(terminal) => return Ok(terminal),
            SupadataOutcome::Pending { job_id } => job_id,
        };
        sleep(SUPADATA_JOB_POLL_INTERVAL);
        outcome = parse_supadata_job(transport.supadata_job(api_key, &job_id)?, &job_id);
    }
    Ok(match outcome {
        SupadataOutcome::Pending { .. } => TranscriptOutcome::Unavailable {
            detail: "Supadata transcript job did not finish within 30 seconds".to_owned(),
        },
        SupadataOutcome::Terminal(terminal) => terminal,
    })
}

fn supadata_success(payload: &Value) -> TranscriptOutcome {
    let text = transcript_text(payload.get("content"));
    if text.is_empty() {
        return TranscriptOutcome::Unavailable {
            detail: "Supadata returned no native captions".to_owned(),
        };
    }
    TranscriptOutcome::Success {
        text,
        language: language(payload),
    }
}

fn transcript_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.trim().to_owned(),
        Some(Value::Array(segments)) => segments
            .iter()
            .filter_map(|segment| segment.get("text").and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

#[must_use]
pub fn parse_apify(response: HttpResponse) -> TranscriptOutcome {
    let payload = match provider_payload(&response) {
        Ok(payload) => payload,
        Err(detail) => return TranscriptOutcome::Unavailable { detail },
    };
    let item = payload
        .as_array()
        .and_then(|items| items.first())
        .unwrap_or(&Value::Null);
    if item.get("success").and_then(Value::as_bool) == Some(false) {
        return TranscriptOutcome::Unavailable {
            detail: response_detail(item),
        };
    }
    let text = item
        .get("transcript")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|segment| segment.get("text").and_then(Value::as_str))
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        return TranscriptOutcome::Unavailable {
            detail: "Apify returned no native captions".to_owned(),
        };
    }
    TranscriptOutcome::Success {
        text,
        language: language(item),
    }
}

fn provider_payload(response: &HttpResponse) -> Result<Value, String> {
    let payload = serde_json::from_str::<Value>(&response.body).map_err(|_| {
        format!(
            "transcript provider returned invalid JSON (HTTP {})",
            response.status_code
        )
    })?;
    if !(200..300).contains(&response.status_code) {
        return Err(format!(
            "transcript provider returned HTTP {}: {}",
            response.status_code,
            response_detail(&payload)
        ));
    }
    Ok(payload)
}

fn language(payload: &Value) -> String {
    payload
        .get("lang")
        .or_else(|| payload.get("language"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn response_detail(payload: &Value) -> String {
    let detail = payload
        .get("error")
        .or_else(|| payload.get("message"))
        .and_then(|value| {
            value.as_str().map(str::to_owned).or_else(|| {
                value
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "transcript unavailable".to_owned());
    detail
        .split_whitespace()
        .take(80)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use serde_json::json;

    use super::*;

    #[test]
    fn parses_supadata_native_text_and_failures() {
        assert_eq!(
            parse_supadata(HttpResponse {
                status_code: 200,
                body: json!({"content": " native transcript ", "lang": "en"}).to_string(),
            }),
            SupadataOutcome::Terminal(TranscriptOutcome::Success {
                text: "native transcript".to_owned(),
                language: "en".to_owned(),
            })
        );
        assert!(matches!(
            parse_supadata(HttpResponse {
                status_code: 429,
                body: json!({"error": "quota reached"}).to_string(),
            }),
            SupadataOutcome::Terminal(TranscriptOutcome::Unavailable { detail })
                if detail.contains("quota reached")
        ));
        assert!(matches!(
            parse_supadata(HttpResponse {
                status_code: 200,
                body: json!({"content": ""}).to_string(),
            }),
            SupadataOutcome::Terminal(TranscriptOutcome::Unavailable { detail })
                if detail.contains("no native captions")
        ));
        assert_eq!(
            parse_supadata(HttpResponse {
                status_code: 202,
                body: json!({"jobId": "synthetic-job"}).to_string(),
            }),
            SupadataOutcome::Pending {
                job_id: "synthetic-job".to_owned(),
            }
        );
        assert!(matches!(
            parse_supadata(HttpResponse {
                status_code: 202,
                body: json!({}).to_string(),
            }),
            SupadataOutcome::Terminal(TranscriptOutcome::Unavailable { detail })
                if detail.contains("without an ID")
        ));
    }

    #[test]
    fn parses_supadata_job_states_and_segment_results() {
        for status in ["queued", "active"] {
            assert_eq!(
                parse_supadata_job(
                    HttpResponse {
                        status_code: 200,
                        body: json!({"status": status}).to_string(),
                    },
                    "synthetic-job",
                ),
                SupadataOutcome::Pending {
                    job_id: "synthetic-job".to_owned(),
                }
            );
        }
        assert_eq!(
            parse_supadata_job(
                HttpResponse {
                    status_code: 200,
                    body: json!({
                        "status": "completed",
                        "content": [{"text": " first "}, {"text": "second"}],
                        "lang": "es"
                    })
                    .to_string(),
                },
                "synthetic-job",
            ),
            SupadataOutcome::Terminal(TranscriptOutcome::Success {
                text: "first\nsecond".to_owned(),
                language: "es".to_owned(),
            })
        );
        assert!(matches!(
            parse_supadata_job(
                HttpResponse {
                    status_code: 200,
                    body: json!({
                        "status": "failed",
                        "error": {"message": "native captions unavailable"}
                    })
                    .to_string(),
                },
                "synthetic-job",
            ),
            SupadataOutcome::Terminal(TranscriptOutcome::Unavailable { detail })
                if detail == "native captions unavailable"
        ));
        for body in [json!({"status": "unexpected"}), json!({})] {
            assert!(matches!(
                parse_supadata_job(
                    HttpResponse {
                        status_code: 200,
                        body: body.to_string(),
                    },
                    "synthetic-job",
                ),
                SupadataOutcome::Terminal(TranscriptOutcome::Unavailable { .. })
            ));
        }
    }

    struct PollTransport {
        start: RefCell<Option<Result<HttpResponse, TranscriptTransportError>>>,
        jobs: RefCell<Vec<Result<HttpResponse, TranscriptTransportError>>>,
    }

    impl YoutubeTranscriptTransport for PollTransport {
        fn supadata(&self, _: &str, _: &str) -> Result<HttpResponse, TranscriptTransportError> {
            self.start.borrow_mut().take().unwrap_or_else(|| {
                Err(TranscriptTransportError::Other(
                    "unexpected start request".to_owned(),
                ))
            })
        }

        fn supadata_job(&self, _: &str, _: &str) -> Result<HttpResponse, TranscriptTransportError> {
            self.jobs.borrow_mut().remove(0)
        }

        fn apify(&self, _: &str, _: &str) -> Result<HttpResponse, TranscriptTransportError> {
            Err(TranscriptTransportError::Other(
                "unexpected Apify request".to_owned(),
            ))
        }
    }

    fn json_response(
        status_code: u16,
        body: Value,
    ) -> Result<HttpResponse, TranscriptTransportError> {
        Ok(HttpResponse {
            status_code,
            body: body.to_string(),
        })
    }

    #[test]
    fn polls_supadata_jobs_until_completion() {
        let transport = PollTransport {
            start: RefCell::new(Some(json_response(202, json!({"jobId": "synthetic-job"})))),
            jobs: RefCell::new(vec![
                json_response(200, json!({"status": "queued"})),
                json_response(200, json!({"status": "active"})),
                json_response(
                    200,
                    json!({"status": "completed", "content": "transcript", "lang": "en"}),
                ),
            ]),
        };
        let delays = RefCell::new(Vec::new());
        let outcome = fetch_supadata_with(
            &transport,
            "synthetic-key",
            "https://youtu.be/synthetic",
            |delay| delays.borrow_mut().push(delay),
        );
        assert_eq!(
            outcome,
            Ok(TranscriptOutcome::Success {
                text: "transcript".to_owned(),
                language: "en".to_owned(),
            })
        );
        assert_eq!(*delays.borrow(), vec![SUPADATA_JOB_POLL_INTERVAL; 3]);
    }

    #[test]
    fn bounded_supadata_polling_returns_an_explicit_timeout() {
        let transport = PollTransport {
            start: RefCell::new(Some(json_response(202, json!({"jobId": "synthetic-job"})))),
            jobs: RefCell::new(
                (0..SUPADATA_JOB_POLL_ATTEMPTS)
                    .map(|_| json_response(200, json!({"status": "active"})))
                    .collect(),
            ),
        };
        assert!(matches!(
            fetch_supadata_with(
                &transport,
                "synthetic-key",
                "https://youtu.be/synthetic",
                |_| {}
            ),
            Ok(TranscriptOutcome::Unavailable { detail }) if detail.contains("within 30 seconds")
        ));
    }

    #[test]
    fn parses_apify_segments_without_translation() {
        let outcome = parse_apify(HttpResponse {
            status_code: 201,
            body: json!([{
                "success": true,
                "language": "ja",
                "transcript": [{"text": "日本語"}, {"text": "の字幕"}]
            }])
            .to_string(),
        });
        assert_eq!(
            outcome,
            TranscriptOutcome::Success {
                text: "日本語\nの字幕".to_owned(),
                language: "ja".to_owned(),
            }
        );
        assert!(matches!(
            parse_apify(HttpResponse {
                status_code: 200,
                body: json!([]).to_string(),
            }),
            TranscriptOutcome::Unavailable { .. }
        ));
        assert!(matches!(
            parse_apify(HttpResponse {
                status_code: 200,
                body: json!([{
                    "success": false,
                    "error": {"message": "captions disabled"}
                }])
                .to_string(),
            }),
            TranscriptOutcome::Unavailable { detail } if detail == "captions disabled"
        ));
        assert!(matches!(
            parse_apify(HttpResponse {
                status_code: 503,
                body: json!({"message": "provider unavailable"}).to_string(),
            }),
            TranscriptOutcome::Unavailable { detail } if detail.contains("HTTP 503")
        ));
        assert!(matches!(
            parse_apify(HttpResponse {
                status_code: 200,
                body: "not-json".to_owned(),
            }),
            TranscriptOutcome::Unavailable { detail } if detail.contains("invalid JSON")
        ));
    }

    #[test]
    fn transports_send_provider_specific_auth_and_native_only_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            for expected in [
                "GET /supadata?",
                "GET /supadata/synthetic-job HTTP/1.1",
                "POST /apify?",
            ] {
                let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
                let mut request = [0_u8; 8_192];
                let bytes = stream.read(&mut request).unwrap_or_default();
                let request = String::from_utf8_lossy(&request[..bytes]);
                assert!(request.starts_with(expected));
                if expected == "GET /supadata?" {
                    assert!(request.contains("x-api-key: synthetic-supadata"));
                    assert!(request.contains("mode=native"));
                    assert!(request.contains("text=true"));
                } else if expected.starts_with("GET") {
                    assert!(request.contains("x-api-key: synthetic-supadata"));
                } else {
                    assert!(request.contains("authorization: Bearer synthetic-apify"));
                    assert!(request.contains(r#"{"metadata":true,"videoId":"video123"}"#));
                }
                let body = "{}";
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap_or_else(|_| unreachable!());
            }
        });
        let transport = ReqwestYoutubeTranscriptTransport::with_urls(
            &format!("http://{address}/supadata"),
            &format!("http://{address}/apify"),
        )
        .unwrap_or_else(|_| unreachable!());
        assert!(
            transport
                .supadata("synthetic-supadata", "https://youtu.be/video123")
                .is_ok()
        );
        assert!(
            transport
                .supadata_job("synthetic-supadata", "synthetic-job")
                .is_ok()
        );
        assert!(transport.apify("synthetic-apify", "video123").is_ok());
        assert!(server.join().is_ok());

        let unavailable = ReqwestYoutubeTranscriptTransport::with_urls(
            "http://127.0.0.1:1/supadata",
            "http://127.0.0.1:1/apify",
        )
        .unwrap_or_else(|_| unreachable!());
        assert!(matches!(
            unavailable.supadata("synthetic", "https://youtu.be/video123"),
            Err(TranscriptTransportError::Connection)
        ));
        let malformed = ReqwestYoutubeTranscriptTransport::with_urls("://invalid", "://invalid")
            .unwrap_or_else(|_| unreachable!());
        assert!(matches!(
            malformed.apify("synthetic", "video123"),
            Err(TranscriptTransportError::Other(_))
        ));
        assert!(matches!(
            malformed.supadata_job("synthetic", "synthetic-job"),
            Err(TranscriptTransportError::Other(_))
        ));
    }

    #[test]
    fn transport_rejects_oversized_and_non_utf8_responses() {
        for (content_length, body, expected) in [
            (Some(RESPONSE_MAX_BYTES + 1), Vec::new(), "size limit"),
            (Some(1), vec![0xff], "invalid UTF-8"),
            (
                None,
                vec![b'x'; (RESPONSE_MAX_BYTES + 1) as usize],
                "size limit",
            ),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
            let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
            let server = thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
                let mut request = [0_u8; 1_024];
                let _bytes = stream.read(&mut request).unwrap_or_default();
                let length = content_length
                    .map(|length| format!("Content-Length: {length}\r\n"))
                    .unwrap_or_default();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\n{length}Connection: close\r\n\r\n"
                )
                .unwrap_or_else(|_| unreachable!());
                stream.write_all(&body).unwrap_or_else(|_| unreachable!());
            });
            let transport = ReqwestYoutubeTranscriptTransport::with_urls(
                &format!("http://{address}/transcript"),
                &format!("http://{address}/transcript"),
            )
            .unwrap_or_else(|_| unreachable!());
            let result = transport.supadata("synthetic", "https://youtu.be/video123");
            assert!(matches!(
                result,
                Err(TranscriptTransportError::Other(detail)) if detail.contains(expected)
            ));
            assert!(server.join().is_ok());
        }
    }

    #[test]
    fn transport_classifies_request_timeouts() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
            let mut request = [0_u8; 1_024];
            let _bytes = stream.read(&mut request).unwrap_or_default();
            thread::sleep(Duration::from_millis(100));
        });
        let transport = ReqwestYoutubeTranscriptTransport {
            client: Client::builder()
                .timeout(Duration::from_millis(10))
                .build()
                .unwrap_or_else(|_| unreachable!()),
            supadata_url: format!("http://{address}/transcript"),
            apify_url: format!("http://{address}/transcript"),
        };
        assert!(matches!(
            transport.supadata("synthetic", "https://youtu.be/video123"),
            Err(TranscriptTransportError::Timeout)
        ));
        assert!(server.join().is_ok());
    }
}
