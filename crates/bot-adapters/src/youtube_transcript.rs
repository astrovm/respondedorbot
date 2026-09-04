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

#[must_use]
pub fn parse_supadata(response: HttpResponse) -> TranscriptOutcome {
    let payload = match provider_payload(&response) {
        Ok(payload) => payload,
        Err(detail) => return TranscriptOutcome::Unavailable { detail },
    };
    let text = payload
        .get("content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty());
    let Some(text) = text else {
        return TranscriptOutcome::Unavailable {
            detail: "Supadata returned no native captions".to_owned(),
        };
    };
    TranscriptOutcome::Success {
        text: text.to_owned(),
        language: language(&payload),
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
            TranscriptOutcome::Success {
                text: "native transcript".to_owned(),
                language: "en".to_owned(),
            }
        );
        assert!(matches!(
            parse_supadata(HttpResponse {
                status_code: 429,
                body: json!({"error": "quota reached"}).to_string(),
            }),
            TranscriptOutcome::Unavailable { detail } if detail.contains("quota reached")
        ));
        assert!(matches!(
            parse_supadata(HttpResponse {
                status_code: 200,
                body: json!({"content": ""}).to_string(),
            }),
            TranscriptOutcome::Unavailable { detail } if detail.contains("no native captions")
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
            for expected in ["GET /supadata?", "POST /apify?"] {
                let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
                let mut request = [0_u8; 8_192];
                let bytes = stream.read(&mut request).unwrap_or_default();
                let request = String::from_utf8_lossy(&request[..bytes]);
                assert!(request.starts_with(expected));
                if expected.starts_with("GET") {
                    assert!(request.contains("x-api-key: synthetic-supadata"));
                    assert!(request.contains("mode=native"));
                    assert!(request.contains("text=true"));
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
