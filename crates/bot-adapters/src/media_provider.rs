//! Typed image-description and audio-transcription provider boundaries.

use std::collections::BTreeMap;
use std::io::Read;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::blocking::Client;
use reqwest::blocking::multipart::{Form, Part};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::openrouter_chat::{
    ChatCompletion, ChatCompletionRequest, ChatMessage, ChatRole, OpenRouterChatError,
    OpenRouterTransport, complete_with,
};

const GROQ_TRANSCRIPTION_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
const MAX_RESPONSE_BYTES: u64 = 1_048_576;

#[derive(Debug, Clone, PartialEq)]
pub struct MediaProviderResult {
    pub text: String,
    pub billing_segment: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionRequest<'a> {
    pub api_key: &'a str,
    pub base_url: &'a str,
    pub model: &'a str,
    pub system_prompt: &'a str,
    pub user_prompt: &'a str,
    pub image_bytes: &'a [u8],
    pub image_mime: &'a str,
    pub max_tokens: u64,
    pub file_id: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroqTranscriptionRequest {
    pub url: String,
    pub bearer_token: String,
    pub model: String,
    pub file_name: String,
    pub audio: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroqTranscriptionResponse {
    pub status_code: u16,
    pub headers: BTreeMap<String, String>,
    pub body: String,
}

pub trait GroqTranscriptionTransport {
    fn transcribe(
        &self,
        request: &GroqTranscriptionRequest,
    ) -> Result<GroqTranscriptionResponse, MediaProviderError>;
}

pub struct ReqwestGroqTranscriptionTransport {
    client: Client,
}

impl ReqwestGroqTranscriptionTransport {
    pub fn new() -> Result<Self, MediaProviderError> {
        Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(90))
            .build()
            .map(|client| Self { client })
            .map_err(|error| MediaProviderError::Transport(error.to_string()))
    }
}

impl GroqTranscriptionTransport for ReqwestGroqTranscriptionTransport {
    fn transcribe(
        &self,
        request: &GroqTranscriptionRequest,
    ) -> Result<GroqTranscriptionResponse, MediaProviderError> {
        let audio = Part::bytes(request.audio.clone())
            .file_name(request.file_name.clone())
            .mime_str("application/octet-stream")
            .map_err(|error| MediaProviderError::Transport(error.to_string()))?;
        let form = Form::new()
            .text("model", request.model.clone())
            .part("file", audio);
        let mut response = self
            .client
            .post(&request.url)
            .bearer_auth(&request.bearer_token)
            .multipart(form)
            .send()
            .map_err(|error| MediaProviderError::Transport(error.to_string()))?;
        let status_code = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_ascii_lowercase(), value.to_owned()))
            })
            .collect();
        let mut body = Vec::new();
        response
            .by_ref()
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_end(&mut body)
            .map_err(|error| MediaProviderError::Transport(error.to_string()))?;
        if body.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(MediaProviderError::ResponseTooLarge);
        }
        Ok(GroqTranscriptionResponse {
            status_code,
            headers,
            body: String::from_utf8_lossy(&body).into_owned(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MediaProviderError {
    #[error(transparent)]
    OpenRouter(#[from] OpenRouterChatError),
    #[error("provider credential is missing")]
    MissingCredential,
    #[error("provider transport failed: {0}")]
    Transport(String),
    #[error("provider response exceeded the safe size limit")]
    ResponseTooLarge,
    #[error("provider returned HTTP {status_code}: {message}")]
    Http {
        status_code: u16,
        code: String,
        message: String,
        retry_after_seconds: Option<u64>,
    },
    #[error("provider returned malformed JSON: {0}")]
    InvalidJson(String),
    #[error("provider response did not contain text")]
    MissingText,
}

impl MediaProviderError {
    #[must_use]
    pub const fn status_code(&self) -> Option<u16> {
        match self {
            Self::Http { status_code, .. } => Some(*status_code),
            Self::OpenRouter(OpenRouterChatError::RateLimited { .. }) => Some(429),
            Self::OpenRouter(OpenRouterChatError::Http { status_code, .. }) => Some(*status_code),
            _ => None,
        }
    }

    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::Http { code, .. } => code,
            _ => "",
        }
    }

    #[must_use]
    pub const fn retry_after_seconds(&self) -> Option<u64> {
        match self {
            Self::Http {
                retry_after_seconds,
                ..
            }
            | Self::OpenRouter(OpenRouterChatError::RateLimited {
                retry_after_seconds,
                ..
            }) => *retry_after_seconds,
            _ => None,
        }
    }
}

pub fn describe_image_with<T: OpenRouterTransport>(
    transport: &T,
    vision: VisionRequest<'_>,
) -> Result<MediaProviderResult, MediaProviderError> {
    let image_url = format!(
        "data:{};base64,{}",
        vision.image_mime,
        BASE64.encode(vision.image_bytes)
    );
    let messages = vec![
        ChatMessage::text(ChatRole::System, vision.system_prompt),
        ChatMessage {
            role: ChatRole::User,
            content: Some(json!([
                {"type": "text", "text": vision.user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        },
    ];
    let mut request = ChatCompletionRequest::new(vision.model, messages);
    request.max_tokens = Some(vision.max_tokens);
    let completion = complete_with(transport, vision.api_key, vision.base_url, &request)?;
    result_from_completion("vision", completion, "openrouter", vision.file_id, None)
}

pub fn transcribe_audio_openrouter_with<T: OpenRouterTransport>(
    transport: &T,
    api_key: &str,
    base_url: &str,
    model: &str,
    audio_bytes: &[u8],
    file_id: Option<&str>,
) -> Result<MediaProviderResult, MediaProviderError> {
    let audio_format = detect_audio_format(audio_bytes);
    let message = ChatMessage {
        role: ChatRole::User,
        content: Some(json!([
            {
                "type": "input_audio",
                "input_audio": {
                    "format": audio_format,
                    "data": BASE64.encode(audio_bytes),
                }
            },
            {"type": "text", "text": "Transcribe this audio exactly as spoken."}
        ])),
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    };
    let mut request = ChatCompletionRequest::new(model, vec![message]);
    request.max_tokens = Some(4_096);
    let completion = complete_with(transport, api_key, base_url, &request)?;
    result_from_completion("transcribe", completion, "openrouter", file_id, Some(0.0))
}

pub fn transcribe_audio_groq_with<T: GroqTranscriptionTransport>(
    transport: &T,
    api_key: &str,
    model: &str,
    audio_bytes: &[u8],
    file_id: Option<&str>,
    audio_seconds: f64,
    account: &str,
) -> Result<MediaProviderResult, MediaProviderError> {
    if api_key.trim().is_empty() {
        return Err(MediaProviderError::MissingCredential);
    }
    let response = transport.transcribe(&GroqTranscriptionRequest {
        url: GROQ_TRANSCRIPTION_URL.to_owned(),
        bearer_token: api_key.trim().to_owned(),
        model: model.to_owned(),
        file_name: "audio.webm".to_owned(),
        audio: audio_bytes.to_vec(),
    })?;
    let payload = serde_json::from_str::<Value>(&response.body)
        .map_err(|error| MediaProviderError::InvalidJson(error.to_string()))?;
    if response.status_code >= 400 {
        let error = payload.get("error").unwrap_or(&payload);
        return Err(MediaProviderError::Http {
            status_code: response.status_code,
            code: error
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("provider request failed")
                .to_owned(),
            retry_after_seconds: retry_after(&response.headers),
        });
    }
    let text = payload
        .get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or(MediaProviderError::MissingText)?
        .to_owned();
    let response_model = payload
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(model);
    let usage = payload
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    Ok(MediaProviderResult {
        text,
        billing_segment: json!({
            "kind": "transcribe",
            "model": response_model,
            "usage": usage,
            "audio_seconds": audio_seconds.max(0.0),
            "source": "groq",
            "metadata": {
                "file_id": file_id,
                "cache_hit": false,
                "provider": "groq",
                "groq_account": account,
            }
        }),
    })
}

fn result_from_completion(
    kind: &str,
    completion: ChatCompletion,
    source: &str,
    file_id: Option<&str>,
    audio_seconds: Option<f64>,
) -> Result<MediaProviderResult, MediaProviderError> {
    if completion.text.is_empty() {
        return Err(MediaProviderError::MissingText);
    }
    let mut metadata = Map::from_iter([
        ("file_id".to_owned(), json!(file_id)),
        ("cache_hit".to_owned(), json!(false)),
        ("provider".to_owned(), json!(source)),
    ]);
    if let Some(generation_id) = completion.generation_id {
        metadata.insert("provider_generation_id".to_owned(), json!(generation_id));
    }
    if let Some(provider) = completion.upstream_provider {
        metadata.insert("upstream_provider".to_owned(), json!(provider));
    }
    if let Some(service_tier) = completion.service_tier {
        metadata.insert("service_tier".to_owned(), json!(service_tier));
    }
    let mut segment = Map::from_iter([
        ("kind".to_owned(), json!(kind)),
        ("model".to_owned(), json!(completion.model)),
        ("usage".to_owned(), Value::Object(completion.usage)),
        ("source".to_owned(), json!(source)),
        ("metadata".to_owned(), Value::Object(metadata)),
    ]);
    if let Some(audio_seconds) = audio_seconds {
        segment.insert("audio_seconds".to_owned(), json!(audio_seconds.max(0.0)));
    }
    Ok(MediaProviderResult {
        text: completion.text,
        billing_segment: Value::Object(segment),
    })
}

fn detect_audio_format(audio: &[u8]) -> &'static str {
    if audio.starts_with(b"\x1aE\xdf\xa3") || audio.starts_with(b"ID3") {
        "mp3"
    } else if audio.starts_with(b"OggS") {
        "ogg"
    } else {
        "webm"
    }
}

fn retry_after(headers: &BTreeMap<String, String>) -> Option<u64> {
    [
        "retry-after",
        "x-ratelimit-reset",
        "x-ratelimit-reset-requests",
    ]
    .into_iter()
    .find_map(|name| {
        headers
            .get(name)
            .and_then(|value| value.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| value as u64)
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use crate::openrouter_chat::{HttpRequest, HttpResponse};

    use super::*;

    #[test]
    fn reqwest_groq_transport_posts_multipart_audio_and_preserves_response_metadata() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
            let mut request = [0_u8; 16_384];
            let count = stream.read(&mut request).unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..count]);
            let request = request.to_ascii_lowercase();
            assert!(request.contains("authorization: bearer synthetic-key"));
            assert!(request.contains("multipart/form-data"));
            let body = r#"{"text":"synthetic transcript"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Synthetic: yes\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .unwrap_or_else(|_| unreachable!());
        });
        let transport = ReqwestGroqTranscriptionTransport::new().unwrap_or_else(|_| unreachable!());
        let response = transport
            .transcribe(&GroqTranscriptionRequest {
                url: format!("http://{address}/transcribe"),
                bearer_token: "synthetic-key".to_owned(),
                model: "synthetic-model".to_owned(),
                file_name: "synthetic.ogg".to_owned(),
                audio: b"synthetic audio".to_vec(),
            })
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(response.status_code, 200);
        assert_eq!(
            response.headers.get("x-synthetic").map(String::as_str),
            Some("yes")
        );
        assert!(response.body.contains("synthetic transcript"));
        assert!(server.join().is_ok());
    }

    struct OpenRouter {
        request: RefCell<Option<HttpRequest>>,
        response: HttpResponse,
    }

    impl OpenRouterTransport for OpenRouter {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.request.replace(Some(request.clone()));
            Ok(self.response.clone())
        }
    }

    struct Groq {
        request: RefCell<Option<GroqTranscriptionRequest>>,
        response: GroqTranscriptionResponse,
    }

    impl GroqTranscriptionTransport for Groq {
        fn transcribe(
            &self,
            request: &GroqTranscriptionRequest,
        ) -> Result<GroqTranscriptionResponse, MediaProviderError> {
            self.request.replace(Some(request.clone()));
            Ok(self.response.clone())
        }
    }

    fn chat_response(text: &str) -> HttpResponse {
        HttpResponse {
            status_code: 200,
            headers: BTreeMap::new(),
            body: json!({
                "id": "generation-1",
                "model": "resolved/model",
                "provider": "SyntheticProvider",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}
            })
            .to_string(),
        }
    }

    #[test]
    fn vision_request_uses_typed_multimodal_content_and_normalizes_usage() {
        let transport = OpenRouter {
            request: RefCell::new(None),
            response: chat_response("a synthetic image"),
        };
        let result = describe_image_with(
            &transport,
            VisionRequest {
                api_key: "key",
                base_url: "https://synthetic.invalid/api/v1",
                model: "requested/model",
                system_prompt: "system",
                user_prompt: "describe",
                image_bytes: b"image",
                image_mime: "image/webp",
                max_tokens: 500,
                file_id: Some("file-1"),
            },
        );
        assert!(result.is_ok());
        let Some(result) = result.ok() else {
            return;
        };
        assert_eq!(result.text, "a synthetic image");
        assert_eq!(result.billing_segment["kind"], "vision");
        assert_eq!(result.billing_segment["model"], "resolved/model");
        assert_eq!(result.billing_segment["metadata"]["file_id"], "file-1");
        let body: Value = serde_json::from_str(
            &transport
                .request
                .borrow()
                .as_ref()
                .map_or_else(String::new, |request| request.body.clone()),
        )
        .unwrap_or(Value::Null);
        assert!(
            body["messages"][1]["content"][1]["image_url"]["url"]
                .as_str()
                .is_some_and(|value| value.starts_with("data:image/webp;base64,"))
        );
    }

    #[test]
    fn openrouter_audio_detects_container_and_preserves_provider_usage() {
        let transport = OpenRouter {
            request: RefCell::new(None),
            response: chat_response("spoken words"),
        };
        let result = transcribe_audio_openrouter_with(
            &transport,
            "key",
            "https://synthetic.invalid/api/v1",
            "requested/model",
            b"OggS synthetic",
            Some("audio-1"),
        );
        assert!(result.is_ok());
        let body: Value = serde_json::from_str(
            &transport
                .request
                .borrow()
                .as_ref()
                .map_or_else(String::new, |request| request.body.clone()),
        )
        .unwrap_or(Value::Null);
        assert_eq!(
            body["messages"][0]["content"][0]["input_audio"]["format"],
            "ogg"
        );
    }

    #[test]
    fn groq_transcription_normalizes_success_and_rate_limits() {
        let transport = Groq {
            request: RefCell::new(None),
            response: GroqTranscriptionResponse {
                status_code: 200,
                headers: BTreeMap::new(),
                body: json!({
                    "text": "spoken words",
                    "usage": {"total_time": 1.5}
                })
                .to_string(),
            },
        };
        let result = transcribe_audio_groq_with(
            &transport,
            "key",
            "whisper-large-v3",
            b"audio",
            Some("audio-1"),
            4.5,
            "free",
        );
        assert!(result.is_ok());
        let Some(result) = result.ok() else {
            return;
        };
        assert_eq!(result.text, "spoken words");
        assert_eq!(result.billing_segment["audio_seconds"], 4.5);
        assert_eq!(result.billing_segment["metadata"]["groq_account"], "free");

        let limited = Groq {
            request: RefCell::new(None),
            response: GroqTranscriptionResponse {
                status_code: 429,
                headers: BTreeMap::from([("retry-after".to_owned(), "12".to_owned())]),
                body: json!({"error": {"code": "rate_limit", "message": "slow down"}}).to_string(),
            },
        };
        let error = transcribe_audio_groq_with(
            &limited,
            "key",
            "whisper-large-v3",
            b"audio",
            None,
            0.0,
            "free",
        )
        .err();
        assert!(matches!(
            error,
            Some(MediaProviderError::Http {
                status_code: 429,
                retry_after_seconds: Some(12),
                ..
            })
        ));
    }

    #[test]
    fn media_errors_expose_stable_retry_metadata() {
        let http = MediaProviderError::Http {
            status_code: 503,
            code: "synthetic_unavailable".to_owned(),
            message: "synthetic failure".to_owned(),
            retry_after_seconds: Some(4),
        };
        assert_eq!(http.status_code(), Some(503));
        assert_eq!(http.code(), "synthetic_unavailable");
        assert_eq!(http.retry_after_seconds(), Some(4));

        let limited = MediaProviderError::OpenRouter(OpenRouterChatError::RateLimited {
            retry_after_seconds: Some(7),
            message: "synthetic limit".to_owned(),
        });
        assert_eq!(limited.status_code(), Some(429));
        assert_eq!(limited.code(), "");
        assert_eq!(limited.retry_after_seconds(), Some(7));

        let provider_http = MediaProviderError::OpenRouter(OpenRouterChatError::Http {
            status_code: 502,
            message: "synthetic upstream".to_owned(),
        });
        assert_eq!(provider_http.status_code(), Some(502));
        assert_eq!(provider_http.retry_after_seconds(), None);
        assert_eq!(MediaProviderError::MissingText.status_code(), None);
    }

    #[test]
    fn media_validation_handles_missing_and_malformed_provider_results() {
        let response = GroqTranscriptionResponse {
            status_code: 200,
            headers: BTreeMap::new(),
            body: "not-json".to_owned(),
        };
        for (api_key, response, expected) in [
            ("", response.clone(), MediaProviderError::MissingCredential),
            (
                "synthetic-key",
                response,
                MediaProviderError::InvalidJson("expected value at line 1 column 1".to_owned()),
            ),
            (
                "synthetic-key",
                GroqTranscriptionResponse {
                    status_code: 200,
                    headers: BTreeMap::new(),
                    body: json!({"text":""}).to_string(),
                },
                MediaProviderError::MissingText,
            ),
        ] {
            let transport = Groq {
                request: RefCell::new(None),
                response,
            };
            let result = transcribe_audio_groq_with(
                &transport,
                api_key,
                "synthetic-model",
                b"synthetic audio",
                None,
                -3.0,
                "synthetic-account",
            );
            let error = result.err().unwrap_or_else(|| unreachable!());
            if matches!(expected, MediaProviderError::InvalidJson(_)) {
                assert!(matches!(error, MediaProviderError::InvalidJson(_)));
            } else {
                assert_eq!(error, expected);
            }
        }

        assert_eq!(detect_audio_format(b"ID3 synthetic"), "mp3");
        assert_eq!(detect_audio_format(b"\x1aE\xdf\xa3 synthetic"), "mp3");
        assert_eq!(detect_audio_format(b"synthetic"), "webm");
        assert_eq!(
            retry_after(&BTreeMap::from([(
                "x-ratelimit-reset-requests".to_owned(),
                "2.9".to_owned(),
            )])),
            Some(2)
        );
    }

    #[test]
    fn completion_metadata_preserves_optional_provider_fields() {
        let result = result_from_completion(
            "synthetic-kind",
            ChatCompletion {
                generation_id: Some("synthetic-generation".to_owned()),
                text: "synthetic output".to_owned(),
                tool_calls: Vec::new(),
                finish_reason: Some("stop".to_owned()),
                model: "synthetic-model".to_owned(),
                upstream_provider: Some("synthetic-provider".to_owned()),
                service_tier: Some("synthetic-tier".to_owned()),
                annotations: Vec::new(),
                usage: Map::new(),
            },
            "synthetic-source",
            Some("synthetic-file"),
            Some(-1.0),
        )
        .unwrap_or_else(|_| unreachable!());
        assert_eq!(
            result.billing_segment["metadata"]["provider_generation_id"],
            "synthetic-generation"
        );
        assert_eq!(
            result.billing_segment["metadata"]["upstream_provider"],
            "synthetic-provider"
        );
        assert_eq!(
            result.billing_segment["metadata"]["service_tier"],
            "synthetic-tier"
        );
        assert_eq!(result.billing_segment["audio_seconds"], 0.0);
    }

    #[test]
    fn reqwest_groq_transport_rejects_oversized_responses() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
            let mut request = [0_u8; 8_192];
            let _read = stream.read(&mut request).unwrap_or_default();
            let body = vec![b'x'; (MAX_RESPONSE_BYTES + 1) as usize];
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .unwrap_or_else(|_| unreachable!());
            stream.write_all(&body).unwrap_or_else(|_| unreachable!());
        });
        let transport = ReqwestGroqTranscriptionTransport::new().unwrap_or_else(|_| unreachable!());
        let result = transport.transcribe(&GroqTranscriptionRequest {
            url: format!("http://{address}/transcribe"),
            bearer_token: "synthetic-key".to_owned(),
            model: "synthetic-model".to_owned(),
            file_name: "synthetic.webm".to_owned(),
            audio: b"synthetic audio".to_vec(),
        });
        assert_eq!(result, Err(MediaProviderError::ResponseTooLarge));
        assert!(server.join().is_ok());
    }
}
