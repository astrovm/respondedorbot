//! Typed image-description and audio-transcription provider boundaries.

use std::collections::BTreeMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::openrouter_chat::{
    ChatCompletion, ChatCompletionRequest, ChatMessage, ChatRole, HttpRequest, HttpResponse,
    OpenRouterChatError, OpenRouterTransport, complete_with,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MediaProviderResult {
    pub text: String,
    pub billing_segment: Value,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionRequest<'a> {
    pub api_key: &'a str,
    pub base_url: &'a str,
    pub model: &'a str,
    pub system_prompt: &'a str,
    pub user_prompt: &'a str,
    pub image_bytes: &'a [u8],
    pub image_mime: &'a str,
    pub max_tokens: u64,
    pub price_ceiling: Option<(f64, f64)>,
    pub file_id: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MediaProviderError {
    #[error(transparent)]
    OpenRouter(#[from] OpenRouterChatError),
    #[error("provider credential is missing")]
    MissingCredential,
    #[error("provider transport failed: {0}")]
    Transport(String),
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
            reasoning: None,
            reasoning_details: Vec::new(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        },
    ];
    let mut request = ChatCompletionRequest::new(vision.model, messages);
    request.max_tokens = Some(vision.max_tokens);
    if let Some((prompt, completion)) = vision.price_ceiling {
        request.set_price_ceiling(prompt, completion);
    }
    let completion = complete_with(transport, vision.api_key, vision.base_url, &request)?;
    result_from_completion("vision", completion, "openrouter", vision.file_id, None)
}

pub fn transcribe_audio_openrouter_with<T: OpenRouterTransport>(
    transport: &T,
    api_key: &str,
    base_url: &str,
    model: &str,
    audio_bytes: &[u8],
    audio_seconds: f64,
    file_id: Option<&str>,
) -> Result<MediaProviderResult, MediaProviderError> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(MediaProviderError::MissingCredential);
    }
    let model = model.trim();
    if model.is_empty() {
        return Err(MediaProviderError::OpenRouter(
            OpenRouterChatError::MissingModel,
        ));
    }
    let audio_format = detect_audio_format(audio_bytes);
    let response = transport.post(&HttpRequest {
        url: transcription_url(base_url)?,
        bearer_token: api_key.to_owned(),
        body: serde_json::to_string(&json!({
            "model": model,
            "input_audio": {
                "format": audio_format,
                "data": BASE64.encode(audio_bytes),
            }
        }))
        .map_err(|error| {
            MediaProviderError::OpenRouter(OpenRouterChatError::RequestJson(error.to_string()))
        })?,
    })?;
    transcription_result(response, model, audio_seconds, file_id)
}

fn transcription_result(
    response: HttpResponse,
    requested_model: &str,
    audio_seconds: f64,
    file_id: Option<&str>,
) -> Result<MediaProviderResult, MediaProviderError> {
    if response.status_code >= 400 {
        let payload = serde_json::from_str::<Value>(&response.body).unwrap_or(Value::Null);
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
    let payload = serde_json::from_str::<Value>(&response.body)
        .map_err(|error| MediaProviderError::InvalidJson(error.to_string()))?;
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
        .unwrap_or(requested_model);
    let usage = payload
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let audio_seconds = usage
        .get("seconds")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(audio_seconds)
        .max(0.0);
    Ok(MediaProviderResult {
        text,
        billing_segment: json!({
            "kind": "transcribe",
            "model": response_model,
            "usage": usage,
            "audio_seconds": audio_seconds,
            "source": "openrouter",
            "metadata": {
                "file_id": file_id,
                "cache_hit": false,
                "provider": "openrouter",
            }
        }),
    })
}

fn transcription_url(base_url: &str) -> Result<String, MediaProviderError> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let parsed = reqwest::Url::parse(trimmed)
        .map_err(|_| MediaProviderError::OpenRouter(OpenRouterChatError::InvalidBaseUrl))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(MediaProviderError::OpenRouter(
            OpenRouterChatError::InvalidBaseUrl,
        ));
    }
    Ok(format!("{trimmed}/audio/transcriptions"))
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
    if audio.starts_with(b"ID3") {
        "mp3"
    } else if audio.starts_with(b"\x1aE\xdf\xa3") {
        "webm"
    } else if audio.starts_with(b"OggS") {
        "ogg"
    } else if audio.starts_with(b"fLaC") {
        "flac"
    } else if audio.starts_with(b"RIFF") && audio.get(8..12) == Some(b"WAVE") {
        "wav"
    } else if audio.get(4..8) == Some(b"ftyp") {
        "m4a"
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

    use crate::openrouter_chat::{HttpRequest, HttpResponse};

    use super::*;

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
                price_ceiling: None,
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
            response: HttpResponse {
                status_code: 200,
                headers: BTreeMap::new(),
                body: json!({
                    "text": "spoken words",
                    "model": "resolved/model",
                    "usage": {"seconds": 3.25, "cost": "0.0000903"}
                })
                .to_string(),
            },
        };
        let result = transcribe_audio_openrouter_with(
            &transport,
            "key",
            "https://synthetic.invalid/api/v1",
            "requested/model",
            b"OggS synthetic",
            4.5,
            Some("audio-1"),
        );
        assert!(result.is_ok());
        let Some(result) = result.ok() else {
            return;
        };
        assert_eq!(result.text, "spoken words");
        assert_eq!(result.billing_segment["kind"], "transcribe");
        assert_eq!(result.billing_segment["model"], "resolved/model");
        assert_eq!(result.billing_segment["audio_seconds"], 3.25);
        assert_eq!(result.billing_segment["usage"]["cost"], "0.0000903");
        assert_eq!(result.billing_segment["source"], "openrouter");
        assert_eq!(result.billing_segment["metadata"]["provider"], "openrouter");
        let body: Value = serde_json::from_str(
            &transport
                .request
                .borrow()
                .as_ref()
                .map_or_else(String::new, |request| request.body.clone()),
        )
        .unwrap_or(Value::Null);
        assert_eq!(body["model"], "requested/model");
        assert_eq!(body["input_audio"]["format"], "ogg");
        assert_eq!(
            body["input_audio"]["data"],
            BASE64.encode(b"OggS synthetic")
        );
        assert_eq!(
            transport
                .request
                .borrow()
                .as_ref()
                .map(|request| request.url.as_str()),
            Some("https://synthetic.invalid/api/v1/audio/transcriptions")
        );
        assert_eq!(
            transport
                .request
                .borrow()
                .as_ref()
                .map(|request| request.bearer_token.as_str()),
            Some("key")
        );
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
        let response = HttpResponse {
            status_code: 200,
            headers: BTreeMap::new(),
            body: "not-json".to_owned(),
        };
        let invalid_json = OpenRouter {
            request: RefCell::new(None),
            response: response.clone(),
        };
        assert_eq!(
            transcribe_audio_openrouter_with(
                &invalid_json,
                "synthetic-key",
                "https://synthetic.invalid/api/v1",
                "synthetic-model",
                b"synthetic audio",
                3.0,
                None,
            )
            .err()
            .map(|error| matches!(error, MediaProviderError::InvalidJson(_))),
            Some(true)
        );

        let missing_text = OpenRouter {
            request: RefCell::new(None),
            response: HttpResponse {
                status_code: 200,
                headers: BTreeMap::new(),
                body: json!({"text":""}).to_string(),
            },
        };
        assert_eq!(
            transcribe_audio_openrouter_with(
                &missing_text,
                "synthetic-key",
                "https://synthetic.invalid/api/v1",
                "synthetic-model",
                b"synthetic audio",
                3.0,
                None,
            )
            .err(),
            Some(MediaProviderError::MissingText)
        );

        let missing_key = OpenRouter {
            request: RefCell::new(None),
            response,
        };
        assert_eq!(
            transcribe_audio_openrouter_with(
                &missing_key,
                "",
                "https://synthetic.invalid/api/v1",
                "synthetic-model",
                b"synthetic audio",
                3.0,
                None,
            )
            .err(),
            Some(MediaProviderError::MissingCredential)
        );

        let limited = OpenRouter {
            request: RefCell::new(None),
            response: HttpResponse {
                status_code: 429,
                headers: BTreeMap::from([("retry-after".to_owned(), "12".to_owned())]),
                body: json!({"error": {"code": "rate_limit", "message": "slow down"}}).to_string(),
            },
        };
        assert!(matches!(
            transcribe_audio_openrouter_with(
                &limited,
                "synthetic-key",
                "https://synthetic.invalid/api/v1",
                "synthetic-model",
                b"synthetic audio",
                3.0,
                None,
            ),
            Err(MediaProviderError::Http {
                status_code: 429,
                code,
                retry_after_seconds: Some(12),
                ..
            }) if code == "rate_limit"
        ));

        let malformed_url = OpenRouter {
            request: RefCell::new(None),
            response: HttpResponse {
                status_code: 200,
                headers: BTreeMap::new(),
                body: json!({"text":"synthetic"}).to_string(),
            },
        };
        assert!(matches!(
            transcribe_audio_openrouter_with(
                &malformed_url,
                "synthetic-key",
                "not a url",
                "synthetic-model",
                b"synthetic audio",
                3.0,
                None,
            ),
            Err(MediaProviderError::OpenRouter(
                OpenRouterChatError::InvalidBaseUrl
            ))
        ));

        assert_eq!(detect_audio_format(b"ID3 synthetic"), "mp3");
        assert_eq!(detect_audio_format(b"\x1aE\xdf\xa3 synthetic"), "webm");
        assert_eq!(detect_audio_format(b"OggS synthetic"), "ogg");
        assert_eq!(detect_audio_format(b"fLaC synthetic"), "flac");
        assert_eq!(detect_audio_format(b"RIFFxxxxWAVE"), "wav");
        assert_eq!(detect_audio_format(b"xxxxftyp"), "m4a");
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
}
