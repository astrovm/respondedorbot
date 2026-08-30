//! Typed blocking OpenRouter chat-completion boundary.

use std::collections::BTreeMap;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const DEFAULT_OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    #[must_use]
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(Value::String(content.into())),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    #[must_use]
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: Some(Value::String(content.into())),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
        }
    }

    #[must_use]
    pub fn assistant_tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: calls,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    pub stream: bool,
}

impl ChatCompletionRequest {
    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            max_tokens: None,
            temperature: None,
            stream: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    pub generation_id: Option<String>,
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub model: String,
    pub upstream_provider: Option<String>,
    pub service_tier: Option<String>,
    pub annotations: Vec<Value>,
    pub usage: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub url: String,
    pub bearer_token: String,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
    pub headers: BTreeMap<String, String>,
}

pub trait OpenRouterTransport {
    fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError>;
}

pub struct ReqwestOpenRouterTransport {
    client: Client,
}

impl ReqwestOpenRouterTransport {
    pub fn new() -> Result<Self, OpenRouterChatError> {
        Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(90))
            .build()
            .map(|client| Self { client })
            .map_err(|error| OpenRouterChatError::Transport(error.to_string()))
    }
}

impl OpenRouterTransport for ReqwestOpenRouterTransport {
    fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
        let response = self
            .client
            .post(&request.url)
            .bearer_auth(&request.bearer_token)
            .header("Content-Type", "application/json")
            .body(request.body.clone())
            .send()
            .map_err(|error| OpenRouterChatError::Transport(error.to_string()))?;
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
        let body = response
            .text()
            .map_err(|error| OpenRouterChatError::Transport(error.to_string()))?;
        Ok(HttpResponse {
            status_code,
            body,
            headers,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OpenRouterChatError {
    #[error("OpenRouter API key is missing")]
    MissingApiKey,
    #[error("OpenRouter model is missing")]
    MissingModel,
    #[error("OpenRouter base URL is invalid")]
    InvalidBaseUrl,
    #[error("OpenRouter request could not be serialized: {0}")]
    RequestJson(String),
    #[error("OpenRouter transport failed: {0}")]
    Transport(String),
    #[error("OpenRouter rate limited the request")]
    RateLimited {
        retry_after_seconds: Option<u64>,
        message: String,
    },
    #[error("OpenRouter returned HTTP {status_code}: {message}")]
    Http { status_code: u16, message: String },
    #[error("OpenRouter returned malformed JSON: {0}")]
    InvalidJson(String),
    #[error("OpenRouter response did not contain a valid completion")]
    MalformedResponse,
}

#[derive(Deserialize)]
struct RawEnvelope {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    choices: Vec<RawChoice>,
    #[serde(default)]
    usage: Map<String, Value>,
}

#[derive(Deserialize)]
struct RawChoice {
    #[serde(default)]
    message: Option<RawMessage>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
    #[serde(default)]
    annotations: Vec<Value>,
}

fn completion_url(base_url: &str) -> Result<String, OpenRouterChatError> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let parsed = reqwest::Url::parse(trimmed).map_err(|_| OpenRouterChatError::InvalidBaseUrl)?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(OpenRouterChatError::InvalidBaseUrl);
    }
    Ok(format!("{trimmed}/chat/completions"))
}

fn response_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| "upstream request failed".to_owned())
}

fn retry_after_seconds(headers: &BTreeMap<String, String>) -> Option<u64> {
    headers
        .get("retry-after")
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn content_text(content: &Value) -> Result<String, OpenRouterChatError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => Ok(parts
            .iter()
            .filter_map(|part| {
                part.as_object()
                    .and_then(|part| part.get("text"))
                    .and_then(Value::as_str)
            })
            .collect::<String>()),
        Value::Bool(_) | Value::Number(_) | Value::Object(_) => {
            Err(OpenRouterChatError::MalformedResponse)
        }
    }
}

pub fn parse_chat_completion(
    response: HttpResponse,
    requested_model: &str,
) -> Result<ChatCompletion, OpenRouterChatError> {
    if response.status_code == 429 {
        return Err(OpenRouterChatError::RateLimited {
            retry_after_seconds: retry_after_seconds(&response.headers),
            message: response_message(&response.body),
        });
    }
    if response.status_code >= 400 {
        return Err(OpenRouterChatError::Http {
            status_code: response.status_code,
            message: response_message(&response.body),
        });
    }
    let envelope = serde_json::from_str::<RawEnvelope>(&response.body)
        .map_err(|error| OpenRouterChatError::InvalidJson(error.to_string()))?;
    let choice = envelope
        .choices
        .into_iter()
        .next()
        .ok_or(OpenRouterChatError::MalformedResponse)?;
    let message = choice
        .message
        .ok_or(OpenRouterChatError::MalformedResponse)?;
    let text = content_text(&message.content)?;
    if text.is_empty() && message.tool_calls.is_empty() {
        return Err(OpenRouterChatError::MalformedResponse);
    }
    Ok(ChatCompletion {
        generation_id: envelope.id.filter(|value| !value.is_empty()),
        text,
        tool_calls: message.tool_calls,
        finish_reason: choice.finish_reason.filter(|value| !value.is_empty()),
        model: envelope
            .model
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| requested_model.to_owned()),
        upstream_provider: envelope.provider.filter(|value| !value.is_empty()),
        service_tier: envelope.service_tier.filter(|value| !value.is_empty()),
        annotations: message.annotations,
        usage: envelope.usage,
    })
}

pub fn complete_with<T: OpenRouterTransport>(
    transport: &T,
    api_key: &str,
    base_url: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletion, OpenRouterChatError> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(OpenRouterChatError::MissingApiKey);
    }
    if request.model.trim().is_empty() {
        return Err(OpenRouterChatError::MissingModel);
    }
    if request.stream {
        return Err(OpenRouterChatError::MalformedResponse);
    }
    let body = serde_json::to_string(request)
        .map_err(|error| OpenRouterChatError::RequestJson(error.to_string()))?;
    parse_chat_completion(
        transport.post(&HttpRequest {
            url: completion_url(base_url)?,
            bearer_token: api_key.to_owned(),
            body,
        })?,
        &request.model,
    )
}

pub fn complete(
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletion, OpenRouterChatError> {
    complete_with(
        &ReqwestOpenRouterTransport::new()?,
        api_key,
        DEFAULT_OPENROUTER_BASE_URL,
        request,
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use serde_json::{Value, json};

    use super::{
        ChatCompletionRequest, ChatMessage, ChatRole, HttpRequest, HttpResponse,
        OpenRouterChatError, OpenRouterTransport, ToolCall, ToolFunctionCall, complete_with,
        parse_chat_completion,
    };

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, OpenRouterChatError>>>,
        requests: RefCell<Vec<HttpRequest>>,
    }

    impl OpenRouterTransport for Transport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or_else(|| Err(OpenRouterChatError::Transport("missing response".into())))
        }
    }

    fn response(status_code: u16, body: Value) -> HttpResponse {
        HttpResponse {
            status_code,
            body: body.to_string(),
            headers: BTreeMap::new(),
        }
    }

    fn request() -> ChatCompletionRequest {
        ChatCompletionRequest::new(
            "synthetic/model",
            vec![
                ChatMessage::text(ChatRole::System, "synthetic system"),
                ChatMessage::text(ChatRole::User, "synthetic question"),
            ],
        )
    }

    #[test]
    fn sends_authenticated_typed_request_and_normalizes_usage() {
        let transport = Transport {
            response: RefCell::new(Some(Ok(response(
                200,
                json!({
                    "id": "generation-1",
                    "model": "resolved/model",
                    "provider": "SyntheticProvider",
                    "service_tier": "priority",
                    "choices": [{
                        "message": {
                            "content": "synthetic answer",
                            "annotations": [{"type": "url_citation"}]
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4}
                }),
            )))),
            requests: RefCell::new(Vec::new()),
        };
        let actual = complete_with(
            &transport,
            " synthetic-key ",
            "https://openrouter.example/api/v1/",
            &request(),
        );
        assert_eq!(
            actual.as_ref().map(|completion| completion.text.as_str()),
            Ok("synthetic answer")
        );
        assert_eq!(
            actual.as_ref().map(|completion| completion.model.as_str()),
            Ok("resolved/model")
        );
        assert_eq!(
            actual
                .as_ref()
                .map(|completion| completion.upstream_provider.as_deref()),
            Ok(Some("SyntheticProvider"))
        );
        assert_eq!(
            actual
                .as_ref()
                .map(|completion| completion.service_tier.as_deref()),
            Ok(Some("priority"))
        );
        assert_eq!(
            actual
                .as_ref()
                .map(|completion| completion.usage["prompt_tokens"].clone()),
            Ok(json!(10))
        );
        let requests = transport.requests.borrow();
        assert_eq!(
            requests[0].url,
            "https://openrouter.example/api/v1/chat/completions"
        );
        assert_eq!(requests[0].bearer_token, "synthetic-key");
        let body: Value = serde_json::from_str(&requests[0].body).unwrap_or(Value::Null);
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn preserves_function_tool_calls_and_builds_followup_messages() {
        let call = ToolCall {
            id: "call-1".to_owned(),
            call_type: "function".to_owned(),
            function: ToolFunctionCall {
                name: "weather".to_owned(),
                arguments: "{\"location\":\"Synthetic City\"}".to_owned(),
            },
        };
        let completion = parse_chat_completion(
            response(
                200,
                json!({
                    "choices": [{"message": {"content": null, "tool_calls": [call.clone()]}}]
                }),
            ),
            "synthetic/model",
        );
        assert!(completion.is_ok());
        assert_eq!(
            completion.map(|value| value.tool_calls),
            Ok(vec![call.clone()])
        );
        let assistant = serde_json::to_value(ChatMessage::assistant_tool_calls(vec![call]));
        assert_eq!(
            assistant.unwrap_or(Value::Null)["tool_calls"][0]["function"]["name"],
            "weather"
        );
        let result = serde_json::to_value(ChatMessage::tool_result("call-1", "sunny"));
        assert_eq!(result.unwrap_or(Value::Null)["tool_call_id"], "call-1");
    }

    #[test]
    fn joins_text_parts_and_uses_requested_model_when_response_omits_it() {
        let actual = parse_chat_completion(
            response(
                200,
                json!({
                    "choices": [{"message": {"content": [
                        {"type": "text", "text": "hello "},
                        {"type": "image", "url": "ignored"},
                        {"type": "text", "text": "world"}
                    ]}}]
                }),
            ),
            "requested/model",
        );
        assert!(actual.is_ok());
        assert_eq!(
            actual.map(|value| (value.text, value.model)),
            Ok(("hello world".into(), "requested/model".into()))
        );
    }

    #[test]
    fn classifies_rate_limits_with_retry_after_and_safe_error_message() {
        let mut headers = BTreeMap::new();
        headers.insert("retry-after".to_owned(), "17".to_owned());
        assert_eq!(
            parse_chat_completion(
                HttpResponse {
                    status_code: 429,
                    body: json!({"error": {"message": "capacity exhausted"}}).to_string(),
                    headers,
                },
                "model",
            ),
            Err(OpenRouterChatError::RateLimited {
                retry_after_seconds: Some(17),
                message: "capacity exhausted".to_owned(),
            })
        );
        assert_eq!(
            parse_chat_completion(response(503, json!({})), "model"),
            Err(OpenRouterChatError::Http {
                status_code: 503,
                message: "upstream request failed".to_owned(),
            })
        );
    }

    #[test]
    fn malformed_json_choices_content_and_empty_outputs_are_distinct() {
        assert!(matches!(
            parse_chat_completion(
                HttpResponse {
                    status_code: 200,
                    body: "not-json".to_owned(),
                    headers: BTreeMap::new(),
                },
                "model"
            ),
            Err(OpenRouterChatError::InvalidJson(_))
        ));
        for payload in [
            json!({"choices": []}),
            json!({"choices": [{"message": {"content": {"bad": true}}}]}),
            json!({"choices": [{"message": {"content": ""}}]}),
        ] {
            assert_eq!(
                parse_chat_completion(response(200, payload), "model"),
                Err(OpenRouterChatError::MalformedResponse)
            );
        }
    }

    #[test]
    fn rejects_credentials_model_url_and_stream_mode_before_transport() {
        for (key, base_url, mut request, expected) in [
            (
                "",
                "https://example.com",
                request(),
                OpenRouterChatError::MissingApiKey,
            ),
            (
                "key",
                "https://example.com",
                ChatCompletionRequest::new("", Vec::new()),
                OpenRouterChatError::MissingModel,
            ),
            (
                "key",
                "file:///tmp/provider",
                request(),
                OpenRouterChatError::InvalidBaseUrl,
            ),
        ] {
            let transport = Transport {
                response: RefCell::new(None),
                requests: RefCell::new(Vec::new()),
            };
            assert_eq!(
                complete_with(&transport, key, base_url, &request),
                Err(expected)
            );
            assert!(transport.requests.borrow().is_empty());
            request.stream = false;
        }
        let mut streaming = request();
        streaming.stream = true;
        let transport = Transport {
            response: RefCell::new(None),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            complete_with(&transport, "key", "https://example.com", &streaming),
            Err(OpenRouterChatError::MalformedResponse)
        );
    }

    #[test]
    fn transport_failures_propagate_without_leaking_the_api_key() {
        let transport = Transport {
            response: RefCell::new(Some(Err(OpenRouterChatError::Transport(
                "synthetic timeout".to_owned(),
            )))),
            requests: RefCell::new(Vec::new()),
        };
        let actual = complete_with(
            &transport,
            "synthetic-secret",
            "https://example.com/api/v1",
            &request(),
        );
        assert_eq!(
            actual,
            Err(OpenRouterChatError::Transport(
                "synthetic timeout".to_owned()
            ))
        );
        assert!(!format!("{actual:?}").contains("synthetic-secret"));
    }
}
