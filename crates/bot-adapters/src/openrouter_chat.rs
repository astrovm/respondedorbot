//! Typed blocking OpenRouter chat-completion boundary.

use std::collections::BTreeMap;
use std::io::Read;
use std::time::Duration;

use bot_core::provider_stream_policy::StreamToolCallFragment;
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

#[derive(Debug, Clone, PartialEq)]
pub struct ChatStreamChunk {
    pub generation_id: Option<String>,
    pub text: String,
    pub tool_call_fragments: Vec<StreamToolCallFragment>,
    pub finish_reason: Option<String>,
    pub model: Option<String>,
    pub upstream_provider: Option<String>,
    pub service_tier: Option<String>,
    pub annotations: Vec<Value>,
    pub usage: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamEvent {
    Chunk(Box<ChatStreamChunk>),
    Done,
}

pub trait OpenRouterTransport {
    fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError>;
}

pub trait OpenRouterStreamTransport {
    fn post_stream(
        &self,
        request: &HttpRequest,
        on_bytes: &mut dyn FnMut(&[u8]) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError>;
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

impl OpenRouterStreamTransport for ReqwestOpenRouterTransport {
    fn post_stream(
        &self,
        request: &HttpRequest,
        on_bytes: &mut dyn FnMut(&[u8]) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError> {
        let mut response = self
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
            .collect::<BTreeMap<_, _>>();
        if status_code >= 400 {
            let body = response
                .text()
                .map_err(|error| OpenRouterChatError::Transport(error.to_string()))?;
            return Err(http_error(status_code, &body, &headers));
        }
        let mut buffer = [0_u8; 8_192];
        loop {
            let count = response
                .read(&mut buffer)
                .map_err(|error| OpenRouterChatError::Transport(error.to_string()))?;
            if count == 0 {
                return Ok(());
            }
            on_bytes(&buffer[..count])?;
        }
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
    #[error("OpenRouter stream ended with an incomplete UTF-8 or SSE frame")]
    IncompleteStream,
    #[error("OpenRouter stream returned an error: {0}")]
    Stream(String),
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

#[derive(Deserialize)]
struct RawStreamEnvelope {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    choices: Vec<RawStreamChoice>,
    #[serde(default)]
    usage: Map<String, Value>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct RawStreamChoice {
    #[serde(default)]
    delta: RawStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Default, Deserialize)]
struct RawStreamDelta {
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Vec<RawStreamToolCall>,
    #[serde(default)]
    annotations: Vec<Value>,
}

#[derive(Deserialize)]
struct RawStreamToolCall {
    #[serde(default)]
    index: Value,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    call_type: Option<String>,
    #[serde(default)]
    function: Option<RawStreamFunction>,
}

#[derive(Deserialize)]
struct RawStreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
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

fn http_error(
    status_code: u16,
    body: &str,
    headers: &BTreeMap<String, String>,
) -> OpenRouterChatError {
    if status_code == 429 {
        OpenRouterChatError::RateLimited {
            retry_after_seconds: retry_after_seconds(headers),
            message: response_message(body),
        }
    } else {
        OpenRouterChatError::Http {
            status_code,
            message: response_message(body),
        }
    }
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
    if response.status_code >= 400 {
        return Err(http_error(
            response.status_code,
            &response.body,
            &response.headers,
        ));
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

#[derive(Debug, Default)]
struct SseDecoder {
    pending: Vec<u8>,
    saw_done: bool,
}

impl SseDecoder {
    fn feed(
        &mut self,
        bytes: &[u8],
        on_event: &mut dyn FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError> {
        self.pending.extend_from_slice(bytes);
        while let Some((frame_end, delimiter_length)) = next_sse_frame(&self.pending) {
            let frame = self.pending[..frame_end].to_vec();
            self.pending.drain(..frame_end + delimiter_length);
            if let Some(event) = parse_sse_frame(&frame)? {
                if event == ChatStreamEvent::Done {
                    self.saw_done = true;
                }
                on_event(event)?;
            }
        }
        Ok(())
    }

    fn finish(
        &mut self,
        on_event: &mut dyn FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError> {
        if !self.pending.iter().all(u8::is_ascii_whitespace) {
            let frame = std::mem::take(&mut self.pending);
            if let Some(event) = parse_sse_frame(&frame)? {
                if event == ChatStreamEvent::Done {
                    self.saw_done = true;
                }
                on_event(event)?;
            }
        }
        if self.pending.iter().all(u8::is_ascii_whitespace) {
            self.pending.clear();
        }
        if self.saw_done {
            Ok(())
        } else {
            Err(OpenRouterChatError::IncompleteStream)
        }
    }
}

fn next_sse_frame(bytes: &[u8]) -> Option<(usize, usize)> {
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4))
        .or_else(|| {
            bytes
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|position| (position, 2))
        })
}

fn parse_sse_frame(frame: &[u8]) -> Result<Option<ChatStreamEvent>, OpenRouterChatError> {
    let frame = std::str::from_utf8(frame).map_err(|_| OpenRouterChatError::IncompleteStream)?;
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        return Ok(None);
    }
    if data.trim() == "[DONE]" {
        return Ok(Some(ChatStreamEvent::Done));
    }
    let envelope = serde_json::from_str::<RawStreamEnvelope>(&data)
        .map_err(|error| OpenRouterChatError::InvalidJson(error.to_string()))?;
    if let Some(error) = envelope.error.as_ref() {
        return Err(OpenRouterChatError::Stream(stream_error_message(error)));
    }
    let choice = envelope.choices.into_iter().next();
    if let Some(error) = choice.as_ref().and_then(|choice| choice.error.as_ref()) {
        return Err(OpenRouterChatError::Stream(stream_error_message(error)));
    }
    let (text, fragments, finish_reason, annotations) = choice.map_or_else(
        || Ok::<_, OpenRouterChatError>((String::new(), Vec::new(), None, Vec::new())),
        |choice| {
            let text = content_text(&choice.delta.content)?;
            let fragments = choice
                .delta
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(position, fragment)| {
                    let function = fragment.function;
                    StreamToolCallFragment {
                        position: i64::try_from(position).unwrap_or(i64::MAX),
                        index: fragment.index,
                        id: fragment.id,
                        call_type: fragment.call_type,
                        name: function.as_ref().and_then(|value| value.name.clone()),
                        arguments: function.and_then(|value| value.arguments),
                    }
                })
                .collect();
            Ok::<_, OpenRouterChatError>((
                text,
                fragments,
                choice.finish_reason.filter(|value| !value.is_empty()),
                choice.delta.annotations,
            ))
        },
    )?;
    Ok(Some(ChatStreamEvent::Chunk(Box::new(ChatStreamChunk {
        generation_id: envelope.id.filter(|value| !value.is_empty()),
        text,
        tool_call_fragments: fragments,
        finish_reason,
        model: envelope.model.filter(|value| !value.is_empty()),
        upstream_provider: envelope.provider.filter(|value| !value.is_empty()),
        service_tier: envelope.service_tier.filter(|value| !value.is_empty()),
        annotations,
        usage: envelope.usage,
    }))))
}

fn stream_error_message(error: &Value) -> String {
    error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .filter(|message| !message.is_empty())
        .unwrap_or("unknown provider error")
        .to_owned()
}

pub fn stream_with<T, F>(
    transport: &T,
    api_key: &str,
    base_url: &str,
    request: &ChatCompletionRequest,
    mut on_event: F,
) -> Result<(), OpenRouterChatError>
where
    T: OpenRouterStreamTransport,
    F: FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
{
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(OpenRouterChatError::MissingApiKey);
    }
    if request.model.trim().is_empty() {
        return Err(OpenRouterChatError::MissingModel);
    }
    if !request.stream {
        return Err(OpenRouterChatError::MalformedResponse);
    }
    let body = serde_json::to_string(request)
        .map_err(|error| OpenRouterChatError::RequestJson(error.to_string()))?;
    let mut decoder = SseDecoder::default();
    transport.post_stream(
        &HttpRequest {
            url: completion_url(base_url)?,
            bearer_token: api_key.to_owned(),
            body,
        },
        &mut |bytes| decoder.feed(bytes, &mut on_event),
    )?;
    decoder.finish(&mut on_event)
}

pub fn stream<F>(
    api_key: &str,
    request: &ChatCompletionRequest,
    on_event: F,
) -> Result<(), OpenRouterChatError>
where
    F: FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
{
    stream_with(
        &ReqwestOpenRouterTransport::new()?,
        api_key,
        DEFAULT_OPENROUTER_BASE_URL,
        request,
        on_event,
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use serde_json::{Value, json};

    use super::{
        ChatCompletionRequest, ChatMessage, ChatRole, ChatStreamEvent, HttpRequest, HttpResponse,
        OpenRouterChatError, OpenRouterStreamTransport, OpenRouterTransport, ToolCall,
        ToolFunctionCall, complete_with, parse_chat_completion, stream_with,
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

    struct StreamTransport {
        chunks: Vec<Vec<u8>>,
        failure: Option<OpenRouterChatError>,
        requests: RefCell<Vec<HttpRequest>>,
    }

    impl OpenRouterStreamTransport for StreamTransport {
        fn post_stream(
            &self,
            request: &HttpRequest,
            on_bytes: &mut dyn FnMut(&[u8]) -> Result<(), OpenRouterChatError>,
        ) -> Result<(), OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            for chunk in &self.chunks {
                on_bytes(chunk)?;
            }
            self.failure.clone().map_or(Ok(()), Err)
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

    #[test]
    fn incremental_sse_preserves_text_tool_fragments_usage_and_metadata() {
        let body = [
            ": keepalive\r\n\r\n".to_owned(),
            format!(
                "data: {}\r\n\r\n",
                json!({
                    "id": "gen-1",
                    "model": "resolved/model",
                    "provider": "Synthetic",
                    "service_tier": "paid",
                    "choices": [{"delta": {"content": "holá "}}]
                })
            ),
            format!(
                "data: {}\n\n",
                json!({
                    "choices": [{"delta": {
                        "content": "mundo",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "wea", "arguments": "{\"city\":"}
                        }]
                    }}]
                })
            ),
            format!(
                "data: {}\n\n",
                json!({
                    "choices": [{
                        "delta": {"tool_calls": [{
                            "index": "0",
                            "function": {"name": "ther", "arguments": "\"Synthetic\"}"}
                        }]},
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "cost": "0.001"
                    }
                })
            ),
            "data: [DONE]\n\n".to_owned(),
        ]
        .concat();
        let chunks = body
            .as_bytes()
            .chunks(7)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let transport = StreamTransport {
            chunks,
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let mut request = request();
        request.stream = true;
        let mut events = Vec::new();
        let result = stream_with(
            &transport,
            " synthetic-key ",
            "https://openrouter.example/api/v1/",
            &request,
            |event| {
                events.push(event);
                Ok(())
            },
        );
        assert_eq!(result, Ok(()));
        assert_eq!(events.len(), 4);
        let ChatStreamEvent::Chunk(first) = &events[0] else {
            return;
        };
        assert_eq!(first.text, "holá ");
        assert_eq!(first.generation_id.as_deref(), Some("gen-1"));
        assert_eq!(first.model.as_deref(), Some("resolved/model"));
        let ChatStreamEvent::Chunk(second) = &events[1] else {
            return;
        };
        assert_eq!(second.text, "mundo");
        assert_eq!(second.tool_call_fragments[0].name.as_deref(), Some("wea"));
        let ChatStreamEvent::Chunk(final_chunk) = &events[2] else {
            return;
        };
        assert_eq!(
            final_chunk.tool_call_fragments[0].arguments.as_deref(),
            Some("\"Synthetic\"}")
        );
        assert_eq!(final_chunk.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(final_chunk.usage["cost"], "0.001");
        assert_eq!(events[3], ChatStreamEvent::Done);
        let requests = transport.requests.borrow();
        assert_eq!(
            requests[0].url,
            "https://openrouter.example/api/v1/chat/completions"
        );
        assert_eq!(requests[0].bearer_token, "synthetic-key");
        let request_body: Value = serde_json::from_str(&requests[0].body).unwrap_or(Value::Null);
        assert_eq!(request_body["stream"], true);
    }

    #[test]
    fn stream_reports_provider_errors_interruption_and_consumer_failure() {
        let mut request = request();
        request.stream = true;
        for (body, expected) in [
            (
                "data: {\"error\":{\"message\":\"provider exploded\"}}\n\n",
                OpenRouterChatError::Stream("provider exploded".to_owned()),
            ),
            (
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
                OpenRouterChatError::IncompleteStream,
            ),
        ] {
            let transport = StreamTransport {
                chunks: vec![body.as_bytes().to_vec()],
                failure: None,
                requests: RefCell::new(Vec::new()),
            };
            assert_eq!(
                stream_with(
                    &transport,
                    "key",
                    "https://example.com",
                    &request,
                    |_event| Ok(())
                ),
                Err(expected)
            );
        }

        let transport = StreamTransport {
            chunks: vec![b"data: [DONE]\n\n".to_vec()],
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            stream_with(
                &transport,
                "key",
                "https://example.com",
                &request,
                |_event| Err(OpenRouterChatError::Stream("consumer stopped".to_owned()))
            ),
            Err(OpenRouterChatError::Stream("consumer stopped".to_owned()))
        );
    }

    #[test]
    fn stream_validates_mode_and_propagates_transport_failure() {
        let transport = StreamTransport {
            chunks: Vec::new(),
            failure: Some(OpenRouterChatError::Transport("timeout".to_owned())),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            stream_with(
                &transport,
                "key",
                "https://example.com",
                &request(),
                |_event| Ok(())
            ),
            Err(OpenRouterChatError::MalformedResponse)
        );
        let mut streaming = request();
        streaming.stream = true;
        assert_eq!(
            stream_with(
                &transport,
                "key",
                "https://example.com",
                &streaming,
                |_event| Ok(())
            ),
            Err(OpenRouterChatError::Transport("timeout".to_owned()))
        );
    }
}
