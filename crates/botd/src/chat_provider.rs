//! OpenRouter streaming rounds with partial-usage preservation.

use std::sync::Arc;

use bot_adapters::openrouter_chat::{
    ChatCompletionRequest, ChatMessage, ChatRole, ChatStreamEvent, OpenRouterChatError,
    OpenRouterPricingCache, OpenRouterStreamTransport, ReasoningConfig, ToolCall, ToolFunctionCall,
    stream_with,
};
use bot_core::ai_prompt::{PromptContent, PromptMessage, PromptRole};
use bot_core::ai_reserve::chat_output_token_limit;
use bot_core::provider_stream_policy::{
    ProviderStreamEvent, StreamToolCall, accumulate_stream_tool_calls,
};
use serde_json::{Map, Value, json};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatRoundResult {
    pub text: String,
    pub reasoning: String,
    pub reasoning_details: Vec<Value>,
    pub tool_calls: Vec<StreamToolCall>,
    pub finish_reason: Option<String>,
    pub billing_segment: Option<Value>,
}

impl ChatRoundResult {
    fn empty() -> Self {
        Self {
            text: String::new(),
            reasoning: String::new(),
            reasoning_details: Vec::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            billing_segment: None,
        }
    }
}

#[derive(Debug, Error)]
#[error("native OpenRouter stream failed: {source}")]
pub struct ChatRoundError {
    pub source: OpenRouterChatError,
    pub partial: Box<ChatRoundResult>,
}

#[derive(Default)]
struct ProviderRoundMetadata {
    generation_id: Option<String>,
    resolved_model: Option<String>,
    upstream_provider: Option<String>,
    service_tier: Option<String>,
    annotation_types: Vec<String>,
}

pub struct OpenRouterChatStreamer<Transport> {
    transport: Transport,
    api_key: String,
    base_url: String,
    model: String,
    pricing: Option<Arc<OpenRouterPricingCache>>,
}

impl<Transport> OpenRouterChatStreamer<Transport> {
    #[must_use]
    pub fn new(transport: Transport, api_key: &str, base_url: &str, model: &str) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            base_url: base_url.to_owned(),
            model: model.to_owned(),
            pricing: None,
        }
    }

    #[must_use]
    pub fn with_openrouter_pricing(mut self, pricing: Arc<OpenRouterPricingCache>) -> Self {
        self.pricing = Some(pricing);
        self
    }

    fn request(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
    ) -> Result<ChatCompletionRequest, OpenRouterChatError> {
        let mut request = ChatCompletionRequest::new(
            &self.model,
            messages.iter().map(openrouter_message).collect(),
        );
        if let Some(pricing) = self.pricing.as_ref() {
            pricing.apply_to_request(&mut request)?;
        }
        request.tools = tools.to_vec();
        request.max_tokens = u64::try_from(chat_output_token_limit(&self.model)).ok();
        request.reasoning = Some(ReasoningConfig { enabled: true });
        request.stream = true;
        Ok(request)
    }
}

impl<Transport: OpenRouterStreamTransport> OpenRouterChatStreamer<Transport> {
    pub fn stream_round<F>(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
        mut on_text: F,
    ) -> Result<ChatRoundResult, ChatRoundError>
    where
        F: FnMut(&str) -> Result<(), OpenRouterChatError>,
    {
        self.stream_round_events(messages, tools, |event| match event {
            ProviderStreamEvent::ReasoningDelta(_) => Ok(()),
            ProviderStreamEvent::TextDelta(text) => on_text(&text),
        })
    }

    pub fn stream_round_events<F>(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
        mut on_event: F,
    ) -> Result<ChatRoundResult, ChatRoundError>
    where
        F: FnMut(ProviderStreamEvent) -> Result<(), OpenRouterChatError>,
    {
        let mut result = ChatRoundResult::empty();
        let request = match self.request(messages, tools) {
            Ok(request) => request,
            Err(source) => {
                return Err(ChatRoundError {
                    source,
                    partial: Box::new(result),
                });
            }
        };
        let mut metadata = ProviderRoundMetadata::default();
        let mut usage = Map::new();
        let stream_result = stream_with(
            &self.transport,
            &self.api_key,
            &self.base_url,
            &request,
            |event| {
                let ChatStreamEvent::Chunk(chunk) = event else {
                    return Ok(());
                };
                let reasoning = if !chunk.reasoning.is_empty() {
                    chunk.reasoning.clone()
                } else {
                    reasoning_details_text(&chunk.reasoning_details)
                };
                if !reasoning.is_empty() {
                    on_event(ProviderStreamEvent::ReasoningDelta(reasoning.clone()))?;
                    result.reasoning.push_str(&reasoning);
                }
                if !chunk.text.is_empty() {
                    on_event(ProviderStreamEvent::TextDelta(chunk.text.clone()))?;
                    result.text.push_str(&chunk.text);
                }
                accumulate_reasoning_details(
                    &mut result.reasoning_details,
                    chunk.reasoning_details,
                );
                result.tool_calls = accumulate_stream_tool_calls(
                    std::mem::take(&mut result.tool_calls),
                    chunk.tool_call_fragments,
                );
                if chunk.finish_reason.is_some() {
                    result.finish_reason = chunk.finish_reason;
                }
                replace_nonempty(&mut metadata.generation_id, chunk.generation_id);
                replace_nonempty(&mut metadata.resolved_model, chunk.model);
                replace_nonempty(&mut metadata.upstream_provider, chunk.upstream_provider);
                replace_nonempty(&mut metadata.service_tier, chunk.service_tier);
                if !chunk.usage.is_empty() {
                    usage = chunk.usage;
                }
                metadata
                    .annotation_types
                    .extend(chunk.annotations.into_iter().filter_map(|annotation| {
                        annotation
                            .get("type")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    }));
                Ok(())
            },
        );
        result.billing_segment =
            provider_segment(&self.model, metadata, usage, stream_result.is_err());
        match stream_result {
            Ok(()) => Ok(result),
            Err(source) => Err(ChatRoundError {
                source,
                partial: Box::new(result),
            }),
        }
    }
}

fn replace_nonempty(target: &mut Option<String>, value: Option<String>) {
    if value.as_deref().is_some_and(|value| !value.is_empty()) {
        *target = value;
    }
}

fn provider_segment(
    requested_model: &str,
    metadata: ProviderRoundMetadata,
    usage: Map<String, Value>,
    interrupted: bool,
) -> Option<Value> {
    if usage.is_empty() && metadata.generation_id.is_none() {
        return None;
    }
    let model = metadata
        .resolved_model
        .unwrap_or_else(|| requested_model.to_owned());
    let mut segment_metadata = Map::from_iter([("provider".to_owned(), json!("openrouter"))]);
    if model != requested_model {
        segment_metadata.insert("requested_model".to_owned(), json!(requested_model));
    }
    insert_optional(
        &mut segment_metadata,
        "provider_generation_id",
        metadata.generation_id,
    );
    insert_optional(
        &mut segment_metadata,
        "upstream_provider",
        metadata.upstream_provider,
    );
    insert_optional(&mut segment_metadata, "service_tier", metadata.service_tier);
    if interrupted || usage.is_empty() {
        segment_metadata.insert("provider_usage_pending".to_owned(), json!(true));
    }
    let citation_count = metadata
        .annotation_types
        .iter()
        .filter(|kind| kind.as_str() == "url_citation")
        .count();
    if citation_count > 0 {
        segment_metadata.insert(
            "web_search_citation_count".to_owned(),
            json!(citation_count),
        );
    }
    Some(json!({
        "kind": "chat",
        "model": model,
        "usage": usage,
        "source": "openrouter",
        "metadata": segment_metadata,
    }))
}

fn insert_optional(metadata: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        metadata.insert(key.to_owned(), json!(value));
    }
}

fn openrouter_message(message: &PromptMessage) -> ChatMessage {
    let content = match &message.content {
        PromptContent::Text(text) => Some(Value::String(text.clone())),
        PromptContent::TextParts(parts) => Some(Value::Array(
            parts
                .iter()
                .map(|text| json!({"type": "text", "text": text}))
                .collect(),
        )),
        PromptContent::Empty => None,
    };
    let reasoning_details = message.reasoning_details.clone();
    let reasoning = if reasoning_details.is_empty() {
        message.reasoning.clone()
    } else {
        None
    };
    ChatMessage {
        role: match message.role {
            PromptRole::System => ChatRole::System,
            PromptRole::User => ChatRole::User,
            PromptRole::Assistant => ChatRole::Assistant,
            PromptRole::Tool => ChatRole::Tool,
        },
        content,
        reasoning,
        reasoning_details,
        name: None,
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(openrouter_tool_call)
            .collect(),
    }
}

// Text and summary deltas extend the preceding logical block. Opaque encrypted
// blocks must remain separate and in order, even when their indices are equal.
fn accumulate_reasoning_details(details: &mut Vec<Value>, fragments: Vec<Value>) {
    for fragment in fragments {
        let kind = fragment.get("type").and_then(Value::as_str);
        let previous = details.last_mut().filter(|previous| {
            matches!(kind, Some("reasoning.text" | "reasoning.summary"))
                && previous.get("type") == fragment.get("type")
                && ["id", "format", "index"].iter().all(|key| {
                    match (previous.get(key), fragment.get(key)) {
                        (Some(left), Some(right)) if !left.is_null() && !right.is_null() => {
                            left == right
                        }
                        _ => true,
                    }
                })
        });
        if let Some(Value::Object(previous)) = previous
            && let Some(fragment) = fragment.as_object()
        {
            for (key, value) in fragment {
                if matches!(key.as_str(), "text" | "summary" | "signature")
                    && let Some(delta) = value.as_str()
                {
                    let mut text = previous
                        .get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    text.push_str(delta);
                    previous.insert(key.clone(), Value::String(text));
                } else if !value.is_null() {
                    previous.insert(key.clone(), value.clone());
                }
            }
        } else {
            details.push(fragment);
        }
    }
}

fn reasoning_details_text(details: &[Value]) -> String {
    details
        .iter()
        .filter_map(|detail| {
            detail
                .get("text")
                .and_then(Value::as_str)
                .or_else(|| detail.get("summary").and_then(Value::as_str))
        })
        .collect()
}

fn openrouter_tool_call(call: &bot_core::ai_prompt::PromptToolCall) -> ToolCall {
    ToolCall {
        id: call.id.clone(),
        call_type: call.call_type.clone(),
        function: ToolFunctionCall {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use bot_adapters::openrouter_chat::{
        HttpRequest, OpenRouterChatError, OpenRouterStreamTransport,
    };
    use bot_core::ai_prompt::{PromptContent, PromptMessage, PromptRole, PromptToolCall};
    use bot_core::provider_stream_policy::ProviderStreamEvent;
    use serde_json::{Value, json};

    use super::OpenRouterChatStreamer;

    struct Transport {
        chunks: Vec<Vec<u8>>,
        failure: Option<OpenRouterChatError>,
        requests: RefCell<Vec<HttpRequest>>,
    }

    impl OpenRouterStreamTransport for Transport {
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

    fn stream_body(done: bool) -> Vec<Vec<u8>> {
        let mut body = [
            format!(
                "data: {}\n\n",
                json!({
                    "id": "generation-1",
                    "model": "resolved/model",
                    "provider": "SyntheticProvider",
                    "choices": [{"delta": {
                        "reasoning": "checking ",
                        "reasoning_details": [{"type": "reasoning.text", "text": "checking "}],
                        "content": "hello ",
                        "annotations": [{"type": "url_citation"}],
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
                        "delta": {
                            "content": "world",
                            "tool_calls": [{
                                "index": 0,
                                "function": {"name": "ther", "arguments": "\"x\"}"}
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": "0.001"}
                })
            ),
        ]
        .concat();
        if done {
            body.push_str("data: [DONE]\n\n");
        }
        body.as_bytes().chunks(5).map(<[u8]>::to_vec).collect()
    }

    fn details_only_stream_body() -> Vec<Vec<u8>> {
        let body = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({
                "choices": [{"delta": {
                    "reasoning_details": [
                        {"type": "reasoning.summary", "summary": "checking "},
                        {"type": "reasoning.text", "text": "details"}
                    ],
                    "content": "answer"
                }}]
            })
        );
        body.as_bytes().chunks(5).map(<[u8]>::to_vec).collect()
    }

    fn messages() -> Vec<PromptMessage> {
        vec![
            PromptMessage::text(PromptRole::System, "system"),
            PromptMessage {
                role: PromptRole::User,
                content: PromptContent::TextParts(vec!["question".to_owned()]),
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning: None,
                reasoning_details: Vec::new(),
            },
        ]
    }

    #[test]
    fn stream_round_accumulates_text_tools_and_billable_usage() {
        let transport = Transport {
            chunks: stream_body(true),
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let provider = OpenRouterChatStreamer::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "requested/model",
        );
        let mut emitted = Vec::new();
        let result = provider.stream_round(&messages(), &[json!({"type": "function"})], |text| {
            emitted.push(text.to_owned());
            Ok(())
        });
        assert!(result.is_ok());
        let Some(result) = result.ok() else {
            return;
        };
        assert_eq!(result.text, "hello world");
        assert_eq!(result.reasoning, "checking ");
        assert_eq!(
            result.reasoning_details,
            [json!({
                "type": "reasoning.text",
                "text": "checking "
            })]
        );
        assert_eq!(emitted, ["hello ", "world"]);
        assert_eq!(result.tool_calls[0].name, "weather");
        assert_eq!(result.tool_calls[0].arguments, "{\"city\":\"x\"}");
        assert_eq!(result.finish_reason.as_deref(), Some("tool_calls"));
        let segment = result.billing_segment.unwrap_or(Value::Null);
        assert_eq!(segment["model"], "resolved/model");
        assert_eq!(segment["metadata"]["requested_model"], "requested/model");
        assert_eq!(
            segment["metadata"]["provider_generation_id"],
            "generation-1"
        );
        assert_eq!(segment["metadata"]["web_search_citation_count"], 1);
        assert_eq!(segment["usage"]["cost"], "0.001");
        let body: Value = serde_json::from_str(&provider.transport.requests.borrow()[0].body)
            .unwrap_or(Value::Null);
        assert_eq!(body["stream"], true);
        assert_eq!(body["reasoning"]["enabled"], true);
        assert_eq!(body["messages"][1]["content"][0]["type"], "text");
        assert_eq!(body["tools"][0]["type"], "function");
    }

    #[test]
    fn stream_round_events_exposes_reasoning_before_text_and_keeps_tool_context() {
        let transport = Transport {
            chunks: stream_body(true),
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let provider = OpenRouterChatStreamer::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "requested/model",
        );
        let mut events = Vec::new();
        let result = provider
            .stream_round_events(&messages(), &[], |event| {
                events.push(event);
                Ok(())
            })
            .unwrap_or_else(|error| error.partial.as_ref().clone());

        assert_eq!(
            events[..3],
            [
                ProviderStreamEvent::ReasoningDelta("checking ".to_owned()),
                ProviderStreamEvent::TextDelta("hello ".to_owned()),
                ProviderStreamEvent::TextDelta("world".to_owned()),
            ]
        );
        assert_eq!(result.reasoning, "checking ");
        assert_eq!(result.text, "hello world");
    }

    #[test]
    fn interrupted_round_returns_partial_text_and_reconcilable_usage() {
        let transport = Transport {
            chunks: stream_body(false),
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let provider = OpenRouterChatStreamer::new(
            transport,
            "key",
            "https://synthetic.invalid",
            "requested/model",
        );
        let result = provider.stream_round(&messages(), &[], |_text| Ok(()));
        assert!(result.is_err());
        let Some(error) = result.err() else {
            return;
        };
        assert_eq!(error.source, OpenRouterChatError::IncompleteStream);
        assert_eq!(error.partial.text, "hello world");
        let segment = error.partial.billing_segment.unwrap_or(Value::Null);
        assert_eq!(segment["metadata"]["provider_usage_pending"], true);
        assert_eq!(segment["usage"]["completion_tokens"], 4);
    }

    #[test]
    fn consumer_failure_preserves_the_completed_prefix_without_fake_usage() {
        let transport = Transport {
            chunks: stream_body(true),
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let provider = OpenRouterChatStreamer::new(
            transport,
            "key",
            "https://synthetic.invalid",
            "requested/model",
        );
        let result = provider.stream_round(&messages(), &[], |_text| {
            Err(OpenRouterChatError::Stream("delivery stopped".to_owned()))
        });
        let Some(error) = result.err() else {
            return;
        };
        assert_eq!(error.partial.text, "");
        assert!(error.partial.billing_segment.is_none());
    }

    #[test]
    fn request_conversion_preserves_empty_content_roles_and_tool_calls() {
        let transport = Transport {
            chunks: vec![b"data: [DONE]\n\n".to_vec()],
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let provider = OpenRouterChatStreamer::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid",
            "requested/model",
        );
        let messages = [
            PromptMessage {
                role: PromptRole::Assistant,
                content: PromptContent::Empty,
                tool_call_id: None,
                tool_calls: vec![PromptToolCall {
                    id: "synthetic-call".to_owned(),
                    call_type: "function".to_owned(),
                    name: "calculate".to_owned(),
                    arguments: r#"{"expression":"2+2"}"#.to_owned(),
                }],
                reasoning: Some("checking".to_owned()),
                reasoning_details: vec![json!({"type": "reasoning.text", "text": "checking"})],
            },
            PromptMessage {
                role: PromptRole::Tool,
                content: PromptContent::Text("4".to_owned()),
                tool_call_id: Some("synthetic-call".to_owned()),
                tool_calls: Vec::new(),
                reasoning: None,
                reasoning_details: Vec::new(),
            },
        ];
        let result = provider.stream_round(&messages, &[], |_text| Ok(()));
        assert!(result.is_ok_and(|result| result.text.is_empty()));
        let body = serde_json::from_str::<Value>(&provider.transport.requests.borrow()[0].body)
            .unwrap_or(Value::Null);
        assert_eq!(body["messages"][0]["role"], "assistant");
        assert!(body["messages"][0]["content"].is_null());
        assert!(body["messages"][0].get("reasoning").is_none());
        assert_eq!(
            body["messages"][0]["reasoning_details"][0]["type"],
            "reasoning.text"
        );
        assert_eq!(body["messages"][0]["tool_calls"][0]["id"], "synthetic-call");
        assert_eq!(body["messages"][1]["role"], "tool");
        assert_eq!(body["messages"][1]["tool_call_id"], "synthetic-call");
    }

    #[test]
    fn stream_round_events_exposes_textual_reasoning_details_without_legacy_field() {
        let transport = Transport {
            chunks: details_only_stream_body(),
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let provider = OpenRouterChatStreamer::new(
            transport,
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "requested/model",
        );
        let mut events = Vec::new();
        let result = provider
            .stream_round_events(&messages(), &[], |event| {
                events.push(event);
                Ok(())
            })
            .unwrap_or_else(|error| error.partial.as_ref().clone());

        assert_eq!(
            events,
            [
                ProviderStreamEvent::ReasoningDelta("checking details".to_owned()),
                ProviderStreamEvent::TextDelta("answer".to_owned()),
            ]
        );
        assert_eq!(result.reasoning, "checking details");
    }
    #[test]
    fn streamed_reasoning_is_reassembled_for_the_tool_continuation() {
        let fragments = vec![
            json!({"type":"reasoning.text","text":"check ","id":"text-1","index":0}),
            json!({"type":"reasoning.text","text":"the fixture","signature":"sig-","index":0}),
            json!({"type":"reasoning.text","signature":"end","index":0}),
            json!({"type":"reasoning.summary","summary":"checking ","index":0}),
            json!({"type":"reasoning.summary","summary":"schedule","index":0}),
            json!({"type":"reasoning.encrypted","data":"opaque-one","index":0}),
            json!({"type":"reasoning.encrypted","data":"opaque-two","index":0}),
            json!({"type":"reasoning.text","text":"new block","id":"text-2","index":0}),
            json!({"type":"reasoning.text","text":"separate block","id":"text-3","index":0}),
        ];
        let mut body = fragments
            .iter()
            .map(|detail| {
                format!(
                    "data: {}\n\n",
                    json!({
                        "choices":[{"delta":{"reasoning_details":[detail]}}]
                    })
                )
            })
            .collect::<String>();
        body.push_str("data: [DONE]\n\n");
        let provider = OpenRouterChatStreamer::new(
            Transport {
                chunks: vec![body.into_bytes()],
                failure: None,
                requests: RefCell::new(Vec::new()),
            },
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "synthetic/model",
        );
        let round = provider
            .stream_round_events(&messages(), &[], |_| Ok(()))
            .unwrap_or_else(|error| *error.partial);
        let continuation = PromptMessage::assistant_tool_calls_with_reasoning(
            None,
            vec![],
            Some(round.reasoning),
            round.reasoning_details,
        );
        provider
            .stream_round(&[continuation], &[], |_| Ok(()))
            .unwrap_or_else(|error| *error.partial);
        let requests = provider.transport.requests.borrow();
        let body: Value = serde_json::from_str(&requests[1].body).unwrap_or(Value::Null);
        assert_eq!(
            body["messages"][0]["reasoning_details"],
            json!([
                {"type":"reasoning.text","text":"check the fixture","signature":"sig-end","id":"text-1","index":0},
                {"type":"reasoning.summary","summary":"checking schedule","index":0},
                {"type":"reasoning.encrypted","data":"opaque-one","index":0},
                {"type":"reasoning.encrypted","data":"opaque-two","index":0},
                {"type":"reasoning.text","text":"new block","id":"text-2","index":0},
                {"type":"reasoning.text","text":"separate block","id":"text-3","index":0}
            ])
        );
        assert!(body["messages"][0].get("reasoning").is_none());
    }
}
