//! Bounded native chat/tool orchestration with durable per-round usage.

use bot_adapters::openrouter_chat::{OpenRouterChatError, OpenRouterStreamTransport};
use bot_core::ai_prompt::{PromptMessage, PromptToolCall};
use bot_core::provider_runtime_policy::{
    ProviderExceptionFacts, is_retryable_provider_exception, response_has_billable_usage,
    retry_wait_seconds,
};
use bot_core::provider_stream_policy::{ProviderStreamEvent, StreamToolCall};
use serde_json::{Value, json};
use std::time::Duration;
use thiserror::Error;

use crate::chat_provider::{ChatRoundError, ChatRoundResult, OpenRouterChatStreamer};

pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 5;
const MAX_PROVIDER_RETRIES: usize = 2;
const MAX_RATE_LIMIT_RETRY_SECONDS: u64 = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct ToolExecutionResult {
    pub output: String,
    pub failure_fallback: Option<String>,
    pub billing_segment: Option<Value>,
    pub diagnostics: Vec<String>,
}

impl ToolExecutionResult {
    #[must_use]
    pub fn output(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            failure_fallback: None,
            billing_segment: None,
            diagnostics: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_diagnostics(output: impl Into<String>, diagnostics: Vec<String>) -> Self {
        Self {
            output: output.into(),
            failure_fallback: None,
            billing_segment: None,
            diagnostics,
        }
    }

    #[must_use]
    pub fn confirmed_output(output: impl Into<String>) -> Self {
        let output = output.into();
        Self {
            failure_fallback: Some(output.clone()),
            output,
            billing_segment: None,
            diagnostics: Vec::new(),
        }
    }
}

/// A prepared tool call that owns everything it needs, so it can run on its
/// own thread while the round's other calls run.
pub type ConcurrentToolCall = Box<dyn FnOnce() -> ToolExecutionResult + Send>;

pub trait NativeToolRuntime {
    fn schemas(&self, task_mode: bool) -> Vec<Value>;

    fn contains(&self, name: &str, task_mode: bool) -> bool;

    fn execute(&mut self, name: &str, arguments: &Value, tool_call_id: &str)
    -> ToolExecutionResult;

    /// Prepare a read-only call to run concurrently with the other calls of
    /// its round. Tools with side effects (billing, tasks, configuration)
    /// keep the default `None` and run sequentially through `execute`.
    fn concurrent_call(
        &mut self,
        _name: &str,
        _arguments: &Value,
        _tool_call_id: &str,
    ) -> Option<ConcurrentToolCall> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatToolLoopEvent {
    ReasoningDelta(String),
    ResetToTrace,
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        id: String,
        name: String,
        output: String,
    },
    FinalText(String),
}

pub trait ChatRoundStream {
    fn stream_round(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
        on_text: &mut dyn FnMut(&str) -> Result<(), OpenRouterChatError>,
    ) -> Result<ChatRoundResult, ChatRoundError>;

    fn stream_round_events(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
        on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<(), OpenRouterChatError>,
    ) -> Result<ChatRoundResult, ChatRoundError> {
        self.stream_round(messages, tools, &mut |text| {
            on_event(ProviderStreamEvent::TextDelta(text.to_owned()))
        })
    }
}

impl<Transport: OpenRouterStreamTransport> ChatRoundStream for OpenRouterChatStreamer<Transport> {
    fn stream_round(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
        on_text: &mut dyn FnMut(&str) -> Result<(), OpenRouterChatError>,
    ) -> Result<ChatRoundResult, ChatRoundError> {
        self.stream_round(messages, tools, on_text)
    }

    fn stream_round_events(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
        on_event: &mut dyn FnMut(ProviderStreamEvent) -> Result<(), OpenRouterChatError>,
    ) -> Result<ChatRoundResult, ChatRoundError> {
        OpenRouterChatStreamer::stream_round_events(self, messages, tools, on_event)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatToolLoopResult {
    pub text: String,
    pub messages: Vec<PromptMessage>,
    pub billing_segments: Vec<Value>,
    pub provider_rounds: usize,
    pub tool_calls_executed: usize,
    pub diagnostics: Vec<String>,
    pub failure_fallbacks: Vec<String>,
    pub stopped_at_limit: bool,
}

impl ChatToolLoopResult {
    fn new(messages: &[PromptMessage]) -> Self {
        Self {
            text: String::new(),
            messages: messages.to_vec(),
            billing_segments: Vec::new(),
            provider_rounds: 0,
            tool_calls_executed: 0,
            diagnostics: Vec::new(),
            failure_fallbacks: Vec::new(),
            stopped_at_limit: false,
        }
    }
}

#[derive(Debug, Error)]
#[error("native chat/tool loop failed after {provider_rounds} provider rounds: {source}")]
pub struct ChatToolLoopError {
    pub source: OpenRouterChatError,
    pub failed_round: Box<ChatRoundResult>,
    pub partial: Box<ChatToolLoopResult>,
    pub provider_rounds: usize,
}

pub fn run_chat_tool_loop<Provider, Tools>(
    operation_id: &str,
    provider: &Provider,
    tools: &mut Tools,
    initial_messages: &[PromptMessage],
    task_mode: bool,
    max_rounds: usize,
    mut on_text: impl FnMut(&str) -> Result<(), OpenRouterChatError>,
) -> Result<ChatToolLoopResult, ChatToolLoopError>
where
    Provider: ChatRoundStream,
    Tools: NativeToolRuntime,
{
    run_chat_tool_loop_events_with_policy(
        operation_id,
        provider,
        tools,
        initial_messages,
        task_mode,
        max_rounds,
        true,
        |event| match event {
            ChatToolLoopEvent::FinalText(text) => on_text(&text),
            ChatToolLoopEvent::ReasoningDelta(_)
            | ChatToolLoopEvent::ResetToTrace
            | ChatToolLoopEvent::ToolCall { .. }
            | ChatToolLoopEvent::ToolResult { .. } => Ok(()),
        },
    )
}

pub fn run_chat_tool_loop_events<Provider, Tools>(
    operation_id: &str,
    provider: &Provider,
    tools: &mut Tools,
    initial_messages: &[PromptMessage],
    task_mode: bool,
    max_rounds: usize,
    on_event: impl FnMut(ChatToolLoopEvent) -> Result<(), OpenRouterChatError>,
) -> Result<ChatToolLoopResult, ChatToolLoopError>
where
    Provider: ChatRoundStream,
    Tools: NativeToolRuntime,
{
    run_chat_tool_loop_events_with_policy(
        operation_id,
        provider,
        tools,
        initial_messages,
        task_mode,
        max_rounds,
        false,
        on_event,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_chat_tool_loop_events_with_policy<Provider, Tools>(
    operation_id: &str,
    provider: &Provider,
    tools: &mut Tools,
    initial_messages: &[PromptMessage],
    task_mode: bool,
    max_rounds: usize,
    include_intermediate_text: bool,
    mut on_event: impl FnMut(ChatToolLoopEvent) -> Result<(), OpenRouterChatError>,
) -> Result<ChatToolLoopResult, ChatToolLoopError>
where
    Provider: ChatRoundStream,
    Tools: NativeToolRuntime,
{
    let schemas = tools.schemas(task_mode);
    let mut result = ChatToolLoopResult::new(initial_messages);
    trace(
        operation_id,
        0,
        "start",
        json!({
            "available_tools": schemas.iter().filter_map(|schema| schema.pointer("/function/name").and_then(Value::as_str)).collect::<Vec<_>>(),
        }),
    );

    for logical_round in 0..max_rounds {
        let mut retry = 0;
        let round =
            loop {
                let round =
                    provider.stream_round_events(&result.messages, &schemas, &mut |event| {
                        match event {
                            ProviderStreamEvent::ReasoningDelta(text) => {
                                on_event(ChatToolLoopEvent::ReasoningDelta(text))
                            }
                            ProviderStreamEvent::TextDelta(text) => {
                                on_event(ChatToolLoopEvent::FinalText(text))
                            }
                        }
                    });
                match round {
                    Ok(round) => break round,
                    Err(error) => {
                        result.provider_rounds += 1;
                        trace(
                            operation_id,
                            result.provider_rounds,
                            "provider_error",
                            json!({
                                "error_kind": provider_error_kind(&error.source),
                                "provider": round_trace(&error.partial),
                            }),
                        );
                        record_failed_round(&mut result, &error.partial);
                        let retry_delay = (retry < MAX_PROVIDER_RETRIES
                            && retryable_provider_error(&error.source)
                            && error.partial.text.is_empty()
                            && error.partial.reasoning.is_empty()
                            && error.partial.tool_calls.is_empty()
                            && !round_has_billable_usage(&error.partial))
                        .then(|| provider_retry_delay(&error.source, retry))
                        .flatten();
                        if let Some(delay) = retry_delay {
                            result.diagnostics.push(format!(
                                "AI provider retry: round={} attempt={} error_kind={} delay_ms={}",
                                logical_round + 1,
                                retry + 1,
                                provider_error_kind(&error.source),
                                delay.as_millis(),
                            ));
                            wait_before_retry(delay);
                            retry += 1;
                            continue;
                        }
                        return Err(ChatToolLoopError {
                            source: error.source,
                            failed_round: error.partial,
                            provider_rounds: result.provider_rounds,
                            partial: Box::new(result),
                        });
                    }
                }
            };
        result.provider_rounds += 1;
        trace(
            operation_id,
            result.provider_rounds,
            "provider_round",
            round_trace(&round),
        );
        record_round(&mut result, &round);

        let known_calls = round
            .tool_calls
            .iter()
            .filter(|call| tools.contains(&call.name, task_mode))
            .cloned()
            .collect::<Vec<_>>();
        if known_calls.is_empty() {
            result.text.push_str(&round.text);
            trace(
                operation_id,
                result.provider_rounds,
                "finish",
                json!({
                    "tool_calls_executed": result.tool_calls_executed,
                    "stopped_at_limit": false,
                }),
            );
            return Ok(result);
        }

        if include_intermediate_text {
            result.text.push_str(&round.text);
        } else {
            emit_event(
                &mut on_event,
                &result,
                &round,
                ChatToolLoopEvent::ResetToTrace,
            )?;
        }
        result
            .messages
            .push(PromptMessage::assistant_tool_calls_with_reasoning(
                (!round.text.is_empty()).then_some(round.text.as_str()),
                known_calls.iter().map(prompt_tool_call).collect(),
                (!round.reasoning.is_empty()).then_some(round.reasoning.clone()),
                round.reasoning_details.clone(),
            ));
        let arguments = known_calls
            .iter()
            .map(|call| parse_arguments(&call.arguments))
            .collect::<Vec<_>>();
        let concurrent = prepare_concurrent_calls(tools, &known_calls, &arguments);
        // Read-only calls start on their own threads right away; results are
        // still consumed in the provider's call order, so events, billing and
        // the tool messages sent back to the model keep that order.
        std::thread::scope(|scope| -> Result<(), ChatToolLoopError> {
            let mut running = concurrent
                .into_iter()
                .zip(&known_calls)
                .map(|(prepared, call)| {
                    prepared.map(|run| {
                        let name = call.name.clone();
                        scope.spawn(move || {
                            let started = std::time::Instant::now();
                            // A panicking tool must not take the reply down
                            // with it; report it like any failed tool.
                            let tool_result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(run))
                                    .unwrap_or_else(|_| {
                                        ToolExecutionResult::with_diagnostics(
                                            format!("{name} failed"),
                                            vec![format!("concurrent tool {name} panicked")],
                                        )
                                    });
                            (tool_result, started.elapsed())
                        })
                    })
                })
                .collect::<Vec<_>>()
                .into_iter();
            for (call, arguments) in known_calls.iter().zip(&arguments) {
                let handle = running.next().flatten();
                emit_event(
                    &mut on_event,
                    &result,
                    &round,
                    ChatToolLoopEvent::ToolCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                )?;
                trace(
                    operation_id,
                    result.provider_rounds,
                    "tool_start",
                    tool_trace(call, arguments, None),
                );
                let joined = handle.and_then(|handle| handle.join().ok());
                let (tool_result, elapsed) = if let Some(joined) = joined {
                    joined
                } else {
                    let started = std::time::Instant::now();
                    let tool_result = tools.execute(&call.name, arguments, &call.id);
                    (tool_result, started.elapsed())
                };
                let mut details = tool_trace(call, arguments, Some(&tool_result));
                details["elapsed_ms"] = json!(elapsed.as_millis());
                trace(operation_id, result.provider_rounds, "tool_result", details);
                result.tool_calls_executed += 1;
                if let Some(segment) = tool_result.billing_segment {
                    result.billing_segments.push(segment);
                }
                result.diagnostics.extend(tool_result.diagnostics);
                if let Some(fallback) = tool_result.failure_fallback {
                    result.failure_fallbacks.push(fallback);
                }
                emit_event(
                    &mut on_event,
                    &result,
                    &round,
                    ChatToolLoopEvent::ToolResult {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        output: tool_result.output.clone(),
                    },
                )?;
                result
                    .messages
                    .push(PromptMessage::tool_result(&call.id, tool_result.output));
            }
            Ok(())
        })?;
    }

    result.stopped_at_limit = true;
    trace(
        operation_id,
        result.provider_rounds,
        "finish",
        json!({
            "tool_calls_executed": result.tool_calls_executed,
            "stopped_at_limit": true,
        }),
    );
    Ok(result)
}

/// Prepared concurrent calls, aligned with `calls`. Concurrency only pays off
/// with at least two read-only calls in the round; otherwise every call runs
/// sequentially through `execute`.
fn prepare_concurrent_calls<Tools: NativeToolRuntime>(
    tools: &mut Tools,
    calls: &[StreamToolCall],
    arguments: &[Value],
) -> Vec<Option<ConcurrentToolCall>> {
    if calls.len() < 2 {
        return Vec::new();
    }
    let prepared = calls
        .iter()
        .zip(arguments)
        .map(|(call, arguments)| tools.concurrent_call(&call.name, arguments, &call.id))
        .collect::<Vec<_>>();
    if prepared.iter().filter(|call| call.is_some()).count() < 2 {
        return Vec::new();
    }
    prepared
}

fn emit_event(
    on_event: &mut impl FnMut(ChatToolLoopEvent) -> Result<(), OpenRouterChatError>,
    result: &ChatToolLoopResult,
    round: &ChatRoundResult,
    event: ChatToolLoopEvent,
) -> Result<(), ChatToolLoopError> {
    on_event(event).map_err(|source| ChatToolLoopError {
        source,
        failed_round: Box::new(round.clone()),
        partial: Box::new(result.clone()),
        provider_rounds: result.provider_rounds,
    })
}

// Keep traces separate from user-facing diagnostics. JSON escapes embedded newlines.
fn trace(operation_id: &str, round: usize, event: &str, details: Value) {
    eprintln!(
        "AI trace: {}",
        json!({
            "operation_id": bounded(operation_id, 160),
            "round": round,
            "event": event,
            "details": details,
        })
    );
}

fn bounded(text: &str, limit: usize) -> String {
    let mut chars = text.chars();
    let mut result: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        result.push('…');
    }
    result
}

fn round_trace(round: &ChatRoundResult) -> Value {
    let segment = round.billing_segment.as_ref();
    json!({
        "model": segment.and_then(|s| s.get("model")).and_then(Value::as_str).map(|s| bounded(s, 160)),
        "generation_id": segment.and_then(|s| s.pointer("/metadata/provider_generation_id")).and_then(Value::as_str).map(|s| bounded(s, 160)),
        "finish_reason": round.finish_reason.as_deref().map(|s| bounded(s, 80)),
        "text_chars": round.text.chars().count(),
        "tool_calls": round.tool_calls.iter().take(20).map(|call| json!({
            "id": bounded(&call.id, 160), "name": bounded(&call.name, 80),
        })).collect::<Vec<_>>(),
    })
}

// URLs can contain credentials or signed query parameters; retain only the location.
fn source_location(raw: &str) -> String {
    let Ok(mut url) = url::Url::parse(raw) else {
        return "[invalid URL]".to_owned();
    };
    if !matches!(url.scheme(), "http" | "https") {
        return "[unsupported URL]".to_owned();
    }
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    bounded(url.as_str(), 500)
}

fn tool_trace(
    call: &StreamToolCall,
    arguments: &Value,
    result: Option<&ToolExecutionResult>,
) -> Value {
    let mut details =
        json!({"tool": bounded(&call.name, 80), "tool_call_id": bounded(&call.id, 160)});
    match call.name.as_str() {
        "web_search" => {
            details["query"] = json!(bounded(
                arguments
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                500
            ));
        }
        "web_fetch" => {
            details["url"] = json!(source_location(
                arguments
                    .get("url")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            ));
        }
        _ => {}
    }
    if let Some(result) = result {
        details["output_chars"] = json!(result.output.chars().count());
        details["diagnostic_count"] = json!(result.diagnostics.len());
        if call.name == "web_search" {
            let payload = serde_json::from_str::<Value>(&result.output).ok();
            if let Some(results) = payload
                .as_ref()
                .and_then(|p| p.get("results"))
                .and_then(Value::as_array)
            {
                details["result_count"] = json!(results.len());
                details["sources"] = json!(results.iter().take(5).map(|source| json!({
                    "url": source_location(source.get("url").and_then(Value::as_str).unwrap_or_default()),
                    "title": bounded(source.get("title").and_then(Value::as_str).unwrap_or_default(), 300),
                    "description": bounded(source.get("description").and_then(Value::as_str).unwrap_or_default(), 1200),
                })).collect::<Vec<_>>());
            } else {
                details["search_error"] = json!(bounded(&result.output, 500));
            }
            details["provider_request_id"] = json!(
                result
                    .billing_segment
                    .as_ref()
                    .and_then(|s| s.pointer("/metadata/provider_request_id"))
                    .and_then(Value::as_str)
                    .map(|s| bounded(s, 160))
            );
        }
    }
    details
}

fn retryable_provider_error(error: &OpenRouterChatError) -> bool {
    is_retryable_provider_exception(provider_exception_facts(error))
}

fn provider_exception_facts(error: &OpenRouterChatError) -> ProviderExceptionFacts {
    ProviderExceptionFacts {
        json_decode_error: matches!(error, OpenRouterChatError::InvalidJson(_)),
        connection_error: matches!(
            error,
            OpenRouterChatError::Transport(_) | OpenRouterChatError::IncompleteStream
        ),
        timeout_error: false,
        rate_limit_error: matches!(error, OpenRouterChatError::RateLimited { .. }),
        api_status_code: match error {
            OpenRouterChatError::Http { status_code, .. } => Some(i64::from(*status_code)),
            _ => None,
        },
    }
}

fn round_has_billable_usage(round: &ChatRoundResult) -> bool {
    round
        .billing_segment
        .as_ref()
        .and_then(|segment| segment.get("usage"))
        .and_then(Value::as_object)
        .is_some_and(response_has_billable_usage)
}

fn provider_retry_delay(error: &OpenRouterChatError, retry: usize) -> Option<Duration> {
    let retry = u32::try_from(retry).ok()?;
    let default = Duration::from_secs(retry_wait_seconds(retry)?);
    let OpenRouterChatError::RateLimited {
        retry_after_seconds: Some(retry_after),
        ..
    } = error
    else {
        return Some(default);
    };
    if *retry_after > MAX_RATE_LIMIT_RETRY_SECONDS {
        return None;
    }
    Some(default.max(Duration::from_secs(*retry_after)))
}

pub(crate) fn provider_error_kind(error: &OpenRouterChatError) -> String {
    match error {
        OpenRouterChatError::MissingApiKey => "missing_api_key".to_owned(),
        OpenRouterChatError::MissingModel => "missing_model".to_owned(),
        OpenRouterChatError::MissingModelPricing { .. } => "missing_model_pricing".to_owned(),
        OpenRouterChatError::InvalidBaseUrl => "invalid_base_url".to_owned(),
        OpenRouterChatError::RequestJson(_) => "request_json".to_owned(),
        OpenRouterChatError::Transport(_) => "transport".to_owned(),
        OpenRouterChatError::RateLimited { .. } => "rate_limited".to_owned(),
        OpenRouterChatError::Http { status_code, .. } => format!("http_{status_code}"),
        OpenRouterChatError::InvalidJson(_) => "invalid_json".to_owned(),
        OpenRouterChatError::ResponseTooLarge => "response_too_large".to_owned(),
        OpenRouterChatError::MalformedResponse => "malformed_response".to_owned(),
        OpenRouterChatError::IncompleteStream => "incomplete_stream".to_owned(),
        OpenRouterChatError::Stream(_) => "stream_consumer_or_provider".to_owned(),
    }
}

fn wait_before_retry(delay: Duration) {
    #[cfg(not(test))]
    std::thread::sleep(delay);
    #[cfg(test)]
    let _ = delay;
}

fn record_round(result: &mut ChatToolLoopResult, round: &ChatRoundResult) {
    if let Some(segment) = round.billing_segment.clone() {
        result.billing_segments.push(segment);
    }
}

fn record_failed_round(result: &mut ChatToolLoopResult, round: &ChatRoundResult) {
    result.text.push_str(&round.text);
    record_round(result, round);
}

fn parse_arguments(raw: &str) -> Value {
    serde_json::from_str(raw)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}))
}

fn prompt_tool_call(call: &StreamToolCall) -> PromptToolCall {
    PromptToolCall {
        id: call.id.clone(),
        call_type: call.call_type.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use bot_core::ai_prompt::{PromptContent, PromptRole};

    use super::*;

    #[test]
    fn search_trace_preserves_evidence_but_bounds_text_and_omits_extra_fields() {
        let call = call("web_search", "{}");
        let result = ToolExecutionResult::output(
            json!({"results": [{
                "title": "Fixture 2026", "description": "á".repeat(1400),
                "url": "https://user:password@example.com/Fixture?token=secret#private",
                "extra": "not logged"
            }]})
            .to_string(),
        );
        let trace = tool_trace(
            &call,
            &json!({"query": "boca river 2026\npróximo", "api_key": "secret-key"}),
            Some(&result),
        );
        assert_eq!(trace["query"], "boca river 2026\npróximo");
        assert_eq!(trace["sources"][0]["url"], "https://example.com/Fixture");
        assert_eq!(trace["result_count"], 1);
        assert_eq!(
            trace["sources"][0]["description"]
                .as_str()
                .map(|s| s.chars().count()),
            Some(1201)
        );
        let encoded = trace.to_string();
        assert!(!encoded.contains('\n'));
        for excluded in ["password", "secret", "not logged"] {
            assert!(!encoded.contains(excluded));
        }
    }

    #[test]
    fn search_trace_distinguishes_empty_results_from_errors_and_links_provider_request() {
        let call = call("web_search", "{}");
        let mut result = ToolExecutionResult::output(r#"{"results":[]}"#);
        result.billing_segment = Some(json!({"metadata": {"provider_request_id": "request-123"}}));
        let empty = tool_trace(&call, &json!({}), Some(&result));
        assert_eq!(empty["result_count"], 0);
        assert_eq!(empty["provider_request_id"], "request-123");
        assert!(empty.get("search_error").is_none());
        result.output = "Search error: timed out".to_owned();
        let error = tool_trace(&call, &json!({}), Some(&result));
        assert_eq!(error["search_error"], "Search error: timed out");
        assert!(error.get("result_count").is_none());
    }

    #[test]
    fn traces_do_not_dump_private_tool_contents_or_assistant_text() {
        let call = call("task_set", "{}");
        let result = ToolExecutionResult::output("private reminder");
        let trace = tool_trace(&call, &json!({"text": "private reminder"}), Some(&result));
        assert!(!trace.to_string().contains("private reminder"));
        assert_eq!(trace["output_chars"], 16);
        let trace = round_trace(&round(
            "private reply",
            vec![],
            json!({"model": "test-model", "metadata": {"provider_generation_id": "generation-1"}}),
        ));
        assert_eq!(trace["tool_calls"], json!([]));
        assert_eq!(trace["generation_id"], "generation-1");
        assert!(!trace.to_string().contains("private reply"));
        assert_eq!(source_location("not a url"), "[invalid URL]");
        assert_eq!(
            source_location("data:text/plain,private"),
            "[unsupported URL]"
        );
    }

    struct Provider {
        rounds: RefCell<Vec<Result<ChatRoundResult, ChatRoundError>>>,
        observed: RefCell<Vec<Vec<PromptMessage>>>,
    }

    impl ChatRoundStream for Provider {
        fn stream_round(
            &self,
            messages: &[PromptMessage],
            _tools: &[Value],
            on_text: &mut dyn FnMut(&str) -> Result<(), OpenRouterChatError>,
        ) -> Result<ChatRoundResult, ChatRoundError> {
            self.observed.borrow_mut().push(messages.to_vec());
            let round = self.rounds.borrow_mut().remove(0)?;
            if !round.text.is_empty() {
                on_text(&round.text).map_err(|source| ChatRoundError {
                    source,
                    partial: Box::new(round.clone()),
                })?;
            }
            Ok(round)
        }
    }

    #[derive(Default)]
    struct Tools {
        calls: Vec<(String, Value, String)>,
        confirm: bool,
    }

    impl NativeToolRuntime for Tools {
        fn schemas(&self, task_mode: bool) -> Vec<Value> {
            vec![json!({"task_mode": task_mode})]
        }

        fn contains(&self, name: &str, _task_mode: bool) -> bool {
            name == "calculate"
        }

        fn execute(
            &mut self,
            name: &str,
            arguments: &Value,
            tool_call_id: &str,
        ) -> ToolExecutionResult {
            self.calls
                .push((name.to_owned(), arguments.clone(), tool_call_id.to_owned()));
            ToolExecutionResult {
                output: "4".to_owned(),
                failure_fallback: self.confirm.then(|| "synthetic confirmation".to_owned()),
                billing_segment: Some(json!({"kind": "tool"})),
                diagnostics: vec!["synthetic tool diagnostic".to_owned()],
            }
        }
    }

    fn round(text: &str, calls: Vec<StreamToolCall>, segment: Value) -> ChatRoundResult {
        ChatRoundResult {
            text: text.to_owned(),
            reasoning: String::new(),
            reasoning_details: Vec::new(),
            tool_calls: calls,
            finish_reason: Some("tool_calls".to_owned()),
            billing_segment: Some(segment),
        }
    }

    fn call(name: &str, arguments: &str) -> StreamToolCall {
        StreamToolCall {
            index: 0,
            id: "call-1".to_owned(),
            call_type: "function".to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        }
    }

    #[test]
    fn executes_known_calls_and_supplies_typed_results_to_the_next_round() {
        let provider = Provider {
            rounds: RefCell::new(vec![
                Ok(round(
                    "checking",
                    vec![call("calculate", r#"{"expression":"2+2"}"#)],
                    json!({"round": 1}),
                )),
                Ok(round("answer", Vec::new(), json!({"round": 2}))),
            ]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools::default();
        let mut streamed = String::new();
        let result = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[PromptMessage::text(PromptRole::User, "question")],
            false,
            DEFAULT_MAX_TOOL_ROUNDS,
            |text| {
                streamed.push_str(text);
                Ok(())
            },
        )
        .unwrap_or_else(|error| *error.partial);

        assert_eq!(result.text, "checkinganswer");
        assert_eq!(streamed, result.text);
        assert_eq!(result.provider_rounds, 2);
        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.billing_segments.len(), 3);
        assert_eq!(tools.calls[0].1["expression"], "2+2");
        let observed = provider.observed.borrow();
        assert_eq!(observed[1][1].role, PromptRole::Assistant);
        assert_eq!(observed[1][1].tool_calls[0].name, "calculate");
        assert_eq!(observed[1][2].role, PromptRole::Tool);
        assert_eq!(observed[1][2].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(observed[1][2].content, PromptContent::Text("4".to_owned()));
    }

    #[test]
    fn event_loop_resets_provisional_text_when_tool_calls_arrive() {
        let provider = Provider {
            rounds: RefCell::new(vec![
                Ok(round(
                    "checking",
                    vec![call("calculate", r#"{"expression":"2+2"}"#)],
                    json!({"round": 1}),
                )),
                Ok(round("answer", Vec::new(), json!({"round": 2}))),
            ]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools::default();
        let mut events = Vec::new();
        let result = run_chat_tool_loop_events(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            DEFAULT_MAX_TOOL_ROUNDS,
            |event| {
                events.push(event);
                Ok(())
            },
        )
        .unwrap_or_else(|error| *error.partial);

        assert_eq!(
            events,
            [
                ChatToolLoopEvent::FinalText("checking".to_owned()),
                ChatToolLoopEvent::ResetToTrace,
                ChatToolLoopEvent::ToolCall {
                    id: "call-1".to_owned(),
                    name: "calculate".to_owned(),
                    arguments: r#"{"expression":"2+2"}"#.to_owned(),
                },
                ChatToolLoopEvent::ToolResult {
                    id: "call-1".to_owned(),
                    name: "calculate".to_owned(),
                    output: "4".to_owned(),
                },
                ChatToolLoopEvent::FinalText("answer".to_owned()),
            ]
        );
        assert_eq!(result.text, "answer");
    }

    #[test]
    fn event_loop_emits_text_before_provider_round_returns() {
        struct StreamingProvider {
            order: Rc<RefCell<Vec<&'static str>>>,
        }

        impl ChatRoundStream for StreamingProvider {
            fn stream_round(
                &self,
                _messages: &[PromptMessage],
                _tools: &[Value],
                on_text: &mut dyn FnMut(&str) -> Result<(), OpenRouterChatError>,
            ) -> Result<ChatRoundResult, ChatRoundError> {
                let round = round("streamed", Vec::new(), json!({"round": 1}));
                on_text(&round.text).map_err(|source| ChatRoundError {
                    source,
                    partial: Box::new(round.clone()),
                })?;
                self.order.borrow_mut().push("provider_returned");
                Ok(round)
            }
        }

        let order = Rc::new(RefCell::new(Vec::new()));
        let provider = StreamingProvider {
            order: Rc::clone(&order),
        };
        let mut tools = Tools::default();
        let result = run_chat_tool_loop_events(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            1,
            |event| {
                if matches!(event, ChatToolLoopEvent::FinalText(_)) {
                    order.borrow_mut().push("final_text");
                }
                Ok(())
            },
        )
        .unwrap_or_else(|error| *error.partial);

        assert_eq!(&*order.borrow(), &["final_text", "provider_returned"]);
        assert_eq!(result.text, "streamed");
    }

    #[test]
    fn skips_unknown_calls_and_normalizes_malformed_known_arguments() {
        let provider = Provider {
            rounds: RefCell::new(vec![Ok(round(
                "",
                vec![call("missing", "not-json")],
                json!({"round": 1}),
            ))]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools::default();
        let result = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            5,
            |_text| Ok(()),
        );
        assert!(result.is_ok());
        assert!(tools.calls.is_empty());

        let provider = Provider {
            rounds: RefCell::new(vec![Ok(round(
                "",
                vec![call("calculate", "not-json")],
                json!({"round": 1}),
            ))]),
            observed: RefCell::new(Vec::new()),
        };
        let result = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            1,
            |_text| Ok(()),
        )
        .unwrap_or_else(|error| *error.partial);
        assert_eq!(tools.calls[0].1, json!({}));
        assert!(result.stopped_at_limit);
    }

    #[test]
    fn preserves_partial_round_usage_and_text_when_streaming_fails() {
        let provider = Provider {
            rounds: RefCell::new(vec![Err(ChatRoundError {
                source: OpenRouterChatError::IncompleteStream,
                partial: Box::new(round("partial", Vec::new(), json!({"pending": true}))),
            })]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools::default();
        let error = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            5,
            |_text| Ok(()),
        )
        .err();
        assert!(error.is_some());
        let Some(error) = error else {
            return;
        };
        assert_eq!(error.provider_rounds, 1);
        assert_eq!(error.partial.text, "partial");
        assert_eq!(error.partial.billing_segments[0]["pending"], true);
    }

    #[test]
    fn retries_an_empty_transient_round_without_repeating_completed_tools() {
        let provider = Provider {
            rounds: RefCell::new(vec![
                Ok(round(
                    "",
                    vec![call("calculate", r#"{"expression":"2+2"}"#)],
                    json!({"round": 1}),
                )),
                Err(ChatRoundError {
                    source: OpenRouterChatError::IncompleteStream,
                    partial: Box::new(round("", Vec::new(), json!({"pending": true}))),
                }),
                Ok(round("synthetic answer", Vec::new(), json!({"round": 3}))),
            ]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools {
            confirm: true,
            ..Tools::default()
        };

        let result = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            5,
            |_text| Ok(()),
        )
        .unwrap_or_else(|error| *error.partial);

        assert_eq!(result.text, "synthetic answer");
        assert_eq!(result.provider_rounds, 3);
        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(tools.calls.len(), 1);
        assert_eq!(result.failure_fallbacks, ["synthetic confirmation"]);
        assert_eq!(
            result
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.contains("AI provider retry"))
                .count(),
            1
        );
    }

    #[test]
    fn does_not_retry_after_a_partial_reasoning_round() {
        let mut partial = round("", Vec::new(), json!({"pending": true}));
        partial.reasoning = "visible reasoning".to_owned();
        let provider = Provider {
            rounds: RefCell::new(vec![
                Err(ChatRoundError {
                    source: OpenRouterChatError::IncompleteStream,
                    partial: Box::new(partial),
                }),
                Ok(round("unexpected retry", Vec::new(), json!({"round": 2}))),
            ]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools::default();

        let error = run_chat_tool_loop_events(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            DEFAULT_MAX_TOOL_ROUNDS,
            |_event| Ok(()),
        )
        .err()
        .unwrap_or_else(|| unreachable!());

        assert_eq!(error.provider_rounds, 1);
        assert_eq!(provider.observed.borrow().len(), 1);
    }

    #[test]
    fn exhausted_retries_preserve_the_completed_tool_confirmation() {
        let failed_round = || {
            Err(ChatRoundError {
                source: OpenRouterChatError::Transport("synthetic timeout".to_owned()),
                partial: Box::new(round("", Vec::new(), json!({"pending": true}))),
            })
        };
        let provider = Provider {
            rounds: RefCell::new(vec![
                Ok(round(
                    "",
                    vec![call("calculate", r#"{"expression":"2+2"}"#)],
                    json!({"round": 1}),
                )),
                failed_round(),
                failed_round(),
                failed_round(),
            ]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools {
            confirm: true,
            ..Tools::default()
        };

        let error = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            5,
            |_text| Ok(()),
        )
        .err()
        .unwrap_or_else(|| unreachable!());

        assert_eq!(error.provider_rounds, 4);
        assert_eq!(error.partial.tool_calls_executed, 1);
        assert_eq!(tools.calls.len(), 1);
        assert_eq!(error.partial.failure_fallbacks, ["synthetic confirmation"]);
    }

    #[test]
    fn preserves_every_completed_tool_confirmation() {
        let mut second_call = call("calculate", r#"{"expression":"3+3"}"#);
        second_call.index = 1;
        second_call.id = "call-2".to_owned();
        let failed_round = || {
            Err(ChatRoundError {
                source: OpenRouterChatError::IncompleteStream,
                partial: Box::new(round("", Vec::new(), json!({"pending": true}))),
            })
        };
        let provider = Provider {
            rounds: RefCell::new(vec![
                Ok(round(
                    "",
                    vec![call("calculate", r#"{"expression":"2+2"}"#), second_call],
                    json!({"round": 1}),
                )),
                failed_round(),
                failed_round(),
                failed_round(),
            ]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = Tools {
            confirm: true,
            ..Tools::default()
        };

        let error = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            5,
            |_text| Ok(()),
        )
        .err()
        .unwrap_or_else(|| unreachable!());

        assert_eq!(tools.calls.len(), 2);
        assert_eq!(
            error.partial.failure_fallbacks,
            ["synthetic confirmation", "synthetic confirmation"]
        );
    }

    #[test]
    fn does_not_retry_a_failed_round_with_any_billable_usage_shape() {
        for usage in [
            json!({"cost": "0.001"}),
            json!({"cost_details": {"upstream_inference_cost": "0.001"}}),
        ] {
            let provider = Provider {
                rounds: RefCell::new(vec![Err(ChatRoundError {
                    source: OpenRouterChatError::IncompleteStream,
                    partial: Box::new(round("", Vec::new(), json!({"usage": usage}))),
                })]),
                observed: RefCell::new(Vec::new()),
            };
            let mut tools = Tools::default();

            let error = run_chat_tool_loop(
                "test-operation",
                &provider,
                &mut tools,
                &[],
                false,
                5,
                |_text| Ok(()),
            )
            .err()
            .unwrap_or_else(|| unreachable!());

            assert_eq!(error.provider_rounds, 1);
            assert_eq!(provider.observed.borrow().len(), 1);
            assert_eq!(error.partial.billing_segments.len(), 1);
        }
    }

    #[test]
    fn retry_delay_respects_short_rate_limit_advice_and_rejects_long_waits() {
        let rate_limit = |retry_after_seconds| OpenRouterChatError::RateLimited {
            retry_after_seconds,
            message: "synthetic".to_owned(),
        };

        assert_eq!(
            provider_retry_delay(&rate_limit(Some(3)), 0),
            Some(Duration::from_secs(3))
        );
        assert_eq!(provider_retry_delay(&rate_limit(Some(6)), 0), None);
        assert_eq!(
            provider_retry_delay(&rate_limit(None), 1),
            Some(Duration::from_secs(2))
        );
    }

    #[test]
    fn retry_policy_rejects_permanent_and_output_delivery_failures() {
        assert!(retryable_provider_error(
            &OpenRouterChatError::RateLimited {
                retry_after_seconds: Some(1),
                message: "synthetic".to_owned(),
            }
        ));
        assert!(retryable_provider_error(&OpenRouterChatError::Http {
            status_code: 503,
            message: "synthetic".to_owned(),
        }));
        assert!(!retryable_provider_error(&OpenRouterChatError::Http {
            status_code: 400,
            message: "synthetic".to_owned(),
        }));
        assert!(!retryable_provider_error(&OpenRouterChatError::Stream(
            "synthetic delivery failure".to_owned()
        )));
    }
    /// `fetch` calls are read-only and run concurrently; `record` calls have
    /// side effects and run sequentially.
    struct ConcurrentTools {
        barrier: std::sync::Arc<std::sync::Barrier>,
        sequential: Vec<String>,
    }

    impl NativeToolRuntime for ConcurrentTools {
        fn schemas(&self, _task_mode: bool) -> Vec<Value> {
            Vec::new()
        }

        fn contains(&self, name: &str, _task_mode: bool) -> bool {
            matches!(name, "fetch" | "record" | "explode")
        }

        fn execute(
            &mut self,
            name: &str,
            _arguments: &Value,
            tool_call_id: &str,
        ) -> ToolExecutionResult {
            self.sequential.push(tool_call_id.to_owned());
            ToolExecutionResult::output(format!("{name} {tool_call_id}"))
        }

        fn concurrent_call(
            &mut self,
            name: &str,
            arguments: &Value,
            _tool_call_id: &str,
        ) -> Option<ConcurrentToolCall> {
            let barrier = std::sync::Arc::clone(&self.barrier);
            let url = arguments["url"].as_str().unwrap_or_default().to_owned();
            match name {
                // Every concurrent call waits for the others: a sequential
                // loop would deadlock here.
                "fetch" => Some(Box::new(move || {
                    barrier.wait();
                    ToolExecutionResult {
                        output: format!("page {url}"),
                        failure_fallback: None,
                        billing_segment: Some(json!({"url": url})),
                        diagnostics: Vec::new(),
                    }
                })),
                "explode" => Some(Box::new(move || {
                    barrier.wait();
                    std::panic::resume_unwind(Box::new("synthetic tool panic"))
                })),
                _ => None,
            }
        }
    }

    fn numbered_call(index: i64, name: &str, url: &str) -> StreamToolCall {
        StreamToolCall {
            index,
            id: format!("call-{index}"),
            call_type: "function".to_owned(),
            name: name.to_owned(),
            arguments: json!({"url": url}).to_string(),
        }
    }

    #[test]
    fn runs_read_only_calls_concurrently_and_keeps_results_in_call_order() {
        let provider = Provider {
            rounds: RefCell::new(vec![
                Ok(round(
                    "",
                    vec![
                        numbered_call(0, "fetch", "https://a.example"),
                        numbered_call(1, "record", ""),
                        numbered_call(2, "fetch", "https://b.example"),
                        numbered_call(3, "explode", ""),
                        numbered_call(4, "fetch", "https://c.example"),
                    ],
                    json!({"round": 1}),
                )),
                Ok(round("done", Vec::new(), json!({"round": 2}))),
            ]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = ConcurrentTools {
            barrier: std::sync::Arc::new(std::sync::Barrier::new(4)),
            sequential: Vec::new(),
        };
        let mut results = Vec::new();
        let result = run_chat_tool_loop_events(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            DEFAULT_MAX_TOOL_ROUNDS,
            |event| {
                if let ChatToolLoopEvent::ToolResult { id, output, .. } = event {
                    results.push((id, output));
                }
                Ok(())
            },
        )
        .unwrap_or_else(|error| *error.partial);

        assert_eq!(result.text, "done");
        assert_eq!(result.tool_calls_executed, 5);
        assert_eq!(tools.sequential, ["call-1"]);
        assert_eq!(
            results,
            [
                ("call-0".to_owned(), "page https://a.example".to_owned()),
                ("call-1".to_owned(), "record call-1".to_owned()),
                ("call-2".to_owned(), "page https://b.example".to_owned()),
                ("call-3".to_owned(), "explode failed".to_owned()),
                ("call-4".to_owned(), "page https://c.example".to_owned()),
            ]
        );
        assert!(
            result
                .diagnostics
                .contains(&"concurrent tool explode panicked".to_owned())
        );
        assert_eq!(result.billing_segments[1]["url"], "https://a.example");
        assert_eq!(result.billing_segments[3]["url"], "https://c.example");
        let observed = provider.observed.borrow();
        let tool_ids = observed[1]
            .iter()
            .filter_map(|message| message.tool_call_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(tool_ids, ["call-0", "call-1", "call-2", "call-3", "call-4"]);
    }

    #[test]
    fn a_single_read_only_call_runs_inline_and_event_failures_stop_the_round() {
        let provider = Provider {
            rounds: RefCell::new(vec![Ok(round(
                "",
                vec![
                    numbered_call(0, "fetch", "https://a.example"),
                    numbered_call(1, "record", ""),
                ],
                json!({"round": 1}),
            ))]),
            observed: RefCell::new(Vec::new()),
        };
        // A one-party barrier never blocks, yet the lone fetch still goes
        // through `execute` because concurrency needs two read-only calls.
        let mut tools = ConcurrentTools {
            barrier: std::sync::Arc::new(std::sync::Barrier::new(1)),
            sequential: Vec::new(),
        };
        let result = run_chat_tool_loop(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            1,
            |_| Ok(()),
        )
        .unwrap_or_else(|error| *error.partial);
        assert_eq!(tools.sequential, ["call-0", "call-1"]);
        assert!(result.stopped_at_limit);

        let provider = Provider {
            rounds: RefCell::new(vec![Ok(round(
                "",
                vec![
                    numbered_call(0, "fetch", "https://a.example"),
                    numbered_call(1, "fetch", "https://b.example"),
                ],
                json!({"round": 1}),
            ))]),
            observed: RefCell::new(Vec::new()),
        };
        let mut tools = ConcurrentTools {
            barrier: std::sync::Arc::new(std::sync::Barrier::new(2)),
            sequential: Vec::new(),
        };
        let error = run_chat_tool_loop_events(
            "test-operation",
            &provider,
            &mut tools,
            &[],
            false,
            1,
            |event| match event {
                ChatToolLoopEvent::ToolResult { .. } => Err(OpenRouterChatError::Stream(
                    "synthetic delivery failure".to_owned(),
                )),
                _ => Ok(()),
            },
        )
        .err()
        .unwrap_or_else(|| unreachable!());
        // The running fetches are joined before the error is returned.
        assert_eq!(error.partial.tool_calls_executed, 1);
        assert!(tools.sequential.is_empty());
    }
}
