//! Bounded native chat/tool orchestration with durable per-round usage.

use bot_adapters::openrouter_chat::{OpenRouterChatError, OpenRouterStreamTransport};
use bot_core::ai_prompt::{PromptMessage, PromptToolCall};
use bot_core::provider_stream_policy::StreamToolCall;
use serde_json::{Value, json};
use std::time::Duration;
use thiserror::Error;

use crate::chat_provider::{ChatRoundError, ChatRoundResult, OpenRouterChatStreamer};

pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 5;
const PROVIDER_RETRY_DELAYS: [Duration; 2] =
    [Duration::from_millis(250), Duration::from_millis(500)];

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

pub trait NativeToolRuntime {
    fn schemas(&self, task_mode: bool) -> Vec<Value>;

    fn contains(&self, name: &str, task_mode: bool) -> bool;

    fn execute(&mut self, name: &str, arguments: &Value, tool_call_id: &str)
    -> ToolExecutionResult;
}

pub trait ChatRoundStream {
    fn stream_round(
        &self,
        messages: &[PromptMessage],
        tools: &[Value],
        on_text: &mut dyn FnMut(&str) -> Result<(), OpenRouterChatError>,
    ) -> Result<ChatRoundResult, ChatRoundError>;
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatToolLoopResult {
    pub text: String,
    pub messages: Vec<PromptMessage>,
    pub billing_segments: Vec<Value>,
    pub provider_rounds: usize,
    pub tool_calls_executed: usize,
    pub diagnostics: Vec<String>,
    pub failure_fallback: Option<String>,
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
            failure_fallback: None,
            stopped_at_limit: false,
        }
    }
}

#[derive(Debug, Error)]
#[error("native chat/tool loop failed after {provider_rounds} provider rounds: {source}")]
pub struct ChatToolLoopError {
    pub source: OpenRouterChatError,
    pub partial: Box<ChatToolLoopResult>,
    pub provider_rounds: usize,
}

pub fn run_chat_tool_loop<Provider, Tools>(
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
    let schemas = tools.schemas(task_mode);
    let mut result = ChatToolLoopResult::new(initial_messages);

    for logical_round in 0..max_rounds {
        let mut retry = 0;
        let round = loop {
            let round = provider.stream_round(&result.messages, &schemas, &mut on_text);
            match round {
                Ok(round) => break round,
                Err(error) => {
                    result.provider_rounds += 1;
                    record_round(&mut result, &error.partial);
                    let retryable = retry < PROVIDER_RETRY_DELAYS.len()
                        && retryable_provider_error(&error.source)
                        && error.partial.text.is_empty()
                        && error.partial.tool_calls.is_empty();
                    if retryable {
                        result.diagnostics.push(format!(
                            "AI provider retry: round={} attempt={} error_kind={}",
                            logical_round + 1,
                            retry + 1,
                            provider_error_kind(&error.source)
                        ));
                        wait_before_retry(PROVIDER_RETRY_DELAYS[retry]);
                        retry += 1;
                        continue;
                    }
                    return Err(ChatToolLoopError {
                        source: error.source,
                        provider_rounds: result.provider_rounds,
                        partial: Box::new(result),
                    });
                }
            }
        };
        result.provider_rounds += 1;
        record_round(&mut result, &round);

        let known_calls = round
            .tool_calls
            .iter()
            .filter(|call| tools.contains(&call.name, task_mode))
            .cloned()
            .collect::<Vec<_>>();
        if known_calls.is_empty() {
            return Ok(result);
        }

        result.messages.push(PromptMessage::assistant_tool_calls(
            (!round.text.is_empty()).then_some(round.text.as_str()),
            known_calls.iter().map(prompt_tool_call).collect(),
        ));
        for call in known_calls {
            let arguments = parse_arguments(&call.arguments);
            let tool_result = tools.execute(&call.name, &arguments, &call.id);
            result.tool_calls_executed += 1;
            if let Some(segment) = tool_result.billing_segment {
                result.billing_segments.push(segment);
            }
            result.diagnostics.extend(tool_result.diagnostics);
            if let Some(fallback) = tool_result.failure_fallback {
                result.failure_fallback = Some(fallback);
            }
            result
                .messages
                .push(PromptMessage::tool_result(&call.id, tool_result.output));
        }
    }

    result.stopped_at_limit = true;
    Ok(result)
}

fn retryable_provider_error(error: &OpenRouterChatError) -> bool {
    matches!(
        error,
        OpenRouterChatError::Transport(_)
            | OpenRouterChatError::IncompleteStream
            | OpenRouterChatError::RateLimited { .. }
            | OpenRouterChatError::Http {
                status_code: 408 | 425 | 500..=599,
                ..
            }
    )
}

pub(crate) fn provider_error_kind(error: &OpenRouterChatError) -> String {
    match error {
        OpenRouterChatError::MissingApiKey => "missing_api_key".to_owned(),
        OpenRouterChatError::MissingModel => "missing_model".to_owned(),
        OpenRouterChatError::InvalidBaseUrl => "invalid_base_url".to_owned(),
        OpenRouterChatError::RequestJson(_) => "request_json".to_owned(),
        OpenRouterChatError::Transport(_) => "transport".to_owned(),
        OpenRouterChatError::RateLimited { .. } => "rate_limited".to_owned(),
        OpenRouterChatError::Http { status_code, .. } => format!("http_{status_code}"),
        OpenRouterChatError::InvalidJson(_) => "invalid_json".to_owned(),
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
    result.text.push_str(&round.text);
    if let Some(segment) = round.billing_segment.clone() {
        result.billing_segments.push(segment);
    }
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

    use bot_core::ai_prompt::{PromptContent, PromptRole};

    use super::*;

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
        let result = run_chat_tool_loop(&provider, &mut tools, &[], false, 5, |_text| Ok(()));
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
        let result = run_chat_tool_loop(&provider, &mut tools, &[], false, 1, |_text| Ok(()))
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
        let error = run_chat_tool_loop(&provider, &mut tools, &[], false, 5, |_text| Ok(())).err();
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

        let result = run_chat_tool_loop(&provider, &mut tools, &[], false, 5, |_text| Ok(()))
            .unwrap_or_else(|error| *error.partial);

        assert_eq!(result.text, "synthetic answer");
        assert_eq!(result.provider_rounds, 3);
        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(tools.calls.len(), 1);
        assert_eq!(
            result.failure_fallback.as_deref(),
            Some("synthetic confirmation")
        );
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

        let error = run_chat_tool_loop(&provider, &mut tools, &[], false, 5, |_text| Ok(()))
            .err()
            .unwrap_or_else(|| unreachable!());

        assert_eq!(error.provider_rounds, 4);
        assert_eq!(error.partial.tool_calls_executed, 1);
        assert_eq!(tools.calls.len(), 1);
        assert_eq!(
            error.partial.failure_fallback.as_deref(),
            Some("synthetic confirmation")
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
}
