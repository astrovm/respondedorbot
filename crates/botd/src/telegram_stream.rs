//! Incremental Telegram delivery for native AI responses.

use std::time::Instant;

use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_input::{ChatId, MessageId};
use bot_core::telegram_streaming::{StreamAction, plan_feed, plan_finalize};
use serde_json::Value;

use crate::ai_dispatch::AiStreamEvent;
use crate::dispatcher::{ActionReceipt, ActionSink};

const DEFAULT_MIN_EDIT_INTERVAL_SECONDS: f64 = 0.3;
const DEFAULT_MIN_CHARS_BETWEEN_EDITS: usize = 15;
const MAX_TRACE_HEAD_CHARS: usize = 800;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamDelivery {
    pub message_id: MessageId,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StreamFinalizeError<Error> {
    Action(Error),
    MissingMessageId,
}

/// Owns one Telegram draft while a provider emits text. Intermediate edit
/// failures are non-fatal: the final edit gets another chance to converge.
pub struct TelegramStream<'a, Actions> {
    actions: &'a mut Actions,
    chat_id: ChatId,
    reply_to_message_id: MessageId,
    started: Instant,
    last_edit_seconds: f64,
    min_edit_interval_seconds: f64,
    min_chars_between_edits: usize,
    buffer: String,
    sent_text: String,
    // Queue acceptance is not final delivery: the final edit must supersede
    // any accepted draft and restore link previews.
    draft_edit_pending: bool,
    message_id: Option<MessageId>,
    send_attempted: bool,
    ignored_edit_failures: usize,
}

impl<'a, Actions: ActionSink> TelegramStream<'a, Actions> {
    #[must_use]
    pub fn new(actions: &'a mut Actions, chat_id: ChatId, reply_to_message_id: MessageId) -> Self {
        Self::with_policy(
            actions,
            chat_id,
            reply_to_message_id,
            DEFAULT_MIN_EDIT_INTERVAL_SECONDS,
            DEFAULT_MIN_CHARS_BETWEEN_EDITS,
        )
    }

    #[must_use]
    fn with_policy(
        actions: &'a mut Actions,
        chat_id: ChatId,
        reply_to_message_id: MessageId,
        min_edit_interval_seconds: f64,
        min_chars_between_edits: usize,
    ) -> Self {
        Self {
            actions,
            chat_id,
            reply_to_message_id,
            started: Instant::now(),
            last_edit_seconds: 0.0,
            min_edit_interval_seconds,
            min_chars_between_edits,
            buffer: String::new(),
            sent_text: String::new(),
            draft_edit_pending: false,
            message_id: None,
            send_attempted: false,
            ignored_edit_failures: 0,
        }
    }

    fn elapsed_seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn feed(&mut self, token: &str) -> Result<(), Actions::Error> {
        let now = self.elapsed_seconds();
        self.feed_at(token, now)
    }

    fn feed_at(&mut self, token: &str, now_seconds: f64) -> Result<(), Actions::Error> {
        let plan = plan_feed(
            false,
            self.message_id.is_some(),
            self.send_attempted,
            &self.buffer,
            &self.sent_text,
            token,
            now_seconds,
            self.last_edit_seconds,
            self.min_edit_interval_seconds,
            self.min_chars_between_edits,
        );
        self.buffer = plan.buffer;
        match plan.action {
            StreamAction::None => {}
            StreamAction::Send => {
                self.send_attempted = true;
                let receipt = self.actions.execute(self.send_action(&self.buffer, true))?;
                self.accept_send(receipt, now_seconds);
            }
            StreamAction::Edit => self.try_edit(now_seconds, true),
        }
        Ok(())
    }

    fn send_action(&self, text: &str, disable_web_page_preview: bool) -> TelegramAction {
        let mut message = SendMessage::new(self.chat_id, text);
        message.reply_to_message_id = Some(self.reply_to_message_id);
        message.disable_web_page_preview = disable_web_page_preview;
        TelegramAction::SendMessage(message)
    }

    fn accept_send(&mut self, receipt: ActionReceipt, now_seconds: f64) {
        self.message_id = receipt.message_id;
        self.sent_text.clone_from(&self.buffer);
        self.last_edit_seconds = now_seconds;
    }

    fn edit_action(
        &self,
        message_id: MessageId,
        text: &str,
        disable_web_page_preview: bool,
    ) -> TelegramAction {
        if disable_web_page_preview {
            TelegramAction::EditMessageNoPreview {
                chat_id: self.chat_id,
                message_id,
                text: text.to_owned(),
                reply_markup: None,
            }
        } else {
            TelegramAction::EditMessage {
                chat_id: self.chat_id,
                message_id,
                text: text.to_owned(),
                reply_markup: None,
            }
        }
    }

    fn try_edit(&mut self, now_seconds: f64, disable_web_page_preview: bool) {
        let Some(message_id) = self.message_id else {
            return;
        };
        match self.actions.enqueue_stream_edit(self.edit_action(
            message_id,
            &self.buffer,
            disable_web_page_preview,
        )) {
            Ok(true) => {
                self.sent_text.clone_from(&self.buffer);
                self.draft_edit_pending = true;
                self.last_edit_seconds = now_seconds;
            }
            Ok(false) | Err(_) => self.ignored_edit_failures += 1,
        }
    }

    fn replace_snapshot(
        &mut self,
        text: &str,
        now_seconds: f64,
        force: bool,
        disable_web_page_preview: bool,
    ) -> Result<(), Actions::Error> {
        self.buffer = bot_core::telegram_actions::truncate_text(text);
        if self.buffer.trim().is_empty() {
            return Ok(());
        }
        if self.message_id.is_none() && !self.send_attempted {
            self.send_attempted = true;
            let receipt = self
                .actions
                .execute(self.send_action(&self.buffer, disable_web_page_preview))?;
            self.accept_send(receipt, now_seconds);
        } else if self.message_id.is_some()
            && self.buffer != self.sent_text
            && (force
                || bot_core::telegram_streaming::should_edit(
                    false,
                    true,
                    now_seconds,
                    self.last_edit_seconds,
                    self.buffer.chars().count(),
                    self.sent_text.chars().count(),
                    self.min_edit_interval_seconds,
                    self.min_chars_between_edits,
                ))
        {
            self.try_edit(now_seconds, disable_web_page_preview);
        }
        Ok(())
    }

    fn replace_draft(&mut self, text: &str, force: bool) -> Result<(), Actions::Error> {
        self.replace_snapshot(text, self.elapsed_seconds(), force, true)
    }

    fn start_thinking_animation(&mut self, text: &str) {
        if let Some(message_id) = self.message_id {
            self.actions
                .start_stream_thinking(self.chat_id, message_id, text);
        }
    }

    fn stop_thinking_animation(&mut self) {
        if let Some(message_id) = self.message_id {
            self.actions.stop_stream_thinking(self.chat_id, message_id);
        }
    }

    pub fn finalize(
        &mut self,
        final_text: &str,
    ) -> Result<StreamDelivery, StreamFinalizeError<Actions::Error>> {
        self.stop_thinking_animation();
        let plan = plan_finalize(
            &self.buffer,
            &self.sent_text,
            self.message_id.is_some(),
            Some(final_text),
        );
        let action = if self.draft_edit_pending && plan.action == StreamAction::None {
            StreamAction::Edit
        } else {
            plan.action
        };
        match action {
            StreamAction::None => {}
            StreamAction::Send => {
                self.send_attempted = true;
                let receipt = self
                    .actions
                    .execute(self.send_action(&plan.text, false))
                    .map_err(StreamFinalizeError::Action)?;
                self.message_id = receipt.message_id;
                self.sent_text = plan.text;
            }
            StreamAction::Edit => {
                let Some(message_id) = self.message_id else {
                    unreachable!("stream planner only edits an existing message")
                };
                match self
                    .actions
                    .finalize_stream_edit(self.edit_action(message_id, &plan.text, false))
                {
                    Ok(true) => {
                        self.sent_text = plan.text;
                        self.draft_edit_pending = false;
                    }
                    Ok(false) | Err(_) => self.ignored_edit_failures += 1,
                }
            }
        }
        self.message_id
            .map(|message_id| StreamDelivery { message_id })
            .ok_or(StreamFinalizeError::MissingMessageId)
    }

    pub fn cancel(&mut self) {
        if let Some(message_id) = self.message_id {
            self.stop_thinking_animation();
            self.actions.cancel_stream_edits(self.chat_id, message_id);
            let _result = self.actions.execute(TelegramAction::DeleteMessage {
                chat_id: self.chat_id,
                message_id,
            });
            self.message_id = None;
        }
    }

    #[must_use]
    pub const fn ignored_edit_failures(&self) -> usize {
        self.ignored_edit_failures
    }
}

/// Shows a localized thinking status or tool activity in one replaceable
/// Telegram draft before handing the message over to the final answer stream.
pub struct TelegramAiStream<'a, Actions> {
    stream: TelegramStream<'a, Actions>,
    trace: String,
    thinking_text: String,
    final_text: String,
    final_started: bool,
    final_message_started: bool,
}

impl<'a, Actions: ActionSink> TelegramAiStream<'a, Actions> {
    #[must_use]
    pub fn new(actions: &'a mut Actions, chat_id: ChatId, reply_to_message_id: MessageId) -> Self {
        Self::with_policy(
            actions,
            chat_id,
            reply_to_message_id,
            DEFAULT_MIN_EDIT_INTERVAL_SECONDS,
            DEFAULT_MIN_CHARS_BETWEEN_EDITS,
        )
    }

    #[must_use]
    fn with_policy(
        actions: &'a mut Actions,
        chat_id: ChatId,
        reply_to_message_id: MessageId,
        min_edit_interval_seconds: f64,
        min_chars_between_edits: usize,
    ) -> Self {
        Self {
            stream: TelegramStream::with_policy(
                actions,
                chat_id,
                reply_to_message_id,
                min_edit_interval_seconds,
                min_chars_between_edits,
            ),
            trace: String::new(),
            thinking_text: "☁️ Thinking".to_owned(),
            final_text: String::new(),
            final_started: false,
            final_message_started: false,
        }
    }

    #[must_use]
    pub fn with_thinking_text(mut self, text: impl Into<String>) -> Self {
        self.thinking_text = text.into().trim_end_matches('.').to_owned();
        self
    }

    fn thinking_status(&self) -> String {
        if self.thinking_text.is_empty() {
            String::new()
        } else {
            format!("{}.", self.thinking_text)
        }
    }

    fn render_thinking(&mut self) -> Result<(), Actions::Error> {
        self.trace = self.thinking_status();
        self.stream.replace_draft(&self.trace, true)
    }

    pub fn show_thinking(&mut self) -> Result<(), Actions::Error> {
        self.final_started = false;
        self.final_message_started = false;
        self.final_text.clear();
        let result = self.render_thinking();
        if result.is_ok() {
            self.stream.start_thinking_animation(&self.thinking_text);
        }
        result
    }

    pub fn feed(&mut self, event: AiStreamEvent) -> Result<(), Actions::Error> {
        match event {
            // Keep reasoning internal to the model; Telegram only receives a
            // localized status message while the provider is thinking.
            AiStreamEvent::Thought(_) if !self.final_started => Ok(()),
            AiStreamEvent::ResetToTrace => self.show_thinking(),
            AiStreamEvent::ToolCall {
                name, arguments, ..
            } if !self.final_started => {
                self.stream.stop_thinking_animation();
                self.replace_trace_line(&format!(
                    "🔧 {}({})",
                    name,
                    format_tool_arguments(&arguments)
                ));
                self.stream.replace_draft(&self.trace, true)
            }
            AiStreamEvent::ToolResult { .. } if !self.final_started => self.show_thinking(),
            AiStreamEvent::FinalText(text) => {
                if !self.final_started {
                    self.stream.stop_thinking_animation();
                }
                self.final_started = true;
                self.final_text.push_str(&text);
                let force = !self.final_message_started && !self.final_text.trim().is_empty();
                self.final_message_started |= force;
                self.stream.replace_draft(&self.final_text, force)
            }
            AiStreamEvent::Thought(_)
            | AiStreamEvent::ToolCall { .. }
            | AiStreamEvent::ToolResult { .. } => Ok(()),
        }
    }

    fn replace_trace_line(&mut self, line: &str) {
        self.trace.clear();
        self.trace.push_str(line);
        self.bound_trace();
    }

    fn bound_trace(&mut self) {
        let max_chars = bot_core::telegram_actions::MAX_TELEGRAM_TEXT_LENGTH;
        let chars = self.trace.chars().collect::<Vec<_>>();
        if chars.len() <= max_chars {
            return;
        }
        let marker = ['\n', '…', '\n'];
        let tail_chars = max_chars.saturating_sub(MAX_TRACE_HEAD_CHARS + marker.len());
        let mut bounded = chars[..MAX_TRACE_HEAD_CHARS].to_vec();
        bounded.extend(marker);
        bounded.extend(
            chars[chars.len().saturating_sub(tail_chars)..]
                .iter()
                .copied(),
        );
        self.trace = bounded.into_iter().collect();
    }

    pub fn finalize(
        &mut self,
        final_text: &str,
    ) -> Result<StreamDelivery, StreamFinalizeError<Actions::Error>> {
        self.stream
            .finalize(&bot_core::telegram_actions::truncate_text(final_text))
    }

    pub fn cancel(&mut self) {
        self.stream.cancel();
    }

    #[must_use]
    pub const fn ignored_edit_failures(&self) -> usize {
        self.stream.ignored_edit_failures()
    }
}

fn format_tool_arguments(raw: &str) -> String {
    let Ok(Value::Object(arguments)) = serde_json::from_str::<Value>(raw) else {
        return raw.trim().to_owned();
    };
    arguments
        .into_iter()
        .map(|(name, value)| format!("{name}={}", format_tool_value(&value)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_tool_value(value: &Value) -> String {
    match value {
        Value::String(value) => serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_owned()),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use bot_core::telegram_actions::TelegramAction;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SyntheticError;

    #[derive(Default)]
    struct Actions {
        actions: Vec<TelegramAction>,
        next_message_id: Option<MessageId>,
        edit_fails: bool,
        queue_stream_edits: bool,
        queued_stream_edits: Vec<TelegramAction>,
        finalized_stream_edits: Vec<TelegramAction>,
    }

    impl ActionSink for Actions {
        type Error = SyntheticError;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            self.actions.push(action);
            Ok(ActionReceipt {
                message_id: self.next_message_id,
            })
        }

        fn try_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
            self.actions.push(action);
            if self.edit_fails {
                Err(SyntheticError)
            } else {
                Ok(true)
            }
        }

        fn enqueue_stream_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
            if self.queue_stream_edits {
                self.queued_stream_edits.push(action);
                Ok(true)
            } else {
                self.try_edit(action)
            }
        }

        fn finalize_stream_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
            if self.queue_stream_edits {
                self.finalized_stream_edits.push(action);
                Ok(true)
            } else {
                self.try_edit(action)
            }
        }
    }

    #[test]
    fn sends_first_token_edits_by_policy_and_converges_to_cleaned_text() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            ..Actions::default()
        };
        let mut stream = TelegramStream::with_policy(&mut actions, ChatId(7), MessageId(4), 0.3, 5);
        assert_eq!(stream.feed_at("hello", 0.0), Ok(()));
        assert_eq!(stream.feed_at(" there", 0.2), Ok(()));
        assert_eq!(stream.feed_at(" friend", 0.3), Ok(()));
        assert_eq!(
            stream.finalize("cleaned response"),
            Ok(StreamDelivery {
                message_id: MessageId(80)
            })
        );
        drop(stream);

        assert!(matches!(
            &actions.actions[0],
            TelegramAction::SendMessage(message)
                if message.text == "hello"
                    && message.reply_to_message_id == Some(MessageId(4))
                    && message.disable_web_page_preview
        ));
        assert!(matches!(
            &actions.actions[1],
            TelegramAction::EditMessageNoPreview { text, .. } if text == "hello there friend"
        ));
        assert!(matches!(
            &actions.actions[2],
            TelegramAction::EditMessage { text, .. } if text == "cleaned response"
        ));
    }

    #[test]
    fn finalization_replaces_a_queued_same_text_draft() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            queue_stream_edits: true,
            ..Actions::default()
        };
        let mut stream = TelegramStream::with_policy(&mut actions, ChatId(7), MessageId(4), 0.0, 1);
        assert_eq!(stream.feed_at("draft", 0.0), Ok(()));
        assert_eq!(stream.feed_at(" answer", 0.1), Ok(()));
        assert_eq!(
            stream.finalize("draft answer"),
            Ok(StreamDelivery {
                message_id: MessageId(80)
            })
        );
        drop(stream);

        assert!(matches!(
            actions.queued_stream_edits.first(),
            Some(TelegramAction::EditMessageNoPreview { text, .. }) if text == "draft answer"
        ));
        assert!(matches!(
            actions.finalized_stream_edits.first(),
            Some(TelegramAction::EditMessage { text, .. }) if text == "draft answer"
        ));
    }

    #[test]
    fn public_feed_and_status_events_handle_empty_and_post_final_inputs() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            ..Actions::default()
        };
        {
            let mut stream = TelegramStream::with_policy(
                &mut actions,
                ChatId(7),
                MessageId(4),
                0.3,
                DEFAULT_MIN_CHARS_BETWEEN_EDITS,
            );
            assert_eq!(stream.feed("hello"), Ok(()));
        }

        let mut no_message_actions = Actions::default();
        let mut no_message_stream = TelegramStream::with_policy(
            &mut no_message_actions,
            ChatId(7),
            MessageId(4),
            0.3,
            DEFAULT_MIN_CHARS_BETWEEN_EDITS,
        );
        no_message_stream.try_edit(0.0, true);

        let mut empty_actions = Actions::default();
        let mut empty_status = TelegramAiStream::with_policy(
            &mut empty_actions,
            ChatId(7),
            MessageId(4),
            0.3,
            DEFAULT_MIN_CHARS_BETWEEN_EDITS,
        )
        .with_thinking_text("");
        assert_eq!(empty_status.show_thinking(), Ok(()));

        let mut final_actions = Actions {
            next_message_id: Some(MessageId(81)),
            ..Actions::default()
        };
        let mut final_stream =
            TelegramAiStream::with_policy(&mut final_actions, ChatId(7), MessageId(4), 0.0, 1);
        assert_eq!(
            final_stream.feed(AiStreamEvent::FinalText("answer".to_owned())),
            Ok(())
        );
        assert_eq!(
            final_stream.feed(AiStreamEvent::ToolResult {
                id: "synthetic".to_owned(),
                name: "calculate".to_owned(),
                output: "2".to_owned(),
            }),
            Ok(())
        );
    }

    #[test]
    fn bounds_long_tool_traces_and_formats_non_string_arguments() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            ..Actions::default()
        };
        let mut stream =
            TelegramAiStream::with_policy(&mut actions, ChatId(7), MessageId(4), 0.0, 1);
        assert_eq!(
            stream.feed(AiStreamEvent::ToolCall {
                id: "long-call".to_owned(),
                name: "synthetic".to_owned(),
                arguments: "x".repeat(bot_core::telegram_actions::MAX_TELEGRAM_TEXT_LENGTH),
            }),
            Ok(())
        );
        assert_eq!(
            stream.feed(AiStreamEvent::ToolCall {
                id: "numeric-call".to_owned(),
                name: "calculate".to_owned(),
                arguments: r#"{"count":2}"#.to_owned(),
            }),
            Ok(())
        );
        drop(stream);
        assert!(matches!(
            actions.actions.first(),
            Some(TelegramAction::SendMessage(message))
                if message.text.chars().count() <= bot_core::telegram_actions::MAX_TELEGRAM_TEXT_LENGTH
        ));
    }

    #[test]
    fn ignores_draft_edit_failures_and_keeps_confirmed_delivery() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            edit_fails: true,
            ..Actions::default()
        };
        let mut stream = TelegramStream::with_policy(&mut actions, ChatId(7), MessageId(4), 0.0, 1);
        assert_eq!(stream.feed_at("a", 0.0), Ok(()));
        assert_eq!(stream.feed_at("b", 0.1), Ok(()));
        assert_eq!(
            stream.finalize("final"),
            Ok(StreamDelivery {
                message_id: MessageId(80)
            })
        );
        assert_eq!(stream.ignored_edit_failures(), 2);
    }

    #[test]
    fn cancellation_removes_a_partially_streamed_spontaneous_reply() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            ..Actions::default()
        };
        let mut stream = TelegramStream::new(&mut actions, ChatId(7), MessageId(4));
        assert_eq!(stream.feed_at("partial", 0.0), Ok(()));
        stream.cancel();
        drop(stream);
        assert!(matches!(
            actions.actions.last(),
            Some(TelegramAction::DeleteMessage {
                chat_id: ChatId(7),
                message_id: MessageId(80),
            })
        ));
    }

    #[test]
    fn ai_stream_hides_reasoning_and_shows_localized_status_and_tools() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            ..Actions::default()
        };
        let mut stream =
            TelegramAiStream::with_policy(&mut actions, ChatId(7), MessageId(4), 0.0, 1)
                .with_thinking_text("☁️ Pensando");
        stream.show_thinking().unwrap_or_else(|_| unreachable!());
        stream
            .feed(AiStreamEvent::Thought("checking the match".to_owned()))
            .unwrap_or_else(|_| unreachable!());
        stream
            .feed(AiStreamEvent::ResetToTrace)
            .unwrap_or_else(|_| unreachable!());
        stream
            .feed(AiStreamEvent::ToolCall {
                id: "call-1".to_owned(),
                name: "web_search".to_owned(),
                arguments: r#"{"query":"cuando juegan river y huracán"}"#.to_owned(),
            })
            .unwrap_or_else(|_| unreachable!());
        stream
            .feed(AiStreamEvent::ToolResult {
                id: "call-1".to_owned(),
                name: "web_search".to_owned(),
                output: "fixture result".to_owned(),
            })
            .unwrap_or_else(|_| unreachable!());
        stream
            .feed(AiStreamEvent::FinalText("River juega ".to_owned()))
            .unwrap_or_else(|_| unreachable!());
        let delivery = stream
            .finalize("River juega el sábado")
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(delivery.message_id, MessageId(80));
        drop(stream);

        assert!(matches!(
            &actions.actions[0],
            TelegramAction::SendMessage(message) if message.text == "☁️ Pensando."
                && message.disable_web_page_preview
        ));
        assert!(matches!(
            &actions.actions[1],
            TelegramAction::EditMessageNoPreview { text, .. }
                if text == "🔧 web_search(query=\"cuando juegan river y huracán\")"
        ));
        assert!(matches!(
            &actions.actions[2],
            TelegramAction::EditMessageNoPreview { text, .. } if text == "☁️ Pensando."
        ));
        assert!(matches!(
            &actions.actions[3],
            TelegramAction::EditMessageNoPreview { text, .. } if text == "River juega "
        ));
        assert!(matches!(
            &actions.actions[4],
            TelegramAction::EditMessage { text, .. } if text == "River juega el sábado"
        ));
    }
    #[test]
    fn provisional_text_is_replaced_by_tools_and_only_the_final_round_remains() {
        for thought in [None, Some("checking")] {
            let mut actions = Actions {
                next_message_id: Some(MessageId(80)),
                ..Actions::default()
            };
            let mut stream =
                TelegramAiStream::with_policy(&mut actions, ChatId(7), MessageId(4), 60.0, 100);
            if let Some(thought) = thought {
                assert_eq!(
                    stream.feed(AiStreamEvent::Thought(thought.to_owned())),
                    Ok(())
                );
            }
            for event in [
                AiStreamEvent::FinalText("provisional".to_owned()),
                AiStreamEvent::ResetToTrace,
                AiStreamEvent::ToolCall {
                    id: "synthetic-call".to_owned(),
                    name: "calculate".to_owned(),
                    arguments: "{}".to_owned(),
                },
                AiStreamEvent::FinalText("answer".to_owned()),
            ] {
                assert_eq!(stream.feed(event), Ok(()));
            }
            assert!(stream.finalize("answer").is_ok());
            drop(stream);
            assert!(actions.actions.iter().any(|action| matches!(action,
                TelegramAction::EditMessageNoPreview { text, .. } if text.starts_with("☁️ Thinking")
            )));
            assert!(actions.actions.iter().any(|action| matches!(action,
                TelegramAction::EditMessageNoPreview { text, .. } if text.contains("calculate")
            )));
            assert!(matches!(
                actions.actions.last(),
                Some(TelegramAction::EditMessage { text, .. }) if text == "answer"
            ));
        }
    }
    #[test]
    fn shorter_activities_replace_previous_actions_despite_edit_throttling() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            ..Actions::default()
        };
        let mut stream =
            TelegramAiStream::with_policy(&mut actions, ChatId(7), MessageId(4), 60.0, 100);
        assert_eq!(stream.show_thinking(), Ok(()));
        for event in [
            AiStreamEvent::Thought("a long initial reasoning block".to_owned()),
            AiStreamEvent::ToolCall {
                id: "first".to_owned(),
                name: "web_search".to_owned(),
                arguments: r#"{"query":"synthetic fixture"}"#.to_owned(),
            },
            AiStreamEvent::ToolCall {
                id: "second".to_owned(),
                name: "calculate".to_owned(),
                arguments: "{}".to_owned(),
            },
            AiStreamEvent::ToolResult {
                id: "second".to_owned(),
                name: "calculate".to_owned(),
                output: "2".to_owned(),
            },
            AiStreamEvent::Thought(String::new()),
            AiStreamEvent::Thought("done".to_owned()),
            AiStreamEvent::FinalText("answer".to_owned()),
        ] {
            assert_eq!(stream.feed(event), Ok(()));
        }
        assert!(stream.finalize("answer").is_ok());
        drop(stream);
        let texts = actions
            .actions
            .iter()
            .filter_map(|action| match action {
                TelegramAction::SendMessage(message) => Some(message.text.as_str()),
                TelegramAction::EditMessage { text, .. } => Some(text.as_str()),
                TelegramAction::EditMessageNoPreview { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            texts,
            [
                "☁️ Thinking.",
                "🔧 web_search(query=\"synthetic fixture\")",
                "🔧 calculate()",
                "☁️ Thinking.",
                "answer",
                "answer",
            ]
        );
    }

    #[test]
    fn reasoning_deltas_are_hidden_while_status_returns_after_tool_results() {
        let mut actions = Actions {
            next_message_id: Some(MessageId(80)),
            ..Actions::default()
        };
        let mut stream =
            TelegramAiStream::with_policy(&mut actions, ChatId(7), MessageId(4), 0.0, 1);
        assert_eq!(stream.show_thinking(), Ok(()));
        for event in [
            AiStreamEvent::Thought("first ".to_owned()),
            AiStreamEvent::Thought("thought".to_owned()),
            AiStreamEvent::ToolCall {
                id: "synthetic".to_owned(),
                name: "calculate".to_owned(),
                arguments: "{}".to_owned(),
            },
            AiStreamEvent::ToolResult {
                id: "synthetic".to_owned(),
                name: "calculate".to_owned(),
                output: "2".to_owned(),
            },
            AiStreamEvent::Thought("next ".to_owned()),
            AiStreamEvent::Thought("thought".to_owned()),
        ] {
            assert_eq!(stream.feed(event), Ok(()));
        }
        drop(stream);
        let texts = actions
            .actions
            .iter()
            .filter_map(|action| match action {
                TelegramAction::SendMessage(message) => Some(message.text.as_str()),
                TelegramAction::EditMessage { text, .. } => Some(text.as_str()),
                TelegramAction::EditMessageNoPreview { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(texts, ["☁️ Thinking.", "🔧 calculate()", "☁️ Thinking."]);
    }
}
