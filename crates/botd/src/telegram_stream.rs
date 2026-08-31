//! Incremental Telegram delivery for native AI responses.

use std::time::Instant;

use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_input::{ChatId, MessageId};
use bot_core::telegram_streaming::{StreamAction, plan_feed, plan_finalize};

use crate::dispatcher::{ActionReceipt, ActionSink};

const DEFAULT_MIN_EDIT_INTERVAL_SECONDS: f64 = 0.3;
const DEFAULT_MIN_CHARS_BETWEEN_EDITS: usize = 15;

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
                let receipt = self.actions.execute(self.send_action(&self.buffer))?;
                self.accept_send(receipt, now_seconds);
            }
            StreamAction::Edit => self.try_edit(now_seconds),
        }
        Ok(())
    }

    fn send_action(&self, text: &str) -> TelegramAction {
        let mut message = SendMessage::new(self.chat_id, text);
        message.reply_to_message_id = Some(self.reply_to_message_id);
        TelegramAction::SendMessage(message)
    }

    fn accept_send(&mut self, receipt: ActionReceipt, now_seconds: f64) {
        self.message_id = receipt.message_id;
        self.sent_text.clone_from(&self.buffer);
        self.last_edit_seconds = now_seconds;
    }

    fn edit_action(&self, message_id: MessageId, text: &str) -> TelegramAction {
        TelegramAction::EditMessage {
            chat_id: self.chat_id,
            message_id,
            text: text.to_owned(),
            reply_markup: None,
        }
    }

    fn try_edit(&mut self, now_seconds: f64) {
        let Some(message_id) = self.message_id else {
            return;
        };
        match self
            .actions
            .try_edit(self.edit_action(message_id, &self.buffer))
        {
            Ok(true) => {
                self.sent_text.clone_from(&self.buffer);
                self.last_edit_seconds = now_seconds;
            }
            Ok(false) | Err(_) => self.ignored_edit_failures += 1,
        }
    }

    pub fn finalize(
        &mut self,
        final_text: &str,
    ) -> Result<StreamDelivery, StreamFinalizeError<Actions::Error>> {
        let plan = plan_finalize(
            &self.buffer,
            &self.sent_text,
            self.message_id.is_some(),
            Some(final_text),
        );
        match plan.action {
            StreamAction::None => {}
            StreamAction::Send => {
                self.send_attempted = true;
                let receipt = self
                    .actions
                    .execute(self.send_action(&plan.text))
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
                    .try_edit(self.edit_action(message_id, &plan.text))
                {
                    Ok(true) => self.sent_text = plan.text,
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
                if message.text == "hello" && message.reply_to_message_id == Some(MessageId(4))
        ));
        assert!(matches!(
            &actions.actions[1],
            TelegramAction::EditMessage { text, .. } if text == "hello there friend"
        ));
        assert!(matches!(
            &actions.actions[2],
            TelegramAction::EditMessage { text, .. } if text == "cleaned response"
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
}
