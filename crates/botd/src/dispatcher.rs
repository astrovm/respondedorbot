//! Native update dispatch for feature-complete command vertical slices.

use bot_adapters::telegram_polling::{IncomingEvent, IncomingMessage, IncomingUpdate};
use bot_core::chat_config::ChatConfig;
use bot_core::command_parsing::parse_command;
use bot_core::command_state::{
    IncomingCommandState, IncomingCommandWritePlan, OutgoingCommandState, OutgoingCommandWritePlan,
    prepare_incoming_command_state, prepare_outgoing_command_state,
};
use bot_core::locale::resolve_locale;
use bot_core::stateless_commands::{
    StatelessCommandPlan, StatelessRuntimeContext, plan_runtime_stateless_command,
    plan_stateless_command,
};
use bot_core::telegram_actions::TelegramAction;
use bot_core::telegram_input::{MessageId, is_group_chat_type};
use thiserror::Error;

use crate::runtime::UpdateHandler;

pub trait ChatConfigSource {
    type Error;

    fn get(&mut self, chat_id: &str) -> Result<ChatConfig, Self::Error>;
}

pub trait ActionSink {
    type Error;

    fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionReceipt {
    pub message_id: Option<MessageId>,
}

pub trait MessageStateSink {
    type Error: std::fmt::Display;

    fn record_incoming(&mut self, plan: &IncomingCommandWritePlan) -> Result<(), Self::Error>;

    fn record_outgoing(&mut self, plan: &OutgoingCommandWritePlan) -> Result<(), Self::Error>;
}

pub trait RuntimeValues {
    fn unix_timestamp(&mut self) -> i64;

    fn instance_name(&self) -> Option<&str>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchOutcome {
    Handled,
    LegacyRequired,
    Unsupported,
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum DispatchError<ConfigError, ActionError> {
    #[error("could not load chat configuration")]
    Config(ConfigError),
    #[error("could not execute Telegram action")]
    Action(ActionError),
}

pub struct NativeDispatcher<Config, Actions, State, Values> {
    config: Config,
    actions: Actions,
    state: State,
    runtime_values: Values,
    bot_name: String,
    last_outcome: Option<DispatchOutcome>,
    state_diagnostics: Vec<String>,
}

impl<Config, Actions, State, Values> NativeDispatcher<Config, Actions, State, Values>
where
    Config: ChatConfigSource,
    Actions: ActionSink,
    State: MessageStateSink,
    Values: RuntimeValues,
{
    #[must_use]
    pub fn new(
        config: Config,
        actions: Actions,
        state: State,
        runtime_values: Values,
        bot_name: &str,
    ) -> Self {
        Self {
            config,
            actions,
            state,
            runtime_values,
            bot_name: bot_name.to_owned(),
            last_outcome: None,
            state_diagnostics: Vec::new(),
        }
    }

    #[must_use]
    pub const fn last_outcome(&self) -> Option<DispatchOutcome> {
        self.last_outcome
    }

    #[must_use]
    pub fn state_diagnostics(&self) -> &[String] {
        &self.state_diagnostics
    }

    fn dispatch_message(
        &mut self,
        message: &IncomingMessage,
    ) -> Result<DispatchOutcome, DispatchError<Config::Error, Actions::Error>> {
        if message.has_reply {
            return Ok(DispatchOutcome::LegacyRequired);
        }
        let (Some(chat_id), Some(message_id), Some(_sender_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let config = self
            .config
            .get(&chat_id.0.to_string())
            .map_err(DispatchError::Config)?;
        let locale = resolve_locale(
            Some(&config.language),
            message.sender_language_code.as_deref(),
            message.chat_type.as_deref().unwrap_or_default(),
        );
        let timestamp = self.runtime_values.unix_timestamp();
        let plan = match plan_stateless_command(
            chat_id,
            message_id,
            &content.text,
            &self.bot_name,
            locale,
        ) {
            StatelessCommandPlan::NotHandled => plan_runtime_stateless_command(
                chat_id,
                message_id,
                &content.text,
                &self.bot_name,
                locale,
                StatelessRuntimeContext {
                    unix_timestamp: timestamp,
                    instance_name: self.runtime_values.instance_name(),
                },
            ),
            plan => plan,
        };
        match plan {
            StatelessCommandPlan::Action(action) => {
                self.state_diagnostics.clear();
                let command = parse_command(&content.text, &self.bot_name).command;
                let incoming = prepare_incoming_command_state(IncomingCommandState {
                    chat_id,
                    message_id,
                    user_id: _sender_id,
                    first_name: message.sender_first_name.as_deref(),
                    username: message.sender_username.as_deref(),
                    text: &content.text,
                    is_group: is_group_chat_type(message.chat_type.as_deref()),
                    timestamp,
                });
                match incoming {
                    Ok(incoming) => {
                        if let Err(error) = self.state.record_incoming(&incoming) {
                            self.state_diagnostics
                                .push(format!("incoming command state: {error}"));
                        }
                    }
                    Err(error) => self
                        .state_diagnostics
                        .push(format!("incoming command state plan: {error}")),
                }
                let response_text = match &action {
                    TelegramAction::SendMessage(message) => Some(message.text.clone()),
                    _ => None,
                };
                let receipt = self
                    .actions
                    .execute(action)
                    .map_err(DispatchError::Action)?;
                if let Some(response_text) = response_text {
                    let outgoing = prepare_outgoing_command_state(OutgoingCommandState {
                        chat_id,
                        incoming_message_id: message_id,
                        sent_message_id: receipt.message_id,
                        text: &response_text,
                        command: &command,
                        timestamp,
                    });
                    match outgoing {
                        Ok(outgoing) => {
                            if let Err(error) = self.state.record_outgoing(&outgoing) {
                                self.state_diagnostics
                                    .push(format!("outgoing command state: {error}"));
                            }
                        }
                        Err(error) => self
                            .state_diagnostics
                            .push(format!("outgoing command state plan: {error}")),
                    }
                }
                Ok(DispatchOutcome::Handled)
            }
            StatelessCommandPlan::NotHandled | StatelessCommandPlan::LegacyFallbackRequired => {
                Ok(DispatchOutcome::LegacyRequired)
            }
        }
    }

    pub fn dispatch(
        &mut self,
        update: IncomingUpdate,
    ) -> Result<DispatchOutcome, DispatchError<Config::Error, Actions::Error>> {
        let outcome = match update.event {
            IncomingEvent::Message(message) => self.dispatch_message(&message)?,
            IncomingEvent::CallbackQuery(_)
            | IncomingEvent::PreCheckoutQuery(_)
            | IncomingEvent::Unsupported => DispatchOutcome::Unsupported,
        };
        self.last_outcome = Some(outcome);
        Ok(outcome)
    }
}

impl<Config, Actions, State, Values> UpdateHandler
    for NativeDispatcher<Config, Actions, State, Values>
where
    Config: ChatConfigSource,
    Actions: ActionSink,
    State: MessageStateSink,
    Values: RuntimeValues,
{
    type Error = DispatchError<Config::Error, Actions::Error>;

    fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
        self.dispatch(update).map(|_outcome| ())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use bot_adapters::telegram_polling::{IncomingEvent, IncomingMessage, IncomingUpdate};
    use bot_core::chat_config::ChatConfig;
    use bot_core::command_state::{IncomingCommandWritePlan, OutgoingCommandWritePlan};
    use bot_core::telegram_actions::TelegramAction;
    use bot_core::telegram_input::{ChatId, MessageContent, MessageId, UserId};

    use super::{
        ActionReceipt, ActionSink, ChatConfigSource, DispatchError, DispatchOutcome,
        MessageStateSink, NativeDispatcher, RuntimeValues,
    };

    struct Config {
        value: Result<ChatConfig, &'static str>,
        chat_ids: Vec<String>,
    }

    impl ChatConfigSource for Config {
        type Error = &'static str;

        fn get(&mut self, chat_id: &str) -> Result<ChatConfig, Self::Error> {
            self.chat_ids.push(chat_id.to_owned());
            self.value.clone()
        }
    }

    #[derive(Default)]
    struct Actions(Vec<TelegramAction>);

    impl ActionSink for Actions {
        type Error = Infallible;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            self.0.push(action);
            Ok(ActionReceipt {
                message_id: Some(MessageId(700)),
            })
        }
    }

    #[derive(Default)]
    struct State {
        incoming: Vec<IncomingCommandWritePlan>,
        outgoing: Vec<OutgoingCommandWritePlan>,
    }

    impl MessageStateSink for State {
        type Error = Infallible;

        fn record_incoming(&mut self, plan: &IncomingCommandWritePlan) -> Result<(), Self::Error> {
            self.incoming.push(plan.clone());
            Ok(())
        }

        fn record_outgoing(&mut self, plan: &OutgoingCommandWritePlan) -> Result<(), Self::Error> {
            self.outgoing.push(plan.clone());
            Ok(())
        }
    }

    struct Values {
        unix_timestamp: i64,
        instance_name: Option<String>,
    }

    impl RuntimeValues for Values {
        fn unix_timestamp(&mut self) -> i64 {
            self.unix_timestamp
        }

        fn instance_name(&self) -> Option<&str> {
            self.instance_name.as_deref()
        }
    }

    fn values() -> Values {
        Values {
            unix_timestamp: 1_672_531_200,
            instance_name: Some("synthetic-instance".to_owned()),
        }
    }

    fn update(text: &str, language: Option<&str>) -> IncomingUpdate {
        IncomingUpdate {
            update_id: 99,
            event: IncomingEvent::Message(IncomingMessage {
                message_id: Some(MessageId(7)),
                chat_id: Some(ChatId(-42)),
                chat_type: Some("private".to_owned()),
                sender_id: Some(UserId(88)),
                sender_first_name: Some("Synthetic".to_owned()),
                sender_username: Some("tester".to_owned()),
                sender_language_code: language.map(ToOwned::to_owned),
                has_reply: false,
                content: Some(MessageContent {
                    text: text.to_owned(),
                    photo_file_id: None,
                    audio_file_id: None,
                }),
            }),
        }
    }

    #[test]
    fn dispatches_localized_stateless_action_with_persisted_configuration() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/convertbase 101, 2, 10", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(dispatcher.config.chat_ids, vec!["-42"]);
        assert_eq!(dispatcher.actions.0.len(), 1);
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(dispatcher.state.outgoing[0].message.message_id, "bot_700");
        assert!(dispatcher.state_diagnostics().is_empty());
        let TelegramAction::SendMessage(message) = &dispatcher.actions.0[0] else {
            return;
        };
        assert_eq!(message.text, "101 in base 2 is 5 in base 10");
        assert_eq!(dispatcher.last_outcome(), Some(DispatchOutcome::Handled));
    }

    #[test]
    fn leaves_unknown_incomplete_and_unicode_messages_for_legacy_runtime() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/other", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert_eq!(
            dispatcher.dispatch(update("/convertbase １２, 10, 2", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        let mut replied = update("/time", None);
        if let IncomingEvent::Message(message) = &mut replied.event {
            message.has_reply = true;
        }
        assert_eq!(
            dispatcher.dispatch(replied),
            Ok(DispatchOutcome::LegacyRequired)
        );
        let incomplete = IncomingUpdate {
            update_id: 100,
            event: IncomingEvent::Message(IncomingMessage {
                message_id: None,
                chat_id: None,
                chat_type: None,
                sender_id: None,
                sender_first_name: None,
                sender_username: None,
                sender_language_code: None,
                has_reply: false,
                content: None,
            }),
        };
        assert_eq!(
            dispatcher.dispatch(incomplete),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn unsupported_updates_do_not_load_config_or_emit_actions() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(IncomingUpdate {
                update_id: 101,
                event: IncomingEvent::Unsupported,
            }),
            Ok(DispatchOutcome::Unsupported)
        );
        assert!(dispatcher.config.chat_ids.is_empty());
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn configuration_and_action_errors_are_not_acknowledged() {
        let config = Config {
            value: Err("synthetic config failure"),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            "@mybot",
        );
        assert!(matches!(
            dispatcher.dispatch(update("/convertbase 1,2,10", None)),
            Err(DispatchError::Config("synthetic config failure"))
        ));

        struct FailingActions;
        impl ActionSink for FailingActions {
            type Error = &'static str;
            fn execute(&mut self, _action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
                Err("synthetic action failure")
            }
        }
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher =
            NativeDispatcher::new(config, FailingActions, State::default(), values(), "@mybot");
        assert!(matches!(
            dispatcher.dispatch(update("/convertbase 1,2,10", None)),
            Err(DispatchError::Action("synthetic action failure"))
        ));
    }

    #[test]
    fn dispatches_time_and_instance_from_injected_runtime_values() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/time", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(update("/instance", None)),
            Ok(DispatchOutcome::Handled)
        );
        let texts = dispatcher
            .actions
            .0
            .iter()
            .filter_map(|action| match action {
                TelegramAction::SendMessage(message) => Some(message.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            texts,
            vec!["1672531200", "I am running on synthetic-instance"]
        );
    }

    #[test]
    fn state_failures_are_diagnostic_and_do_not_duplicate_or_block_delivery() {
        struct FailingState;
        impl MessageStateSink for FailingState {
            type Error = &'static str;

            fn record_incoming(
                &mut self,
                _plan: &IncomingCommandWritePlan,
            ) -> Result<(), Self::Error> {
                Err("synthetic incoming failure")
            }

            fn record_outgoing(
                &mut self,
                _plan: &OutgoingCommandWritePlan,
            ) -> Result<(), Self::Error> {
                Err("synthetic outgoing failure")
            }
        }
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher =
            NativeDispatcher::new(config, Actions::default(), FailingState, values(), "@mybot");
        assert_eq!(
            dispatcher.dispatch(update("/time", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(dispatcher.actions.0.len(), 1);
        assert_eq!(
            dispatcher.state_diagnostics(),
            [
                "incoming command state: synthetic incoming failure",
                "outgoing command state: synthetic outgoing failure",
            ]
        );
    }
}
