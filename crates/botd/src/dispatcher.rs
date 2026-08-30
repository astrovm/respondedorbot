//! Native update dispatch for feature-complete command vertical slices.

use bot_adapters::telegram_polling::{IncomingEvent, IncomingMessage, IncomingUpdate};
use bot_core::chat_config::ChatConfig;
use bot_core::command_parsing::parse_command;
use bot_core::command_state::{
    IncomingCommandState, IncomingCommandWritePlan, OutgoingCommandState, OutgoingCommandWritePlan,
    prepare_incoming_command_state, prepare_outgoing_command_state,
};
use bot_core::config_callbacks::{
    ConfigCallbackDiagnostic, ConfigCallbackOutcome, plan_config_callback,
};
use bot_core::config_command::{plan_config_command, render_config};
use bot_core::language_command::{LanguageCommandPlan, plan_language_command};
use bot_core::locale::resolve_locale;
use bot_core::random_selection::{RandomSelection, parse_random_selection};
use bot_core::stateless_commands::{
    StatelessCommandPlan, StatelessRuntimeContext, plan_runtime_stateless_command,
    plan_stateless_command,
};
use bot_core::telegram_actions::{SendMessage, TelegramAction};
use bot_core::telegram_callbacks::{CallbackContextOutcome, CallbackRoute, parse_callback_context};
use bot_core::telegram_input::{ChatId, MessageId, is_group_chat_type};
use bot_core::telegram_payments::{
    StarPaymentRecord, SuccessfulPaymentDecision, TopupCallbackPlan,
    evaluate_default_successful_payment, invoice_payload_locale, payment_record, plan_pre_checkout,
    plan_topup_callback, plan_topup_command, successful_payment_reply,
};
use num_bigint::BigInt;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::runtime::UpdateHandler;

pub trait ChatConfigSource {
    type Error;

    fn get(&mut self, chat_id: &str) -> Result<ChatConfig, Self::Error>;

    fn set(&mut self, chat_id: &str, config: &ChatConfig) -> Result<ChatConfig, Self::Error>;
}

pub trait ActionSink {
    type Error;

    fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error>;

    fn try_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        self.execute(action).map(|_receipt| true)
    }

    fn try_invoice(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        self.execute(action).map(|_receipt| true)
    }
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

pub trait RandomSource {
    type Error;

    fn choice_index(&mut self, upper_exclusive: usize) -> Result<usize, Self::Error>;

    fn inclusive_integer(&mut self, start: &BigInt, end: &BigInt) -> Result<BigInt, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupAuthorizationDecision {
    pub is_admin: bool,
    pub diagnostics: Vec<String>,
}

pub trait GroupAuthorizer {
    fn authorize(&mut self, chat_id: &str, user_id: &str) -> GroupAuthorizationDecision;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StarPaymentReceipt {
    pub inserted: bool,
    pub user_balance: i64,
}

pub trait StarPaymentSink {
    fn record(&mut self, payment: &StarPaymentRecord) -> Result<StarPaymentReceipt, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchOutcome {
    Handled,
    LegacyRequired,
    Unsupported,
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum DispatchError<ConfigError, ActionError, RandomError> {
    #[error("could not load chat configuration")]
    Config(ConfigError),
    #[error("could not execute Telegram action")]
    Action(ActionError),
    #[error("could not obtain a random value")]
    Random(RandomError),
}

type NativeDispatchResult<Config, Actions, Random> = Result<
    DispatchOutcome,
    DispatchError<
        <Config as ChatConfigSource>::Error,
        <Actions as ActionSink>::Error,
        <Random as RandomSource>::Error,
    >,
>;

pub struct NativeDispatcher<Config, Actions, State, Values, Random, Authorization> {
    config: Config,
    actions: Actions,
    state: State,
    runtime_values: Values,
    random: Random,
    authorization: Authorization,
    bot_name: String,
    billing_available: bool,
    payment_sink: Option<Box<dyn StarPaymentSink>>,
    last_outcome: Option<DispatchOutcome>,
    state_diagnostics: Vec<String>,
}

impl<Config, Actions, State, Values, Random, Authorization>
    NativeDispatcher<Config, Actions, State, Values, Random, Authorization>
where
    Config: ChatConfigSource,
    Actions: ActionSink,
    State: MessageStateSink,
    Values: RuntimeValues,
    Random: RandomSource,
    Authorization: GroupAuthorizer,
{
    #[must_use]
    pub fn new(
        config: Config,
        actions: Actions,
        state: State,
        runtime_values: Values,
        random: Random,
        authorization: Authorization,
        bot_name: &str,
    ) -> Self {
        Self {
            config,
            actions,
            state,
            runtime_values,
            random,
            authorization,
            bot_name: bot_name.to_owned(),
            billing_available: true,
            payment_sink: None,
            last_outcome: None,
            state_diagnostics: Vec::new(),
        }
    }

    /// Override billing availability for startup/readiness and deterministic tests.
    #[must_use]
    pub const fn with_billing_available(mut self, available: bool) -> Self {
        self.billing_available = available;
        self
    }

    /// Connect the exact-once PostgreSQL Stars payment writer.
    #[must_use]
    pub fn with_payment_sink(mut self, sink: Box<dyn StarPaymentSink>) -> Self {
        self.payment_sink = Some(sink);
        self
    }

    fn dispatch_successful_payment(
        &mut self,
        message: Map<String, Value>,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let language_code = message
            .get("from")
            .and_then(Value::as_object)
            .and_then(|user| user.get("language_code"))
            .and_then(Value::as_str);
        let chat_type = message
            .get("chat")
            .and_then(Value::as_object)
            .and_then(|chat| chat.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("private");
        let locale = resolve_locale(None, language_code, chat_type);
        let decision = match evaluate_default_successful_payment(
            &Value::Object(message),
            self.billing_available,
        ) {
            Ok(decision) => decision,
            Err(_) => return Ok(DispatchOutcome::LegacyRequired),
        };
        self.state_diagnostics.clear();
        let (chat_id, text) = match &decision {
            SuccessfulPaymentDecision::Ignore => return Ok(DispatchOutcome::Handled),
            SuccessfulPaymentDecision::BillingUnavailable { chat_id } => (
                chat_id,
                match locale {
                    bot_core::locale::Locale::Es => {
                        "el cobro de ia no está andando, avisale al admin".to_owned()
                    }
                    bot_core::locale::Locale::En => {
                        "AI billing is unavailable, please tell the admin".to_owned()
                    }
                },
            ),
            SuccessfulPaymentDecision::InvalidPayment {
                chat_id,
                user_id,
                currency,
                payload,
                total_amount,
                charge_id,
            } => {
                self.state_diagnostics.push(format!(
                    "Invalid successful payment payload chat_id={chat_id} user_id={user_id} currency={currency} payload={payload} total_amount={total_amount} charge_id={charge_id}"
                ));
                (
                    chat_id,
                    match locale {
                        bot_core::locale::Locale::Es => {
                            "me cayó un pago raro y no lo pude validar, avisale al admin".to_owned()
                        }
                        bot_core::locale::Locale::En => {
                            "I received an invalid payment, please tell the admin".to_owned()
                        }
                    },
                )
            }
            SuccessfulPaymentDecision::Record {
                chat_id,
                credits_awarded,
                ..
            } => {
                let Some(payment) = payment_record(&decision) else {
                    return Ok(DispatchOutcome::LegacyRequired);
                };
                let Some(sink) = self.payment_sink.as_mut() else {
                    return Ok(DispatchOutcome::LegacyRequired);
                };
                let text = match sink.record(&payment) {
                    Ok(receipt) => successful_payment_reply(
                        *credits_awarded,
                        receipt.user_balance,
                        receipt.inserted,
                        locale,
                    ),
                    Err(error) => {
                        self.state_diagnostics.push(format!(
                            "successful payment persistence chat_id={chat_id} user_id={} charge_id={}: {error}",
                            payment.user_id, payment.charge_id
                        ));
                        match locale {
                            bot_core::locale::Locale::Es => "me entró la guita pero se trabó la acreditación, avisale al admin".to_owned(),
                            bot_core::locale::Locale::En => "I received the payment but could not add the credits, please tell the admin".to_owned(),
                        }
                    }
                };
                (chat_id, text)
            }
        };
        let Ok(chat_id) = chat_id.parse::<i64>() else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let _receipt = self
            .actions
            .execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(chat_id),
                &text,
            )))
            .map_err(DispatchError::Action)?;
        Ok(DispatchOutcome::Handled)
    }

    #[must_use]
    pub const fn last_outcome(&self) -> Option<DispatchOutcome> {
        self.last_outcome
    }

    #[must_use]
    pub fn state_diagnostics(&self) -> &[String] {
        &self.state_diagnostics
    }

    fn answer_callback_best_effort(&mut self, callback_id: Option<&str>) {
        if let Some(callback_id) = callback_id {
            let _result = self.actions.execute(TelegramAction::AnswerCallback {
                callback_id: callback_id.to_owned(),
                text: None,
                show_alert: false,
            });
        }
    }

    fn dispatch_callback(
        &mut self,
        callback: &Map<String, Value>,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let username = callback
            .get("from")
            .and_then(Value::as_object)
            .and_then(|user| user.get("username"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parsed = parse_callback_context(&Value::Object(callback.clone()));
        let Ok(CallbackContextOutcome::Context { context }) = parsed else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        if context.route == CallbackRoute::Topup {
            let Ok(chat_id) = context.chat_id.parse::<i64>() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let locale = resolve_locale(
                None,
                context.user_language_code.as_deref(),
                &context.chat_type,
            );
            return match plan_topup_callback(
                context.callback_id.as_deref(),
                &context.data,
                ChatId(chat_id),
                &context.chat_type,
                context.user_id,
                self.billing_available,
                locale,
            ) {
                TopupCallbackPlan::Answer(action) => {
                    if let Some(action) = action {
                        let _receipt = self
                            .actions
                            .execute(action)
                            .map_err(DispatchError::Action)?;
                    }
                    Ok(DispatchOutcome::Handled)
                }
                TopupCallbackPlan::Invoice(plan) => {
                    let sent = self
                        .actions
                        .try_invoice(plan.invoice)
                        .map_err(DispatchError::Action)?;
                    let answer = if sent {
                        plan.success_answer
                    } else {
                        plan.failure_answer
                    };
                    if let Some(answer) = answer {
                        let _receipt = self
                            .actions
                            .execute(answer)
                            .map_err(DispatchError::Action)?;
                    }
                    Ok(DispatchOutcome::Handled)
                }
            };
        }
        if context.route != CallbackRoute::Config {
            return Ok(DispatchOutcome::LegacyRequired);
        }
        let Ok(chat_id_value) = context.chat_id.parse::<i64>() else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let chat_id = ChatId(chat_id_value);
        let message_id = MessageId(context.message_id);
        let config = self
            .config
            .get(&context.chat_id)
            .map_err(DispatchError::Config)?;
        let locale = resolve_locale(
            Some(&config.language),
            context.user_language_code.as_deref(),
            &context.chat_type,
        );
        let is_group = is_group_chat_type(Some(&context.chat_type));
        self.state_diagnostics.clear();
        if is_group {
            let authorization = context.user_id.map_or(
                GroupAuthorizationDecision {
                    is_admin: false,
                    diagnostics: Vec::new(),
                },
                |user_id| {
                    self.authorization
                        .authorize(&context.chat_id, &user_id.to_string())
                },
            );
            self.state_diagnostics.extend(authorization.diagnostics);
            if !authorization.is_admin {
                self.state_diagnostics.push(format!(
                    "Unauthorized config attempt chat_id={} chat_type={} user_id={} username={} action=callback:config callback_data={}",
                    context.chat_id,
                    context.chat_type,
                    context.user_id.map_or_else(String::new, |value| value.to_string()),
                    username,
                    context.data,
                ));
                self.answer_callback_best_effort(context.callback_id.as_deref());
                let text = match locale {
                    bot_core::locale::Locale::Es => "este comando es solo para admins del grupo",
                    bot_core::locale::Locale::En => "Only group admins can use this command",
                };
                let mut response = SendMessage::new(chat_id, text);
                response.reply_to_message_id = Some(message_id);
                let _receipt = self
                    .actions
                    .execute(TelegramAction::SendMessage(response))
                    .map_err(DispatchError::Action)?;
                return Ok(DispatchOutcome::Handled);
            }
        }
        let (outcome, config) = plan_config_callback(&context.data, &config);
        let (changed, diagnostic) = match outcome {
            ConfigCallbackOutcome::Render {
                changed,
                diagnostic,
            } => (changed, diagnostic),
            ConfigCallbackOutcome::Guard => {
                self.answer_callback_best_effort(context.callback_id.as_deref());
                return Ok(DispatchOutcome::Handled);
            }
            ConfigCallbackOutcome::NotHandled | ConfigCallbackOutcome::LegacyRequired => {
                return Ok(DispatchOutcome::LegacyRequired);
            }
        };
        if let Some(diagnostic) = diagnostic {
            let value = context
                .data
                .strip_prefix("cfg:")
                .and_then(|payload| payload.split_once(':'))
                .map_or("", |(_, value)| value);
            let name = match diagnostic {
                ConfigCallbackDiagnostic::InvalidTimezone => "timezone",
                ConfigCallbackDiagnostic::InvalidCreditlessLimit => "creditless",
            };
            self.state_diagnostics.push(format!(
                "Invalid {name} callback value chat_id={} value={value}",
                context.chat_id
            ));
        }
        if changed {
            self.config
                .set(&context.chat_id, &config)
                .map_err(DispatchError::Config)?;
        }
        let rendered_locale = resolve_locale(
            Some(&config.language),
            context.user_language_code.as_deref(),
            &context.chat_type,
        );
        let (rendered_text, rendered_markup) = render_config(&config, rendered_locale, is_group);
        let edit = TelegramAction::EditMessage {
            chat_id,
            message_id,
            text: rendered_text.clone(),
            reply_markup: Some(rendered_markup.clone()),
        };
        let edited = match self.actions.try_edit(edit) {
            Ok(edited) => edited,
            Err(error) => {
                self.answer_callback_best_effort(context.callback_id.as_deref());
                return Err(DispatchError::Action(error));
            }
        };
        let fallback = if edited {
            Ok(())
        } else {
            self.state_diagnostics.push(format!(
                "Falling back to new config message chat_id={} message_id={}",
                context.chat_id, context.message_id
            ));
            let mut message = SendMessage::new(chat_id, &rendered_text);
            message.reply_markup = Some(rendered_markup);
            self.actions
                .execute(TelegramAction::SendMessage(message))
                .map(|_receipt| ())
        };
        self.answer_callback_best_effort(context.callback_id.as_deref());
        fallback.map_err(DispatchError::Action)?;
        Ok(DispatchOutcome::Handled)
    }

    fn dispatch_message(
        &mut self,
        message: &IncomingMessage,
    ) -> NativeDispatchResult<Config, Actions, Random> {
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
        let parsed = parse_command(&content.text, &self.bot_name);
        let is_group = is_group_chat_type(message.chat_type.as_deref());
        let is_settings_command = matches!(
            parsed.command.as_str(),
            "/language" | "/idioma" | "/config" | "/configs" | "/settings"
        );
        let mut language_needs_legacy_group = is_group;
        if is_group && is_settings_command {
            let authorization = self
                .authorization
                .authorize(&chat_id.0.to_string(), &_sender_id.0.to_string());
            self.state_diagnostics.clear();
            self.state_diagnostics.extend(authorization.diagnostics);
            if !authorization.is_admin {
                self.state_diagnostics.push(format!(
                    "Unauthorized config attempt chat_id={} chat_type={} user_id={} username={} action=command:{}",
                    chat_id.0,
                    message.chat_type.as_deref().unwrap_or_default(),
                    _sender_id.0,
                    message.sender_username.as_deref().unwrap_or_default(),
                    parsed.command,
                ));
                let text = match locale {
                    bot_core::locale::Locale::Es => "este comando es solo para admins del grupo",
                    bot_core::locale::Locale::En => "Only group admins can use this command",
                };
                let mut response = SendMessage::new(chat_id, text);
                response.reply_to_message_id = Some(message_id);
                let _receipt = self
                    .actions
                    .execute(TelegramAction::SendMessage(response))
                    .map_err(DispatchError::Action)?;
                return Ok(DispatchOutcome::Handled);
            }
            language_needs_legacy_group = false;
        }
        let language_plan = plan_language_command(
            chat_id,
            message_id,
            &content.text,
            &self.bot_name,
            locale,
            &config,
            language_needs_legacy_group,
        );
        let (plan, updated_config) = match language_plan {
            LanguageCommandPlan::LegacyGroupRequired => {
                return Ok(DispatchOutcome::LegacyRequired);
            }
            LanguageCommandPlan::Action {
                action,
                updated_config,
            } => (StatelessCommandPlan::Action(action), updated_config),
            LanguageCommandPlan::NotHandled => (StatelessCommandPlan::NotHandled, None),
        };
        let plan = if plan != StatelessCommandPlan::NotHandled {
            plan
        } else if let Some(action) = plan_config_command(
            chat_id,
            message_id,
            &content.text,
            &self.bot_name,
            locale,
            &config,
            is_group,
        ) {
            StatelessCommandPlan::Action(action)
        } else if let Some(action) = plan_topup_command(
            chat_id,
            message_id,
            &content.text,
            &self.bot_name,
            locale,
            message.chat_type.as_deref().unwrap_or_default(),
            self.billing_available,
        ) {
            StatelessCommandPlan::Action(action)
        } else if parsed.command == "/random" {
            match parse_random_selection(&parsed.message_text) {
                Err(_) => StatelessCommandPlan::LegacyFallbackRequired,
                Ok(RandomSelection::Invalid) => {
                    let text = match locale {
                        bot_core::locale::Locale::Es => {
                            "mandate algo como 'pizza, carne, sushi' o '1-10' boludo, no me hagas laburar al pedo"
                        }
                        bot_core::locale::Locale::En => {
                            "send options like 'pizza, steak, sushi' or a range like '1-10'"
                        }
                    };
                    let mut message = SendMessage::new(chat_id, text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                Ok(RandomSelection::Choices { values }) => {
                    let index = self
                        .random
                        .choice_index(values.len())
                        .map_err(DispatchError::Random)?;
                    let Some(text) = values.get(index) else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    let mut message = SendMessage::new(chat_id, text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                Ok(RandomSelection::InclusiveRange { start, end }) => {
                    let value = self
                        .random
                        .inclusive_integer(&start, &end)
                        .map_err(DispatchError::Random)?;
                    let mut message = SendMessage::new(chat_id, &value.to_string());
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
            }
        } else {
            match plan_stateless_command(chat_id, message_id, &content.text, &self.bot_name, locale)
            {
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
            }
        };
        match plan {
            StatelessCommandPlan::Action(action) => {
                if let Some(updated_config) = updated_config {
                    self.config
                        .set(&chat_id.0.to_string(), &updated_config)
                        .map_err(DispatchError::Config)?;
                }
                self.state_diagnostics.clear();
                let command = parsed.command;
                let incoming = prepare_incoming_command_state(IncomingCommandState {
                    chat_id,
                    message_id,
                    user_id: _sender_id,
                    first_name: message.sender_first_name.as_deref(),
                    username: message.sender_username.as_deref(),
                    text: &content.text,
                    is_group,
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
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let outcome = match update.event {
            IncomingEvent::Message(message) => self.dispatch_message(&message)?,
            IncomingEvent::SuccessfulPayment(message) => {
                self.dispatch_successful_payment(message)?
            }
            IncomingEvent::CallbackQuery(callback) => self.dispatch_callback(&callback)?,
            IncomingEvent::PreCheckoutQuery(query) => {
                let language_code = query
                    .get("from")
                    .and_then(Value::as_object)
                    .and_then(|user| user.get("language_code"))
                    .and_then(Value::as_str);
                let payload_locale = query
                    .get("invoice_payload")
                    .and_then(Value::as_str)
                    .and_then(invoice_payload_locale);
                let locale = resolve_locale(payload_locale, language_code, "private");
                match plan_pre_checkout(&Value::Object(query), self.billing_available, locale) {
                    Ok(Some(action)) => {
                        let _receipt = self
                            .actions
                            .execute(action)
                            .map_err(DispatchError::Action)?;
                        DispatchOutcome::Handled
                    }
                    Ok(None) => DispatchOutcome::Handled,
                    Err(_) => DispatchOutcome::LegacyRequired,
                }
            }
            IncomingEvent::Unsupported => DispatchOutcome::Unsupported,
        };
        self.last_outcome = Some(outcome);
        Ok(outcome)
    }
}

impl<Config, Actions, State, Values, Random, Authorization> UpdateHandler
    for NativeDispatcher<Config, Actions, State, Values, Random, Authorization>
where
    Config: ChatConfigSource,
    Actions: ActionSink,
    State: MessageStateSink,
    Values: RuntimeValues,
    Random: RandomSource,
    Authorization: GroupAuthorizer,
{
    type Error = DispatchError<Config::Error, Actions::Error, Random::Error>;

    fn handle(&mut self, update: IncomingUpdate) -> Result<(), Self::Error> {
        self.dispatch(update).map(|_outcome| ())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::convert::Infallible;
    use std::rc::Rc;

    use bot_adapters::telegram_polling::{IncomingEvent, IncomingMessage, IncomingUpdate};
    use bot_core::chat_config::ChatConfig;
    use bot_core::command_state::{IncomingCommandWritePlan, OutgoingCommandWritePlan};
    use bot_core::telegram_actions::TelegramAction;
    use bot_core::telegram_input::{ChatId, MessageContent, MessageId, UserId};
    use bot_core::telegram_payments::StarPaymentRecord;
    use num_bigint::BigInt;
    use serde_json::{Map, json};

    use super::{
        ActionReceipt, ActionSink, ChatConfigSource, DispatchError, DispatchOutcome,
        GroupAuthorizationDecision, GroupAuthorizer, MessageStateSink, NativeDispatcher,
        RandomSource, RuntimeValues, StarPaymentReceipt, StarPaymentSink,
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

        fn set(&mut self, chat_id: &str, config: &ChatConfig) -> Result<ChatConfig, Self::Error> {
            self.chat_ids.push(format!("set:{chat_id}"));
            self.value = Ok(config.clone());
            Ok(config.clone())
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

    struct Samples {
        choice_index: usize,
        integer: BigInt,
    }

    impl RandomSource for Samples {
        type Error = Infallible;

        fn choice_index(&mut self, _upper_exclusive: usize) -> Result<usize, Self::Error> {
            Ok(self.choice_index)
        }

        fn inclusive_integer(
            &mut self,
            _start: &BigInt,
            _end: &BigInt,
        ) -> Result<BigInt, Self::Error> {
            Ok(self.integer.clone())
        }
    }

    fn random() -> Samples {
        Samples {
            choice_index: 1,
            integer: BigInt::from(2_u8),
        }
    }

    struct Authorization {
        is_admin: bool,
        diagnostics: Vec<String>,
        checks: Vec<(String, String)>,
    }

    impl GroupAuthorizer for Authorization {
        fn authorize(&mut self, chat_id: &str, user_id: &str) -> GroupAuthorizationDecision {
            self.checks.push((chat_id.to_owned(), user_id.to_owned()));
            GroupAuthorizationDecision {
                is_admin: self.is_admin,
                diagnostics: self.diagnostics.clone(),
            }
        }
    }

    fn authorization() -> Authorization {
        Authorization {
            is_admin: true,
            diagnostics: Vec::new(),
            checks: Vec::new(),
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

    fn callback_update(data: &str, chat_type: &str, language: Option<&str>) -> IncomingUpdate {
        let callback = Map::from_iter([
            ("id".to_owned(), json!("callback-1")),
            ("data".to_owned(), json!(data)),
            (
                "from".to_owned(),
                json!({
                    "id": 88,
                    "username": "tester",
                    "language_code": language,
                }),
            ),
            (
                "message".to_owned(),
                json!({
                    "message_id": 7,
                    "chat": {"id": -42, "type": chat_type},
                }),
            ),
        ]);
        IncomingUpdate {
            update_id: 100,
            event: IncomingEvent::CallbackQuery(callback),
        }
    }

    fn pre_checkout_update(
        query_id: Option<&str>,
        pack_id: &str,
        user_id: serde_json::Value,
        language: Option<&str>,
    ) -> IncomingUpdate {
        let mut query = Map::from_iter([
            (
                "from".to_owned(),
                json!({"id":user_id,"language_code":language}),
            ),
            (
                "invoice_payload".to_owned(),
                json!(format!("topup:{pack_id}:42:en")),
            ),
            ("currency".to_owned(), json!("XTR")),
            ("total_amount".to_owned(), json!(25)),
        ]);
        if let Some(query_id) = query_id {
            query.insert("id".to_owned(), json!(query_id));
        }
        IncomingUpdate {
            update_id: 101,
            event: IncomingEvent::PreCheckoutQuery(query),
        }
    }

    fn successful_payment_update(
        pack_id: &str,
        user_id: i64,
        total_amount: i64,
        language: Option<&str>,
    ) -> IncomingUpdate {
        IncomingUpdate {
            update_id: 102,
            event: IncomingEvent::SuccessfulPayment(Map::from_iter([
                ("chat".to_owned(), json!({"id":42,"type":"private"})),
                (
                    "from".to_owned(),
                    json!({"id":user_id,"language_code":language}),
                ),
                (
                    "successful_payment".to_owned(),
                    json!({
                        "currency":"XTR",
                        "invoice_payload":format!("topup:{pack_id}:42:en"),
                        "telegram_payment_charge_id":"charge-1",
                        "total_amount":total_amount,
                    }),
                ),
            ])),
        }
    }

    struct Payments {
        result: Result<StarPaymentReceipt, String>,
        records: Rc<RefCell<Vec<StarPaymentRecord>>>,
    }

    impl StarPaymentSink for Payments {
        fn record(&mut self, payment: &StarPaymentRecord) -> Result<StarPaymentReceipt, String> {
            self.records.borrow_mut().push(payment.clone());
            self.result.clone()
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
            random(),
            authorization(),
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
            random(),
            authorization(),
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
        assert_eq!(
            dispatcher.dispatch(update("/random １-３", None)),
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
            random(),
            authorization(),
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
            random(),
            authorization(),
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
        let mut dispatcher = NativeDispatcher::new(
            config,
            FailingActions,
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
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
            random(),
            authorization(),
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
    fn dispatches_help_with_persisted_locale_and_command_state() {
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
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/help", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert!(message.text.starts_with("what I can do:"));
        assert!(message.text.contains("/summary focus on crypto"));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn dispatches_private_language_reads_and_persisted_updates() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/language", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(update("/idioma en", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher
                .config
                .value
                .as_ref()
                .map(|config| config.language.as_str()),
            Ok("en")
        );
        assert!(dispatcher.config.chat_ids.contains(&"set:-42".to_owned()));
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.last() else {
            return;
        };
        assert_eq!(message.text, "done, I will speak English now");
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .and_then(|markup| markup.inline_keyboard.first())
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn group_language_command_authorizes_admin_and_persists_update() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        let mut group_update = update("/language en", None);
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("supergroup".to_owned());
        }
        assert_eq!(
            dispatcher.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.authorization.checks,
            [("-42".to_owned(), "88".to_owned())]
        );
        assert!(dispatcher.config.chat_ids.contains(&"set:-42".to_owned()));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn group_language_command_denies_non_admin_without_writing_command_state() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let denied = Authorization {
            is_admin: false,
            diagnostics: vec!["synthetic lookup diagnostic".to_owned()],
            checks: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            denied,
            "@mybot",
        );
        let mut group_update = update("/idioma en", None);
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("group".to_owned());
        }
        assert_eq!(
            dispatcher.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "este comando es solo para admins del grupo");
        assert!(dispatcher.state.incoming.is_empty());
        assert!(dispatcher.state.outgoing.is_empty());
        assert_eq!(
            dispatcher.state_diagnostics(),
            [
                "synthetic lookup diagnostic",
                "Unauthorized config attempt chat_id=-42 chat_type=group user_id=88 username=tester action=command:/idioma",
            ]
        );
    }

    #[test]
    fn private_settings_render_native_english_configuration_without_authorization() {
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
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/settings@mybot", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.authorization.checks.is_empty());
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert!(message.text.starts_with("Bot settings"));
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(5)
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn group_config_authorizes_admin_and_renders_group_only_settings() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        let mut group_update = update("/configs", None);
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("group".to_owned());
        }
        assert_eq!(
            dispatcher.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.authorization.checks,
            [("-42".to_owned(), "88".to_owned())]
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert!(message.text.starts_with("config del gordo"));
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(7)
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn group_config_denial_uses_the_shared_admin_boundary() {
        let denied = Authorization {
            is_admin: false,
            diagnostics: Vec::new(),
            checks: Vec::new(),
        };
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
            random(),
            denied,
            "@mybot",
        );
        let mut group_update = update("/config", None);
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("supergroup".to_owned());
        }
        assert_eq!(
            dispatcher.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "Only group admins can use this command");
        assert!(dispatcher.state.incoming.is_empty());
        assert!(dispatcher.state.outgoing.is_empty());
        assert!(dispatcher.state_diagnostics()[0].contains("action=command:/config"));
    }

    #[test]
    fn private_config_callback_updates_persists_edits_and_acknowledges() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("cfg:random:toggle", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher
                .config
                .value
                .as_ref()
                .map(|config| config.ai_random_replies),
            Ok(false)
        );
        assert!(dispatcher.config.chat_ids.contains(&"set:-42".to_owned()));
        assert!(matches!(
            dispatcher.actions.0.first(),
            Some(TelegramAction::EditMessage { reply_markup: Some(markup), .. })
                if markup.inline_keyboard.len() == 5
        ));
        assert!(matches!(
            dispatcher.actions.0.get(1),
            Some(TelegramAction::AnswerCallback { callback_id, .. })
                if callback_id == "callback-1"
        ));
        assert!(dispatcher.state.incoming.is_empty());
        assert!(dispatcher.state.outgoing.is_empty());
    }

    #[test]
    fn language_callback_immediately_renders_the_new_locale() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("cfg:language:en", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.first(),
            Some(TelegramAction::EditMessage { text, .. }) if text.starts_with("Bot settings")
        ));
    }

    #[test]
    fn config_callback_current_and_malformed_buttons_only_acknowledge() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        for data in ["cfg:timezone:current", "cfg:broken"] {
            assert_eq!(
                dispatcher.dispatch(callback_update(data, "private", None)),
                Ok(DispatchOutcome::Handled)
            );
        }
        assert_eq!(dispatcher.actions.0.len(), 2);
        assert!(
            dispatcher
                .actions
                .0
                .iter()
                .all(|action| matches!(action, TelegramAction::AnswerCallback { .. }))
        );
        assert!(
            !dispatcher
                .config
                .chat_ids
                .iter()
                .any(|chat_id| chat_id.starts_with("set:"))
        );
    }

    #[test]
    fn group_config_callback_denial_acknowledges_replies_and_audits() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let denied = Authorization {
            is_admin: false,
            diagnostics: vec!["synthetic callback lookup".to_owned()],
            checks: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            denied,
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("cfg:link:off", "group", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.authorization.checks,
            [("-42".to_owned(), "88".to_owned())]
        );
        assert!(matches!(
            dispatcher.actions.0.first(),
            Some(TelegramAction::AnswerCallback { .. })
        ));
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.get(1) else {
            return;
        };
        assert_eq!(message.text, "este comando es solo para admins del grupo");
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert!(dispatcher.state_diagnostics()[1].contains("callback_data=cfg:link:off"));
    }

    #[test]
    fn config_callback_uses_new_message_fallback_before_acknowledging() {
        #[derive(Default)]
        struct EditFallbackActions(Vec<TelegramAction>);

        impl ActionSink for EditFallbackActions {
            type Error = Infallible;

            fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
                self.0.push(action);
                Ok(ActionReceipt { message_id: None })
            }

            fn try_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
                self.0.push(action);
                Ok(false)
            }
        }

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            EditFallbackActions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("cfg:link:delete", "private", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::EditMessage { .. },
                TelegramAction::SendMessage(_),
                TelegramAction::AnswerCallback { .. }
            ]
        ));
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|message| message.starts_with("Falling back to new config message"))
        );
    }

    #[test]
    fn non_config_callbacks_remain_owned_by_the_legacy_runtime() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("task:delete:1", "private", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(dispatcher.config.chat_ids.is_empty());
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn pre_checkout_dispatches_valid_and_localized_fail_closed_answers() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(pre_checkout_update(
                Some("checkout-valid"),
                "p50",
                json!(42),
                Some("en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(pre_checkout_update(
                Some("checkout-invalid"),
                "missing",
                json!(42),
                Some("es"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.actions.0,
            vec![
                TelegramAction::AnswerPreCheckout {
                    query_id: "checkout-valid".to_owned(),
                    ok: true,
                    error_message: None,
                },
                TelegramAction::AnswerPreCheckout {
                    query_id: "checkout-invalid".to_owned(),
                    ok: false,
                    error_message: Some("I could not validate this payment".to_owned()),
                },
            ]
        );
        assert!(dispatcher.config.chat_ids.is_empty());
        assert!(dispatcher.state.incoming.is_empty());
    }

    #[test]
    fn pre_checkout_respects_billing_readiness_and_ignores_missing_query_ids() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_billing_available(false);
        assert_eq!(
            dispatcher.dispatch(pre_checkout_update(
                Some("checkout-unavailable"),
                "p50",
                json!(42),
                Some("en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(pre_checkout_update(None, "p50", json!(42), Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.actions.0,
            vec![TelegramAction::AnswerPreCheckout {
                query_id: "checkout-unavailable".to_owned(),
                ok: false,
                error_message: Some("AI billing is unavailable, please tell the admin".to_owned()),
            }]
        );
    }

    #[test]
    fn malformed_pre_checkout_sender_remains_legacy_owned_during_shadowing() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        let query = Map::from_iter([
            ("id".to_owned(), json!("checkout-malformed")),
            ("from".to_owned(), json!("invalid sender")),
            ("invoice_payload".to_owned(), json!("topup:p50:42:en")),
            ("currency".to_owned(), json!("XTR")),
            ("total_amount".to_owned(), json!(25)),
        ]);
        assert_eq!(
            dispatcher.dispatch(IncomingUpdate {
                update_id: 101,
                event: IncomingEvent::PreCheckoutQuery(query),
            }),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn topup_command_and_callback_complete_the_native_invoice_flow() {
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
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/topup", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(command)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(command.text, "choose how much you want to add:");
        assert_eq!(
            command
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(6)
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);

        assert_eq!(
            dispatcher.dispatch(callback_update("topup:p50", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(_),
                TelegramAction::SendInvoice { payload, .. },
                TelegramAction::AnswerCallback {
                    text: Some(text),
                    show_alert: false,
                    ..
                }
            ] if payload == "topup:p50:88:en" && text == "invoice ready"
        ));
    }

    #[test]
    fn topup_invoice_failure_answers_with_an_alert_without_retrying_the_charge() {
        #[derive(Default)]
        struct InvoiceFailure(Vec<TelegramAction>);

        impl ActionSink for InvoiceFailure {
            type Error = Infallible;

            fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
                self.0.push(action);
                Ok(ActionReceipt { message_id: None })
            }

            fn try_invoice(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
                self.0.push(action);
                Ok(false)
            }
        }

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            InvoiceFailure::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:p50", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendInvoice { .. },
                TelegramAction::AnswerCallback {
                    text: Some(text),
                    show_alert: true,
                    ..
                }
            ] if text == "no pude armar la factura, probá de nuevo"
        ));
    }

    #[test]
    fn topup_guards_are_native_and_do_not_load_chat_configuration() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:missing", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        dispatcher.billing_available = false;
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:p50", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.config.chat_ids.is_empty());
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::AnswerCallback {
                    text: Some(invalid),
                    show_alert: true,
                    ..
                },
                TelegramAction::AnswerCallback {
                    text: Some(unavailable),
                    show_alert: true,
                    ..
                }
            ] if invalid == "that credit pack is invalid"
                && unavailable == "el cobro de ia no está andando, avisale al admin"
        ));
    }

    #[test]
    fn successful_payment_records_once_and_sends_the_localized_balance() {
        let records = Rc::new(RefCell::new(Vec::new()));
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_payment_sink(Box::new(Payments {
            result: Ok(StarPaymentReceipt {
                inserted: true,
                user_balance: 5_300,
            }),
            records: Rc::clone(&records),
        }));
        assert_eq!(
            dispatcher.dispatch(successful_payment_update("p50", 42, 25, Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            records.borrow().as_slice(),
            [StarPaymentRecord {
                charge_id: "charge-1".to_owned(),
                user_id: 42,
                pack_id: "p50".to_owned(),
                xtr_amount: 25,
                credits_awarded: 5_000,
                payload: "topup:p50:42:en".to_owned(),
            }]
        );
        let [TelegramAction::SendMessage(message)] = dispatcher.actions.0.as_slice() else {
            return;
        };
        assert_eq!(message.chat_id, ChatId(42));
        assert_eq!(
            message.text,
            "listo, te cargué 50.00 créditos\nahora te quedaron 53.00\nsi querés mandarle al grupo: /transfer <monto>"
        );
    }

    #[test]
    fn duplicate_and_failed_payment_writes_have_distinct_safe_replies() {
        for (result, expected, diagnostic) in [
            (
                Ok(StarPaymentReceipt {
                    inserted: false,
                    user_balance: 5_300,
                }),
                "this payment was already credited\nyour balance is 53.00",
                false,
            ),
            (
                Err("synthetic database failure".to_owned()),
                "I received the payment but could not add the credits, please tell the admin",
                true,
            ),
        ] {
            let config = Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            };
            let mut dispatcher = NativeDispatcher::new(
                config,
                Actions::default(),
                State::default(),
                values(),
                random(),
                authorization(),
                "@mybot",
            )
            .with_payment_sink(Box::new(Payments {
                result,
                records: Rc::new(RefCell::new(Vec::new())),
            }));
            assert_eq!(
                dispatcher.dispatch(successful_payment_update("p50", 42, 25, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            let [TelegramAction::SendMessage(message)] = dispatcher.actions.0.as_slice() else {
                return;
            };
            assert_eq!(message.text, expected);
            assert_eq!(!dispatcher.state_diagnostics().is_empty(), diagnostic);
        }
    }

    #[test]
    fn invalid_and_unavailable_successful_payments_never_reach_the_ledger() {
        let records = Rc::new(RefCell::new(Vec::new()));
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_payment_sink(Box::new(Payments {
            result: Ok(StarPaymentReceipt {
                inserted: true,
                user_balance: 5_000,
            }),
            records: Rc::clone(&records),
        }));
        assert_eq!(
            dispatcher.dispatch(successful_payment_update("p50", 99, 25, Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(records.borrow().is_empty());
        assert_eq!(dispatcher.actions.0.len(), 1);
        assert!(dispatcher.state_diagnostics()[0].contains("user_id=99"));

        dispatcher.billing_available = false;
        assert_eq!(
            dispatcher.dispatch(successful_payment_update("p50", 42, 25, Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(records.borrow().is_empty());
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.last() else {
            return;
        };
        assert_eq!(
            message.text,
            "el cobro de ia no está andando, avisale al admin"
        );
    }

    #[test]
    fn valid_successful_payment_waits_for_a_native_ledger_sink_during_shadowing() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(successful_payment_update("p50", 42, 25, Some("en"))),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn dispatches_random_choices_ranges_and_localized_validation() {
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
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/random alpha, beta", None)),
            Ok(DispatchOutcome::Handled)
        );
        dispatcher.random.integer =
            BigInt::from(100_u8) * BigInt::from(10_u8).pow(18) + BigInt::from(2_u8);
        assert_eq!(
            dispatcher.dispatch(update(
                "/random 100000000000000000000-100000000000000000002",
                None,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(update("/random invalid", None)),
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
            [
                "beta",
                "100000000000000000002",
                "send options like 'pizza, steak, sushi' or a range like '1-10'",
            ]
        );
        assert_eq!(dispatcher.state.incoming.len(), 3);
        assert_eq!(dispatcher.state.outgoing.len(), 3);
    }

    #[test]
    fn random_source_errors_are_not_acknowledged() {
        struct FailingRandom;
        impl RandomSource for FailingRandom {
            type Error = &'static str;

            fn choice_index(&mut self, _upper_exclusive: usize) -> Result<usize, Self::Error> {
                Err("synthetic random failure")
            }

            fn inclusive_integer(
                &mut self,
                _start: &BigInt,
                _end: &BigInt,
            ) -> Result<BigInt, Self::Error> {
                Err("synthetic random failure")
            }
        }
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            FailingRandom,
            authorization(),
            "@mybot",
        );
        assert!(matches!(
            dispatcher.dispatch(update("/random alpha, beta", None)),
            Err(DispatchError::Random("synthetic random failure"))
        ));
        assert!(dispatcher.actions.0.is_empty());
        assert!(dispatcher.state.incoming.is_empty());
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
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            FailingState,
            values(),
            random(),
            authorization(),
            "@mybot",
        );
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
