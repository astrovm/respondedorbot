//! Native update dispatch for feature-complete command vertical slices.

use bot_adapters::telegram_polling::{IncomingEvent, IncomingMessage, IncomingUpdate};
use bot_core::admin_commands::{
    CreditLogEntry, CreditLogPlan, PrintCreditsContext, PrintCreditsPlan, plan_creditlog_command,
    plan_printcredits_command, printcredits_result_reply, render_creditlog,
};
use bot_core::bcra::classify_bcra_command;
use bot_core::billing_commands::{
    TransferCommandContext, TransferCommandPlan, TransferResult, plan_transfer_command,
    transfer_result_reply,
};
use bot_core::bitcoin_commands::{
    BitcoinCommand, bitcoin_price_error, classify_bitcoin_command, render_market_model,
    render_satoshi,
};
use bot_core::charge_history::{
    ChargeHistoryCallbackPlan, ChargeHistoryPage, ChargesCommandContext, ChargesCommandPlan,
    charge_callback_answer, plan_charge_history_callback, plan_charges_command,
    render_charge_history_page,
};
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
use bot_core::devo::{
    DevoCommandPlan, DevoQuotes, DevoReply, calculate_devo, plan_devo_command, render_devo_reply,
    render_devo_result,
};
use bot_core::dollar::{
    DollarCommandPlan, classify_dollar_command, invalid_timeframe_message, plan_dollar_command,
};
use bot_core::greeting_commands::{GreetingCategory, classify_greeting_command, greeting_fallback};
use bot_core::language_command::{LanguageCommandPlan, plan_language_command};
use bot_core::links::{
    LinkActionContext, LinkMode, LinkReplacement, has_replaceable_link, plan_link_actions,
};
use bot_core::locale::resolve_locale;
use bot_core::market_prices::{MarketPriceCommand, classify_market_price_command};
use bot_core::polymarket::{ElectionEvent, classify_election_command, render_elections};
use bot_core::random_selection::{RandomSelection, parse_random_selection};
use bot_core::routing::{
    ResponseRoutingEvaluation, ResponseRoutingInput, evaluate_response_routing,
};
use bot_core::rulo::{RuloInput, evaluate_rulo, render_rulo};
use bot_core::scheduled_tasks::{ScheduledTask, TaskId};
use bot_core::stateless_commands::{
    StatelessCommandPlan, StatelessRuntimeContext, plan_runtime_stateless_command,
    plan_stateless_command,
};
use bot_core::stocks::{
    StockQuote, classify_oil_command, classify_stock_command, render_oil_quotes,
    render_stock_quotes,
};
use bot_core::task_commands::{
    TaskCallbackParse, can_delete_task, parse_task_callback, render_task_list,
    task_delete_forbidden, task_deleted, task_not_found,
};
use bot_core::telegram_actions::{ParseMode, SendMessage, TelegramAction};
use bot_core::telegram_callbacks::{
    CallbackContext, CallbackContextOutcome, CallbackRoute, parse_callback_context,
};
use bot_core::telegram_commands::telegram_commands;
use bot_core::telegram_input::{ChatId, MessageId, is_group_chat_type};
use bot_core::telegram_payments::{
    BalanceCommandContext, BalanceCommandPlan, StarPaymentRecord, SuccessfulPaymentDecision,
    TopupCallbackPlan, balance_reply, evaluate_default_successful_payment, invoice_payload_locale,
    payment_record, plan_balance_command, plan_pre_checkout, plan_topup_callback,
    plan_topup_command, successful_payment_reply,
};
use bot_core::weather::{
    WeatherObservation, classify_weather_command, render_weather, requested_location,
    weather_load_error,
};
use num_bigint::BigInt;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::ai_dispatch::{
    AiConversationInput, AiConversationSource, AiDelivery, AiPreparation, reply_context,
};
use crate::runtime::UpdateHandler;
use crate::telegram_stream::{StreamFinalizeError, TelegramStream};

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

    fn try_animation(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        self.execute(action).map(|_receipt| true)
    }

    fn try_video(&mut self, action: TelegramAction) -> Result<Option<ActionReceipt>, Self::Error> {
        self.execute(action).map(Some)
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

    fn unit_interval(&mut self) -> Result<f64, Self::Error> {
        self.choice_index(10_000)
            .map(|sample| sample.min(9_999) as f64 / 10_000.0)
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BillingBalances {
    pub user_balance: i64,
    pub chat_balance: Option<i64>,
    pub diagnostics: Vec<String>,
}

pub trait BillingBalanceSource {
    fn load(&mut self, user_id: i64, chat_id: Option<i64>) -> Result<BillingBalances, String>;
}

pub trait BillingTransferSink {
    fn transfer(
        &mut self,
        user_id: i64,
        chat_id: i64,
        amount: i64,
    ) -> Result<TransferResult, String>;
}

pub trait ChargeHistorySource {
    fn load(
        &mut self,
        user_id: i64,
        limit: usize,
        cursor_id: Option<i64>,
        direction: &str,
    ) -> Result<ChargeHistoryPage, String>;
}

pub trait AdminCreditSink {
    fn mint(&mut self, user_id: i64, amount: i64) -> Result<i64, String>;
}

pub trait AdminCreditLogSource {
    fn load(&mut self, limit: usize) -> Result<Vec<CreditLogEntry>, String>;
}

pub trait BitcoinPriceSource {
    fn price(&mut self, currency: &str) -> Result<Option<f64>, String>;
}

pub trait DollarQuotesSource {
    fn devo_quotes(&mut self) -> Result<Option<DevoQuotes>, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct DollarMarketLoad {
    pub text: Option<String>,
    pub diagnostics: Vec<String>,
}

pub trait DollarMarketSource {
    fn load(
        &mut self,
        hours_ago: i64,
        locale: bot_core::locale::Locale,
        now_unix: i64,
    ) -> DollarMarketLoad;
}

#[derive(Debug, Clone, PartialEq)]
pub struct BcraLoad {
    pub text: Option<String>,
    pub diagnostics: Vec<String>,
}

pub trait BcraSource {
    fn load(&mut self, locale: bot_core::locale::Locale, now_unix: i64) -> BcraLoad;
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuloInputLoad {
    pub input: RuloInput,
    pub diagnostics: Vec<String>,
}

pub trait RuloSource {
    fn rulo_input(&mut self) -> Result<RuloInputLoad, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreetingPoolLoad {
    pub urls: Vec<String>,
    pub diagnostics: Vec<String>,
}

pub trait GreetingPoolSource {
    fn pool(&mut self, category: GreetingCategory) -> GreetingPoolLoad;
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeatherObservationLoad {
    pub observation: Option<WeatherObservation>,
    pub diagnostics: Vec<String>,
}

pub trait WeatherSource {
    fn load(&mut self, location: &str, now_unix: i64) -> WeatherObservationLoad;
}

#[derive(Debug, Clone, PartialEq)]
pub struct OilQuoteLoad {
    pub brent: Option<StockQuote>,
    pub wti: Option<StockQuote>,
    pub diagnostics: Vec<String>,
}

pub trait OilPriceSource {
    fn load(&mut self, now_unix: i64) -> OilQuoteLoad;
}

#[derive(Debug, Clone, PartialEq)]
pub struct StockQuotesLoad {
    pub quotes: Option<Vec<(String, Option<StockQuote>)>>,
    pub diagnostics: Vec<String>,
}

pub trait StockPriceSource {
    fn load(&mut self, query: &str, now_unix: i64) -> StockQuotesLoad;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketPriceLoad {
    pub text: String,
    pub diagnostics: Vec<String>,
}

pub trait MarketPriceSource {
    fn load(
        &mut self,
        query: &str,
        command: MarketPriceCommand,
        locale: bot_core::locale::Locale,
        now_unix: i64,
    ) -> MarketPriceLoad;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkReplacementLoad {
    pub replacement: LinkReplacement,
    pub context: Option<String>,
    pub oversized_video: Option<Vec<u8>>,
    pub diagnostics: Vec<String>,
}

pub trait LinkReplacementSource {
    fn load(&mut self, text: &str, now_unix: i64) -> LinkReplacementLoad;
}

pub trait ScheduledTaskSource {
    fn list(&mut self, chat_id: &str) -> Result<Vec<ScheduledTask>, String>;

    fn cancel(&mut self, task_id: &TaskId, chat_id: &str) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectionLoad {
    pub events: Vec<ElectionEvent>,
    pub live_prices: std::collections::HashMap<String, f64>,
    pub diagnostics: Vec<String>,
}

pub trait ElectionSource {
    fn load(&mut self, now_unix: i64) -> ElectionLoad;
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
    balance_source: Option<Box<dyn BillingBalanceSource>>,
    transfer_sink: Option<Box<dyn BillingTransferSink>>,
    charge_history_source: Option<Box<dyn ChargeHistorySource>>,
    admin_user_id: Option<i64>,
    admin_credit_sink: Option<Box<dyn AdminCreditSink>>,
    admin_creditlog_source: Option<Box<dyn AdminCreditLogSource>>,
    bitcoin_price_source: Option<Box<dyn BitcoinPriceSource>>,
    dollar_quotes_source: Option<Box<dyn DollarQuotesSource>>,
    dollar_market_source: Option<Box<dyn DollarMarketSource>>,
    bcra_source: Option<Box<dyn BcraSource>>,
    rulo_source: Option<Box<dyn RuloSource>>,
    greeting_pool_source: Option<Box<dyn GreetingPoolSource>>,
    weather_source: Option<Box<dyn WeatherSource>>,
    oil_price_source: Option<Box<dyn OilPriceSource>>,
    stock_price_source: Option<Box<dyn StockPriceSource>>,
    market_price_source: Option<Box<dyn MarketPriceSource>>,
    election_source: Option<Box<dyn ElectionSource>>,
    link_replacement_source: Option<Box<dyn LinkReplacementSource>>,
    scheduled_task_source: Option<Box<dyn ScheduledTaskSource>>,
    ai_conversation_source: Option<Box<dyn AiConversationSource>>,
    trigger_words: Vec<String>,
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
            balance_source: None,
            transfer_sink: None,
            charge_history_source: None,
            admin_user_id: None,
            admin_credit_sink: None,
            admin_creditlog_source: None,
            bitcoin_price_source: None,
            dollar_quotes_source: None,
            dollar_market_source: None,
            bcra_source: None,
            rulo_source: None,
            greeting_pool_source: None,
            weather_source: None,
            oil_price_source: None,
            stock_price_source: None,
            market_price_source: None,
            election_source: None,
            link_replacement_source: None,
            scheduled_task_source: None,
            ai_conversation_source: None,
            trigger_words: vec!["bot".to_owned(), "assistant".to_owned()],
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

    #[must_use]
    pub fn with_balance_source(mut self, source: Box<dyn BillingBalanceSource>) -> Self {
        self.balance_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_transfer_sink(mut self, sink: Box<dyn BillingTransferSink>) -> Self {
        self.transfer_sink = Some(sink);
        self
    }

    #[must_use]
    pub fn with_charge_history_source(mut self, source: Box<dyn ChargeHistorySource>) -> Self {
        self.charge_history_source = Some(source);
        self
    }

    #[must_use]
    pub const fn with_admin_user_id(mut self, user_id: Option<i64>) -> Self {
        self.admin_user_id = user_id;
        self
    }

    #[must_use]
    pub fn with_admin_credit_sink(mut self, sink: Box<dyn AdminCreditSink>) -> Self {
        self.admin_credit_sink = Some(sink);
        self
    }

    #[must_use]
    pub fn with_admin_creditlog_source(mut self, source: Box<dyn AdminCreditLogSource>) -> Self {
        self.admin_creditlog_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_bitcoin_price_source(mut self, source: Box<dyn BitcoinPriceSource>) -> Self {
        self.bitcoin_price_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_dollar_quotes_source(mut self, source: Box<dyn DollarQuotesSource>) -> Self {
        self.dollar_quotes_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_dollar_market_source(mut self, source: Box<dyn DollarMarketSource>) -> Self {
        self.dollar_market_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_bcra_source(mut self, source: Box<dyn BcraSource>) -> Self {
        self.bcra_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_rulo_source(mut self, source: Box<dyn RuloSource>) -> Self {
        self.rulo_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_greeting_pool_source(mut self, source: Box<dyn GreetingPoolSource>) -> Self {
        self.greeting_pool_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_weather_source(mut self, source: Box<dyn WeatherSource>) -> Self {
        self.weather_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_oil_price_source(mut self, source: Box<dyn OilPriceSource>) -> Self {
        self.oil_price_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_stock_price_source(mut self, source: Box<dyn StockPriceSource>) -> Self {
        self.stock_price_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_market_price_source(mut self, source: Box<dyn MarketPriceSource>) -> Self {
        self.market_price_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_election_source(mut self, source: Box<dyn ElectionSource>) -> Self {
        self.election_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_link_replacement_source(mut self, source: Box<dyn LinkReplacementSource>) -> Self {
        self.link_replacement_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_scheduled_task_source(mut self, source: Box<dyn ScheduledTaskSource>) -> Self {
        self.scheduled_task_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_ai_conversation_source(mut self, source: Box<dyn AiConversationSource>) -> Self {
        self.ai_conversation_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_trigger_words(mut self, trigger_words: Vec<String>) -> Self {
        self.trigger_words = trigger_words;
        self
    }

    fn dispatch_link_replacement(
        &mut self,
        message: &IncomingMessage,
        config: &ChatConfig,
        locale: bot_core::locale::Locale,
        timestamp: i64,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(sender_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let text = content.text.as_str();
        let mode = LinkMode::parse(&config.link_mode);
        if mode == LinkMode::Off || text.is_empty() || text.starts_with('/') {
            return Ok(DispatchOutcome::LegacyRequired);
        }
        if message.has_reply {
            let without_links = regex::Regex::new(r"https?://[^\s]+").ok().map_or_else(
                || text.to_owned(),
                |pattern| pattern.replace_all(text, "").into_owned(),
            );
            if !without_links.trim().is_empty() {
                return Ok(DispatchOutcome::LegacyRequired);
            }
        }
        if !has_replaceable_link(text) {
            return Ok(DispatchOutcome::LegacyRequired);
        }
        let Some(source) = self.link_replacement_source.as_mut() else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let load = source.load(text, timestamp);
        self.state_diagnostics.extend(load.diagnostics);
        if !load.replacement.changed {
            let incoming = prepare_incoming_command_state(IncomingCommandState {
                chat_id,
                message_id,
                user_id: sender_id,
                first_name: message.sender_first_name.as_deref(),
                username: message.sender_username.as_deref(),
                text,
                is_group: is_group_chat_type(message.chat_type.as_deref()),
                timestamp,
            });
            if let Ok(incoming) = incoming
                && let Err(error) = self.state.record_incoming(&incoming)
            {
                self.state_diagnostics
                    .push(format!("unreplaced link state: {error}"));
            }
            return Ok(DispatchOutcome::Handled);
        }
        let shared_by = message.sender_username.as_deref().map_or_else(
            || {
                [
                    message.sender_first_name.as_deref(),
                    message.sender_last_name.as_deref(),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" ")
            },
            |username| format!("@{username}"),
        );
        let Some(plan) = plan_link_actions(
            &load.replacement,
            mode,
            LinkActionContext {
                chat_id,
                incoming_message_id: message_id,
                replied_message_id: message.replied_message_id,
                shared_by: (!shared_by.is_empty()).then_some(shared_by.as_str()),
                locale,
                link_context: load.context.as_deref(),
            },
        ) else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let video_action = load.oversized_video.map(|video| {
            let TelegramAction::SendMessage(message) = &plan.send else {
                return plan.send.clone();
            };
            TelegramAction::SendVideo {
                chat_id: message.chat_id,
                video,
                reply_to_message_id: message.reply_to_message_id,
                caption: message.text.clone(),
                reply_markup: message.reply_markup.clone(),
            }
        });
        let receipt = if let Some(video_action) = video_action {
            match self
                .actions
                .try_video(video_action)
                .map_err(DispatchError::Action)?
            {
                Some(receipt) => receipt,
                None => self
                    .actions
                    .execute(plan.send)
                    .map_err(DispatchError::Action)?,
            }
        } else {
            self.actions
                .execute(plan.send)
                .map_err(DispatchError::Action)?
        };
        if let Some(delete) = plan.delete_original {
            let _receipt = self
                .actions
                .execute(delete)
                .map_err(DispatchError::Action)?;
        }
        let outgoing = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id,
            incoming_message_id: message_id,
            sent_message_id: receipt.message_id,
            text: &plan.stored_text,
            command: "fixed_link",
            timestamp,
        });
        if let Ok(outgoing) = outgoing
            && let Err(error) = self.state.record_outgoing(&outgoing)
        {
            self.state_diagnostics
                .push(format!("fixed link state: {error}"));
        }
        Ok(DispatchOutcome::Handled)
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

    fn dispatch_task_callback(
        &mut self,
        context: &CallbackContext,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        if self.scheduled_task_source.is_none() {
            return Ok(DispatchOutcome::LegacyRequired);
        }
        let config = self
            .config
            .get(&context.chat_id)
            .map_err(DispatchError::Config)?;
        let locale = resolve_locale(
            Some(&config.language),
            context.user_language_code.as_deref(),
            &context.chat_type,
        );
        let TaskCallbackParse::Delete(task_id) = parse_task_callback(&context.data) else {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        };
        let Some(source) = self.scheduled_task_source.as_mut() else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let tasks = match source.list(&context.chat_id) {
            Ok(tasks) => tasks,
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "scheduled task list callback chat_id={}: {error}",
                    context.chat_id
                ));
                Vec::new()
            }
        };
        let Some(target) = tasks.iter().find(|task| task.id == task_id).cloned() else {
            if let Some(callback_id) = context.callback_id.as_deref() {
                let _receipt = self
                    .actions
                    .execute(TelegramAction::AnswerCallback {
                        callback_id: callback_id.to_owned(),
                        text: Some(task_not_found(locale).to_owned()),
                        show_alert: true,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        };
        let is_group = is_group_chat_type(Some(&context.chat_type));
        let authorization = if is_group {
            context.user_id.map_or(
                GroupAuthorizationDecision {
                    is_admin: false,
                    diagnostics: Vec::new(),
                },
                |user_id| {
                    self.authorization
                        .authorize(&context.chat_id, &user_id.to_string())
                },
            )
        } else {
            GroupAuthorizationDecision {
                is_admin: true,
                diagnostics: Vec::new(),
            }
        };
        self.state_diagnostics.extend(authorization.diagnostics);
        if !can_delete_task(
            is_group,
            context.user_id,
            target.user_id,
            authorization.is_admin,
        ) {
            if let Some(callback_id) = context.callback_id.as_deref() {
                let _receipt = self
                    .actions
                    .execute(TelegramAction::AnswerCallback {
                        callback_id: callback_id.to_owned(),
                        text: Some(task_delete_forbidden(locale).to_owned()),
                        show_alert: true,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
        if let Err(error) = source.cancel(&task_id, &context.chat_id) {
            self.state_diagnostics.push(format!(
                "scheduled task cancellation chat_id={} task_id={}: {error}",
                context.chat_id,
                task_id.as_str()
            ));
        }
        if let Some(callback_id) = context.callback_id.as_deref() {
            let _receipt = self
                .actions
                .execute(TelegramAction::AnswerCallback {
                    callback_id: callback_id.to_owned(),
                    text: Some(task_deleted(&task_id, locale)),
                    show_alert: false,
                })
                .map_err(DispatchError::Action)?;
        }
        let tasks = source.list(&context.chat_id).unwrap_or_else(|error| {
            self.state_diagnostics.push(format!(
                "scheduled task list after cancellation chat_id={}: {error}",
                context.chat_id
            ));
            Vec::new()
        });
        let view = render_task_list(&tasks, locale);
        let Ok(chat_id) = context.chat_id.parse::<i64>() else {
            return Ok(DispatchOutcome::Handled);
        };
        if self
            .actions
            .try_edit(TelegramAction::EditMessage {
                chat_id: ChatId(chat_id),
                message_id: MessageId(context.message_id),
                text: view.text,
                reply_markup: view.keyboard,
            })
            .is_err()
        {
            self.state_diagnostics.push(format!(
                "scheduled task edit failed chat_id={} message_id={}",
                context.chat_id, context.message_id
            ));
        }
        Ok(DispatchOutcome::Handled)
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
        if context.route == CallbackRoute::Task {
            return self.dispatch_task_callback(&context);
        }
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
        if context.route == CallbackRoute::Charges {
            let Ok(chat_id_value) = context.chat_id.parse::<i64>() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let config = self
                .config
                .get(&context.chat_id)
                .map_err(DispatchError::Config)?;
            let locale = resolve_locale(
                Some(&config.language),
                context.user_language_code.as_deref(),
                &context.chat_type,
            );
            let load = match plan_charge_history_callback(
                context.callback_id.as_deref(),
                &context.data,
                context.user_id,
                locale,
            ) {
                ChargeHistoryCallbackPlan::Answer(action) => {
                    if let Some(action) = action {
                        let _receipt = self
                            .actions
                            .execute(action)
                            .map_err(DispatchError::Action)?;
                    }
                    return Ok(DispatchOutcome::Handled);
                }
                ChargeHistoryCallbackPlan::Load {
                    owner_id,
                    limit,
                    direction,
                    cursor_id,
                    timezone_minutes,
                } => (owner_id, limit, direction, cursor_id, timezone_minutes),
            };
            let (owner_id, limit, direction, cursor_id, timezone_minutes) = load;
            let Some(source) = self.charge_history_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let page = match source.load(owner_id, limit, Some(cursor_id), direction.as_str()) {
                Ok(page) => page,
                Err(error) => {
                    self.state_diagnostics.push(format!(
                        "charge history pagination chat_id={} user_id={owner_id} cursor_id={cursor_id} direction={}: {error}",
                        context.chat_id,
                        direction.as_str()
                    ));
                    if let Some(action) = charge_callback_answer(
                        context.callback_id.as_deref(),
                        Some(match locale {
                            bot_core::locale::Locale::Es => "se trabó leyendo tus gastos",
                            bot_core::locale::Locale::En => "I could not load your expenses",
                        }),
                        true,
                    ) {
                        let _receipt = self
                            .actions
                            .execute(action)
                            .map_err(DispatchError::Action)?;
                    }
                    return Ok(DispatchOutcome::Handled);
                }
            };
            if page.groups.is_empty() {
                if let Some(action) = charge_callback_answer(
                    context.callback_id.as_deref(),
                    Some(match locale {
                        bot_core::locale::Locale::Es => "no hay más gastos",
                        bot_core::locale::Locale::En => "there are no more expenses",
                    }),
                    false,
                ) {
                    let _receipt = self
                        .actions
                        .execute(action)
                        .map_err(DispatchError::Action)?;
                }
                return Ok(DispatchOutcome::Handled);
            }
            let (text, keyboard) =
                render_charge_history_page(&page, owner_id, limit, timezone_minutes, locale);
            let edited = match self.actions.try_edit(TelegramAction::EditMessage {
                chat_id: ChatId(chat_id_value),
                message_id: MessageId(context.message_id),
                text,
                reply_markup: Some(keyboard.unwrap_or(
                    bot_core::telegram_actions::InlineKeyboardMarkup {
                        inline_keyboard: Vec::new(),
                    },
                )),
            }) {
                Ok(edited) => edited,
                Err(_error) => {
                    self.state_diagnostics.push(format!(
                        "charge history edit failed chat_id={} message_id={}",
                        context.chat_id, context.message_id
                    ));
                    if let Some(action) = charge_callback_answer(
                        context.callback_id.as_deref(),
                        Some(match locale {
                            bot_core::locale::Locale::Es => "se trabó leyendo tus gastos",
                            bot_core::locale::Locale::En => "I could not load your expenses",
                        }),
                        true,
                    ) {
                        let _receipt = self
                            .actions
                            .execute(action)
                            .map_err(DispatchError::Action)?;
                    }
                    return Ok(DispatchOutcome::Handled);
                }
            };
            if let Some(action) = charge_callback_answer(
                context.callback_id.as_deref(),
                (!edited).then_some(match locale {
                    bot_core::locale::Locale::Es => "no pude actualizar el historial",
                    bot_core::locale::Locale::En => "I could not update the history",
                }),
                !edited,
            ) {
                let _receipt = self
                    .actions
                    .execute(action)
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
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

    fn dispatch_ai_message(
        &mut self,
        message: &IncomingMessage,
        config: &ChatConfig,
        locale: bot_core::locale::Locale,
        timestamp: i64,
        command: &str,
        prompt_text: &str,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(sender_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        if self.ai_conversation_source.is_none() {
            return Ok(DispatchOutcome::LegacyRequired);
        }

        let bot_username = self.bot_name.trim().trim_start_matches('@');
        let mention = (!bot_username.is_empty())
            && content
                .text
                .to_lowercase()
                .contains(&format!("@{}", bot_username.to_lowercase()));
        let reply_to_bot = message.replied_sender_username.as_deref() == Some(bot_username);
        let command_name = command
            .trim_start_matches('/')
            .split('@')
            .next()
            .unwrap_or_default();
        let known_command = telegram_commands(locale)
            .iter()
            .any(|candidate| candidate.command == command_name);
        let reply_metadata = if reply_to_bot {
            message.replied_message_id.and_then(|reply_id| {
                let source = self.ai_conversation_source.as_mut()?;
                match source.reply_metadata(&chat_id.0.to_string(), &reply_id.0.to_string()) {
                    Ok(metadata) => metadata,
                    Err(error) => {
                        self.state_diagnostics
                            .push(format!("AI reply metadata: {error}"));
                        None
                    }
                }
            })
        } else {
            None
        };
        let mut routing = ResponseRoutingInput {
            known_command,
            command_starts_with_slash: command.starts_with('/'),
            message_text: prompt_text.to_owned(),
            is_private: message.chat_type.as_deref() == Some("private"),
            is_mention: mention,
            is_reply: reply_to_bot,
            reply_text: message.replied_text.clone().unwrap_or_default(),
            ignore_link_fix_followups: config.ignore_link_fix_followups,
            is_non_ai_command_followup: reply_metadata
                .as_ref()
                .is_some_and(crate::ai_dispatch::AiReplyMetadata::is_non_ai_command),
            ai_command_followups: config.ai_command_followups,
            random_replies_enabled: config.ai_random_replies,
            trigger_words: Some(self.trigger_words.clone()),
            random_sample: None,
        };
        let evaluation = loop {
            match evaluate_response_routing(&routing) {
                ResponseRoutingEvaluation::NeedsTriggerWords => {
                    routing.trigger_words = Some(self.trigger_words.clone());
                }
                ResponseRoutingEvaluation::NeedsRandomSample => {
                    routing.random_sample =
                        Some(self.random.unit_interval().map_err(DispatchError::Random)?);
                }
                resolved => break resolved,
            }
        };
        let spontaneous = !routing.is_private
            && !routing.known_command
            && !routing.is_mention
            && !routing.is_reply;
        let ai_prompt_text = if known_command {
            prompt_text
        } else {
            content.text.as_str()
        };
        let input = AiConversationInput {
            chat_id,
            message_id,
            chat_type: message.chat_type.clone().unwrap_or_default(),
            chat_title: message.chat_title.clone().unwrap_or_default(),
            sender_id,
            sender_first_name: message.sender_first_name.clone().unwrap_or_default(),
            sender_username: message.sender_username.clone().unwrap_or_default(),
            message_text: ai_prompt_text.to_owned(),
            command: command.to_owned(),
            reply_to_message_id: message.replied_message_id,
            reply_context: reply_context(
                message.replied_sender_first_name.as_deref(),
                message.replied_sender_username.as_deref(),
                message.replied_text.as_deref(),
            ),
            has_reply: message.has_reply,
            visual_media_kind: message.visual_media_kind.clone(),
            audio_media_kind: message.audio_media_kind.clone(),
            photo_file_id: content.photo_file_id.clone(),
            audio_file_id: content.audio_file_id.clone(),
            audio_duration_seconds: message.audio_duration_seconds.map(|value| value as f64),
            locale,
            timezone_offset_hours: config.timezone_offset,
            timestamp,
            spontaneous,
        };
        if evaluation == ResponseRoutingEvaluation::Ignore {
            if let Some(source) = self.ai_conversation_source.as_mut()
                && let Err(error) = source.record_ignored(input)
            {
                self.state_diagnostics
                    .push(format!("ignored AI message state: {error}"));
            }
            return Ok(DispatchOutcome::Handled);
        }

        let (preparation, stream_finalize, ignored_edit_failures) = {
            let Some(source) = self.ai_conversation_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let mut stream = TelegramStream::new(&mut self.actions, chat_id, message_id);
            let preparation = source.prepare_streaming(input, &mut |token| {
                stream
                    .feed(token)
                    .map_err(|_error| "Telegram rejected the initial streamed response".to_owned())
            });
            match preparation {
                Err(error) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (Err(error), None, ignored)
                }
                Ok(AiPreparation::Silent { diagnostics }) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (Ok(AiPreparation::Silent { diagnostics }), None, ignored)
                }
                Ok(AiPreparation::Reply {
                    text,
                    completion_id,
                    diagnostics,
                }) => {
                    let finalized = stream.finalize(&text);
                    let ignored = stream.ignored_edit_failures();
                    (
                        Ok(AiPreparation::Reply {
                            text,
                            completion_id,
                            diagnostics,
                        }),
                        Some(finalized),
                        ignored,
                    )
                }
            }
        };
        if ignored_edit_failures > 0 {
            self.state_diagnostics.push(format!(
                "AI Telegram stream ignored {ignored_edit_failures} intermediate edit failures"
            ));
        }
        let preparation = match preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                self.state_diagnostics
                    .push(format!("AI conversation: {error}"));
                return Ok(DispatchOutcome::LegacyRequired);
            }
        };
        let (completion_id, diagnostics) = match preparation {
            AiPreparation::Silent { diagnostics } => {
                self.state_diagnostics.extend(diagnostics);
                return Ok(DispatchOutcome::Handled);
            }
            AiPreparation::Reply {
                completion_id,
                diagnostics,
                ..
            } => (completion_id, diagnostics),
        };
        self.state_diagnostics.extend(diagnostics);
        let Some(stream_finalize) = stream_finalize else {
            return Ok(DispatchOutcome::Handled);
        };
        match stream_finalize {
            Ok(delivery) => {
                if let Some(completion_id) = completion_id
                    && let Some(source) = self.ai_conversation_source.as_mut()
                    && let Err(error) = source.complete_delivery(AiDelivery {
                        completion_id,
                        delivered: true,
                        sent_message_id: Some(delivery.message_id),
                    })
                {
                    self.state_diagnostics
                        .push(format!("AI delivery completion: {error}"));
                }
                Ok(DispatchOutcome::Handled)
            }
            Err(StreamFinalizeError::MissingMessageId) => {
                if let Some(completion_id) = completion_id
                    && let Some(source) = self.ai_conversation_source.as_mut()
                    && let Err(error) = source.complete_delivery(AiDelivery {
                        completion_id,
                        delivered: false,
                        sent_message_id: None,
                    })
                {
                    self.state_diagnostics
                        .push(format!("AI unconfirmed delivery completion: {error}"));
                }
                self.state_diagnostics
                    .push("AI Telegram send returned no message identifier".to_owned());
                Ok(DispatchOutcome::Handled)
            }
            Err(StreamFinalizeError::Action(error)) => {
                if let Some(completion_id) = completion_id
                    && let Some(source) = self.ai_conversation_source.as_mut()
                    && let Err(finalize_error) = source.complete_delivery(AiDelivery {
                        completion_id,
                        delivered: false,
                        sent_message_id: None,
                    })
                {
                    self.state_diagnostics
                        .push(format!("AI delivery failure completion: {finalize_error}"));
                }
                Err(DispatchError::Action(error))
            }
        }
    }

    fn dispatch_media_command(
        &mut self,
        message: &IncomingMessage,
        config: &ChatConfig,
        locale: bot_core::locale::Locale,
        timestamp: i64,
        command: &str,
        prompt_text: &str,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(sender_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let Some(source) = self.ai_conversation_source.as_mut() else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        let input = AiConversationInput {
            chat_id,
            message_id,
            chat_type: message.chat_type.clone().unwrap_or_default(),
            chat_title: message.chat_title.clone().unwrap_or_default(),
            sender_id,
            sender_first_name: message.sender_first_name.clone().unwrap_or_default(),
            sender_username: message.sender_username.clone().unwrap_or_default(),
            message_text: prompt_text.to_owned(),
            command: command.to_owned(),
            reply_to_message_id: message.replied_message_id,
            reply_context: reply_context(
                message.replied_sender_first_name.as_deref(),
                message.replied_sender_username.as_deref(),
                message.replied_text.as_deref(),
            ),
            has_reply: message.has_reply,
            visual_media_kind: message.visual_media_kind.clone(),
            audio_media_kind: message.audio_media_kind.clone(),
            photo_file_id: content.photo_file_id.clone(),
            audio_file_id: content.audio_file_id.clone(),
            audio_duration_seconds: message.audio_duration_seconds.map(|value| value as f64),
            locale,
            timezone_offset_hours: config.timezone_offset,
            timestamp,
            spontaneous: false,
        };
        let preparation = match source.prepare_media_command(input) {
            Ok(Some(preparation)) => preparation,
            Ok(None) => return Ok(DispatchOutcome::LegacyRequired),
            Err(error) => {
                self.state_diagnostics
                    .push(format!("media command: {error}"));
                return Ok(DispatchOutcome::LegacyRequired);
            }
        };
        let AiPreparation::Reply {
            text,
            completion_id,
            diagnostics,
        } = preparation
        else {
            return Ok(DispatchOutcome::Handled);
        };
        self.state_diagnostics.extend(diagnostics);

        let incoming = prepare_incoming_command_state(IncomingCommandState {
            chat_id,
            message_id,
            user_id: sender_id,
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
                        .push(format!("incoming media command state: {error}"));
                }
            }
            Err(error) => self
                .state_diagnostics
                .push(format!("incoming media command state plan: {error}")),
        }

        let mut response = SendMessage::new(chat_id, &text);
        response.reply_to_message_id = Some(message_id);
        let receipt = match self.actions.execute(TelegramAction::SendMessage(response)) {
            Ok(receipt) => receipt,
            Err(error) => {
                if let Some(completion_id) = completion_id
                    && let Some(source) = self.ai_conversation_source.as_mut()
                    && let Err(finalize_error) = source.complete_delivery(AiDelivery {
                        completion_id,
                        delivered: false,
                        sent_message_id: None,
                    })
                {
                    self.state_diagnostics.push(format!(
                        "media command delivery failure completion: {finalize_error}"
                    ));
                }
                return Err(DispatchError::Action(error));
            }
        };
        if let Some(completion_id) = completion_id
            && let Some(source) = self.ai_conversation_source.as_mut()
            && let Err(error) = source.complete_delivery(AiDelivery {
                completion_id,
                delivered: receipt.message_id.is_some(),
                sent_message_id: receipt.message_id,
            })
        {
            self.state_diagnostics
                .push(format!("media command delivery completion: {error}"));
        }
        let outgoing = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id,
            incoming_message_id: message_id,
            sent_message_id: receipt.message_id,
            text: &text,
            command,
            timestamp,
        });
        match outgoing {
            Ok(outgoing) => {
                if let Err(error) = self.state.record_outgoing(&outgoing) {
                    self.state_diagnostics
                        .push(format!("outgoing media command state: {error}"));
                }
            }
            Err(error) => self
                .state_diagnostics
                .push(format!("outgoing media command state plan: {error}")),
        }
        Ok(DispatchOutcome::Handled)
    }

    fn dispatch_summary_command(
        &mut self,
        message: &IncomingMessage,
        config: &ChatConfig,
        locale: bot_core::locale::Locale,
        timestamp: i64,
        command: &str,
        prompt_text: &str,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(sender_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        if self.ai_conversation_source.is_none() {
            return Ok(DispatchOutcome::LegacyRequired);
        }
        let input = AiConversationInput {
            chat_id,
            message_id,
            chat_type: message.chat_type.clone().unwrap_or_default(),
            chat_title: message.chat_title.clone().unwrap_or_default(),
            sender_id,
            sender_first_name: message.sender_first_name.clone().unwrap_or_default(),
            sender_username: message.sender_username.clone().unwrap_or_default(),
            message_text: prompt_text.to_owned(),
            command: command.to_owned(),
            reply_to_message_id: message.replied_message_id,
            reply_context: reply_context(
                message.replied_sender_first_name.as_deref(),
                message.replied_sender_username.as_deref(),
                message.replied_text.as_deref(),
            ),
            has_reply: message.has_reply,
            visual_media_kind: message.visual_media_kind.clone(),
            audio_media_kind: message.audio_media_kind.clone(),
            photo_file_id: content.photo_file_id.clone(),
            audio_file_id: content.audio_file_id.clone(),
            audio_duration_seconds: message.audio_duration_seconds.map(|value| value as f64),
            locale,
            timezone_offset_hours: config.timezone_offset,
            timestamp,
            spontaneous: false,
        };
        let (preparation, stream_finalize, ignored_edit_failures) = {
            let Some(source) = self.ai_conversation_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let mut stream = TelegramStream::new(&mut self.actions, chat_id, message_id);
            let preparation = source.prepare_summary_command_streaming(input, &mut |token| {
                stream
                    .feed(token)
                    .map_err(|_error| "Telegram rejected the initial summary stream".to_owned())
            });
            match preparation {
                Err(error) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (Err(error), None, ignored)
                }
                Ok(None) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (Ok(None), None, ignored)
                }
                Ok(Some(AiPreparation::Silent { diagnostics })) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (
                        Ok(Some(AiPreparation::Silent { diagnostics })),
                        None,
                        ignored,
                    )
                }
                Ok(Some(AiPreparation::Reply {
                    text,
                    completion_id,
                    diagnostics,
                })) => {
                    let finalized = stream.finalize(&text);
                    let ignored = stream.ignored_edit_failures();
                    (
                        Ok(Some(AiPreparation::Reply {
                            text,
                            completion_id,
                            diagnostics,
                        })),
                        Some(finalized),
                        ignored,
                    )
                }
            }
        };
        if ignored_edit_failures > 0 {
            self.state_diagnostics.push(format!(
                "summary Telegram stream ignored {ignored_edit_failures} intermediate edit failures"
            ));
        }
        let preparation = match preparation {
            Ok(Some(preparation)) => preparation,
            Ok(None) => return Ok(DispatchOutcome::LegacyRequired),
            Err(error) => {
                self.state_diagnostics
                    .push(format!("summary command: {error}"));
                return Ok(DispatchOutcome::LegacyRequired);
            }
        };
        let AiPreparation::Reply {
            text,
            completion_id,
            diagnostics,
        } = preparation
        else {
            return Ok(DispatchOutcome::Handled);
        };
        self.state_diagnostics.extend(diagnostics);

        if let Ok(incoming) = prepare_incoming_command_state(IncomingCommandState {
            chat_id,
            message_id,
            user_id: sender_id,
            first_name: message.sender_first_name.as_deref(),
            username: message.sender_username.as_deref(),
            text: &content.text,
            is_group: is_group_chat_type(message.chat_type.as_deref()),
            timestamp,
        }) && let Err(error) = self.state.record_incoming(&incoming)
        {
            self.state_diagnostics
                .push(format!("incoming summary command state: {error}"));
        }
        let Some(stream_finalize) = stream_finalize else {
            return Ok(DispatchOutcome::Handled);
        };
        let receipt = match stream_finalize {
            Ok(receipt) => receipt,
            Err(StreamFinalizeError::MissingMessageId) => {
                if let Some(completion_id) = completion_id
                    && let Some(source) = self.ai_conversation_source.as_mut()
                {
                    let _result = source.complete_delivery(AiDelivery {
                        completion_id,
                        delivered: false,
                        sent_message_id: None,
                    });
                }
                self.state_diagnostics
                    .push("summary Telegram send returned no message identifier".to_owned());
                return Ok(DispatchOutcome::Handled);
            }
            Err(StreamFinalizeError::Action(error)) => {
                if let Some(completion_id) = completion_id
                    && let Some(source) = self.ai_conversation_source.as_mut()
                {
                    let _result = source.complete_delivery(AiDelivery {
                        completion_id,
                        delivered: false,
                        sent_message_id: None,
                    });
                }
                return Err(DispatchError::Action(error));
            }
        };
        if let Some(completion_id) = completion_id
            && let Some(source) = self.ai_conversation_source.as_mut()
            && let Err(error) = source.complete_delivery(AiDelivery {
                completion_id,
                delivered: true,
                sent_message_id: Some(receipt.message_id),
            })
        {
            self.state_diagnostics
                .push(format!("summary delivery completion: {error}"));
        }
        if let Ok(outgoing) = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id,
            incoming_message_id: message_id,
            sent_message_id: Some(receipt.message_id),
            text: &text,
            command,
            timestamp,
        }) && let Err(error) = self.state.record_outgoing(&outgoing)
        {
            self.state_diagnostics
                .push(format!("outgoing summary command state: {error}"));
        }
        Ok(DispatchOutcome::Handled)
    }

    fn dispatch_message(
        &mut self,
        message: &IncomingMessage,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(sender_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return Ok(DispatchOutcome::LegacyRequired);
        };
        self.state_diagnostics.clear();
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
        if !content.text.starts_with('/') && has_replaceable_link(&content.text) {
            return self.dispatch_link_replacement(message, &config, locale, timestamp);
        }
        if message.has_reply && self.ai_conversation_source.is_none() {
            return Ok(DispatchOutcome::LegacyRequired);
        }
        let parsed = parse_command(&content.text, &self.bot_name);
        if matches!(parsed.command.as_str(), "/transcribe" | "/describe") {
            return self.dispatch_media_command(
                message,
                &config,
                locale,
                timestamp,
                &parsed.command,
                &parsed.message_text,
            );
        }
        if matches!(parsed.command.as_str(), "/resumen" | "/summary" | "/tldr") {
            return self.dispatch_summary_command(
                message,
                &config,
                locale,
                timestamp,
                &parsed.command,
                &parsed.message_text,
            );
        }
        let is_group = is_group_chat_type(message.chat_type.as_deref());
        let is_settings_command = matches!(
            parsed.command.as_str(),
            "/language" | "/idioma" | "/config" | "/configs" | "/settings"
        );
        let mut language_needs_legacy_group = is_group;
        if is_group && is_settings_command {
            let authorization = self
                .authorization
                .authorize(&chat_id.0.to_string(), &sender_id.0.to_string());
            self.state_diagnostics.clear();
            self.state_diagnostics.extend(authorization.diagnostics);
            if !authorization.is_admin {
                self.state_diagnostics.push(format!(
                    "Unauthorized config attempt chat_id={} chat_type={} user_id={} username={} action=command:{}",
                    chat_id.0,
                    message.chat_type.as_deref().unwrap_or_default(),
                    sender_id.0,
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
        } else if matches!(
            parsed.command.as_str(),
            "/tarea" | "/tareas" | "/task" | "/tasks"
        ) {
            if !parsed.message_text.is_empty() {
                return Ok(DispatchOutcome::LegacyRequired);
            }
            let Some(source) = self.scheduled_task_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let tasks = match source.list(&chat_id.0.to_string()) {
                Ok(tasks) => tasks,
                Err(error) => {
                    self.state_diagnostics.push(format!(
                        "scheduled task list command chat_id={}: {error}",
                        chat_id.0
                    ));
                    Vec::new()
                }
            };
            let view = render_task_list(&tasks, locale);
            let mut message = SendMessage::new(chat_id, &view.text);
            message.reply_to_message_id = Some(message_id);
            message.reply_markup = view.keyboard;
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if parsed.command == "/balance" {
            match plan_balance_command(
                &content.text,
                &self.bot_name,
                BalanceCommandContext {
                    chat_id,
                    message_id,
                    user_id: Some(sender_id.0),
                    locale,
                    is_group,
                    billing_available: self.billing_available,
                },
            ) {
                BalanceCommandPlan::Reply(action) => StatelessCommandPlan::Action(action),
                BalanceCommandPlan::Load {
                    user_id,
                    chat_id,
                    is_group,
                } => {
                    let Some(source) = self.balance_source.as_mut() else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    self.state_diagnostics.clear();
                    let balances = source.load(user_id, is_group.then_some(chat_id.0));
                    let text = match balances {
                        Ok(balances) => {
                            self.state_diagnostics.extend(balances.diagnostics);
                            balance_reply(balances.user_balance, balances.chat_balance, locale)
                        }
                        Err(error) => {
                            self.state_diagnostics.push(format!(
                                "balance load chat_id={} user_id={user_id}: {error}",
                                chat_id.0
                            ));
                            match locale {
                                bot_core::locale::Locale::Es => {
                                    "se trabó leyendo tu saldo, probá de nuevo".to_owned()
                                }
                                bot_core::locale::Locale::En => {
                                    "I could not load your balance, try again".to_owned()
                                }
                            }
                        }
                    };
                    let mut message = SendMessage::new(chat_id, &text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                BalanceCommandPlan::NotHandled => StatelessCommandPlan::NotHandled,
            }
        } else if matches!(parsed.command.as_str(), "/charges" | "/history" | "/gastos") {
            match plan_charges_command(
                &content.text,
                &self.bot_name,
                ChargesCommandContext {
                    chat_id,
                    message_id,
                    user_id: Some(sender_id.0),
                    locale,
                    timezone_offset_hours: config.timezone_offset,
                    billing_available: self.billing_available,
                },
            ) {
                ChargesCommandPlan::Reply(action) => StatelessCommandPlan::Action(action),
                ChargesCommandPlan::Load {
                    user_id,
                    limit,
                    timezone_minutes,
                } => {
                    let Some(source) = self.charge_history_source.as_mut() else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    let (text, keyboard) = match source.load(user_id, limit, None, "older") {
                        Ok(page) => render_charge_history_page(
                            &page,
                            user_id,
                            limit,
                            timezone_minutes,
                            locale,
                        ),
                        Err(error) => {
                            self.state_diagnostics.push(format!(
                                "charge history load chat_id={} user_id={user_id} limit={limit}: {error}",
                                chat_id.0
                            ));
                            let text = match locale {
                                bot_core::locale::Locale::Es => {
                                    "se trabó leyendo tus gastos, probá de nuevo"
                                }
                                bot_core::locale::Locale::En => {
                                    "I could not load your expenses, try again"
                                }
                            };
                            (text.to_owned(), None)
                        }
                    };
                    let mut message = SendMessage::new(chat_id, &text);
                    message.reply_to_message_id = Some(message_id);
                    message.reply_markup = keyboard;
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                ChargesCommandPlan::NotHandled => StatelessCommandPlan::NotHandled,
            }
        } else if parsed.command == "/transfer" {
            match plan_transfer_command(
                &content.text,
                &self.bot_name,
                TransferCommandContext {
                    chat_id,
                    message_id,
                    user_id: Some(sender_id.0),
                    locale,
                    is_group,
                    billing_available: self.billing_available,
                },
            ) {
                TransferCommandPlan::Reply(action) => StatelessCommandPlan::Action(action),
                TransferCommandPlan::Transfer {
                    user_id,
                    chat_id,
                    amount,
                } => {
                    let Some(sink) = self.transfer_sink.as_mut() else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    let text = match sink.transfer(user_id, chat_id, amount) {
                        Ok(result) => transfer_result_reply(amount, result, locale),
                        Err(error) => {
                            self.state_diagnostics.push(format!(
                                "credit transfer chat_id={chat_id} user_id={user_id} amount={amount}: {error}"
                            ));
                            match locale {
                                bot_core::locale::Locale::Es => {
                                    "se trabó la transferencia, probá de nuevo".to_owned()
                                }
                                bot_core::locale::Locale::En => {
                                    "the transfer failed, try again".to_owned()
                                }
                            }
                        }
                    };
                    let mut message = SendMessage::new(ChatId(chat_id), &text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                TransferCommandPlan::NotHandled => StatelessCommandPlan::NotHandled,
            }
        } else if parsed.command == "/printcredits" {
            match plan_printcredits_command(
                &content.text,
                &self.bot_name,
                PrintCreditsContext {
                    chat_id,
                    message_id,
                    user_id: sender_id.0,
                    admin_user_id: self.admin_user_id,
                    billing_available: self.billing_available,
                    locale,
                },
            ) {
                PrintCreditsPlan::Reply(action) => StatelessCommandPlan::Action(action),
                PrintCreditsPlan::Mint { user_id, amount } => {
                    let Some(sink) = self.admin_credit_sink.as_mut() else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    let text = match sink.mint(user_id, amount) {
                        Ok(balance) => printcredits_result_reply(amount, balance, locale),
                        Err(error) => {
                            self.state_diagnostics.push(format!(
                                "admin credit mint chat_id={} user_id={user_id} amount={amount}: {error}",
                                chat_id.0
                            ));
                            match locale {
                                bot_core::locale::Locale::Es => {
                                    "se trabó imprimiendo créditos, probá de nuevo".to_owned()
                                }
                                bot_core::locale::Locale::En => {
                                    "I could not mint credits, try again".to_owned()
                                }
                            }
                        }
                    };
                    let mut message = SendMessage::new(chat_id, &text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                PrintCreditsPlan::NotHandled => StatelessCommandPlan::NotHandled,
            }
        } else if parsed.command == "/creditlog" {
            match plan_creditlog_command(
                &content.text,
                &self.bot_name,
                PrintCreditsContext {
                    chat_id,
                    message_id,
                    user_id: sender_id.0,
                    admin_user_id: self.admin_user_id,
                    billing_available: self.billing_available,
                    locale,
                },
            ) {
                CreditLogPlan::Reply(action) => StatelessCommandPlan::Action(action),
                CreditLogPlan::Load { limit } => {
                    let Some(source) = self.admin_creditlog_source.as_mut() else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    let text = match source.load(limit) {
                        Ok(entries) if entries.is_empty() => match locale {
                            bot_core::locale::Locale::Es => {
                                "no hay liquidaciones IA recientes".to_owned()
                            }
                            bot_core::locale::Locale::En => {
                                "there are no recent AI settlements".to_owned()
                            }
                        },
                        Ok(entries) => render_creditlog(&entries, locale),
                        Err(error) => {
                            self.state_diagnostics.push(format!(
                                "admin creditlog chat_id={} user_id={} limit={limit}: {error}",
                                chat_id.0, sender_id.0
                            ));
                            match locale {
                                bot_core::locale::Locale::Es => {
                                    "se trabó leyendo el creditlog, probá de nuevo".to_owned()
                                }
                                bot_core::locale::Locale::En => {
                                    "I could not load the credit log, try again".to_owned()
                                }
                            }
                        }
                    };
                    let mut message = SendMessage::new(chat_id, &text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                CreditLogPlan::NotHandled => StatelessCommandPlan::NotHandled,
                CreditLogPlan::LegacyRequired => StatelessCommandPlan::LegacyFallbackRequired,
            }
        } else if classify_bcra_command(&parsed.command) {
            let Some(source) = self.bcra_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let load = source.load(locale, timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = load.text.unwrap_or_else(|| match locale {
                bot_core::locale::Locale::Es => {
                    "No pude obtener las variables del BCRA en este momento, probá más tarde"
                        .to_owned()
                }
                bot_core::locale::Locale::En => {
                    "I could not load the BCRA variables right now".to_owned()
                }
            });
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if classify_dollar_command(&parsed.command) {
            match plan_dollar_command(&parsed.message_text) {
                DollarCommandPlan::InvalidTimeframe => {
                    let text = invalid_timeframe_message(&parsed.message_text, locale);
                    let mut message = SendMessage::new(chat_id, &text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
                DollarCommandPlan::Load { hours_ago } => {
                    let Some(source) = self.dollar_market_source.as_mut() else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    let load = source.load(hours_ago, locale, timestamp);
                    self.state_diagnostics.extend(load.diagnostics);
                    let text = load.text.unwrap_or_else(|| match locale {
                        bot_core::locale::Locale::Es => {
                            "no pude traer cotizaciones del dólar boludo".to_owned()
                        }
                        bot_core::locale::Locale::En => "I could not load dollar rates".to_owned(),
                    });
                    let mut message = SendMessage::new(chat_id, &text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
            }
        } else if classify_election_command(&parsed.command) {
            let Some(source) = self.election_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let load = source.load(timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = render_elections(&load.events, &load.live_prices, locale);
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            message.parse_mode = Some(ParseMode::Html);
            message.disable_web_page_preview = true;
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let Some(command) = classify_market_price_command(&parsed.command) {
            let Some(source) = self.market_price_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let load = source.load(&parsed.message_text, command, locale, timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let mut message = SendMessage::new(chat_id, &load.text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if classify_stock_command(&parsed.command) {
            let Some(source) = self.stock_price_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let load = source.load(&parsed.message_text, timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = render_stock_quotes(load.quotes.as_deref(), locale);
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if classify_oil_command(&parsed.command) {
            let Some(source) = self.oil_price_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let load = source.load(timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = render_oil_quotes(load.brent.as_ref(), load.wti.as_ref(), locale);
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if classify_weather_command(&parsed.command) {
            let location = requested_location(&parsed.message_text);
            let Some(source) = self.weather_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let load = source.load(location, timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = load.observation.as_ref().map_or_else(
                || weather_load_error(location, locale),
                |observation| render_weather(observation, locale),
            );
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let Some(category) = classify_greeting_command(&parsed.command) {
            let Some(source) = self.greeting_pool_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let load = source.pool(category);
            self.state_diagnostics.extend(load.diagnostics);
            if load.urls.is_empty() {
                let mut message = SendMessage::new(chat_id, greeting_fallback(category, locale));
                message.reply_to_message_id = Some(message_id);
                StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
            } else {
                let index = self
                    .random
                    .choice_index(load.urls.len())
                    .map_err(DispatchError::Random)?;
                let Some(animation) = load.urls.get(index) else {
                    return Ok(DispatchOutcome::LegacyRequired);
                };
                if animation.starts_with("http") {
                    StatelessCommandPlan::Action(TelegramAction::SendAnimation {
                        chat_id,
                        animation: animation.clone(),
                        reply_to_message_id: Some(message_id),
                        caption: None,
                    })
                } else {
                    let mut message = SendMessage::new(chat_id, animation);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
            }
        } else if parsed.command == "/rulo" {
            let Some(source) = self.rulo_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let text = match source.rulo_input() {
                Ok(load) => {
                    self.state_diagnostics.extend(load.diagnostics);
                    render_rulo(&evaluate_rulo(&load.input), locale)
                }
                Err(error) => {
                    self.state_diagnostics.push(format!(
                        "rulo quotes chat_id={} user_id={}: {error}",
                        chat_id.0, sender_id.0
                    ));
                    render_devo_reply(DevoReply::LoadError, locale)
                }
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if parsed.command == "/devo" {
            let text = match plan_devo_command(&parsed.message_text) {
                Err(_) => return Ok(DispatchOutcome::LegacyRequired),
                Ok(DevoCommandPlan::Reply(reply)) => render_devo_reply(reply, locale),
                Ok(DevoCommandPlan::Load { fee, purchase }) => {
                    let Some(source) = self.dollar_quotes_source.as_mut() else {
                        return Ok(DispatchOutcome::LegacyRequired);
                    };
                    let quotes = source.devo_quotes().unwrap_or_else(|error| {
                        self.state_diagnostics.push(format!(
                            "devo quotes chat_id={} user_id={}: {error}",
                            chat_id.0, sender_id.0
                        ));
                        None
                    });
                    match quotes.and_then(|quotes| calculate_devo(fee, purchase, quotes).ok()) {
                        Some(result) => render_devo_result(&result, locale),
                        None => render_devo_reply(DevoReply::LoadError, locale),
                    }
                }
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let Some(bitcoin_command) = classify_bitcoin_command(&parsed.command) {
            let Some(source) = self.bitcoin_price_source.as_mut() else {
                return Ok(DispatchOutcome::LegacyRequired);
            };
            let mut price = |source: &mut Box<dyn BitcoinPriceSource>, currency: &str| {
                source.price(currency).unwrap_or_else(|error| {
                    self.state_diagnostics.push(format!(
                        "bitcoin price chat_id={} command={} currency={currency}: {error}",
                        chat_id.0, parsed.command
                    ));
                    None
                })
            };
            let text = match bitcoin_command {
                BitcoinCommand::Satoshi => match price(source, "USD") {
                    None => bitcoin_price_error(bitcoin_command, "USD", locale),
                    Some(price_usd) => match price(source, "ARS") {
                        None => bitcoin_price_error(bitcoin_command, "ARS", locale),
                        Some(price_ars) => render_satoshi(price_usd, price_ars, locale),
                    },
                },
                BitcoinCommand::PowerLaw | BitcoinCommand::Rainbow => match price(source, "USD") {
                    Some(price) => render_market_model(bitcoin_command, timestamp, price, locale),
                    None => bitcoin_price_error(bitcoin_command, "USD", locale),
                },
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
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
                let command = parsed.command;
                let incoming = prepare_incoming_command_state(IncomingCommandState {
                    chat_id,
                    message_id,
                    user_id: sender_id,
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
                let receipt = if matches!(&action, TelegramAction::SendAnimation { .. }) {
                    let _sent = self
                        .actions
                        .try_animation(action)
                        .map_err(DispatchError::Action)?;
                    ActionReceipt { message_id: None }
                } else {
                    self.actions
                        .execute(action)
                        .map_err(DispatchError::Action)?
                };
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
            StatelessCommandPlan::NotHandled | StatelessCommandPlan::LegacyFallbackRequired => self
                .dispatch_ai_message(
                    message,
                    &config,
                    locale,
                    timestamp,
                    &parsed.command,
                    &parsed.message_text,
                ),
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
    use std::collections::HashMap;
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

    use crate::ai_dispatch::{
        AiConversationInput, AiConversationSource, AiDelivery, AiPreparation, AiReplyMetadata,
    };

    use super::{
        ActionReceipt, ActionSink, AdminCreditLogSource, AdminCreditSink, BcraLoad, BcraSource,
        BillingBalanceSource, BillingBalances, BillingTransferSink, BitcoinPriceSource,
        ChargeHistoryPage, ChargeHistorySource, ChatConfigSource, DispatchError, DispatchOutcome,
        DollarMarketLoad, DollarMarketSource, DollarQuotesSource, ElectionLoad, ElectionSource,
        GreetingPoolLoad, GreetingPoolSource, GroupAuthorizationDecision, GroupAuthorizer,
        LinkReplacementLoad, LinkReplacementSource, MarketPriceLoad, MarketPriceSource,
        MessageStateSink, NativeDispatcher, OilPriceSource, OilQuoteLoad, RandomSource,
        RuloInputLoad, RuloSource, RuntimeValues, ScheduledTaskSource, StarPaymentReceipt,
        StarPaymentSink, StockPriceSource, StockQuotesLoad, TransferResult, WeatherObservationLoad,
        WeatherSource,
    };
    use bot_core::charge_history::{ChargeHistoryEntry, ChargeHistoryGroup};
    use bot_core::devo::DevoQuotes;
    use bot_core::greeting_commands::GreetingCategory;
    use bot_core::links::LinkReplacement;
    use bot_core::polymarket::parse_election_events;
    use bot_core::rulo::{ExchangeQuote, RuloInput};
    use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule, TaskStateError};
    use bot_core::stocks::StockQuote;
    use bot_core::weather::WeatherObservation;

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

    struct AiSource {
        metadata: Option<AiReplyMetadata>,
        preparation: Option<Result<AiPreparation, String>>,
        media_preparation: Option<Result<AiPreparation, String>>,
        summary_preparation: Option<Result<AiPreparation, String>>,
        tokens: Vec<String>,
        prepared: Rc<RefCell<Vec<AiConversationInput>>>,
        ignored: Rc<RefCell<Vec<AiConversationInput>>>,
        deliveries: Rc<RefCell<Vec<AiDelivery>>>,
    }

    impl AiConversationSource for AiSource {
        fn reply_metadata(
            &mut self,
            _chat_id: &str,
            _message_id: &str,
        ) -> Result<Option<AiReplyMetadata>, String> {
            Ok(self.metadata.clone())
        }

        fn prepare(&mut self, input: AiConversationInput) -> Result<AiPreparation, String> {
            self.prepared.borrow_mut().push(input);
            self.preparation
                .take()
                .unwrap_or_else(|| Ok(AiPreparation::silent()))
        }

        fn prepare_streaming(
            &mut self,
            input: AiConversationInput,
            on_token: &mut dyn FnMut(&str) -> Result<(), String>,
        ) -> Result<AiPreparation, String> {
            self.prepared.borrow_mut().push(input);
            for token in &self.tokens {
                on_token(token)?;
            }
            self.preparation
                .take()
                .unwrap_or_else(|| Ok(AiPreparation::silent()))
        }

        fn prepare_media_command(
            &mut self,
            input: AiConversationInput,
        ) -> Result<Option<AiPreparation>, String> {
            let Some(preparation) = self.media_preparation.take() else {
                return Ok(None);
            };
            self.prepared.borrow_mut().push(input);
            preparation.map(Some)
        }

        fn prepare_summary_command_streaming(
            &mut self,
            input: AiConversationInput,
            on_token: &mut dyn FnMut(&str) -> Result<(), String>,
        ) -> Result<Option<AiPreparation>, String> {
            let Some(preparation) = self.summary_preparation.take() else {
                return Ok(None);
            };
            self.prepared.borrow_mut().push(input);
            for token in &self.tokens {
                on_token(token)?;
            }
            preparation.map(Some)
        }

        fn record_ignored(&mut self, input: AiConversationInput) -> Result<(), String> {
            self.ignored.borrow_mut().push(input);
            Ok(())
        }

        fn complete_delivery(&mut self, delivery: AiDelivery) -> Result<(), String> {
            self.deliveries.borrow_mut().push(delivery);
            Ok(())
        }
    }

    type AiObservations = (
        Rc<RefCell<Vec<AiConversationInput>>>,
        Rc<RefCell<Vec<AiConversationInput>>>,
        Rc<RefCell<Vec<AiDelivery>>>,
    );

    fn ai_source(preparation: Result<AiPreparation, String>) -> (AiSource, AiObservations) {
        let prepared = Rc::new(RefCell::new(Vec::new()));
        let ignored = Rc::new(RefCell::new(Vec::new()));
        let deliveries = Rc::new(RefCell::new(Vec::new()));
        (
            AiSource {
                metadata: None,
                preparation: Some(preparation),
                media_preparation: None,
                summary_preparation: None,
                tokens: Vec::new(),
                prepared: Rc::clone(&prepared),
                ignored: Rc::clone(&ignored),
                deliveries: Rc::clone(&deliveries),
            },
            (prepared, ignored, deliveries),
        )
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

    struct Links {
        replacement: LinkReplacement,
        context: Option<String>,
        oversized_video: Option<Vec<u8>>,
        diagnostics: Vec<String>,
        calls: Vec<(String, i64)>,
    }

    impl LinkReplacementSource for Links {
        fn load(&mut self, text: &str, now_unix: i64) -> LinkReplacementLoad {
            self.calls.push((text.to_owned(), now_unix));
            LinkReplacementLoad {
                replacement: self.replacement.clone(),
                context: self.context.clone(),
                oversized_video: self.oversized_video.clone(),
                diagnostics: self.diagnostics.clone(),
            }
        }
    }

    fn links(changed: bool) -> Links {
        Links {
            replacement: LinkReplacement {
                text: if changed {
                    "https://fixupx.com/a/status/1".to_owned()
                } else {
                    "https://x.com/a/status/1".to_owned()
                },
                changed,
                original_links: if changed {
                    vec!["https://x.com/a/status/1".to_owned()]
                } else {
                    Vec::new()
                },
            },
            context: changed.then(|| {
                "LINKS DEL MENSAJE:\n1. https://fixupx.com/a/status/1\ntitulo: example".to_owned()
            }),
            oversized_video: None,
            diagnostics: Vec::new(),
            calls: Vec::new(),
        }
    }

    struct Tasks {
        lists: Vec<Vec<ScheduledTask>>,
        cancellations: Rc<RefCell<Vec<(String, String)>>>,
    }

    impl ScheduledTaskSource for Tasks {
        fn list(&mut self, _chat_id: &str) -> Result<Vec<ScheduledTask>, String> {
            if self.lists.len() > 1 {
                Ok(self.lists.remove(0))
            } else {
                Ok(self.lists.first().cloned().unwrap_or_default())
            }
        }

        fn cancel(&mut self, task_id: &TaskId, chat_id: &str) -> Result<bool, String> {
            self.cancellations
                .borrow_mut()
                .push((task_id.as_str().to_owned(), chat_id.to_owned()));
            Ok(true)
        }
    }

    fn scheduled_task(owner_user_id: i64) -> Result<ScheduledTask, TaskStateError> {
        Ok(ScheduledTask {
            id: TaskId::new("task0001")?,
            chat_id: "-42".to_owned(),
            text: "synthetic reminder".to_owned(),
            user_name: "tester".to_owned(),
            user_id: Some(owner_user_id),
            schedule: TaskSchedule::IntervalSeconds { seconds: 3_600 },
            timezone_offset: -3,
            locale: "es".to_owned(),
            schedule_anchor_at: Some(1_777_523_400),
            next_run_at: Some(1_777_527_000),
            last_execution_id: None,
        })
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
            event: IncomingEvent::Message(Box::new(IncomingMessage {
                message_id: Some(MessageId(7)),
                chat_id: Some(ChatId(-42)),
                chat_type: Some("private".to_owned()),
                chat_title: None,
                sender_id: Some(UserId(88)),
                sender_first_name: Some("Synthetic".to_owned()),
                sender_last_name: None,
                sender_username: Some("tester".to_owned()),
                sender_language_code: language.map(ToOwned::to_owned),
                has_reply: false,
                replied_message_id: None,
                replied_sender_first_name: None,
                replied_sender_username: None,
                replied_text: None,
                visual_media_kind: None,
                audio_media_kind: None,
                audio_duration_seconds: None,
                content: Some(MessageContent {
                    text: text.to_owned(),
                    photo_file_id: None,
                    audio_file_id: None,
                }),
            })),
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

    type BalanceCalls = Rc<RefCell<Vec<(i64, Option<i64>)>>>;

    struct Balances {
        result: Result<BillingBalances, String>,
        calls: BalanceCalls,
    }

    impl BillingBalanceSource for Balances {
        fn load(&mut self, user_id: i64, chat_id: Option<i64>) -> Result<BillingBalances, String> {
            self.calls.borrow_mut().push((user_id, chat_id));
            self.result.clone()
        }
    }

    type TransferCalls = Rc<RefCell<Vec<(i64, i64, i64)>>>;

    struct Transfers {
        result: Result<TransferResult, String>,
        calls: TransferCalls,
    }

    impl BillingTransferSink for Transfers {
        fn transfer(
            &mut self,
            user_id: i64,
            chat_id: i64,
            amount: i64,
        ) -> Result<TransferResult, String> {
            self.calls.borrow_mut().push((user_id, chat_id, amount));
            self.result.clone()
        }
    }

    type AdminCreditCalls = Rc<RefCell<Vec<(i64, i64)>>>;

    struct AdminCredits {
        result: Result<i64, String>,
        calls: AdminCreditCalls,
    }

    impl AdminCreditSink for AdminCredits {
        fn mint(&mut self, user_id: i64, amount: i64) -> Result<i64, String> {
            self.calls.borrow_mut().push((user_id, amount));
            self.result.clone()
        }
    }

    type AdminCreditLogCalls = Rc<RefCell<Vec<usize>>>;

    struct AdminCreditLogs {
        result: Result<Vec<bot_core::admin_commands::CreditLogEntry>, String>,
        calls: AdminCreditLogCalls,
    }

    impl AdminCreditLogSource for AdminCreditLogs {
        fn load(
            &mut self,
            limit: usize,
        ) -> Result<Vec<bot_core::admin_commands::CreditLogEntry>, String> {
            self.calls.borrow_mut().push(limit);
            self.result.clone()
        }
    }

    type BitcoinPriceCalls = Rc<RefCell<Vec<String>>>;

    struct BitcoinPrices {
        results: Vec<Result<Option<f64>, String>>,
        calls: BitcoinPriceCalls,
    }

    impl BitcoinPriceSource for BitcoinPrices {
        fn price(&mut self, currency: &str) -> Result<Option<f64>, String> {
            self.calls.borrow_mut().push(currency.to_owned());
            if self.results.is_empty() {
                return Err("no synthetic price".to_owned());
            }
            self.results.remove(0)
        }
    }

    struct DollarQuotes {
        result: Result<Option<DevoQuotes>, String>,
        calls: Rc<RefCell<usize>>,
    }

    impl DollarQuotesSource for DollarQuotes {
        fn devo_quotes(&mut self) -> Result<Option<DevoQuotes>, String> {
            *self.calls.borrow_mut() += 1;
            self.result.clone()
        }
    }

    struct DollarMarket {
        result: DollarMarketLoad,
        calls: Rc<RefCell<Vec<(i64, bot_core::locale::Locale, i64)>>>,
    }

    struct BcraVariables {
        result: BcraLoad,
        calls: Rc<RefCell<Vec<(bot_core::locale::Locale, i64)>>>,
    }

    impl BcraSource for BcraVariables {
        fn load(&mut self, locale: bot_core::locale::Locale, now_unix: i64) -> BcraLoad {
            self.calls.borrow_mut().push((locale, now_unix));
            self.result.clone()
        }
    }

    impl DollarMarketSource for DollarMarket {
        fn load(
            &mut self,
            hours_ago: i64,
            locale: bot_core::locale::Locale,
            now_unix: i64,
        ) -> DollarMarketLoad {
            self.calls.borrow_mut().push((hours_ago, locale, now_unix));
            self.result.clone()
        }
    }

    struct RuloInputs {
        result: Result<RuloInputLoad, String>,
        calls: Rc<RefCell<usize>>,
    }

    impl RuloSource for RuloInputs {
        fn rulo_input(&mut self) -> Result<RuloInputLoad, String> {
            *self.calls.borrow_mut() += 1;
            self.result.clone()
        }
    }

    struct GreetingPools {
        result: GreetingPoolLoad,
        calls: Rc<RefCell<Vec<GreetingCategory>>>,
    }

    struct WeatherObservations {
        result: WeatherObservationLoad,
        calls: Rc<RefCell<Vec<(String, i64)>>>,
    }

    impl WeatherSource for WeatherObservations {
        fn load(&mut self, location: &str, now_unix: i64) -> WeatherObservationLoad {
            self.calls
                .borrow_mut()
                .push((location.to_owned(), now_unix));
            self.result.clone()
        }
    }

    struct OilQuotes {
        result: OilQuoteLoad,
        calls: Rc<RefCell<Vec<i64>>>,
    }

    impl OilPriceSource for OilQuotes {
        fn load(&mut self, now_unix: i64) -> OilQuoteLoad {
            self.calls.borrow_mut().push(now_unix);
            self.result.clone()
        }
    }

    struct StockQuotes {
        result: StockQuotesLoad,
        calls: Rc<RefCell<Vec<(String, i64)>>>,
    }

    type MarketPriceCalls = Rc<
        RefCell<
            Vec<(
                String,
                bot_core::market_prices::MarketPriceCommand,
                bot_core::locale::Locale,
                i64,
            )>,
        >,
    >;

    struct MarketPrices {
        result: MarketPriceLoad,
        calls: MarketPriceCalls,
    }

    impl MarketPriceSource for MarketPrices {
        fn load(
            &mut self,
            query: &str,
            command: bot_core::market_prices::MarketPriceCommand,
            locale: bot_core::locale::Locale,
            now_unix: i64,
        ) -> MarketPriceLoad {
            self.calls
                .borrow_mut()
                .push((query.to_owned(), command, locale, now_unix));
            self.result.clone()
        }
    }

    impl StockPriceSource for StockQuotes {
        fn load(&mut self, query: &str, now_unix: i64) -> StockQuotesLoad {
            self.calls.borrow_mut().push((query.to_owned(), now_unix));
            self.result.clone()
        }
    }

    struct Elections {
        result: ElectionLoad,
        calls: Rc<RefCell<Vec<i64>>>,
    }

    impl ElectionSource for Elections {
        fn load(&mut self, now_unix: i64) -> ElectionLoad {
            self.calls.borrow_mut().push(now_unix);
            self.result.clone()
        }
    }

    impl GreetingPoolSource for GreetingPools {
        fn pool(&mut self, category: GreetingCategory) -> GreetingPoolLoad {
            self.calls.borrow_mut().push(category);
            self.result.clone()
        }
    }

    type ChargeHistoryCalls = Rc<RefCell<Vec<(i64, usize, Option<i64>, String)>>>;

    struct ChargeHistories {
        result: Result<ChargeHistoryPage, String>,
        calls: ChargeHistoryCalls,
    }

    impl ChargeHistorySource for ChargeHistories {
        fn load(
            &mut self,
            user_id: i64,
            limit: usize,
            cursor_id: Option<i64>,
            direction: &str,
        ) -> Result<ChargeHistoryPage, String> {
            self.calls
                .borrow_mut()
                .push((user_id, limit, cursor_id, direction.to_owned()));
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
            event: IncomingEvent::Message(Box::new(IncomingMessage {
                message_id: None,
                chat_id: None,
                chat_type: None,
                chat_title: None,
                sender_id: None,
                sender_first_name: None,
                sender_last_name: None,
                sender_username: None,
                sender_language_code: None,
                has_reply: false,
                replied_message_id: None,
                replied_sender_first_name: None,
                replied_sender_username: None,
                replied_text: None,
                visual_media_kind: None,
                audio_media_kind: None,
                audio_duration_seconds: None,
                content: None,
            })),
        };
        assert_eq!(
            dispatcher.dispatch(incomplete),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn private_ai_turn_crosses_the_transaction_seam_and_acknowledges_delivery() {
        let (source, (prepared, ignored, deliveries)) = ai_source(Ok(AiPreparation::Reply {
            text: "native answer".to_owned(),
            completion_id: Some("conversation-1".to_owned()),
            diagnostics: vec!["provider diagnostic".to_owned()],
        }));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig {
                    language: "en".to_owned(),
                    timezone_offset: 4,
                    ..ChatConfig::default()
                }),
                chat_ids: Vec::new(),
            },
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_ai_conversation_source(Box::new(source));

        assert_eq!(
            dispatcher.dispatch(update("tell me something", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let prepared = prepared.borrow();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].message_text, "tell me something");
        assert_eq!(prepared[0].chat_type, "private");
        assert_eq!(prepared[0].sender_id, UserId(88));
        assert_eq!(prepared[0].timezone_offset_hours, 4);
        assert!(!prepared[0].spontaneous);
        assert!(ignored.borrow().is_empty());
        let [TelegramAction::SendMessage(message)] = dispatcher.actions.0.as_slice() else {
            return;
        };
        assert_eq!(message.text, "native answer");
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(
            deliveries.borrow().as_slice(),
            [AiDelivery {
                completion_id: "conversation-1".to_owned(),
                delivered: true,
                sent_message_id: Some(MessageId(700)),
            }]
        );
        assert_eq!(dispatcher.state_diagnostics(), ["provider diagnostic"]);
    }

    #[test]
    fn explicit_media_command_uses_its_native_transaction_and_command_state() {
        let (mut source, (prepared, ignored, deliveries)) = ai_source(Ok(AiPreparation::silent()));
        source.media_preparation = Some(Ok(AiPreparation::Reply {
            text: "🎵 audio transcription: synthetic transcript".to_owned(),
            completion_id: Some("media-1".to_owned()),
            diagnostics: vec!["media diagnostic".to_owned()],
        }));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig {
                    language: "en".to_owned(),
                    ..ChatConfig::default()
                }),
                chat_ids: Vec::new(),
            },
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_ai_conversation_source(Box::new(source));
        let mut incoming = update("/transcribe", Some("en"));
        let IncomingEvent::Message(message) = &mut incoming.event else {
            return;
        };
        message.has_reply = true;
        message.replied_message_id = Some(MessageId(6));
        message.audio_media_kind = Some("voice".to_owned());
        message.audio_duration_seconds = Some(4);
        if let Some(content) = message.content.as_mut() {
            content.audio_file_id = Some("voice-1".to_owned());
        }

        assert_eq!(dispatcher.dispatch(incoming), Ok(DispatchOutcome::Handled));
        let prepared = prepared.borrow();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].command, "/transcribe");
        assert!(prepared[0].has_reply);
        assert_eq!(prepared[0].audio_file_id.as_deref(), Some("voice-1"));
        assert_eq!(prepared[0].audio_duration_seconds, Some(4.0));
        assert!(ignored.borrow().is_empty());
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert!(
            dispatcher.state.outgoing[0]
                .metadata
                .as_ref()
                .is_some_and(|metadata| metadata.payload.contains("/transcribe"))
        );
        let [TelegramAction::SendMessage(message)] = dispatcher.actions.0.as_slice() else {
            return;
        };
        assert_eq!(message.text, "🎵 audio transcription: synthetic transcript");
        assert_eq!(
            deliveries.borrow().as_slice(),
            [AiDelivery {
                completion_id: "media-1".to_owned(),
                delivered: true,
                sent_message_id: Some(MessageId(700)),
            }]
        );
        assert_eq!(dispatcher.state_diagnostics(), ["media diagnostic"]);
    }

    #[test]
    fn summary_command_uses_native_stream_delivery_and_command_state() {
        let (mut source, (prepared, ignored, deliveries)) = ai_source(Ok(AiPreparation::silent()));
        source.tokens = vec!["raw summary".to_owned()];
        source.summary_preparation = Some(Ok(AiPreparation::Reply {
            text: "clean summary".to_owned(),
            completion_id: Some("summary-1".to_owned()),
            diagnostics: vec!["summary diagnostic".to_owned()],
        }));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig {
                    language: "en".to_owned(),
                    ..ChatConfig::default()
                }),
                chat_ids: Vec::new(),
            },
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_ai_conversation_source(Box::new(source));

        assert_eq!(
            dispatcher.dispatch(update("/summary focus on decisions", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let prepared = prepared.borrow();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].command, "/summary");
        assert_eq!(prepared[0].message_text, "focus on decisions");
        assert!(ignored.borrow().is_empty());
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(dispatcher.actions.0.len(), 2);
        assert!(matches!(
            &dispatcher.actions.0[0],
            TelegramAction::SendMessage(message) if message.text == "raw summary"
        ));
        assert!(matches!(
            &dispatcher.actions.0[1],
            TelegramAction::EditMessage { text, .. } if text == "clean summary"
        ));
        assert_eq!(
            deliveries.borrow().as_slice(),
            [AiDelivery {
                completion_id: "summary-1".to_owned(),
                delivered: true,
                sent_message_id: Some(MessageId(700)),
            }]
        );
        assert_eq!(dispatcher.state_diagnostics(), ["summary diagnostic"]);
    }

    #[test]
    fn private_ai_turn_streams_a_draft_then_finalizes_the_cleaned_response() {
        let (mut source, (_prepared, _ignored, deliveries)) = ai_source(Ok(AiPreparation::Reply {
            text: "cleaned answer".to_owned(),
            completion_id: Some("conversation-1".to_owned()),
            diagnostics: Vec::new(),
        }));
        source.tokens = vec!["raw ".to_owned(), "answer".to_owned()];
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_ai_conversation_source(Box::new(source));

        assert_eq!(
            dispatcher.dispatch(update("tell me something", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(dispatcher.actions.0.len(), 2);
        assert!(matches!(
            &dispatcher.actions.0[0],
            TelegramAction::SendMessage(message) if message.text == "raw "
        ));
        assert!(matches!(
            &dispatcher.actions.0[1],
            TelegramAction::EditMessage { text, .. } if text == "cleaned answer"
        ));
        assert_eq!(
            deliveries.borrow().as_slice(),
            [AiDelivery {
                completion_id: "conversation-1".to_owned(),
                delivered: true,
                sent_message_id: Some(MessageId(700)),
            }]
        );
    }

    #[test]
    fn ai_routing_records_ignored_groups_and_honors_non_ai_followup_config() {
        let (source, (prepared, ignored, deliveries)) = ai_source(Ok(AiPreparation::silent()));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig {
                    ai_command_followups: false,
                    ..ChatConfig::default()
                }),
                chat_ids: Vec::new(),
            },
            Actions::default(),
            State::default(),
            values(),
            Samples {
                choice_index: 9_999,
                integer: BigInt::from(2_u8),
            },
            authorization(),
            "@mybot",
        )
        .with_ai_conversation_source(Box::new(AiSource {
            metadata: Some(AiReplyMetadata {
                kind: "command".to_owned(),
                uses_ai: false,
            }),
            ..source
        }));
        let mut ordinary = update("ordinary group message", None);
        let IncomingEvent::Message(message) = &mut ordinary.event else {
            return;
        };
        message.chat_type = Some("group".to_owned());
        assert_eq!(dispatcher.dispatch(ordinary), Ok(DispatchOutcome::Handled));

        let mut followup = update("and why?", None);
        let IncomingEvent::Message(message) = &mut followup.event else {
            return;
        };
        message.chat_type = Some("group".to_owned());
        message.has_reply = true;
        message.replied_message_id = Some(MessageId(3));
        message.replied_sender_first_name = Some("Gordo".to_owned());
        message.replied_sender_username = Some("mybot".to_owned());
        message.replied_text = Some("command answer".to_owned());
        assert_eq!(dispatcher.dispatch(followup), Ok(DispatchOutcome::Handled));
        assert!(prepared.borrow().is_empty());
        assert_eq!(ignored.borrow().len(), 2);
        assert_eq!(
            ignored.borrow()[1].reply_context.as_deref(),
            Some("Gordo (mybot): command answer")
        );
        assert!(deliveries.borrow().is_empty());
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
    fn dispatches_ascii_command_conversion_and_keeps_preprocessing_on_legacy() {
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
            dispatcher.dispatch(update("/command hello! world", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(update("/comando", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(update("/command もうすぐです", Some("es"))),
            Ok(DispatchOutcome::LegacyRequired)
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
            vec![
                "/HELLO_SIGNODEEXCLAMACION_WORLD",
                "send the text you want to convert"
            ]
        );
        assert_eq!(dispatcher.state.incoming.len(), 2);
        assert_eq!(dispatcher.state.outgoing.len(), 2);
    }

    #[test]
    fn bcra_commands_localize_load_failures_diagnostics_state_and_legacy_boundary() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        )
        .with_bcra_source(Box::new(BcraVariables {
            result: BcraLoad {
                text: Some("synthetic BCRA variables".to_owned()),
                diagnostics: vec!["synthetic stale source".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/variables", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls.borrow().as_slice(),
            &[(bot_core::locale::Locale::En, 1_672_531_200)]
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "synthetic BCRA variables");
        assert_eq!(dispatcher.state_diagnostics(), &["synthetic stale source"]);
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_bcra_source(Box::new(BcraVariables {
            result: BcraLoad {
                text: None,
                diagnostics: Vec::new(),
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(update("/bcra", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert!(message.text.contains("No pude obtener las variables"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/bcra", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn dollar_commands_pass_timeframe_locale_and_record_diagnostics_and_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        )
        .with_dollar_market_source(Box::new(DollarMarket {
            result: DollarMarketLoad {
                text: Some("synthetic dollar rates".to_owned()),
                diagnostics: vec!["synthetic stale cache".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/usd 6h", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls.borrow().as_slice(),
            &[(6, bot_core::locale::Locale::En, 1_672_531_200)]
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "synthetic dollar rates");
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(dispatcher.state_diagnostics(), &["synthetic stale cache"]);
    }

    #[test]
    fn dollar_invalid_timeframe_and_failure_are_localized_without_losing_legacy_fallback() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut invalid = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            invalid.dispatch(update("/dolar 7d", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = invalid.actions.0.first() else {
            return;
        };
        assert!(message.text.contains("7d"));
        assert!(message.text.contains("no soportado"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_dollar_market_source(Box::new(DollarMarket {
            result: DollarMarketLoad {
                text: None,
                diagnostics: vec!["synthetic provider failure".to_owned()],
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(update("/dollar", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "no pude traer cotizaciones del dólar boludo");

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/usd", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn weather_commands_use_default_or_requested_location_and_record_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        )
        .with_weather_source(Box::new(WeatherObservations {
            result: WeatherObservationLoad {
                observation: Some(WeatherObservation {
                    location: "Example City, Exampleland".to_owned(),
                    apparent_temperature: "19.5".to_owned(),
                    precipitation_probability: "20".to_owned(),
                    weather_code: 1,
                    cloud_cover: "30".to_owned(),
                    visibility_meters: 15_000.0,
                }),
                diagnostics: vec!["synthetic cache diagnostic".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/weather Example City, Exampleland", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls.borrow().as_slice(),
            &[("Example City, Exampleland".to_owned(), 1_672_531_200)]
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert!(message.text.contains("Location: Example City, Exampleland"));
        assert!(message.text.contains("Condition: mostly clear"));
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(
            dispatcher.state_diagnostics(),
            &["synthetic cache diagnostic"]
        );

        let default_calls = Rc::new(RefCell::new(Vec::new()));
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut default = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_weather_source(Box::new(WeatherObservations {
            result: WeatherObservationLoad {
                observation: None,
                diagnostics: Vec::new(),
            },
            calls: Rc::clone(&default_calls),
        }));
        assert_eq!(
            default.dispatch(update("/clima", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(default_calls.borrow()[0].0, "Buenos Aires");
        let Some(TelegramAction::SendMessage(message)) = default.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "no se pudo obtener el clima de Buenos Aires");
    }

    #[test]
    fn election_commands_render_html_live_prices_diagnostics_and_command_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let events = parse_election_events(&json!([{
            "title":"US election","slug":"us-election","liquidity":2500000,"tags":[{"slug":"united-states"}],"markets":[
                {"groupItemTitle":"Candidate A","outcomes":["Yes","No"],"outcomePrices":[0.4,0.6],"clobTokenIds":["a","a-no"]}
            ]
        }]));
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
        )
        .with_election_source(Box::new(Elections {
            result: ElectionLoad {
                events,
                live_prices: HashMap::from([("a".to_owned(), 0.72)]),
                diagnostics: vec!["synthetic midpoint fallback".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/elections", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), &[1_672_531_200]);
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert!(message.text.contains("Global elections by liquidity"));
        assert!(message.text.contains("Candidate A 72%"));
        assert_eq!(
            message.parse_mode,
            Some(bot_core::telegram_actions::ParseMode::Html)
        );
        assert!(message.disable_web_page_preview);
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(
            dispatcher.state_diagnostics(),
            &["synthetic midpoint fallback"]
        );
    }

    #[test]
    fn election_failure_is_localized_and_missing_source_stays_legacy() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_election_source(Box::new(Elections {
            result: ElectionLoad {
                events: Vec::new(),
                live_prices: HashMap::new(),
                diagnostics: vec!["synthetic Gamma failure".to_owned()],
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(update("/eleccion", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "No pude traer las elecciones desde Polymarket"
        );

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/election", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(missing.actions.0.is_empty());
    }

    #[test]
    fn stock_commands_pass_query_render_quotes_and_record_command_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let quote = StockQuote {
            symbol: "AAPL".to_owned(),
            name: "Apple".to_owned(),
            price: 205.5,
            currency: "USD".to_owned(),
            exchange: "NMS".to_owned(),
            variation: 1.25,
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
            authorization(),
            "@mybot",
        )
        .with_stock_price_source(Box::new(StockQuotes {
            result: StockQuotesLoad {
                quotes: Some(vec![
                    ("Apple Inc".to_owned(), Some(quote)),
                    ("Unknown".to_owned(), None),
                ]),
                diagnostics: vec!["synthetic stale search".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/stocks Apple Inc", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls.borrow().as_slice(),
            &[("Apple Inc".to_owned(), 1_672_531_200)]
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "AAPL: 205.50 USD (+1.25% 24h)\nUnknown: not found"
        );
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(dispatcher.state_diagnostics(), &["synthetic stale search"]);
    }

    #[test]
    fn market_price_aliases_use_native_source_locale_diagnostics_and_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        )
        .with_market_price_source(Box::new(MarketPrices {
            result: MarketPriceLoad {
                text: "BTC: 50000 USD (+2.5% 24h)".to_owned(),
                diagnostics: vec!["synthetic stale CMC cache".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/bresios btc", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls.borrow().as_slice(),
            &[(
                "btc".to_owned(),
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            )]
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "BTC: 50000 USD (+2.5% 24h)");
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(
            dispatcher.state_diagnostics(),
            &["synthetic stale CMC cache"]
        );

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/crypto btc", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn stock_top_failure_is_localized_and_missing_source_stays_legacy() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_stock_price_source(Box::new(StockQuotes {
            result: StockQuotesLoad {
                quotes: None,
                diagnostics: vec!["synthetic Finviz failure".to_owned()],
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(update("/acciones", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "no pude traer el top de acciones, probá de nuevo"
        );
        assert!(failed.state_diagnostics()[0].contains("Finviz failure"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/stocks AAPL", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(missing.actions.0.is_empty());
    }

    #[test]
    fn oil_commands_render_partial_quotes_diagnostics_and_command_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let quote = |symbol: &str, price: f64, variation: f64| StockQuote {
            symbol: symbol.to_owned(),
            name: String::new(),
            price,
            currency: "USD".to_owned(),
            exchange: String::new(),
            variation,
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
            authorization(),
            "@mybot",
        )
        .with_oil_price_source(Box::new(OilQuotes {
            result: OilQuoteLoad {
                brent: Some(quote("BZ=F", 98.15, -8.78)),
                wti: Some(quote("CL=F", 95.45, 1.25)),
                diagnostics: vec!["synthetic stale quote".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/oil", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), &[1_672_531_200]);
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "Brent: 98.15 USD (-8.78% 24hs)\nWTI: 95.45 USD (+1.25% 24hs)"
        );
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert_eq!(dispatcher.state_diagnostics(), &["synthetic stale quote"]);
    }

    #[test]
    fn oil_failure_is_localized_and_missing_source_stays_legacy() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_oil_price_source(Box::new(OilQuotes {
            result: OilQuoteLoad {
                brent: None,
                wti: None,
                diagnostics: vec!["synthetic Yahoo failure".to_owned()],
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(update("/petroleo", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "no pude traer el precio del petróleo boludo");
        assert!(failed.state_diagnostics()[0].contains("Yahoo failure"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/oil", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(missing.actions.0.is_empty());
    }

    #[test]
    fn weather_without_native_source_stays_on_legacy() {
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
            dispatcher.dispatch(update("/weather Rosario", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(dispatcher.actions.0.is_empty());
        assert!(dispatcher.state.incoming.is_empty());
    }

    #[test]
    fn greeting_commands_choose_animation_and_fall_back_to_localized_text() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        )
        .with_greeting_pool_source(Box::new(GreetingPools {
            result: GreetingPoolLoad {
                urls: vec![
                    "https://example.test/first.gif".to_owned(),
                    "https://example.test/second.gif".to_owned(),
                ],
                diagnostics: vec!["synthetic stale pool".to_owned()],
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/gm", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), &[GreetingCategory::Morning]);
        assert_eq!(
            dispatcher.actions.0,
            vec![TelegramAction::SendAnimation {
                chat_id: ChatId(-42),
                animation: "https://example.test/second.gif".to_owned(),
                reply_to_message_id: Some(MessageId(7)),
                caption: None,
            }]
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert!(dispatcher.state.outgoing.is_empty());
        assert_eq!(dispatcher.state_diagnostics(), &["synthetic stale pool"]);

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut fallback = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_greeting_pool_source(Box::new(GreetingPools {
            result: GreetingPoolLoad {
                urls: Vec::new(),
                diagnostics: Vec::new(),
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            fallback.dispatch(update("/gn", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = fallback.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "buenas noches boludo");

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut non_http = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            Samples {
                choice_index: 0,
                integer: BigInt::from(0_u8),
            },
            authorization(),
            "@mybot",
        )
        .with_greeting_pool_source(Box::new(GreetingPools {
            result: GreetingPoolLoad {
                urls: vec!["cached greeting".to_owned()],
                diagnostics: Vec::new(),
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            non_http.dispatch(update("/gm", None)),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = non_http.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "cached greeting");
        assert_eq!(non_http.state.outgoing.len(), 1);
    }

    #[test]
    fn greeting_animation_delivery_failure_is_silent_and_missing_source_stays_legacy() {
        struct DroppingAnimations;

        impl ActionSink for DroppingAnimations {
            type Error = Infallible;

            fn execute(&mut self, _action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
                Ok(ActionReceipt { message_id: None })
            }

            fn try_animation(&mut self, _action: TelegramAction) -> Result<bool, Self::Error> {
                Ok(false)
            }
        }

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            DroppingAnimations,
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_greeting_pool_source(Box::new(GreetingPools {
            result: GreetingPoolLoad {
                urls: vec![
                    "https://example.test/first.gif".to_owned(),
                    "https://example.test/greeting.gif".to_owned(),
                ],
                diagnostics: Vec::new(),
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/gm", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert!(dispatcher.state.outgoing.is_empty());

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/gn", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn rulo_renders_all_routes_and_records_nonfatal_exchange_diagnostics() {
        let calls = Rc::new(RefCell::new(0));
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
        .with_rulo_source(Box::new(RuloInputs {
            result: Ok(RuloInputLoad {
                input: RuloInput {
                    official: Some(1440.0),
                    mep: Some(1459.73),
                    blue: Some(1430.0),
                    usd_to_usdt: vec![ExchangeQuote {
                        exchange: "buenbit".to_owned(),
                        price: Some(1.031),
                    }],
                    usdt_to_ars: vec![ExchangeQuote {
                        exchange: "buenbit".to_owned(),
                        price: Some(1458.44),
                    }],
                    usd_amount: 1000.0,
                },
                diagnostics: vec!["synthetic stale USD book".to_owned()],
            }),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/rulo", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(*calls.borrow(), 1);
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert!(
            message
                .text
                .starts_with("Rulos desde Oficial (precio oficial: 1.440 ARS/USD)")
        );
        assert!(message.text.contains("  • Ganancia: +19.730 ARS"));
        assert!(
            message
                .text
                .contains("  • Tramos: USD→USDT BUENBIT, USDT→ARS BUENBIT")
        );
        assert_eq!(
            dispatcher.state_diagnostics(),
            &["synthetic stale USD book".to_owned()]
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn rulo_primary_failure_and_missing_source_are_safe() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_rulo_source(Box::new(RuloInputs {
            result: Err("synthetic primary failure".to_owned()),
            calls: Rc::new(RefCell::new(0)),
        }));
        assert_eq!(
            failed.dispatch(update("/rulo ignored", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "I could not load dollar rates");
        assert!(failed.state_diagnostics()[0].contains("synthetic primary failure"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/rulo", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn devo_loads_quotes_renders_projection_and_records_command_state() {
        let calls = Rc::new(RefCell::new(0));
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
        )
        .with_dollar_quotes_source(Box::new(DollarQuotes {
            result: Ok(Some(DevoQuotes {
                official: 100.0,
                card: 150.0,
                usdt_ask: 200.0,
                usdt_bid: 190.0,
            })),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/devo 0.5, 100", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(*calls.borrow(), 1);
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "100 USD card = 15000 ARS = 76.92 USDT\nProfit: 9402.5 ARS / 48.22 USDT\nTotal: 24402.5 ARS / 125.14 USDT\n\nprofit: 62.68%\n\nfee: 0.5%\nofficial: 100\nusdt: 195\ncard: 150"
        );
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn devo_preserves_guards_safe_failures_and_legacy_boundary() {
        let calls = Rc::new(RefCell::new(0));
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
        .with_dollar_quotes_source(Box::new(DollarQuotes {
            result: Err("synthetic upstream failure".to_owned()),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/devo nan", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(*calls.borrow(), 0);
        assert_eq!(
            dispatcher.dispatch(update("/devo 0.5", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(*calls.borrow(), 1);
        assert!(dispatcher.state_diagnostics()[0].contains("synthetic upstream failure"));
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
            vec![
                "mandá bien los datos: fee entre 0 y 100 y monto de compra positivo",
                "no pude traer cotizaciones del dólar boludo"
            ]
        );
        assert_eq!(
            dispatcher.dispatch(update("/devo ０.５", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/devo 0.5", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn dispatches_bitcoin_quote_and_reference_model_commands() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        )
        .with_bitcoin_price_source(Box::new(BitcoinPrices {
            results: vec![
                Ok(Some(50_000.0)),
                Ok(Some(10_000_000.0)),
                Ok(Some(50_000.0)),
                Ok(Some(50_000.0)),
            ],
            calls: Rc::clone(&calls),
        }));
        for command in ["/sats", "/powerlaw", "/rainbow"] {
            assert_eq!(
                dispatcher.dispatch(update(command, Some("es"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        assert_eq!(calls.borrow().as_slice(), &["USD", "ARS", "USD", "USD"]);
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
            texts[0],
            "1 satoshi = $0.00050000 USD\n1 satoshi = $0.1000 ARS\n\n$1 USD = 2,000 sats\n$1 ARS = 10.000 sats"
        );
        assert!(texts[1].starts_with("power law estimates BTC at "));
        assert!(texts[2].starts_with("rainbow chart estimates BTC at "));
        assert_eq!(dispatcher.state.incoming.len(), 3);
        assert_eq!(dispatcher.state.outgoing.len(), 3);
    }

    #[test]
    fn bitcoin_price_failures_are_localized_diagnostic_and_legacy_safe() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        .with_bitcoin_price_source(Box::new(BitcoinPrices {
            results: vec![
                Err("synthetic USD failure".to_owned()),
                Ok(Some(50_000.0)),
                Ok(None),
            ],
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/powerlaw", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.state_diagnostics()[0].contains("synthetic USD failure"));
        assert_eq!(
            dispatcher.dispatch(update("/satoshi", None)),
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
            vec![
                "no pude traer el precio de BTC para calcular power law",
                "no pude traer el precio de BTC en ARS"
            ]
        );
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            missing.dispatch(update("/sat", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn printcredits_authorizes_parses_mints_and_records_command_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        .with_admin_user_id(Some(88))
        .with_admin_credit_sink(Box::new(AdminCredits {
            result: Ok(12_000),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/printcredits 100.0", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), &[(88, 10_000)]);
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "minted 100.00 credits\nyour balance is 120.00"
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn printcredits_guards_and_failures_are_safe_without_duplicate_mints() {
        let denied_calls = Rc::new(RefCell::new(Vec::new()));
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut denied = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_admin_user_id(Some(99))
        .with_admin_credit_sink(Box::new(AdminCredits {
            result: Ok(0),
            calls: Rc::clone(&denied_calls),
        }));
        assert_eq!(
            denied.dispatch(update("/printcredits 100", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(denied_calls.borrow().is_empty());
        let Some(TelegramAction::SendMessage(message)) = denied.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "this command is only for the admin");

        let failed_calls = Rc::new(RefCell::new(Vec::new()));
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_admin_user_id(Some(88))
        .with_admin_credit_sink(Box::new(AdminCredits {
            result: Err("synthetic database failure".to_owned()),
            calls: Rc::clone(&failed_calls),
        }));
        assert_eq!(
            failed.dispatch(update("/printcredits 1", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(failed_calls.borrow().as_slice(), &[(88, 100)]);
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "se trabó imprimiendo créditos, probá de nuevo"
        );
        assert!(failed.state_diagnostics()[0].contains("synthetic database failure"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_admin_user_id(Some(88));
        assert_eq!(
            missing.dispatch(update("/printcredits 1", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
    }

    #[test]
    fn creditlog_loads_formats_and_records_admin_command_state() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        )
        .with_admin_user_id(Some(88))
        .with_admin_creditlog_source(Box::new(AdminCreditLogs {
            result: Ok(vec![bot_core::admin_commands::CreditLogEntry {
                user_id: Some(88),
                chat_id: Some(-42),
                metadata: json!({
                    "command":"/ask",
                    "reserved_credit_units_total":200,
                    "settled_credit_units":100,
                    "refunded_credit_units":100,
                    "billing_segments":[{"kind":"chat"}],
                    "model_breakdown":[{"model":"m1","usd_micros":5}],
                    "tool_breakdown":[{"tool":"web","usd_micros":7,"count":2}]
                }),
                created_at: "2026-03-11T17:35:10+00:00".to_owned(),
            }]),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/creditlog 2", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), &[2]);
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert!(message.text.starts_with("latest AI settlements:"));
        assert!(
            message
                .text
                .contains("reserved=2.00 charged=1.00 refund=1.00")
        );
        assert!(message.text.contains("requests: chat=1"));
        assert!(message.text.contains("models: m1=5"));
        assert!(message.text.contains("tools: web=7 (2x)"));
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn creditlog_handles_empty_failure_legacy_and_missing_source_paths() {
        for (result, expected) in [
            (
                Ok(Vec::new()),
                "no hay liquidaciones IA recientes".to_owned(),
            ),
            (
                Err("synthetic read failure".to_owned()),
                "se trabó leyendo el creditlog, probá de nuevo".to_owned(),
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
            .with_admin_user_id(Some(88))
            .with_admin_creditlog_source(Box::new(AdminCreditLogs {
                result,
                calls: Rc::new(RefCell::new(Vec::new())),
            }));
            assert_eq!(
                dispatcher.dispatch(update("/creditlog", None)),
                Ok(DispatchOutcome::Handled)
            );
            let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
                return;
            };
            assert_eq!(message.text, expected);
        }

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut missing = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_admin_user_id(Some(88));
        assert_eq!(
            missing.dispatch(update("/creditlog", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert_eq!(
            missing.dispatch(update("/creditlog ２", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
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
    fn task_list_aliases_render_native_keyboard_but_creation_stays_legacy()
    -> Result<(), TaskStateError> {
        let cancellations = Rc::new(RefCell::new(Vec::new()));
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
        .with_scheduled_task_source(Box::new(Tasks {
            lists: vec![vec![scheduled_task(88)?]],
            cancellations,
        }));

        assert_eq!(
            dispatcher.dispatch(update("/tasks", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.first(),
            Some(TelegramAction::SendMessage(_))
        ));
        if let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() {
            assert!(message.text.starts_with("• [task0001] synthetic reminder"));
            assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
            let callback = message
                .reply_markup
                .as_ref()
                .and_then(|keyboard| keyboard.inline_keyboard.first())
                .and_then(|row| row.first())
                .and_then(|button| button.callback_data.as_deref());
            assert_eq!(callback, Some("task:del:task0001"));
        }

        let before = dispatcher.actions.0.len();
        assert_eq!(
            dispatcher.dispatch(update("/tarea create something", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert_eq!(dispatcher.actions.0.len(), before);
        Ok(())
    }

    #[test]
    fn task_owner_can_delete_in_group_and_message_is_refreshed() -> Result<(), TaskStateError> {
        let cancellations = Rc::new(RefCell::new(Vec::new()));
        let mut denied_admin = authorization();
        denied_admin.is_admin = false;
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
            denied_admin,
            "@mybot",
        )
        .with_scheduled_task_source(Box::new(Tasks {
            lists: vec![vec![scheduled_task(88)?], Vec::new()],
            cancellations: Rc::clone(&cancellations),
        }));

        assert_eq!(
            dispatcher.dispatch(callback_update("task:del:task0001", "group", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            cancellations.borrow().as_slice(),
            &[("task0001".to_owned(), "-42".to_owned())]
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::AnswerCallback {
                    text: Some(text),
                    show_alert: false,
                    ..
                },
                TelegramAction::EditMessage { text: edit_text, .. }
            ] if text == "tarea task0001 borrada" && edit_text == "no hay tareas"
        ));
        Ok(())
    }

    #[test]
    fn unrelated_group_member_cannot_delete_a_task() -> Result<(), TaskStateError> {
        let cancellations = Rc::new(RefCell::new(Vec::new()));
        let mut denied_admin = authorization();
        denied_admin.is_admin = false;
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
            denied_admin,
            "@mybot",
        )
        .with_scheduled_task_source(Box::new(Tasks {
            lists: vec![vec![scheduled_task(42)?]],
            cancellations: Rc::clone(&cancellations),
        }));

        assert_eq!(
            dispatcher.dispatch(callback_update("task:del:task0001", "group", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(cancellations.borrow().is_empty());
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }] if text == "solo el creador o un admin pueden borrar esta tarea"
        ));
        Ok(())
    }

    #[test]
    fn charge_history_callback_loads_edits_and_acknowledges_owned_pages() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Ok(ChargeHistoryPage {
                groups: vec![ChargeHistoryGroup {
                    cursor_id: 20,
                    created_at: "2026-08-26T17:00:00+00:00".to_owned(),
                    entries: vec![ChargeHistoryEntry {
                        id: 20,
                        event_type: "ai_settlement_result".to_owned(),
                        metadata: json!({"charged_credit_units_total":4}),
                    }],
                }],
                has_newer: true,
                has_older: false,
                newer_cursor: Some(20),
                older_cursor: Some(20),
            }),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(callback_update("chg:88:2:o:29:-180", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls.borrow().as_slice(),
            [(88, 2, Some(29), "older".to_owned())]
        );
        let Some(TelegramAction::EditMessage {
            text, reply_markup, ..
        }) = dispatcher.actions.0.first()
        else {
            return;
        };
        assert_eq!(text, "Gastos IA\n\n26/08 14:00 · respuesta · 0.04 cr");
        assert_eq!(
            reply_markup
                .as_ref()
                .and_then(|keyboard| keyboard.inline_keyboard[0][0].callback_data.as_deref()),
            Some("chg:88:2:n:20:-180")
        );
        assert!(matches!(
            dispatcher.actions.0.get(1),
            Some(TelegramAction::AnswerCallback {
                text: None,
                show_alert: false,
                ..
            })
        ));
    }

    #[test]
    fn charge_history_callback_handles_guards_empty_pages_and_load_failures() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut guards = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            guards.dispatch(callback_update("chg:bad", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            guards.dispatch(callback_update("chg:55:2:o:29:-180", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        for (action, expected) in guards
            .actions
            .0
            .iter()
            .zip(["this button expired", "this history is not yours"])
        {
            let TelegramAction::AnswerCallback {
                text, show_alert, ..
            } = action
            else {
                return;
            };
            assert_eq!(text.as_deref(), Some(expected));
            assert!(*show_alert);
        }

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut empty = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Ok(ChargeHistoryPage {
                groups: Vec::new(),
                has_newer: false,
                has_older: false,
                newer_cursor: None,
                older_cursor: None,
            }),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            empty.dispatch(callback_update("chg:88:2:n:30:-180", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            empty.actions.0.first(),
            Some(TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: false,
                ..
            }) if text == "no hay más gastos"
        ));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Err("synthetic callback read failure".to_owned()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(callback_update("chg:88:2:o:29:-180", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            failed.actions.0.first(),
            Some(TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }) if text == "se trabó leyendo tus gastos"
        ));
        assert!(failed.state_diagnostics()[0].contains("synthetic callback read failure"));
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
    fn balance_command_loads_private_and_group_accounts_with_diagnostics() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut private = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_balance_source(Box::new(Balances {
            result: Ok(BillingBalances {
                user_balance: 4_200,
                chat_balance: None,
                diagnostics: vec!["synthetic onboarding diagnostic".to_owned()],
            }),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            private.dispatch(update("/balance", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), [(88, None)]);
        let Some(TelegramAction::SendMessage(message)) = private.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "tenés 42.00 créditos ia\nsi querés cargar más mandale /topup"
        );
        assert_eq!(
            private.state_diagnostics(),
            ["synthetic onboarding diagnostic"]
        );

        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut group = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_balance_source(Box::new(Balances {
            result: Ok(BillingBalances {
                user_balance: 3_000,
                chat_balance: Some(12_000),
                diagnostics: Vec::new(),
            }),
            calls: Rc::clone(&calls),
        }));
        let mut group_update = update("/balance", Some("es"));
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("group".to_owned());
        }
        assert_eq!(group.dispatch(group_update), Ok(DispatchOutcome::Handled));
        assert_eq!(calls.borrow().as_slice(), [(88, None), (88, Some(-42))]);
        let Some(TelegramAction::SendMessage(message)) = group.actions.0.first() else {
            return;
        };
        assert!(
            message
                .text
                .starts_with("AI balances:\n- yours: 30.00\n- group: 120.00")
        );
    }

    #[test]
    fn balance_load_failure_is_localized_and_missing_native_source_stays_legacy_owned() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_balance_source(Box::new(Balances {
            result: Err("synthetic database failure".to_owned()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(update("/balance", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "I could not load your balance, try again");
        assert!(failed.state_diagnostics()[0].contains("synthetic database failure"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut shadow = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            shadow.dispatch(update("/balance", Some("es"))),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(shadow.actions.0.is_empty());
    }

    #[test]
    fn charges_command_loads_formats_and_paginates_the_calling_users_history() {
        let calls = Rc::new(RefCell::new(Vec::new()));
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
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Ok(ChargeHistoryPage {
                groups: vec![ChargeHistoryGroup {
                    cursor_id: 30,
                    created_at: "2026-08-26T17:32:00+00:00".to_owned(),
                    entries: vec![ChargeHistoryEntry {
                        id: 30,
                        event_type: "ai_settlement_result".to_owned(),
                        metadata: json!({
                            "charged_credit_units_total":8,
                            "model_breakdown":[{"kind":"chat","usd_micros":30}],
                            "tool_breakdown":[{"tool":"web_search","count":1,"usd_micros":50}]
                        }),
                    }],
                }],
                has_newer: false,
                has_older: true,
                newer_cursor: Some(30),
                older_cursor: Some(30),
            }),
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/charges 2", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls.borrow().as_slice(),
            [(88, 2, None, "older".to_owned())]
        );
        let Some(TelegramAction::SendMessage(message)) = dispatcher.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "Gastos IA\n\n26/08 14:32 · 0.08 cr\n  respuesta 0.03 cr\n  web 0.05 cr"
        );
        let Some(keyboard) = message.reply_markup.as_ref() else {
            return;
        };
        assert_eq!(
            keyboard.inline_keyboard[0][0].callback_data.as_deref(),
            Some("chg:88:2:o:30:-180")
        );
    }

    #[test]
    fn charges_command_handles_empty_invalid_failed_and_shadow_paths() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut empty = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Ok(ChargeHistoryPage {
                groups: Vec::new(),
                has_newer: false,
                has_older: false,
                newer_cursor: None,
                older_cursor: None,
            }),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            empty.dispatch(update("/history", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = empty.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "you have no recent AI expenses");

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut invalid = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            invalid.dispatch(update("/gastos 0", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = invalid.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "mandalo bien: /charges [cantidad]");

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Err("synthetic history failure".to_owned()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failed.dispatch(update("/charges", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "se trabó leyendo tus gastos, probá de nuevo");
        assert!(failed.state_diagnostics()[0].contains("synthetic history failure"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut shadow = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            shadow.dispatch(update("/charges", Some("es"))),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(shadow.actions.0.is_empty());
    }

    #[test]
    fn transfer_command_moves_fractional_credits_and_reports_insufficient_balance() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut success = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_transfer_sink(Box::new(Transfers {
            result: Ok(TransferResult {
                transferred: true,
                user_balance: 285,
                chat_balance: 1_215,
            }),
            calls: Rc::clone(&calls),
        }));
        let mut group_update = update("/transfer 0.1", Some("es"));
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("group".to_owned());
        }
        assert_eq!(success.dispatch(group_update), Ok(DispatchOutcome::Handled));
        assert_eq!(calls.borrow().as_slice(), [(88, -42, 10)]);
        let Some(TelegramAction::SendMessage(message)) = success.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "listo, le pasé 0.10 créditos al grupo\n- lo tuyo: 2.85\n- lo del grupo: 12.15"
        );

        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let mut insufficient = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_transfer_sink(Box::new(Transfers {
            result: Ok(TransferResult {
                transferred: false,
                user_balance: 70,
                chat_balance: 0,
            }),
            calls,
        }));
        let mut group_update = update("/transfer 1.5", Some("es"));
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("supergroup".to_owned());
        }
        assert_eq!(
            insufficient.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = insufficient.actions.0.first() else {
            return;
        };
        assert_eq!(
            message.text,
            "you do not have enough personal credits\nyou have: 0.70"
        );
    }

    #[test]
    fn transfer_guards_are_native_and_transaction_failures_are_safe() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut private = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            private.dispatch(update("/transfer 1", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let Some(TelegramAction::SendMessage(message)) = private.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "esto es para grupos, capo: /transfer <monto>");

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut failed = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_transfer_sink(Box::new(Transfers {
            result: Err("synthetic uncertain transaction".to_owned()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        let mut group_update = update("/transfer 1", Some("es"));
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("group".to_owned());
        }
        assert_eq!(failed.dispatch(group_update), Ok(DispatchOutcome::Handled));
        let Some(TelegramAction::SendMessage(message)) = failed.actions.0.first() else {
            return;
        };
        assert_eq!(message.text, "se trabó la transferencia, probá de nuevo");
        assert!(failed.state_diagnostics()[0].contains("synthetic uncertain transaction"));

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut shadow = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        let mut group_update = update("/transfer 1", Some("es"));
        if let IncomingEvent::Message(message) = &mut group_update.event {
            message.chat_type = Some("group".to_owned());
        }
        assert_eq!(
            shadow.dispatch(group_update),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(shadow.actions.0.is_empty());
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
    fn dispatches_fixed_links_with_reply_buttons_context_and_state() {
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
        .with_link_replacement_source(Box::new(links(true)));
        assert_eq!(
            dispatcher.dispatch(update("https://x.com/a/status/1", None)),
            Ok(DispatchOutcome::Handled)
        );
        let [TelegramAction::SendMessage(message)] = dispatcher.actions.0.as_slice() else {
            return;
        };
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(
            message.text,
            "https://fixupx.com/a/status/1\n\ncompartido por @tester"
        );
        let Some(markup) = &message.reply_markup else {
            return;
        };
        assert_eq!(
            markup.inline_keyboard[0][0].url.as_deref(),
            Some("https://x.com/a/status/1")
        );
        assert!(dispatcher.state.incoming.is_empty());
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        assert!(
            dispatcher.state.outgoing[0]
                .message
                .text
                .contains("titulo: example")
        );
        assert_eq!(dispatcher.state.outgoing[0].message.message_id, "bot_700");
    }

    #[test]
    fn oversized_instagram_video_uploads_and_falls_back_to_text_on_rejection() {
        let config = || Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut source = links(true);
        source.oversized_video = Some(vec![1, 2, 3]);
        let mut dispatcher = NativeDispatcher::new(
            config(),
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_link_replacement_source(Box::new(source));
        assert_eq!(
            dispatcher.dispatch(update("https://instagram.com/reel/a", None)),
            Ok(DispatchOutcome::Handled)
        );
        let [
            TelegramAction::SendVideo {
                video,
                caption,
                reply_to_message_id,
                ..
            },
        ] = dispatcher.actions.0.as_slice()
        else {
            return;
        };
        assert_eq!(video, &[1, 2, 3]);
        assert!(caption.contains("compartido por @tester"));
        assert_eq!(*reply_to_message_id, Some(MessageId(7)));

        #[derive(Default)]
        struct RejectVideo(Vec<TelegramAction>);
        impl ActionSink for RejectVideo {
            type Error = Infallible;

            fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
                self.0.push(action);
                Ok(ActionReceipt {
                    message_id: Some(MessageId(701)),
                })
            }

            fn try_video(
                &mut self,
                _action: TelegramAction,
            ) -> Result<Option<ActionReceipt>, Self::Error> {
                Ok(None)
            }
        }
        let mut source = links(true);
        source.oversized_video = Some(vec![4, 5, 6]);
        let mut fallback = NativeDispatcher::new(
            config(),
            RejectVideo::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_link_replacement_source(Box::new(source));
        assert_eq!(
            fallback.dispatch(update("https://instagram.com/reel/a", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            fallback.actions.0.as_slice(),
            [TelegramAction::SendMessage(_)]
        ));
        assert_eq!(fallback.state.outgoing[0].message.message_id, "bot_701");
    }

    #[test]
    fn delete_mode_preserves_reply_target_and_deletes_only_after_send() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                link_mode: "delete".to_owned(),
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
        )
        .with_link_replacement_source(Box::new(links(true)));
        let mut incoming = update("https://x.com/a/status/1", Some("en"));
        let IncomingEvent::Message(message) = &mut incoming.event else {
            return;
        };
        message.has_reply = true;
        message.replied_message_id = Some(MessageId(3));
        message.sender_username = None;
        message.sender_first_name = Some("Ana".to_owned());
        message.sender_last_name = Some("Test".to_owned());
        assert_eq!(dispatcher.dispatch(incoming), Ok(DispatchOutcome::Handled));
        let [
            TelegramAction::SendMessage(message),
            TelegramAction::DeleteMessage {
                chat_id,
                message_id,
            },
        ] = dispatcher.actions.0.as_slice()
        else {
            return;
        };
        assert_eq!(message.reply_to_message_id, Some(MessageId(3)));
        assert_eq!(
            message.text,
            "https://fixupx.com/a/status/1\n\nshared by Ana Test"
        );
        assert_eq!((*chat_id, *message_id), (ChatId(-42), MessageId(7)));
    }

    #[test]
    fn failed_supported_preview_is_suppressed_and_stored_as_user_context() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut source = links(false);
        source.diagnostics.push("preview unavailable".to_owned());
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_link_replacement_source(Box::new(source));
        assert_eq!(
            dispatcher.dispatch(update("https://x.com/a/status/1", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.actions.0.is_empty());
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert!(dispatcher.state.outgoing.is_empty());
        assert_eq!(dispatcher.state_diagnostics(), ["preview unavailable"]);
    }

    #[test]
    fn link_mode_off_commands_and_non_plain_replies_remain_legacy_owned() {
        let config = Config {
            value: Ok(ChatConfig {
                link_mode: "off".to_owned(),
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
        )
        .with_link_replacement_source(Box::new(links(true)));
        assert_eq!(
            dispatcher.dispatch(update("https://x.com/a/status/1", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        dispatcher.config.value = Ok(ChatConfig::default());
        assert_eq!(
            dispatcher.dispatch(update("/ask https://x.com/a/status/1", None)),
            Ok(DispatchOutcome::LegacyRequired)
        );
        let mut reply = update("mirá https://x.com/a/status/1", None);
        let IncomingEvent::Message(message) = &mut reply.event else {
            return;
        };
        message.has_reply = true;
        message.replied_message_id = Some(MessageId(3));
        assert_eq!(
            dispatcher.dispatch(reply),
            Ok(DispatchOutcome::LegacyRequired)
        );
        assert!(dispatcher.actions.0.is_empty());
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
