//! Update dispatch for commands, callbacks, payments, media, and AI messages.

use bot_adapters::telegram_polling::{IncomingEvent, IncomingMessage, IncomingUpdate};
use bot_core::admin_commands::{
    CreditLogEntry, CreditLogPlan, PrintCreditsContext, PrintCreditsPlan, plan_creditlog_command,
    plan_printcredits_command, printcredits_result_reply, render_creditlog,
};
use bot_core::bcra::classify_bcra_command;
use bot_core::billing_commands::{
    TransferCommandContext, TransferCommandPlan, TransferRecipient, TransferResult,
    plan_transfer_command, transfer_result_reply, user_transfer_result_reply,
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
use bot_core::chat_bans::{
    BanCommand, BanCommandContext, BanCommandPlan, BanTarget, BannedUser, ban_admin_target,
    ban_list_failed, ban_reply, ban_result_reply, ban_store_failed, bans_group_only,
    classify_ban_command, is_shared_ban_command, plan_ban_command, render_ban_list,
    unban_result_reply,
};
use bot_core::chat_config::ChatConfig;
use bot_core::chat_limits::{
    LimitCommand, LimitCommandContext, LimitCommandPlan, LimitedUser, classify_limit_command,
    limit_admin_target, limit_result_reply, plan_limit_command, render_limit_list,
    unlimit_result_reply,
};
use bot_core::command_parsing::parse_command;
use bot_core::command_state::{
    IncomingCommandState, IncomingCommandWritePlan, OutgoingCommandState, OutgoingCommandWritePlan,
    prepare_incoming_command_state, prepare_outgoing_command_state,
};
use bot_core::config_callbacks::{
    ConfigCallbackDiagnostic, ConfigCallbackOutcome, plan_config_callback,
};
use bot_core::config_command::{plan_config_command, render_config_page};
use bot_core::devo::{
    DevoCommandPlan, DevoQuotes, DevoReply, calculate_devo, plan_devo_command, render_devo_reply,
    render_devo_result,
};
use bot_core::dollar::{
    DollarCommandPlan, classify_dollar_command, invalid_timeframe_message, plan_dollar_command,
};
use bot_core::greeting_commands::{GreetingCategory, classify_greeting_command, greeting_fallback};
use bot_core::group_charges::{
    GROUP_CHARGES_LIMIT, GroupChargesPlan, GroupSpender, classify_group_charges_command,
    group_charges_failed, plan_group_charges_command, render_group_charges,
};
use bot_core::language_command::{LanguageCommandPlan, plan_language_command};
use bot_core::lightning_topup::{
    LightningCallback, LightningInvoice, lightning_invoice_failed, lightning_invoice_message,
    lightning_invoice_ready, parse_lightning_callback,
};
use bot_core::links::{
    LinkActionContext, LinkMode, LinkReplacement, has_replaceable_link, plan_link_actions,
};
use bot_core::locale::resolve_locale;
use bot_core::market_prices::{
    MarketCandidate, MarketConversion, MarketPriceCommand, MarketSelection,
    classify_market_price_command, format_market_selection,
};
use bot_core::mention_targets::{
    NamedMember, find_member_by_username, known_member_target, mention_lookup_failed_reply,
    picked_member_target, split_named_member, unknown_mention_reply,
};
use bot_core::polymarket::{ElectionEvent, classify_election_command, render_elections};
use bot_core::random_selection::{RandomSelection, parse_random_selection};
use bot_core::routing::{
    ResponseRoutingEvaluation, ResponseRoutingInput, evaluate_response_routing,
};
use bot_core::rulo::{RuloInput, evaluate_rulo, render_rulo};
use bot_core::scheduled_tasks::{ScheduledTask, TaskId};
use bot_core::stateless_commands::{
    StatelessCommandPlan, StatelessRuntimeContext, plan_runtime_stateless_command,
    plan_stateless_command_with_reply,
};
use bot_core::stocks::{
    StockQuote, classify_oil_command, classify_stock_command, render_oil_quotes,
    render_stock_quotes,
};
use bot_core::task_commands::{
    TaskCallbackParse, can_delete_task, parse_task_callback, render_task_list, task_delete_failed,
    task_delete_forbidden, task_deleted, task_load_failed, task_not_found,
};
use bot_core::telegram_actions::{
    InlineKeyboardMarkup, MAX_TELEGRAM_TEXT_LENGTH, ParseMode, SendMessage, TelegramAction,
};
use bot_core::telegram_callbacks::{
    CallbackContext, CallbackContextOutcome, CallbackRoute, parse_callback_context,
};
use bot_core::telegram_commands::telegram_commands;
use bot_core::telegram_input::{ChatId, MessageId, UserId, is_group_chat_type};
use bot_core::telegram_payments::{
    BalanceCommandContext, BalanceCommandPlan, BillingPackTerms, DEFAULT_TOPUP_CREDITS,
    MAX_TOPUP_CREDITS, MIN_TOPUP_CREDITS, StarPaymentRecord, SuccessfulPaymentDecision,
    TopupCallbackPlan, balance_reply, evaluate_default_successful_payment, invoice_payload_locale,
    parse_topup_amount, payment_record, plan_balance_command, plan_pre_checkout,
    plan_topup_callback, plan_topup_command, successful_payment_reply, topup_menu,
    topup_menu_title,
};
use bot_core::token_signals::{
    SIGNAL_REFRESH_COOLDOWN_SECONDS, SignalQuery, SignalState, TokenAddress, TokenSignal,
    TokenSignalCandidates, build_signal_keyboard_localized, callback_text as signal_callback_text,
    detect_signal_query, format_signal_caption_for_period_with_candles, normalize_token_name,
    signal_market_values_with_candles, stable_signal_id,
};
use bot_core::weather::{
    WeatherObservation, classify_weather_command, render_weather, requested_location,
    weather_load_error,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::ai_dispatch::{
    AiConversationInput, AiConversationSource, AiDelivery, AiPreparation, CreditlessLimit,
    reply_context,
};
use crate::runtime::{HandlerErrorDisposition, UpdateHandler};
use crate::telegram_stream::{StreamFinalizeError, TelegramAiStream};

fn text_without_links(text: &str) -> String {
    bot_core::links::HTTP_URL
        .as_ref()
        .map_or(std::borrow::Cow::Borrowed(text), |pattern| {
            pattern.replace_all(text, "")
        })
        .into_owned()
}

fn thinking_text(locale: bot_core::locale::Locale) -> &'static str {
    match locale {
        bot_core::locale::Locale::Es => "Pensando",
        bot_core::locale::Locale::En => "Thinking",
    }
}

pub trait ChatConfigSource {
    type Error: std::fmt::Display;

    fn get(&mut self, chat_id: &str) -> Result<ChatConfig, Self::Error>;

    fn set(&mut self, chat_id: &str, config: &ChatConfig) -> Result<ChatConfig, Self::Error>;

    fn set_changed(
        &mut self,
        chat_id: &str,
        _previous: &ChatConfig,
        config: &ChatConfig,
    ) -> Result<ChatConfig, Self::Error> {
        self.set(chat_id, config)
    }
}

pub trait ActionSink {
    type Error: std::fmt::Display;

    fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error>;

    /// Whether `error` would repeat if the same action were sent again, so
    /// retrying the whole update cannot help.
    fn is_permanent_failure(&self, _error: &Self::Error) -> bool {
        false
    }

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

    fn try_photo(&mut self, action: TelegramAction) -> Result<Option<ActionReceipt>, Self::Error> {
        self.execute(action).map(Some)
    }

    /// Queue an intermediate AI stream edit without making the provider wait
    /// for Telegram. Implementations without a background delivery queue keep
    /// the synchronous behavior as a safe fallback.
    fn enqueue_stream_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        self.try_edit(action)
    }

    /// Deliver the final AI stream edit and wait for its result.
    fn finalize_stream_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        self.try_edit(action)
    }

    /// Drop queued intermediate edits for one AI stream before deleting its
    /// temporary Telegram message.
    fn cancel_stream_edits(&mut self, _chat_id: ChatId, _message_id: MessageId) {}

    /// Start a periodic thinking-status animation for one AI stream.
    fn start_stream_thinking(&mut self, _chat_id: ChatId, _message_id: MessageId, _text: &str) {}

    /// Stop a periodic thinking-status animation for one AI stream.
    fn stop_stream_thinking(&mut self, _chat_id: ChatId, _message_id: MessageId) {}
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
    type Error: std::fmt::Display;

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

/// Credit-moving commands carry an operation id unique to their Telegram
/// message, so a retried update can't apply them twice.
pub trait BillingTransferSink {
    fn transfer(
        &mut self,
        user_id: i64,
        chat_id: i64,
        amount: i64,
        operation_id: &str,
    ) -> Result<TransferResult, String>;

    /// Moves personal credits to another user; `chat_balance` in the result
    /// is the recipient's balance.
    fn transfer_to_user(
        &mut self,
        user_id: i64,
        recipient_id: i64,
        amount: i64,
        operation_id: &str,
    ) -> Result<TransferResult, String>;
}

/// The person a `/transfer` replies to. Replying to this bot (or to nothing)
/// keeps the transfer going to the group.
fn transfer_recipient(message: &IncomingMessage, bot_name: &str) -> Option<TransferRecipient> {
    let bot_username = bot_name.trim().trim_start_matches('@');
    let to_this_bot = !bot_username.is_empty()
        && message
            .replied_sender_username
            .as_deref()
            .is_some_and(|username| username.eq_ignore_ascii_case(bot_username));
    if to_this_bot {
        return None;
    }
    message.replied_sender_id.map(|user_id| TransferRecipient {
        user_id: user_id.0,
        is_bot: message.replied_sender_is_bot,
    })
}

fn recipient_display_name(message: &IncomingMessage, locale: bot_core::locale::Locale) -> String {
    let first_name = message
        .replied_sender_first_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let username = message
        .replied_sender_username
        .as_deref()
        .filter(|name| !name.is_empty());
    match (first_name, username) {
        (Some(name), _) => name.to_owned(),
        (None, Some(username)) => format!("@{username}"),
        (None, None) => match locale {
            bot_core::locale::Locale::Es => "esa persona".to_owned(),
            bot_core::locale::Locale::En => "them".to_owned(),
        },
    }
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
    fn mint(&mut self, user_id: i64, amount: i64, operation_id: &str) -> Result<i64, String>;
}

pub trait AdminCreditLogSource {
    fn load(&mut self, limit: usize) -> Result<Vec<CreditLogEntry>, String>;
}

/// Creates Lightning charges at the payment provider and tracks them until
/// the background poller credits them.
pub trait LightningCheckout {
    fn create(
        &mut self,
        user_id: i64,
        chat_id: i64,
        pack: &BillingPackTerms,
        locale: bot_core::locale::Locale,
    ) -> Result<LightningInvoice, String>;

    /// Remembers the invoice message so the payment notice can reply to it.
    fn attach_message(&mut self, charge_id: &str, message_id: i64) -> Result<(), String>;
}

/// Members a group's admins banned from using the bot in that group.
pub trait ChatBanStore {
    fn is_banned(&mut self, chat_id: i64, user_id: i64) -> Result<bool, String>;

    /// Returns whether the ban is new.
    fn ban(
        &mut self,
        chat_id: i64,
        user_id: i64,
        name: &str,
        banned_by: i64,
    ) -> Result<bool, String>;

    /// Returns whether a ban was lifted.
    fn unban(&mut self, chat_id: i64, user_id: i64) -> Result<bool, String>;

    fn list(&mut self, chat_id: i64) -> Result<Vec<BannedUser>, String>;
}

/// Members a group's admins gave their own hourly limit of AI messages paid
/// by the group, in place of the group's limit.
pub trait ChatLimitStore {
    fn hourly_limit(&mut self, chat_id: i64, user_id: i64) -> Result<Option<i64>, String>;

    /// Setting again replaces the previous limit and name.
    fn set(
        &mut self,
        chat_id: i64,
        user_id: i64,
        name: &str,
        hourly_limit: i64,
        set_by: i64,
    ) -> Result<(), String>;

    /// Returns whether a limit was removed.
    fn clear(&mut self, chat_id: i64, user_id: i64) -> Result<bool, String>;

    fn list(&mut self, chat_id: i64) -> Result<Vec<LimitedUser>, String>;
}

/// What members spent from a group's own balance on AI.
pub trait GroupSpendingSource {
    /// The biggest spenders over the last `days`, at most `limit` of them,
    /// with their names when the bot knows them.
    fn load(&mut self, chat_id: i64, days: i64, limit: usize) -> Result<Vec<GroupSpender>, String>;

    /// How many days back the ledger still has.
    fn max_days(&self) -> i64;
}

/// The member a ban or limit command replies to, bots included so the
/// planner can refuse them.
/// A number typed in private as a reply to the bot's top-up menu.
fn typed_topup_amount(message: &IncomingMessage, text: &str) -> Option<i64> {
    let replies_to_menu = message.chat_type.as_deref() == Some("private")
        && message.replied_sender_is_bot
        && message.replied_text.as_deref().is_some_and(|replied| {
            [bot_core::locale::Locale::Es, bot_core::locale::Locale::En]
                .iter()
                .any(|locale| replied.starts_with(topup_menu_title(*locale)))
        });
    replies_to_menu.then(|| parse_topup_amount(text)).flatten()
}

fn ban_target(message: &IncomingMessage, locale: bot_core::locale::Locale) -> Option<BanTarget> {
    message.replied_sender_id.map(|user_id| BanTarget {
        user_id: user_id.0,
        is_bot: message.replied_sender_is_bot,
        name: recipient_display_name(message, locale),
    })
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

    fn load_with_timeframe(
        &mut self,
        query: &str,
        _timeframe: Option<&str>,
        now_unix: i64,
    ) -> StockQuotesLoad {
        self.load(query, now_unix)
    }

    fn render_chart(
        &mut self,
        _quote: &bot_core::stocks::StockQuote,
        _now_unix: i64,
    ) -> Result<Vec<u8>, String> {
        Err("stock chart unavailable".to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketPriceLoad {
    pub chart: Option<bot_core::market_prices::MarketChart>,
    pub selection: Option<MarketSelection>,
    pub no_assets_found: bool,
    pub text: String,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StoredMarketSelection {
    selection: MarketSelection,
    chat_id: String,
    message_id: i64,
    /// The user's original message. The callback message is deleted after a
    /// selection, so any replacement quote or chart must reply to this ID.
    #[serde(default)]
    source_message_id: Option<i64>,
    requester_id: i64,
    command: String,
}

impl StoredMarketSelection {
    /// Every field is a string, an integer, a bool, or a list or option of
    /// those, which serde_json always serializes.
    fn encode(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

// Zero stores menu state until it is consumed, without a time limit.
const MARKET_SELECTION_TTL_SECONDS: i64 = 0;

// An in-flight invoice marker lives long enough to swallow double taps
// while still allowing an intentional repurchase shortly after.
const TOPUP_INVOICE_CLAIM_TTL_SECONDS: i64 = 120;

/// How long a sent link fix for an addressed message is remembered, so a
/// retried update answers again without posting the fixed link twice.
const SENT_LINK_FIX_TTL_SECONDS: i64 = 600;

fn sent_link_fix_key(chat_id: ChatId, message_id: MessageId) -> String {
    format!("sent_link_fix:{}:{}", chat_id.0, message_id.0)
}

fn topup_invoice_claim_key(user_id: i64, pack_id: &str) -> String {
    format!("topup_invoice:{user_id}:{pack_id}")
}

fn market_selection_key(selection_id: &str) -> String {
    format!("market_selection:{selection_id}")
}

fn market_selection_id(
    chat_id: i64,
    message_id: i64,
    requester_id: i64,
    timestamp: i64,
    selection_index: usize,
) -> String {
    let base = stable_signal_id(chat_id, message_id, requester_id, timestamp);
    if selection_index == 0 {
        base
    } else {
        format!("{base}-{selection_index}")
    }
}

fn market_selection_text(selection: &MarketSelection, locale: bot_core::locale::Locale) -> String {
    format_market_selection(selection, locale)
}

fn period_seconds(period: &str) -> i64 {
    bot_core::price_queries::ChartPeriod::parse(period).map_or(86_400, |range| range.seconds)
}

fn has_period_change(candles: &[Vec<f64>], period: &str) -> bool {
    bot_core::token_signals::has_candle_change(candles, period_seconds(period))
}

fn wider_periods(period: &str) -> &'static [&'static str] {
    match period {
        "1h" | "24h" | "1d" => &["7d"],
        "7d" => &["30d"],
        _ => &[],
    }
}

fn market_token_candidate(
    signal: &TokenSignal,
    timeframe: Option<&str>,
    candles: Option<&[Vec<f64>]>,
) -> MarketCandidate {
    let (price, change) = signal_market_values_with_candles(signal, timeframe, candles);
    let symbol = signal.pair.base_token.symbol.clone();
    let name = if signal.pair.base_token.name.is_empty() {
        symbol.clone()
    } else {
        signal.pair.base_token.name.clone()
    };
    MarketCandidate {
        id: format!(
            "token:{}:{}:{}",
            signal.token.chain_id, signal.token.network, signal.token.address
        ),
        symbol: symbol.clone(),
        name,
        slug: symbol.to_ascii_lowercase(),
        price,
        change,
        currency: String::new(),
        exchange: String::new(),
        asset_type: String::new(),
        contracts: vec![signal.token.clone()],
    }
}

fn market_contracts_match(left: &TokenAddress, right: &TokenAddress) -> bool {
    let left_is_evm = left.address.len() == 42
        && left.address.starts_with("0x")
        && left.address[2..]
            .chars()
            .all(|character| character.is_ascii_hexdigit());
    let right_is_evm = right.address.len() == 42
        && right.address.starts_with("0x")
        && right.address[2..]
            .chars()
            .all(|character| character.is_ascii_hexdigit());
    if left_is_evm && right_is_evm {
        let same_chain = left.chain_id.eq_ignore_ascii_case(&right.chain_id)
            && left.network.eq_ignore_ascii_case(&right.network);
        // A bare EVM address has no chain information. The detector keeps the
        // historical Ethereum-shaped placeholder until DexScreener resolves
        // the actual chain, so compare the contract across EVM networks while
        // that placeholder is present.
        let unqualified_left = left.chain_id.eq_ignore_ascii_case("ethereum")
            && left.network.eq_ignore_ascii_case("eth")
            && left.tag.eq_ignore_ascii_case("ETH");
        let unqualified_right = right.chain_id.eq_ignore_ascii_case("ethereum")
            && right.network.eq_ignore_ascii_case("eth")
            && right.tag.eq_ignore_ascii_case("ETH");
        (same_chain || unqualified_left || unqualified_right)
            && left.address.eq_ignore_ascii_case(&right.address)
    } else {
        if !left.chain_id.eq_ignore_ascii_case(&right.chain_id)
            || !left.network.eq_ignore_ascii_case(&right.network)
        {
            return false;
        }
        // Base58 addresses, including Solana mints, are case-sensitive.
        left.address == right.address
    }
}

fn market_candidates_share_contract(left: &MarketCandidate, right: &MarketCandidate) -> bool {
    left.contracts.iter().any(|left_contract| {
        right
            .contracts
            .iter()
            .any(|right_contract| market_contracts_match(left_contract, right_contract))
    })
}

fn deduplicate_market_candidates(candidates: &mut Vec<MarketCandidate>) {
    let mut distinct = Vec::with_capacity(candidates.len());
    for candidate in std::mem::take(candidates) {
        let Some(existing) = distinct
            .iter_mut()
            .find(|existing| market_candidates_share_contract(existing, &candidate))
        else {
            distinct.push(candidate);
            continue;
        };
        let existing_is_market = !existing.id.starts_with("token:");
        let candidate_is_market = !candidate.id.starts_with("token:");
        if candidate_is_market && !existing_is_market {
            // A provider-resolved candidate carries the identity used by
            // load_candidate; keep it when it overlaps the DEX candidate.
            *existing = candidate;
        }
    }
    *candidates = distinct;
}

fn token_signal_matches_query(signal: &TokenSignal, query: &SignalQuery) -> bool {
    match query {
        SignalQuery::Address(address) => market_contracts_match(address, &signal.token),
        SignalQuery::Symbol(symbol) => {
            let symbol = symbol.trim_start_matches('$');
            signal.pair.base_token.symbol.eq_ignore_ascii_case(symbol)
                || signal.pair.base_token.name.eq_ignore_ascii_case(symbol)
        }
        SignalQuery::Slug(slug) => {
            let slug = normalize_token_name(slug);
            normalize_token_name(&signal.pair.base_token.name) == slug
                || normalize_token_name(&signal.pair.base_token.symbol) == slug
        }
    }
}

fn provider_query_text(text: &str) -> String {
    match detect_signal_query(text) {
        Some(SignalQuery::Slug(slug)) => slug,
        _ => text.to_owned(),
    }
}

fn provider_request_text(request: &str, query: &SignalQuery) -> String {
    match query {
        SignalQuery::Slug(slug) => slug.clone(),
        SignalQuery::Address(token) if !request.eq_ignore_ascii_case(&token.address) => {
            token.address.clone()
        }
        _ => request.to_owned(),
    }
}

fn market_selection_prompt(
    selection: &MarketSelection,
    locale: bot_core::locale::Locale,
) -> String {
    format!(
        "{}: {}",
        bot_core::menu_ui::localized(locale, "Elegí un activo", "Choose an asset"),
        selection.query
    )
}

fn market_selection_page(
    selection_id: &str,
    selection: &MarketSelection,
    locale: bot_core::locale::Locale,
    page: usize,
) -> InlineKeyboardMarkup {
    use bot_core::menu_ui::{button, close};
    let identity = |candidate: &MarketCandidate| {
        let name = if candidate.name.is_empty() {
            &candidate.symbol
        } else {
            &candidate.name
        };
        let detail = if let Some(symbol) = candidate.id.strip_prefix("stock:") {
            if candidate.exchange.trim().is_empty() {
                symbol.to_owned()
            } else {
                format!("{symbol} ({})", candidate.exchange)
            }
        } else {
            format!(
                "{} ({})",
                candidate.symbol,
                candidate
                    .contracts
                    .first()
                    .map_or("Crypto", |contract| contract.chain_id.as_str())
            )
        };
        // Keep the ticker and exchange visible even with long company names.
        let short_name = if name.chars().count() > 24 {
            format!("{}…", name.chars().take(23).collect::<String>())
        } else {
            name.clone()
        };
        if name == &candidate.symbol {
            detail
        } else {
            format!("{short_name}, {detail}")
        }
    };
    let mut rows = selection
        .candidates
        .iter()
        .enumerate()
        .skip(page.saturating_mul(5))
        .take(5)
        .map(|(index, candidate)| {
            let mut label = identity(candidate);
            if selection
                .candidates
                .iter()
                .filter(|other| identity(other) == label)
                .count()
                > 1
            {
                let suffix = candidate.contracts.first().map_or_else(
                    || candidate.id.clone(),
                    |contract| short_market_address(&contract.address),
                );
                label.push_str(&format!(" [{suffix}]"));
            }
            vec![button(label, format!("mkt:select:{selection_id}:{index}"))]
        })
        .collect::<Vec<_>>();
    let pages = selection.candidates.len().div_ceil(5);
    let mut navigation = Vec::new();
    if page > 0 {
        navigation.push(button("‹", format!("mkt:page:{selection_id}:{}", page - 1)));
    }
    if pages > 1 {
        navigation.push(button(
            format!("{} / {pages}", page + 1),
            format!("mkt:page:{selection_id}:{page}"),
        ));
    }
    if page + 1 < pages {
        navigation.push(button("›", format!("mkt:page:{selection_id}:{}", page + 1)));
    }
    if !navigation.is_empty() {
        rows.push(navigation);
    }
    rows.push(vec![close(locale, format!("mkt:close:{selection_id}:0"))]);
    InlineKeyboardMarkup {
        inline_keyboard: rows,
    }
}

fn short_market_address(value: &str) -> String {
    if value.chars().count() <= 14 {
        return value.to_owned();
    }
    let start = value.chars().take(6).collect::<String>();
    let end = value
        .chars()
        .rev()
        .take(4)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{start}…{end}")
}

fn market_selection_command(command: &str) -> MarketPriceCommand {
    if command == "crypto" {
        MarketPriceCommand::CryptoOnly
    } else {
        MarketPriceCommand::Unified
    }
}

/// Chart media and an optional quote caption for the requested period.
pub struct MarketChartRender {
    pub photo: Vec<u8>,
    pub caption: Option<String>,
}

pub trait MarketPriceSource {
    fn load(
        &mut self,
        query: &str,
        command: MarketPriceCommand,
        _locale: bot_core::locale::Locale,
        now_unix: i64,
    ) -> MarketPriceLoad;

    #[allow(clippy::too_many_arguments)]
    fn load_candidate(
        &mut self,
        _candidate: &MarketCandidate,
        _timeframe: Option<&str>,
        _target_symbol: &str,
        _target_parameter: &str,
        _conversion: Option<&MarketConversion>,
        _command: MarketPriceCommand,
        _locale: bot_core::locale::Locale,
        _now_unix: i64,
    ) -> MarketPriceLoad {
        MarketPriceLoad {
            chart: None,
            selection: None,
            no_assets_found: true,
            text: String::new(),
            diagnostics: vec!["market candidate resolution unavailable".to_owned()],
        }
    }
    fn render_chart(
        &mut self,
        _chart: &bot_core::market_prices::MarketChart,
        _now_unix: i64,
    ) -> Result<MarketChartRender, String> {
        Err("market chart unavailable".to_owned())
    }

    fn save_selection(
        &mut self,
        _key: &str,
        _value: &str,
        _ttl_seconds: i64,
    ) -> Result<(), String> {
        Err("market selection storage unavailable".to_owned())
    }

    fn load_selection(&mut self, _key: &str) -> Result<Option<String>, String> {
        Err("market selection storage unavailable".to_owned())
    }

    /// Atomically reads and removes a stored selection. Concurrent takers are
    /// serialized: exactly one of them observes the value, so a double-tapped
    /// inline button can only resolve into a single quote.
    fn take_selection(&mut self, _key: &str) -> Result<Option<String>, String> {
        Err("market selection storage unavailable".to_owned())
    }

    /// Atomically stores a guard key only when absent. Concurrent claimants
    /// are serialized: exactly one of them wins. Used for single-flight
    /// guards such as one in-flight invoice per user and pack.
    fn claim(&mut self, _key: &str, _value: &str, _ttl_seconds: i64) -> Result<bool, String> {
        Err("market selection storage unavailable".to_owned())
    }

    fn clear_selection(&mut self, _key: &str) -> Result<(), String> {
        Ok(())
    }
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

    /// Bounded title/description context for links in a message the AI will
    /// answer, fetched under a short deadline. `None` when nothing is known.
    fn preview_context(&mut self, _text: &str) -> Option<String> {
        None
    }
}

pub trait ScheduledTaskSource {
    fn list(&mut self, chat_id: &str) -> Result<Vec<ScheduledTask>, String>;

    fn cancel(&mut self, task_id: &TaskId, chat_id: &str) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenSignalLoad {
    pub signal: Option<TokenSignal>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenSignalPhotoFailure {
    HistoryUnavailable,
    DeliveryFailed,
    DeliverySkipped,
    DeliveryUnconfirmed,
}

struct TokenSignalPhotoRequest<'a> {
    signal: &'a TokenSignal,
    signal_id: &'a str,
    chat_id: ChatId,
    reply_to_message_id: Option<MessageId>,
    timeframe: Option<&'a str>,
    locale: bot_core::locale::Locale,
    timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenSignalPhotoDelivery {
    Sent {
        caption: String,
        message_id: MessageId,
    },
    Failed {
        caption: String,
        failure: TokenSignalPhotoFailure,
    },
}

/// Stores votes and state for polls the bot sent. Both return whether the
/// update belonged to a stored poll.
pub trait PollUpdateSink {
    fn record_answer(&mut self, answer: &bot_core::polls::PollAnswer) -> Result<bool, String>;

    fn apply_state(&mut self, state: bot_core::polls::PollState) -> Result<bool, String>;
}

pub trait TokenSignalSource {
    fn load(&mut self, query: &SignalQuery) -> TokenSignalLoad;

    fn load_token(&mut self, token: &TokenAddress) -> TokenSignalLoad;

    fn load_candidates(&mut self, query: &SignalQuery) -> TokenSignalCandidates {
        let load = self.load(query);
        TokenSignalCandidates {
            signals: load.signal.into_iter().collect(),
            diagnostics: load.diagnostics,
        }
    }

    fn render_period_photo(
        &mut self,
        _signal: &TokenSignal,
        _period: &str,
        _now: i64,
    ) -> Result<Vec<u8>, String> {
        Err("requested token history unavailable".into())
    }

    fn period_candles(
        &mut self,
        _signal: &TokenSignal,
        _period: &str,
        _now: i64,
    ) -> Result<Vec<Vec<f64>>, String> {
        Ok(Vec::new())
    }

    fn load_state(&mut self, signal_id: &str) -> Result<Option<SignalState>, String>;

    fn save_state(&mut self, signal_id: &str, state: &SignalState) -> Result<(), String>;

    fn clear_state(&mut self, _signal_id: &str) -> Result<(), String> {
        Ok(())
    }
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
    Unsupported,
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum DispatchError<ConfigError, ActionError, RandomError> {
    #[error("could not load chat configuration: {0}")]
    Config(ConfigError),
    #[error("could not execute Telegram action: {0}")]
    Action(ActionError),
    #[error("could not obtain a random value: {0}")]
    Random(RandomError),
    #[error("required native service is not configured: {0}")]
    MissingService(&'static str),
    #[error("native dispatch invariant failed: {0}")]
    Invariant(&'static str),
    /// A write that must not be lost failed; the update is retried.
    #[error("could not persist {0}")]
    Persistence(String),
}

type NativeDispatchResult<Config, Actions, Random> = Result<
    DispatchOutcome,
    DispatchError<
        <Config as ChatConfigSource>::Error,
        <Actions as ActionSink>::Error,
        <Random as RandomSource>::Error,
    >,
>;

type OptionalNativeDispatchResult<Config, Actions, Random> = Result<
    Option<DispatchOutcome>,
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
    lightning_checkout: Option<Box<dyn LightningCheckout>>,
    ban_store: Option<Box<dyn ChatBanStore>>,
    /// Resolves the @username in ban and limit commands.
    member_source: Option<Box<dyn crate::chat_members_tool::ChatMemberSource>>,
    limit_store: Option<Box<dyn ChatLimitStore>>,
    group_spending_source: Option<Box<dyn GroupSpendingSource>>,
    /// Set while a message is routed to the AI turn only to be recorded as
    /// ignored: a banned member's, or a bare `/ban` left to moderation bots.
    listen_only: bool,
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
    token_signal_source: Option<Box<dyn TokenSignalSource>>,
    poll_update_sink: Option<Box<dyn PollUpdateSink>>,
    ai_conversation_source: Option<Box<dyn AiConversationSource>>,
    trigger_words: Vec<String>,
    /// Link preview context already fetched by link replacement for the
    /// current message, so the AI turn does not fetch it again.
    prefetched_link_context: Option<Option<String>>,
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
            lightning_checkout: None,
            ban_store: None,
            member_source: None,
            limit_store: None,
            group_spending_source: None,
            listen_only: false,
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
            token_signal_source: None,
            poll_update_sink: None,
            ai_conversation_source: None,
            trigger_words: vec!["bot".to_owned(), "assistant".to_owned()],
            prefetched_link_context: None,
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
    pub fn with_lightning_checkout(mut self, checkout: Box<dyn LightningCheckout>) -> Self {
        self.lightning_checkout = Some(checkout);
        self
    }

    #[must_use]
    pub fn with_ban_store(mut self, store: Box<dyn ChatBanStore>) -> Self {
        self.ban_store = Some(store);
        self
    }

    #[must_use]
    pub fn with_member_source(
        mut self,
        source: Box<dyn crate::chat_members_tool::ChatMemberSource>,
    ) -> Self {
        self.member_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_limit_store(mut self, store: Box<dyn ChatLimitStore>) -> Self {
        self.limit_store = Some(store);
        self
    }

    #[must_use]
    pub fn with_group_spending_source(mut self, source: Box<dyn GroupSpendingSource>) -> Self {
        self.group_spending_source = Some(source);
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
    pub fn with_token_signal_source(mut self, source: Box<dyn TokenSignalSource>) -> Self {
        self.token_signal_source = Some(source);
        self
    }

    #[must_use]
    pub fn with_poll_update_sink(mut self, sink: Box<dyn PollUpdateSink>) -> Self {
        self.poll_update_sink = Some(sink);
        self
    }

    /// Votes and poll state only reach the bot for polls it sent.
    fn dispatch_poll_update(
        &mut self,
        payload: &Map<String, Value>,
        answer: bool,
    ) -> DispatchOutcome {
        let Some(sink) = self.poll_update_sink.as_mut() else {
            return DispatchOutcome::Unsupported;
        };
        let stored = if answer {
            bot_core::polls::parse_poll_answer(payload).map(|answer| sink.record_answer(&answer))
        } else {
            bot_core::polls::parse_poll_state(payload).map(|state| sink.apply_state(state))
        };
        match stored {
            Some(Ok(true)) => DispatchOutcome::Handled,
            Some(Ok(false)) | None => DispatchOutcome::Unsupported,
            Some(Err(error)) => {
                self.state_diagnostics
                    .push(format!("poll update not stored: {error}"));
                DispatchOutcome::Unsupported
            }
        }
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

    /// Whether the AI would answer this message on its own merits: it
    /// mentions or replies to the bot, or is a private message with more
    /// than bare links (a bare link in private is only a link-fix request).
    /// Replies routing ignores (to a link fix, or to a non-AI command when
    /// those followups are off) are not addressed, so their original is
    /// still handled like any other link message.
    fn is_addressed_to_bot(
        &mut self,
        message: &IncomingMessage,
        text: &str,
        config: &ChatConfig,
    ) -> bool {
        let bot_username = self.bot_name.trim().trim_start_matches('@');
        if bot_username.is_empty() {
            return message.chat_type.as_deref() == Some("private")
                && !text_without_links(text).trim().is_empty();
        }
        let mention = text
            .to_lowercase()
            .contains(&format!("@{}", bot_username.to_lowercase()));
        let reply_to_bot = message.replied_sender_username.as_deref() == Some(bot_username);
        let addressed = mention
            || reply_to_bot
            || (message.chat_type.as_deref() == Some("private")
                && !text_without_links(text).trim().is_empty());
        if !addressed || !reply_to_bot {
            return addressed;
        }
        if config.ignore_link_fix_followups
            && bot_core::routing::is_link_fix_text(
                message.replied_text.as_deref().unwrap_or_default(),
            )
        {
            return false;
        }
        config.ai_command_followups
            || !self
                .replied_bot_metadata(message)
                .is_some_and(|metadata| metadata.is_non_ai_command())
    }

    /// Stored metadata for the bot message this message replies to.
    fn replied_bot_metadata(
        &mut self,
        message: &IncomingMessage,
    ) -> Option<crate::ai_dispatch::AiReplyMetadata> {
        let (chat_id, reply_id) = (message.chat_id?, message.replied_message_id?);
        let source = self.ai_conversation_source.as_mut()?;
        match source.reply_metadata(&chat_id.0.to_string(), &reply_id.0.to_string()) {
            Ok(metadata) => metadata,
            Err(error) => {
                self.state_diagnostics
                    .push(format!("AI reply metadata: {error}"));
                None
            }
        }
    }

    /// Returns `Some` when link replacement fully handled the message. A
    /// message addressed to the bot gets its links fixed (without deleting
    /// the original, which the answer replies to) and then `None`, so the AI
    /// still answers with the fetched preview context.
    fn dispatch_link_replacement(
        &mut self,
        message: &IncomingMessage,
        config: &ChatConfig,
        locale: bot_core::locale::Locale,
        timestamp: i64,
        addressed: bool,
    ) -> OptionalNativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(sender_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return Ok(Some(DispatchOutcome::Unsupported));
        };
        let text = content.text.as_str();
        let mode = LinkMode::parse(&config.link_mode);
        if mode == LinkMode::Off || text.is_empty() || text.starts_with('/') {
            return Ok(None);
        }
        if message.has_reply && !text_without_links(text).trim().is_empty() {
            return Ok(None);
        }
        if !has_replaceable_link(text) {
            return Ok(None);
        }
        let Some(source) = self.link_replacement_source.as_mut() else {
            return Err(DispatchError::MissingService("link replacement"));
        };
        let load = source.load(text, timestamp);
        self.state_diagnostics.extend(load.diagnostics);
        if addressed {
            self.prefetched_link_context = Some(load.context.clone());
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
            // Only changed links produce a plan (the mode is not off here).
            if addressed {
                return Ok(None);
            }
            let incoming = prepare_incoming_command_state(IncomingCommandState {
                chat_id,
                message_id,
                user_id: sender_id,
                first_name: message.sender_first_name.as_deref(),
                username: message.sender_username.as_deref(),
                is_bot: message.sender_is_bot,
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
            return Ok(Some(DispatchOutcome::Handled));
        };
        let sent_key = addressed.then(|| sent_link_fix_key(chat_id, message_id));
        if let Some(key) = sent_key.as_deref()
            && let Some(source) = self.market_price_source.as_mut()
        {
            match source.load_selection(key) {
                Ok(Some(_)) => return Ok(None),
                Ok(None) => {}
                Err(error) => self
                    .state_diagnostics
                    .push(format!("sent link fix lookup: {error}")),
            }
        }
        // Link plans always send a message, which carries the video caption.
        let video_action = match (&plan.send, load.oversized_video) {
            (TelegramAction::SendMessage(message), Some(video)) => {
                Some(TelegramAction::SendVideo {
                    chat_id: message.chat_id,
                    video: video.into(),
                    reply_to_message_id: message.reply_to_message_id,
                    caption: message.text.clone(),
                    reply_markup: message.reply_markup.clone(),
                })
            }
            _ => None,
        };
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
        if let Some(key) = sent_key.as_deref()
            && let Some(source) = self.market_price_source.as_mut()
            && let Err(error) = source.save_selection(key, "1", SENT_LINK_FIX_TTL_SECONDS)
        {
            self.state_diagnostics
                .push(format!("sent link fix marker: {error}"));
        }
        if let Some(delete) = plan.delete_original.filter(|_| !addressed) {
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
        Ok((!addressed).then_some(DispatchOutcome::Handled))
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
        let payload_locale = message
            .get("successful_payment")
            .and_then(Value::as_object)
            .and_then(|payment| payment.get("invoice_payload"))
            .and_then(Value::as_str)
            .and_then(invoice_payload_locale);
        let locale = resolve_locale(payload_locale, language_code, chat_type);
        let decision = match evaluate_default_successful_payment(
            &Value::Object(message),
            self.billing_available,
        ) {
            Ok(decision) => decision,
            Err(error) => {
                self.state_diagnostics
                    .push(format!("invalid successful payment: {error}"));
                return Ok(DispatchOutcome::Handled);
            }
        };
        self.state_diagnostics.clear();
        let (chat_id, text) = match &decision {
            SuccessfulPaymentDecision::Ignore => return Ok(DispatchOutcome::Handled),
            SuccessfulPaymentDecision::BillingUnavailable { chat_id } => (
                chat_id,
                bot_core::billing_commands::billing_unavailable(locale).to_owned(),
            ),
            SuccessfulPaymentDecision::InvalidPayment {
                chat_id,
                user_id,
                currency,
                payload,
                total_amount,
                charge_id,
            } => {
                // The user paid, so the charge id must survive for a refund.
                eprintln!("Invalid successful payment chat_id={chat_id} charge_id={charge_id}");
                self.state_diagnostics.push(format!(
                    "Invalid successful payment payload chat_id={chat_id} user_id={user_id} currency={currency} payload={payload} total_amount={total_amount} charge_id={charge_id}"
                ));
                (
                    chat_id,
                    match locale {
                        bot_core::locale::Locale::Es => {
                            "Me cayó un pago raro y no lo pude validar. Avisale al admin".to_owned()
                        }
                        bot_core::locale::Locale::En => {
                            "I received an invalid payment. Please tell the admin".to_owned()
                        }
                    },
                )
            }
            SuccessfulPaymentDecision::Record {
                chat_id,
                credits_awarded,
                ..
            } => {
                let invariant = "recordable payment did not produce a ledger record";
                let payment =
                    payment_record(&decision).ok_or(DispatchError::Invariant(invariant))?;
                let Some(sink) = self.payment_sink.as_mut() else {
                    return Err(DispatchError::MissingService("payment persistence"));
                };
                // Recording is idempotent per charge id, so a failed write is
                // retried with the update instead of leaving the user paid
                // but uncredited. The log keeps the charge id either way.
                let receipt = sink.record(&payment).map_err(|error| {
                    // The charge id is enough to find and refund the payment.
                    eprintln!(
                        "Payment not recorded yet: chat_id={chat_id} charge_id={}",
                        payment.charge_id
                    );
                    DispatchError::Persistence(format!(
                        "successful payment chat_id={chat_id} charge_id={}: {error}",
                        payment.charge_id
                    ))
                })?;
                let text = successful_payment_reply(
                    *credits_awarded,
                    receipt.user_balance,
                    receipt.inserted,
                    locale,
                );
                (chat_id, text)
            }
        };
        let Ok(chat_id) = chat_id.parse::<i64>() else {
            return Err(DispatchError::Invariant(
                "validated payment chat id was not numeric",
            ));
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

    fn send_failure_reply(
        &mut self,
        chat_id: ChatId,
        message_id: MessageId,
        text: &str,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let mut reply = SendMessage::new(chat_id, text);
        reply.reply_to_message_id = Some(message_id);
        let _receipt = self
            .actions
            .execute(TelegramAction::SendMessage(reply))
            .map_err(DispatchError::Action)?;
        Ok(DispatchOutcome::Handled)
    }

    /// Keep the chat's `/` menu in the language its replies use. Best effort:
    /// a failure is logged and never blocks the reply.
    fn sync_chat_command_menu(&mut self, chat_id: ChatId, locale: bot_core::locale::Locale) {
        if self
            .actions
            .execute(bot_core::telegram_commands::chat_command_menu_action(
                chat_id, locale,
            ))
            .is_err()
        {
            self.state_diagnostics.push(format!(
                "chat command menu update failed chat_id={}",
                chat_id.0
            ));
        }
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

    fn record_price_delivery(
        &mut self,
        message: &IncomingMessage,
        text: &str,
        sent_message_id: Option<MessageId>,
        timestamp: i64,
    ) {
        let (Some(chat_id), Some(message_id), Some(user_id), Some(content)) = (
            message.chat_id,
            message.message_id,
            message.sender_id,
            message.content.as_ref(),
        ) else {
            return;
        };
        if let Ok(incoming) = prepare_incoming_command_state(IncomingCommandState {
            chat_id,
            message_id,
            user_id,
            first_name: message.sender_first_name.as_deref(),
            username: message.sender_username.as_deref(),
            is_bot: message.sender_is_bot,
            text: &content.text,
            is_group: is_group_chat_type(message.chat_type.as_deref()),
            timestamp,
        }) && let Err(error) = self.state.record_incoming(&incoming)
        {
            self.state_diagnostics
                .push(format!("incoming price state: {error}"));
        }
        let command = parse_command(&content.text, &self.bot_name).command;
        if let Ok(outgoing) = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id,
            incoming_message_id: message_id,
            sent_message_id,
            text,
            command: &command,
            timestamp,
        }) && let Err(error) = self.state.record_outgoing(&outgoing)
        {
            self.state_diagnostics
                .push(format!("outgoing price state: {error}"));
        }
    }

    /// `chat_id` is the callback chat id, already validated as numeric.
    fn record_market_callback_delivery(
        &mut self,
        context: &CallbackContext,
        chat_id: i64,
        text: &str,
        sent_message_id: Option<MessageId>,
        command: &str,
        timestamp: i64,
    ) {
        if let Ok(outgoing) = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id: ChatId(chat_id),
            incoming_message_id: MessageId(context.message_id),
            sent_message_id,
            text,
            command,
            timestamp,
        }) && let Err(error) = self.state.record_outgoing(&outgoing)
        {
            self.state_diagnostics
                .push(format!("market callback history: {error}"));
        }
    }

    /// Puts an atomically consumed selection back so a retried update can
    /// resolve it again. Only used on paths where no quote was delivered.
    fn restore_market_selection(
        &mut self,
        key: &str,
        value: &str,
        selection_id: &str,
        chat_id: &str,
    ) {
        if let Some(source) = self.market_price_source.as_mut()
            && let Err(error) = source.save_selection(key, value, MARKET_SELECTION_TTL_SECONDS)
        {
            self.state_diagnostics.push(format!(
                "market selection restore failed chat_id={chat_id} selection_id={selection_id}: {error}"
            ));
        }
    }

    fn persist_market_selection(
        &mut self,
        message: &IncomingMessage,
        selection: &MarketSelection,
        command: MarketPriceCommand,
        timestamp: i64,
        locale: bot_core::locale::Locale,
        selection_index: usize,
    ) -> OptionalNativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(requester_id)) =
            (message.chat_id, message.message_id, message.sender_id)
        else {
            return Ok(None);
        };
        let selection_id = market_selection_id(
            chat_id.0,
            message_id.0,
            requester_id.0,
            timestamp,
            selection_index,
        );
        let mut stored = StoredMarketSelection {
            selection: selection.clone(),
            chat_id: chat_id.0.to_string(),
            // Bind the callback to the requesting message immediately. The
            // second write below replaces this with the bot's sent message id
            // once Telegram confirms delivery.
            message_id: message_id.0,
            source_message_id: Some(message_id.0),
            requester_id: requester_id.0,
            command: if command == MarketPriceCommand::CryptoOnly {
                "crypto".to_owned()
            } else {
                "unified".to_owned()
            },
        };
        let encoded = stored.encode();
        let key = market_selection_key(&selection_id);
        let Some(source) = self.market_price_source.as_mut() else {
            self.state_diagnostics
                .push("market selection storage unavailable".to_owned());
            return Ok(None);
        };
        if let Err(error) = source.save_selection(&key, &encoded, MARKET_SELECTION_TTL_SECONDS) {
            self.state_diagnostics
                .push(format!("market selection storage unavailable: {error}"));
            return Ok(None);
        }
        let text = &market_selection_text(selection, locale);
        let mut reply = SendMessage::new(chat_id, &market_selection_prompt(selection, locale));
        reply.reply_to_message_id = Some(message_id);
        reply.reply_markup = Some(market_selection_page(&selection_id, selection, locale, 0));
        let receipt = self
            .actions
            .execute(TelegramAction::SendMessage(reply))
            .map_err(|error| {
                if let Some(source) = self.market_price_source.as_mut()
                    && let Err(clear_error) = source.clear_selection(&key)
                {
                    self.state_diagnostics.push(format!(
                        "market selection clear after send failure: {clear_error}"
                    ));
                }
                DispatchError::Action(error)
            })?;
        let Some(sent_message_id) = receipt.message_id else {
            self.state_diagnostics
                .push("market selection delivery was unconfirmed".to_owned());
            if let Some(source) = self.market_price_source.as_mut()
                && let Err(error) = source.clear_selection(&key)
            {
                self.state_diagnostics.push(format!(
                    "market selection clear after unconfirmed delivery: {error}"
                ));
            }
            self.record_price_delivery(message, text, None, timestamp);
            return Ok(Some(DispatchOutcome::Handled));
        };
        stored.message_id = sent_message_id.0;
        let encoded = stored.encode();
        if let Some(source) = self.market_price_source.as_mut()
            && let Err(error) = source.save_selection(&key, &encoded, MARKET_SELECTION_TTL_SECONDS)
        {
            self.state_diagnostics
                .push(format!("market selection update unavailable: {error}"));
            if let Err(clear_error) = source.clear_selection(&key) {
                self.state_diagnostics.push(format!(
                    "market selection clear after update failure: {clear_error}"
                ));
            }
            let _ = self.actions.try_edit(TelegramAction::EditMessage {
                chat_id,
                message_id: sent_message_id,
                text: text.to_owned(),
                reply_markup: Some(InlineKeyboardMarkup {
                    inline_keyboard: Vec::new(),
                }),
            });
        }
        self.record_price_delivery(message, text, Some(sent_message_id), timestamp);
        Ok(Some(DispatchOutcome::Handled))
    }

    fn dispatch_market_price_query(
        &mut self,
        message: &IncomingMessage,
        text: &str,
        command: MarketPriceCommand,
        locale: bot_core::locale::Locale,
        timestamp: i64,
    ) -> OptionalNativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id)) = (message.chat_id, message.message_id) else {
            return Ok(Some(DispatchOutcome::Unsupported));
        };
        let Some(source) = self.market_price_source.as_mut() else {
            return Err(DispatchError::MissingService("market prices"));
        };
        let load = source.load(text, command, locale, timestamp);
        self.state_diagnostics.extend(load.diagnostics.clone());
        if let Some(selection) = load.selection {
            let selection_text = format_market_selection(&selection, locale);
            if let Some(outcome) =
                self.persist_market_selection(message, &selection, command, timestamp, locale, 0)?
            {
                return Ok(Some(outcome));
            }
            let mut reply = SendMessage::new(chat_id, &selection_text);
            reply.reply_to_message_id = Some(message_id);
            let receipt = self
                .actions
                .execute(TelegramAction::SendMessage(reply))
                .map_err(DispatchError::Action)?;
            self.record_price_delivery(message, &selection_text, receipt.message_id, timestamp);
            return Ok(Some(DispatchOutcome::Handled));
        }
        let output = if load.text.trim().is_empty() {
            match locale {
                bot_core::locale::Locale::Es => {
                    "No pude conseguir una cotización. Probá más tarde".to_owned()
                }
                bot_core::locale::Locale::En => {
                    "I could not get a quote. Try again later".to_owned()
                }
            }
        } else {
            load.text
        };
        let mut reply = SendMessage::new(chat_id, &output);
        reply.reply_to_message_id = Some(message_id);
        let receipt = self
            .actions
            .execute(TelegramAction::SendMessage(reply))
            .map_err(DispatchError::Action)?;
        self.record_price_delivery(message, &output, receipt.message_id, timestamp);
        Ok(Some(DispatchOutcome::Handled))
    }

    fn dispatch_asset_prices(
        &mut self,
        message: &IncomingMessage,
        text: &str,
        command: MarketPriceCommand,
        locale: bot_core::locale::Locale,
        timestamp: i64,
    ) -> OptionalNativeDispatchResult<Config, Actions, Random> {
        use bot_core::price_queries::{PriceQuery, ProviderScope, parse_price_query};
        let query_text = provider_query_text(text);
        let mut valid_periods = vec!["1h".into(), "24h".into(), "7d".into(), "30d".into()];
        if let Some(candidate) = query_text.split_whitespace().last()
            && bot_core::price_queries::ChartPeriod::parse(candidate).is_some()
        {
            valid_periods.push(candidate.to_ascii_lowercase());
        }
        let parsed_query = parse_price_query(&query_text, &valid_periods);
        if matches!(
            &parsed_query,
            PriceQuery::AmountConversion(_)
                | PriceQuery::Assets {
                    conversion_requested: true,
                    ..
                }
        ) {
            return self.dispatch_market_price_query(
                message,
                &query_text,
                command,
                locale,
                timestamp,
            );
        }
        let PriceQuery::Assets {
            query,
            timeframe,
            conversion_requested: false,
            provider_scope,
            ..
        } = parsed_query
        else {
            return Ok(None);
        };
        if query
            .split(|c: char| c.is_whitespace() || c == ',')
            .any(|part| part.parse::<u64>().is_ok())
        {
            return Ok(None);
        }
        let (Some(chat_id), Some(message_id)) = (message.chat_id, message.message_id) else {
            return Ok(Some(DispatchOutcome::Unsupported));
        };
        let requests = if query.trim().is_empty() {
            vec![query.as_str()]
        } else {
            query
                .split(',')
                .map(str::trim)
                .filter(|query| !query.is_empty())
                .take(20)
                .collect::<Vec<_>>()
        };
        if requests.is_empty() {
            return Ok(None);
        }
        if self.market_price_source.is_none() && self.token_signal_source.is_none() {
            return Err(DispatchError::MissingService(
                if detect_signal_query(text).is_some() {
                    "token signals"
                } else {
                    "market prices"
                },
            ));
        }
        let single = requests.len() == 1;
        let direct_query = single.then(|| detect_signal_query(text)).flatten();
        let mut lines = Vec::new();
        let mut pending_selections = Vec::new();
        for request in requests {
            let detected = direct_query
                .clone()
                .or_else(|| detect_signal_query(request));
            let is_address = matches!(detected, Some(SignalQuery::Address(_)));
            let provider_request = detected.as_ref().map_or_else(
                || request.to_owned(),
                |query| provider_request_text(request, query),
            );
            let market_alias = provider_request.trim().to_ascii_lowercase();
            let token_query = if provider_scope == Some(ProviderScope::Stock)
                || market_alias.is_empty()
                || matches!(market_alias.as_str(), "stables" | "stablecoins")
            {
                None
            } else {
                detected
                    .clone()
                    .or_else(|| detect_signal_query(&format!("${request}")))
            };
            let mut market_query = match provider_scope {
                Some(ProviderScope::Stock) => format!("stock:{provider_request}"),
                Some(ProviderScope::Crypto) => format!("crypto:{provider_request}"),
                None => provider_request.clone(),
            };
            if let Some(timeframe) = &timeframe {
                market_query.push_str(&format!(" {timeframe}"));
            }
            let mut load = if is_address && provider_scope != Some(ProviderScope::Stock) {
                None
            } else {
                self.market_price_source
                    .as_mut()
                    .map(|source| source.load(&market_query, command, locale, timestamp))
            };
            if let Some(chart) = load.as_mut().and_then(|load| load.chart.as_mut()) {
                chart.timeframe.clone_from(&timeframe);
            }
            if let Some(load) = &load {
                self.state_diagnostics.extend(load.diagnostics.clone());
            }
            let probe_token_signal = provider_scope != Some(ProviderScope::Stock)
                && token_query.is_some()
                && (load.as_ref().is_none_or(|load| {
                    load.selection.is_some()
                        || load.no_assets_found
                        || load
                            .chart
                            .as_ref()
                            .is_some_and(|chart| chart.candidate.is_some())
                }));
            let mut token_candidates = Vec::new();
            if probe_token_signal
                && let Some(token_query) = token_query.as_ref()
                && let Some(source) = self.token_signal_source.as_mut()
            {
                let canonical_token = load
                    .as_ref()
                    .and_then(|load| load.chart.as_ref())
                    .and_then(|chart| chart.token.as_ref())
                    .cloned();
                // A provider's symbol match can be a namesake. Discover DEX
                // identities first, then add the provider's canonical contract
                // as another candidate when it is distinct. Address queries
                // never consult the provider, so they have no canonical token.
                let loaded = source.load_candidates(token_query);
                self.state_diagnostics.extend(loaded.diagnostics);
                token_candidates = loaded.signals;
                if let Some(token) = canonical_token.as_ref() {
                    let loaded = source.load_token(token);
                    self.state_diagnostics.extend(loaded.diagnostics);
                    if let Some(signal) = loaded.signal
                        && !token_candidates.iter().any(|candidate| {
                            market_contracts_match(&candidate.token, &signal.token)
                        })
                    {
                        token_candidates.push(signal);
                    }
                }
                token_candidates.retain(|signal| {
                    self.market_price_source.is_none()
                        || token_signal_matches_query(signal, token_query)
                });
            }
            let token_signal = (token_candidates.len() == 1)
                .then(|| token_candidates.first().cloned())
                .flatten();
            if token_candidates.len() > 1 {
                let mut menu_candidates = Vec::new();
                for signal in &token_candidates {
                    let history =
                        self.period_change_history(signal, timeframe.as_deref(), timestamp);
                    menu_candidates.push(market_token_candidate(
                        signal,
                        Some(history.1.as_str()),
                        history.0.as_deref(),
                    ));
                }
                let token_candidates = menu_candidates;
                let mut selection = if let Some(load) = load.as_ref()
                    && let Some(existing) = load.selection.as_ref()
                {
                    existing.clone()
                } else if let Some(chart) = load.as_ref().and_then(|load| load.chart.as_ref())
                    && let Some(market_candidate) = chart.candidate.clone()
                {
                    MarketSelection {
                        query: provider_request.clone(),
                        timeframe: timeframe.clone(),
                        target_symbol: "USD".to_owned(),
                        target_parameter: "USD".to_owned(),
                        conversion: None,
                        candidates: vec![market_candidate],
                    }
                } else {
                    MarketSelection {
                        query: provider_request.clone(),
                        timeframe: timeframe.clone(),
                        target_symbol: "USD".to_owned(),
                        target_parameter: "USD".to_owned(),
                        conversion: None,
                        candidates: Vec::new(),
                    }
                };
                selection.candidates.extend(token_candidates);
                deduplicate_market_candidates(&mut selection.candidates);
                if selection.candidates.len() > 1 {
                    pending_selections.push(selection);
                    continue;
                }
            }
            if let Some(signal) = token_signal.as_ref() {
                let history = self.period_change_history(signal, timeframe.as_deref(), timestamp);
                let token_candidate =
                    market_token_candidate(signal, Some(history.1.as_str()), history.0.as_deref());
                if let Some(load) = load.as_mut()
                    && let Some(selection) = load.selection.as_mut()
                {
                    selection.candidates.push(token_candidate);
                    deduplicate_market_candidates(&mut selection.candidates);
                    selection.timeframe.clone_from(&timeframe);
                } else if let Some(load) = load.as_ref()
                    && let Some(chart) = load.chart.as_ref()
                    && let Some(market_candidate) = chart.candidate.clone()
                {
                    let mut selection = MarketSelection {
                        query: provider_request.clone(),
                        timeframe: timeframe.clone(),
                        target_symbol: "USD".to_owned(),
                        target_parameter: "USD".to_owned(),
                        conversion: None,
                        candidates: vec![market_candidate, token_candidate],
                    };
                    deduplicate_market_candidates(&mut selection.candidates);
                    if selection.candidates.len() > 1 {
                        pending_selections.push(selection);
                        continue;
                    }
                }
            }
            let mut direct_market_candidate = None;
            if let Some(load) = load.as_ref()
                && let Some(selection) = &load.selection
            {
                let mut selection = selection.clone();
                selection.timeframe.clone_from(&timeframe);
                // Each comma-separated request owns a separate candidate
                // group.  Keeping them separate preserves all candidates,
                // the request-specific conversion context, and an
                // independently retryable callback menu.
                if selection.candidates.len() > 1 || token_signal.is_none() {
                    pending_selections.push(selection);
                    continue;
                }
                let MarketSelection {
                    candidates,
                    target_symbol,
                    target_parameter,
                    conversion,
                    ..
                } = selection;
                if let Some(candidate) = candidates.into_iter().next() {
                    direct_market_candidate =
                        Some((candidate, target_symbol, target_parameter, conversion));
                }
            }
            if let Some((candidate, target_symbol, target_parameter, conversion)) =
                direct_market_candidate
            {
                load = Some(
                    self.market_price_source
                        .as_mut()
                        .map(|source| {
                            source.load_candidate(
                                &candidate,
                                timeframe.as_deref(),
                                &target_symbol,
                                &target_parameter,
                                conversion.as_ref(),
                                command,
                                locale,
                                timestamp,
                            )
                        })
                        .unwrap_or(MarketPriceLoad {
                            chart: None,
                            selection: None,
                            no_assets_found: true,
                            text: String::new(),
                            diagnostics: vec!["market price source disappeared".to_owned()],
                        }),
                );
                if let Some(load) = &load {
                    self.state_diagnostics.extend(load.diagnostics.clone());
                }
            }
            if let Some(load) = load.as_ref().filter(|load| !load.no_assets_found) {
                if single && let Some(chart) = &load.chart {
                    let mut caption = if load.text.trim().is_empty() {
                        match locale {
                            bot_core::locale::Locale::Es => {
                                format!("{}: cotización disponible", chart.symbol)
                            }
                            bot_core::locale::Locale::En => {
                                format!("{}: quote available", chart.symbol)
                            }
                        }
                    } else {
                        load.text.clone()
                    };
                    let photo = self
                        .market_price_source
                        .as_mut()
                        .and_then(|source| source.render_chart(chart, timestamp).ok());
                    if let Some(rendered) = photo {
                        if let Some(chart_caption) = rendered.caption {
                            caption = chart_caption;
                        }
                        let delivered = self.actions.try_photo(TelegramAction::SendPhoto {
                            chat_id,
                            photo: rendered.photo.into(),
                            reply_to_message_id: Some(message_id),
                            caption: caption.clone(),
                            parse_mode: None,
                            reply_markup: None,
                        });
                        if let Ok(Some(receipt)) = delivered
                            && receipt.message_id.is_some()
                        {
                            self.record_price_delivery(
                                message,
                                &caption,
                                receipt.message_id,
                                timestamp,
                            );
                            return Ok(Some(DispatchOutcome::Handled));
                        }
                    }
                    // CMC may know an exact contract while Yahoo has no
                    // symbol for a token. Use the verified token identity for
                    // the historical chart before falling back to the quote.
                    if let Some(token) = chart.token.as_ref()
                        && let Some(source) = self.token_signal_source.as_mut()
                    {
                        let token_load = source.load_token(token);
                        self.state_diagnostics.extend(token_load.diagnostics);
                        if message.sender_id.is_some()
                            && let Some(outcome) = self.dispatch_token_signal_loaded_message(
                                message,
                                &SignalQuery::Address(token.clone()),
                                token_load.signal,
                                timeframe.as_deref(),
                                locale,
                                timestamp,
                            )?
                        {
                            return Ok(Some(outcome));
                        }
                    }
                    self.state_diagnostics.push(format!(
                        "market chart unavailable or undelivered: {}",
                        chart.symbol
                    ));
                    lines.push(format!(
                        "{}\n{}",
                        caption,
                        match locale {
                            bot_core::locale::Locale::Es =>
                                "Gráfico no disponible. Probá más tarde",
                            bot_core::locale::Locale::En => "Chart unavailable. Try again later",
                        }
                    ));
                    continue;
                }
                if !load.text.trim().is_empty() {
                    lines.push(load.text.clone());
                } else {
                    lines.push(match locale {
                        bot_core::locale::Locale::Es => {
                            format!(
                                "No pude conseguir una cotización para {}",
                                load.chart
                                    .as_ref()
                                    .map_or(request, |chart| chart.symbol.as_str())
                            )
                        }
                        bot_core::locale::Locale::En => {
                            format!(
                                "I could not get a quote for {}",
                                load.chart
                                    .as_ref()
                                    .map_or(request, |chart| chart.symbol.as_str())
                            )
                        }
                    });
                }
                continue;
            }
            if let Some(token_query) = token_query {
                if single {
                    if let Some(outcome) = self.dispatch_token_signal_loaded_message(
                        message,
                        &token_query,
                        token_signal,
                        timeframe.as_deref(),
                        locale,
                        timestamp,
                    )? {
                        return Ok(Some(outcome));
                    }
                } else if let Some(signal) = token_signal {
                    let history =
                        self.period_change_history(&signal, timeframe.as_deref(), timestamp);
                    let quote = bot_core::token_signals::format_signal_quote_with_candles(
                        &signal,
                        Some(history.1.as_str()),
                        history.0.as_deref(),
                    );
                    lines.push(quote);
                    continue;
                }
            }
            lines.push(
                load.map(|load| {
                    if load.text.trim().is_empty() {
                        match locale {
                            bot_core::locale::Locale::Es => {
                                format!("No pude conseguir una cotización para {request}")
                            }
                            bot_core::locale::Locale::En => {
                                format!("I could not get a quote for {request}")
                            }
                        }
                    } else {
                        load.text
                    }
                })
                .unwrap_or_else(|| match locale {
                    bot_core::locale::Locale::Es => {
                        format!("No encontré datos para {request}")
                    }
                    bot_core::locale::Locale::En => format!("I could not find data for {request}"),
                }),
            );
        }
        if !pending_selections.is_empty() {
            if !lines.is_empty() {
                let text = lines.join("\n");
                let mut reply = SendMessage::new(chat_id, &text);
                reply.reply_to_message_id = Some(message_id);
                let receipt = self
                    .actions
                    .execute(TelegramAction::SendMessage(reply))
                    .map_err(DispatchError::Action)?;
                self.record_price_delivery(message, &text, receipt.message_id, timestamp);
            }
            for (selection_index, selection) in pending_selections.into_iter().enumerate() {
                let selection_text = market_selection_text(&selection, locale);
                let persisted = self
                    .persist_market_selection(
                        message,
                        &selection,
                        command,
                        timestamp,
                        locale,
                        selection_index,
                    )?
                    .is_some();
                if persisted {
                    continue;
                }
                let mut reply = SendMessage::new(chat_id, &selection_text);
                reply.reply_to_message_id = Some(message_id);
                let receipt = self
                    .actions
                    .execute(TelegramAction::SendMessage(reply))
                    .map_err(DispatchError::Action)?;
                self.record_price_delivery(message, &selection_text, receipt.message_id, timestamp);
            }
            return Ok(Some(DispatchOutcome::Handled));
        }
        let text = lines.join("\n");
        let mut reply = SendMessage::new(chat_id, &text);
        reply.reply_to_message_id = Some(message_id);
        let receipt = self
            .actions
            .execute(TelegramAction::SendMessage(reply))
            .map_err(DispatchError::Action)?;
        self.record_price_delivery(message, &text, receipt.message_id, timestamp);
        Ok(Some(DispatchOutcome::Handled))
    }

    fn save_token_signal_state(&mut self, signal_id: &str, state: &SignalState) {
        if let Some(source) = self.token_signal_source.as_mut()
            && let Err(error) = source.save_state(signal_id, state)
        {
            self.state_diagnostics.push(format!(
                "token signal state write failed chat_id={} signal_id={signal_id}: {error}",
                state.chat_id
            ));
        }
    }

    fn render_token_signal_photo(
        &mut self,
        signal: &TokenSignal,
        timeframe: Option<&str>,
        timestamp: i64,
    ) -> Result<Vec<u8>, String> {
        self.token_signal_source.as_mut().map_or(
            Err("native token-signal source disappeared".to_owned()),
            |source| source.render_period_photo(signal, timeframe.unwrap_or("24h"), timestamp),
        )
    }

    fn period_change_history(
        &mut self,
        signal: &TokenSignal,
        timeframe: Option<&str>,
        timestamp: i64,
    ) -> (Option<Vec<Vec<f64>>>, String) {
        let period = timeframe.unwrap_or("24h");
        let mut period_candles = |period: &str| {
            self.token_signal_source
                .as_mut()
                .and_then(|source| source.period_candles(signal, period, timestamp).ok())
                .filter(|candles| !candles.is_empty())
        };
        let candles = period_candles(period);
        if candles
            .as_deref()
            .is_some_and(|candles| has_period_change(candles, period))
        {
            return (candles, period.to_owned());
        }
        for wider in wider_periods(period) {
            let wider_candles = period_candles(wider);
            if wider_candles
                .as_deref()
                .is_some_and(|candles| has_period_change(candles, wider))
            {
                return (wider_candles, (*wider).to_owned());
            }
        }
        (candles, period.to_owned())
    }

    fn try_send_token_signal_photo(
        &mut self,
        request: TokenSignalPhotoRequest<'_>,
    ) -> TokenSignalPhotoDelivery {
        let TokenSignalPhotoRequest {
            signal,
            signal_id,
            chat_id,
            reply_to_message_id,
            timeframe,
            locale,
            timestamp,
        } = request;
        let history = self.period_change_history(signal, timeframe, timestamp);
        let caption = format_signal_caption_for_period_with_candles(
            signal,
            timestamp,
            Some(history.1.as_str()),
            history.0.as_deref(),
        );
        let photo =
            match self.render_token_signal_photo(signal, Some(history.1.as_str()), timestamp) {
                Ok(photo) => photo,
                Err(_) => {
                    return TokenSignalPhotoDelivery::Failed {
                        caption,
                        failure: TokenSignalPhotoFailure::HistoryUnavailable,
                    };
                }
            };
        match self.actions.try_photo(TelegramAction::SendPhoto {
            chat_id,
            photo: photo.into(),
            reply_to_message_id,
            caption: caption.clone(),
            parse_mode: Some(ParseMode::Html),
            reply_markup: Some(build_signal_keyboard_localized(
                signal_id,
                &signal.token,
                &signal.pair,
                locale,
            )),
        }) {
            Ok(Some(receipt)) => match receipt.message_id {
                Some(message_id) => TokenSignalPhotoDelivery::Sent {
                    caption,
                    message_id,
                },
                None => TokenSignalPhotoDelivery::Failed {
                    caption,
                    failure: TokenSignalPhotoFailure::DeliveryUnconfirmed,
                },
            },
            Ok(None) => TokenSignalPhotoDelivery::Failed {
                caption,
                failure: TokenSignalPhotoFailure::DeliverySkipped,
            },
            Err(_) => TokenSignalPhotoDelivery::Failed {
                caption,
                failure: TokenSignalPhotoFailure::DeliveryFailed,
            },
        }
    }

    fn token_signal_state(
        signal: &TokenSignal,
        chart_period: Option<String>,
        chat_id: i64,
        message_id: MessageId,
        source_message_id: i64,
        requester_id: i64,
    ) -> SignalState {
        SignalState {
            chart_period,
            chat_id: chat_id.to_string(),
            message_id: message_id.0,
            source_message_id,
            requester_id: requester_id.to_string(),
            chain_id: signal.token.chain_id.clone(),
            network: signal.token.network.clone(),
            tag: signal.token.tag.clone(),
            address: signal.token.address.clone(),
            last_refresh_at: None,
        }
    }

    fn dispatch_token_signal_loaded_message(
        &mut self,
        message: &IncomingMessage,
        query: &SignalQuery,
        signal: Option<TokenSignal>,
        timeframe: Option<&str>,
        locale: bot_core::locale::Locale,
        timestamp: i64,
    ) -> OptionalNativeDispatchResult<Config, Actions, Random> {
        let (Some(chat_id), Some(message_id), Some(sender_id)) =
            (message.chat_id, message.message_id, message.sender_id)
        else {
            return Ok(Some(DispatchOutcome::Unsupported));
        };
        let Some(signal) = signal else {
            return Ok(None);
        };
        let signal_id = stable_signal_id(chat_id.0, message_id.0, sender_id.0, timestamp);
        match self.try_send_token_signal_photo(TokenSignalPhotoRequest {
            signal: &signal,
            signal_id: &signal_id,
            chat_id,
            reply_to_message_id: Some(message_id),
            timeframe,
            locale,
            timestamp,
        }) {
            TokenSignalPhotoDelivery::Sent {
                caption,
                message_id: sent_message_id,
            } => {
                self.record_price_delivery(message, &caption, Some(sent_message_id), timestamp);
                let state = Self::token_signal_state(
                    &signal,
                    timeframe.map(str::to_owned),
                    chat_id.0,
                    sent_message_id,
                    message_id.0,
                    sender_id.0,
                );
                self.save_token_signal_state(&signal_id, &state);
            }
            TokenSignalPhotoDelivery::Failed { caption, failure } => {
                match failure {
                    TokenSignalPhotoFailure::HistoryUnavailable => {
                        self.state_diagnostics.push(format!(
                            "token signal photo failed chat_id={} query={}",
                            chat_id.0,
                            match query {
                                SignalQuery::Address(token) => token.address.as_str(),
                                SignalQuery::Symbol(symbol) => symbol,
                                SignalQuery::Slug(slug) => slug,
                            }
                        ));
                    }
                    TokenSignalPhotoFailure::DeliveryFailed => self
                        .state_diagnostics
                        .push(format!(
                            "token signal photo delivery failed chat_id={} signal_id={signal_id}",
                            chat_id.0
                        )),
                    TokenSignalPhotoFailure::DeliverySkipped => self
                        .state_diagnostics
                        .push(format!(
                            "token signal photo delivery failed chat_id={} signal_id={signal_id}",
                            chat_id.0
                        )),
                    TokenSignalPhotoFailure::DeliveryUnconfirmed => self
                        .state_diagnostics
                        .push(format!(
                            "token signal photo delivery was unconfirmed chat_id={} signal_id={signal_id}",
                            chat_id.0
                        )),
                }
                let reply_text = if failure == TokenSignalPhotoFailure::HistoryUnavailable {
                    let period = timeframe.unwrap_or("24h");
                    let unavailable = match locale {
                        bot_core::locale::Locale::Es => {
                            format!("No tengo historial de {period}; te dejo la cotización")
                        }
                        bot_core::locale::Locale::En => {
                            format!("No {period} history available; showing the quote")
                        }
                    };
                    format!("{caption}\n{unavailable}")
                } else {
                    caption
                };
                let mut reply = SendMessage::new(chat_id, &reply_text);
                reply.reply_to_message_id = Some(message_id);
                reply.parse_mode = Some(ParseMode::Html);
                let receipt = self
                    .actions
                    .execute(TelegramAction::SendMessage(reply))
                    .map_err(DispatchError::Action)?;
                self.record_price_delivery(message, &reply_text, receipt.message_id, timestamp);
            }
        }
        Ok(Some(DispatchOutcome::Handled))
    }

    fn dispatch_token_signal_callback(
        &mut self,
        context: &CallbackContext,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        if self.token_signal_source.is_none() {
            return Err(DispatchError::MissingService("token signals"));
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
        let mut parts = context.data.splitn(3, ':');
        let valid_prefix = parts.next() == Some("sig");
        let action = parts.next().unwrap_or_default();
        let signal_id = parts.next().unwrap_or_default();
        if !valid_prefix || signal_id.is_empty() {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        }
        let state_load = self
            .token_signal_source
            .as_mut()
            .map_or(Ok(None), |source| source.load_state(signal_id));
        let state = match state_load {
            Ok(Some(state)) => state,
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "token signal state read failed chat_id={} signal_id={signal_id}: {error}",
                    context.chat_id
                ));
                if let Some(callback_id) = context.callback_id.as_deref() {
                    let _receipt = self
                        .actions
                        .execute(TelegramAction::AnswerCallback {
                            callback_id: callback_id.to_owned(),
                            text: Some(signal_callback_text("expired", locale).to_owned()),
                            show_alert: true,
                        })
                        .map_err(DispatchError::Action)?;
                }
                return Ok(DispatchOutcome::Handled);
            }
            Ok(None) => {
                if let Some(callback_id) = context.callback_id.as_deref() {
                    let _receipt = self
                        .actions
                        .execute(TelegramAction::AnswerCallback {
                            callback_id: callback_id.to_owned(),
                            text: Some(signal_callback_text("expired", locale).to_owned()),
                            show_alert: true,
                        })
                        .map_err(DispatchError::Action)?;
                }
                return Ok(DispatchOutcome::Handled);
            }
        };
        let user_id = context.user_id.map(|value| value.to_string());
        let mut allowed = user_id.as_deref() == Some(state.requester_id.as_str());
        if !allowed && is_group_chat_type(Some(&context.chat_type)) {
            let authorization = user_id.as_deref().map_or(
                GroupAuthorizationDecision {
                    is_admin: false,
                    diagnostics: Vec::new(),
                },
                |user_id| self.authorization.authorize(&context.chat_id, user_id),
            );
            self.state_diagnostics.extend(authorization.diagnostics);
            allowed = authorization.is_admin;
        }
        if !allowed {
            if let Some(callback_id) = context.callback_id.as_deref() {
                let _receipt = self
                    .actions
                    .execute(TelegramAction::AnswerCallback {
                        callback_id: callback_id.to_owned(),
                        text: Some(signal_callback_text("owner_only", locale).to_owned()),
                        show_alert: true,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
        let Ok(chat_id) = context.chat_id.parse::<i64>() else {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        };
        if action == "del" {
            // Best effort: a double-tapped delete removes an already gone
            // message, which must not fail the update into retries.
            if self
                .actions
                .execute(TelegramAction::DeleteMessage {
                    chat_id: ChatId(chat_id),
                    message_id: MessageId(context.message_id),
                })
                .is_err()
            {
                self.state_diagnostics.push(format!(
                    "callback delete failed chat_id={} message_id={}",
                    context.chat_id, context.message_id
                ));
            }
            if let Some(source) = self.token_signal_source.as_mut()
                && let Err(error) = source.clear_state(signal_id)
            {
                self.state_diagnostics
                    .push(format!("token signal state cleanup failed: {error}"));
            }
            if let Some(callback_id) = context.callback_id.as_deref() {
                let _receipt = self
                    .actions
                    .execute(TelegramAction::AnswerCallback {
                        callback_id: callback_id.to_owned(),
                        text: Some(signal_callback_text("deleted", locale).to_owned()),
                        show_alert: false,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
        if action != "ref" {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        }
        let timestamp = self.runtime_values.unix_timestamp();
        if state.last_refresh_at.is_some_and(|last_refresh_at| {
            timestamp.saturating_sub(last_refresh_at) < SIGNAL_REFRESH_COOLDOWN_SECONDS
        }) {
            if let Some(callback_id) = context.callback_id.as_deref() {
                let _receipt = self
                    .actions
                    .execute(TelegramAction::AnswerCallback {
                        callback_id: callback_id.to_owned(),
                        text: Some(signal_callback_text("cooldown", locale).to_owned()),
                        show_alert: true,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
        let load = self.token_signal_source.as_mut().map_or(
            TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            |source| source.load_token(&state.token()),
        );
        self.state_diagnostics.extend(load.diagnostics);
        let Some(signal) = load.signal else {
            if let Some(callback_id) = context.callback_id.as_deref() {
                let _receipt = self
                    .actions
                    .execute(TelegramAction::AnswerCallback {
                        callback_id: callback_id.to_owned(),
                        text: Some(signal_callback_text("no_data", locale).to_owned()),
                        show_alert: true,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        };
        let history = self.period_change_history(&signal, state.chart_period.as_deref(), timestamp);
        let photo = self.render_token_signal_photo(&signal, Some(history.1.as_str()), timestamp);
        let edited = match photo {
            Ok(photo) => match self.actions.try_edit(TelegramAction::EditMessagePhoto {
                chat_id: ChatId(chat_id),
                message_id: MessageId(context.message_id),
                photo: photo.into(),
                caption: format_signal_caption_for_period_with_candles(
                    &signal,
                    timestamp,
                    Some(history.1.as_str()),
                    history.0.as_deref(),
                ),
                parse_mode: Some(ParseMode::Html),
                reply_markup: Some(build_signal_keyboard_localized(
                    signal_id,
                    &signal.token,
                    &signal.pair,
                    locale,
                )),
            }) {
                Ok(edited) => edited,
                Err(_) => {
                    self.state_diagnostics.push(format!(
                        "token signal Telegram refresh failed chat_id={chat_id} signal_id={signal_id}"
                    ));
                    false
                }
            },
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "token signal photo refresh failed chat_id={chat_id} signal_id={signal_id}: {error}"
                ));
                false
            }
        };
        if !edited {
            if let Some(callback_id) = context.callback_id.as_deref() {
                let _receipt = self
                    .actions
                    .execute(TelegramAction::AnswerCallback {
                        callback_id: callback_id.to_owned(),
                        text: Some(signal_callback_text("refresh_failed", locale).to_owned()),
                        show_alert: true,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
        let mut refreshed_state = state;
        refreshed_state.last_refresh_at = Some(timestamp);
        if let Some(source) = self.token_signal_source.as_mut()
            && let Err(error) = source.save_state(signal_id, &refreshed_state)
        {
            self.state_diagnostics.push(format!(
                "token signal refresh state write failed chat_id={chat_id} signal_id={signal_id}: {error}"
            ));
        }
        if let Some(callback_id) = context.callback_id.as_deref() {
            let _receipt = self
                .actions
                .execute(TelegramAction::AnswerCallback {
                    callback_id: callback_id.to_owned(),
                    text: Some(signal_callback_text("refreshed", locale).to_owned()),
                    show_alert: false,
                })
                .map_err(DispatchError::Action)?;
        }
        Ok(DispatchOutcome::Handled)
    }

    fn dispatch_market_callback(
        &mut self,
        context: &CallbackContext,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        if self.market_price_source.is_none() {
            return Err(DispatchError::MissingService("market prices"));
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
        let mut parts = context.data.splitn(4, ':');
        let valid_prefix = parts.next() == Some("mkt");
        let action = parts.next().unwrap_or_default();
        let selection_id = parts.next().unwrap_or_default();
        let candidate_index = parts.next().and_then(|value| value.parse::<usize>().ok());
        if !valid_prefix
            || !matches!(action, "select" | "page" | "close")
            || selection_id.is_empty()
            || candidate_index.is_none()
        {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        }
        let candidate_index = candidate_index.unwrap_or_default();
        let key = market_selection_key(selection_id);
        let stored_value = self
            .market_price_source
            .as_mut()
            .map_or(Ok(None), |source| source.load_selection(&key));
        let stored_value = match stored_value {
            Ok(Some(value)) => value,
            Ok(None) => {
                self.answer_market_callback(context, locale, "expired", true)?;
                return Ok(DispatchOutcome::Handled);
            }
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "market selection read failed chat_id={} selection_id={selection_id}: {error}",
                    context.chat_id
                ));
                self.answer_market_callback(context, locale, "expired", true)?;
                return Ok(DispatchOutcome::Handled);
            }
        };
        let stored = match serde_json::from_str::<StoredMarketSelection>(&stored_value) {
            Ok(stored) => stored,
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "market selection decode failed chat_id={} selection_id={selection_id}: {error}",
                    context.chat_id
                ));
                self.answer_market_callback(context, locale, "expired", true)?;
                return Ok(DispatchOutcome::Handled);
            }
        };
        if stored.chat_id != context.chat_id {
            self.answer_market_callback(context, locale, "owner_only", true)?;
            return Ok(DispatchOutcome::Handled);
        }
        if stored.message_id != context.message_id {
            self.answer_market_callback(context, locale, "invalid", true)?;
            return Ok(DispatchOutcome::Handled);
        }
        let user_id = context.user_id;
        let mut allowed = user_id.is_some_and(|id| id == stored.requester_id);
        if !allowed && is_group_chat_type(Some(&context.chat_type)) {
            let authorization = user_id.map_or(
                GroupAuthorizationDecision {
                    is_admin: false,
                    diagnostics: Vec::new(),
                },
                |id| {
                    self.authorization
                        .authorize(&context.chat_id, &id.to_string())
                },
            );
            self.state_diagnostics.extend(authorization.diagnostics);
            allowed = authorization.is_admin;
        }
        if !allowed {
            self.answer_market_callback(context, locale, "owner_only", true)?;
            return Ok(DispatchOutcome::Handled);
        }
        let Ok(chat_id_value) = context.chat_id.parse::<i64>() else {
            self.state_diagnostics
                .push("invalid market callback chat id".to_owned());
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        };
        if action == "close" {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return self.clear_market_selection_callback(
                context,
                chat_id_value,
                &key,
                selection_id,
            );
        }
        if action == "page" {
            if candidate_index >= stored.selection.candidates.len().div_ceil(5) {
                self.answer_market_callback(context, locale, "invalid", true)?;
                return Ok(DispatchOutcome::Handled);
            }
            self.answer_callback_best_effort(context.callback_id.as_deref());
            self.actions
                .try_edit(TelegramAction::EditMessage {
                    chat_id: ChatId(chat_id_value),
                    message_id: MessageId(context.message_id),
                    text: market_selection_prompt(&stored.selection, locale),
                    reply_markup: Some(market_selection_page(
                        selection_id,
                        &stored.selection,
                        locale,
                        candidate_index,
                    )),
                })
                .map_err(DispatchError::Action)?;
            return Ok(DispatchOutcome::Handled);
        }
        // Atomically consume the menu before doing slow provider work. A
        // double-tapped button reaches this point twice across parallel
        // workers; exactly one taker observes the value and the loser gets the
        // expired toast below instead of sending a second quote.
        let taken = self
            .market_price_source
            .as_mut()
            .map_or(Ok(None), |source| source.take_selection(&key));
        let taken = match taken {
            Ok(Some(value)) => value,
            Ok(None) => {
                self.answer_market_callback(context, locale, "expired", true)?;
                return Ok(DispatchOutcome::Handled);
            }
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "market selection take failed chat_id={} selection_id={selection_id}: {error}",
                    context.chat_id
                ));
                self.answer_market_callback(context, locale, "expired", true)?;
                return Ok(DispatchOutcome::Handled);
            }
        };
        let stored = match serde_json::from_str::<StoredMarketSelection>(&taken) {
            Ok(stored) => stored,
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "market selection take decode failed chat_id={} selection_id={selection_id}: {error}",
                    context.chat_id
                ));
                self.answer_market_callback(context, locale, "expired", true)?;
                return Ok(DispatchOutcome::Handled);
            }
        };
        let Some(candidate) = stored.selection.candidates.get(candidate_index).cloned() else {
            self.restore_market_selection(&key, &taken, selection_id, &context.chat_id);
            self.answer_market_callback(context, locale, "invalid", true)?;
            return Ok(DispatchOutcome::Handled);
        };
        let timestamp = self.runtime_values.unix_timestamp();
        if candidate.id.starts_with("token:") {
            return self.dispatch_token_market_candidate(
                context,
                chat_id_value,
                &key,
                selection_id,
                &taken,
                &stored,
                &candidate,
                locale,
                timestamp,
            );
        }
        let command = market_selection_command(&stored.command);
        let mut load = self
            .market_price_source
            .as_mut()
            .map(|source| {
                source.load_candidate(
                    &candidate,
                    stored.selection.timeframe.as_deref(),
                    &stored.selection.target_symbol,
                    &stored.selection.target_parameter,
                    stored.selection.conversion.as_ref(),
                    command,
                    locale,
                    timestamp,
                )
            })
            .unwrap_or(MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: true,
                text: String::new(),
                diagnostics: vec!["market price source disappeared".to_owned()],
            });
        self.state_diagnostics.append(&mut load.diagnostics);
        let text = if load.text.trim().is_empty() {
            match locale {
                bot_core::locale::Locale::Es => {
                    format!("No pude conseguir una cotización para {}", candidate.symbol)
                }
                bot_core::locale::Locale::En => {
                    format!("I could not get a quote for {}", candidate.symbol)
                }
            }
        } else {
            load.text.clone()
        };
        let command_name = if command == MarketPriceCommand::CryptoOnly {
            "/c"
        } else {
            "/p"
        };
        let reply_to_message_id = stored.source_message_id.map(MessageId);
        if load.no_assets_found || load.text.trim().is_empty() {
            self.restore_market_selection(&key, &taken, selection_id, &context.chat_id);
            let mut reply = SendMessage::new(ChatId(chat_id_value), &text);
            reply.reply_to_message_id = reply_to_message_id;
            let receipt = self
                .actions
                .execute(TelegramAction::SendMessage(reply))
                .map_err(DispatchError::Action)?;
            self.record_market_callback_delivery(
                context,
                chat_id_value,
                &text,
                receipt.message_id,
                command_name,
                timestamp,
            );
            // The retry text was delivered; a toast failure must not retry
            // the update into a second copy of it.
            if let Err(error) = self.answer_market_callback(context, locale, "retry", true) {
                self.state_diagnostics.push(format!(
                    "market selection answer failed chat_id={} selection_id={selection_id}: {error}",
                    context.chat_id
                ));
            }
            return Ok(DispatchOutcome::Handled);
        }
        let mut delivered = false;
        let mut delivered_text = None;
        if let Some(chart) = load.chart.as_ref() {
            let rendered = match self.market_price_source.as_mut().map_or(
                Err("market price source disappeared".to_owned()),
                |source| source.render_chart(chart, timestamp),
            ) {
                Ok(rendered) => Some(rendered),
                Err(error) => {
                    self.state_diagnostics
                        .push(format!("market chart render failed: {error}"));
                    None
                }
            };
            if let Some(rendered) = rendered {
                let caption = rendered.caption.unwrap_or_else(|| text.clone());
                match self.actions.try_photo(TelegramAction::SendPhoto {
                    chat_id: ChatId(chat_id_value),
                    photo: rendered.photo.into(),
                    reply_to_message_id,
                    caption: caption.clone(),
                    parse_mode: None,
                    reply_markup: None,
                }) {
                    Ok(Some(receipt)) if receipt.message_id.is_some() => {
                        delivered = true;
                        delivered_text = Some((caption, receipt.message_id));
                    }
                    Ok(_) => self
                        .state_diagnostics
                        .push("market chart photo delivery was unconfirmed".to_owned()),
                    Err(_) => self
                        .state_diagnostics
                        .push("market chart photo delivery failed".to_owned()),
                }
            }
            if !delivered && let Some(token) = chart.token.as_ref() {
                let token_load = self
                    .token_signal_source
                    .as_mut()
                    .map(|source| source.load_token(token));
                if let Some(token_load) = token_load {
                    self.state_diagnostics.extend(token_load.diagnostics);
                    if let Some(signal) = token_load.signal {
                        let signal_id = stable_signal_id(
                            chat_id_value,
                            context.message_id,
                            stored.requester_id,
                            timestamp,
                        );
                        match self.try_send_token_signal_photo(TokenSignalPhotoRequest {
                            signal: &signal,
                            signal_id: &signal_id,
                            chat_id: ChatId(chat_id_value),
                            reply_to_message_id,
                            timeframe: stored.selection.timeframe.as_deref(),
                            locale,
                            timestamp,
                        }) {
                            TokenSignalPhotoDelivery::Sent {
                                caption,
                                message_id,
                            } => {
                                delivered = true;
                                delivered_text = Some((caption, Some(message_id)));
                                let state = Self::token_signal_state(
                                    &signal,
                                    stored.selection.timeframe.clone(),
                                    chat_id_value,
                                    message_id,
                                    stored.source_message_id.unwrap_or(context.message_id),
                                    stored.requester_id,
                                );
                                self.save_token_signal_state(&signal_id, &state);
                            }
                            TokenSignalPhotoDelivery::Failed { failure, .. } => match failure {
                                TokenSignalPhotoFailure::HistoryUnavailable => {}
                                TokenSignalPhotoFailure::DeliverySkipped
                                | TokenSignalPhotoFailure::DeliveryUnconfirmed => self
                                    .state_diagnostics
                                    .push("token chart photo delivery was unconfirmed".to_owned()),
                                TokenSignalPhotoFailure::DeliveryFailed => self
                                    .state_diagnostics
                                    .push("token chart photo delivery failed".to_owned()),
                            },
                        }
                    }
                }
            }
        }
        if !delivered {
            let chat_id = ChatId(chat_id_value);
            let mut reply = SendMessage::new(chat_id, &text);
            reply.reply_to_message_id = reply_to_message_id;
            let receipt = match self.actions.execute(TelegramAction::SendMessage(reply)) {
                Ok(receipt) => receipt,
                Err(error) => {
                    self.restore_market_selection(&key, &taken, selection_id, &context.chat_id);
                    return Err(DispatchError::Action(error));
                }
            };
            self.record_market_callback_delivery(
                context,
                chat_id_value,
                &text,
                receipt.message_id,
                command_name,
                timestamp,
            );
        } else if let Some((caption, sent_message_id)) = delivered_text {
            self.record_market_callback_delivery(
                context,
                chat_id_value,
                &caption,
                sent_message_id,
                command_name,
                timestamp,
            );
        }
        self.clear_market_selection_callback(context, chat_id_value, &key, selection_id)?;
        // The quote was delivered; a toast failure must not retry the update
        // into a second quote.
        if let Err(error) = self.answer_market_callback(
            context,
            locale,
            if delivered { "selected" } else { "quote" },
            false,
        ) {
            self.state_diagnostics.push(format!(
                "market selection answer failed chat_id={} selection_id={selection_id}: {error}",
                context.chat_id
            ));
        }
        Ok(DispatchOutcome::Handled)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_token_market_candidate(
        &mut self,
        context: &CallbackContext,
        chat_id_value: i64,
        selection_key: &str,
        selection_id: &str,
        taken: &str,
        stored: &StoredMarketSelection,
        candidate: &MarketCandidate,
        locale: bot_core::locale::Locale,
        timestamp: i64,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let Some(token) = candidate.contracts.first().cloned() else {
            self.restore_market_selection(selection_key, taken, selection_id, &context.chat_id);
            self.answer_market_callback(context, locale, "retry", true)?;
            return Ok(DispatchOutcome::Handled);
        };
        let command_name = if stored.command == "crypto" {
            "/c"
        } else {
            "/p"
        };
        let reply_to_message_id = stored.source_message_id.map(MessageId);
        let load = self.token_signal_source.as_mut().map_or(
            TokenSignalLoad {
                signal: None,
                diagnostics: vec!["token signal source disappeared".to_owned()],
            },
            |source| source.load_token(&token),
        );
        self.state_diagnostics.extend(load.diagnostics);
        let Some(signal) = load.signal else {
            let text = match locale {
                bot_core::locale::Locale::Es => {
                    format!("No pude conseguir una cotización para {}", candidate.symbol)
                }
                bot_core::locale::Locale::En => {
                    format!("I could not get a quote for {}", candidate.symbol)
                }
            };
            let mut reply = SendMessage::new(ChatId(chat_id_value), &text);
            reply.reply_to_message_id = reply_to_message_id;
            self.restore_market_selection(selection_key, taken, selection_id, &context.chat_id);
            let receipt = self
                .actions
                .execute(TelegramAction::SendMessage(reply))
                .map_err(DispatchError::Action)?;
            self.record_market_callback_delivery(
                context,
                chat_id_value,
                &text,
                receipt.message_id,
                command_name,
                timestamp,
            );
            // The retry text was delivered; a toast failure must not retry
            // the update into a second copy of it.
            if let Err(error) = self.answer_market_callback(context, locale, "retry", true) {
                self.state_diagnostics.push(format!(
                    "market selection answer failed chat_id={} selection_id={selection_id}: {error}",
                    context.chat_id
                ));
            }
            return Ok(DispatchOutcome::Handled);
        };
        let timeframe = stored.selection.timeframe.as_deref();
        let signal_id = stable_signal_id(
            chat_id_value,
            context.message_id,
            stored.requester_id,
            timestamp,
        );
        let (delivered, delivered_text, sent_message_id) =
            match self.try_send_token_signal_photo(TokenSignalPhotoRequest {
                signal: &signal,
                signal_id: &signal_id,
                chat_id: ChatId(chat_id_value),
                reply_to_message_id,
                timeframe,
                locale,
                timestamp,
            }) {
                TokenSignalPhotoDelivery::Sent {
                    caption,
                    message_id,
                } => (true, caption, Some(message_id)),
                TokenSignalPhotoDelivery::Failed { caption, .. } => (false, caption, None),
            };
        if delivered {
            self.record_market_callback_delivery(
                context,
                chat_id_value,
                &delivered_text,
                sent_message_id,
                command_name,
                timestamp,
            );
            if let Some(sent_message_id) = sent_message_id {
                let state = Self::token_signal_state(
                    &signal,
                    stored.selection.timeframe.clone(),
                    chat_id_value,
                    sent_message_id,
                    stored.source_message_id.unwrap_or(context.message_id),
                    stored.requester_id,
                );
                self.save_token_signal_state(&signal_id, &state);
            }
        } else {
            let mut reply = SendMessage::new(ChatId(chat_id_value), &delivered_text);
            reply.reply_to_message_id = reply_to_message_id;
            reply.parse_mode = Some(ParseMode::Html);
            let receipt = match self.actions.execute(TelegramAction::SendMessage(reply)) {
                Ok(receipt) => receipt,
                Err(error) => {
                    self.restore_market_selection(
                        selection_key,
                        taken,
                        selection_id,
                        &context.chat_id,
                    );
                    return Err(DispatchError::Action(error));
                }
            };
            self.record_market_callback_delivery(
                context,
                chat_id_value,
                &delivered_text,
                receipt.message_id,
                command_name,
                timestamp,
            );
        }
        self.clear_market_selection_callback(context, chat_id_value, selection_key, selection_id)?;
        // The quote was delivered; a toast failure must not retry the update
        // into a second quote.
        if let Err(error) = self.answer_market_callback(
            context,
            locale,
            if delivered { "selected" } else { "quote" },
            false,
        ) {
            self.state_diagnostics.push(format!(
                "market selection answer failed chat_id={} selection_id={selection_id}: {error}",
                context.chat_id
            ));
        }
        Ok(DispatchOutcome::Handled)
    }

    fn clear_market_selection_callback(
        &mut self,
        context: &CallbackContext,
        chat_id_value: i64,
        selection_key: &str,
        selection_id: &str,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        if let Err(error) = self
            .market_price_source
            .as_mut()
            .map_or(Ok(()), |source| source.clear_selection(selection_key))
        {
            self.state_diagnostics.push(format!(
                "market selection clear failed chat_id={} selection_id={selection_id}: {error}",
                context.chat_id
            ));
        }
        let _ = self.actions.execute(TelegramAction::DeleteMessage {
            chat_id: ChatId(chat_id_value),
            message_id: MessageId(context.message_id),
        });
        Ok(DispatchOutcome::Handled)
    }

    fn answer_market_callback(
        &mut self,
        context: &CallbackContext,
        locale: bot_core::locale::Locale,
        kind: &str,
        show_alert: bool,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let Some(callback_id) = context.callback_id.as_deref() else {
            return Ok(DispatchOutcome::Handled);
        };
        let text = match (kind, locale) {
            ("expired", bot_core::locale::Locale::Es) => "Esta selección ya venció",
            ("expired", bot_core::locale::Locale::En) => "This selection expired",
            ("owner_only", bot_core::locale::Locale::Es) => "Esa selección es de otra persona",
            ("owner_only", bot_core::locale::Locale::En) => {
                "That selection belongs to someone else"
            }
            ("invalid", bot_core::locale::Locale::Es) => "Esa opción no es válida",
            ("invalid", bot_core::locale::Locale::En) => "That option is not valid",
            ("selected", bot_core::locale::Locale::Es) => "Cotización cargada",
            ("selected", bot_core::locale::Locale::En) => "Quote loaded",
            ("quote", bot_core::locale::Locale::Es) => "Te dejé la cotización",
            ("quote", bot_core::locale::Locale::En) => "Showing the quote",
            // "retry", the only remaining kind callers pass.
            (_, bot_core::locale::Locale::Es) => "No pude cargarla. Probá de nuevo",
            (_, bot_core::locale::Locale::En) => "I could not load it. Try again",
        };
        let _receipt = self
            .actions
            .execute(TelegramAction::AnswerCallback {
                callback_id: callback_id.to_owned(),
                text: Some(text.to_owned()),
                show_alert,
            })
            .map_err(DispatchError::Action)?;
        Ok(DispatchOutcome::Handled)
    }

    fn dispatch_task_callback(
        &mut self,
        context: &CallbackContext,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        if self.scheduled_task_source.is_none() {
            return Err(DispatchError::MissingService("scheduled tasks"));
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
        // Either a task list page or a task with its detail mode.
        let request = match parse_task_callback(&context.data) {
            TaskCallbackParse::Close => {
                self.answer_callback_best_effort(context.callback_id.as_deref());
                if let Ok(chat_id) = context.chat_id.parse::<i64>() {
                    // Best effort: see the signal delete handler above.
                    if self
                        .actions
                        .execute(TelegramAction::DeleteMessage {
                            chat_id: ChatId(chat_id),
                            message_id: MessageId(context.message_id),
                        })
                        .is_err()
                    {
                        self.state_diagnostics.push(format!(
                            "callback delete failed chat_id={} message_id={}",
                            context.chat_id, context.message_id
                        ));
                    }
                }
                return Ok(DispatchOutcome::Handled);
            }
            TaskCallbackParse::Guard => {
                self.answer_callback_best_effort(context.callback_id.as_deref());
                return Ok(DispatchOutcome::Handled);
            }
            TaskCallbackParse::Page(page) => Err(page),
            TaskCallbackParse::Delete(id) => Ok((id, None)),
            TaskCallbackParse::View(id) => Ok((id, Some(false))),
            TaskCallbackParse::Confirm(id) => Ok((id, Some(true))),
        };
        let missing = DispatchError::MissingService("scheduled tasks");
        let source = self.scheduled_task_source.as_mut().ok_or(missing)?;
        let tasks = match source.list(&context.chat_id) {
            Ok(tasks) => tasks,
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "scheduled task list callback chat_id={}: {error}",
                    context.chat_id
                ));
                if let Some(callback_id) = context.callback_id.as_deref() {
                    let _receipt = self
                        .actions
                        .execute(TelegramAction::AnswerCallback {
                            callback_id: callback_id.to_owned(),
                            text: Some(task_load_failed(locale).to_owned()),
                            show_alert: true,
                        })
                        .map_err(DispatchError::Action)?;
                }
                return Ok(DispatchOutcome::Handled);
            }
        };
        let (task_id, detail) = match request {
            Err(page) => {
                let view = bot_core::task_commands::render_task_page(&tasks, locale, page);
                self.answer_callback_best_effort(context.callback_id.as_deref());
                if let Ok(chat_id) = context.chat_id.parse::<i64>() {
                    self.actions
                        .try_edit(TelegramAction::EditMessage {
                            chat_id: ChatId(chat_id),
                            message_id: MessageId(context.message_id),
                            text: view.text,
                            reply_markup: view.keyboard,
                        })
                        .map_err(DispatchError::Action)?;
                }
                return Ok(DispatchOutcome::Handled);
            }
            Ok(target) => target,
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
        if let Some(confirm) = detail {
            let view = bot_core::task_commands::render_task_detail(&target, locale, confirm);
            self.answer_callback_best_effort(context.callback_id.as_deref());
            if let Ok(chat_id) = context.chat_id.parse::<i64>() {
                self.actions
                    .try_edit(TelegramAction::EditMessage {
                        chat_id: ChatId(chat_id),
                        message_id: MessageId(context.message_id),
                        text: view.text,
                        reply_markup: view.keyboard,
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
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
        match source.cancel(&task_id, &context.chat_id) {
            Ok(true) => {}
            Ok(false) => {
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
            }
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "scheduled task cancellation chat_id={} task_id={}: {error}",
                    context.chat_id,
                    task_id.as_str()
                ));
                if let Some(callback_id) = context.callback_id.as_deref() {
                    let _receipt = self
                        .actions
                        .execute(TelegramAction::AnswerCallback {
                            callback_id: callback_id.to_owned(),
                            text: Some(task_delete_failed(locale).to_owned()),
                            show_alert: true,
                        })
                        .map_err(DispatchError::Action)?;
                }
                return Ok(DispatchOutcome::Handled);
            }
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
        let tasks = match source.list(&context.chat_id) {
            Ok(tasks) => tasks,
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "scheduled task list after cancellation chat_id={}: {error}",
                    context.chat_id
                ));
                return Ok(DispatchOutcome::Handled);
            }
        };
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
            self.answer_callback_best_effort(callback.get("id").and_then(Value::as_str));
            if let Err(error) = parsed {
                self.state_diagnostics
                    .push(format!("invalid callback query: {error}"));
            }
            return Ok(DispatchOutcome::Handled);
        };
        if is_group_chat_type(Some(&context.chat_type))
            && let (Ok(chat_id), Some(user_id)) = (context.chat_id.parse::<i64>(), context.user_id)
            && self.sender_is_banned(ChatId(chat_id), user_id)
        {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        }
        if context.data == "topup:close" || context.data.starts_with("chg:close:") {
            let allowed = if context.data == "topup:close" {
                context.chat_type == "private"
            } else {
                context
                    .data
                    .strip_prefix("chg:close:")
                    .and_then(|id| id.parse::<i64>().ok())
                    .is_some_and(|id| context.user_id == Some(id))
            };
            self.answer_callback_best_effort(context.callback_id.as_deref());
            if allowed && let Ok(chat_id) = context.chat_id.parse::<i64>() {
                // Best effort: a double-tapped close deletes an already gone
                // message, which must not fail the update into retries.
                if self
                    .actions
                    .execute(TelegramAction::DeleteMessage {
                        chat_id: ChatId(chat_id),
                        message_id: MessageId(context.message_id),
                    })
                    .is_err()
                {
                    self.state_diagnostics.push(format!(
                        "callback delete failed chat_id={} message_id={}",
                        context.chat_id, context.message_id
                    ));
                }
            }
            return Ok(DispatchOutcome::Handled);
        }
        if let Some(page) = context.data.strip_prefix("help:") {
            let Ok(id) = context.chat_id.parse::<i64>() else {
                self.answer_callback_best_effort(context.callback_id.as_deref());
                return Ok(DispatchOutcome::Handled);
            };
            self.answer_callback_best_effort(context.callback_id.as_deref());
            if page == "close" {
                // Best effort: see the signal delete handler above.
                if self
                    .actions
                    .execute(TelegramAction::DeleteMessage {
                        chat_id: ChatId(id),
                        message_id: MessageId(context.message_id),
                    })
                    .is_err()
                {
                    self.state_diagnostics.push(format!(
                        "callback delete failed chat_id={} message_id={}",
                        context.chat_id, context.message_id
                    ));
                }
            } else {
                let config = self
                    .config
                    .get(&context.chat_id)
                    .map_err(DispatchError::Config)?;
                let locale = resolve_locale(
                    Some(&config.language),
                    context.user_language_code.as_deref(),
                    &context.chat_type,
                );
                let (text, keyboard) = bot_core::help_catalog::render_help_page(locale, page);
                self.actions
                    .try_edit(TelegramAction::EditMessage {
                        chat_id: ChatId(id),
                        message_id: MessageId(context.message_id),
                        text,
                        reply_markup: Some(keyboard),
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
        if context.route == CallbackRoute::Task {
            return self.dispatch_task_callback(&context);
        }
        if context.route == CallbackRoute::Signal {
            return self.dispatch_token_signal_callback(&context);
        }
        if context.route == CallbackRoute::Market {
            return self.dispatch_market_callback(&context);
        }
        if context.route == CallbackRoute::Topup {
            let Ok(chat_id) = context.chat_id.parse::<i64>() else {
                self.answer_callback_best_effort(context.callback_id.as_deref());
                self.state_diagnostics
                    .push("invalid top-up callback chat id".to_owned());
                return Ok(DispatchOutcome::Handled);
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
            if let Some(callback) = parse_lightning_callback(&context.data) {
                return self.dispatch_lightning_callback(
                    &context,
                    ChatId(chat_id),
                    callback,
                    locale,
                );
            }
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
                TopupCallbackPlan::Menu(credits) => {
                    self.answer_callback_best_effort(context.callback_id.as_deref());
                    let (text, keyboard) =
                        topup_menu(locale, credits, self.lightning_checkout.is_some());
                    // Tapping the amount already shown changes nothing, and
                    // Telegram refuses that edit, so its result is ignored.
                    let _edited = self
                        .actions
                        .try_edit(TelegramAction::EditMessage {
                            chat_id: ChatId(chat_id),
                            message_id: MessageId(context.message_id),
                            text,
                            reply_markup: Some(keyboard),
                        })
                        .map_err(DispatchError::Action)?;
                    Ok(DispatchOutcome::Handled)
                }
                TopupCallbackPlan::Invoice(plan) => {
                    // One in-flight invoice per user and pack: a double tap
                    // must not emit two independently payable invoices. The
                    // marker expires on its own, so an intentional repurchase
                    // stays possible. The plan guarantees a user and pack id
                    // on this path, so the key below always identifies the tap.
                    let user_id = context.user_id.unwrap_or(0);
                    let pack_id = context.data.strip_prefix("topup:").unwrap_or_default();
                    let claim_key = topup_invoice_claim_key(user_id, pack_id);
                    let claimed = self
                        .market_price_source
                        .as_mut()
                        .map_or(Ok(true), |source| {
                            source.claim(claim_key.as_str(), "1", TOPUP_INVOICE_CLAIM_TTL_SECONDS)
                        });
                    let claimed = match claimed {
                        Ok(claimed) => claimed,
                        Err(error) => {
                            self.state_diagnostics.push(format!(
                                "topup invoice claim failed chat_id={} user_id={}: {error}",
                                context.chat_id,
                                context
                                    .user_id
                                    .map_or_else(String::new, |value| value.to_string()),
                            ));
                            true
                        }
                    };
                    if !claimed {
                        let text = match locale {
                            bot_core::locale::Locale::Es => "Ya te dejé la factura más arriba",
                            bot_core::locale::Locale::En => "The invoice is already above",
                        };
                        if let Some(callback_id) = context.callback_id.as_deref() {
                            let _receipt = self
                                .actions
                                .execute(TelegramAction::AnswerCallback {
                                    callback_id: callback_id.to_owned(),
                                    text: Some(text.to_owned()),
                                    show_alert: false,
                                })
                                .map_err(DispatchError::Action)?;
                        }
                        return Ok(DispatchOutcome::Handled);
                    }
                    let release_claim = |dispatcher: &mut Self| {
                        if let Some(source) = dispatcher.market_price_source.as_mut()
                            && let Err(error) = source.take_selection(claim_key.as_str())
                        {
                            dispatcher.state_diagnostics.push(format!(
                                "topup invoice claim release failed chat_id={} key={claim_key}: {error}",
                                context.chat_id,
                            ));
                        }
                    };
                    let sent = match self.actions.try_invoice(plan.invoice) {
                        Ok(sent) => sent,
                        Err(error) => {
                            release_claim(self);
                            return Err(DispatchError::Action(error));
                        }
                    };
                    if !sent {
                        release_claim(self);
                    }
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
                self.answer_callback_best_effort(context.callback_id.as_deref());
                self.state_diagnostics
                    .push("invalid charge-history callback chat id".to_owned());
                return Ok(DispatchOutcome::Handled);
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
                return Err(DispatchError::MissingService("charge history"));
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
                            bot_core::locale::Locale::Es => {
                                "Se trabó leyendo tus gastos. Probá de nuevo"
                            }
                            bot_core::locale::Locale::En => {
                                "I could not load your spending. Try again"
                            }
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
                        bot_core::locale::Locale::Es => "No hay más gastos para mostrar",
                        bot_core::locale::Locale::En => "There is no more spending to show",
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
                            bot_core::locale::Locale::Es => {
                                "Se trabó leyendo tus gastos. Probá de nuevo"
                            }
                            bot_core::locale::Locale::En => {
                                "I could not load your spending. Try again"
                            }
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
                    bot_core::locale::Locale::Es => {
                        "No pude actualizar el historial. Probá de nuevo"
                    }
                    bot_core::locale::Locale::En => "I could not update the history. Try again",
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
            self.answer_callback_best_effort(context.callback_id.as_deref());
            return Ok(DispatchOutcome::Handled);
        }
        let Ok(chat_id_value) = context.chat_id.parse::<i64>() else {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            self.state_diagnostics
                .push("invalid configuration callback chat id".to_owned());
            return Ok(DispatchOutcome::Handled);
        };
        let chat_id = ChatId(chat_id_value);
        let message_id = MessageId(context.message_id);
        let current_config = self
            .config
            .get(&context.chat_id)
            .map_err(DispatchError::Config)?;
        let locale = resolve_locale(
            Some(&current_config.language),
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
                // A toast answers the tapper's own callback: a double tap
                // shows the warning twice on their screen instead of
                // delivering a duplicate message to the chat.
                if let Some(callback_id) = context.callback_id.as_deref() {
                    let text = match locale {
                        bot_core::locale::Locale::Es => {
                            "Este comando es solo para admins del grupo"
                        }
                        bot_core::locale::Locale::En => "Only group admins can use this command",
                    };
                    let _receipt = self
                        .actions
                        .execute(TelegramAction::AnswerCallback {
                            callback_id: callback_id.to_owned(),
                            text: Some(text.to_owned()),
                            show_alert: true,
                        })
                        .map_err(DispatchError::Action)?;
                }
                return Ok(DispatchOutcome::Handled);
            }
        }
        if let Some(page) = context.data.strip_prefix("cfg:page:") {
            self.answer_callback_best_effort(context.callback_id.as_deref());
            if page == "close" {
                self.actions
                    .execute(TelegramAction::DeleteMessage {
                        chat_id,
                        message_id,
                    })
                    .map_err(DispatchError::Action)?;
            } else {
                let (text, keyboard) = render_config_page(&current_config, locale, is_group, page);
                self.actions
                    .try_edit(TelegramAction::EditMessage {
                        chat_id,
                        message_id,
                        text,
                        reply_markup: Some(keyboard),
                    })
                    .map_err(DispatchError::Action)?;
            }
            return Ok(DispatchOutcome::Handled);
        }
        let (outcome, config) = plan_config_callback(&context.data, &current_config);
        let (changed, diagnostic) = match outcome {
            ConfigCallbackOutcome::Render {
                changed,
                diagnostic,
            } => (changed, diagnostic),
            // Config routes always carry the `cfg:` prefix, so "not handled"
            // only mirrors a guard here.
            ConfigCallbackOutcome::Guard | ConfigCallbackOutcome::NotHandled => {
                self.answer_callback_best_effort(context.callback_id.as_deref());
                return Ok(DispatchOutcome::Handled);
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
                .set_changed(&context.chat_id, &current_config, &config)
                .map_err(DispatchError::Config)?;
        }
        let rendered_locale = resolve_locale(
            Some(&config.language),
            context.user_language_code.as_deref(),
            &context.chat_type,
        );
        let (rendered_text, rendered_markup) = render_config_page(
            &config,
            rendered_locale,
            is_group,
            context.data.split(':').nth(1).unwrap_or("home"),
        );
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
        // Re-tapping the selected option leaves the menu identical, which
        // Telegram rejects as an edit; that must not post a duplicate menu.
        let fallback = if edited || config == current_config {
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
        if config.language != current_config.language {
            self.sync_chat_command_menu(chat_id, rendered_locale);
        }
        Ok(DispatchOutcome::Handled)
    }

    fn dispatch_lightning_callback(
        &mut self,
        context: &CallbackContext,
        chat_id: ChatId,
        callback: LightningCallback,
        locale: bot_core::locale::Locale,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let callback_id = context.callback_id.as_deref();
        let alert_text = if !self.billing_available {
            Some(bot_core::billing_commands::billing_unavailable(locale))
        } else if context.chat_type != "private" {
            Some(match locale {
                bot_core::locale::Locale::Es => "Cargá por privado, maestro",
                bot_core::locale::Locale::En => "Open this in a private chat",
            })
        } else if self.lightning_checkout.is_none() {
            Some(lightning_invoice_failed(locale))
        } else if callback == LightningCallback::InvalidPack {
            Some(match locale {
                bot_core::locale::Locale::Es => "Ese pack es fruta, elegí otro",
                bot_core::locale::Locale::En => "That credit pack is invalid, choose another one",
            })
        } else {
            None
        };
        if let Some(text) = alert_text {
            return self.answer_callback_alert(callback_id, text);
        }
        let message_id = MessageId(context.message_id);
        let (pack, user_id) = match (callback, context.user_id) {
            (LightningCallback::Pack(pack), Some(user_id)) => (pack, user_id),
            (LightningCallback::Pack(_), None) => {
                self.answer_callback_best_effort(callback_id);
                return Ok(DispatchOutcome::Handled);
            }
            // The old pack lists' menu buttons open the amount menu.
            (_, _) => {
                self.answer_callback_best_effort(callback_id);
                let (text, keyboard) = topup_menu(locale, DEFAULT_TOPUP_CREDITS, true);
                self.actions
                    .try_edit(TelegramAction::EditMessage {
                        chat_id,
                        message_id,
                        text,
                        reply_markup: Some(keyboard),
                    })
                    .map_err(DispatchError::Action)?;
                return Ok(DispatchOutcome::Handled);
            }
        };
        // Same single-flight guard as Stars invoices: a double tap must not
        // create two payable charges.
        let claim_key = topup_invoice_claim_key(user_id, &format!("ln:{}", pack.id));
        let claimed = self
            .market_price_source
            .as_mut()
            .map_or(Ok(true), |source| {
                source.claim(claim_key.as_str(), "1", TOPUP_INVOICE_CLAIM_TTL_SECONDS)
            })
            .unwrap_or_else(|error| {
                self.state_diagnostics.push(format!(
                    "lightning invoice claim failed user_id={user_id}: {error}"
                ));
                true
            });
        if !claimed {
            return self.answer_callback_alert(
                callback_id,
                match locale {
                    bot_core::locale::Locale::Es => "Ya te dejé la factura más arriba",
                    bot_core::locale::Locale::En => "The invoice is already above",
                },
            );
        }
        let created = self
            .lightning_checkout
            .as_mut()
            .map(|checkout| checkout.create(user_id, chat_id.0, &pack, locale));
        let Some(Ok(invoice)) = created else {
            if let Some(Err(error)) = created {
                self.state_diagnostics.push(format!(
                    "lightning invoice user_id={user_id} pack={}: {error}",
                    pack.id
                ));
            }
            if let Some(source) = self.market_price_source.as_mut()
                && let Err(error) = source.take_selection(claim_key.as_str())
            {
                self.state_diagnostics
                    .push(format!("lightning invoice claim release failed: {error}"));
            }
            return self.answer_callback_alert(callback_id, lightning_invoice_failed(locale));
        };
        let message = lightning_invoice_message(chat_id, &pack, &invoice, locale);
        // A QR code to scan from another device; plain text if Telegram
        // refuses the photo.
        let photo_receipt = bot_adapters::qr_code::lightning_invoice_png(&invoice.payreq)
            .map(|png| {
                self.actions.try_photo(TelegramAction::SendPhoto {
                    chat_id,
                    photo: png.into(),
                    reply_to_message_id: None,
                    caption: message.text.clone(),
                    parse_mode: message.parse_mode,
                    reply_markup: message.reply_markup.clone(),
                })
            })
            .transpose()
            .map_err(DispatchError::Action)?
            .flatten();
        let receipt = match photo_receipt {
            Some(receipt) => receipt,
            None => self
                .actions
                .execute(TelegramAction::SendMessage(message))
                .map_err(DispatchError::Action)?,
        };
        if let (Some(sent), Some(checkout)) = (receipt.message_id, self.lightning_checkout.as_mut())
            && let Err(error) = checkout.attach_message(&invoice.charge_id, sent.0)
        {
            self.state_diagnostics.push(format!(
                "lightning invoice message charge_id={}: {error}",
                invoice.charge_id
            ));
        }
        if let Some(callback_id) = callback_id {
            self.actions
                .execute(TelegramAction::AnswerCallback {
                    callback_id: callback_id.to_owned(),
                    text: Some(lightning_invoice_ready(locale).to_owned()),
                    show_alert: false,
                })
                .map_err(DispatchError::Action)?;
        }
        Ok(DispatchOutcome::Handled)
    }

    fn answer_callback_alert(
        &mut self,
        callback_id: Option<&str>,
        text: &str,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        if let Some(callback_id) = callback_id {
            self.actions
                .execute(TelegramAction::AnswerCallback {
                    callback_id: callback_id.to_owned(),
                    text: Some(text.to_owned()),
                    show_alert: true,
                })
                .map_err(DispatchError::Action)?;
        }
        Ok(DispatchOutcome::Handled)
    }

    /// Admins are never treated as banned, so a ban left from before they
    /// were promoted can't lock them out. A failed lookup lets the message
    /// through rather than silencing the group.
    fn sender_is_banned(&mut self, chat_id: ChatId, user_id: i64) -> bool {
        let Some(store) = self.ban_store.as_mut() else {
            return false;
        };
        match store.is_banned(chat_id.0, user_id) {
            Ok(false) => false,
            Ok(true) => {
                let authorization = self
                    .authorization
                    .authorize(&chat_id.0.to_string(), &user_id.to_string());
                self.state_diagnostics.extend(authorization.diagnostics);
                !authorization.is_admin
            }
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "chat ban check chat_id={} user_id={user_id}: {error}",
                    chat_id.0
                ));
                false
            }
        }
    }

    /// The member a ban or limit command is about: the one it names, either
    /// picked from Telegram's mention list or by @username looked up among
    /// the members the bot has seen write, or else the one it replies to.
    /// `Err` is the reply for a username the bot can't resolve.
    fn moderation_target(
        &mut self,
        message: &IncomingMessage,
        chat_id: ChatId,
        named: Option<NamedMember<'_>>,
        locale: bot_core::locale::Locale,
    ) -> Result<Option<BanTarget>, String> {
        let username = match named {
            None => return Ok(ban_target(message, locale)),
            Some(NamedMember::Picked(mention)) => return Ok(Some(picked_member_target(mention))),
            Some(NamedMember::Username(username)) => username,
        };
        let entries = match self.member_source.as_mut() {
            Some(source) => source.members(&chat_id.0.to_string()).map_err(|error| {
                self.state_diagnostics
                    .push(format!("chat members chat_id={}: {error}", chat_id.0));
                mention_lookup_failed_reply(username, locale)
            })?,
            None => Vec::new(),
        };
        let members = bot_core::chat_members::decode_chat_members(&entries);
        find_member_by_username(&members, username)
            .and_then(known_member_target)
            .map(Some)
            .ok_or_else(|| unknown_mention_reply(username, locale))
    }

    fn dispatch_ban_command(
        &mut self,
        message: &IncomingMessage,
        (chat_id, message_id, sender_id): (ChatId, MessageId, UserId),
        command: BanCommand,
        argument: &str,
        locale: bot_core::locale::Locale,
        is_group: bool,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let (named, _) = if command == BanCommand::List {
            (None, "")
        } else {
            split_named_member(argument, &message.text_mentions)
        };
        let target = self.moderation_target(message, chat_id, named, locale);
        let Some(store) = self.ban_store.as_mut() else {
            return Err(DispatchError::MissingService("chat bans"));
        };
        let chat = chat_id.0.to_string();
        let reply = |text: &str| ban_reply(chat_id, message_id, text);
        let authorization = if is_group {
            let authorization = self
                .authorization
                .authorize(&chat, &sender_id.0.to_string());
            self.state_diagnostics.extend(authorization.diagnostics);
            Some(authorization.is_admin)
        } else {
            None
        };
        let action = match authorization {
            None => reply(bans_group_only(locale)),
            Some(false) => {
                self.state_diagnostics.push(format!(
                    "Unauthorized ban attempt chat_id={chat} user_id={}",
                    sender_id.0
                ));
                reply(match locale {
                    bot_core::locale::Locale::Es => "Este comando es solo para admins del grupo",
                    bot_core::locale::Locale::En => "Only group admins can use this command",
                })
            }
            Some(true) if target.is_err() => reply(&target.err().unwrap_or_default()),
            Some(true) => {
                let context = BanCommandContext {
                    chat_id,
                    message_id,
                    sender_id: sender_id.0,
                    locale,
                    target: target.ok().flatten(),
                };
                match plan_ban_command(command, context) {
                    BanCommandPlan::Reply(action) => action,
                    BanCommandPlan::List => match store.list(chat_id.0) {
                        Ok(users) => reply(&render_ban_list(&users, locale)),
                        Err(error) => {
                            self.state_diagnostics
                                .push(format!("chat ban list chat_id={chat}: {error}"));
                            reply(ban_list_failed(locale))
                        }
                    },
                    BanCommandPlan::Ban { user_id, name } => {
                        let target_authorization =
                            self.authorization.authorize(&chat, &user_id.to_string());
                        self.state_diagnostics
                            .extend(target_authorization.diagnostics);
                        if target_authorization.is_admin {
                            reply(ban_admin_target(locale))
                        } else {
                            match store.ban(chat_id.0, user_id, &name, sender_id.0) {
                                Ok(inserted) => reply(&ban_result_reply(&name, inserted, locale)),
                                Err(error) => {
                                    self.state_diagnostics.push(format!(
                                        "chat ban chat_id={chat} user_id={user_id}: {error}"
                                    ));
                                    reply(ban_store_failed(locale))
                                }
                            }
                        }
                    }
                    BanCommandPlan::Unban { user_id, name } => {
                        match store.unban(chat_id.0, user_id) {
                            Ok(removed) => reply(&unban_result_reply(&name, removed, locale)),
                            Err(error) => {
                                self.state_diagnostics.push(format!(
                                    "chat unban chat_id={chat} user_id={user_id}: {error}"
                                ));
                                reply(ban_store_failed(locale))
                            }
                        }
                    }
                }
            }
        };
        let _receipt = self
            .actions
            .execute(action)
            .map_err(DispatchError::Action)?;
        Ok(DispatchOutcome::Handled)
    }

    /// A member's own limit replaces the group's for AI messages the group
    /// pays for. Admins keep the group's limit, so a limit left from before
    /// they were promoted can't hold them back, and a failed lookup keeps it
    /// too rather than blocking the member.
    fn creditless_limit(
        &mut self,
        message: &IncomingMessage,
        (chat_id, sender_id): (ChatId, UserId),
        config: &ChatConfig,
    ) -> CreditlessLimit {
        let group_limit = CreditlessLimit::Group(config.creditless_user_hourly_limit);
        if !is_group_chat_type(message.chat_type.as_deref()) {
            return group_limit;
        }
        let Some(store) = self.limit_store.as_mut() else {
            return group_limit;
        };
        match store.hourly_limit(chat_id.0, sender_id.0) {
            Ok(None) => group_limit,
            Ok(Some(own_limit)) => {
                let authorization = self
                    .authorization
                    .authorize(&chat_id.0.to_string(), &sender_id.0.to_string());
                self.state_diagnostics.extend(authorization.diagnostics);
                if authorization.is_admin {
                    group_limit
                } else {
                    CreditlessLimit::Member(own_limit)
                }
            }
            Err(error) => {
                self.state_diagnostics.push(format!(
                    "chat limit check chat_id={} user_id={}: {error}",
                    chat_id.0, sender_id.0
                ));
                group_limit
            }
        }
    }

    fn dispatch_group_charges_command(
        &mut self,
        (chat_id, message_id, sender_id): (ChatId, MessageId, UserId),
        argument: &str,
        locale: bot_core::locale::Locale,
        is_group: bool,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let Some(source) = self.group_spending_source.as_mut() else {
            return Err(DispatchError::MissingService("group charges"));
        };
        let chat = chat_id.0.to_string();
        let reply = |text: &str| ban_reply(chat_id, message_id, text);
        let authorization = if is_group {
            let authorization = self
                .authorization
                .authorize(&chat, &sender_id.0.to_string());
            self.state_diagnostics.extend(authorization.diagnostics);
            Some(authorization.is_admin)
        } else {
            None
        };
        let action = match authorization {
            None => reply(bans_group_only(locale)),
            Some(false) => {
                self.state_diagnostics.push(format!(
                    "Unauthorized group charges attempt chat_id={chat} user_id={}",
                    sender_id.0
                ));
                reply(match locale {
                    bot_core::locale::Locale::Es => "Este comando es solo para admins del grupo",
                    bot_core::locale::Locale::En => "Only group admins can use this command",
                })
            }
            Some(true) => match plan_group_charges_command(
                argument,
                source.max_days(),
                chat_id,
                message_id,
                locale,
            ) {
                GroupChargesPlan::Reply(action) => action,
                GroupChargesPlan::Load { days } => {
                    match source.load(chat_id.0, days, GROUP_CHARGES_LIMIT) {
                        Ok(spenders) => reply(&render_group_charges(&spenders, days, locale)),
                        Err(error) => {
                            self.state_diagnostics
                                .push(format!("group charges chat_id={chat}: {error}"));
                            reply(group_charges_failed(locale))
                        }
                    }
                }
            },
        };
        let _receipt = self
            .actions
            .execute(action)
            .map_err(DispatchError::Action)?;
        Ok(DispatchOutcome::Handled)
    }

    fn dispatch_limit_command(
        &mut self,
        message: &IncomingMessage,
        (chat_id, message_id, sender_id): (ChatId, MessageId, UserId),
        command: LimitCommand,
        locale: bot_core::locale::Locale,
        is_group: bool,
    ) -> NativeDispatchResult<Config, Actions, Random> {
        let (command, target) = match command {
            LimitCommand::Change(argument) => {
                let (named, rest) = split_named_member(&argument, &message.text_mentions);
                let target = self.moderation_target(message, chat_id, named, locale);
                (LimitCommand::Change(rest.to_owned()), target)
            }
            LimitCommand::List => (
                LimitCommand::List,
                self.moderation_target(message, chat_id, None, locale),
            ),
        };
        let Some(store) = self.limit_store.as_mut() else {
            return Err(DispatchError::MissingService("chat limits"));
        };
        let chat = chat_id.0.to_string();
        let reply = |text: &str| ban_reply(chat_id, message_id, text);
        let authorization = if is_group {
            let authorization = self
                .authorization
                .authorize(&chat, &sender_id.0.to_string());
            self.state_diagnostics.extend(authorization.diagnostics);
            Some(authorization.is_admin)
        } else {
            None
        };
        let action = match authorization {
            None => reply(bans_group_only(locale)),
            Some(false) => {
                self.state_diagnostics.push(format!(
                    "Unauthorized limit attempt chat_id={chat} user_id={}",
                    sender_id.0
                ));
                reply(match locale {
                    bot_core::locale::Locale::Es => "Este comando es solo para admins del grupo",
                    bot_core::locale::Locale::En => "Only group admins can use this command",
                })
            }
            Some(true) if target.is_err() => reply(&target.err().unwrap_or_default()),
            Some(true) => {
                let context = LimitCommandContext {
                    chat_id,
                    message_id,
                    sender_id: sender_id.0,
                    locale,
                    target: target.ok().flatten(),
                };
                match plan_limit_command(command, context) {
                    LimitCommandPlan::Reply(action) => action,
                    LimitCommandPlan::List => match store.list(chat_id.0) {
                        Ok(users) => reply(&render_limit_list(&users, locale)),
                        Err(error) => {
                            self.state_diagnostics
                                .push(format!("chat limit list chat_id={chat}: {error}"));
                            reply(ban_list_failed(locale))
                        }
                    },
                    LimitCommandPlan::Set {
                        user_id,
                        name,
                        hourly_limit,
                    } => {
                        let target_authorization =
                            self.authorization.authorize(&chat, &user_id.to_string());
                        self.state_diagnostics
                            .extend(target_authorization.diagnostics);
                        if target_authorization.is_admin {
                            reply(limit_admin_target(locale))
                        } else {
                            match store.set(chat_id.0, user_id, &name, hourly_limit, sender_id.0) {
                                Ok(()) => reply(&limit_result_reply(&name, hourly_limit, locale)),
                                Err(error) => {
                                    self.state_diagnostics.push(format!(
                                        "chat limit chat_id={chat} user_id={user_id}: {error}"
                                    ));
                                    reply(ban_store_failed(locale))
                                }
                            }
                        }
                    }
                    LimitCommandPlan::Clear { user_id, name } => {
                        match store.clear(chat_id.0, user_id) {
                            Ok(removed) => reply(&unlimit_result_reply(&name, removed, locale)),
                            Err(error) => {
                                self.state_diagnostics.push(format!(
                                    "chat unlimit chat_id={chat} user_id={user_id}: {error}"
                                ));
                                reply(ban_store_failed(locale))
                            }
                        }
                    }
                }
            }
        };
        let _receipt = self
            .actions
            .execute(action)
            .map_err(DispatchError::Action)?;
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
            return Ok(DispatchOutcome::Unsupported);
        };
        if self.ai_conversation_source.is_none() {
            return Err(DispatchError::MissingService("AI conversation"));
        }

        let bot_username = self.bot_name.trim().trim_start_matches('@');
        let mention = (!bot_username.is_empty())
            && content
                .text
                .to_lowercase()
                .contains(&format!("@{}", bot_username.to_lowercase()));
        let reply_to_bot = message.replied_sender_username.as_deref() == Some(bot_username);
        let command_starts_with_slash = command.starts_with('/');
        let command_name = command.strip_prefix('/').filter(|name| !name.contains('@'));
        let known_command = command_name.is_some_and(|command_name| {
            command_starts_with_slash
                && telegram_commands(locale)
                    .iter()
                    .any(|candidate| candidate.command == command_name)
        });
        let reply_metadata = if reply_to_bot {
            self.replied_bot_metadata(message)
        } else {
            None
        };
        let mut routing = ResponseRoutingInput {
            known_command,
            command_starts_with_slash,
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
            ignore_media_replies: config.ignore_media_replies,
            bare_attachment: message.attachment.filter(|_| content.text.is_empty()),
            random_replies_enabled: config.ai_random_replies,
            trigger_words: Some(self.trigger_words.clone()),
            random_sample: None,
        };
        let evaluation = loop {
            if self.listen_only {
                break ResponseRoutingEvaluation::Ignore;
            }
            // Trigger words are provided up front, so routing never asks for
            // them.
            match evaluate_response_routing(&routing) {
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
        // Ignored messages are never charged, so they skip the lookup.
        let creditless_limit = if evaluation == ResponseRoutingEvaluation::Ignore {
            CreditlessLimit::Group(config.creditless_user_hourly_limit)
        } else {
            self.creditless_limit(message, (chat_id, sender_id), config)
        };
        let input = AiConversationInput {
            chat_id,
            message_id,
            chat_type: message.chat_type.clone().unwrap_or_default(),
            chat_title: message.chat_title.clone().unwrap_or_default(),
            sender_id,
            sender_first_name: message.sender_first_name.clone().unwrap_or_default(),
            sender_username: message.sender_username.clone().unwrap_or_default(),
            sender_is_bot: message.sender_is_bot,
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
            creditless_limit,
            group_pays_first: config.group_pays_first,
            timestamp,
            spontaneous,
            link_context: None,
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
        let link_context = match self.prefetched_link_context.take() {
            Some(context) => context,
            None if text_without_links(ai_prompt_text) != ai_prompt_text => self
                .link_replacement_source
                .as_mut()
                .and_then(|source| source.preview_context(ai_prompt_text)),
            None => None,
        };
        let input = AiConversationInput {
            link_context,
            ..input
        };

        let (preparation, stream_finalize, ignored_edit_failures, thinking_status_failed) = {
            let missing = DispatchError::MissingService("AI conversation");
            let source = self.ai_conversation_source.as_mut().ok_or(missing)?;
            let mut stream = TelegramAiStream::new(&mut self.actions, chat_id, message_id)
                .with_thinking_text(thinking_text(locale));
            // The thinking status waits for the source to admit the turn, so
            // denied or spontaneous turns that end silently never flash it.
            // Live updates are best effort: a failed send or edit must not
            // abort a model call that is already running. `finalize` still
            // delivers the reply.
            let mut thinking_status_failed = false;
            let preparation = source.prepare_streaming_events(input, &mut |event| {
                thinking_status_failed |= stream.feed(event).is_err();
                Ok(())
            });
            match preparation {
                Err(error) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (Err(error), None, ignored, thinking_status_failed)
                }
                Ok(AiPreparation::Silent { diagnostics }) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (
                        Ok(AiPreparation::Silent { diagnostics }),
                        None,
                        ignored,
                        thinking_status_failed,
                    )
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
                        thinking_status_failed,
                    )
                }
            }
        };
        if thinking_status_failed {
            self.state_diagnostics.push(
                "AI Telegram live update failed; continuing so response delivery can retry"
                    .to_owned(),
            );
        }
        if ignored_edit_failures > 0 {
            self.state_diagnostics.push(format!(
                "AI Telegram stream ignored {ignored_edit_failures} intermediate edit failures"
            ));
        }
        let preparation = match preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                // Dispatcher diagnostics never reach the log on their own,
                // and this is the only trace of why the reply failed.
                eprintln!("AI conversation failed: {error:?}");
                self.state_diagnostics
                    .push(format!("AI conversation: {error}"));
                let text = match locale {
                    bot_core::locale::Locale::Es => {
                        "Me quedé reculando y no te pude responder. Probá de nuevo"
                    }
                    bot_core::locale::Locale::En => "I could not answer. Try again",
                };
                return self.send_failure_reply(chat_id, message_id, text);
            }
        };
        // Only a reply finalizes the stream, so a silent turn has nothing to
        // deliver.
        let (completion_id, diagnostics, stream_finalize) = match (preparation, stream_finalize) {
            (
                AiPreparation::Reply {
                    completion_id,
                    diagnostics,
                    ..
                },
                Some(stream_finalize),
            ) => (completion_id, diagnostics, stream_finalize),
            (
                AiPreparation::Silent { diagnostics } | AiPreparation::Reply { diagnostics, .. },
                _,
            ) => {
                self.state_diagnostics.extend(diagnostics);
                return Ok(DispatchOutcome::Handled);
            }
        };
        self.state_diagnostics.extend(diagnostics);
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
            return Ok(DispatchOutcome::Unsupported);
        };
        let creditless_limit = self.creditless_limit(message, (chat_id, sender_id), config);
        let Some(source) = self.ai_conversation_source.as_mut() else {
            return Err(DispatchError::MissingService("AI conversation"));
        };
        let input = AiConversationInput {
            chat_id,
            message_id,
            chat_type: message.chat_type.clone().unwrap_or_default(),
            chat_title: message.chat_title.clone().unwrap_or_default(),
            sender_id,
            sender_first_name: message.sender_first_name.clone().unwrap_or_default(),
            sender_username: message.sender_username.clone().unwrap_or_default(),
            sender_is_bot: message.sender_is_bot,
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
            creditless_limit,
            group_pays_first: config.group_pays_first,
            timestamp,
            spontaneous: false,
            link_context: None,
        };
        // Downloading and converting media can take a while; show that the
        // bot is working. Best effort: the reply does not depend on it.
        if let Err(error) = self.actions.execute(TelegramAction::SendTyping { chat_id }) {
            self.state_diagnostics
                .push(format!("media command typing status: {error}"));
        }
        let preparation = match source.prepare_media_command(input) {
            Ok(Some(preparation)) => preparation,
            result => {
                if let Err(error) = result {
                    self.state_diagnostics
                        .push(format!("media command: {error}"));
                }
                let text = match locale {
                    bot_core::locale::Locale::Es => {
                        format!("Se trabó el {command}. Probá más tarde")
                    }
                    bot_core::locale::Locale::En => format!("{command} failed. Try again later"),
                };
                return self.send_failure_reply(chat_id, message_id, &text);
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
            is_bot: message.sender_is_bot,
            text: &content.text,
            is_group: is_group_chat_type(message.chat_type.as_deref()),
            timestamp,
        });
        if let Ok(incoming) = incoming
            && let Err(error) = self.state.record_incoming(&incoming)
        {
            self.state_diagnostics
                .push(format!("incoming media command state: {error}"));
        }

        let action = if text.chars().count() > MAX_TELEGRAM_TEXT_LENGTH {
            TelegramAction::SendDocument {
                chat_id,
                document: text.as_bytes().to_vec().into(),
                file_name: "transcript.txt".to_owned(),
                reply_to_message_id: Some(message_id),
                caption: String::new(),
            }
        } else {
            let mut response = SendMessage::new(chat_id, &text);
            response.reply_to_message_id = Some(message_id);
            TelegramAction::SendMessage(response)
        };
        let receipt = match self.actions.execute(action) {
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
        if let Ok(outgoing) = outgoing
            && let Err(error) = self.state.record_outgoing(&outgoing)
        {
            self.state_diagnostics
                .push(format!("outgoing media command state: {error}"));
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
            return Ok(DispatchOutcome::Unsupported);
        };
        let creditless_limit = self.creditless_limit(message, (chat_id, sender_id), config);
        let input = AiConversationInput {
            chat_id,
            message_id,
            chat_type: message.chat_type.clone().unwrap_or_default(),
            chat_title: message.chat_title.clone().unwrap_or_default(),
            sender_id,
            sender_first_name: message.sender_first_name.clone().unwrap_or_default(),
            sender_username: message.sender_username.clone().unwrap_or_default(),
            sender_is_bot: message.sender_is_bot,
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
            creditless_limit,
            group_pays_first: config.group_pays_first,
            timestamp,
            spontaneous: false,
            link_context: None,
        };
        let (preparation, stream_finalize, ignored_edit_failures, thinking_status_failed) = {
            let Some(source) = self.ai_conversation_source.as_mut() else {
                return Err(DispatchError::MissingService("AI conversation"));
            };
            let mut stream = TelegramAiStream::new(&mut self.actions, chat_id, message_id)
                .with_thinking_text(thinking_text(locale));
            // Live updates are best effort, as for chat replies.
            let mut thinking_status_failed = stream.show_thinking().is_err();
            let preparation =
                source.prepare_summary_command_streaming_events(input, &mut |event| {
                    thinking_status_failed |= stream.feed(event).is_err();
                    Ok(())
                });
            match preparation {
                Err(error) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (Err(error), None, ignored, thinking_status_failed)
                }
                Ok(None) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (Ok(None), None, ignored, thinking_status_failed)
                }
                Ok(Some(AiPreparation::Silent { diagnostics })) => {
                    stream.cancel();
                    let ignored = stream.ignored_edit_failures();
                    (
                        Ok(Some(AiPreparation::Silent { diagnostics })),
                        None,
                        ignored,
                        thinking_status_failed,
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
                        thinking_status_failed,
                    )
                }
            }
        };
        if thinking_status_failed {
            self.state_diagnostics.push(
                "summary Telegram live update failed; continuing so response delivery can retry"
                    .to_owned(),
            );
        }
        if ignored_edit_failures > 0 {
            self.state_diagnostics.push(format!(
                "summary Telegram stream ignored {ignored_edit_failures} intermediate edit failures"
            ));
        }
        let preparation = match preparation {
            Ok(Some(preparation)) => preparation,
            Ok(None) => {
                let text = match locale {
                    bot_core::locale::Locale::Es => "No pude generar el resumen. Probá de nuevo",
                    bot_core::locale::Locale::En => "I could not generate the summary. Try again",
                };
                return self.send_failure_reply(chat_id, message_id, text);
            }
            Err(error) => {
                self.state_diagnostics
                    .push(format!("summary command: {error}"));
                let text = match locale {
                    bot_core::locale::Locale::Es => "No pude generar el resumen. Probá de nuevo",
                    bot_core::locale::Locale::En => "I could not generate the summary. Try again",
                };
                return self.send_failure_reply(chat_id, message_id, text);
            }
        };
        // Only a reply finalizes the stream, so a silent summary has nothing
        // to deliver.
        let (
            AiPreparation::Reply {
                text,
                completion_id,
                diagnostics,
            },
            Some(stream_finalize),
        ) = (preparation, stream_finalize)
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
            is_bot: message.sender_is_bot,
            text: &content.text,
            is_group: is_group_chat_type(message.chat_type.as_deref()),
            timestamp,
        }) && let Err(error) = self.state.record_incoming(&incoming)
        {
            self.state_diagnostics
                .push(format!("incoming summary command state: {error}"));
        }
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
            return Ok(DispatchOutcome::Unsupported);
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
        // Banned members get no reply, but their messages still reach the chat
        // history so summaries and replies to others keep context. A bare
        // `/ban` is the same: Rose and GroupHelp answer it, and acting on it
        // too would ignore someone an admin only meant to ban from the group.
        if is_group_chat_type(message.chat_type.as_deref())
            && (self.sender_is_banned(chat_id, sender_id.0) || is_shared_ban_command(&content.text))
        {
            if self.ai_conversation_source.is_none() {
                return Ok(DispatchOutcome::Handled);
            }
            let parsed = parse_command(&content.text, &self.bot_name);
            self.listen_only = true;
            let outcome = self.dispatch_ai_message(
                message,
                &config,
                locale,
                timestamp,
                &parsed.command,
                &parsed.message_text,
            );
            self.listen_only = false;
            return outcome;
        }
        let cashtag_with_period = {
            let mut parts = content.text.split_whitespace();
            parts.next().and_then(detect_signal_query).is_some_and(|_| {
                parts.next().is_some_and(|period| {
                    bot_core::price_queries::ChartPeriod::parse(period).is_some()
                }) && parts.next().is_none()
            })
        };
        if (detect_signal_query(&content.text).is_some() || cashtag_with_period)
            && let Some(outcome) = self.dispatch_asset_prices(
                message,
                &content.text,
                MarketPriceCommand::Unified,
                locale,
                timestamp,
            )?
        {
            return Ok(outcome);
        }
        self.prefetched_link_context = None;
        if !content.text.starts_with('/') && has_replaceable_link(&content.text) {
            let addressed = self.is_addressed_to_bot(message, &content.text, &config);
            if let Some(outcome) =
                self.dispatch_link_replacement(message, &config, locale, timestamp, addressed)?
            {
                return Ok(outcome);
            }
        }
        if let Some(credits) = typed_topup_amount(message, &content.text) {
            // A number sent as a reply to the top-up menu redraws it on that
            // amount, nearest end if out of range.
            let credits = credits.clamp(MIN_TOPUP_CREDITS, MAX_TOPUP_CREDITS);
            let menu = plan_topup_command(
                chat_id,
                message_id,
                &format!("/topup {credits}"),
                &self.bot_name,
                locale,
                "private",
                self.billing_available,
                self.lightning_checkout.is_some(),
            );
            let _receipt = menu
                .map(|action| self.actions.execute(action))
                .transpose()
                .map_err(DispatchError::Action)?;
            return Ok(DispatchOutcome::Handled);
        }
        if message.has_reply && self.ai_conversation_source.is_none() {
            return Err(DispatchError::MissingService("AI conversation"));
        }
        let parsed = parse_command(&content.text, &self.bot_name);
        if matches!(
            parsed.command.as_str(),
            "/transcribe" | "/transcript" | "/describe"
        ) {
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
        if matches!(
            parsed.command.as_str(),
            "/tarea" | "/tareas" | "/task" | "/tasks"
        ) && !parsed.message_text.is_empty()
        {
            return self.dispatch_ai_message(
                message,
                &config,
                locale,
                timestamp,
                &parsed.command,
                &parsed.message_text,
            );
        }
        let is_group = is_group_chat_type(message.chat_type.as_deref());
        if let Some(command) = classify_ban_command(&content.text, &self.bot_name) {
            let ids = (chat_id, message_id, sender_id);
            let argument = parsed.message_text.as_str();
            return self.dispatch_ban_command(message, ids, command, argument, locale, is_group);
        }
        if let Some(command) = classify_limit_command(&content.text, &self.bot_name) {
            let ids = (chat_id, message_id, sender_id);
            return self.dispatch_limit_command(message, ids, command, locale, is_group);
        }
        if let Some(argument) = classify_group_charges_command(&content.text, &self.bot_name) {
            let ids = (chat_id, message_id, sender_id);
            return self.dispatch_group_charges_command(ids, &argument, locale, is_group);
        }
        let is_settings_command = matches!(
            parsed.command.as_str(),
            "/language" | "/idioma" | "/config" | "/configs" | "/settings"
        );
        let mut language_requires_group_authorization = is_group;
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
                    bot_core::locale::Locale::Es => "Este comando es solo para admins del grupo",
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
            language_requires_group_authorization = false;
        }
        let language_plan = plan_language_command(
            chat_id,
            message_id,
            &content.text,
            &self.bot_name,
            locale,
            &config,
            language_requires_group_authorization,
        );
        let (plan, updated_config) = match language_plan {
            LanguageCommandPlan::Action {
                action,
                updated_config,
            } => (StatelessCommandPlan::Action(action), updated_config),
            // Group language commands are settings commands, which the admin
            // check above either rejects or unlocks, so the planner never
            // asks for group authorization here.
            LanguageCommandPlan::GroupAuthorizationRequired | LanguageCommandPlan::NotHandled => {
                (StatelessCommandPlan::NotHandled, None)
            }
        };
        // Billing and admin planners claim only their own commands; any
        // other command is NotHandled and falls through the chain below.
        let balance_plan = plan_balance_command(
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
        );
        let charges_plan = plan_charges_command(
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
        );
        let transfer_plan = plan_transfer_command(
            &content.text,
            &self.bot_name,
            TransferCommandContext {
                chat_id,
                message_id,
                user_id: Some(sender_id.0),
                locale,
                is_group,
                billing_available: self.billing_available,
                recipient: transfer_recipient(message, &self.bot_name),
            },
        );
        let printcredits_plan = plan_printcredits_command(
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
        );
        let creditlog_plan = plan_creditlog_command(
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
        );
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
            self.lightning_checkout.is_some(),
        ) {
            StatelessCommandPlan::Action(action)
        } else if matches!(
            parsed.command.as_str(),
            "/tarea" | "/tareas" | "/task" | "/tasks"
        ) {
            // Task prompts with text were routed to the AI turn above.
            let Some(source) = self.scheduled_task_source.as_mut() else {
                return Err(DispatchError::MissingService("scheduled tasks"));
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
        } else if let BalanceCommandPlan::Reply(action) = balance_plan {
            StatelessCommandPlan::Action(action)
        } else if let BalanceCommandPlan::Load {
            user_id,
            chat_id,
            is_group,
        } = balance_plan
        {
            let Some(source) = self.balance_source.as_mut() else {
                return Err(DispatchError::MissingService("billing balances"));
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
                            "Se trabó leyendo tu saldo. Probá de nuevo".to_owned()
                        }
                        bot_core::locale::Locale::En => {
                            "I could not load your balance. Try again".to_owned()
                        }
                    }
                }
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let ChargesCommandPlan::Reply(action) = charges_plan {
            StatelessCommandPlan::Action(action)
        } else if let ChargesCommandPlan::Load {
            user_id,
            limit,
            timezone_minutes,
        } = charges_plan
        {
            let Some(source) = self.charge_history_source.as_mut() else {
                return Err(DispatchError::MissingService("charge history"));
            };
            let (text, keyboard) = match source.load(user_id, limit, None, "older") {
                Ok(page) => {
                    render_charge_history_page(&page, user_id, limit, timezone_minutes, locale)
                }
                Err(error) => {
                    self.state_diagnostics.push(format!(
                        "charge history load chat_id={} user_id={user_id} limit={limit}: {error}",
                        chat_id.0
                    ));
                    let text = match locale {
                        bot_core::locale::Locale::Es => {
                            "Se trabó leyendo tus gastos. Probá de nuevo"
                        }
                        bot_core::locale::Locale::En => "I could not load your spending. Try again",
                    };
                    (text.to_owned(), None)
                }
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            message.reply_markup = keyboard;
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let TransferCommandPlan::Reply(action) = transfer_plan {
            StatelessCommandPlan::Action(action)
        } else if let TransferCommandPlan::Transfer {
            user_id,
            chat_id,
            amount,
        } = transfer_plan
        {
            let Some(sink) = self.transfer_sink.as_mut() else {
                return Err(DispatchError::MissingService("credit transfers"));
            };
            let operation_id = format!("transfer:{chat_id}:{}", message_id.0);
            let text = match sink.transfer(user_id, chat_id, amount, &operation_id) {
                Ok(result) => transfer_result_reply(amount, result, locale),
                Err(error) => {
                    self.state_diagnostics.push(format!(
                            "credit transfer chat_id={chat_id} user_id={user_id} amount={amount}: {error}"
                        ));
                    match locale {
                        bot_core::locale::Locale::Es => {
                            "Se trabó la transferencia. Probá de nuevo".to_owned()
                        }
                        bot_core::locale::Locale::En => "The transfer failed. Try again".to_owned(),
                    }
                }
            };
            let mut message = SendMessage::new(ChatId(chat_id), &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let TransferCommandPlan::TransferToUser {
            user_id,
            recipient_id,
            amount,
        } = transfer_plan
        {
            let Some(sink) = self.transfer_sink.as_mut() else {
                return Err(DispatchError::MissingService("credit transfers"));
            };
            let operation_id = format!("transfer:{}:{}", chat_id.0, message_id.0);
            let text = match sink.transfer_to_user(user_id, recipient_id, amount, &operation_id) {
                Ok(result) => {
                    let name = recipient_display_name(message, locale);
                    user_transfer_result_reply(amount, &name, result, locale)
                }
                Err(error) => {
                    self.state_diagnostics.push(format!(
                        "user credit transfer chat_id={} amount={amount}: {error}",
                        chat_id.0
                    ));
                    match locale {
                        bot_core::locale::Locale::Es => {
                            "Se trabó la transferencia. Probá de nuevo".to_owned()
                        }
                        bot_core::locale::Locale::En => "The transfer failed. Try again".to_owned(),
                    }
                }
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let PrintCreditsPlan::Reply(action) = printcredits_plan {
            StatelessCommandPlan::Action(action)
        } else if let PrintCreditsPlan::Mint { user_id, amount } = printcredits_plan {
            let Some(sink) = self.admin_credit_sink.as_mut() else {
                return Err(DispatchError::MissingService("admin credit minting"));
            };
            let operation_id = format!("printcredits:{}:{}", chat_id.0, message_id.0);
            let text = match sink.mint(user_id, amount, &operation_id) {
                Ok(balance) => printcredits_result_reply(amount, balance, locale),
                Err(error) => {
                    self.state_diagnostics.push(format!(
                        "admin credit mint chat_id={} user_id={user_id} amount={amount}: {error}",
                        chat_id.0
                    ));
                    match locale {
                        bot_core::locale::Locale::Es => {
                            "Se trabó imprimiendo créditos. Probá de nuevo".to_owned()
                        }
                        bot_core::locale::Locale::En => {
                            "I could not mint credits. Try again".to_owned()
                        }
                    }
                }
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if let CreditLogPlan::Reply(action) = creditlog_plan {
            StatelessCommandPlan::Action(action)
        } else if let CreditLogPlan::Load { limit } = creditlog_plan {
            let Some(source) = self.admin_creditlog_source.as_mut() else {
                return Err(DispatchError::MissingService("admin credit log"));
            };
            let text = match source.load(limit) {
                Ok(entries) if entries.is_empty() => match locale {
                    bot_core::locale::Locale::Es => {
                        "No hay liquidaciones de IA recientes".to_owned()
                    }
                    bot_core::locale::Locale::En => "There are no recent AI settlements".to_owned(),
                },
                Ok(entries) => render_creditlog(&entries, locale),
                Err(error) => {
                    self.state_diagnostics.push(format!(
                        "admin creditlog chat_id={} user_id={} limit={limit}: {error}",
                        chat_id.0, sender_id.0
                    ));
                    match locale {
                        bot_core::locale::Locale::Es => {
                            "Se trabó leyendo el creditlog. Probá de nuevo".to_owned()
                        }
                        bot_core::locale::Locale::En => {
                            "I could not load the credit log. Try again".to_owned()
                        }
                    }
                }
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if classify_bcra_command(&parsed.command) {
            let Some(source) = self.bcra_source.as_mut() else {
                return Err(DispatchError::MissingService("BCRA market data"));
            };
            let load = source.load(locale, timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = load.text.unwrap_or_else(|| match locale {
                bot_core::locale::Locale::Es => {
                    "No pude conseguir las variables del BCRA. Probá más tarde".to_owned()
                }
                bot_core::locale::Locale::En => {
                    "I could not load the BCRA variables. Try again later".to_owned()
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
                        return Err(DispatchError::MissingService("dollar market data"));
                    };
                    let load = source.load(hours_ago, locale, timestamp);
                    self.state_diagnostics.extend(load.diagnostics);
                    let text = load.text.unwrap_or_else(|| match locale {
                        bot_core::locale::Locale::Es => {
                            "No pude traer las cotizaciones del dólar, boludo. Probá más tarde"
                                .to_owned()
                        }
                        bot_core::locale::Locale::En => {
                            "I could not load dollar rates. Try again later".to_owned()
                        }
                    });
                    let mut message = SendMessage::new(chat_id, &text);
                    message.reply_to_message_id = Some(message_id);
                    StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
                }
            }
        } else if classify_election_command(&parsed.command) {
            let Some(source) = self.election_source.as_mut() else {
                return Err(DispatchError::MissingService("election markets"));
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
            if let Some(outcome) = self.dispatch_asset_prices(
                message,
                &parsed.message_text,
                command,
                locale,
                timestamp,
            )? {
                return Ok(outcome);
            }
            let Some(source) = self.market_price_source.as_mut() else {
                return Err(DispatchError::MissingService("market prices"));
            };
            let load = source.load(&parsed.message_text, command, locale, timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = if load.text.trim().is_empty() {
                load.selection.as_ref().map_or_else(
                    || match locale {
                        bot_core::locale::Locale::Es => {
                            "No pude conseguir una cotización. Probá más tarde".to_owned()
                        }
                        bot_core::locale::Locale::En => {
                            "I could not get a quote. Try again later".to_owned()
                        }
                    },
                    |selection| format_market_selection(selection, locale),
                )
            } else {
                load.text
            };
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if classify_stock_command(&parsed.command) {
            if !parsed.message_text.trim().is_empty()
                && self.market_price_source.is_some()
                && let Some(outcome) = self.dispatch_asset_prices(
                    message,
                    &format!("stock:{}", parsed.message_text),
                    MarketPriceCommand::Unified,
                    locale,
                    timestamp,
                )?
            {
                return Ok(outcome);
            }
            let Some(source) = self.stock_price_source.as_mut() else {
                return Err(DispatchError::MissingService("stock prices"));
            };
            let load = source.load(&parsed.message_text, timestamp);
            self.state_diagnostics.extend(load.diagnostics);
            let text = render_stock_quotes(load.quotes.as_deref(), locale);
            if !parsed.message_text.trim().is_empty()
                && let Some(rows) = &load.quotes
                && rows.len() == 1
                && let Some(quote) = &rows[0].1
                && let Some(source) = self.stock_price_source.as_mut()
                && let Ok(photo) = source.render_chart(quote, timestamp)
                && let Ok(Some(receipt)) = self.actions.try_photo(TelegramAction::SendPhoto {
                    chat_id,
                    photo: photo.into(),
                    reply_to_message_id: Some(message_id),
                    caption: text.clone(),
                    parse_mode: None,
                    reply_markup: None,
                })
                && receipt.message_id.is_some()
            {
                self.record_price_delivery(message, &text, receipt.message_id, timestamp);
                return Ok(DispatchOutcome::Handled);
            }
            let mut message = SendMessage::new(chat_id, &text);
            message.reply_to_message_id = Some(message_id);
            StatelessCommandPlan::Action(TelegramAction::SendMessage(message))
        } else if classify_oil_command(&parsed.command) {
            let Some(source) = self.oil_price_source.as_mut() else {
                return Err(DispatchError::MissingService("oil prices"));
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
                return Err(DispatchError::MissingService("weather"));
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
                return Err(DispatchError::MissingService("greeting media"));
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
                    return Err(DispatchError::Invariant(
                        "random greeting index out of bounds",
                    ));
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
                return Err(DispatchError::MissingService("rulo market data"));
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
                Err(_) => render_devo_reply(DevoReply::Usage, locale),
                Ok(DevoCommandPlan::Reply(reply)) => render_devo_reply(reply, locale),
                Ok(DevoCommandPlan::Load { fee, purchase }) => {
                    let Some(source) = self.dollar_quotes_source.as_mut() else {
                        return Err(DispatchError::MissingService("dollar quotes"));
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
                return Err(DispatchError::MissingService("bitcoin prices"));
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
                Err(_) | Ok(RandomSelection::Invalid) => {
                    let text = match locale {
                        bot_core::locale::Locale::Es => {
                            "Mandate algo como 'pizza, carne, sushi' o '1-10', boludo, no me hagas laburar al pedo"
                        }
                        bot_core::locale::Locale::En => {
                            "Send options like 'pizza, steak, sushi' or a range like '1-10'"
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
                        return Err(DispatchError::Invariant("random reply index out of bounds"));
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
            match plan_stateless_command_with_reply(
                chat_id,
                message_id,
                &content.text,
                message.replied_text.as_deref(),
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
            }
        };
        match plan {
            StatelessCommandPlan::Action(action) => {
                if let Some(updated_config) = &updated_config {
                    self.config
                        .set_changed(&chat_id.0.to_string(), &config, updated_config)
                        .map_err(DispatchError::Config)?;
                }
                let command = parsed.command;
                let incoming = prepare_incoming_command_state(IncomingCommandState {
                    chat_id,
                    message_id,
                    user_id: sender_id,
                    first_name: message.sender_first_name.as_deref(),
                    username: message.sender_username.as_deref(),
                    is_bot: message.sender_is_bot,
                    text: &content.text,
                    is_group,
                    timestamp,
                });
                if let Ok(incoming) = incoming
                    && let Err(error) = self.state.record_incoming(&incoming)
                {
                    self.state_diagnostics
                        .push(format!("incoming command state: {error}"));
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
                // Only chats with an explicit language get their own menu; "auto"
                // chats keep Telegram's app-language and all-groups menus, which
                // already match how they are answered.
                let menu_language = &updated_config.as_ref().unwrap_or(&config).language;
                if is_settings_command && matches!(menu_language.as_str(), "es" | "en") {
                    let menu_locale = resolve_locale(
                        Some(menu_language),
                        message.sender_language_code.as_deref(),
                        message.chat_type.as_deref().unwrap_or_default(),
                    );
                    self.sync_chat_command_menu(chat_id, menu_locale);
                }
                if let Some(response_text) = response_text {
                    let outgoing = prepare_outgoing_command_state(OutgoingCommandState {
                        chat_id,
                        incoming_message_id: message_id,
                        sent_message_id: receipt.message_id,
                        text: &response_text,
                        command: &command,
                        timestamp,
                    });
                    if let Ok(outgoing) = outgoing
                        && let Err(error) = self.state.record_outgoing(&outgoing)
                    {
                        self.state_diagnostics
                            .push(format!("outgoing command state: {error}"));
                    }
                }
                Ok(DispatchOutcome::Handled)
            }
            StatelessCommandPlan::NotHandled => self.dispatch_ai_message(
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
            IncomingEvent::Poll(poll) => self.dispatch_poll_update(&poll, false),
            IncomingEvent::PollAnswer(answer) => self.dispatch_poll_update(&answer, true),
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
                let query_id = query.get("id").and_then(Value::as_str).map(str::to_owned);
                match plan_pre_checkout(&Value::Object(query), self.billing_available, locale) {
                    Ok(Some(action)) => {
                        let _receipt = self
                            .actions
                            .execute(action)
                            .map_err(DispatchError::Action)?;
                        DispatchOutcome::Handled
                    }
                    Ok(None) => DispatchOutcome::Handled,
                    Err(error) => {
                        self.state_diagnostics
                            .push(format!("invalid pre-checkout query: {error}"));
                        if let Some(query_id) = query_id {
                            let _receipt = self
                                .actions
                                .execute(TelegramAction::AnswerPreCheckout {
                                    query_id,
                                    ok: false,
                                    error_message: Some(match locale {
                                        bot_core::locale::Locale::Es => {
                                            "Ese pago vino raro y no te lo pude validar".to_owned()
                                        }
                                        bot_core::locale::Locale::En => {
                                            "I could not validate this payment".to_owned()
                                        }
                                    }),
                                })
                                .map_err(DispatchError::Action)?;
                        }
                        DispatchOutcome::Handled
                    }
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

    fn error_disposition(&self, error: &Self::Error) -> HandlerErrorDisposition {
        match error {
            DispatchError::Action(error) if self.actions.is_permanent_failure(error) => {
                HandlerErrorDisposition::DiscardUpdate
            }
            _ => HandlerErrorDisposition::RetryUpdate,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};
    use std::rc::Rc;

    use bot_adapters::telegram_polling::{IncomingEvent, IncomingMessage, IncomingUpdate};
    use bot_core::chat_config::ChatConfig;
    use bot_core::command_state::{IncomingCommandWritePlan, OutgoingCommandWritePlan};
    use bot_core::telegram_actions::{MAX_TELEGRAM_TEXT_LENGTH, SendMessage, TelegramAction};
    use bot_core::telegram_input::{Attachment, ChatId, MessageContent, MessageId, UserId};
    use bot_core::telegram_payments::StarPaymentRecord;
    use bot_core::token_signals::{
        PairLiquidity, PairPriceChange, PairToken, PairTransactionWindows, PairTransactions,
        PairVolume, SignalQuery, SignalState, TokenAddress, TokenPair, TokenSignal,
    };
    use num_bigint::BigInt;
    use serde_json::{Map, Value, json};

    use crate::ai_dispatch::{
        AiConversationInput, AiConversationSource, AiDelivery, AiPreparation, AiReplyMetadata,
        AiStreamEvent, CreditlessLimit,
    };

    use super::LightningCheckout;
    use super::{
        ActionReceipt, ActionSink, AdminCreditLogSource, AdminCreditSink, BcraLoad, BcraSource,
        BillingBalanceSource, BillingBalances, BillingTransferSink, BitcoinPriceSource,
        ChargeHistoryPage, ChargeHistorySource, ChatConfigSource, DispatchError, DispatchOutcome,
        DollarMarketLoad, DollarMarketSource, DollarQuotesSource, ElectionLoad, ElectionSource,
        GreetingPoolLoad, GreetingPoolSource, GroupAuthorizationDecision, GroupAuthorizer,
        LinkReplacementLoad, LinkReplacementSource, MarketPriceLoad, MarketPriceSource,
        MarketSelection, MessageStateSink, NativeDispatcher, OilPriceSource, OilQuoteLoad,
        RandomSource, RuloInputLoad, RuloSource, RuntimeValues, ScheduledTaskSource,
        StarPaymentReceipt, StarPaymentSink, StockPriceSource, StockQuotesLoad,
        StoredMarketSelection, TokenSignalLoad, TokenSignalSource, TransferResult,
        WeatherObservationLoad, WeatherSource, deduplicate_market_candidates,
        market_contracts_match, market_selection_command, market_selection_id,
        market_selection_key, market_selection_text, provider_query_text, short_market_address,
    };
    use crate::runtime::{HandlerErrorDisposition, UpdateHandler as _};
    use bot_core::charge_history::{ChargeHistoryEntry, ChargeHistoryGroup};
    use bot_core::devo::DevoQuotes;
    use bot_core::greeting_commands::GreetingCategory;
    use bot_core::lightning_topup::{LightningInvoice, lightning_invoice_failed};
    use bot_core::links::LinkReplacement;
    use bot_core::polymarket::parse_election_events;
    use bot_core::rulo::{ExchangeQuote, RuloInput};
    use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule, TaskStateError};
    use bot_core::stocks::StockQuote;
    use bot_core::telegram_payments::{BillingPackTerms, topup_menu};
    use bot_core::weather::WeatherObservation;

    struct Config {
        value: Result<ChatConfig, &'static str>,
        chat_ids: Vec<String>,
    }

    #[test]
    fn bare_evm_address_matches_the_chain_resolved_by_dexscreener() {
        let address = "0xb095274743941e953c746f9c228da9c18bb6ec29";
        let query = TokenAddress {
            chain_id: "ethereum".to_owned(),
            network: "eth".to_owned(),
            tag: "ETH".to_owned(),
            address: address.to_owned(),
        };
        let resolved = TokenAddress {
            chain_id: "base".to_owned(),
            network: "base".to_owned(),
            tag: "BASE".to_owned(),
            address: address.to_owned(),
        };
        assert!(market_contracts_match(&query, &resolved));
        assert!(!market_contracts_match(
            &query,
            &TokenAddress {
                address: "0x0000000000000000000000000000000000000001".to_owned(),
                ..resolved
            }
        ));
    }

    #[test]
    fn provider_url_is_normalized_before_market_lookup() {
        assert_eq!(
            provider_query_text(
                "https://www.coingecko.com/en/coins/hunter-bidens-laptop?utm_source=test"
            ),
            "hunter-bidens-laptop"
        );
        assert_eq!(provider_query_text("$LAPTOP"), "$LAPTOP");
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

    /// Which executed actions a scripted failure applies to.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ActionKind {
        Any,
        SendMessage,
        DeleteMessage,
        AnswerCallback,
        SetCommands,
    }

    impl ActionKind {
        fn matches(self, action: &TelegramAction) -> bool {
            match self {
                Self::Any => true,
                Self::SendMessage => matches!(action, TelegramAction::SendMessage(_)),
                Self::DeleteMessage => matches!(action, TelegramAction::DeleteMessage { .. }),
                Self::AnswerCallback => matches!(action, TelegramAction::AnswerCallback { .. }),
                Self::SetCommands => matches!(action, TelegramAction::SetCommands { .. }),
            }
        }
    }

    /// Message ids the fake sink confirms for executed actions.
    #[derive(Debug, Clone, Copy)]
    enum Receipts {
        Fixed(Option<MessageId>),
        /// Sent messages and photos get increasing ids; other actions get 0.
        Sequential(i64),
    }

    impl Default for Receipts {
        fn default() -> Self {
            Self::Fixed(Some(MessageId(700)))
        }
    }

    /// How a fake sink answers one of the `try_*` delivery attempts.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    enum Attempt {
        #[default]
        Deliver,
        Skip,
        Unconfirmed,
        Fail,
    }

    #[derive(Debug)]
    struct ExecuteFailure {
        kind: ActionKind,
        remaining: usize,
        error: &'static str,
        /// Whether the rejected action is still recorded as attempted.
        record: bool,
    }

    #[derive(Debug, Default)]
    struct ActionScript {
        receipts: Receipts,
        failure: Option<ExecuteFailure>,
        photo: Attempt,
        edit: Attempt,
        video: Attempt,
        animation: Attempt,
        invoice: Attempt,
        /// Leading invoice attempts Telegram refuses before `invoice` applies.
        invoice_refusals: usize,
    }

    impl ActionScript {
        fn photo(attempt: Attempt) -> Self {
            Self {
                photo: attempt,
                ..Self::default()
            }
        }

        fn edit(attempt: Attempt) -> Self {
            Self {
                edit: attempt,
                ..Self::default()
            }
        }

        fn invoice(attempt: Attempt) -> Self {
            Self {
                invoice: attempt,
                ..Self::default()
            }
        }
    }

    /// The single Telegram sink every dispatcher test uses, so all of them
    /// exercise one `NativeDispatcher` instantiation. It records every
    /// delivered action and follows its script for failures.
    #[derive(Default)]
    struct Actions(Vec<TelegramAction>, ActionScript);

    impl Actions {
        fn scripted(script: ActionScript) -> Self {
            Self(Vec::new(), script)
        }
    }

    fn attempt_actions(attempt: Attempt, script: fn(Attempt) -> ActionScript) -> Actions {
        Actions::scripted(script(attempt))
    }

    fn failing_actions(
        kind: ActionKind,
        remaining: usize,
        error: &'static str,
        record: bool,
    ) -> Actions {
        Actions::scripted(ActionScript {
            failure: Some(ExecuteFailure {
                kind,
                remaining,
                error,
                record,
            }),
            ..ActionScript::default()
        })
    }

    impl ActionSink for Actions {
        type Error = &'static str;

        fn is_permanent_failure(&self, error: &Self::Error) -> bool {
            *error == "synthetic permanent failure"
        }

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            if let Some(failure) = self.1.failure.as_mut()
                && failure.remaining > 0
                && failure.kind.matches(&action)
            {
                failure.remaining -= 1;
                if failure.record {
                    self.0.push(action);
                }
                return Err(failure.error);
            }
            let message_id = match &mut self.1.receipts {
                Receipts::Fixed(message_id) => *message_id,
                Receipts::Sequential(next) => Some(MessageId(
                    if matches!(
                        action,
                        TelegramAction::SendMessage(_) | TelegramAction::SendPhoto { .. }
                    ) {
                        *next += 1;
                        *next - 1
                    } else {
                        0
                    },
                )),
            };
            self.0.push(action);
            Ok(ActionReceipt { message_id })
        }

        fn try_photo(
            &mut self,
            action: TelegramAction,
        ) -> Result<Option<ActionReceipt>, Self::Error> {
            match self.1.photo {
                Attempt::Deliver => self.execute(action).map(Some),
                Attempt::Skip => Ok(None),
                Attempt::Unconfirmed => Ok(Some(ActionReceipt { message_id: None })),
                Attempt::Fail => Err("synthetic photo failure"),
            }
        }

        fn try_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
            if self.1.edit == Attempt::Deliver {
                return self.execute(action).map(|_receipt| true);
            }
            self.0.push(action);
            if self.1.edit == Attempt::Fail {
                return Err("synthetic edit failure");
            }
            Ok(false)
        }

        fn try_video(
            &mut self,
            action: TelegramAction,
        ) -> Result<Option<ActionReceipt>, Self::Error> {
            if self.1.video == Attempt::Deliver {
                return self.execute(action).map(Some);
            }
            Ok(None)
        }

        fn try_animation(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
            if self.1.animation == Attempt::Deliver {
                return self.execute(action).map(|_receipt| true);
            }
            Ok(false)
        }

        fn try_invoice(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
            if self.1.invoice_refusals == 0 && self.1.invoice == Attempt::Deliver {
                return self.execute(action).map(|_receipt| true);
            }
            self.0.push(action);
            if self.1.invoice_refusals > 0 {
                self.1.invoice_refusals -= 1;
                return Ok(false);
            }
            if self.1.invoice == Attempt::Fail {
                return Err("synthetic invoice transport failure");
            }
            Ok(false)
        }
    }

    fn photo_actions(attempt: Attempt) -> Actions {
        attempt_actions(attempt, ActionScript::photo)
    }

    #[derive(Clone, Copy)]
    enum DeliveryOutcome {
        Confirmed,
        Unconfirmed,
        Rejected,
    }

    fn delivery_actions(outcome: DeliveryOutcome) -> Actions {
        match outcome {
            DeliveryOutcome::Confirmed => Actions::default(),
            DeliveryOutcome::Unconfirmed => Actions::scripted(ActionScript {
                receipts: Receipts::Fixed(None),
                ..ActionScript::default()
            }),
            DeliveryOutcome::Rejected => failing_actions(
                ActionKind::Any,
                usize::MAX,
                "synthetic delivery rejection",
                true,
            ),
        }
    }

    #[derive(Default)]
    struct State {
        incoming: Vec<IncomingCommandWritePlan>,
        outgoing: Vec<OutgoingCommandWritePlan>,
        /// Rejects every write, like unavailable message storage.
        failing: bool,
    }

    impl State {
        fn failing() -> Self {
            Self {
                failing: true,
                ..Self::default()
            }
        }
    }

    impl MessageStateSink for State {
        type Error = &'static str;

        fn record_incoming(&mut self, plan: &IncomingCommandWritePlan) -> Result<(), Self::Error> {
            if self.failing {
                return Err("synthetic incoming failure");
            }
            self.incoming.push(plan.clone());
            Ok(())
        }

        fn record_outgoing(&mut self, plan: &OutgoingCommandWritePlan) -> Result<(), Self::Error> {
            if self.failing {
                return Err("synthetic outgoing failure");
            }
            self.outgoing.push(plan.clone());
            Ok(())
        }
    }

    struct AiSource {
        metadata: Option<AiReplyMetadata>,
        metadata_error: Option<String>,
        preparation: Option<Result<AiPreparation, String>>,
        media_preparation: Option<Result<AiPreparation, String>>,
        summary_preparation: Option<Result<AiPreparation, String>>,
        tokens: Vec<String>,
        prepared: Rc<RefCell<Vec<AiConversationInput>>>,
        ignored: Rc<RefCell<Vec<AiConversationInput>>>,
        deliveries: Rc<RefCell<Vec<AiDelivery>>>,
        delivery_error: Option<String>,
        ignored_error: Option<String>,
        /// Whether the synthetic turn passes its credit check (and so emits
        /// `Admitted`) before preparing, mirroring the native conversation.
        admit: bool,
    }

    impl AiConversationSource for AiSource {
        fn reply_metadata(
            &mut self,
            _chat_id: &str,
            _message_id: &str,
        ) -> Result<Option<AiReplyMetadata>, String> {
            if let Some(error) = self.metadata_error.clone() {
                return Err(error);
            }
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
            let preparation = self.prepare(input);
            for token in &self.tokens {
                on_token(token)?;
            }
            preparation
        }

        fn prepare_streaming_events(
            &mut self,
            input: AiConversationInput,
            on_event: &mut dyn FnMut(AiStreamEvent) -> Result<(), String>,
        ) -> Result<AiPreparation, String> {
            if self.admit && !input.spontaneous {
                on_event(AiStreamEvent::Admitted)?;
            }
            self.prepare_streaming(input, &mut |token| {
                on_event(AiStreamEvent::FinalText(token.to_owned()))
            })
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
            self.ignored_error.clone().map_or(Ok(()), Err)
        }

        fn complete_delivery(&mut self, delivery: AiDelivery) -> Result<(), String> {
            self.deliveries.borrow_mut().push(delivery);
            self.delivery_error.clone().map_or(Ok(()), Err)
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
                metadata_error: None,
                preparation: Some(preparation),
                media_preparation: None,
                summary_preparation: None,
                tokens: Vec::new(),
                prepared: Rc::clone(&prepared),
                ignored: Rc::clone(&ignored),
                deliveries: Rc::clone(&deliveries),
                delivery_error: None,
                ignored_error: None,
                admit: true,
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
        failing: bool,
    }

    impl RandomSource for Samples {
        type Error = &'static str;

        fn choice_index(&mut self, _upper_exclusive: usize) -> Result<usize, Self::Error> {
            if self.failing {
                return Err("synthetic random failure");
            }
            Ok(self.choice_index)
        }

        fn inclusive_integer(
            &mut self,
            _start: &BigInt,
            _end: &BigInt,
        ) -> Result<BigInt, Self::Error> {
            if self.failing {
                return Err("synthetic random failure");
            }
            Ok(self.integer.clone())
        }
    }

    fn random() -> Samples {
        Samples {
            choice_index: 1,
            integer: BigInt::from(2_u8),
            failing: false,
        }
    }

    struct Links {
        replacement: LinkReplacement,
        context: Option<String>,
        oversized_video: Option<Vec<u8>>,
        diagnostics: Vec<String>,
        calls: Vec<(String, i64)>,
        preview: Option<String>,
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

        fn preview_context(&mut self, text: &str) -> Option<String> {
            self.preview
                .as_ref()
                .map(|preview| format!("{preview} for {text}"))
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
            preview: Some("PREVIEW".to_owned()),
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

    struct FallibleTasks {
        list_result: Result<Vec<ScheduledTask>, String>,
        cancel_result: Result<bool, String>,
    }

    impl ScheduledTaskSource for FallibleTasks {
        fn list(&mut self, _chat_id: &str) -> Result<Vec<ScheduledTask>, String> {
            self.list_result.clone()
        }

        fn cancel(&mut self, _task_id: &TaskId, _chat_id: &str) -> Result<bool, String> {
            self.cancel_result.clone()
        }
    }

    struct SequencedTasks {
        lists: VecDeque<Result<Vec<ScheduledTask>, String>>,
    }

    impl ScheduledTaskSource for SequencedTasks {
        fn list(&mut self, _chat_id: &str) -> Result<Vec<ScheduledTask>, String> {
            self.lists.pop_front().unwrap_or(Ok(Vec::new()))
        }

        fn cancel(&mut self, _task_id: &TaskId, _chat_id: &str) -> Result<bool, String> {
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

    fn dispatcher() -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization> {
        NativeDispatcher::new(
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
    }

    fn delivery_dispatcher(
        source: AiSource,
        outcome: DeliveryOutcome,
    ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization> {
        NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig {
                    language: "en".to_owned(),
                    ..ChatConfig::default()
                }),
                chat_ids: Vec::new(),
            },
            delivery_actions(outcome),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_ai_conversation_source(Box::new(source))
    }

    fn sent_message(action: &TelegramAction) -> Option<&SendMessage> {
        match action {
            TelegramAction::SendMessage(message) => Some(message),
            _ => None,
        }
    }

    /// Messages a fake sink recorded, in order, skipping every other action.
    fn sent_messages(actions: &[TelegramAction]) -> Vec<&SendMessage> {
        actions.iter().filter_map(sent_message).collect()
    }

    fn sent_texts(actions: &[TelegramAction]) -> Vec<&str> {
        sent_messages(actions)
            .into_iter()
            .map(|message| message.text.as_str())
            .collect()
    }

    /// The first recorded action, which must be a sent message; indexing
    /// fails the test otherwise.
    fn first_sent(actions: &[TelegramAction]) -> &SendMessage {
        sent_messages(&actions[..1])[0]
    }

    /// The last recorded action, which must be a sent message.
    fn last_sent(actions: &[TelegramAction]) -> &SendMessage {
        sent_messages(&actions[actions.len() - 1..])[0]
    }

    /// Media commands show "typing" first; returns what follows it.
    fn after_typing(actions: &[TelegramAction]) -> &[TelegramAction] {
        assert!(
            matches!(actions.first(), Some(TelegramAction::SendTyping { .. })),
            "expected a typing status first: {actions:?}"
        );
        &actions[1..]
    }

    /// The only recorded action, which must be a sent message.
    fn only_sent(actions: &[TelegramAction]) -> &SendMessage {
        assert_eq!(actions.len(), 1, "expected one action: {actions:?}");
        sent_messages(actions)[0]
    }

    fn update(text: &str, language: Option<&str>) -> IncomingUpdate {
        message_update(text, language, |_| {})
    }

    /// A private-chat message update adjusted by `edit` before it is wrapped.
    fn message_update(
        text: &str,
        language: Option<&str>,
        edit: impl FnOnce(&mut IncomingMessage),
    ) -> IncomingUpdate {
        let mut message = incoming_message(text, language);
        edit(&mut message);
        wrap_message(message)
    }

    fn wrap_message(message: IncomingMessage) -> IncomingUpdate {
        IncomingUpdate {
            update_id: 99,
            event: IncomingEvent::Message(Box::new(message)),
        }
    }

    fn incoming_message(text: &str, language: Option<&str>) -> IncomingMessage {
        IncomingMessage {
            message_id: Some(MessageId(7)),
            chat_id: Some(ChatId(-42)),
            chat_type: Some("private".to_owned()),
            chat_title: None,
            sender_id: Some(UserId(88)),
            sender_first_name: Some("Synthetic".to_owned()),
            sender_last_name: None,
            sender_username: Some("tester".to_owned()),
            sender_language_code: language.map(ToOwned::to_owned),
            sender_is_bot: false,
            text_mentions: Vec::new(),
            has_reply: false,
            replied_message_id: None,
            replied_sender_first_name: None,
            replied_sender_username: None,
            replied_sender_id: None,
            replied_sender_is_bot: false,
            replied_text: None,
            visual_media_kind: None,
            audio_media_kind: None,
            audio_duration_seconds: None,
            attachment: None,
            content: Some(MessageContent {
                text: text.to_owned(),
                photo_file_id: None,
                audio_file_id: None,
            }),
        }
    }

    fn callback_update(data: &str, chat_type: &str, language: Option<&str>) -> IncomingUpdate {
        callback_update_for_message(data, chat_type, language, 7)
    }

    fn callback_update_for_message(
        data: &str,
        chat_type: &str,
        language: Option<&str>,
        message_id: i64,
    ) -> IncomingUpdate {
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
                    "message_id": message_id,
                    "chat": {"id": -42, "type": chat_type},
                }),
            ),
        ]);
        IncomingUpdate {
            update_id: 100,
            event: IncomingEvent::CallbackQuery(callback),
        }
    }

    fn callback_update_with_context(
        data: &str,
        chat_id: Value,
        chat_type: &str,
        message_id: i64,
        user_id: Option<i64>,
        language: Option<&str>,
        callback_id: Option<&str>,
    ) -> IncomingUpdate {
        let mut callback = Map::from_iter([
            ("data".to_owned(), json!(data)),
            (
                "from".to_owned(),
                user_id.map_or_else(
                    || json!({}),
                    |id| json!({"id": id, "language_code": language}),
                ),
            ),
            (
                "message".to_owned(),
                json!({
                    "message_id": message_id,
                    "chat": {"id": chat_id, "type": chat_type},
                }),
            ),
        ]);
        if let Some(callback_id) = callback_id {
            callback.insert("id".to_owned(), json!(callback_id));
        }
        IncomingUpdate {
            update_id: 101,
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
            operation_id: &str,
        ) -> Result<TransferResult, String> {
            // One id per Telegram message: "transfer:<chat>:<message>".
            assert!(
                operation_id.starts_with(&format!("transfer:{chat_id}:"))
                    && operation_id.len() > format!("transfer:{chat_id}:").len()
            );
            self.calls.borrow_mut().push((user_id, chat_id, amount));
            self.result.clone()
        }

        fn transfer_to_user(
            &mut self,
            user_id: i64,
            recipient_id: i64,
            amount: i64,
            operation_id: &str,
        ) -> Result<TransferResult, String> {
            // Keyed by chat and message like group transfers.
            assert!(
                operation_id.starts_with("transfer:") && operation_id.matches(':').count() == 2
            );
            self.calls
                .borrow_mut()
                .push((user_id, recipient_id, amount));
            self.result.clone()
        }
    }

    type AdminCreditCalls = Rc<RefCell<Vec<(i64, i64)>>>;

    struct AdminCredits {
        result: Result<i64, String>,
        calls: AdminCreditCalls,
    }

    impl AdminCreditSink for AdminCredits {
        fn mint(&mut self, user_id: i64, amount: i64, operation_id: &str) -> Result<i64, String> {
            assert!(
                operation_id.starts_with("printcredits:") && operation_id.matches(':').count() == 2
            );
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
            let next = (!self.results.is_empty()).then(|| self.results.remove(0));
            next.unwrap_or(Err("no synthetic price".to_owned()))
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

    struct Signals {
        query_load: TokenSignalLoad,
        token_load: TokenSignalLoad,
        photo: Result<Vec<u8>, String>,
        state: Option<SignalState>,
        queries: Rc<RefCell<Vec<SignalQuery>>>,
        saved: Rc<RefCell<Vec<(String, SignalState)>>>,
    }

    impl TokenSignalSource for Signals {
        fn load(&mut self, query: &SignalQuery) -> TokenSignalLoad {
            self.queries.borrow_mut().push(query.clone());
            self.query_load.clone()
        }

        fn load_token(&mut self, _token: &TokenAddress) -> TokenSignalLoad {
            self.token_load.clone()
        }

        fn render_period_photo(
            &mut self,
            _: &TokenSignal,
            _: &str,
            _: i64,
        ) -> Result<Vec<u8>, String> {
            self.photo.clone()
        }

        fn load_state(&mut self, _signal_id: &str) -> Result<Option<SignalState>, String> {
            Ok(self.state.clone())
        }

        fn save_state(&mut self, signal_id: &str, state: &SignalState) -> Result<(), String> {
            self.saved
                .borrow_mut()
                .push((signal_id.to_owned(), state.clone()));
            Ok(())
        }
    }

    struct WideningSignals {
        query_load: TokenSignalLoad,
        photo: Result<Vec<u8>, String>,
        periods: Rc<RefCell<Vec<String>>>,
        state: Option<SignalState>,
    }

    impl TokenSignalSource for WideningSignals {
        fn load(&mut self, _query: &SignalQuery) -> TokenSignalLoad {
            self.query_load.clone()
        }

        fn load_token(&mut self, _token: &TokenAddress) -> TokenSignalLoad {
            TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            }
        }

        fn render_period_photo(
            &mut self,
            _signal: &TokenSignal,
            period: &str,
            _now: i64,
        ) -> Result<Vec<u8>, String> {
            self.periods.borrow_mut().push(period.to_owned());
            self.photo.clone()
        }

        fn period_candles(
            &mut self,
            _signal: &TokenSignal,
            period: &str,
            now: i64,
        ) -> Result<Vec<Vec<f64>>, String> {
            if period == "24h" {
                Ok(vec![vec![now as f64, 1.0, 1.0, 1.0, 100.0]])
            } else {
                Ok(vec![
                    vec![now as f64 - 172_800.0, 1.0, 1.0, 1.0, 100.0],
                    vec![now as f64, 2.0, 2.0, 2.0, 125.0],
                ])
            }
        }

        fn load_state(&mut self, _signal_id: &str) -> Result<Option<SignalState>, String> {
            Ok(self.state.clone())
        }

        fn save_state(&mut self, _signal_id: &str, state: &SignalState) -> Result<(), String> {
            self.state = Some(state.clone());
            Ok(())
        }
    }

    struct StatefulSignals {
        signal: TokenSignal,
        token_results: Rc<RefCell<VecDeque<TokenSignalLoad>>>,
        state: Rc<RefCell<Option<SignalState>>>,
        saved: Rc<RefCell<Vec<(String, SignalState)>>>,
        periods: Rc<RefCell<Vec<String>>>,
    }

    impl TokenSignalSource for StatefulSignals {
        fn load(&mut self, _query: &SignalQuery) -> TokenSignalLoad {
            TokenSignalLoad {
                signal: Some(self.signal.clone()),
                diagnostics: Vec::new(),
            }
        }

        fn load_token(&mut self, _token: &TokenAddress) -> TokenSignalLoad {
            self.token_results
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| TokenSignalLoad {
                    signal: Some(self.signal.clone()),
                    diagnostics: Vec::new(),
                })
        }

        fn render_period_photo(
            &mut self,
            _signal: &TokenSignal,
            period: &str,
            _now: i64,
        ) -> Result<Vec<u8>, String> {
            self.periods.borrow_mut().push(period.to_owned());
            Ok(b"stateful-token-card".to_vec())
        }

        fn period_candles(
            &mut self,
            _signal: &TokenSignal,
            period: &str,
            now: i64,
        ) -> Result<Vec<Vec<f64>>, String> {
            let span = super::period_seconds(period);
            Ok(vec![
                vec![(now - span) as f64, 1.0, 1.0, 1.0, 100.0],
                vec![now as f64, 2.0, 2.0, 2.0, 125.0],
            ])
        }

        fn load_state(&mut self, _signal_id: &str) -> Result<Option<SignalState>, String> {
            Ok(self.state.borrow().clone())
        }

        fn save_state(&mut self, signal_id: &str, state: &SignalState) -> Result<(), String> {
            self.saved
                .borrow_mut()
                .push((signal_id.to_owned(), state.clone()));
            *self.state.borrow_mut() = Some(state.clone());
            Ok(())
        }

        fn clear_state(&mut self, _signal_id: &str) -> Result<(), String> {
            *self.state.borrow_mut() = None;
            Ok(())
        }
    }

    struct FallibleSignals {
        query_load: TokenSignalLoad,
        token_load: TokenSignalLoad,
        photo: Result<Vec<u8>, String>,
        state: Result<Option<SignalState>, String>,
        save_error: Option<String>,
    }

    impl TokenSignalSource for FallibleSignals {
        fn load(&mut self, _query: &SignalQuery) -> TokenSignalLoad {
            self.query_load.clone()
        }

        fn load_token(&mut self, _token: &TokenAddress) -> TokenSignalLoad {
            self.token_load.clone()
        }

        fn render_period_photo(
            &mut self,
            _signal: &TokenSignal,
            _period: &str,
            _now: i64,
        ) -> Result<Vec<u8>, String> {
            self.photo.clone()
        }

        fn period_candles(
            &mut self,
            _signal: &TokenSignal,
            _period: &str,
            _now: i64,
        ) -> Result<Vec<Vec<f64>>, String> {
            Err("synthetic period candles failure".into())
        }

        fn load_state(&mut self, _signal_id: &str) -> Result<Option<SignalState>, String> {
            self.state.clone()
        }

        fn save_state(&mut self, _signal_id: &str, _state: &SignalState) -> Result<(), String> {
            self.save_error.clone().map_or(Ok(()), Err)
        }
    }

    fn token_signal() -> TokenSignal {
        TokenSignal {
            token: TokenAddress {
                chain_id: "solana".to_owned(),
                network: "solana".to_owned(),
                tag: "SOL".to_owned(),
                address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            },
            pair: TokenPair {
                chain_id: "solana".to_owned(),
                url: "https://dexscreener.com/solana/pair".to_owned(),
                pair_address: "pair1".to_owned(),
                base_token: PairToken {
                    address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
                    name: "Synthetic Token".to_owned(),
                    symbol: "SYN".to_owned(),
                },
                price_usd: json!("0.01"),
                price_change: PairPriceChange {
                    h1: json!(1),
                    h24: json!(2),
                },
                market_cap: json!(1_000_000),
                volume: PairVolume {
                    h24: json!(100_000),
                },
                liquidity: PairLiquidity { usd: json!(50_000) },
                txns: PairTransactionWindows {
                    h1: PairTransactions {
                        buys: json!(10),
                        sells: json!(5),
                    },
                },
                ..TokenPair::default()
            },
            candles: vec![
                vec![1.0, 1.0, 2.0, 0.8, 1.5],
                vec![2.0, 1.5, 2.5, 1.2, 1.3],
                vec![3.0, 1.3, 1.8, 1.0, 1.7],
                vec![4.0, 1.7, 2.2, 1.4, 2.0],
                vec![5.0, 2.0, 2.4, 1.8, 2.1],
            ],
            supply: None,
            token_image_url: None,
            socials: std::collections::BTreeMap::new(),
            pump: None,
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
        let message = first_sent(&dispatcher.actions.0);
        assert_eq!(message.text, "101 in base 2 is 5 in base 10");
        assert_eq!(dispatcher.last_outcome(), Some(DispatchOutcome::Handled));
    }

    #[test]
    fn compatibility_digits_are_native_and_missing_ai_is_explicit() {
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
            Err(DispatchError::MissingService("AI conversation"))
        );
        assert_eq!(
            dispatcher.dispatch(update("/convertbase １２, 10, 2", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(update("/random １-３", None)),
            Ok(DispatchOutcome::Handled)
        );
        let replied = message_update("/time", None, |message| {
            message.has_reply = true;
        });
        assert_eq!(
            dispatcher.dispatch(replied),
            Err(DispatchError::MissingService("AI conversation"))
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
                sender_is_bot: false,
                text_mentions: Vec::new(),
                has_reply: false,
                replied_message_id: None,
                replied_sender_first_name: None,
                replied_sender_username: None,
                replied_sender_id: None,
                replied_sender_is_bot: false,
                replied_text: None,
                visual_media_kind: None,
                audio_media_kind: None,
                audio_duration_seconds: None,
                attachment: None,
                content: None,
            })),
        };
        assert_eq!(
            dispatcher.dispatch(incomplete),
            Ok(DispatchOutcome::Unsupported)
        );
        let texts = sent_texts(&dispatcher.actions.0);
        assert_eq!(
            texts,
            ["Ahí tenés, boludo: １２ en base 10 es 1100 en base 2", "2"]
        );
    }

    #[test]
    fn optional_native_routes_report_the_exact_missing_service() {
        let make = || {
            NativeDispatcher::new(
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
        };

        assert_eq!(
            make().dispatch(update("https://x.com/example/status/1", Some("en"))),
            Err(DispatchError::MissingService("link replacement"))
        );
        assert_eq!(
            make().dispatch(update(
                "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump",
                Some("en"),
            )),
            Err(DispatchError::MissingService("token signals"))
        );
        assert_eq!(
            make().dispatch(callback_update("task:del:task0001", "private", Some("en"))),
            Err(DispatchError::MissingService("scheduled tasks"))
        );
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
        // The draft placeholder is edited into the final answer.
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(draft),
                TelegramAction::EditMessage { message_id: MessageId(700), text, .. },
            ] if draft.text == "Thinking."
                && draft.reply_to_message_id == Some(MessageId(7))
                && text == "native answer"
        ));
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
    fn native_ai_failure_sends_the_localized_retry_reply() {
        let (source, _observations) = ai_source(Err("synthetic provider failure".to_owned()));
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
            dispatcher.dispatch(update("answer me", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        // The draft placeholder is removed before the retry reply.
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(draft),
                TelegramAction::DeleteMessage { message_id: MessageId(700), .. },
                TelegramAction::SendMessage(message),
            ] if draft.text == "Thinking."
                && message.text == "I could not answer. Try again"
                && message.reply_to_message_id == Some(MessageId(7))
        ));
        assert_eq!(
            dispatcher.state_diagnostics(),
            ["AI conversation: synthetic provider failure"]
        );
    }

    #[test]
    fn explicit_media_command_uses_its_native_transaction_and_command_state() {
        let (mut source, (prepared, ignored, deliveries)) = ai_source(Ok(AiPreparation::silent()));
        source.media_preparation = Some(Ok(AiPreparation::Reply {
            text: "synthetic transcript".to_owned(),
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
        let incoming = message_update("/transcript", Some("en"), |message| {
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(6));
            message.audio_media_kind = Some("voice".to_owned());
            message.audio_duration_seconds = Some(4);
            if let Some(content) = message.content.as_mut() {
                content.audio_file_id = Some("voice-1".to_owned());
            }
        });

        assert_eq!(dispatcher.dispatch(incoming), Ok(DispatchOutcome::Handled));
        let prepared = prepared.borrow();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].command, "/transcript");
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
                .is_some_and(|metadata| metadata.payload.contains("/transcript"))
        );
        let message = only_sent(after_typing(&dispatcher.actions.0));
        assert_eq!(message.text, "synthetic transcript");
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
    fn long_youtube_transcript_is_sent_as_a_complete_text_document() {
        for completion_id in [None, Some("youtube-paid".to_owned())] {
            let transcript = "texto sintético 🦀 ".repeat(300);
            assert!(transcript.chars().count() > MAX_TELEGRAM_TEXT_LENGTH);
            let (mut source, (_prepared, _ignored, deliveries)) =
                ai_source(Ok(AiPreparation::silent()));
            source.media_preparation = Some(Ok(AiPreparation::Reply {
                text: transcript.clone(),
                completion_id: completion_id.clone(),
                diagnostics: Vec::new(),
            }));
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
                dispatcher.dispatch(update("/transcript https://youtu.be/synthetic", None)),
                Ok(DispatchOutcome::Handled)
            );
            // Long transcripts are sent as a complete text document.
            assert!(matches!(
                after_typing(&dispatcher.actions.0),
                [TelegramAction::SendDocument {
                    document,
                    file_name,
                    reply_to_message_id: Some(MessageId(7)),
                    caption,
                    ..
                }] if document.as_ref() == transcript.as_bytes()
                    && file_name == "transcript.txt"
                    && caption.is_empty()
            ));
            let expected = completion_id
                .map(|completion_id| AiDelivery {
                    completion_id,
                    delivered: true,
                    sent_message_id: Some(MessageId(700)),
                })
                .into_iter()
                .collect::<Vec<_>>();
            assert_eq!(*deliveries.borrow(), expected);
            assert_eq!(dispatcher.state.incoming.len(), 1);
            assert_eq!(dispatcher.state.outgoing.len(), 1);
        }
    }

    #[test]
    fn media_command_failure_sends_the_exact_localized_error() {
        let (mut source, _observations) = ai_source(Ok(AiPreparation::silent()));
        source.media_preparation = Some(Err("synthetic media failure".to_owned()));
        // The typing status is rejected too; the error reply still goes out
        // and names the command the user sent.
        let mut dispatcher = configured(
            ChatConfig::default(),
            failing_actions(ActionKind::Any, 1, "synthetic typing failure", false),
        )
        .with_ai_conversation_source(Box::new(source));
        assert_eq!(
            dispatcher.dispatch(update("/describe", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&dispatcher.actions.0),
            ["Se trabó el /describe. Probá más tarde"]
        );
        assert_eq!(
            dispatcher.state_diagnostics(),
            [
                "media command typing status: synthetic typing failure",
                "media command: synthetic media failure"
            ]
        );
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
        assert_eq!(dispatcher.actions.0.len(), 3);
        assert!(matches!(
            &dispatcher.actions.0[0],
            TelegramAction::SendMessage(message) if message.text == "Thinking."
        ));
        assert!(matches!(
            &dispatcher.actions.0[1],
            TelegramAction::EditMessageNoPreview { text, .. } if text == "raw summary"
        ));
        assert!(matches!(
            &dispatcher.actions.0[2],
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
    fn summary_failure_sends_the_exact_localized_error() {
        let (mut source, _observations) = ai_source(Ok(AiPreparation::silent()));
        source.summary_preparation = Some(Err("synthetic summary failure".to_owned()));
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
            dispatcher.dispatch(update("/summary", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(draft),
                TelegramAction::DeleteMessage { message_id: MessageId(700), .. },
                TelegramAction::SendMessage(message),
            ] if draft.text == "Thinking."
                && message.text == "I could not generate the summary. Try again"
        ));
        assert_eq!(
            dispatcher.state_diagnostics(),
            ["summary command: synthetic summary failure"]
        );
    }

    #[test]
    fn media_and_summary_commands_handle_empty_and_silent_preparations() {
        #[derive(Clone, Copy)]
        enum CommandCase {
            MediaNone,
            MediaSilent,
            SummaryNone,
            SummarySilent,
        }

        for case in [
            CommandCase::MediaNone,
            CommandCase::MediaSilent,
            CommandCase::SummaryNone,
            CommandCase::SummarySilent,
        ] {
            let (mut source, _observations) = ai_source(Ok(AiPreparation::silent()));
            let command = match case {
                CommandCase::MediaNone => "/transcribe",
                CommandCase::MediaSilent => {
                    source.media_preparation = Some(Ok(AiPreparation::Silent {
                        diagnostics: vec!["synthetic silent media".to_owned()],
                    }));
                    "/transcribe"
                }
                CommandCase::SummaryNone => "/summary",
                CommandCase::SummarySilent => {
                    source.summary_preparation = Some(Ok(AiPreparation::Silent {
                        diagnostics: vec!["synthetic silent summary".to_owned()],
                    }));
                    "/summary"
                }
            };
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
                dispatcher.dispatch(update(command, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            match case {
                CommandCase::MediaNone => {
                    assert_eq!(after_typing(&dispatcher.actions.0).len(), 1);
                }
                CommandCase::MediaSilent => {
                    assert!(after_typing(&dispatcher.actions.0).is_empty());
                }
                CommandCase::SummaryNone => {
                    assert_eq!(dispatcher.actions.0.len(), 3);
                    assert!(matches!(
                        &dispatcher.actions.0[0],
                        TelegramAction::SendMessage(message) if message.text == "Thinking."
                    ));
                    assert!(matches!(
                        &dispatcher.actions.0[1],
                        TelegramAction::DeleteMessage { .. }
                    ));
                    assert!(matches!(
                        &dispatcher.actions.0[2],
                        TelegramAction::SendMessage(message)
                            if message.text == "I could not generate the summary. Try again"
                    ));
                }
                CommandCase::SummarySilent => {
                    assert_eq!(dispatcher.actions.0.len(), 2);
                    assert!(matches!(
                        &dispatcher.actions.0[0],
                        TelegramAction::SendMessage(message) if message.text == "Thinking."
                    ));
                    assert!(matches!(
                        &dispatcher.actions.0[1],
                        TelegramAction::DeleteMessage { .. }
                    ));
                }
            }
        }
    }

    #[test]
    fn ai_delivery_records_confirmed_unconfirmed_and_rejected_outcomes() {
        let cases = [
            (DeliveryOutcome::Confirmed, true, false),
            (DeliveryOutcome::Unconfirmed, false, false),
            (DeliveryOutcome::Rejected, false, true),
        ];
        for (outcome, delivered, rejected) in cases {
            let (mut source, (_prepared, _ignored, deliveries)) =
                ai_source(Ok(AiPreparation::Reply {
                    text: "synthetic answer".to_owned(),
                    completion_id: Some("synthetic-completion".to_owned()),
                    diagnostics: Vec::new(),
                }));
            source.delivery_error = Some("synthetic completion failure".to_owned());
            let mut dispatcher = delivery_dispatcher(source, outcome);
            let result = dispatcher.dispatch(update("synthetic question", Some("en")));
            if rejected {
                assert!(matches!(
                    result,
                    Err(DispatchError::Action("synthetic delivery rejection"))
                ));
            } else {
                assert_eq!(result, Ok(DispatchOutcome::Handled));
            }
            assert_eq!(deliveries.borrow().len(), 1);
            assert_eq!(deliveries.borrow()[0].delivered, delivered);
            assert!(
                dispatcher
                    .state_diagnostics()
                    .iter()
                    .any(|entry| entry.contains("completion"))
            );
            if !delivered && !rejected {
                assert!(
                    dispatcher
                        .state_diagnostics()
                        .iter()
                        .any(|entry| entry.contains("no message identifier"))
                );
            }
        }
    }

    #[test]
    fn media_delivery_records_unconfirmed_rejected_and_completion_failures() {
        for (outcome, rejected) in [
            (DeliveryOutcome::Unconfirmed, false),
            (DeliveryOutcome::Rejected, true),
        ] {
            let (mut source, (_prepared, _ignored, deliveries)) =
                ai_source(Ok(AiPreparation::silent()));
            source.media_preparation = Some(Ok(AiPreparation::Reply {
                text: "synthetic transcript".to_owned(),
                completion_id: Some("synthetic-media".to_owned()),
                diagnostics: Vec::new(),
            }));
            source.delivery_error = Some("synthetic media completion failure".to_owned());
            let mut dispatcher = delivery_dispatcher(source, outcome);
            let result = dispatcher.dispatch(update("/transcribe", Some("en")));
            if rejected {
                assert!(matches!(
                    result,
                    Err(DispatchError::Action("synthetic delivery rejection"))
                ));
            } else {
                assert_eq!(result, Ok(DispatchOutcome::Handled));
            }
            assert_eq!(deliveries.borrow().len(), 1);
            assert!(!deliveries.borrow()[0].delivered);
            assert!(
                dispatcher
                    .state_diagnostics()
                    .iter()
                    .any(|entry| entry.contains("completion"))
            );
        }
    }

    #[test]
    fn summary_delivery_records_unconfirmed_rejected_and_completion_failures() {
        for (outcome, rejected) in [
            (DeliveryOutcome::Confirmed, false),
            (DeliveryOutcome::Unconfirmed, false),
            (DeliveryOutcome::Rejected, true),
        ] {
            let (mut source, (_prepared, _ignored, deliveries)) =
                ai_source(Ok(AiPreparation::silent()));
            source.summary_preparation = Some(Ok(AiPreparation::Reply {
                text: "synthetic summary".to_owned(),
                completion_id: Some("synthetic-summary".to_owned()),
                diagnostics: Vec::new(),
            }));
            source.delivery_error = Some("synthetic summary completion failure".to_owned());
            let mut dispatcher = delivery_dispatcher(source, outcome);
            let result = dispatcher.dispatch(update("/summary", Some("en")));
            if rejected {
                assert!(matches!(
                    result,
                    Err(DispatchError::Action("synthetic delivery rejection"))
                ));
            } else {
                assert_eq!(result, Ok(DispatchOutcome::Handled));
            }
            assert_eq!(deliveries.borrow().len(), 1);
            assert_eq!(
                deliveries.borrow()[0].delivered,
                !rejected && matches!(outcome, DeliveryOutcome::Confirmed)
            );
            if matches!(outcome, DeliveryOutcome::Confirmed) {
                assert!(
                    dispatcher
                        .state_diagnostics()
                        .iter()
                        .any(|entry| entry.contains("summary delivery completion"))
                );
            }
        }
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
        assert_eq!(dispatcher.actions.0.len(), 3);
        assert!(matches!(
            &dispatcher.actions.0[0],
            TelegramAction::SendMessage(message) if message.text == "Thinking."
        ));
        assert!(matches!(
            &dispatcher.actions.0[1],
            TelegramAction::EditMessageNoPreview { text, .. } if text == "raw "
        ));
        assert!(matches!(
            &dispatcher.actions.0[2],
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
                failing: false,
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
        let ordinary = message_update("ordinary group message", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(dispatcher.dispatch(ordinary), Ok(DispatchOutcome::Handled));

        let followup = message_update("and why?", None, |message| {
            message.chat_type = Some("group".to_owned());
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(3));
            message.replied_sender_first_name = Some("Gordo".to_owned());
            message.replied_sender_username = Some("mybot".to_owned());
            message.replied_text = Some("command answer".to_owned());
        });
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
    fn media_only_replies_to_the_bot_follow_the_chat_setting() {
        for (ignore_media_replies, text, attachment, answered) in [
            (true, "", Some(Attachment::Sticker), false),
            (true, "", Some(Attachment::Animation), false),
            // A one-word caption is still text.
            (true, "jaja", Some(Attachment::Photo), true),
            (true, "", Some(Attachment::Voice), true),
            // No attachment of its own, like a dice or a location.
            (true, "", None, true),
            (false, "", Some(Attachment::Sticker), true),
            (false, "", Some(Attachment::Animation), true),
        ] {
            let (source, (prepared, ignored, _)) = ai_source(Ok(AiPreparation::silent()));
            let mut dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig {
                        ignore_media_replies,
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
            let reply = message_update(text, None, |message| {
                message.chat_type = Some("group".to_owned());
                message.has_reply = true;
                message.replied_message_id = Some(MessageId(3));
                message.replied_sender_username = Some("mybot".to_owned());
                message.attachment = attachment;
            });
            let case = format!("{ignore_media_replies} {text:?} {attachment:?}");
            assert_eq!(
                dispatcher.dispatch(reply),
                Ok(DispatchOutcome::Handled),
                "{case}"
            );
            assert_eq!(prepared.borrow().len(), usize::from(answered), "{case}");
            assert_eq!(ignored.borrow().len(), usize::from(!answered), "{case}");
        }
    }

    #[test]
    fn group_ai_commands_require_a_leading_slash() {
        let (source, (prepared, ignored, deliveries)) = ai_source(Ok(AiPreparation::silent()));
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

        let plain = message_update("che", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(dispatcher.dispatch(plain), Ok(DispatchOutcome::Handled));

        let command = message_update("/che seguís ahí?", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(dispatcher.dispatch(command), Ok(DispatchOutcome::Handled));

        let other_bot = message_update("/balance@otherbot", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(dispatcher.dispatch(other_bot), Ok(DispatchOutcome::Handled));

        assert_eq!(ignored.borrow().len(), 2);
        assert_eq!(ignored.borrow()[0].message_text, "che");
        assert_eq!(ignored.borrow()[1].command, "/balance@otherbot");
        assert_eq!(prepared.borrow().len(), 1);
        assert_eq!(prepared.borrow()[0].command, "/che");
        assert_eq!(prepared.borrow()[0].message_text, "seguís ahí?");
        assert!(deliveries.borrow().is_empty());
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(message),
                TelegramAction::DeleteMessage { .. },
            ] if message.text == "Pensando."
        ));
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

        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            failing_actions(
                ActionKind::Any,
                usize::MAX,
                "synthetic action failure",
                false,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        let result = dispatcher.dispatch(update("/convertbase 1,2,10", None));
        assert!(matches!(
            result,
            Err(DispatchError::Action("synthetic action failure"))
        ));
        assert_eq!(
            result.err().map(|error| error.to_string()),
            Some("could not execute Telegram action: synthetic action failure".to_owned())
        );
    }

    #[test]
    fn only_permanent_action_failures_skip_update_retries() {
        let dispatcher = dispatcher();
        assert_eq!(
            dispatcher.error_disposition(&DispatchError::Action("synthetic permanent failure")),
            HandlerErrorDisposition::DiscardUpdate
        );
        assert_eq!(
            dispatcher.error_disposition(&DispatchError::Action("synthetic action failure")),
            HandlerErrorDisposition::RetryUpdate
        );
        assert_eq!(
            dispatcher.error_disposition(&DispatchError::MissingService("synthetic")),
            HandlerErrorDisposition::RetryUpdate
        );
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
        let texts = sent_texts(&dispatcher.actions.0);
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
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.starts_with("Help\n\n"));
        assert!(message.reply_markup.is_some());
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn dispatches_ascii_emoji_and_japanese_command_conversion_natively() {
        let (source, _observations) = ai_source(Ok(AiPreparation::silent()));
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
            "mybot",
        )
        .with_ai_conversation_source(Box::new(source));
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
            Ok(DispatchOutcome::Handled)
        );
        let replied = message_update("/comando@mybot", Some("es"), |message| {
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(6));
            message.replied_text = Some("quoted content".to_owned());
        });
        assert_eq!(dispatcher.dispatch(replied), Ok(DispatchOutcome::Handled));
        let texts = sent_texts(&dispatcher.actions.0);
        assert_eq!(
            texts,
            vec![
                "/HELLO_SIGNODEEXCLAMACION_WORLD",
                "Send the text you want to convert",
                "/MOUSUGUDESU",
                "/QUOTED_CONTENT"
            ]
        );
        assert_eq!(dispatcher.state.incoming.len(), 4);
        assert_eq!(dispatcher.state.outgoing.len(), 4);
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
        let message = first_sent(&dispatcher.actions.0);
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
        let message = first_sent(&failed.actions.0);
        assert!(message.text.contains("No pude conseguir las variables"));

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
            Err(DispatchError::MissingService("BCRA market data"))
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
        let message = first_sent(&dispatcher.actions.0);
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
        let message = first_sent(&invalid.actions.0);
        assert!(message.text.contains("7d"));
        assert!(message.text.contains("No conozco el período"));

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
        let message = first_sent(&failed.actions.0);
        assert_eq!(
            message.text,
            "No pude traer las cotizaciones del dólar, boludo. Probá más tarde"
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
            missing.dispatch(update("/usd", None)),
            Err(DispatchError::MissingService("dollar market data"))
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
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.contains("Example City, Exampleland"));
        assert!(message.text.contains("Mostly clear, feels like"));
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
        let message = first_sent(&default.actions.0);
        assert_eq!(
            message.text,
            "No pude conseguir el clima de Buenos Aires. Probá más tarde"
        );
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
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.contains("Polymarket elections by liquidity"));
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
        let message = first_sent(&failed.actions.0);
        assert_eq!(
            message.text,
            "No pude traer las elecciones de Polymarket. Probá más tarde"
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
            Err(DispatchError::MissingService("election markets"))
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
            asset_type: String::new(),
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
        let message = first_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "AAPL: 205.5 USD (+1.25% 24h)\nUnknown: not found"
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
                chart: None,
                selection: None,
                no_assets_found: false,
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
        let message = first_sent(&dispatcher.actions.0);
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
            Err(DispatchError::MissingService("market prices"))
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
        let message = first_sent(&failed.actions.0);
        assert_eq!(
            message.text,
            "No pude traer el top de acciones. Probá de nuevo"
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
            Err(DispatchError::MissingService("stock prices"))
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
            asset_type: String::new(),
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
        let message = first_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "Brent: 98.15 USD (-8.78% 24h)\nWTI: 95.45 USD (+1.25% 24h)"
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
        let message = first_sent(&failed.actions.0);
        assert_eq!(
            message.text,
            "No pude traer el precio del petróleo, boludo. Probá más tarde"
        );
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
            Err(DispatchError::MissingService("oil prices"))
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
            Err(DispatchError::MissingService("weather"))
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
        let message = first_sent(&fallback.actions.0);
        assert_eq!(message.text, "Buenas noches, boludo");

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
                failing: false,
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
        let message = first_sent(&non_http.actions.0);
        assert_eq!(message.text, "cached greeting");
        assert_eq!(non_http.state.outgoing.len(), 1);
    }

    #[test]
    fn greeting_animation_delivery_failure_is_silent_and_missing_source_stays_legacy() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::scripted(ActionScript {
                receipts: Receipts::Fixed(None),
                animation: Attempt::Skip,
                ..ActionScript::default()
            }),
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
        // A later text reply without a confirmed message id is still
        // recorded against the requesting message.
        assert_eq!(
            dispatcher.dispatch(update("/time", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(dispatcher.state.incoming.len(), 2);

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
            Err(DispatchError::MissingService("greeting media"))
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
        let message = first_sent(&dispatcher.actions.0);
        assert!(
            message
                .text
                .starts_with("Rulos desde el oficial\nOficial: $1,440")
        );
        assert!(message.text.contains("Ganancia: +19,730 ARS"));
        assert!(
            message
                .text
                .contains("Ruta: USD→USDT BUENBIT, USDT→ARS BUENBIT")
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
        let message = first_sent(&failed.actions.0);
        assert_eq!(
            message.text,
            "I could not load dollar rates. Try again later"
        );
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
            Err(DispatchError::MissingService("rulo market data"))
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
        let message = first_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "Card and crypto arbitrage\nProfit: 62.68% (fee 0.5%)\n\nRates in ARS\nOfficial: 100\nUSDT: 195\nCard: 150\n\n100 USD card purchase\n= 15,000 ARS = 76.92 USDT\nProfit: 9,402.5 ARS / 48.22 USDT\nTotal: 24,402.5 ARS / 125.14 USDT"
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
        let texts = sent_texts(&dispatcher.actions.0);
        assert_eq!(
            texts,
            vec![
                "Mandá bien los datos: comisión entre 0 y 100 y monto de compra positivo",
                "No pude traer las cotizaciones del dólar, boludo. Probá más tarde"
            ]
        );
        assert_eq!(
            dispatcher.dispatch(update("/devo ０.５", None)),
            Ok(DispatchOutcome::Handled)
        );
        let message = last_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "Usá: /devo <comisión %>[, <monto de la compra en USD>]\nEjemplo: /devo 0.5, 100"
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
            Err(DispatchError::MissingService("dollar quotes"))
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
        let texts = sent_texts(&dispatcher.actions.0);
        assert_eq!(
            texts[0],
            "1 satoshi = $0.00050000 USD\n1 satoshi = $0.1000 ARS\n\n$1 USD = 2,000 sats\n$1 ARS = 10.000 sats"
        );
        assert!(texts[1].starts_with("Power law estimates BTC at "));
        assert!(texts[2].starts_with("The rainbow chart estimates BTC at "));
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
        let texts = sent_texts(&dispatcher.actions.0);
        assert_eq!(
            texts,
            vec![
                "No pude traer el precio de BTC para calcular power law. Probá más tarde",
                "No pude traer el precio de BTC en ARS. Probá más tarde"
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
            Err(DispatchError::MissingService("bitcoin prices"))
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
        let message = first_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "Minted 100.00 credits\nYour balance is 120.00"
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
        let message = first_sent(&denied.actions.0);
        assert_eq!(message.text, "This command is only for the admin");

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
        let message = first_sent(&failed.actions.0);
        assert_eq!(
            message.text,
            "Se trabó imprimiendo créditos. Probá de nuevo"
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
            Err(DispatchError::MissingService("admin credit minting"))
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
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.starts_with("Latest AI settlements:"));
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
                "No hay liquidaciones de IA recientes".to_owned(),
            ),
            (
                Err("synthetic read failure".to_owned()),
                "Se trabó leyendo el creditlog. Probá de nuevo".to_owned(),
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
            let message = first_sent(&dispatcher.actions.0);
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
            Err(DispatchError::MissingService("admin credit log"))
        );
        assert_eq!(
            missing.dispatch(update("/creditlog ２", None)),
            Err(DispatchError::MissingService("admin credit log"))
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
        // The update replies in English and then moves the chat's command
        // menu to English as well.
        assert!(matches!(
            dispatcher.actions.0.last(),
            Some(TelegramAction::SetCommands { commands, .. })
                if commands.first().is_some_and(|command| command.description == "ask me anything")
        ));
        let message = last_sent(&dispatcher.actions.0[..dispatcher.actions.0.len() - 1]);
        assert_eq!(message.text, "Done, I will speak English now");
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
        let group_update = message_update("/language en", None, |message| {
            message.chat_type = Some("supergroup".to_owned());
        });
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
        let group_update = message_update("/idioma en", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(
            dispatcher.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        let message = first_sent(&dispatcher.actions.0);
        assert_eq!(message.text, "Este comando es solo para admins del grupo");
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
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.starts_with("Settings"));
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(6)
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        // The chat's own menu is described in its configured language, not the app's.
        assert!(matches!(
            dispatcher.actions.0.last(),
            Some(TelegramAction::SetCommands {
                commands,
                language_code: None,
                scope: bot_core::telegram_actions::CommandScope::Chat(_),
            }) if commands[0].command == "ask" && commands[0].description == "ask me anything"
        ));
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
        let group_update = message_update("/configs", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(
            dispatcher.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.authorization.checks,
            [("-42".to_owned(), "88".to_owned())]
        );
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.starts_with("Configuración"));
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(10)
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        // An "auto" chat keeps Telegram's app-language and all-groups menus.
        assert!(
            !dispatcher
                .actions
                .0
                .iter()
                .any(|action| matches!(action, TelegramAction::SetCommands { .. }))
        );
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
        let group_update = message_update("/config", None, |message| {
            message.chat_type = Some("supergroup".to_owned());
        });
        assert_eq!(
            dispatcher.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        let message = first_sent(&dispatcher.actions.0);
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
                if markup.inline_keyboard.len() == 6
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
            Some(TelegramAction::EditMessage { text, .. }) if text.starts_with("Language")
        ));
        assert!(matches!(
            dispatcher.actions.0.last(),
            Some(TelegramAction::SetCommands {
                commands,
                scope: bot_core::telegram_actions::CommandScope::Chat(_),
                ..
            }) if commands[0].command == "ask" && commands[0].description == "ask me anything"
        ));

        let mut other_setting = NativeDispatcher::new(
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
        );
        assert_eq!(
            other_setting.dispatch(callback_update("cfg:link:off", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            !other_setting
                .actions
                .0
                .iter()
                .any(|action| matches!(action, TelegramAction::SetCommands { .. }))
        );
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
            dispatcher.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }] if text == "Este comando es solo para admins del grupo"
        ));
        assert!(dispatcher.state_diagnostics()[1].contains("callback_data=cfg:link:off"));
    }

    #[test]
    fn group_config_callback_denial_without_callback_id_stays_handled() -> Result<(), String> {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let denied = Authorization {
            is_admin: false,
            diagnostics: Vec::new(),
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
            dispatcher.dispatch(callback_update_with_context(
                "cfg:link:off",
                json!(-42),
                "group",
                7,
                Some(88),
                None,
                None,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.actions.0.is_empty());
        assert!(dispatcher.state_diagnostics()[0].contains("callback_data=cfg:link:off"));
        Ok(())
    }

    #[test]
    fn config_callback_uses_new_message_fallback_before_acknowledging() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::scripted(ActionScript {
                receipts: Receipts::Fixed(None),
                edit: Attempt::Skip,
                ..ActionScript::default()
            }),
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

        let mut unchanged = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            Actions::scripted(ActionScript {
                receipts: Receipts::Fixed(None),
                edit: Attempt::Skip,
                ..ActionScript::default()
            }),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            unchanged.dispatch(callback_update("cfg:link:reply", "private", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            unchanged.actions.0.as_slice(),
            [
                TelegramAction::EditMessage { .. },
                TelegramAction::AnswerCallback { .. }
            ]
        ));
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
            Err(DispatchError::MissingService("scheduled tasks"))
        );
        assert!(dispatcher.config.chat_ids.is_empty());
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn malformed_and_unknown_callbacks_are_acknowledged_natively() {
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
        );
        let malformed = Map::from_iter([
            ("id".to_owned(), json!("malformed-callback")),
            ("data".to_owned(), json!("cfg:language:en")),
            ("message".to_owned(), json!("invalid")),
        ]);
        assert_eq!(
            dispatcher.dispatch(IncomingUpdate {
                update_id: 101,
                event: IncomingEvent::CallbackQuery(malformed),
            }),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("unknown:value", "private", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(dispatcher.actions.0.len(), 2);
        assert!(
            dispatcher
                .actions
                .0
                .iter()
                .all(|action| matches!(action, TelegramAction::AnswerCallback { .. }))
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|value| value.starts_with("invalid callback query:"))
        );
    }

    #[test]
    fn task_list_aliases_render_native_keyboard_and_creation_uses_ai_transaction()
    -> Result<(), TaskStateError> {
        let cancellations = Rc::new(RefCell::new(Vec::new()));
        let (ai, (prepared, ignored, deliveries)) = ai_source(Ok(AiPreparation::Reply {
            text: "task scheduled".to_owned(),
            completion_id: Some("task-command-1".to_owned()),
            diagnostics: Vec::new(),
        }));
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
        }))
        .with_ai_conversation_source(Box::new(ai));

        assert_eq!(
            dispatcher.dispatch(update("/tasks", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.first(),
            Some(TelegramAction::SendMessage(_))
        ));
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.starts_with("Tareas"));
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        let callback = message
            .reply_markup
            .as_ref()
            .and_then(|keyboard| keyboard.inline_keyboard.first())
            .and_then(|row| row.first())
            .and_then(|button| button.callback_data.as_deref());
        assert_eq!(callback, Some("task:view:task0001"));

        assert_eq!(
            dispatcher.dispatch(update("/tarea create something", None)),
            Ok(DispatchOutcome::Handled)
        );
        let prepared = prepared.borrow();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].command, "/tarea");
        assert_eq!(prepared[0].message_text, "create something");
        assert!(ignored.borrow().is_empty());
        assert_eq!(
            deliveries.borrow().as_slice(),
            [AiDelivery {
                completion_id: "task-command-1".to_owned(),
                delivered: true,
                sent_message_id: Some(MessageId(700)),
            }]
        );
        assert!(matches!(
            dispatcher.actions.0.last(),
            Some(TelegramAction::EditMessage { text, .. }) if text == "task scheduled"
        ));
        Ok(())
    }

    #[test]
    fn token_address_sends_native_photo_card_and_persists_requester_bound_state() {
        let signal = token_signal();
        let queries = Rc::new(RefCell::new(Vec::new()));
        let saved = Rc::new(RefCell::new(Vec::new()));
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
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: Some(signal.clone()),
                diagnostics: vec!["synthetic token diagnostic".to_owned()],
            },
            token_load: TokenSignalLoad {
                signal: Some(signal),
                diagnostics: Vec::new(),
            },
            photo: Ok(b"synthetic-png".to_vec()),
            state: None,
            queries: Rc::clone(&queries),
            saved: Rc::clone(&saved),
        }));

        assert_eq!(
            dispatcher.dispatch(update(
                "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump",
                Some("es")
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            queries.borrow().as_slice(),
            [SignalQuery::Address(TokenAddress { chain_id, .. })] if chain_id == "solana"
        ));
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendPhoto {
                photo,
                reply_to_message_id: Some(MessageId(7)),
                caption,
                parse_mode: Some(bot_core::telegram_actions::ParseMode::Html),
                reply_markup: Some(keyboard),
                ..
            }] if photo.as_ref() == b"synthetic-png"
                && caption.contains("Synthetic Token")
                && keyboard.inline_keyboard[0][1]
                    .copy_text
                    .as_ref()
                    .map(|copy| copy.text.as_str())
                    == Some("J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump")
        ));
        let saved = saved.borrow();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].0.len(), 12);
        assert_eq!(saved[0].1.message_id, 700);
        assert_eq!(saved[0].1.source_message_id, 7);
        assert_eq!(saved[0].1.requester_id, "88");
        assert_eq!(
            dispatcher.state_diagnostics(),
            ["synthetic token diagnostic"]
        );
    }

    struct ChartPrices {
        charts: Rc<RefCell<Vec<String>>>,
        fail_chart: bool,
    }
    impl super::MarketPriceSource for ChartPrices {
        fn load(
            &mut self,
            query: &str,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            let key = query
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_start_matches("stock:")
                .trim_start_matches('$')
                .to_ascii_lowercase();
            let symbol = match key.as_str() {
                "btc" | "bitcoin" => Some("BTC"),
                "apple" | "aapl" => Some("AAPL"),
                "f" => Some("F"),
                _ => None,
            };
            let stable_list = matches!(key.as_str(), "stables" | "stablecoins");
            MarketPriceLoad {
                chart: symbol.map(|symbol| bot_core::market_prices::MarketChart {
                    timeframe: None,
                    symbol: symbol.to_owned(),
                    name: symbol.to_owned(),
                    yahoo_symbol: if symbol == "BTC" {
                        "BTC-USD".to_owned()
                    } else {
                        symbol.to_owned()
                    },
                    token: None,
                    candidate: None,
                }),
                selection: None,
                no_assets_found: symbol.is_none() && !stable_list,
                text: symbol.map_or_else(
                    || "missing".to_owned(),
                    |symbol| format!("{symbol}: 123 USD"),
                ),
                diagnostics: Vec::new(),
            }
        }
        fn render_chart(
            &mut self,
            chart: &bot_core::market_prices::MarketChart,
            _: i64,
        ) -> Result<super::MarketChartRender, String> {
            self.charts.borrow_mut().push(format!(
                "{}:{}",
                chart.yahoo_symbol,
                chart.timeframe.as_deref().unwrap_or("default")
            ));
            if self.fail_chart {
                Err("history unavailable".to_owned())
            } else {
                Ok(super::MarketChartRender {
                    photo: b"market-chart".to_vec(),
                    caption: chart
                        .timeframe
                        .as_ref()
                        .map(|period| format!("{}: 123 USD (+5.00% {period})", chart.symbol)),
                })
            }
        }
    }

    struct CandidateChartPrices {
        initial: MarketPriceLoad,
        rendered: Rc<RefCell<Vec<Option<String>>>>,
    }

    impl super::MarketPriceSource for CandidateChartPrices {
        fn load(
            &mut self,
            _: &str,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            self.initial.clone()
        }

        fn render_chart(
            &mut self,
            chart: &bot_core::market_prices::MarketChart,
            _: i64,
        ) -> Result<super::MarketChartRender, String> {
            self.rendered.borrow_mut().push(chart.timeframe.clone());
            Ok(super::MarketChartRender {
                photo: b"provider-chart".to_vec(),
                caption: Some(format!(
                    "provider chart {}",
                    chart.timeframe.as_deref().unwrap_or("default")
                )),
            })
        }
    }

    struct SelectableMarketPrices {
        initial: MarketPriceLoad,
        candidate: MarketPriceLoad,
        stored: Rc<RefCell<HashMap<String, String>>>,
        selected: Rc<RefCell<Vec<String>>>,
    }

    impl super::MarketPriceSource for SelectableMarketPrices {
        fn load(
            &mut self,
            _: &str,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            self.initial.clone()
        }

        fn load_candidate(
            &mut self,
            candidate: &bot_core::market_prices::MarketCandidate,
            timeframe: Option<&str>,
            _: &str,
            _: &str,
            _: Option<&bot_core::market_prices::MarketConversion>,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            self.selected.borrow_mut().push(format!(
                "{}:{}",
                candidate.id,
                timeframe.unwrap_or_default()
            ));
            self.candidate.clone()
        }

        fn render_chart(
            &mut self,
            _: &bot_core::market_prices::MarketChart,
            _: i64,
        ) -> Result<super::MarketChartRender, String> {
            Err("synthetic chart unavailable".to_owned())
        }

        fn save_selection(&mut self, key: &str, value: &str, _: i64) -> Result<(), String> {
            self.stored
                .borrow_mut()
                .insert(key.to_owned(), value.to_owned());
            Ok(())
        }

        fn load_selection(&mut self, key: &str) -> Result<Option<String>, String> {
            Ok(self.stored.borrow().get(key).cloned())
        }

        fn take_selection(&mut self, key: &str) -> Result<Option<String>, String> {
            Ok(self.stored.borrow_mut().remove(key))
        }

        fn claim(&mut self, key: &str, value: &str, _: i64) -> Result<bool, String> {
            if self.stored.borrow().contains_key(key) {
                Ok(false)
            } else {
                self.stored
                    .borrow_mut()
                    .insert(key.to_owned(), value.to_owned());
                Ok(true)
            }
        }

        fn clear_selection(&mut self, key: &str) -> Result<(), String> {
            self.stored.borrow_mut().remove(key);
            Ok(())
        }
    }

    struct MultiSelectableMarketPrices {
        loads: HashMap<String, MarketPriceLoad>,
        candidate: MarketPriceLoad,
        candidate_results: VecDeque<MarketPriceLoad>,
        stored: Rc<RefCell<HashMap<String, String>>>,
        selected: Rc<RefCell<Vec<String>>>,
    }

    impl super::MarketPriceSource for MultiSelectableMarketPrices {
        fn load(
            &mut self,
            query: &str,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            let key = query
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_start_matches("crypto:")
                .trim_start_matches("stock:")
                .to_ascii_lowercase();
            self.loads.get(&key).cloned().unwrap_or(MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: true,
                text: String::new(),
                diagnostics: vec![format!("unexpected market request: {key}")],
            })
        }

        fn load_candidate(
            &mut self,
            candidate: &bot_core::market_prices::MarketCandidate,
            timeframe: Option<&str>,
            _: &str,
            _: &str,
            _: Option<&bot_core::market_prices::MarketConversion>,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            self.selected.borrow_mut().push(format!(
                "{}:{}",
                candidate.id,
                timeframe.unwrap_or_default()
            ));
            self.candidate_results
                .pop_front()
                .unwrap_or_else(|| self.candidate.clone())
        }

        fn save_selection(&mut self, key: &str, value: &str, _: i64) -> Result<(), String> {
            self.stored
                .borrow_mut()
                .insert(key.to_owned(), value.to_owned());
            Ok(())
        }

        fn load_selection(&mut self, key: &str) -> Result<Option<String>, String> {
            Ok(self.stored.borrow().get(key).cloned())
        }

        fn take_selection(&mut self, key: &str) -> Result<Option<String>, String> {
            Ok(self.stored.borrow_mut().remove(key))
        }

        fn clear_selection(&mut self, key: &str) -> Result<(), String> {
            self.stored.borrow_mut().remove(key);
            Ok(())
        }
    }

    struct SelectionStorageMarketPrices {
        initial: MarketPriceLoad,
        candidate: MarketPriceLoad,
        stored: Rc<RefCell<HashMap<String, String>>>,
        save_calls: Rc<RefCell<usize>>,
        fail_save_at: Option<usize>,
        fail_load: Option<String>,
        fail_clear: Option<String>,
        render_success: bool,
        render_caption: Option<String>,
    }

    impl MarketPriceSource for SelectionStorageMarketPrices {
        fn load(
            &mut self,
            _: &str,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            self.initial.clone()
        }

        fn load_candidate(
            &mut self,
            _: &bot_core::market_prices::MarketCandidate,
            _: Option<&str>,
            _: &str,
            _: &str,
            _: Option<&bot_core::market_prices::MarketConversion>,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            self.candidate.clone()
        }

        fn save_selection(
            &mut self,
            key: &str,
            value: &str,
            ttl_seconds: i64,
        ) -> Result<(), String> {
            assert_eq!(ttl_seconds, 0, "selection menus must not expire");
            let mut calls = self.save_calls.borrow_mut();
            *calls += 1;
            if self.fail_save_at == Some(*calls) {
                return Err(format!("synthetic save failure {}", *calls));
            }
            self.stored
                .borrow_mut()
                .insert(key.to_owned(), value.to_owned());
            Ok(())
        }

        fn load_selection(&mut self, key: &str) -> Result<Option<String>, String> {
            if let Some(error) = &self.fail_load {
                return Err(error.clone());
            }
            Ok(self.stored.borrow().get(key).cloned())
        }

        fn take_selection(&mut self, key: &str) -> Result<Option<String>, String> {
            // A failing `fail_load` read already stops the callback before
            // the take, so taking never fails here.
            Ok(self.stored.borrow_mut().remove(key))
        }

        fn clear_selection(&mut self, key: &str) -> Result<(), String> {
            if let Some(error) = &self.fail_clear {
                return Err(error.clone());
            }
            self.stored.borrow_mut().remove(key);
            Ok(())
        }

        fn render_chart(
            &mut self,
            _: &bot_core::market_prices::MarketChart,
            _: i64,
        ) -> Result<super::MarketChartRender, String> {
            if self.render_success {
                Ok(super::MarketChartRender {
                    photo: b"callback-chart".to_vec(),
                    caption: self.render_caption.clone(),
                })
            } else {
                Err("synthetic callback chart unavailable".to_owned())
            }
        }
    }

    fn market_selection_fixture(
        timeframe: Option<&str>,
    ) -> bot_core::market_prices::MarketSelection {
        bot_core::market_prices::MarketSelection {
            query: "libra".to_owned(),
            timeframe: timeframe.map(ToOwned::to_owned),
            target_symbol: "USD".to_owned(),
            target_parameter: "USD".to_owned(),
            conversion: None,
            candidates: vec![bot_core::market_prices::MarketCandidate {
                id: "1001".to_owned(),
                symbol: "LIBRA".to_owned(),
                name: "Libra Finance".to_owned(),
                slug: "libra-finance".to_owned(),
                price: "0.007".to_owned(),
                change: "N/A".to_owned(),
                currency: String::new(),
                exchange: String::new(),
                asset_type: String::new(),
                contracts: Vec::new(),
            }],
        }
    }

    fn market_selection_load(timeframe: Option<&str>) -> MarketPriceLoad {
        MarketPriceLoad {
            chart: None,
            selection: Some(market_selection_fixture(timeframe)),
            no_assets_found: false,
            text: String::new(),
            diagnostics: Vec::new(),
        }
    }

    fn market_candidate_quote() -> MarketPriceLoad {
        MarketPriceLoad {
            chart: None,
            selection: None,
            no_assets_found: false,
            text: "LIBRA: 0.007 USD (N/A 24h)".to_owned(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn market_selection_persistence_handles_missing_context_and_delivery_failures() {
        let mut no_chat = incoming_message("/p libra", Some("en"));
        no_chat.chat_id = None;
        let message = &no_chat;
        let selection = market_selection_fixture(None);
        let mut missing_context = dispatcher();
        assert_eq!(
            missing_context.persist_market_selection(
                message,
                &selection,
                bot_core::market_prices::MarketPriceCommand::Unified,
                1_672_531_200,
                bot_core::locale::Locale::En,
                0,
            ),
            Ok(None)
        );

        let mut no_source = dispatcher();
        let message = incoming_message("/p libra", Some("en"));
        assert_eq!(
            no_source.persist_market_selection(
                &message,
                &selection,
                bot_core::market_prices::MarketPriceCommand::Unified,
                1_672_531_200,
                bot_core::locale::Locale::En,
                0,
            ),
            Ok(None)
        );
        assert!(
            no_source
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic == "market selection storage unavailable")
        );

        let failing_save = Rc::new(RefCell::new(0));
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut storage_failure =
            dispatcher().with_market_price_source(Box::new(SelectionStorageMarketPrices {
                initial: market_selection_load(None),
                candidate: market_candidate_quote(),
                stored: Rc::clone(&stored),
                save_calls: Rc::clone(&failing_save),
                fail_save_at: Some(1),
                fail_load: None,
                fail_clear: None,
                render_success: false,
                render_caption: None,
            }));
        assert_eq!(
            storage_failure.dispatch(update("/p libra", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            storage_failure.actions.0.as_slice(),
            [TelegramAction::SendMessage(message)] if message.reply_markup.is_none()
        ));
        assert!(
            storage_failure
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection storage unavailable"))
        );

        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut rejected = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            delivery_actions(DeliveryOutcome::Rejected),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectionStorageMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            save_calls: Rc::new(RefCell::new(0)),
            fail_save_at: None,
            fail_load: None,
            fail_clear: Some("synthetic clear after rejection".to_owned()),
            render_success: false,
            render_caption: None,
        }));
        assert_eq!(
            rejected.dispatch(update("/p libra", Some("en"))),
            Err(DispatchError::Action("synthetic delivery rejection"))
        );
        assert_eq!(stored.borrow().len(), 1);
        assert!(
            rejected
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("clear after send failure"))
        );

        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut unconfirmed = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            delivery_actions(DeliveryOutcome::Unconfirmed),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectionStorageMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            save_calls: Rc::new(RefCell::new(0)),
            fail_save_at: None,
            fail_load: None,
            fail_clear: Some("synthetic clear after unconfirmed".to_owned()),
            render_success: false,
            render_caption: None,
        }));
        assert_eq!(
            unconfirmed.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(stored.borrow().len(), 1);
        assert!(
            unconfirmed
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("delivery was unconfirmed"))
        );
        assert!(
            unconfirmed
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("clear after unconfirmed delivery"))
        );

        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut update_failure =
            dispatcher().with_market_price_source(Box::new(SelectionStorageMarketPrices {
                initial: market_selection_load(None),
                candidate: market_candidate_quote(),
                stored: Rc::clone(&stored),
                save_calls: Rc::new(RefCell::new(0)),
                fail_save_at: Some(2),
                fail_load: None,
                fail_clear: Some("synthetic clear after update".to_owned()),
                render_success: false,
                render_caption: None,
            }));
        assert_eq!(
            update_failure.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(stored.borrow().len(), 1);
        assert!(
            update_failure
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection update unavailable"))
        );
        assert!(update_failure
            .actions
            .0
            .iter()
            .any(|action| matches!(action, TelegramAction::EditMessage { reply_markup: Some(markup), .. } if markup.inline_keyboard.is_empty())));
    }

    #[test]
    fn market_pages_preserve_identity_range_and_original_reply() -> Result<(), String> {
        let mut selection = market_selection_fixture(Some("1m"));
        let prototype = selection.candidates[0].clone();
        selection.candidates = (0..7)
            .map(|i| bot_core::market_prices::MarketCandidate {
                id: (1000 + i).to_string(),
                name: format!("Libra {i}"),
                ..prototype.clone()
            })
            .collect();
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    selection: Some(selection),
                    ..market_selection_load(Some("1m"))
                },
                candidate: market_candidate_quote(),
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra 1m", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let key = stored
            .borrow()
            .keys()
            .next()
            .cloned()
            .ok_or("missing storage")?;
        let id = key
            .strip_prefix("market_selection:")
            .ok_or("selection key")?;
        let before = stored.borrow().clone();
        for (page, message_id) in [(1, 701), (99, 700)] {
            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &format!("mkt:page:{id}:{page}"),
                    "private",
                    Some("en"),
                    message_id
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(*stored.borrow(), before);
        }
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &format!("mkt:page:{id}:1"),
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        let page_edits = dispatcher
            .actions
            .0
            .iter()
            .filter_map(|action| match action {
                TelegramAction::EditMessage {
                    message_id: MessageId(700),
                    reply_markup: Some(markup),
                    ..
                } => Some(markup),
                _ => None,
            })
            .collect::<Vec<_>>();
        let callback = page_edits
            .last()
            .and_then(|markup| {
                markup
                    .inline_keyboard
                    .first()?
                    .first()?
                    .callback_data
                    .clone()
            })
            .ok_or("missing page edit")?;
        assert_eq!(callback, format!("mkt:select:{id}:5"));
        assert_eq!(*stored.borrow(), before);
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(*selected.borrow(), vec!["1005:1m"]);
        assert!(stored.borrow().is_empty());
        assert!(dispatcher.actions.0.iter().any(|action| matches!(action, TelegramAction::SendMessage(message) if message.reply_to_message_id == Some(MessageId(7)) && message.reply_markup.is_none())));
        Ok(())
    }

    #[test]
    fn closing_market_menu_clears_state_without_fetching_a_quote() -> Result<(), String> {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: market_selection_load(None),
                candidate: market_candidate_quote(),
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let key = stored
            .borrow()
            .keys()
            .next()
            .cloned()
            .ok_or("missing storage")?;
        let id = key
            .strip_prefix("market_selection:")
            .ok_or("selection key")?;
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &format!("mkt:close:{id}:0"),
                "private",
                Some("en"),
                701
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(!stored.borrow().is_empty());
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &format!("mkt:close:{id}:0"),
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(stored.borrow().is_empty());
        assert!(selected.borrow().is_empty());
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::DeleteMessage {
                message_id: MessageId(700),
                ..
            }
        )));
        Ok(())
    }

    #[test]
    fn help_and_settings_navigation_edit_in_place_without_persistence() {
        for data in [
            "help:home",
            "help:markets",
            "help:ai",
            "help:tasks",
            "help:credits",
            "help:tools",
            "help:settings",
            "help:close",
            "cfg:page:home",
            "cfg:page:link",
            "cfg:page:help",
            "cfg:page:close",
        ] {
            let mut dispatcher = dispatcher();
            assert_eq!(
                dispatcher.dispatch(callback_update(data, "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert!(
                !dispatcher
                    .config
                    .chat_ids
                    .iter()
                    .any(|id| id.starts_with("set:"))
            );
            assert!(
                !dispatcher
                    .actions
                    .0
                    .iter()
                    .any(|a| matches!(a, TelegramAction::SendMessage(_)))
            );
            assert!(
                dispatcher
                    .actions
                    .0
                    .iter()
                    .any(|a| matches!(a, TelegramAction::AnswerCallback { text: None, .. }))
            );
            assert!(dispatcher.actions.0.iter().any(|a| matches!(
                a,
                TelegramAction::EditMessage { .. } | TelegramAction::DeleteMessage { .. }
            )));
        }
        let mut denied = dispatcher();
        denied.authorization.is_admin = false;
        assert_eq!(
            denied.dispatch(callback_update("cfg:page:close", "group", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(!denied.actions.0.iter().any(|a| matches!(
            a,
            TelegramAction::EditMessage { .. } | TelegramAction::DeleteMessage { .. }
        )));
    }

    #[test]
    fn market_callbacks_cover_expiry_ownership_authorization_and_invalid_state() {
        let selection = market_selection_fixture(Some("7d"));
        let candidate_quote = market_candidate_quote();
        let source =
            |stored: Rc<RefCell<HashMap<String, String>>>,
             fail_load: Option<String>,
             fail_clear: Option<String>| SelectionStorageMarketPrices {
                initial: market_selection_load(Some("7d")),
                candidate: candidate_quote.clone(),
                stored,
                save_calls: Rc::new(RefCell::new(0)),
                fail_save_at: None,
                fail_load,
                fail_clear,
                render_success: false,
                render_caption: None,
            };
        let stored_value = |chat_id: &str, message_id: i64, requester_id: i64| {
            StoredMarketSelection {
                selection: selection.clone(),
                chat_id: chat_id.to_owned(),
                message_id,
                source_message_id: Some(message_id),
                requester_id,
                command: "unified".to_owned(),
            }
            .encode()
        };

        let mut malformed = dispatcher().with_market_price_source(Box::new(source(
            Rc::new(RefCell::new(HashMap::new())),
            None,
            None,
        )));
        assert_eq!(
            malformed.dispatch(callback_update("mkt:bad:id:0", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            malformed.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: None,
                show_alert: false,
                ..
            }]
        ));

        let mut expired = dispatcher().with_market_price_source(Box::new(source(
            Rc::new(RefCell::new(HashMap::new())),
            None,
            None,
        )));
        assert_eq!(
            expired.dispatch(callback_update(
                "mkt:select:missing:0",
                "private",
                Some("en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            expired.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }] if text == "This selection expired"
        ));

        let mut read_failed = dispatcher().with_market_price_source(Box::new(source(
            Rc::new(RefCell::new(HashMap::new())),
            Some("synthetic selection read failure".to_owned()),
            None,
        )));
        assert_eq!(
            read_failed.dispatch(callback_update(
                "mkt:select:read-failed:0",
                "private",
                Some("en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            read_failed
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("selection read failed"))
        );

        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:decode-failed".to_owned(),
            "not-json".to_owned(),
        )])));
        let mut decode_failed =
            dispatcher().with_market_price_source(Box::new(source(Rc::clone(&stored), None, None)));
        assert_eq!(
            decode_failed.dispatch(callback_update(
                "mkt:select:decode-failed:0",
                "private",
                Some("en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            decode_failed
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("selection decode failed"))
        );

        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:wrong-chat".to_owned(),
            stored_value("-99", 7, 88),
        )])));
        let mut wrong_chat =
            dispatcher().with_market_price_source(Box::new(source(stored, None, None)));
        assert_eq!(
            wrong_chat.dispatch(callback_update(
                "mkt:select:wrong-chat:0",
                "private",
                Some("en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            wrong_chat.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }] if text == "That selection belongs to someone else"
        ));

        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:private-owner".to_owned(),
            stored_value("-42", 7, 999),
        )])));
        let mut private_owner =
            dispatcher().with_market_price_source(Box::new(source(stored, None, None)));
        assert_eq!(
            private_owner.dispatch(callback_update(
                "mkt:select:private-owner:0",
                "private",
                Some("en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(private_owner.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            } if text == "That selection belongs to someone else"
        )));

        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:group-admin".to_owned(),
            stored_value("-42", 7, 999),
        )])));
        let mut group_admin =
            dispatcher().with_market_price_source(Box::new(source(Rc::clone(&stored), None, None)));
        assert_eq!(
            group_admin.dispatch(callback_update_with_context(
                "mkt:select:group-admin:0",
                json!(-42),
                "group",
                7,
                Some(88),
                Some("en"),
                Some("callback-admin"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            group_admin
                .authorization
                .checks
                .iter()
                .any(|check| check == &("-42".to_owned(), "88".to_owned()))
        );
        assert!(stored.borrow().is_empty());
        assert!(group_admin.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message) if message.text.contains("LIBRA: 0.007")
        )));

        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:no-user".to_owned(),
            stored_value("-42", 7, 999),
        )])));
        let mut no_user =
            dispatcher().with_market_price_source(Box::new(source(stored, None, None)));
        assert_eq!(
            no_user.dispatch(callback_update_with_context(
                "mkt:select:no-user:0",
                json!(-42),
                "supergroup",
                7,
                None,
                Some("en"),
                Some("callback-no-user"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            no_user.actions.0.last(),
            Some(TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }) if text == "Esa selección es de otra persona"
        ));

        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:bad-chat-id".to_owned(),
            stored_value("not-number", 7, 88),
        )])));
        let mut bad_chat_id =
            dispatcher().with_market_price_source(Box::new(source(stored, None, None)));
        assert_eq!(
            bad_chat_id.dispatch(callback_update_with_context(
                "mkt:select:bad-chat-id:0",
                json!("not-number"),
                "private",
                7,
                Some(88),
                Some("en"),
                Some("callback-bad-chat"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            bad_chat_id
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic == "invalid market callback chat id")
        );

        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:bad-index".to_owned(),
            stored_value("-42", 7, 88),
        )])));
        let mut bad_index =
            dispatcher().with_market_price_source(Box::new(source(stored, None, None)));
        assert_eq!(
            bad_index.dispatch(callback_update(
                "mkt:select:bad-index:9",
                "private",
                Some("es"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            bad_index.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }] if text == "Esa opción no es válida"
        ));

        let mut no_callback_id = dispatcher().with_market_price_source(Box::new(source(
            Rc::new(RefCell::new(HashMap::new())),
            None,
            None,
        )));
        assert_eq!(
            no_callback_id.dispatch(callback_update_with_context(
                "mkt:select:missing:0",
                json!(-42),
                "private",
                7,
                Some(88),
                Some("en"),
                None,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(no_callback_id.actions.0.is_empty());
    }

    #[test]
    fn market_callbacks_deliver_chart_photos_and_fallbacks() {
        let chart = |timeframe: Option<&str>, token: Option<TokenAddress>| {
            bot_core::market_prices::MarketChart {
                timeframe: timeframe.map(ToOwned::to_owned),
                symbol: "LIBRA".to_owned(),
                name: "Libra Finance".to_owned(),
                yahoo_symbol: "LIBRA-USD".to_owned(),
                token,
                candidate: None,
            }
        };
        let candidate_load = |chart: bot_core::market_prices::MarketChart| MarketPriceLoad {
            chart: Some(chart),
            selection: None,
            no_assets_found: false,
            text: "LIBRA: 0.007 USD (+2.00% 24h)".to_owned(),
            diagnostics: Vec::new(),
        };
        let callback = |timeframe: Option<&str>| {
            let selection_id = market_selection_id(-42, 7, 88, 1_672_531_200, 0);
            let data = format!("mkt:select:{selection_id}:0");
            let _ = timeframe;
            data
        };

        for render_caption in [Some("provider caption".to_owned()), None] {
            let stored = Rc::new(RefCell::new(HashMap::new()));
            let mut dispatcher =
                dispatcher().with_market_price_source(Box::new(SelectionStorageMarketPrices {
                    initial: market_selection_load(None),
                    candidate: candidate_load(chart(None, None)),
                    stored: Rc::clone(&stored),
                    save_calls: Rc::new(RefCell::new(0)),
                    fail_save_at: None,
                    fail_load: None,
                    fail_clear: None,
                    render_success: true,
                    render_caption,
                }));
            assert_eq!(
                dispatcher.dispatch(update("/p libra", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &callback(None),
                    "private",
                    Some("en"),
                    700,
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert!(stored.borrow().is_empty());
            assert!(dispatcher.actions.0.iter().any(|action| matches!(
                action,
                TelegramAction::SendPhoto {
                    photo,
                    caption,
                    parse_mode: None,
                    reply_to_message_id: Some(MessageId(7)),
                    ..
                }
                    if photo.as_ref() == b"callback-chart"
                        && (caption == "provider caption" || caption.contains("LIBRA: 0.007"))
            )));
            assert!(dispatcher.actions.0.iter().any(|action| matches!(
                action,
                TelegramAction::DeleteMessage {
                    chat_id: ChatId(-42),
                    message_id: MessageId(700),
                }
            )));
            assert!(dispatcher.actions.0.iter().any(|action| matches!(
                action,
                TelegramAction::AnswerCallback {
                    show_alert: false,
                    ..
                }
            )));
        }

        for outcome in [Attempt::Skip, Attempt::Unconfirmed] {
            let stored = Rc::new(RefCell::new(HashMap::new()));
            let mut dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig::default()),
                    chat_ids: Vec::new(),
                },
                photo_actions(outcome),
                State::default(),
                values(),
                random(),
                authorization(),
                "@mybot",
            )
            .with_market_price_source(Box::new(SelectionStorageMarketPrices {
                initial: market_selection_load(None),
                candidate: candidate_load(chart(None, None)),
                stored: Rc::clone(&stored),
                save_calls: Rc::new(RefCell::new(0)),
                fail_save_at: None,
                fail_load: None,
                fail_clear: None,
                render_success: true,
                render_caption: None,
            }));
            assert_eq!(
                dispatcher.dispatch(update("/p libra", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &callback(None),
                    "private",
                    Some("en"),
                    700,
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert!(
                dispatcher
                    .state_diagnostics()
                    .iter()
                    .any(|diagnostic| diagnostic.contains("market chart photo delivery"))
            );
            assert!(stored.borrow().is_empty());
        }

        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut photo_failed = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            attempt_actions(Attempt::Fail, ActionScript::photo),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectionStorageMarketPrices {
            initial: market_selection_load(None),
            candidate: candidate_load(chart(None, None)),
            stored: Rc::clone(&stored),
            save_calls: Rc::new(RefCell::new(0)),
            fail_save_at: None,
            fail_load: None,
            fail_clear: None,
            render_success: true,
            render_caption: None,
        }));
        assert_eq!(
            photo_failed.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            photo_failed.dispatch(callback_update_for_message(
                &callback(None),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            photo_failed
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market chart photo delivery failed"))
        );
        assert!(stored.borrow().is_empty());

        for timeframe in [None, Some("7d")] {
            let token = TokenAddress {
                chain_id: "solana".to_owned(),
                network: "solana".to_owned(),
                tag: "SOL".to_owned(),
                address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            };
            let stored = Rc::new(RefCell::new(HashMap::new()));
            let saved = Rc::new(RefCell::new(Vec::new()));
            let mut dispatcher = dispatcher()
                .with_market_price_source(Box::new(SelectionStorageMarketPrices {
                    initial: market_selection_load(timeframe),
                    candidate: candidate_load(chart(timeframe, Some(token))),
                    stored: Rc::clone(&stored),
                    save_calls: Rc::new(RefCell::new(0)),
                    fail_save_at: None,
                    fail_load: None,
                    fail_clear: None,
                    render_success: false,
                    render_caption: None,
                }))
                .with_token_signal_source(Box::new(Signals {
                    query_load: TokenSignalLoad {
                        signal: None,
                        diagnostics: Vec::new(),
                    },
                    token_load: TokenSignalLoad {
                        signal: Some(token_signal()),
                        diagnostics: vec!["callback token fallback".to_owned()],
                    },
                    photo: Ok(b"token-callback".to_vec()),
                    state: None,
                    queries: Rc::new(RefCell::new(Vec::new())),
                    saved: Rc::clone(&saved),
                }));
            let input = if timeframe == Some("7d") {
                "/p libra 7d"
            } else {
                "/p libra"
            };
            assert_eq!(
                dispatcher.dispatch(update(input, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &callback(timeframe),
                    "private",
                    Some("en"),
                    700,
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert!(dispatcher.actions.0.iter().any(|action| matches!(
                action,
                TelegramAction::SendPhoto {
                    photo,
                    parse_mode: Some(bot_core::telegram_actions::ParseMode::Html),
                    reply_to_message_id: Some(MessageId(7)),
                    reply_markup: Some(markup),
                    ..
                } if photo.as_ref() == b"token-callback" && !markup.inline_keyboard.is_empty()
            )));
            assert_eq!(saved.borrow().len(), 1);
            assert!(stored.borrow().is_empty());
        }
    }

    #[test]
    fn market_callbacks_consume_selection_despite_clear_failure_is_diagnostic() {
        let selection = market_selection_fixture(Some("7d"));
        let token = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
        };
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            photo_actions(Attempt::Skip),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectionStorageMarketPrices {
            initial: MarketPriceLoad {
                chart: None,
                selection: Some(selection),
                no_assets_found: false,
                text: String::new(),
                diagnostics: Vec::new(),
            },
            candidate: MarketPriceLoad {
                chart: Some(bot_core::market_prices::MarketChart {
                    timeframe: Some("7d".to_owned()),
                    symbol: "LIBRA".to_owned(),
                    name: "Libra Finance".to_owned(),
                    yahoo_symbol: "LIBRA-USD".to_owned(),
                    token: Some(token),
                    candidate: None,
                }),
                selection: None,
                no_assets_found: false,
                text: "LIBRA: 0.007 USD".to_owned(),
                diagnostics: Vec::new(),
            },
            stored: Rc::clone(&stored),
            save_calls: Rc::new(RefCell::new(0)),
            fail_save_at: None,
            fail_load: None,
            fail_clear: Some("synthetic callback clear failure".to_owned()),
            render_success: false,
            render_caption: None,
        }))
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: Some(token_signal()),
                diagnostics: Vec::new(),
            },
            photo: Ok(vec![1]),
            state: None,
            queries: Rc::new(RefCell::new(Vec::new())),
            saved: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra 7d", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let selection_id = market_selection_id(-42, 7, 88, 1_672_531_200, 0);
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &format!("mkt:select:{selection_id}:0"),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher
            .state_diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.contains("token chart photo delivery was unconfirmed")));
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection clear failed"))
        );
        // The quote was delivered, so the atomic take consumed the menu even
        // though the best-effort clear failed: a second tap expires instead of
        // resolving into a duplicate quote.
        assert!(stored.borrow().is_empty());
    }

    #[test]
    fn market_callbacks_localize_empty_candidate_quotes_and_history_failures() {
        let selection = market_selection_fixture(None);
        let stored_value = StoredMarketSelection {
            selection,
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: Some(6),
            requester_id: 88,
            command: "crypto".to_owned(),
        }
        .encode();
        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:empty-candidate".to_owned(),
            stored_value,
        )])));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(SelectionStorageMarketPrices {
                initial: market_selection_load(None),
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: vec!["candidate quote unavailable".to_owned()],
                },
                stored,
                save_calls: Rc::new(RefCell::new(0)),
                fail_save_at: None,
                fail_load: None,
                fail_clear: None,
                render_success: false,
                render_caption: None,
            }));
        assert_eq!(
            dispatcher.dispatch(callback_update_with_context(
                "mkt:select:empty-candidate:0",
                json!(-42),
                "private",
                7,
                Some(88),
                Some("es"),
                Some("callback-empty"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(message),
                TelegramAction::AnswerCallback { show_alert: true, .. }
            ] if message.reply_to_message_id == Some(MessageId(6))
                && message.text.contains("No pude conseguir una cotización")
        ));
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic == "candidate quote unavailable")
        );
    }

    #[test]
    fn market_selection_helpers_and_default_source_methods_are_covered() {
        struct BareStock;
        impl StockPriceSource for BareStock {
            fn load(&mut self, _: &str, _: i64) -> StockQuotesLoad {
                StockQuotesLoad {
                    quotes: None,
                    diagnostics: Vec::new(),
                }
            }
        }

        let quote = StockQuote {
            symbol: "SYN".to_owned(),
            name: "Synthetic".to_owned(),
            price: 1.0,
            currency: "USD".to_owned(),
            exchange: "TEST".to_owned(),
            asset_type: String::new(),
            variation: 0.0,
        };
        assert_eq!(
            BareStock.render_chart(&quote, 1_700_000_000),
            Err("stock chart unavailable".to_owned())
        );
        assert_eq!(
            BareStock
                .load_with_timeframe("SYN", Some("1m"), 1_700_000_000)
                .quotes,
            None
        );

        struct BareMarket;
        impl MarketPriceSource for BareMarket {
            fn load(
                &mut self,
                _: &str,
                _: bot_core::market_prices::MarketPriceCommand,
                _: bot_core::locale::Locale,
                _: i64,
            ) -> MarketPriceLoad {
                MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                }
            }
        }

        let candidate = bot_core::market_prices::MarketCandidate {
            id: "42".to_owned(),
            symbol: "SYN".to_owned(),
            name: String::new(),
            slug: "synthetic".to_owned(),
            price: "1".to_owned(),
            change: "0".to_owned(),
            currency: String::new(),
            exchange: String::new(),
            asset_type: String::new(),
            contracts: vec![TokenAddress {
                chain_id: "solana".to_owned(),
                network: "mainnet".to_owned(),
                tag: "SOL".to_owned(),
                address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            }],
        };
        let mut source = BareMarket;
        let default = source.load_candidate(
            &candidate,
            Some("7d"),
            "USD",
            "USD",
            None,
            bot_core::market_prices::MarketPriceCommand::CryptoOnly,
            bot_core::locale::Locale::En,
            1_700_000_000,
        );
        assert!(default.no_assets_found);
        assert_eq!(
            default.diagnostics,
            ["market candidate resolution unavailable"]
        );
        assert!(
            source
                .render_chart(
                    &bot_core::market_prices::MarketChart {
                        timeframe: None,
                        symbol: "SYN".to_owned(),
                        name: "Synthetic".to_owned(),
                        yahoo_symbol: "SYN-USD".to_owned(),
                        token: None,
                        candidate: None,
                    },
                    1_700_000_000,
                )
                .is_err_and(|error| error == "market chart unavailable")
        );
        assert_eq!(
            source.save_selection("key", "value", 60),
            Err("market selection storage unavailable".to_owned())
        );
        assert_eq!(
            source.load_selection("key"),
            Err("market selection storage unavailable".to_owned())
        );
        assert_eq!(
            source.take_selection("key"),
            Err("market selection storage unavailable".to_owned())
        );
        assert_eq!(
            source.claim("key", "1", 60),
            Err("market selection storage unavailable".to_owned())
        );
        assert_eq!(source.clear_selection("key"), Ok(()));
        // A dispatcher backed only by the defaults answers without a quote.
        let mut bare = dispatcher().with_market_price_source(Box::new(BareMarket));
        assert_eq!(
            bare.dispatch(update("/p btc", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&bare.actions.0),
            ["I could not get a quote for btc"]
        );

        assert_eq!(short_market_address("short"), "short");
        assert_eq!(short_market_address("123456789012345678"), "123456…5678");
        assert_eq!(
            market_selection_command("crypto"),
            bot_core::market_prices::MarketPriceCommand::CryptoOnly
        );
        assert_eq!(
            market_selection_command("unified"),
            bot_core::market_prices::MarketPriceCommand::Unified
        );

        let mut candidates = vec![candidate];
        candidates.push(bot_core::market_prices::MarketCandidate {
            id: "43".to_owned(),
            symbol: "LONG".to_owned(),
            name: "x".repeat(80),
            slug: "long".to_owned(),
            price: "2".to_owned(),
            change: "0".to_owned(),
            currency: String::new(),
            exchange: String::new(),
            asset_type: String::new(),
            contracts: Vec::new(),
        });
        candidates.push(bot_core::market_prices::MarketCandidate {
            id: "stock:EXM-USD".to_owned(),
            symbol: "EXM-USD".to_owned(),
            name: "Example stock".to_owned(),
            slug: "exm-usd".to_owned(),
            price: "42".to_owned(),
            change: "+5% 1m".to_owned(),
            currency: "USD".to_owned(),
            exchange: "Synthetic".to_owned(),
            asset_type: "Equity".to_owned(),
            contracts: Vec::new(),
        });
        for index in 2..12 {
            candidates.push(bot_core::market_prices::MarketCandidate {
                id: index.to_string(),
                symbol: format!("S{index}"),
                name: format!("Synthetic {index}"),
                slug: format!("synthetic-{index}"),
                price: "1".to_owned(),
                change: "0".to_owned(),
                currency: String::new(),
                exchange: String::new(),
                asset_type: String::new(),
                contracts: Vec::new(),
            });
        }
        let selection = bot_core::market_prices::MarketSelection {
            query: "syn".to_owned(),
            timeframe: None,
            target_symbol: "USD".to_owned(),
            target_parameter: "USD".to_owned(),
            conversion: None,
            candidates,
        };
        let keyboard =
            super::market_selection_page("selection", &selection, bot_core::locale::Locale::Es, 0);
        assert_eq!(keyboard.inline_keyboard.len(), 7);
        assert!(keyboard.inline_keyboard[..5].iter().all(|row| {
            row[0]
                .callback_data
                .as_deref()
                .unwrap_or_default()
                .starts_with("mkt:select:selection:")
        }));
        assert!(
            keyboard
                .inline_keyboard
                .iter()
                .any(|row| row[0].text.contains("EXM-USD (Synthetic)"))
        );
        let second =
            super::market_selection_page("selection", &selection, bot_core::locale::Locale::En, 1);
        assert_eq!(
            second.inline_keyboard[0][0].callback_data.as_deref(),
            Some("mkt:select:selection:5")
        );
        assert_eq!(
            second.inline_keyboard[second.inline_keyboard.len() - 1][0].text,
            "Close"
        );
        let third =
            super::market_selection_page("selection", &selection, bot_core::locale::Locale::Es, 2);
        assert_eq!(
            third.inline_keyboard[0][0].callback_data.as_deref(),
            Some("mkt:select:selection:10")
        );
        assert_eq!(
            market_selection_text(&selection, bot_core::locale::Locale::En),
            bot_core::market_prices::format_market_selection(
                &selection,
                bot_core::locale::Locale::En
            )
        );
    }

    #[test]
    fn market_candidate_deduplication_uses_all_contracts_and_keeps_distinct_assets() {
        let solana_target = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: "AbCdEf1234567890".to_owned(),
        };
        let solana_case_variant = TokenAddress {
            address: "aBcDeF1234567890".to_owned(),
            ..solana_target.clone()
        };
        let ethereum_checksum_variant = TokenAddress {
            chain_id: "ethereum".to_owned(),
            network: "eth".to_owned(),
            tag: "ETH".to_owned(),
            address: "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01".to_owned(),
        };
        let ethereum_lowercase = TokenAddress {
            address: ethereum_checksum_variant.address.to_ascii_lowercase(),
            ..ethereum_checksum_variant.clone()
        };
        let candidate =
            |id: &str, contracts: Vec<TokenAddress>| bot_core::market_prices::MarketCandidate {
                id: id.to_owned(),
                symbol: "SYN".to_owned(),
                name: id.to_owned(),
                slug: id.to_ascii_lowercase(),
                price: "1".to_owned(),
                change: "N/A".to_owned(),
                currency: String::new(),
                exchange: String::new(),
                asset_type: String::new(),
                contracts,
            };
        let dex_candidate = candidate(
            "token:solana:solana:AbCdEf1234567890",
            vec![solana_target.clone()],
        );
        let provider_candidate = candidate(
            "1001",
            vec![solana_case_variant.clone(), solana_target.clone()],
        );
        let evm_provider = candidate("1002", vec![ethereum_checksum_variant]);
        let evm_dex = candidate("token:ethereum:eth:0x...", vec![ethereum_lowercase]);
        let other_chain = candidate(
            "1003",
            vec![TokenAddress {
                chain_id: "bsc".to_owned(),
                network: "bsc".to_owned(),
                tag: "BNB".to_owned(),
                address: solana_target.address.clone(),
            }],
        );
        let stock = candidate("stock:SYN", Vec::new());

        let mut candidates = vec![
            provider_candidate,
            dex_candidate,
            evm_provider,
            evm_dex,
            other_chain,
            stock,
        ];
        deduplicate_market_candidates(&mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            ["1001", "1002", "1003", "stock:SYN"]
        );
        assert_eq!(candidates[0].contracts.len(), 2);

        let mut case_sensitive = vec![
            candidate("1004", vec![solana_target]),
            candidate(
                "token:solana:solana:aBcDeF1234567890",
                vec![solana_case_variant],
            ),
        ];
        deduplicate_market_candidates(&mut case_sensitive);
        assert_eq!(case_sensitive.len(), 2);
    }

    #[test]
    fn matching_cmc_and_dex_contracts_do_not_create_duplicate_menus() {
        let signal = token_signal();
        let token = signal.token.clone();
        let market_candidate = bot_core::market_prices::MarketCandidate {
            id: "123".to_owned(),
            symbol: "SYN".to_owned(),
            name: "Provider Synthetic".to_owned(),
            slug: "synthetic".to_owned(),
            price: "0.01".to_owned(),
            change: "N/A 7d".to_owned(),
            currency: String::new(),
            exchange: String::new(),
            asset_type: String::new(),
            contracts: vec![token.clone()],
        };
        let rendered = Rc::new(RefCell::new(Vec::new()));
        let mut single_result = dispatcher()
            .with_market_price_source(Box::new(CandidateChartPrices {
                initial: MarketPriceLoad {
                    chart: Some(bot_core::market_prices::MarketChart {
                        timeframe: None,
                        symbol: "SYN".to_owned(),
                        name: "Provider Synthetic".to_owned(),
                        yahoo_symbol: "SYN-USD".to_owned(),
                        token: Some(token),
                        candidate: Some(market_candidate.clone()),
                    }),
                    selection: None,
                    no_assets_found: false,
                    text: "SYN: 0.01 USD (N/A 7d)".to_owned(),
                    diagnostics: Vec::new(),
                },
                rendered: Rc::clone(&rendered),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: Some(signal.clone()),
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Ok(vec![1]),
                state: None,
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            single_result.dispatch(update("/p syn 7d", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(rendered.borrow().as_slice(), [Some("7d".to_owned())]);
        assert!(matches!(
            single_result.actions.0.as_slice(),
            [TelegramAction::SendPhoto {
                photo,
                caption,
                reply_to_message_id: Some(MessageId(7)),
                reply_markup: None,
                ..
            }] if photo.as_ref() == b"provider-chart" && caption == "provider chart 7d"
        ));

        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut existing_selection = dispatcher()
            .with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: Some(bot_core::market_prices::MarketSelection {
                        query: "syn".to_owned(),
                        timeframe: None,
                        target_symbol: "USD".to_owned(),
                        target_parameter: "USD".to_owned(),
                        conversion: None,
                        candidates: vec![market_candidate],
                    }),
                    no_assets_found: false,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "selected provider quote".to_owned(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: Some(signal),
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Ok(vec![1]),
                state: None,
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            existing_selection.dispatch(update("/p syn 7d", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(selected.borrow().as_slice(), ["123:7d".to_owned()]);
        assert!(stored.borrow().is_empty());
        assert!(matches!(
            existing_selection.actions.0.as_slice(),
            [TelegramAction::SendMessage(message)]
                if message.reply_markup.is_none() && message.text == "selected provider quote"
        ));
    }

    #[test]
    fn default_token_commands_fetch_24h_history_without_preloaded_candles() {
        for input in ["$syn", "/p syn", "/c syn"] {
            let mut signal = token_signal();
            signal.candles.clear();
            let periods = Rc::new(RefCell::new(Vec::new()));
            let mut dispatcher = dispatcher().with_token_signal_source(Box::new(StatefulSignals {
                signal,
                token_results: Rc::new(RefCell::new(VecDeque::new())),
                state: Rc::new(RefCell::new(None)),
                saved: Rc::new(RefCell::new(Vec::new())),
                periods: Rc::clone(&periods),
            }));
            assert_eq!(
                dispatcher.dispatch(update(input, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(periods.borrow().as_slice(), ["24h"], "{input}");
            assert!(dispatcher.actions.0.iter().any(|action| matches!(action,
                TelegramAction::SendPhoto { reply_to_message_id: Some(MessageId(7)), caption, .. }
                    if caption.contains("24h")
            )), "{input}");
        }
    }

    #[test]
    fn narrow_window_without_trades_widens_to_available_history() {
        let mut signal = token_signal();
        signal.candles.clear();
        signal.pair.price_change.h1 = json!(null);
        signal.pair.price_change.h24 = json!(null);
        let periods = Rc::new(RefCell::new(Vec::new()));
        let mut direct = dispatcher().with_token_signal_source(Box::new(WideningSignals {
            query_load: TokenSignalLoad {
                signal: Some(signal),
                diagnostics: Vec::new(),
            },
            photo: Ok(vec![1]),
            periods: Rc::clone(&periods),
            state: None,
        }));
        assert_eq!(
            direct.dispatch(update("/p syn 7d", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(direct.actions.0.iter().any(|action| matches!(action,
            TelegramAction::SendPhoto { reply_to_message_id: Some(MessageId(7)), caption, .. }
                if caption.contains("+25% 2d")
        )));
        assert_eq!(periods.borrow().as_slice(), ["7d"]);

        let mut widened = dispatcher().with_token_signal_source(Box::new(WideningSignals {
            query_load: TokenSignalLoad {
                signal: Some(token_signal()),
                diagnostics: Vec::new(),
            },
            photo: Ok(vec![1]),
            periods: Rc::clone(&periods),
            state: None,
        }));
        assert_eq!(
            widened.dispatch(update("/p syn", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(periods.borrow().as_slice(), ["7d", "7d"]);
        assert!(widened.actions.0.iter().any(|action| matches!(action,
            TelegramAction::SendPhoto { reply_to_message_id: Some(MessageId(7)), caption, .. }
                if caption.contains("+25% 2d")
        )));
        // The widened card remembers its state; refreshing it after the pair
        // disappeared from the provider answers that no data is available.
        let refresh = format!(
            "sig:ref:{}",
            bot_core::token_signals::stable_signal_id(-42, 7, 88, 1_672_531_200)
        );
        assert_eq!(
            widened.dispatch(callback_update(&refresh, "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            widened.actions.0.last(),
            Some(TelegramAction::AnswerCallback {
                text: Some(_),
                show_alert: true,
                ..
            })
        ));
    }

    #[test]
    fn dex_market_selection_persists_card_state_for_refresh_and_delete() -> Result<(), String> {
        for requested in [None, Some("7d")] {
            let expected = requested.unwrap_or("24h");
            let mut signal = token_signal();
            signal.candles.clear();
            let stored = Rc::new(RefCell::new(HashMap::new()));
            let selected = Rc::new(RefCell::new(Vec::new()));
            let token_state = Rc::new(RefCell::new(None));
            let saved = Rc::new(RefCell::new(Vec::new()));
            let periods = Rc::new(RefCell::new(Vec::new()));
            let token_results = Rc::new(RefCell::new(VecDeque::new()));
            let mut dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig::default()),
                    chat_ids: Vec::new(),
                },
                Actions::scripted(ActionScript {
                    receipts: Receipts::Sequential(700),
                    ..ActionScript::default()
                }),
                State::default(),
                values(),
                random(),
                authorization(),
                "@mybot",
            )
            .with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: Some(bot_core::market_prices::MarketSelection {
                        query: "syn".to_owned(),
                        timeframe: requested.map(str::to_owned),
                        target_symbol: "USD".to_owned(),
                        target_parameter: "USD".to_owned(),
                        conversion: None,
                        candidates: vec![bot_core::market_prices::MarketCandidate {
                            id: "1001".to_owned(),
                            symbol: "SYN".to_owned(),
                            name: "Native Synthetic".to_owned(),
                            slug: "synthetic".to_owned(),
                            price: "1".to_owned(),
                            change: "N/A 7d".to_owned(),
                            currency: String::new(),
                            exchange: String::new(),
                            asset_type: String::new(),
                            contracts: Vec::new(),
                        }],
                    }),
                    no_assets_found: false,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "selected provider quote".to_owned(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }))
            .with_token_signal_source(Box::new(StatefulSignals {
                signal,
                token_results,
                state: Rc::clone(&token_state),
                saved: Rc::clone(&saved),
                periods: Rc::clone(&periods),
            }));

            assert_eq!(
                dispatcher.dispatch(update(
                    &format!("/p syn {}", requested.unwrap_or("")),
                    Some("en")
                )),
                Ok(DispatchOutcome::Handled)
            );
            let selection_callback = dispatcher
                .actions
                .0
                .iter()
                .filter_map(sent_message)
                .find_map(|message| {
                    message
                        .reply_markup
                        .as_ref()?
                        .inline_keyboard
                        .get(1)?
                        .first()?
                        .callback_data
                        .clone()
                })
                .ok_or("DEX selection callback".to_owned())?;
            assert!(selection_callback.ends_with(":1"));

            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &selection_callback,
                    "private",
                    Some("en"),
                    700,
                )),
                Ok(DispatchOutcome::Handled)
            );
            let signal_callback = dispatcher
                .actions
                .0
                .iter()
                .find_map(|action| match action {
                    TelegramAction::SendPhoto {
                        reply_markup: Some(markup),
                        ..
                    } => markup.inline_keyboard.first()?.iter().find_map(|button| {
                        button
                            .callback_data
                            .as_deref()
                            .filter(|data| data.starts_with("sig:ref:"))
                            .map(ToOwned::to_owned)
                    }),
                    _ => None,
                })
                .ok_or("refresh callback on delivered DEX card".to_owned())?;
            assert!(dispatcher.actions.0.iter().any(|action| matches!(
                action,
                TelegramAction::SendPhoto {
                    reply_to_message_id: Some(MessageId(7)),
                    caption,
                    ..
                } if caption.contains(expected)
            )));
            assert_eq!(periods.borrow().as_slice(), [expected]);
            assert!(stored.borrow().is_empty());
            let saved = saved.borrow();
            assert_eq!(saved.len(), 1);
            assert_eq!(saved[0].1.message_id, 701);
            assert_eq!(saved[0].1.source_message_id, 7);
            assert_eq!(saved[0].1.requester_id, "88");
            assert_eq!(saved[0].1.chart_period.as_deref(), requested);
            assert_eq!(
                saved[0].1.address,
                "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump"
            );
            drop(saved);

            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &signal_callback,
                    "private",
                    Some("en"),
                    701,
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(periods.borrow().as_slice(), [expected, expected]);
            assert!(dispatcher.actions.0.iter().any(|action| matches!(
                action,
                TelegramAction::EditMessagePhoto {
                    message_id: MessageId(701),
                    caption,
                    ..
                } if caption.contains(expected)
            )));
            assert_eq!(
                token_state
                    .borrow()
                    .as_ref()
                    .and_then(|state| state.last_refresh_at),
                Some(1_672_531_200)
            );

            let delete_callback = signal_callback.replacen("sig:ref:", "sig:del:", 1);
            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &delete_callback,
                    "private",
                    Some("en"),
                    701,
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert!(dispatcher.actions.0.iter().any(|action| matches!(
                action,
                TelegramAction::DeleteMessage {
                    chat_id: ChatId(-42),
                    message_id: MessageId(701),
                }
            )));
            assert!(token_state.borrow().is_none());
            assert!(!dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message)
                if message.text == "Selection processed" || message.text == "selección procesada"
        )));
            assert_eq!(selected.borrow().as_slice(), &[] as &[String]);
        }
        Ok(())
    }

    #[test]
    fn failed_dex_market_lookup_keeps_menu_for_retry() -> Result<(), String> {
        let signal = token_signal();
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let token_results = Rc::new(RefCell::new(VecDeque::from([
            TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            TokenSignalLoad {
                signal: Some(signal.clone()),
                diagnostics: Vec::new(),
            },
        ])));
        let saved = Rc::new(RefCell::new(Vec::new()));
        let token_state = Rc::new(RefCell::new(None));
        let periods = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            Actions::scripted(ActionScript {
                receipts: Receipts::Sequential(700),
                ..ActionScript::default()
            }),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: MarketPriceLoad {
                chart: None,
                selection: Some(bot_core::market_prices::MarketSelection {
                    query: "syn".to_owned(),
                    timeframe: Some("7d".to_owned()),
                    target_symbol: "USD".to_owned(),
                    target_parameter: "USD".to_owned(),
                    conversion: None,
                    candidates: vec![bot_core::market_prices::MarketCandidate {
                        id: "1001".to_owned(),
                        symbol: "SYN".to_owned(),
                        name: "Native Synthetic".to_owned(),
                        slug: "synthetic".to_owned(),
                        price: "1".to_owned(),
                        change: "N/A 7d".to_owned(),
                        currency: String::new(),
                        exchange: String::new(),
                        asset_type: String::new(),
                        contracts: Vec::new(),
                    }],
                }),
                no_assets_found: false,
                text: String::new(),
                diagnostics: Vec::new(),
            },
            candidate: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: false,
                text: "selected provider quote".to_owned(),
                diagnostics: Vec::new(),
            },
            stored: Rc::clone(&stored),
            selected: Rc::clone(&selected),
        }))
        .with_token_signal_source(Box::new(StatefulSignals {
            signal,
            token_results,
            state: Rc::clone(&token_state),
            saved: Rc::clone(&saved),
            periods: Rc::clone(&periods),
        }));

        assert_eq!(
            dispatcher.dispatch(update("/p syn 7d", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let selection_callback = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()?
                    .inline_keyboard
                    .get(1)?
                    .first()?
                    .callback_data
                    .clone()
            })
            .ok_or("DEX selection callback".to_owned())?;

        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &selection_callback,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(stored.borrow().len(), 1);
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message)
                if message.reply_to_message_id == Some(MessageId(7))
                    && message.text.contains("get a quote")
        )));
        assert!(!dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::DeleteMessage {
                message_id: MessageId(700),
                ..
            }
        )));
        assert!(saved.borrow().is_empty());

        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &selection_callback,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(stored.borrow().is_empty());
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendPhoto {
                reply_to_message_id: Some(MessageId(7)),
                ..
            }
        )));
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::DeleteMessage {
                chat_id: ChatId(-42),
                message_id: MessageId(700),
            }
        )));
        assert_eq!(saved.borrow().len(), 1);
        assert_eq!(saved.borrow()[0].1.message_id, 702);
        assert_eq!(periods.borrow().as_slice(), ["7d"]);
        Ok(())
    }

    #[test]
    fn market_asset_fallbacks_cover_empty_quotes_and_verified_token_history() {
        struct EmptyMarket {
            with_chart: bool,
            token: Option<TokenAddress>,
        }

        impl MarketPriceSource for EmptyMarket {
            fn load(
                &mut self,
                _: &str,
                _: bot_core::market_prices::MarketPriceCommand,
                _: bot_core::locale::Locale,
                _: i64,
            ) -> MarketPriceLoad {
                MarketPriceLoad {
                    chart: self
                        .with_chart
                        .then(|| bot_core::market_prices::MarketChart {
                            timeframe: None,
                            symbol: "SYN".to_owned(),
                            name: "Synthetic".to_owned(),
                            yahoo_symbol: String::new(),
                            token: self.token.clone(),
                            candidate: None,
                        }),
                    selection: None,
                    no_assets_found: false,
                    text: String::new(),
                    diagnostics: Vec::new(),
                }
            }
        }

        for (input, language, expected) in [
            ("/p apple", "es", "Gráfico no disponible"),
            ("/p apple,banana", "en", "I could not get a quote for"),
        ] {
            let mut dispatcher = dispatcher().with_market_price_source(Box::new(EmptyMarket {
                with_chart: input == "/p apple",
                token: None,
            }));
            assert_eq!(
                dispatcher.dispatch(update(input, Some(language))),
                Ok(DispatchOutcome::Handled)
            );
            assert!(matches!(
                dispatcher.actions.0.last(),
                Some(TelegramAction::SendMessage(message)) if message.text.contains(expected)
            ));
        }

        for input in ["/p apple", "/p apple 1h"] {
            let token = TokenAddress {
                chain_id: "solana".to_owned(),
                network: "solana".to_owned(),
                tag: "SOL".to_owned(),
                address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            };
            let saved = Rc::new(RefCell::new(Vec::new()));
            let mut dispatcher = dispatcher()
                .with_market_price_source(Box::new(EmptyMarket {
                    with_chart: true,
                    token: Some(token),
                }))
                .with_token_signal_source(Box::new(Signals {
                    query_load: TokenSignalLoad {
                        signal: None,
                        diagnostics: Vec::new(),
                    },
                    token_load: TokenSignalLoad {
                        signal: Some(token_signal()),
                        diagnostics: vec!["token history fixture".to_owned()],
                    },
                    photo: Ok(vec![1, 2, 3]),
                    state: None,
                    queries: Rc::new(RefCell::new(Vec::new())),
                    saved: Rc::clone(&saved),
                }));
            assert_eq!(
                dispatcher.dispatch(update(input, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert!(matches!(
                dispatcher.actions.0.as_slice(),
                [TelegramAction::SendPhoto {
                    parse_mode: Some(bot_core::telegram_actions::ParseMode::Html),
                    reply_markup: Some(markup),
                    ..
                }] if !markup.inline_keyboard.is_empty()
            ));
            assert_eq!(saved.borrow().len(), 1);
            assert!(
                dispatcher
                    .state_diagnostics()
                    .iter()
                    .any(|diagnostic| diagnostic == "token history fixture")
            );
        }

        let mut no_chart_signal = token_signal();
        no_chart_signal.candles.clear();
        let mut dispatcher = dispatcher()
            .with_market_price_source(Box::new(EmptyMarket {
                with_chart: true,
                token: Some(TokenAddress {
                    chain_id: "solana".to_owned(),
                    network: "solana".to_owned(),
                    tag: "SOL".to_owned(),
                    address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
                }),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: Some(no_chart_signal),
                    diagnostics: Vec::new(),
                },
                photo: Err("history unavailable".to_owned()),
                state: None,
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p apple", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendMessage(message)] if message.text.contains("history available")
        ));

        let mut undelivered_token = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            photo_actions(Attempt::Skip),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(EmptyMarket {
            with_chart: true,
            token: Some(TokenAddress {
                chain_id: "solana".to_owned(),
                network: "solana".to_owned(),
                tag: "SOL".to_owned(),
                address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            }),
        }))
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: Some(token_signal()),
                diagnostics: Vec::new(),
            },
            photo: Ok(vec![1]),
            state: None,
            queries: Rc::new(RefCell::new(Vec::new())),
            saved: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            undelivered_token.dispatch(update("/p apple", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            undelivered_token
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("token signal photo delivery failed"))
        );
    }

    #[test]
    fn market_dispatch_boundaries_cover_query_guards_scopes_and_localized_fallbacks() {
        let mut unsupported = incoming_message("/p 0h", Some("en"));
        let message = &mut unsupported;
        let mut query_guards = dispatcher();
        assert_eq!(
            query_guards.dispatch_asset_prices(
                message,
                "0h",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(None)
        );
        assert_eq!(
            query_guards.dispatch_asset_prices(
                message,
                "123",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(None)
        );
        message.chat_id = None;
        assert_eq!(
            query_guards.dispatch_asset_prices(
                message,
                "btc",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(Some(DispatchOutcome::Unsupported))
        );

        let message = incoming_message("/p btc", Some("en"));
        let mut no_market = dispatcher();
        assert_eq!(
            no_market.dispatch_market_price_query(
                &message,
                "btc",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Err(DispatchError::MissingService("market prices"))
        );
        let mut no_chat = message.clone();
        no_chat.chat_id = None;
        assert_eq!(
            no_market.dispatch_market_price_query(
                &no_chat,
                "btc",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(Some(DispatchOutcome::Unsupported))
        );

        let mut no_sender = message.clone();
        no_sender.sender_id = None;
        let mut selection_fallback =
            dispatcher().with_market_price_source(Box::new(MarketPrices {
                result: market_selection_load(None),
                calls: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            selection_fallback.dispatch_market_price_query(
                &no_sender,
                "libra",
                bot_core::market_prices::MarketPriceCommand::CryptoOnly,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(Some(DispatchOutcome::Handled))
        );
        assert!(matches!(
            selection_fallback.actions.0.as_slice(),
            [TelegramAction::SendMessage(message)] if message.reply_markup.is_none()
        ));

        for (locale, expected) in [
            (
                bot_core::locale::Locale::Es,
                "No pude conseguir una cotización. Probá más tarde",
            ),
            (
                bot_core::locale::Locale::En,
                "I could not get a quote. Try again later",
            ),
        ] {
            let mut empty = dispatcher().with_market_price_source(Box::new(MarketPrices {
                result: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                calls: Rc::new(RefCell::new(Vec::new())),
            }));
            assert_eq!(
                empty.dispatch_market_price_query(
                    &message,
                    "btc",
                    bot_core::market_prices::MarketPriceCommand::Unified,
                    locale,
                    1_672_531_200,
                ),
                Ok(Some(DispatchOutcome::Handled))
            );
            assert!(matches!(
                empty.actions.0.as_slice(),
                [TelegramAction::SendMessage(reply)] if reply.text == expected
            ));
        }

        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut scoped = dispatcher().with_market_price_source(Box::new(MarketPrices {
            result: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: false,
                text: "scoped quote".to_owned(),
                diagnostics: Vec::new(),
            },
            calls: Rc::clone(&calls),
        }));
        assert_eq!(
            scoped.dispatch_asset_prices(
                &message,
                "crypto:btc,eth 24h",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(Some(DispatchOutcome::Handled))
        );
        assert_eq!(
            calls
                .borrow()
                .iter()
                .map(|call| call.0.as_str())
                .collect::<Vec<_>>(),
            ["crypto:btc 24h", "crypto:eth 24h"]
        );

        let mut empty_requests = dispatcher().with_market_price_source(Box::new(MarketPrices {
            result: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: true,
                text: String::new(),
                diagnostics: Vec::new(),
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            empty_requests.dispatch_asset_prices(
                &message,
                ",,,",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(None)
        );
    }

    #[test]
    fn ambiguous_market_selection_is_requester_bound_and_resolves_by_provider_id()
    -> Result<(), String> {
        let selection = bot_core::market_prices::MarketSelection {
            query: "$libra".to_owned(),
            timeframe: None,
            target_symbol: "USD".to_owned(),
            target_parameter: "USD".to_owned(),
            conversion: None,
            candidates: vec![
                bot_core::market_prices::MarketCandidate {
                    id: "1001".to_owned(),
                    symbol: "LIBRA".to_owned(),
                    name: "Libra Finance".to_owned(),
                    slug: "libra-finance".to_owned(),
                    price: "0.007".to_owned(),
                    change: "N/A".to_owned(),
                    currency: String::new(),
                    exchange: String::new(),
                    asset_type: String::new(),
                    contracts: Vec::new(),
                },
                bot_core::market_prices::MarketCandidate {
                    id: "1002".to_owned(),
                    symbol: "LIBRA".to_owned(),
                    name: "Libra Protocol".to_owned(),
                    slug: "libra-protocol".to_owned(),
                    price: "0.00009".to_owned(),
                    change: "N/A".to_owned(),
                    currency: String::new(),
                    exchange: String::new(),
                    asset_type: String::new(),
                    contracts: Vec::new(),
                },
            ],
        };
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: Some(selection),
                    no_assets_found: false,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "LIBRA: 0.00009 USD (N/A 24h)".to_owned(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }));
        assert_eq!(
            dispatcher.dispatch(update("$libra 1h", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let callback_data = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()
                    .and_then(|keyboard| keyboard.inline_keyboard.get(1))
                    .and_then(|row| row.first())
                    .and_then(|button| button.callback_data.clone())
            })
            .ok_or("selection callback".to_owned())?;
        assert!(callback_data.starts_with("mkt:select:"));
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback_data,
                "private",
                Some("en"),
                701,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(selected.borrow().is_empty());
        assert!(!stored.borrow().is_empty());
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback_data,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(selected.borrow().as_slice(), &["1002:1h"]);
        assert!(stored.borrow().is_empty());
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message) if message.text.contains("0.00009")
        )));
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::AnswerCallback {
                show_alert: false,
                ..
            }
        )));
        Ok(())
    }

    #[test]
    fn comma_separated_ambiguous_requests_keep_independent_callback_menus() -> Result<(), String> {
        let candidate = |id: &str, name: &str| bot_core::market_prices::MarketCandidate {
            id: id.to_owned(),
            symbol: "LIBRA".to_owned(),
            name: name.to_owned(),
            slug: name.to_ascii_lowercase().replace(' ', "-"),
            price: "0.007".to_owned(),
            change: "N/A".to_owned(),
            currency: String::new(),
            exchange: String::new(),
            asset_type: String::new(),
            contracts: Vec::new(),
        };
        let selection = |query: &str, candidates: Vec<bot_core::market_prices::MarketCandidate>| {
            bot_core::market_prices::MarketSelection {
                query: query.to_owned(),
                timeframe: None,
                target_symbol: "USD".to_owned(),
                target_parameter: "USD".to_owned(),
                conversion: None,
                candidates,
            }
        };
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(MultiSelectableMarketPrices {
                loads: HashMap::from([
                    (
                        "btc".to_owned(),
                        MarketPriceLoad {
                            chart: None,
                            selection: None,
                            no_assets_found: false,
                            text: "BTC: 50000 USD (+2.5% 24h)".to_owned(),
                            diagnostics: Vec::new(),
                        },
                    ),
                    (
                        "libra".to_owned(),
                        MarketPriceLoad {
                            chart: None,
                            selection: Some(selection(
                                "libra",
                                vec![
                                    candidate("L1", "Libra Finance"),
                                    candidate("L2", "Libra Protocol"),
                                ],
                            )),
                            no_assets_found: false,
                            text: String::new(),
                            diagnostics: Vec::new(),
                        },
                    ),
                    (
                        "trump".to_owned(),
                        MarketPriceLoad {
                            chart: None,
                            selection: Some(selection(
                                "trump",
                                vec![
                                    candidate("T1", "Trump Finance"),
                                    candidate("T2", "Trump Protocol"),
                                ],
                            )),
                            no_assets_found: false,
                            text: String::new(),
                            diagnostics: Vec::new(),
                        },
                    ),
                ]),
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "selected quote".to_owned(),
                    diagnostics: Vec::new(),
                },
                candidate_results: VecDeque::new(),
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }));

        assert_eq!(
            dispatcher.dispatch(update("/p btc,libra,trump", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let menus = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .filter_map(|message| {
                message.reply_markup.as_ref().map(|keyboard| {
                    (
                        message.text.clone(),
                        keyboard
                            .inline_keyboard
                            .iter()
                            .filter_map(|row| row.first()?.callback_data.clone())
                            .collect::<Vec<_>>(),
                    )
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(menus.len(), 2);
        assert!(menus[0].0.contains("libra"));
        assert!(menus[1].0.contains("trump"));
        assert_eq!(menus[0].1.len(), 3);
        assert_eq!(menus[1].1.len(), 3);
        let libra_second = menus[0].1[1].clone();
        let trump_first = menus[1].1[0].clone();
        let libra_selection_id = libra_second
            .split(':')
            .nth(2)
            .ok_or("libra callback id".to_owned())?;
        let trump_selection_id = trump_first
            .split(':')
            .nth(2)
            .ok_or("trump callback id".to_owned())?;
        assert_ne!(libra_selection_id, trump_selection_id);
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message) if message.text.starts_with("BTC:")
        )));

        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &libra_second,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(selected.borrow().as_slice(), &["L2:"]);
        assert_eq!(stored.borrow().len(), 1);

        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &trump_first,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(selected.borrow().as_slice(), &["L2:", "T1:"]);
        assert!(stored.borrow().is_empty());
        Ok(())
    }

    #[test]
    fn failed_market_selection_callback_keeps_state_for_retry() -> Result<(), String> {
        let selection = bot_core::market_prices::MarketSelection {
            query: "libra".to_owned(),
            timeframe: Some("7d".to_owned()),
            target_symbol: "USD".to_owned(),
            target_parameter: "USD".to_owned(),
            conversion: None,
            candidates: vec![bot_core::market_prices::MarketCandidate {
                id: "L1".to_owned(),
                symbol: "LIBRA".to_owned(),
                name: "Libra Finance".to_owned(),
                slug: "libra-finance".to_owned(),
                price: "0.007".to_owned(),
                change: "N/A".to_owned(),
                currency: String::new(),
                exchange: String::new(),
                asset_type: String::new(),
                contracts: Vec::new(),
            }],
        };
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(MultiSelectableMarketPrices {
                loads: HashMap::from([(
                    "libra".to_owned(),
                    MarketPriceLoad {
                        chart: None,
                        selection: Some(selection),
                        no_assets_found: false,
                        text: String::new(),
                        diagnostics: Vec::new(),
                    },
                )]),
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "LIBRA: 0.007 USD (N/A 24h)".to_owned(),
                    diagnostics: Vec::new(),
                },
                candidate_results: VecDeque::from([
                    MarketPriceLoad {
                        chart: None,
                        selection: None,
                        no_assets_found: true,
                        text: String::new(),
                        diagnostics: vec!["temporary quote failure".to_owned()],
                    },
                    MarketPriceLoad {
                        chart: None,
                        selection: None,
                        no_assets_found: false,
                        text: "LIBRA: 0.007 USD (N/A 24h)".to_owned(),
                        diagnostics: Vec::new(),
                    },
                ]),
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra 7d", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let callback = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()?
                    .inline_keyboard
                    .first()?
                    .first()?
                    .callback_data
                    .clone()
            })
            .ok_or("selection callback".to_owned())?;
        assert_eq!(stored.borrow().len(), 1);

        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(selected.borrow().as_slice(), &["L1:7d"]);
        assert_eq!(stored.borrow().len(), 1);
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::AnswerCallback {
                show_alert: true,
                ..
            }
        )));

        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(selected.borrow().as_slice(), &["L1:7d", "L1:7d"]);
        assert!(stored.borrow().is_empty());
        Ok(())
    }

    #[test]
    fn double_tapped_market_selection_resolves_into_a_single_quote() -> Result<(), String> {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: market_selection_load(None),
                candidate: market_candidate_quote(),
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let callback = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()?
                    .inline_keyboard
                    .first()?
                    .first()?
                    .callback_data
                    .clone()
            })
            .ok_or("selection callback".to_owned())?;
        assert!(callback.starts_with("mkt:select:"));
        assert_eq!(stored.borrow().len(), 1);

        let select = |dispatcher: &mut NativeDispatcher<
            Config,
            Actions,
            State,
            Values,
            Samples,
            Authorization,
        >| {
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700,
            ))
        };
        assert_eq!(select(&mut dispatcher), Ok(DispatchOutcome::Handled));
        // The same tap delivered twice: the atomic take lets exactly one of
        // them resolve, the loser gets the expired toast.
        assert_eq!(select(&mut dispatcher), Ok(DispatchOutcome::Handled));
        assert_eq!(selected.borrow().len(), 1);
        let quotes = dispatcher
            .actions
            .0
            .iter()
            .filter(|action| {
                matches!(action, TelegramAction::SendMessage(message) if message.text == "LIBRA: 0.007 USD (N/A 24h)")
            })
            .count();
        assert_eq!(quotes, 1);
        assert!(stored.borrow().is_empty());
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::AnswerCallback {
                show_alert: true,
                ..
            }
        )));
        Ok(())
    }

    #[test]
    fn failed_quote_send_restores_the_selection_for_retry() -> Result<(), String> {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            failing_actions(
                ActionKind::SendMessage,
                0,
                "synthetic quote send failure",
                true,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let callback = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()?
                    .inline_keyboard
                    .first()?
                    .first()?
                    .callback_data
                    .clone()
            })
            .ok_or("selection callback".to_owned())?;
        dispatcher.actions.1.failure = Some(ExecuteFailure {
            kind: ActionKind::SendMessage,
            remaining: 1,
            error: "synthetic quote send failure",
            record: true,
        });
        assert!(matches!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700
            )),
            Err(DispatchError::Action(_))
        ));
        // The consumed menu was put back, so the runtime retry resolves it.
        assert_eq!(stored.borrow().len(), 1);
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(stored.borrow().is_empty());
        let delivered = dispatcher
            .state
            .outgoing
            .iter()
            .filter(|plan| plan.message.text == "LIBRA: 0.007 USD (N/A 24h)")
            .count();
        assert_eq!(delivered, 1);
        Ok(())
    }

    struct ScriptedTakeMarketPrices {
        load: Result<Option<String>, String>,
        takes: RefCell<VecDeque<Result<Option<String>, String>>>,
        saves: RefCell<VecDeque<Result<(), String>>>,
        candidate: MarketPriceLoad,
    }

    impl MarketPriceSource for ScriptedTakeMarketPrices {
        fn load(
            &mut self,
            _: &str,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            market_selection_load(None)
        }

        fn load_candidate(
            &mut self,
            _: &bot_core::market_prices::MarketCandidate,
            _: Option<&str>,
            _: &str,
            _: &str,
            _: Option<&bot_core::market_prices::MarketConversion>,
            _: bot_core::market_prices::MarketPriceCommand,
            _: bot_core::locale::Locale,
            _: i64,
        ) -> MarketPriceLoad {
            self.candidate.clone()
        }

        fn save_selection(&mut self, _: &str, _: &str, _: i64) -> Result<(), String> {
            self.saves.borrow_mut().pop_front().unwrap_or(Ok(()))
        }

        fn load_selection(&mut self, _key: &str) -> Result<Option<String>, String> {
            self.load.clone()
        }

        fn take_selection(&mut self, _key: &str) -> Result<Option<String>, String> {
            self.takes.borrow_mut().pop_front().unwrap_or(Ok(None))
        }
    }

    fn stored_selection_value() -> Result<String, String> {
        Ok(StoredMarketSelection {
            selection: market_selection_fixture(None),
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: Some(6),
            requester_id: 88,
            command: "crypto".to_owned(),
        }
        .encode())
    }

    fn race_callback() -> IncomingUpdate {
        callback_update_with_context(
            "mkt:select:race:0",
            json!(-42),
            "private",
            7,
            Some(88),
            Some("en"),
            Some("callback-race"),
        )
    }

    #[test]
    fn losing_the_take_race_answers_expired_without_sending() -> Result<(), String> {
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(ScriptedTakeMarketPrices {
                load: Ok(Some(stored_selection_value()?)),
                takes: RefCell::new(VecDeque::from([Ok(None)])),
                saves: RefCell::new(VecDeque::from([Ok(()), Ok(())])),
                candidate: market_candidate_quote(),
            }));
        // Route a command through first so the scripted load and save stay
        // covered; the raced callback below still takes nothing.
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(race_callback()),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message) if message.reply_markup.is_some()
        )));
        let answers = dispatcher
            .actions
            .0
            .iter()
            .filter(|action| {
                matches!(
                    action,
                    TelegramAction::AnswerCallback {
                        show_alert: true,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(answers, 1);
        assert!(dispatcher.actions.0.iter().all(|action| matches!(
            action,
            TelegramAction::SendMessage(_) | TelegramAction::AnswerCallback { .. }
        )));
        Ok(())
    }

    #[test]
    fn failed_take_answers_expired_with_diagnostics() -> Result<(), String> {
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(ScriptedTakeMarketPrices {
                load: Ok(Some(stored_selection_value()?)),
                takes: RefCell::new(VecDeque::from([Err("synthetic take failure".to_owned())])),
                saves: RefCell::new(VecDeque::new()),
                candidate: market_candidate_quote(),
            }));
        assert_eq!(
            dispatcher.dispatch(race_callback()),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection take failed"))
        );
        Ok(())
    }

    #[test]
    fn undecodable_take_answers_expired_with_diagnostics() -> Result<(), String> {
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(ScriptedTakeMarketPrices {
                load: Ok(Some(stored_selection_value()?)),
                takes: RefCell::new(VecDeque::from([Ok(Some("not-json".to_owned()))])),
                saves: RefCell::new(VecDeque::new()),
                candidate: market_candidate_quote(),
            }));
        assert_eq!(
            dispatcher.dispatch(race_callback()),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection take decode failed"))
        );
        Ok(())
    }

    #[test]
    fn failed_restore_is_diagnostic_and_keeps_the_retry_text() -> Result<(), String> {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(SelectionStorageMarketPrices {
                initial: market_selection_load(None),
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: vec!["candidate quote unavailable".to_owned()],
                },
                stored: Rc::clone(&stored),
                save_calls: Rc::new(RefCell::new(0)),
                // The menu setup itself saves twice (store plus the
                // post-delivery update), so the restore is call three.
                fail_save_at: Some(3),
                fail_load: None,
                fail_clear: None,
                render_success: false,
                render_caption: None,
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let callback = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()?
                    .inline_keyboard
                    .first()?
                    .first()?
                    .callback_data
                    .clone()
            })
            .ok_or("selection callback".to_owned())?;
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection restore failed"))
        );
        assert!(
            dispatcher
                .actions
                .0
                .iter()
                .any(|action| matches!(action, TelegramAction::SendMessage(message) if message.text.contains("I could not get a quote")))
        );
        Ok(())
    }

    #[test]
    fn failed_menu_deletes_stay_handled_with_diagnostics() -> Result<(), String> {
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            failing_actions(
                ActionKind::DeleteMessage,
                usize::MAX,
                "synthetic delete failure",
                false,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_scheduled_task_source(Box::new(Tasks {
            lists: Vec::new(),
            cancellations: Rc::new(RefCell::new(Vec::new())),
        }));
        // A double-tapped close/delete removes an already gone message. The
        // failure is a diagnostic, never a retry into quarantine noise.
        for data in ["topup:close", "help:close", "task:close"] {
            assert_eq!(
                dispatcher.dispatch(callback_update(data, "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        let failures = dispatcher
            .state_diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.contains("callback delete failed"))
            .count();
        assert_eq!(failures, 3);
        Ok(())
    }

    #[test]
    fn failed_signal_delete_stays_handled_with_diagnostics() -> Result<(), String> {
        let signal = token_signal();
        let state = SignalState {
            chart_period: None,
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: 6,
            requester_id: "88".to_owned(),
            chain_id: signal.token.chain_id.clone(),
            network: signal.token.network.clone(),
            tag: signal.token.tag.clone(),
            address: signal.token.address.clone(),
            last_refresh_at: None,
        };
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            failing_actions(
                ActionKind::DeleteMessage,
                usize::MAX,
                "synthetic delete failure",
                false,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: Some(signal),
                diagnostics: Vec::new(),
            },
            photo: Ok(b"unused".to_vec()),
            state: Some(state),
            queries: Rc::new(RefCell::new(Vec::new())),
            saved: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            dispatcher.dispatch(callback_update("sig:del:abc", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("callback delete failed"))
        );
        Ok(())
    }

    #[test]
    fn failed_post_delivery_toast_does_not_retry_into_a_duplicate() -> Result<(), String> {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            failing_actions(
                ActionKind::AnswerCallback,
                usize::MAX,
                "synthetic toast failure",
                false,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let callback = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()?
                    .inline_keyboard
                    .first()?
                    .first()?
                    .callback_data
                    .clone()
            })
            .ok_or("selection callback".to_owned())?;
        // The quote was delivered; the dead toast is a diagnostic, not a
        // retry into a second quote.
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection answer failed"))
        );
        assert!(stored.borrow().is_empty());
        let delivered = dispatcher
            .state
            .outgoing
            .iter()
            .filter(|plan| plan.message.text == "LIBRA: 0.007 USD (N/A 24h)")
            .count();
        assert_eq!(delivered, 1);
        Ok(())
    }

    fn token_selection_value() -> Result<String, String> {
        let token = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
        };
        Ok(StoredMarketSelection {
            selection: MarketSelection {
                query: "syn".to_owned(),
                timeframe: Some("1m".to_owned()),
                target_symbol: "USD".to_owned(),
                target_parameter: "USD".to_owned(),
                conversion: None,
                candidates: vec![bot_core::market_prices::MarketCandidate {
                    id: "token:solana:solana:J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump"
                        .to_owned(),
                    symbol: "SYN".to_owned(),
                    name: "Synthetic Token".to_owned(),
                    slug: "syn".to_owned(),
                    price: "0.01".to_owned(),
                    change: "N/A 1m".to_owned(),
                    currency: String::new(),
                    exchange: String::new(),
                    asset_type: String::new(),
                    contracts: vec![token],
                }],
            },
            chat_id: "-42".to_owned(),
            message_id: 700,
            source_message_id: Some(6),
            requester_id: 88,
            command: "unified".to_owned(),
        }
        .encode())
    }

    fn token_selection_dispatcher<Actions>(
        actions: Actions,
        stored: Rc<RefCell<HashMap<String, String>>>,
    ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization>
    where
        Actions: ActionSink,
        Actions::Error: std::fmt::Display,
    {
        NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            actions,
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectionStorageMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored,
            save_calls: Rc::new(RefCell::new(0)),
            fail_save_at: None,
            fail_load: None,
            fail_clear: None,
            render_success: false,
            render_caption: None,
        }))
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: Some(token_signal()),
                diagnostics: Vec::new(),
            },
            photo: Err("synthetic photo failure".to_owned()),
            state: None,
            queries: Rc::new(RefCell::new(Vec::new())),
            saved: Rc::new(RefCell::new(Vec::new())),
        }))
    }

    #[test]
    fn failed_token_quote_send_restores_the_selection_for_retry() -> Result<(), String> {
        let selection_id = market_selection_id(-42, 700, 88, 1_672_531_200, 0);
        let stored = Rc::new(RefCell::new(HashMap::from([(
            market_selection_key(&selection_id),
            token_selection_value()?,
        )])));
        let mut dispatcher = token_selection_dispatcher(
            failing_actions(
                ActionKind::SendMessage,
                1,
                "synthetic quote send failure",
                true,
            ),
            Rc::clone(&stored),
        );
        let callback = format!("mkt:select:{selection_id}:0");
        let select = |dispatcher: &mut NativeDispatcher<
            Config,
            Actions,
            State,
            Values,
            Samples,
            Authorization,
        >| {
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700,
            ))
        };
        assert!(matches!(
            select(&mut dispatcher),
            Err(DispatchError::Action(_))
        ));
        assert_eq!(stored.borrow().len(), 1);
        assert_eq!(select(&mut dispatcher), Ok(DispatchOutcome::Handled));
        assert!(stored.borrow().is_empty());
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        Ok(())
    }

    #[test]
    fn failed_token_post_delivery_toast_does_not_retry_into_a_duplicate() -> Result<(), String> {
        let selection_id = market_selection_id(-42, 700, 88, 1_672_531_200, 0);
        let stored = Rc::new(RefCell::new(HashMap::from([(
            market_selection_key(&selection_id),
            token_selection_value()?,
        )])));
        let mut dispatcher = token_selection_dispatcher(
            failing_actions(
                ActionKind::AnswerCallback,
                usize::MAX,
                "synthetic toast failure",
                false,
            ),
            Rc::clone(&stored),
        );
        let callback = format!("mkt:select:{selection_id}:0");
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection answer failed"))
        );
        assert!(stored.borrow().is_empty());
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        Ok(())
    }

    #[test]
    fn invalid_taken_candidate_restores_the_menu() -> Result<(), String> {
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(ScriptedTakeMarketPrices {
                load: Ok(Some(stored_selection_value()?)),
                takes: RefCell::new(VecDeque::from([Ok(Some(stored_selection_value()?))])),
                saves: RefCell::new(VecDeque::new()),
                candidate: market_candidate_quote(),
            }));
        assert_eq!(
            dispatcher.dispatch(callback_update_with_context(
                "mkt:select:race:5",
                json!(-42),
                "private",
                7,
                Some(88),
                Some("en"),
                Some("callback-race-invalid"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .actions
                .0
                .iter()
                .all(|action| matches!(action, TelegramAction::AnswerCallback { .. }))
        );
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::AnswerCallback {
                show_alert: true,
                ..
            }
        )));
        Ok(())
    }

    #[test]
    fn failed_retry_toast_after_undeliverable_quote_stays_handled() -> Result<(), String> {
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            failing_actions(
                ActionKind::AnswerCallback,
                usize::MAX,
                "synthetic toast failure",
                false,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(ScriptedTakeMarketPrices {
            load: Ok(Some(stored_selection_value()?)),
            takes: RefCell::new(VecDeque::from([Ok(Some(stored_selection_value()?))])),
            saves: RefCell::new(VecDeque::new()),
            candidate: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: true,
                text: String::new(),
                diagnostics: Vec::new(),
            },
        }));
        assert_eq!(
            dispatcher.dispatch(race_callback()),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection answer failed"))
        );
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message) if message.text.contains("I could not get a quote")
        )));
        Ok(())
    }

    #[test]
    fn failed_token_retry_toast_after_missing_signal_stays_handled() -> Result<(), String> {
        let selection_id = market_selection_id(-42, 700, 88, 1_672_531_200, 0);
        let stored = Rc::new(RefCell::new(HashMap::from([(
            market_selection_key(&selection_id),
            token_selection_value()?,
        )])));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            failing_actions(
                ActionKind::AnswerCallback,
                usize::MAX,
                "synthetic toast failure",
                false,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectionStorageMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            save_calls: Rc::new(RefCell::new(0)),
            fail_save_at: None,
            fail_load: None,
            fail_clear: None,
            render_success: false,
            render_caption: None,
        }))
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            photo: Err("synthetic photo failure".to_owned()),
            state: None,
            queries: Rc::new(RefCell::new(Vec::new())),
            saved: Rc::new(RefCell::new(Vec::new())),
        }));
        let callback = format!("mkt:select:{selection_id}:0");
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("market selection answer failed"))
        );
        assert_eq!(dispatcher.state.outgoing.len(), 1);
        Ok(())
    }

    #[test]
    fn single_assets_get_canonical_charts_and_mixed_lists_resolve_tokens() {
        for (input, photo, expected) in [
            ("/c", false, "missing"),
            ("/c bitcoin", true, "BTC"),
            ("/c btc", true, "BTC"),
            ("/p apple", true, "AAPL"),
            ("/p $apple", true, "AAPL"),
            ("$BTC", true, "BTC"),
            ("$F", true, "F"),
            ("/p btc,timba", false, "TIMBA"),
            ("/c bitcoin,timba", false, "TIMBA"),
            ("/p btc,unknown", false, "missing"),
            ("/c stables", false, "missing"),
            ("/c stablecoins", false, "missing"),
        ] {
            let charts = Rc::new(RefCell::new(Vec::new()));
            let queries = Rc::new(RefCell::new(Vec::new()));
            let mut signal = token_signal();
            signal.pair.base_token.symbol = "TIMBA".to_owned();
            let token_load = if input.contains("unknown") {
                None
            } else {
                Some(signal)
            };
            let mut dispatcher = dispatcher()
                .with_market_price_source(Box::new(ChartPrices {
                    charts: Rc::clone(&charts),
                    fail_chart: false,
                }))
                .with_token_signal_source(Box::new(Signals {
                    query_load: TokenSignalLoad {
                        signal: token_load,
                        diagnostics: Vec::new(),
                    },
                    token_load: TokenSignalLoad {
                        signal: None,
                        diagnostics: Vec::new(),
                    },
                    photo: Ok(vec![1]),
                    state: None,
                    queries: Rc::clone(&queries),
                    saved: Default::default(),
                }));
            assert_eq!(
                dispatcher.dispatch(update(input, Some("en"))),
                Ok(DispatchOutcome::Handled),
                "{input}"
            );
            if photo {
                assert!(
                    matches!(dispatcher.actions.0.as_slice(), [TelegramAction::SendPhoto { caption, reply_to_message_id: Some(MessageId(7)), .. }] if caption.contains(expected)),
                    "{input}"
                );
                assert!(queries.borrow().is_empty());
                assert_eq!(charts.borrow().len(), 1);
            } else {
                assert!(
                    matches!(dispatcher.actions.0.as_slice(), [TelegramAction::SendMessage(reply)] if reply.text.contains(expected)),
                    "{input}"
                );
                assert!(charts.borrow().is_empty());
                if input.contains("timba") {
                    assert_eq!(
                        queries.borrow().as_slice(),
                        &[SignalQuery::Symbol("timba".to_owned())]
                    );
                    assert!(
                        matches!(dispatcher.actions.0.first(), Some(TelegramAction::SendMessage(reply)) if reply.text.contains("BTC:"))
                    );
                }
                if input.contains("stables") || input.contains("stablecoins") {
                    assert!(queries.borrow().is_empty());
                }
            }
            assert_eq!(dispatcher.state.incoming.len(), 1);
            assert_eq!(dispatcher.state.outgoing.len(), 1);
        }
    }

    #[test]
    fn unified_symbol_queries_offer_stock_and_token_candidates_together() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let market_candidate = bot_core::market_prices::MarketCandidate {
            id: "stock:RKHNF".to_owned(),
            symbol: "RKHNF".to_owned(),
            name: "RKHNF".to_owned(),
            slug: "rkhnf".to_owned(),
            price: "0.13".to_owned(),
            change: "+2.77% 1m".to_owned(),
            currency: "USD".to_owned(),
            exchange: "Synthetic".to_owned(),
            asset_type: "Equity".to_owned(),
            contracts: Vec::new(),
        };
        let mut signal = token_signal();
        signal.pair.base_token.symbol = "RKH".to_owned();
        signal.pair.base_token.name = "Roaring Kity Hacked".to_owned();
        let callback_signal = signal.clone();
        let queries = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = dispatcher()
            .with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: Some(bot_core::market_prices::MarketChart {
                        timeframe: None,
                        symbol: "RKHNF".to_owned(),
                        name: "RKHNF".to_owned(),
                        yahoo_symbol: "RKHNF".to_owned(),
                        token: None,
                        candidate: Some(market_candidate),
                    }),
                    selection: None,
                    no_assets_found: false,
                    text: "RKHNF: 0.13 USD (+2.77% 1m)".to_owned(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "RKHNF: 0.13 USD (+2.77% 1m)".to_owned(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&stored),
                selected: Rc::new(RefCell::new(Vec::new())),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: Some(signal),
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: Some(callback_signal),
                    diagnostics: Vec::new(),
                },
                photo: Ok(vec![1]),
                state: None,
                queries: Rc::clone(&queries),
                saved: Default::default(),
            }));

        assert_eq!(
            dispatcher.dispatch(update("/p rkh 1m", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.first(),
            Some(TelegramAction::SendMessage(_))
        ));
        let message = first_sent(&dispatcher.actions.0);
        assert!(
            message
                .reply_markup
                .as_ref()
                .into_iter()
                .flat_map(|markup| markup.inline_keyboard.iter())
                .flatten()
                .any(|b| b.text.contains("RKHNF"))
        );
        assert!(
            message
                .reply_markup
                .as_ref()
                .into_iter()
                .flat_map(|markup| markup.inline_keyboard.iter())
                .flatten()
                .any(|b| b.text.contains("Roaring Kity Hacked"))
        );
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(3)
        );
        assert_eq!(
            queries.borrow().as_slice(),
            &[SignalQuery::Symbol("rkh".to_owned())]
        );
        assert_eq!(stored.borrow().len(), 1);

        let callback = dispatcher
            .actions
            .0
            .iter()
            .filter_map(sent_message)
            .find_map(|message| {
                message
                    .reply_markup
                    .as_ref()?
                    .inline_keyboard
                    .get(1)?
                    .first()?
                    .callback_data
                    .clone()
            });
        assert!(callback.is_some(), "token selection callback");
        let callback = callback.unwrap_or_default();
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendPhoto {
                parse_mode: Some(bot_core::telegram_actions::ParseMode::Html),
                ..
            }
        )));
        assert!(stored.borrow().is_empty());
    }

    #[test]
    fn stock_command_preserves_listing_identity_through_selection_callback() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let selected = Rc::new(RefCell::new(Vec::new()));
        let london = bot_core::market_prices::MarketCandidate {
            id: "stock:RKH.L".to_owned(),
            symbol: "RKH.L".to_owned(),
            name: "Rockhopper Exploration plc".to_owned(),
            slug: "rkh.l".to_owned(),
            price: "12.5".to_owned(),
            change: "+25.00% 1m".to_owned(),
            currency: "GBp".to_owned(),
            exchange: "London".to_owned(),
            asset_type: "Equity".to_owned(),
            contracts: Vec::new(),
        };
        let otc = bot_core::market_prices::MarketCandidate {
            id: "stock:RKHNF".to_owned(),
            symbol: "RKHNF".to_owned(),
            name: "Rockhaven Resources Ltd.".to_owned(),
            slug: "rkhnf".to_owned(),
            price: "0.13".to_owned(),
            change: "+2.77% 1m".to_owned(),
            currency: "USD".to_owned(),
            exchange: "OTC Markets".to_owned(),
            asset_type: "Equity".to_owned(),
            contracts: Vec::new(),
        };
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: Some(bot_core::market_prices::MarketSelection {
                        query: "rkh".to_owned(),
                        timeframe: Some("1m".to_owned()),
                        target_symbol: "USD".to_owned(),
                        target_parameter: "USD".to_owned(),
                        conversion: None,
                        candidates: vec![london.clone(), otc],
                    }),
                    no_assets_found: false,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: Some(bot_core::market_prices::MarketChart {
                        timeframe: Some("1m".to_owned()),
                        symbol: london.symbol.clone(),
                        name: london.name.clone(),
                        yahoo_symbol: london.symbol.clone(),
                        token: None,
                        candidate: Some(london),
                    }),
                    selection: None,
                    no_assets_found: false,
                    text: "RKH.L: 12.50 GBp (+25.00% 1m)".to_owned(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&stored),
                selected: Rc::clone(&selected),
            }));

        assert_eq!(
            dispatcher.dispatch(update("/s rkh 1m", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let message = first_sent(&dispatcher.actions.0);
        assert!(
            message
                .reply_markup
                .as_ref()
                .into_iter()
                .flat_map(|markup| markup.inline_keyboard.iter())
                .flatten()
                .any(|b| b.text.contains("RKH.L"))
        );
        assert!(
            message
                .reply_markup
                .as_ref()
                .into_iter()
                .flat_map(|markup| markup.inline_keyboard.iter())
                .flatten()
                .any(|b| b.text.contains("London"))
        );
        assert!(!message.text.contains("GBp"));
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(3)
        );

        let callback = message
            .reply_markup
            .as_ref()
            .and_then(|markup| markup.inline_keyboard.first())
            .and_then(|row| row.first())
            .and_then(|button| button.callback_data.clone());
        assert!(callback.is_some(), "selection callback");
        let callback = callback.unwrap_or_default();
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &callback,
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(selected.borrow().as_slice(), ["stock:RKH.L:1m"]);
        assert!(stored.borrow().is_empty());
    }

    #[test]
    fn crypto_symbol_queries_merge_direct_native_and_token_candidates() {
        let market_candidate = bot_core::market_prices::MarketCandidate {
            id: "123".to_owned(),
            symbol: "RKH".to_owned(),
            name: "Native RKH".to_owned(),
            slug: "rkh".to_owned(),
            price: "1.25".to_owned(),
            change: "+2% 1m".to_owned(),
            currency: String::new(),
            exchange: String::new(),
            asset_type: String::new(),
            contracts: Vec::new(),
        };
        let mut signal = token_signal();
        signal.pair.base_token.symbol = "RKH".to_owned();
        signal.pair.base_token.name = "Roaring Kity Hacked".to_owned();
        let mut dispatcher = dispatcher()
            .with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: Some(bot_core::market_prices::MarketChart {
                        timeframe: None,
                        symbol: "RKH".to_owned(),
                        name: "Native RKH".to_owned(),
                        yahoo_symbol: String::new(),
                        token: None,
                        candidate: Some(market_candidate),
                    }),
                    selection: None,
                    no_assets_found: false,
                    text: "RKH: 1.25 USD (+2% 1m)".to_owned(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "RKH: 1.25 USD (+2% 1m)".to_owned(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::new(RefCell::new(HashMap::new())),
                selected: Rc::new(RefCell::new(Vec::new())),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: Some(signal),
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Ok(vec![1]),
                state: None,
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Default::default(),
            }));

        assert_eq!(
            dispatcher.dispatch(update("/c rkh 1m", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let message = first_sent(&dispatcher.actions.0);
        assert!(
            message
                .reply_markup
                .as_ref()
                .into_iter()
                .flat_map(|markup| markup.inline_keyboard.iter())
                .flatten()
                .any(|b| b.text.contains("Native RKH"))
        );
        assert!(
            message
                .reply_markup
                .as_ref()
                .into_iter()
                .flat_map(|markup| markup.inline_keyboard.iter())
                .flatten()
                .any(|b| b.text.contains("Roaring Kity Hacked"))
        );
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(3)
        );
    }

    #[test]
    fn provider_symbol_does_not_hide_a_distinct_dex_identity() {
        let mut official = token_signal();
        official.token = TokenAddress {
            chain_id: "base".to_owned(),
            network: "base".to_owned(),
            tag: "BASE".to_owned(),
            address: "0x0000000000000000000000000000000000000001".to_owned(),
        };
        official.pair.chain_id = "base".to_owned();
        official.pair.base_token.address = official.token.address.clone();
        official.pair.base_token.name = "Official Laptop".to_owned();
        official.pair.base_token.symbol = "LAPTOP".to_owned();

        let mut provider = official.clone();
        provider.token.address = "0x0000000000000000000000000000000000000002".to_owned();
        provider.pair.base_token.address = provider.token.address.clone();
        provider.pair.base_token.name = "Provider Laptop".to_owned();

        let provider_candidate = bot_core::market_prices::MarketCandidate {
            id: "provider:laptop".to_owned(),
            symbol: "LAPTOP".to_owned(),
            name: "Provider Laptop".to_owned(),
            slug: "laptop".to_owned(),
            price: "1".to_owned(),
            change: "N/A 24h".to_owned(),
            currency: String::new(),
            exchange: String::new(),
            asset_type: String::new(),
            contracts: vec![provider.token.clone()],
        };
        let mut dispatcher = dispatcher()
            .with_market_price_source(Box::new(CandidateChartPrices {
                initial: MarketPriceLoad {
                    chart: Some(bot_core::market_prices::MarketChart {
                        timeframe: None,
                        symbol: "LAPTOP".to_owned(),
                        name: "Provider Laptop".to_owned(),
                        yahoo_symbol: String::new(),
                        token: Some(provider.token.clone()),
                        candidate: Some(provider_candidate),
                    }),
                    selection: None,
                    no_assets_found: false,
                    text: "LAPTOP: 1 USD".to_owned(),
                    diagnostics: Vec::new(),
                },
                rendered: Rc::new(RefCell::new(Vec::new())),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: Some(official),
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: Some(provider),
                    diagnostics: Vec::new(),
                },
                photo: Ok(vec![1]),
                state: None,
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Rc::new(RefCell::new(Vec::new())),
            }));

        assert_eq!(
            dispatcher.dispatch(update("/p laptop", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let message = first_sent(&dispatcher.actions.0);
        assert!(message.text.contains("Official Laptop"));
        assert!(message.text.contains("Provider Laptop"));
        assert!(message.text.contains("base:BASE 0x000000…000001"));
        assert!(message.text.contains("base:BASE 0x000000…000002"));
    }

    #[test]
    fn token_market_selection_callbacks_cover_missing_data_and_fallbacks() {
        let token = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
        };
        let token_candidate = bot_core::market_prices::MarketCandidate {
            id: "token:solana:solana:J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            symbol: "SYN".to_owned(),
            name: "Synthetic Token".to_owned(),
            slug: "syn".to_owned(),
            price: "0.01".to_owned(),
            change: "N/A 1m".to_owned(),
            currency: String::new(),
            exchange: String::new(),
            asset_type: String::new(),
            contracts: vec![token],
        };
        let selection_id = market_selection_id(-42, 7, 88, 1_672_531_200, 0);
        let selection_key = market_selection_key(&selection_id);
        let stored_value = |candidate: bot_core::market_prices::MarketCandidate,
                            timeframe: Option<&str>,
                            chat_id: &str,
                            command: &str| {
            serde_json::to_string(&StoredMarketSelection {
                selection: MarketSelection {
                    query: "syn".to_owned(),
                    timeframe: timeframe.map(ToOwned::to_owned),
                    target_symbol: "USD".to_owned(),
                    target_parameter: "USD".to_owned(),
                    conversion: None,
                    candidates: vec![candidate],
                },
                chat_id: chat_id.to_owned(),
                message_id: 700,
                source_message_id: Some(6),
                requester_id: 88,
                command: command.to_owned(),
            })
            .ok()
        };

        let missing_stored = Rc::new(RefCell::new(HashMap::from([(
            selection_key.clone(),
            stored_value(token_candidate.clone(), Some("1m"), "-42", "unified").unwrap_or_default(),
        )])));
        let mut missing = dispatcher()
            .with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&missing_stored),
                selected: Rc::new(RefCell::new(Vec::new())),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Ok(vec![1]),
                state: None,
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Default::default(),
            }));
        assert_eq!(
            missing.dispatch(callback_update_for_message(
                &format!("mkt:select:{selection_id}:0"),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(missing.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message)
                if message.reply_to_message_id == Some(MessageId(6))
                    && message.text.contains("get a quote")
        )));

        let mut no_chart_signal = token_signal();
        no_chart_signal.candles.clear();
        let fallback_stored = Rc::new(RefCell::new(HashMap::from([(
            selection_key.clone(),
            stored_value(token_candidate.clone(), None, "-42", "crypto").unwrap_or_default(),
        )])));
        let mut fallback = dispatcher()
            .with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&fallback_stored),
                selected: Rc::new(RefCell::new(Vec::new())),
            }))
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: Some(no_chart_signal),
                    diagnostics: Vec::new(),
                },
                photo: Err("history unavailable".to_owned()),
                state: None,
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Default::default(),
            }));
        assert_eq!(
            fallback.dispatch(callback_update_for_message(
                &format!("mkt:select:{selection_id}:0"),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(fallback.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message)
                if message.reply_to_message_id == Some(MessageId(6))
                    && message.parse_mode == Some(bot_core::telegram_actions::ParseMode::Html)
        )));

        let mut no_contract_candidate = token_candidate.clone();
        no_contract_candidate.contracts.clear();
        let no_contract_stored = Rc::new(RefCell::new(HashMap::from([(
            selection_key.clone(),
            stored_value(no_contract_candidate, Some("1m"), "-42", "unified").unwrap_or_default(),
        )])));
        let mut no_contract =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&no_contract_stored),
                selected: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            no_contract.dispatch(callback_update_for_message(
                &format!("mkt:select:{selection_id}:0"),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(no_contract.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::AnswerCallback {
                show_alert: true,
                ..
            }
        )));

        let invalid_chat_stored = Rc::new(RefCell::new(HashMap::from([(
            selection_key.clone(),
            stored_value(token_candidate.clone(), Some("1m"), "invalid", "unified")
                .unwrap_or_default(),
        )])));
        let mut invalid_chat =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                candidate: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: true,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                stored: Rc::clone(&invalid_chat_stored),
                selected: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            invalid_chat.dispatch(callback_update_with_context(
                &format!("mkt:select:{selection_id}:0"),
                json!("invalid"),
                "private",
                700,
                Some(88),
                Some("en"),
                Some("callback-invalid-chat"),
            )),
            Ok(DispatchOutcome::Handled)
        );

        let photo_failure_stored = Rc::new(RefCell::new(HashMap::from([(
            selection_key,
            stored_value(token_candidate, Some("1m"), "-42", "unified").unwrap_or_default(),
        )])));
        let mut photo_failure = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            attempt_actions(Attempt::Fail, ActionScript::photo),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: true,
                text: String::new(),
                diagnostics: Vec::new(),
            },
            candidate: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: true,
                text: String::new(),
                diagnostics: Vec::new(),
            },
            stored: Rc::clone(&photo_failure_stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }))
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: Some(token_signal()),
                diagnostics: Vec::new(),
            },
            photo: Ok(vec![1]),
            state: None,
            queries: Rc::new(RefCell::new(Vec::new())),
            saved: Default::default(),
        }));
        assert_eq!(
            photo_failure.dispatch(callback_update_for_message(
                &format!("mkt:select:{selection_id}:0"),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(photo_failure.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::SendMessage(message)
                if message.parse_mode == Some(bot_core::telegram_actions::ParseMode::Html)
        )));
    }

    #[test]
    fn explicit_ranges_reach_the_chart_instead_of_becoming_assets() {
        for (command, asset, symbol) in [
            ("/c", "bitcoin", "BTC-USD"),
            ("/p", "apple", "AAPL"),
            ("/s", "apple", "AAPL"),
        ] {
            for period in ["1m", "2h", "7d", "1w", "1mo", "1y", "5y"] {
                let charts = Rc::new(RefCell::new(Vec::new()));
                let mut dispatcher = dispatcher().with_market_price_source(Box::new(ChartPrices {
                    charts: Rc::clone(&charts),
                    fail_chart: false,
                }));
                assert_eq!(
                    dispatcher.dispatch(update(&format!("{command} {asset} {period}"), Some("en"))),
                    Ok(DispatchOutcome::Handled)
                );
                assert_eq!(*charts.borrow(), vec![format!("{symbol}:{period}")]);
                assert!(matches!(
                    dispatcher.actions.0.as_slice(),
                    [TelegramAction::SendPhoto { caption, .. }]
                        if caption.ends_with(&format!("(+5.00% {period})"))
                ));
                assert!(
                    dispatcher.state.outgoing[0]
                        .message
                        .text
                        .ends_with(&format!("(+5.00% {period})"))
                );
            }
        }
    }

    #[test]
    fn mixed_explicit_ranges_reach_each_market_request() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = dispatcher().with_market_price_source(Box::new(MarketPrices {
            result: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: false,
                text: "synthetic quote".to_owned(),
                diagnostics: Vec::new(),
            },
            calls: Rc::clone(&calls),
        }));

        assert_eq!(
            dispatcher.dispatch(update("/p btc,eth 1m", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            calls
                .borrow()
                .iter()
                .map(|call| call.0.as_str())
                .collect::<Vec<_>>(),
            ["btc 1m", "eth 1m"]
        );
    }

    #[test]
    fn single_explicit_range_reaches_fallback_quote_with_requested_period() {
        struct PeriodAwareMarketPrices {
            calls: Rc<RefCell<Vec<String>>>,
        }

        impl MarketPriceSource for PeriodAwareMarketPrices {
            fn load(
                &mut self,
                query: &str,
                _: bot_core::market_prices::MarketPriceCommand,
                _: bot_core::locale::Locale,
                _: i64,
            ) -> MarketPriceLoad {
                self.calls.borrow_mut().push(query.to_owned());
                let period = query.split_whitespace().last().unwrap_or("24h");
                MarketPriceLoad {
                    chart: Some(bot_core::market_prices::MarketChart {
                        timeframe: None,
                        symbol: "BTC".to_owned(),
                        name: "Bitcoin".to_owned(),
                        yahoo_symbol: "BTC-USD".to_owned(),
                        token: None,
                        candidate: None,
                    }),
                    selection: None,
                    no_assets_found: false,
                    text: format!("BTC: 1 USD (+1.00% {period})"),
                    diagnostics: Vec::new(),
                }
            }
        }

        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(PeriodAwareMarketPrices {
                calls: Rc::clone(&calls),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p btc 1m", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), ["btc 1m"]);
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendMessage(message)]
                if message.text.contains("BTC: 1 USD (+1.00% 1m)")
                    && message.text.contains("Chart unavailable")
        ));
    }

    #[test]
    fn single_explicit_range_formats_selection_candidates_with_requested_period() {
        struct PeriodAwareSelection {
            calls: Rc<RefCell<Vec<String>>>,
        }

        impl MarketPriceSource for PeriodAwareSelection {
            fn load(
                &mut self,
                query: &str,
                _: bot_core::market_prices::MarketPriceCommand,
                _: bot_core::locale::Locale,
                _: i64,
            ) -> MarketPriceLoad {
                self.calls.borrow_mut().push(query.to_owned());
                let period = query.split_whitespace().last().unwrap_or("24h");
                MarketPriceLoad {
                    chart: None,
                    selection: Some(bot_core::market_prices::MarketSelection {
                        query: "libra".to_owned(),
                        timeframe: None,
                        target_symbol: "USD".to_owned(),
                        target_parameter: "USD".to_owned(),
                        conversion: None,
                        candidates: vec![bot_core::market_prices::MarketCandidate {
                            id: "1".to_owned(),
                            symbol: "LIBRA".to_owned(),
                            name: "Libra Finance".to_owned(),
                            slug: "libra-finance".to_owned(),
                            price: "0.007".to_owned(),
                            change: format!("N/A {period}"),
                            currency: String::new(),
                            exchange: String::new(),
                            asset_type: String::new(),
                            contracts: Vec::new(),
                        }],
                    }),
                    no_assets_found: false,
                    text: String::new(),
                    diagnostics: Vec::new(),
                }
            }
        }

        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher =
            dispatcher().with_market_price_source(Box::new(PeriodAwareSelection {
                calls: Rc::clone(&calls),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra 1m", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(calls.borrow().as_slice(), ["libra 1m"]);
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendMessage(message)]
                if message.text.contains("N/A 1m") && !message.text.contains("N/A 24h")
        ));
    }

    #[test]
    fn stock_chart_delivery_records_the_quote() {
        struct StockChart;
        impl StockPriceSource for StockChart {
            fn load(&mut self, _: &str, _: i64) -> StockQuotesLoad {
                StockQuotesLoad {
                    quotes: Some(vec![(
                        "AAPL".into(),
                        Some(StockQuote {
                            symbol: "AAPL".into(),
                            name: "Apple".into(),
                            price: 100.0,
                            currency: "USD".into(),
                            exchange: "NMS".into(),
                            asset_type: String::new(),
                            variation: 1.0,
                        }),
                    )]),
                    diagnostics: vec![],
                }
            }
            fn render_chart(&mut self, _: &StockQuote, _: i64) -> Result<Vec<u8>, String> {
                Ok(vec![1, 2, 3])
            }
        }
        let mut dispatcher = dispatcher().with_stock_price_source(Box::new(StockChart));
        assert_eq!(
            dispatcher.dispatch(update("/s apple", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            matches!(dispatcher.actions.0.as_slice(), [TelegramAction::SendPhoto { caption, .. }] if caption.contains("AAPL"))
        );
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn undelivered_market_photos_fall_back_to_recorded_text() {
        for outcome in [Attempt::Skip, Attempt::Unconfirmed] {
            let mut dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig::default()),
                    chat_ids: vec![],
                },
                photo_actions(outcome),
                State::default(),
                values(),
                random(),
                authorization(),
                "@mybot",
            )
            .with_market_price_source(Box::new(ChartPrices {
                charts: Default::default(),
                fail_chart: false,
            }));
            assert_eq!(
                dispatcher.dispatch(update("/p apple", Some("es"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(dispatcher.state.incoming.len(), 1);
            assert_eq!(dispatcher.state.outgoing.len(), 1);
            assert!(
                dispatcher
                    .state_diagnostics()
                    .iter()
                    .any(|line| line.contains("market chart unavailable"))
            );
        }
    }

    #[test]
    fn unavailable_market_charts_preserve_quotes_and_explain_the_fallback() {
        let mut dispatcher = dispatcher().with_market_price_source(Box::new(ChartPrices {
            charts: Default::default(),
            fail_chart: true,
        }));
        assert_eq!(
            dispatcher.dispatch(update("/p apple", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            matches!(dispatcher.actions.0.as_slice(), [TelegramAction::SendMessage(reply)] if reply.text.contains("AAPL:") && reply.text.contains("Chart unavailable"))
        );
    }

    #[test]
    fn token_commands_send_cards_for_mints_symbols_cashtags_and_aliases() {
        let mint = "F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump";
        for command in [
            "/c", "/cripto", "/criptos", "/crypto", "/cryptos", "/c@mybot", "/p", "/price",
            "/prices", "/precios", "/p@mybot",
        ] {
            for argument in [mint, "timba", "$timba"] {
                let mut signal = token_signal();
                signal.token.address = mint.to_owned();
                signal.pair.base_token.address = mint.to_owned();
                signal.pair.base_token.name = "TIMBA".to_owned();
                signal.pair.base_token.symbol = "TIMBA".to_owned();
                let queries = Rc::new(RefCell::new(Vec::new()));
                let market_calls = Rc::new(RefCell::new(Vec::new()));
                let saved = Rc::new(RefCell::new(Vec::new()));
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
                .with_token_signal_source(Box::new(Signals {
                    query_load: TokenSignalLoad {
                        signal: Some(signal),
                        diagnostics: Vec::new(),
                    },
                    token_load: TokenSignalLoad {
                        signal: None,
                        diagnostics: Vec::new(),
                    },
                    photo: Ok(b"png".to_vec()),
                    state: None,
                    queries: Rc::clone(&queries),
                    saved: Rc::clone(&saved),
                }))
                .with_market_price_source(Box::new(MarketPrices {
                    result: MarketPriceLoad {
                        chart: None,
                        selection: None,
                        no_assets_found: true,
                        text: "missing".to_owned(),
                        diagnostics: Vec::new(),
                    },
                    calls: Rc::clone(&market_calls),
                }));
                assert_eq!(
                    dispatcher.dispatch(update(&format!("{command} {argument}"), Some("es"))),
                    Ok(DispatchOutcome::Handled)
                );
                let expected = if argument == mint {
                    SignalQuery::Address(TokenAddress {
                        address: mint.to_owned(),
                        chain_id: "solana".to_owned(),
                        network: "solana".to_owned(),
                        tag: "SOL".to_owned(),
                    })
                } else {
                    SignalQuery::Symbol("timba".to_owned())
                };
                assert_eq!(queries.borrow().as_slice(), &[expected]);
                assert_eq!(market_calls.borrow().len(), usize::from(argument != mint));
                assert!(matches!(dispatcher.actions.0.as_slice(),
                    [TelegramAction::SendPhoto { caption, reply_to_message_id: Some(MessageId(7)), .. }]
                    if caption.contains("TIMBA")));
                assert_eq!(saved.borrow().len(), 1);
                assert_eq!(saved.borrow()[0].1.address, mint);
            }
        }
    }

    #[test]
    fn token_command_fallback_preserves_market_scope_and_complex_queries() {
        use bot_core::market_prices::MarketPriceCommand;
        for (text, argument, scope, token_lookup) in [
            ("/p timba", "timba", MarketPriceCommand::Unified, true),
            ("/p BTC", "BTC", MarketPriceCommand::Unified, false),
            ("/p AAPL", "AAPL", MarketPriceCommand::Unified, false),
            ("/c timba", "timba", MarketPriceCommand::CryptoOnly, true),
            ("/c $timba", "$timba", MarketPriceCommand::CryptoOnly, true),
            (
                "/c btc eth",
                "btc eth",
                MarketPriceCommand::CryptoOnly,
                false,
            ),
            ("/c btc 7d", "btc 7d", MarketPriceCommand::CryptoOnly, false),
            (
                "/c 2 btc to usd",
                "2 btc to usd",
                MarketPriceCommand::CryptoOnly,
                false,
            ),
            ("/prices AAPL", "AAPL", MarketPriceCommand::Unified, false),
            ("/price $timba", "$timba", MarketPriceCommand::Unified, true),
            (
                "/p 0x26449b21EaF982D252956e34E675634b8b15f990",
                "0x26449b21EaF982D252956e34E675634b8b15f990",
                MarketPriceCommand::Unified,
                true,
            ),
            ("/c", "", MarketPriceCommand::CryptoOnly, false),
        ] {
            let calls = Rc::new(RefCell::new(Vec::new()));
            let queries = Rc::new(RefCell::new(Vec::new()));
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
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Err("unused".to_owned()),
                state: None,
                queries: Rc::clone(&queries),
                saved: Rc::new(RefCell::new(Vec::new())),
            }))
            .with_market_price_source(Box::new(MarketPrices {
                result: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: token_lookup,
                    text: "market fallback".to_owned(),
                    diagnostics: Vec::new(),
                },
                calls: Rc::clone(&calls),
            }));
            assert_eq!(
                dispatcher.dispatch(update(text, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(queries.borrow().len(), usize::from(token_lookup));
            if argument.starts_with("0x") {
                assert!(calls.borrow().is_empty());
            } else {
                assert_eq!(calls.borrow().len(), 1);
                assert_eq!(calls.borrow()[0].0, argument);
                assert_eq!(calls.borrow()[0].1, scope);
            }
            assert!(matches!(dispatcher.actions.0.as_slice(),
                [TelegramAction::SendMessage(message)] if message.text == "market fallback" || argument.starts_with("0x")));
        }
    }

    #[test]
    fn unresolved_cashtag_uses_the_existing_unified_market_fallback() {
        let queries = Rc::new(RefCell::new(Vec::new()));
        let saved = Rc::new(RefCell::new(Vec::new()));
        let market_calls = Rc::new(RefCell::new(Vec::new()));
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
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            photo: Err("unused".to_owned()),
            state: None,
            queries,
            saved,
        }))
        .with_market_price_source(Box::new(MarketPrices {
            result: MarketPriceLoad {
                chart: None,
                selection: None,
                no_assets_found: false,
                text: "NVDA: 123.45 USD (+1.25% 24h)".to_owned(),
                diagnostics: Vec::new(),
            },
            calls: Rc::clone(&market_calls),
        }));

        assert_eq!(
            dispatcher.dispatch(update("$NVDA", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            market_calls.borrow().as_slice(),
            [(
                query,
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                _
            )]
                if query == "$NVDA"
        ));
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendMessage(message)]
                if message.text == "NVDA: 123.45 USD (+1.25% 24h)"
                    && message.reply_to_message_id == Some(MessageId(7))
        ));
    }

    #[test]
    fn token_signal_refresh_edits_photo_updates_cooldown_state_and_answers() {
        let signal = token_signal();
        let saved = Rc::new(RefCell::new(Vec::new()));
        let state = SignalState {
            chart_period: None,
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: 6,
            requester_id: "88".to_owned(),
            chain_id: signal.token.chain_id.clone(),
            network: signal.token.network.clone(),
            tag: signal.token.tag.clone(),
            address: signal.token.address.clone(),
            last_refresh_at: None,
        };
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
        .with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: Some(signal),
                diagnostics: vec!["refresh diagnostic".to_owned()],
            },
            photo: Ok(b"refreshed-png".to_vec()),
            state: Some(state),
            queries: Rc::new(RefCell::new(Vec::new())),
            saved: Rc::clone(&saved),
        }));

        assert_eq!(
            dispatcher.dispatch(callback_update("sig:ref:abc", "group", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::EditMessagePhoto { photo, message_id: MessageId(7), .. },
                TelegramAction::AnswerCallback { text: Some(text), show_alert: false, .. },
            ] if photo.as_ref() == b"refreshed-png" && text == "Tarjeta actualizada"
        ));
        let saved = saved.borrow();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].0, "abc");
        assert_eq!(saved[0].1.last_refresh_at, Some(1_672_531_200));
        assert_eq!(dispatcher.state_diagnostics(), ["refresh diagnostic"]);
    }

    #[test]
    fn token_signal_callback_enforces_owner_and_refresh_cooldown() {
        let signal = token_signal();
        let base_state = SignalState {
            chart_period: None,
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: 6,
            requester_id: "7".to_owned(),
            chain_id: signal.token.chain_id.clone(),
            network: signal.token.network.clone(),
            tag: signal.token.tag.clone(),
            address: signal.token.address.clone(),
            last_refresh_at: None,
        };
        let build = |state: SignalState, authorization: Authorization| {
            NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig::default()),
                    chat_ids: Vec::new(),
                },
                Actions::default(),
                State::default(),
                values(),
                random(),
                authorization,
                "@mybot",
            )
            .with_token_signal_source(Box::new(Signals {
                query_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: Some(signal.clone()),
                    diagnostics: Vec::new(),
                },
                photo: Ok(b"unused".to_vec()),
                state: Some(state),
                queries: Rc::new(RefCell::new(Vec::new())),
                saved: Rc::new(RefCell::new(Vec::new())),
            }))
        };
        let mut denied_authorization = authorization();
        denied_authorization.is_admin = false;
        let mut denied = build(base_state.clone(), denied_authorization);
        assert_eq!(
            denied.dispatch(callback_update("sig:del:abc", "group", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            denied.actions.0.as_slice(),
            [TelegramAction::AnswerCallback { text: Some(text), show_alert: true, .. }]
                if text == "Solo quien pidió la tarjeta o un admin puede hacer eso"
        ));

        let mut cooldown_state = base_state;
        cooldown_state.requester_id = "88".to_owned();
        cooldown_state.last_refresh_at = Some(1_672_531_195);
        let mut cooldown = build(cooldown_state, authorization());
        assert_eq!(
            cooldown.dispatch(callback_update("sig:ref:abc", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            cooldown.actions.0.as_slice(),
            [TelegramAction::AnswerCallback { text: Some(text), show_alert: true, .. }]
                if text == "Podés actualizar cada 15s"
        ));
    }

    #[test]
    fn token_signal_callbacks_handle_missing_expired_delete_and_unknown_state() {
        let mut missing = dispatcher();
        assert_eq!(
            missing.dispatch(callback_update("sig:ref:abc", "private", Some("en"))),
            Err(DispatchError::MissingService("token signals"))
        );

        for state in [Err("synthetic state failure".to_owned()), Ok(None)] {
            let mut expired = dispatcher().with_token_signal_source(Box::new(FallibleSignals {
                query_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Err("unused".to_owned()),
                state,
                save_error: None,
            }));
            assert_eq!(
                expired.dispatch(callback_update("sig:ref:abc", "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert!(matches!(
                expired.actions.0.as_slice(),
                [TelegramAction::AnswerCallback { text: Some(text), show_alert: true, .. }]
                    if !text.is_empty()
            ));
        }

        let signal = token_signal();
        let state = SignalState {
            chart_period: None,
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: 6,
            requester_id: "88".to_owned(),
            chain_id: signal.token.chain_id.clone(),
            network: signal.token.network.clone(),
            tag: signal.token.tag.clone(),
            address: signal.token.address.clone(),
            last_refresh_at: None,
        };
        let source = |state: SignalState| FallibleSignals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: Some(signal.clone()),
                diagnostics: Vec::new(),
            },
            photo: Ok(vec![1]),
            state: Ok(Some(state)),
            save_error: None,
        };
        let mut deleted = dispatcher().with_token_signal_source(Box::new(source(state.clone())));
        assert_eq!(
            deleted.dispatch(callback_update("sig:del:abc", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            deleted.actions.0.as_slice(),
            [TelegramAction::DeleteMessage { .. }, TelegramAction::AnswerCallback { text: Some(text), show_alert: false, .. }]
                if !text.is_empty()
        ));

        let mut unknown = dispatcher().with_token_signal_source(Box::new(source(state)));
        assert_eq!(
            unknown.dispatch(callback_update("sig:wat:abc", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            unknown.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: None,
                show_alert: false,
                ..
            }]
        ));
    }

    #[test]
    fn token_signal_refresh_reports_no_data_render_and_state_write_failures() {
        let signal = token_signal();
        let state = SignalState {
            chart_period: None,
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: 6,
            requester_id: "88".to_owned(),
            chain_id: signal.token.chain_id.clone(),
            network: signal.token.network.clone(),
            tag: signal.token.tag.clone(),
            address: signal.token.address.clone(),
            last_refresh_at: None,
        };
        let source = |token_load: TokenSignalLoad,
                      photo: Result<Vec<u8>, String>,
                      save_error: Option<String>| FallibleSignals {
            query_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            token_load,
            photo,
            state: Ok(Some(state.clone())),
            save_error,
        };

        let mut no_data = dispatcher().with_token_signal_source(Box::new(source(
            TokenSignalLoad {
                signal: None,
                diagnostics: vec!["synthetic refresh diagnostic".to_owned()],
            },
            Err("unused".to_owned()),
            None,
        )));
        assert_eq!(
            no_data.dispatch(callback_update("sig:ref:abc", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            no_data.actions.0.as_slice(),
            [TelegramAction::AnswerCallback { text: Some(text), show_alert: true, .. }]
                if !text.is_empty()
        ));

        let loaded = TokenSignalLoad {
            signal: Some(signal.clone()),
            diagnostics: Vec::new(),
        };
        let mut render_failed = dispatcher().with_token_signal_source(Box::new(source(
            loaded.clone(),
            Err("synthetic render failure".to_owned()),
            None,
        )));
        assert_eq!(
            render_failed.dispatch(callback_update("sig:ref:abc", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            render_failed.actions.0.as_slice(),
            [TelegramAction::AnswerCallback { text: Some(text), show_alert: true, .. }]
                if !text.is_empty()
        ));

        let mut save_failed = dispatcher().with_token_signal_source(Box::new(source(
            loaded,
            Ok(vec![1, 2, 3]),
            Some("synthetic save failure".to_owned()),
        )));
        assert_eq!(
            save_failed.dispatch(callback_update("sig:ref:abc", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            save_failed
                .state_diagnostics()
                .iter()
                .any(|entry| entry.contains("refresh state write failed"))
        );
    }

    #[test]
    fn token_range_is_kept_for_refresh_without_caption_annotation() {
        let saved = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = dispatcher().with_token_signal_source(Box::new(Signals {
            query_load: TokenSignalLoad {
                signal: Some(token_signal()),
                diagnostics: vec![],
            },
            token_load: TokenSignalLoad {
                signal: None,
                diagnostics: vec![],
            },
            photo: Ok(vec![1]),
            state: None,
            queries: Default::default(),
            saved: Rc::clone(&saved),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/c timba 1h", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            matches!(dispatcher.actions.0.as_slice(), [TelegramAction::SendPhoto { caption, .. }] if !caption.contains("Chart range:"))
        );
        assert_eq!(saved.borrow()[0].1.chart_period.as_deref(), Some("1h"));
    }

    #[test]
    fn token_signal_message_failures_reply_with_available_data_and_diagnostics() {
        let address = "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump";
        let (ai, _observations) = ai_source(Ok(AiPreparation::silent()));
        let mut missing = dispatcher()
            .with_token_signal_source(Box::new(FallibleSignals {
                query_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: vec!["synthetic query miss".to_owned()],
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Err("unused".to_owned()),
                state: Ok(None),
                save_error: None,
            }))
            .with_ai_conversation_source(Box::new(ai));
        assert_eq!(
            missing.dispatch(update(address, Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            missing
                .state_diagnostics()
                .iter()
                .any(|entry| entry.contains("synthetic query miss"))
        );

        let signal = token_signal();
        let (ai, _observations) = ai_source(Ok(AiPreparation::silent()));
        let mut render_failed = dispatcher()
            .with_token_signal_source(Box::new(FallibleSignals {
                query_load: TokenSignalLoad {
                    signal: Some(signal.clone()),
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Err("synthetic initial render failure".to_owned()),
                state: Ok(None),
                save_error: None,
            }))
            .with_ai_conversation_source(Box::new(ai));
        assert_eq!(
            render_failed.dispatch(update(address, Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            render_failed
                .state_diagnostics()
                .iter()
                .any(|entry| entry.contains("token signal photo failed"))
        );

        assert!(
            matches!(render_failed.actions.0.as_slice(), [TelegramAction::SendMessage(reply)] if reply.text.contains("Synthetic Token"))
        );
        assert_eq!(
            render_failed.dispatch(update(&format!("/c {address} 1m"), Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            matches!(render_failed.actions.0.last(), Some(TelegramAction::SendMessage(reply)) if !reply.text.contains("Chart range:") && reply.text.contains("Synthetic Token"))
        );
        let mut save_failed = dispatcher().with_token_signal_source(Box::new(FallibleSignals {
            query_load: TokenSignalLoad {
                signal: Some(signal),
                diagnostics: Vec::new(),
            },
            token_load: TokenSignalLoad {
                signal: None,
                diagnostics: Vec::new(),
            },
            photo: Ok(vec![1, 2, 3]),
            state: Ok(None),
            save_error: Some("synthetic initial state failure".to_owned()),
        }));
        assert_eq!(
            save_failed.dispatch(update(address, Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            save_failed
                .state_diagnostics()
                .iter()
                .any(|entry| entry.contains("state write failed"))
        );

        for outcome in [Attempt::Skip, Attempt::Unconfirmed] {
            let mut delivery_failed = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig::default()),
                    chat_ids: Vec::new(),
                },
                photo_actions(outcome),
                State::default(),
                values(),
                random(),
                authorization(),
                "@mybot",
            )
            .with_token_signal_source(Box::new(FallibleSignals {
                query_load: TokenSignalLoad {
                    signal: Some(token_signal()),
                    diagnostics: Vec::new(),
                },
                token_load: TokenSignalLoad {
                    signal: None,
                    diagnostics: Vec::new(),
                },
                photo: Ok(vec![1]),
                state: Ok(None),
                save_error: None,
            }));
            assert_eq!(
                delivery_failed.dispatch(update(address, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert!(
                delivery_failed
                    .state_diagnostics()
                    .iter()
                    .any(|entry| entry.contains("token signal photo delivery"))
            );
        }
    }

    #[test]
    fn task_navigation_and_close_never_cancel_a_task() -> Result<(), TaskStateError> {
        let cancellations = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = dispatcher().with_scheduled_task_source(Box::new(Tasks {
            lists: vec![vec![scheduled_task(88)?]],
            cancellations: Rc::clone(&cancellations),
        }));
        for data in [
            "task:page:0",
            "task:page:999",
            "task:view:task0001",
            "task:ask:task0001",
            "task:close",
            "task:unknown:task0001",
        ] {
            assert_eq!(
                dispatcher.dispatch(callback_update(data, "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert!(cancellations.borrow().is_empty());
        }
        assert!(dispatcher.actions.0.iter().any(|a| matches!(a, TelegramAction::EditMessage { text, .. } if text.contains("cancelar esta tarea") || text.contains("cancel this task"))));
        assert!(
            dispatcher
                .actions
                .0
                .iter()
                .any(|a| matches!(a, TelegramAction::DeleteMessage { .. }))
        );
        Ok(())
    }

    #[test]
    fn credit_menu_close_does_not_create_invoices_or_load_history() {
        for (data, chat_type, allowed) in [
            ("topup:close", "private", true),
            ("topup:close", "group", false),
            ("chg:close:88", "private", true),
            ("chg:close:99", "private", false),
            ("chg:close:invalid", "private", false),
        ] {
            let mut dispatcher = dispatcher();
            assert_eq!(
                dispatcher.dispatch(callback_update(data, chat_type, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher
                    .actions
                    .0
                    .iter()
                    .any(|a| matches!(a, TelegramAction::DeleteMessage { .. })),
                allowed
            );
            assert!(
                !dispatcher
                    .actions
                    .0
                    .iter()
                    .any(|a| matches!(a, TelegramAction::SendMessage(_)))
            );
        }
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
            ] if text == "Tarea task0001 cancelada" && edit_text.starts_with("No hay tareas")
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
            }] if text == "Solo quien la creó o un admin puede cancelar esta tarea"
        ));
        Ok(())
    }

    #[test]
    fn task_callback_reports_list_failure_instead_of_not_found() {
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
        .with_scheduled_task_source(Box::new(FallibleTasks {
            list_result: Err("synthetic list failure".to_owned()),
            cancel_result: Ok(true),
        }));

        assert_eq!(
            dispatcher.dispatch(callback_update("task:del:task0001", "private", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }] if text == "No pude leer las tareas. Probá de nuevo"
        ));
        assert!(dispatcher.state_diagnostics()[0].contains("synthetic list failure"));
    }

    #[test]
    fn task_callback_never_claims_failed_cancellation() -> Result<(), TaskStateError> {
        for (cancel_result, expected, has_diagnostic) in [
            (Ok(false), "Esa tarea ya no existe", false),
            (
                Err("synthetic cancellation failure".to_owned()),
                "No pude cancelar la tarea. Probá de nuevo",
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
            .with_scheduled_task_source(Box::new(FallibleTasks {
                list_result: Ok(vec![scheduled_task(88)?]),
                cancel_result,
            }));

            assert_eq!(
                dispatcher.dispatch(callback_update("task:del:task0001", "group", None)),
                Ok(DispatchOutcome::Handled)
            );
            assert!(matches!(
                dispatcher.actions.0.as_slice(),
                [TelegramAction::AnswerCallback {
                    text: Some(text),
                    show_alert: true,
                    ..
                }] if text == expected
            ));
            assert_eq!(!dispatcher.state_diagnostics().is_empty(), has_diagnostic);
        }
        Ok(())
    }

    #[test]
    fn task_callback_handles_missing_tasks_and_post_delete_refresh_failures()
    -> Result<(), TaskStateError> {
        let mut missing = dispatcher().with_scheduled_task_source(Box::new(FallibleTasks {
            list_result: Ok(Vec::new()),
            cancel_result: Ok(true),
        }));
        assert_eq!(
            missing.dispatch(callback_update("task:del:task0001", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            missing.actions.0.as_slice(),
            [TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }] if text == "That task no longer exists"
        ));

        let mut refresh_failed =
            dispatcher().with_scheduled_task_source(Box::new(SequencedTasks {
                lists: VecDeque::from([
                    Ok(vec![scheduled_task(88)?]),
                    Err("synthetic refresh failure".to_owned()),
                ]),
            }));
        assert_eq!(
            refresh_failed.dispatch(callback_update("task:del:task0001", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            refresh_failed
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("synthetic refresh failure"))
        );
        Ok(())
    }

    #[test]
    fn task_callback_tolerates_non_numeric_chat_ids_and_edit_rejection()
    -> Result<(), TaskStateError> {
        let callback_with_chat = |chat_id: Value| {
            callback_update_with_context(
                "task:del:task0001",
                chat_id,
                "private",
                7,
                Some(88),
                Some("en"),
                Some("callback-1"),
            )
        };

        let mut invalid_chat = dispatcher().with_scheduled_task_source(Box::new(Tasks {
            lists: vec![vec![scheduled_task(88)?], Vec::new()],
            cancellations: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            invalid_chat.dispatch(callback_with_chat(json!("synthetic-chat"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(invalid_chat.actions.0.len(), 1);

        let mut rejected = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            attempt_actions(Attempt::Fail, ActionScript::edit),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_scheduled_task_source(Box::new(Tasks {
            lists: vec![vec![scheduled_task(88)?], Vec::new()],
            cancellations: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            rejected.dispatch(callback_with_chat(json!(-42))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            rejected
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("scheduled task edit failed"))
        );
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
        assert!(matches!(
            dispatcher.actions.0.first(),
            Some(TelegramAction::EditMessage {
                text,
                reply_markup: Some(keyboard),
                ..
            }) if text == "Gastos de IA\n\n26/08 14:00 | respuesta: 0.04 cr"
                && keyboard.inline_keyboard[0][0].callback_data.as_deref()
                    == Some("chg:88:2:n:20:-180")
        ));
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
            .zip(["This button expired", "This history is not yours"])
        {
            assert!(matches!(
                action,
                TelegramAction::AnswerCallback {
                    text: Some(text),
                    show_alert: true,
                    ..
                } if text == expected
            ));
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
            }) if text == "No hay más gastos para mostrar"
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
            }) if text == "Se trabó leyendo tus gastos. Probá de nuevo"
        ));
        assert!(failed.state_diagnostics()[0].contains("synthetic callback read failure"));
    }

    #[test]
    fn charge_history_callback_reports_rejected_and_skipped_edits() {
        let page = || ChargeHistoryPage {
            groups: vec![ChargeHistoryGroup {
                cursor_id: 20,
                created_at: "2026-08-26T17:00:00+00:00".to_owned(),
                entries: vec![ChargeHistoryEntry {
                    id: 20,
                    event_type: "ai_settlement_result".to_owned(),
                    metadata: json!({"charged_credit_units_total":4}),
                }],
            }],
            has_newer: false,
            has_older: false,
            newer_cursor: Some(20),
            older_cursor: Some(20),
        };
        let config = || Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };

        let mut rejected = NativeDispatcher::new(
            config(),
            attempt_actions(Attempt::Fail, ActionScript::edit),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Ok(page()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            rejected.dispatch(callback_update("chg:88:2:o:29:-180", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            rejected
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("charge history edit failed"))
        );
        assert!(matches!(
            rejected.actions.0.last(),
            Some(TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }) if text == "I could not load your spending. Try again"
        ));

        let mut skipped = NativeDispatcher::new(
            config(),
            attempt_actions(Attempt::Skip, ActionScript::edit),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_charge_history_source(Box::new(ChargeHistories {
            result: Ok(page()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            skipped.dispatch(callback_update("chg:88:2:o:29:-180", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            skipped.actions.0.last(),
            Some(TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: true,
                ..
            }) if text == "I could not update the history. Try again"
        ));
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
                error_message: Some(
                    "AI credits are unavailable right now. Try again later or tell the admin"
                        .to_owned()
                ),
            }]
        );
    }

    #[test]
    fn malformed_pre_checkout_sender_is_rejected_natively() {
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
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.actions.0,
            [TelegramAction::AnswerPreCheckout {
                query_id: "checkout-malformed".to_owned(),
                ok: false,
                error_message: Some("I could not validate this payment".to_owned()),
            }]
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|value| value.starts_with("invalid pre-checkout query:"))
        );
    }

    #[test]
    fn topup_command_and_callback_complete_the_native_invoice_flow() {
        let config = Config {
            value: Ok(ChatConfig {
                language: "es".to_owned(),
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
        let command = first_sent(&dispatcher.actions.0);
        assert!(command.text.starts_with("Cargar créditos\n\n"));
        assert_eq!(
            command
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard.len()),
            Some(4)
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
            ] if payload == "topup:p50:88:es" && text == "Listo, te dejé la factura"
        ));
    }

    #[test]
    fn topup_invoice_failure_answers_with_an_alert_without_retrying_the_charge() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::scripted(ActionScript {
                receipts: Receipts::Fixed(None),
                invoice: Attempt::Skip,
                ..ActionScript::default()
            }),
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
            ] if text == "No pude armar la factura. Probá de nuevo"
        ));
    }

    #[test]
    fn double_tapped_topup_pack_sends_a_single_invoice() -> Result<(), String> {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        for _ in 0..2 {
            assert_eq!(
                dispatcher.dispatch(callback_update("topup:p50", "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        let invoices = dispatcher
            .actions
            .0
            .iter()
            .filter(|action| matches!(action, TelegramAction::SendInvoice { .. }))
            .count();
        assert_eq!(invoices, 1);
        assert!(dispatcher.actions.0.iter().any(|action| matches!(
            action,
            TelegramAction::AnswerCallback {
                text: Some(text),
                show_alert: false,
                ..
            } if text == "The invoice is already above"
        )));
        assert_eq!(stored.borrow().len(), 1);
        Ok(())
    }

    #[test]
    fn topup_invoice_failure_releases_the_claim_for_retry() -> Result<(), String> {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::scripted(ActionScript {
                invoice_refusals: 1,
                ..ActionScript::default()
            }),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        for _ in 0..2 {
            assert_eq!(
                dispatcher.dispatch(callback_update("topup:p50", "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        // First attempt failed to invoice and released the claim, so the
        // retry invoiced normally instead of answering "already above".
        let invoices = dispatcher
            .actions
            .0
            .iter()
            .filter(|action| matches!(action, TelegramAction::SendInvoice { .. }))
            .count();
        assert_eq!(invoices, 2);
        assert_eq!(stored.borrow().len(), 1);
        Ok(())
    }

    #[test]
    fn topup_without_claim_storage_still_invoices_with_diagnostics() -> Result<(), String> {
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
        .with_market_price_source(Box::new(ScriptedTakeMarketPrices {
            load: Ok(None),
            takes: RefCell::new(VecDeque::new()),
            saves: RefCell::new(VecDeque::new()),
            candidate: market_candidate_quote(),
        }));
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:p50", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.contains("topup invoice claim failed"))
        );
        assert!(
            dispatcher
                .actions
                .0
                .iter()
                .any(|action| matches!(action, TelegramAction::SendInvoice { .. }))
        );
        Ok(())
    }

    #[test]
    fn topup_invoice_transport_failure_releases_the_claim() -> Result<(), String> {
        let config = Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = NativeDispatcher::new(
            config,
            attempt_actions(Attempt::Fail, ActionScript::invoice),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        // The invoice never reached Telegram, so the update fails and the
        // claim is released for the retry instead of blocking it.
        assert!(matches!(
            dispatcher.dispatch(callback_update("topup:p50", "private", Some("en"))),
            Err(DispatchError::Action(_))
        ));
        assert!(stored.borrow().is_empty());
        // Closing the pack menu afterwards still reaches Telegram.
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:close", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendInvoice { .. },
                TelegramAction::AnswerCallback { .. },
                TelegramAction::DeleteMessage {
                    message_id: MessageId(7),
                    ..
                },
            ]
        ));
        Ok(())
    }

    #[derive(Default)]
    struct FakeLightningCheckout {
        refuse: bool,
        attach_error: bool,
        log: Rc<RefCell<Vec<String>>>,
    }

    impl LightningCheckout for FakeLightningCheckout {
        fn create(
            &mut self,
            user_id: i64,
            chat_id: i64,
            pack: &BillingPackTerms,
            locale: bot_core::locale::Locale,
        ) -> Result<LightningInvoice, String> {
            self.log
                .borrow_mut()
                .push(format!("create {user_id} {chat_id} {} {locale:?}", pack.id));
            if self.refuse {
                return Err("provider down".to_owned());
            }
            Ok(LightningInvoice {
                charge_id: "charge-1".to_owned(),
                payreq: "lnbc1synthetic".to_owned(),
                checkout_url: None,
                sats: Some(512),
            })
        }

        fn attach_message(&mut self, charge_id: &str, message_id: i64) -> Result<(), String> {
            self.log
                .borrow_mut()
                .push(format!("attach {charge_id} {message_id}"));
            if self.attach_error {
                return Err("database down".to_owned());
            }
            Ok(())
        }
    }

    fn lightning_dispatcher(
        actions: Actions,
        checkout: Option<FakeLightningCheckout>,
    ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization> {
        lightning_dispatcher_in("en", actions, checkout)
    }

    fn lightning_dispatcher_in(
        language: &str,
        actions: Actions,
        checkout: Option<FakeLightningCheckout>,
    ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization> {
        let config = Config {
            value: Ok(ChatConfig {
                language: language.to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let dispatcher = NativeDispatcher::new(
            config,
            actions,
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        match checkout {
            Some(checkout) => dispatcher.with_lightning_checkout(Box::new(checkout)),
            None => dispatcher,
        }
    }

    fn callback_alerts(actions: &[TelegramAction]) -> Vec<(String, bool)> {
        actions
            .iter()
            .filter_map(|action| match action {
                TelegramAction::AnswerCallback {
                    text: Some(text),
                    show_alert,
                    ..
                } => Some((text.clone(), *show_alert)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn topup_offers_lightning_only_when_a_checkout_is_configured() {
        let has_lightning_button = |checkout: Option<FakeLightningCheckout>| {
            let mut dispatcher = lightning_dispatcher(Actions::default(), checkout);
            assert_eq!(
                dispatcher.dispatch(update("/topup", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            first_sent(&dispatcher.actions.0)
                .reply_markup
                .as_ref()
                .is_some_and(|markup| {
                    markup
                        .inline_keyboard
                        .iter()
                        .flatten()
                        .any(|button| button.callback_data.as_deref() == Some("topup:ln:c100"))
                })
        };
        assert!(has_lightning_button(Some(FakeLightningCheckout::default())));
        assert!(!has_lightning_button(None));
    }

    #[test]
    fn amount_buttons_and_old_menu_buttons_redraw_the_menu_in_place() {
        let mut dispatcher =
            lightning_dispatcher(Actions::default(), Some(FakeLightningCheckout::default()));
        for data in ["topup:ln", "topup:stars", "topup:amt:500"] {
            assert_eq!(
                dispatcher.dispatch(callback_update(data, "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        let edits: Vec<_> = dispatcher
            .actions
            .0
            .iter()
            .filter_map(|action| match action {
                TelegramAction::EditMessage {
                    message_id: MessageId(7),
                    text,
                    reply_markup: Some(markup),
                    ..
                } => Some((text.clone(), markup.inline_keyboard.len())),
                _ => None,
            })
            .collect();
        let menu = |credits| topup_menu(bot_core::locale::Locale::En, credits, true);
        assert_eq!(
            edits,
            [
                (menu(100).0, 5),
                (menu(100).0, 5),
                (menu(500).0, menu(500).1.inline_keyboard.len()),
            ]
        );
    }

    #[test]
    fn lightning_pack_sends_the_invoice_and_links_its_message() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = lightning_dispatcher(
            Actions::default(),
            Some(FakeLightningCheckout {
                log: Rc::clone(&log),
                ..FakeLightningCheckout::default()
            }),
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:ln:p50", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            *log.borrow(),
            ["create 88 -42 p50 En", "attach charge-1 700"]
        );
        let photos = dispatcher
            .actions
            .0
            .iter()
            .filter_map(|action| match action {
                TelegramAction::SendPhoto {
                    chat_id,
                    photo,
                    caption,
                    reply_markup,
                    ..
                } => Some((chat_id, photo, caption, reply_markup)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(photos.len(), 1);
        let (chat_id, photo, caption, reply_markup) = photos[0];
        assert_eq!(*chat_id, ChatId(-42));
        assert!(photo.starts_with(&[0x89, b'P', b'N', b'G']));
        assert!(caption.starts_with("Lightning invoice ⚡"));
        // The invoice is in the copy button, not the caption.
        assert!(!caption.contains("lnbc1synthetic"));
        assert!(reply_markup.as_ref().is_some_and(|markup| {
            markup.inline_keyboard[0][0]
                .copy_text
                .as_ref()
                .map(|copy| copy.text.as_str())
                == Some("lnbc1synthetic")
        }));
        assert_eq!(
            callback_alerts(&dispatcher.actions.0),
            [("Invoice ready".to_owned(), false)]
        );
        assert!(dispatcher.state_diagnostics().is_empty());
    }

    #[test]
    fn lightning_invoice_without_a_message_id_or_callback_id_still_succeeds() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = lightning_dispatcher(
            Actions::scripted(ActionScript {
                receipts: Receipts::Fixed(None),
                ..ActionScript::default()
            }),
            Some(FakeLightningCheckout {
                log: Rc::clone(&log),
                ..FakeLightningCheckout::default()
            }),
        );
        assert_eq!(
            dispatcher.dispatch(callback_update_with_context(
                "topup:ln:p50",
                json!(88),
                "private",
                7,
                Some(88),
                Some("en"),
                None,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(*log.borrow(), ["create 88 88 p50 En"]);
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendPhoto { .. }]
        ));
    }

    #[test]
    fn lightning_invoice_falls_back_to_text_when_the_photo_is_refused() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = lightning_dispatcher(
            Actions::scripted(ActionScript::photo(Attempt::Skip)),
            Some(FakeLightningCheckout {
                log: Rc::clone(&log),
                ..FakeLightningCheckout::default()
            }),
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:ln:c300", "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            *log.borrow(),
            ["create 88 -42 c300 En", "attach charge-1 700"]
        );
        let invoice = first_sent(&dispatcher.actions.0);
        assert!(
            invoice
                .text
                .starts_with("Lightning invoice ⚡\n\n300 credits for US$1.97")
        );
    }

    #[test]
    fn a_number_replying_to_the_topup_menu_redraws_it_on_that_amount() {
        let reply = |text: &str, replied: &str, from_bot: bool, chat_type: &str| {
            let mut update = update(text, Some("en"));
            if let IncomingEvent::Message(message) = &mut update.event {
                message.chat_type = Some(chat_type.to_owned());
                message.has_reply = true;
                message.replied_sender_is_bot = from_bot;
                message.replied_text = Some(replied.to_owned());
            }
            update
        };
        let menu = topup_menu(bot_core::locale::Locale::En, 100, true).0;
        for (text, expected) in [
            ("300", "300 credits"),
            ("1.500", "1,500 credits"),
            ("5", "10 credits"),
            ("999999", "100,000 credits"),
        ] {
            let mut dispatcher =
                lightning_dispatcher(Actions::default(), Some(FakeLightningCheckout::default()));
            assert_eq!(
                dispatcher.dispatch(reply(text, &menu, true, "private")),
                Ok(DispatchOutcome::Handled)
            );
            assert!(
                first_sent(&dispatcher.actions.0)
                    .text
                    .starts_with(&format!("Add credits\n\n{expected}")),
                "{text}"
            );
        }
        // Spanish menus count too.
        let spanish = topup_menu(bot_core::locale::Locale::Es, 100, true).0;
        let mut dispatcher =
            lightning_dispatcher(Actions::default(), Some(FakeLightningCheckout::default()));
        assert_eq!(
            dispatcher.dispatch(reply("200", &spanish, true, "private")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            first_sent(&dispatcher.actions.0)
                .text
                .contains("200 credits")
        );
        // Not a number, not the menu, not from the bot, or not private: no menu.
        for update in [
            reply("hola", &menu, true, "private"),
            reply("300", "Something else", true, "private"),
            reply("300", &menu, false, "private"),
            reply("300", &menu, true, "group"),
        ] {
            let mut dispatcher =
                lightning_dispatcher(Actions::default(), Some(FakeLightningCheckout::default()));
            let _outcome = dispatcher.dispatch(update);
            assert!(
                !sent_messages(&dispatcher.actions.0)
                    .iter()
                    .any(|message| message.text.starts_with("Add credits")),
            );
        }
    }

    #[test]
    fn lightning_invoice_message_link_failure_is_only_a_diagnostic() {
        let mut dispatcher = lightning_dispatcher(
            Actions::default(),
            Some(FakeLightningCheckout {
                attach_error: true,
                ..FakeLightningCheckout::default()
            }),
        );
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:ln:p50", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.state_diagnostics(),
            ["lightning invoice message charge_id=charge-1: database down"]
        );
        assert_eq!(
            callback_alerts(&dispatcher.actions.0),
            [("Invoice ready".to_owned(), false)]
        );
    }

    #[test]
    fn lightning_guards_answer_with_alerts() {
        let unavailable = |dispatcher: &mut NativeDispatcher<
            Config,
            Actions,
            State,
            Values,
            Samples,
            Authorization,
        >,
                           data: &str,
                           chat_type: &str| {
            assert_eq!(
                dispatcher.dispatch(callback_update(data, chat_type, Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            callback_alerts(&dispatcher.actions.0)
                .pop()
                .map(|(text, show_alert)| {
                    assert!(show_alert);
                    text
                })
                .unwrap_or_default()
        };
        let mut dispatcher =
            lightning_dispatcher(Actions::default(), Some(FakeLightningCheckout::default()));
        assert_eq!(
            unavailable(&mut dispatcher, "topup:ln:p50", "group"),
            "Open this in a private chat"
        );
        assert_eq!(
            unavailable(&mut dispatcher, "topup:ln:nope", "private"),
            "That credit pack is invalid, choose another one"
        );
        dispatcher.billing_available = false;
        assert_eq!(
            unavailable(&mut dispatcher, "topup:ln", "private"),
            bot_core::billing_commands::billing_unavailable(bot_core::locale::Locale::En)
        );
        let mut without_checkout = lightning_dispatcher(Actions::default(), None);
        assert_eq!(
            unavailable(&mut without_checkout, "topup:ln:p50", "private"),
            lightning_invoice_failed(bot_core::locale::Locale::En)
        );
    }

    #[test]
    fn lightning_alerts_speak_spanish() {
        let mut dispatcher = lightning_dispatcher_in(
            "es",
            Actions::default(),
            Some(FakeLightningCheckout::default()),
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::new(RefCell::new(HashMap::new())),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        for (data, chat_type) in [
            ("topup:ln:p50", "group"),
            ("topup:ln:nope", "private"),
            ("topup:ln:p50", "private"),
            ("topup:ln:p50", "private"),
        ] {
            assert_eq!(
                dispatcher.dispatch(callback_update(data, chat_type, Some("es"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        assert_eq!(
            callback_alerts(&dispatcher.actions.0),
            [
                ("Cargá por privado, maestro".to_owned(), true),
                ("Ese pack es fruta, elegí otro".to_owned(), true),
                ("Listo, te dejé la factura".to_owned(), false),
                ("Ya te dejé la factura más arriba".to_owned(), true),
            ]
        );
    }

    #[test]
    fn lightning_alerts_without_a_callback_id_stay_silent() {
        let mut dispatcher =
            lightning_dispatcher(Actions::default(), Some(FakeLightningCheckout::default()));
        assert_eq!(
            dispatcher.dispatch(callback_update_with_context(
                "topup:ln:p50",
                json!(-42),
                "group",
                7,
                Some(88),
                Some("en"),
                None,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn lightning_pack_without_a_user_is_ignored() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = lightning_dispatcher(
            Actions::default(),
            Some(FakeLightningCheckout {
                log: Rc::clone(&log),
                ..FakeLightningCheckout::default()
            }),
        );
        assert_eq!(
            dispatcher.dispatch(callback_update_with_context(
                "topup:ln:p50",
                json!(88),
                "private",
                7,
                None,
                None,
                Some("callback-1"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(log.borrow().is_empty());
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::AnswerCallback { text: None, .. }]
        ));
    }

    #[test]
    fn double_tapped_lightning_pack_creates_a_single_charge() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = lightning_dispatcher(
            Actions::default(),
            Some(FakeLightningCheckout {
                log: Rc::clone(&log),
                ..FakeLightningCheckout::default()
            }),
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        for _ in 0..2 {
            assert_eq!(
                dispatcher.dispatch(callback_update("topup:ln:p50", "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        assert_eq!(
            log.borrow()
                .iter()
                .filter(|entry| entry.starts_with("create"))
                .count(),
            1
        );
        assert_eq!(
            callback_alerts(&dispatcher.actions.0),
            [
                ("Invoice ready".to_owned(), false),
                ("The invoice is already above".to_owned(), true),
            ]
        );
        assert_eq!(stored.borrow().len(), 1);
    }

    #[test]
    fn failed_lightning_charge_releases_the_claim_for_retry() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = lightning_dispatcher(
            Actions::default(),
            Some(FakeLightningCheckout {
                refuse: true,
                log: Rc::clone(&log),
                ..FakeLightningCheckout::default()
            }),
        )
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        for _ in 0..2 {
            assert_eq!(
                dispatcher.dispatch(callback_update("topup:ln:p50", "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
        }
        assert_eq!(log.borrow().len(), 2);
        assert!(stored.borrow().is_empty());
        let failed = lightning_invoice_failed(bot_core::locale::Locale::En).to_owned();
        assert_eq!(
            callback_alerts(&dispatcher.actions.0),
            [(failed.clone(), true), (failed, true)]
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .contains(&"lightning invoice user_id=88 pack=p50: provider down".to_owned())
        );
    }

    #[test]
    fn lightning_claim_storage_failures_are_diagnostics() {
        let mut dispatcher = lightning_dispatcher(
            Actions::default(),
            Some(FakeLightningCheckout {
                refuse: true,
                ..FakeLightningCheckout::default()
            }),
        )
        .with_market_price_source(Box::new(ScriptedTakeMarketPrices {
            load: Ok(None),
            takes: RefCell::new(VecDeque::from([Err("redis down".to_owned())])),
            saves: RefCell::new(VecDeque::new()),
            candidate: market_candidate_quote(),
        }));
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:ln:p50", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.state_diagnostics(),
            [
                "lightning invoice claim failed user_id=88: market selection storage unavailable",
                "lightning invoice user_id=88 pack=p50: provider down",
                "lightning invoice claim release failed: redis down",
            ]
        );
    }

    #[test]
    fn topup_guards_use_the_configured_chat_language() {
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
        assert_eq!(dispatcher.config.chat_ids, ["-42", "-42"]);
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
            ] if invalid == "That credit pack is invalid, choose another one"
                && unavailable == "Los créditos de IA no están disponibles en este momento. Probá más tarde o avisale al admin"
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
        let message = first_sent(&private.actions.0);
        assert_eq!(
            message.text,
            "Saldo de IA: 42.00 créditos\n\nCargá más con /topup"
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
        let group_update = message_update("/balance", Some("es"), |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(group.dispatch(group_update), Ok(DispatchOutcome::Handled));
        assert_eq!(calls.borrow().as_slice(), [(88, None), (88, Some(-42))]);
        let message = first_sent(&group.actions.0);
        assert!(
            message
                .text
                .starts_with("AI balances\n\nYours: 30.00 credits\nGroup: 120.00 credits")
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
        let message = first_sent(&failed.actions.0);
        assert_eq!(message.text, "I could not load your balance. Try again");
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
            Err(DispatchError::MissingService("billing balances"))
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
        let message = first_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "Gastos de IA\n\n26/08 14:32 | 0.08 cr\n  respuesta 0.03 cr\n  web 0.05 cr"
        );
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .and_then(|keyboard| keyboard.inline_keyboard[0][0].callback_data.as_deref()),
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
        let message = first_sent(&empty.actions.0);
        assert_eq!(message.text, "You have no recent AI spending");

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
        let message = first_sent(&invalid.actions.0);
        assert_eq!(message.text, "Mandalo así: /charges [cantidad]");

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
        let message = first_sent(&failed.actions.0);
        assert_eq!(message.text, "Se trabó leyendo tus gastos. Probá de nuevo");
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
            Err(DispatchError::MissingService("charge history"))
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
        let group_update = message_update("/transfer 0.1", Some("es"), |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(success.dispatch(group_update), Ok(DispatchOutcome::Handled));
        assert_eq!(calls.borrow().as_slice(), [(88, -42, 10)]);
        let message = first_sent(&success.actions.0);
        assert_eq!(
            message.text,
            "Pasaste 0.10 créditos al grupo\n\nTu saldo: 2.85 créditos\nSaldo del grupo: 12.15 créditos"
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
        let group_update = message_update("/transfer 1.5", Some("es"), |message| {
            message.chat_type = Some("supergroup".to_owned());
        });
        assert_eq!(
            insufficient.dispatch(group_update),
            Ok(DispatchOutcome::Handled)
        );
        let message = first_sent(&insufficient.actions.0);
        assert_eq!(
            message.text,
            "Not enough personal balance: you have 0.70 credits\nTry a smaller amount or add credits with /topup"
        );
    }

    #[test]
    fn transfer_replying_to_a_person_sends_them_personal_credits() {
        let sent = TransferResult {
            transferred: true,
            user_balance: 285,
            chat_balance: 9_999,
        };
        let run = |result: Result<TransferResult, String>, edit: &dyn Fn(&mut IncomingMessage)| {
            let calls = Rc::new(RefCell::new(Vec::new()));
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
            .with_transfer_sink(Box::new(Transfers {
                result,
                calls: Rc::clone(&calls),
            }))
            // Replies need the AI service configured, as in production.
            .with_ai_conversation_source(Box::new(ai_source(Ok(AiPreparation::silent())).0));
            let update = message_update("/transfer 0.1", Some("es"), |message| {
                message.chat_type = Some("group".to_owned());
                message.has_reply = true;
                message.replied_sender_id = Some(UserId(77));
                edit(message);
            });
            assert_eq!(dispatcher.dispatch(update), Ok(DispatchOutcome::Handled));
            let text = first_sent(&dispatcher.actions.0).text.clone();
            let calls = calls.borrow().clone();
            (text, calls)
        };

        let (text, calls) = run(Ok(sent), &|message| {
            message.replied_sender_first_name = Some(" Ana ".to_owned());
            message.replied_sender_username = Some("ana".to_owned());
        });
        assert_eq!(calls, [(88, 77, 10)]);
        assert_eq!(
            text,
            "Le pasaste 0.10 créditos a Ana\n\nTu saldo: 2.85 créditos"
        );
        let (text, _) = run(Ok(sent), &|message| {
            message.replied_sender_first_name = Some("  ".to_owned());
            message.replied_sender_username = Some("ana".to_owned());
        });
        assert_eq!(
            text,
            "Le pasaste 0.10 créditos a @ana\n\nTu saldo: 2.85 créditos"
        );
        let (text, _) = run(Ok(sent), &|_message| {});
        assert_eq!(
            text,
            "Le pasaste 0.10 créditos a esa persona\n\nTu saldo: 2.85 créditos"
        );
        let (text, calls) = run(Err("synthetic database failure".to_owned()), &|_message| {});
        assert_eq!(calls, [(88, 77, 10)]);
        assert_eq!(text, "Se trabó la transferencia. Probá de nuevo");
        let (text, calls) = run(Ok(sent), &|message| {
            message.replied_sender_is_bot = true;
        });
        assert!(calls.is_empty());
        assert_eq!(text, "Los bots no usan créditos, pasáselos a una persona");

        // Replying to this bot keeps the transfer going to the group.
        let (text, calls) = run(Ok(sent), &|message| {
            message.replied_sender_username = Some("mybot".to_owned());
            message.replied_sender_is_bot = true;
        });
        assert_eq!(calls, [(88, -42, 10)]);
        assert!(text.starts_with("Pasaste 0.10 créditos al grupo"));
    }

    #[test]
    fn user_transfers_need_the_transfer_service_and_fail_in_english_too() {
        let reply_to_person = |language: &str| {
            message_update("/transfer 0.1", Some(language), |message| {
                message.chat_type = Some("group".to_owned());
                message.has_reply = true;
                message.replied_sender_id = Some(UserId(77));
            })
        };
        let english = || Config {
            value: Ok(ChatConfig {
                language: "en".to_owned(),
                ..ChatConfig::default()
            }),
            chat_ids: Vec::new(),
        };
        let build = || {
            NativeDispatcher::new(
                english(),
                Actions::default(),
                State::default(),
                values(),
                random(),
                authorization(),
                "@mybot",
            )
            .with_ai_conversation_source(Box::new(ai_source(Ok(AiPreparation::silent())).0))
        };
        assert_eq!(
            build().dispatch(reply_to_person("en")),
            Err(DispatchError::MissingService("credit transfers"))
        );
        let mut failing = build().with_transfer_sink(Box::new(Transfers {
            result: Err("synthetic database failure".to_owned()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            failing.dispatch(reply_to_person("en")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            first_sent(&failing.actions.0).text,
            "The transfer failed. Try again"
        );
    }

    #[test]
    fn user_transfer_names_fall_back_by_locale() {
        let message = IncomingMessage {
            replied_sender_first_name: None,
            replied_sender_username: Some(String::new()),
            ..incoming_message("/transfer 1", None)
        };
        assert_eq!(
            super::recipient_display_name(&message, bot_core::locale::Locale::En),
            "them"
        );
        assert_eq!(super::transfer_recipient(&message, ""), None);
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
        let message = first_sent(&private.actions.0);
        assert_eq!(
            message.text,
            "Esto es para grupos, capo. Usalo ahí: /transfer <monto> se lo pasa al grupo, o respondé a alguien para pasárselo a esa persona"
        );

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
        let group_update = message_update("/transfer 1", Some("es"), |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(failed.dispatch(group_update), Ok(DispatchOutcome::Handled));
        let message = first_sent(&failed.actions.0);
        assert_eq!(message.text, "Se trabó la transferencia. Probá de nuevo");
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
        let group_update = message_update("/transfer 1", Some("es"), |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(
            shadow.dispatch(group_update),
            Err(DispatchError::MissingService("credit transfers"))
        );
        assert!(shadow.actions.0.is_empty());
    }

    #[test]
    fn successful_payment_uses_invoice_locale_and_records_once() {
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
        let message = only_sent(&dispatcher.actions.0);
        assert_eq!(message.chat_id, ChatId(42));
        assert_eq!(
            message.text,
            "Top-up complete: +50.00 credits\nPersonal balance: 53.00 credits"
        );
    }

    #[test]
    fn duplicate_and_failed_payment_writes_have_distinct_safe_replies() {
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
        .with_payment_sink(Box::new(Payments {
            result: Ok(StarPaymentReceipt {
                inserted: false,
                user_balance: 5_300,
            }),
            records: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            dispatcher.dispatch(successful_payment_update("p50", 42, 25, Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let message = only_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "This payment was already credited\nPersonal balance: 53.00 credits"
        );
        assert!(dispatcher.state_diagnostics().is_empty());

        // A failed write is retried with the update (recording is idempotent
        // per charge id), so nothing is sent yet and the error is retryable.
        let mut failing = NativeDispatcher::new(
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
        .with_payment_sink(Box::new(Payments {
            result: Err("synthetic database failure".to_owned()),
            records: Rc::new(RefCell::new(Vec::new())),
        }));
        let failure = failing
            .dispatch(successful_payment_update("p50", 42, 25, Some("en")))
            .err();
        assert!(matches!(
            &failure,
            Some(DispatchError::Persistence(text)) if text.contains("charge_id=") && text.contains("synthetic database failure")
        ));
        assert_eq!(
            failure.map(|error| failing.error_disposition(&error)),
            Some(HandlerErrorDisposition::RetryUpdate)
        );
        assert!(failing.actions.0.is_empty());
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
        let message = last_sent(&dispatcher.actions.0);
        assert_eq!(
            message.text,
            "AI credits are unavailable right now. Try again later or tell the admin"
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
            Err(DispatchError::MissingService("payment persistence"))
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
        let texts = sent_texts(&dispatcher.actions.0);
        assert_eq!(
            texts,
            [
                "beta",
                "100000000000000000002",
                "Send options like 'pizza, steak, sushi' or a range like '1-10'",
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
        let message = only_sent(&dispatcher.actions.0);
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(
            message.text,
            "https://fixupx.com/a/status/1\n\ncompartido por @tester"
        );
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .and_then(|markup| markup.inline_keyboard[0][0].url.as_deref()),
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
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendVideo {
                video,
                caption,
                reply_to_message_id: Some(MessageId(7)),
                ..
            }] if video.as_ref() == [1, 2, 3] && caption.contains("compartido por @tester")
        ));

        let mut source = links(true);
        source.oversized_video = Some(vec![4, 5, 6]);
        let mut fallback = NativeDispatcher::new(
            config(),
            Actions::scripted(ActionScript {
                receipts: Receipts::Fixed(Some(MessageId(701))),
                video: Attempt::Skip,
                ..ActionScript::default()
            }),
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
        let incoming = message_update("https://x.com/a/status/1", Some("en"), |message| {
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(3));
            message.sender_username = None;
            message.sender_first_name = Some("Ana".to_owned());
            message.sender_last_name = Some("Test".to_owned());
        });
        assert_eq!(dispatcher.dispatch(incoming), Ok(DispatchOutcome::Handled));
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(message),
                TelegramAction::DeleteMessage {
                    chat_id: ChatId(-42),
                    message_id: MessageId(7),
                },
            ] if message.reply_to_message_id == Some(MessageId(3))
                && message.text == "https://fixupx.com/a/status/1\n\nshared by Ana Test"
        ));
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
    fn link_cases_outside_replacement_require_the_normal_ai_route() {
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
            Err(DispatchError::MissingService("AI conversation"))
        );
        dispatcher.config.value = Ok(ChatConfig::default());
        assert_eq!(
            dispatcher.dispatch(update("/ask https://x.com/a/status/1", None)),
            Err(DispatchError::MissingService("AI conversation"))
        );
        let reply = message_update("mirá https://x.com/a/status/1", None, |message| {
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(3));
        });
        assert_eq!(
            dispatcher.dispatch(reply),
            Err(DispatchError::MissingService("AI conversation"))
        );
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn link_cases_not_owned_by_replacement_fall_through_to_native_ai() {
        let (source, (prepared, _ignored, _deliveries)) = ai_source(Ok(AiPreparation::silent()));
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig {
                    link_mode: "off".to_owned(),
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
        .with_link_replacement_source(Box::new(links(true)))
        .with_ai_conversation_source(Box::new(source));
        assert_eq!(
            dispatcher.dispatch(update("https://x.com/a/status/1", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(prepared.borrow().len(), 1);
    }

    fn group_link_update(text: &str) -> IncomingUpdate {
        wrap_message(group_link_message(text))
    }

    fn group_link_message(text: &str) -> IncomingMessage {
        let mut message = incoming_message(text, None);
        message.chat_type = Some("group".to_owned());
        message
    }

    #[test]
    fn addressed_link_messages_get_fixed_links_and_an_ai_answer_with_preview_context() {
        for link_mode in ["reply", "delete"] {
            let (source, (prepared, _ignored, deliveries)) =
                ai_source(Ok(AiPreparation::reply("respuesta", Some("c1".to_owned()))));
            let mut dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig {
                        link_mode: link_mode.to_owned(),
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
            .with_link_replacement_source(Box::new(links(true)))
            .with_ai_conversation_source(Box::new(source));
            assert_eq!(
                dispatcher.dispatch(group_link_update(
                    "@mybot qué onda esto https://x.com/a/status/1"
                )),
                Ok(DispatchOutcome::Handled)
            );
            let actions = &dispatcher.actions.0;
            assert!(matches!(
                actions.first(),
                Some(TelegramAction::SendMessage(fixed)) if fixed.text.contains("fixupx.com")
            ));
            // The original stays so the AI answer can reply to it.
            assert!(!actions.contains(&TelegramAction::DeleteMessage {
                chat_id: ChatId(-42),
                message_id: MessageId(7),
            }));
            assert!(actions.iter().any(|action| matches!(
                action,
                TelegramAction::SendMessage(reply) if reply.text == "Pensando."
            )));
            assert_eq!(prepared.borrow().len(), 1);
            // The replacement's preview is reused instead of fetching again.
            assert_eq!(
                prepared.borrow()[0].link_context.as_deref(),
                Some("LINKS DEL MENSAJE:\n1. https://fixupx.com/a/status/1\ntitulo: example")
            );
            assert_eq!(deliveries.borrow().len(), 1);
            assert_eq!(dispatcher.state.outgoing.len(), 1);
        }
    }

    #[test]
    fn retried_addressed_link_messages_answer_again_without_a_second_fixed_link() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let (source, (prepared, _ignored, _deliveries)) =
            ai_source(Ok(AiPreparation::reply("respuesta", Some("c1".to_owned()))));
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
        .with_link_replacement_source(Box::new(links(true)))
        .with_ai_conversation_source(Box::new(source))
        .with_market_price_source(Box::new(SelectableMarketPrices {
            initial: market_selection_load(None),
            candidate: market_candidate_quote(),
            stored: Rc::clone(&stored),
            selected: Rc::new(RefCell::new(Vec::new())),
        }));
        let fixed_links = |actions: &[TelegramAction]| {
            actions
                .iter()
                .filter(|action| {
                    matches!(
                        action,
                        TelegramAction::SendMessage(fixed) if fixed.text.contains("fixupx.com")
                    )
                })
                .count()
        };
        for _ in 0..2 {
            assert_eq!(
                dispatcher.dispatch(group_link_update(
                    "@mybot qué onda esto https://x.com/a/status/1"
                )),
                Ok(DispatchOutcome::Handled)
            );
        }
        assert_eq!(fixed_links(&dispatcher.actions.0), 1);
        assert_eq!(prepared.borrow().len(), 2);
        assert_eq!(
            stored.borrow().keys().cloned().collect::<Vec<_>>(),
            [super::sent_link_fix_key(ChatId(-42), MessageId(7))]
        );
    }

    #[test]
    fn unreplaced_links_addressed_to_the_bot_fall_through_to_the_ai() {
        let (source, (prepared, _ignored, _deliveries)) = ai_source(Ok(AiPreparation::silent()));
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
        .with_link_replacement_source(Box::new(links(false)))
        .with_ai_conversation_source(Box::new(source));
        let mut reply = group_link_message("https://x.com/a/status/1");
        reply.has_reply = true;
        reply.replied_message_id = Some(MessageId(3));
        reply.replied_sender_username = Some("mybot".to_owned());
        reply.replied_text = Some("hola".to_owned());
        let reply = wrap_message(reply);
        assert_eq!(dispatcher.dispatch(reply), Ok(DispatchOutcome::Handled));
        assert_eq!(prepared.borrow().len(), 1);
        // Replacement already inspected the link and found nothing to add.
        assert_eq!(prepared.borrow()[0].link_context, None);
        assert!(dispatcher.state.incoming.is_empty());

        // Links the replacement does not own are previewed for the AI turn.
        assert_eq!(
            dispatcher.dispatch(group_link_update("@mybot leé https://example.com/nota")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            prepared.borrow()[1].link_context.as_deref(),
            Some("PREVIEW for @mybot leé https://example.com/nota")
        );
        // Messages without links never ask for previews.
        assert_eq!(
            dispatcher.dispatch(group_link_update("@mybot hola")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(prepared.borrow()[2].link_context, None);
    }

    #[test]
    fn addressing_requires_a_mention_a_reply_to_the_bot_or_private_commentary() {
        let message = |text: &str, chat_type: &str| {
            let mut message = incoming_message(text, None);
            message.chat_type = Some(chat_type.to_owned());
            message.replied_sender_username = text.starts_with("reply").then(|| "MyBot".to_owned());
            message
        };
        let (group_link, private_link, private_text, group_mention, group_reply, group_text) = (
            message("https://x.com/a/status/1", "group"),
            message("https://x.com/a/status/1", "private"),
            message("mirá https://x.com/a/status/1", "private"),
            message("@mybot https://x.com/a/status/1", "group"),
            message("reply https://x.com/a/status/1", "group"),
            message("mirá https://x.com/a/status/1", "group"),
        );
        let addressed_with = |bot_name: &str,
                              message: &IncomingMessage,
                              config: &ChatConfig,
                              metadata: Option<AiReplyMetadata>| {
            let (source, _observations) = ai_source(Ok(AiPreparation::silent()));
            let mut dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(config.clone()),
                    chat_ids: Vec::new(),
                },
                Actions::default(),
                State::default(),
                values(),
                random(),
                authorization(),
                bot_name,
            )
            .with_ai_conversation_source(Box::new(AiSource { metadata, ..source }));
            let text = message
                .content
                .as_ref()
                .map(|content| content.text.clone())
                .unwrap_or_default();
            dispatcher.is_addressed_to_bot(message, &text, config)
        };
        let addressed = |bot_name: &str, message: &IncomingMessage| {
            addressed_with(bot_name, message, &ChatConfig::default(), None)
        };
        assert!(!addressed("@MyBot", &group_link));
        assert!(!addressed("@MyBot", &private_link));
        assert!(addressed("@MyBot", &private_text));
        assert!(addressed("@MyBot", &group_mention));
        assert!(addressed("@MyBot", &group_reply));
        assert!(!addressed("@MyBot", &group_text));
        assert!(!addressed(" ", &private_link));
        assert!(addressed(" ", &private_text));
        assert!(!addressed(" ", &group_mention));

        // Replies routing ignores are not addressed, so delete mode still
        // removes their original.
        let mut link_fix_reply = group_reply.clone();
        link_fix_reply.replied_message_id = Some(MessageId(3));
        link_fix_reply.replied_text = Some("https://fixupx.com/a/status/9".to_owned());
        assert!(!addressed("@MyBot", &link_fix_reply));
        let followups_on = ChatConfig {
            ignore_link_fix_followups: false,
            ..ChatConfig::default()
        };
        assert!(addressed_with(
            "@MyBot",
            &link_fix_reply,
            &followups_on,
            None
        ));

        let mut command_reply = group_reply;
        command_reply.replied_message_id = Some(MessageId(3));
        let command = || {
            Some(AiReplyMetadata {
                kind: "command".to_owned(),
                uses_ai: false,
            })
        };
        let command_followups_off = ChatConfig {
            ai_command_followups: false,
            ..ChatConfig::default()
        };
        assert!(!addressed_with(
            "@MyBot",
            &command_reply,
            &command_followups_off,
            command()
        ));
        assert!(addressed_with(
            "@MyBot",
            &command_reply,
            &ChatConfig {
                ai_command_followups: true,
                ..ChatConfig::default()
            },
            command()
        ));
        assert!(addressed_with(
            "@MyBot",
            &command_reply,
            &command_followups_off,
            None
        ));
    }

    #[test]
    fn thinking_status_waits_for_the_source_to_admit_the_turn() {
        let (source, (prepared, _ignored, _deliveries)) = ai_source(Ok(AiPreparation::silent()));
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
        .with_ai_conversation_source(Box::new(AiSource {
            admit: false,
            ..source
        }));
        assert_eq!(
            dispatcher.dispatch(update("synthetic question", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(prepared.borrow().len(), 1);
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn random_source_errors_are_not_acknowledged() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::default(),
            values(),
            Samples {
                failing: true,
                ..random()
            },
            authorization(),
            "@mybot",
        );
        assert!(matches!(
            dispatcher.dispatch(update("/random alpha, beta", None)),
            Err(DispatchError::Random("synthetic random failure"))
        ));
        assert!(dispatcher.actions.0.is_empty());
        assert!(dispatcher.state.incoming.is_empty());
        // Numeric ranges draw an integer from the same failing source.
        assert!(matches!(
            dispatcher.dispatch(update("/random 1-10", None)),
            Err(DispatchError::Random("synthetic random failure"))
        ));
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn state_failures_are_diagnostic_and_do_not_duplicate_or_block_delivery() {
        let config = Config {
            value: Ok(ChatConfig::default()),
            chat_ids: Vec::new(),
        };
        let mut dispatcher = NativeDispatcher::new(
            config,
            Actions::default(),
            State::failing(),
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

        let (mut source, _observations) = ai_source(Ok(AiPreparation::silent()));
        source.media_preparation = Some(Ok(AiPreparation::reply("synthetic media result", None)));
        source.summary_preparation =
            Some(Ok(AiPreparation::reply("synthetic summary result", None)));
        let mut dispatcher = dispatcher.with_ai_conversation_source(Box::new(source));
        let media = message_update("/transcribe", None, |message| {
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(6));
            message.audio_media_kind = Some("voice".to_owned());
            if let Some(content) = message.content.as_mut() {
                content.audio_file_id = Some("synthetic-audio".to_owned());
            }
        });
        assert_eq!(dispatcher.dispatch(media), Ok(DispatchOutcome::Handled));
        assert!(dispatcher.state_diagnostics().iter().any(|diagnostic| {
            diagnostic == "incoming media command state: synthetic incoming failure"
        }));
        assert!(dispatcher.state_diagnostics().iter().any(|diagnostic| {
            diagnostic == "outgoing media command state: synthetic outgoing failure"
        }));
        assert_eq!(
            dispatcher.dispatch(update("/summary", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.state_diagnostics().iter().any(|diagnostic| {
            diagnostic == "incoming summary command state: synthetic incoming failure"
        }));
    }

    #[test]
    fn ai_routing_reports_metadata_and_ignored_state_failures() {
        let (mut metadata_source, _observations) = ai_source(Ok(AiPreparation::silent()));
        metadata_source.metadata_error = Some("synthetic metadata failure".to_owned());
        let mut metadata_dispatcher = NativeDispatcher::new(
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
        .with_ai_conversation_source(Box::new(metadata_source));
        let reply = message_update("synthetic follow-up", None, |message| {
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(6));
            message.replied_sender_username = Some("mybot".to_owned());
        });
        assert_eq!(
            metadata_dispatcher.dispatch(reply),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            metadata_dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| {
                    diagnostic == "AI reply metadata: synthetic metadata failure"
                })
        );

        let (mut ignored_source, _observations) = ai_source(Ok(AiPreparation::silent()));
        ignored_source.ignored_error = Some("synthetic ignored-state failure".to_owned());
        let mut ignored_dispatcher = NativeDispatcher::new(
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
        .with_ai_conversation_source(Box::new(ignored_source));
        let ordinary = message_update("synthetic group message", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(
            ignored_dispatcher.dispatch(ordinary),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            ignored_dispatcher
                .state_diagnostics()
                .iter()
                .any(|diagnostic| {
                    diagnostic == "ignored AI message state: synthetic ignored-state failure"
                })
        );
    }

    #[test]
    fn public_handler_and_default_random_boundary_are_usable() {
        let mut samples = Samples {
            choice_index: usize::MAX,
            integer: BigInt::from(0_u8),
            failing: false,
        };
        assert_eq!(samples.unit_interval(), Ok(0.9999));

        let mut native_dispatcher = dispatcher();
        let mut unsupported = update("synthetic message", None);
        unsupported.event = IncomingEvent::Unsupported;
        assert!(crate::runtime::UpdateHandler::handle(&mut native_dispatcher, unsupported).is_ok());
        assert_eq!(
            native_dispatcher.last_outcome(),
            Some(DispatchOutcome::Unsupported)
        );

        let mut missing_tasks = dispatcher();
        assert!(matches!(
            missing_tasks.dispatch(update("/tasks", None)),
            Err(DispatchError::MissingService("scheduled tasks"))
        ));
    }

    #[test]
    fn ai_streaming_and_localized_failure_boundaries_remain_nonfatal() {
        for summary in [false, true] {
            let (mut source, _observations) =
                ai_source(Ok(AiPreparation::reply("synthetic final response", None)));
            source.tokens = vec!["synthetic draft response".to_owned()];
            if summary {
                source.summary_preparation = source.preparation.take();
            }
            let mut dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig::default()),
                    chat_ids: Vec::new(),
                },
                attempt_actions(Attempt::Fail, ActionScript::edit),
                State::default(),
                values(),
                random(),
                authorization(),
                "@mybot",
            )
            .with_ai_conversation_source(Box::new(source));
            let input = if summary {
                update("/summary", None)
            } else {
                update("synthetic question", None)
            };
            assert_eq!(dispatcher.dispatch(input), Ok(DispatchOutcome::Handled));
            assert!(
                dispatcher
                    .state_diagnostics()
                    .iter()
                    .any(|diagnostic| diagnostic.contains("Telegram stream ignored"))
            );
        }

        let (source, _observations) = ai_source(Err("synthetic provider failure".to_owned()));
        let mut failed = NativeDispatcher::new(
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
            failed.dispatch(update("synthetic question", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            failed.actions.0.as_slice(),
            [
                TelegramAction::SendMessage(thinking),
                TelegramAction::DeleteMessage { .. },
                TelegramAction::SendMessage(failure),
            ] if thinking.text == "Pensando."
                && failure.text == "Me quedé reculando y no te pude responder. Probá de nuevo"
        ));

        let (source, _observations) = ai_source(Ok(AiPreparation::silent()));
        let mut missing_media = NativeDispatcher::new(
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
        let media = message_update("/transcribe", None, |message| {
            message.has_reply = true;
            message.replied_message_id = Some(MessageId(6));
            message.audio_media_kind = Some("voice".to_owned());
            if let Some(content) = message.content.as_mut() {
                content.audio_file_id = Some("synthetic-audio".to_owned());
            }
        });
        assert_eq!(missing_media.dispatch(media), Ok(DispatchOutcome::Handled));
        assert_eq!(
            missing_media.dispatch(update("/summary", None)),
            Ok(DispatchOutcome::Handled)
        );

        let (source, _observations) = ai_source(Ok(AiPreparation::silent()));
        let mut random_dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig {
                    ai_random_replies: true,
                    ..ChatConfig::default()
                }),
                chat_ids: Vec::new(),
            },
            Actions::default(),
            State::default(),
            values(),
            Samples {
                choice_index: 9_999,
                integer: BigInt::from(0_u8),
                failing: false,
            },
            authorization(),
            "@mybot",
        )
        .with_ai_conversation_source(Box::new(source));
        let group = message_update("synthetic group message", None, |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(
            random_dispatcher.dispatch(group),
            Ok(DispatchOutcome::Handled)
        );
    }

    fn market_candidate(
        id: &str,
        symbol: &str,
        name: &str,
        exchange: &str,
        contracts: Vec<TokenAddress>,
    ) -> bot_core::market_prices::MarketCandidate {
        bot_core::market_prices::MarketCandidate {
            id: id.to_owned(),
            symbol: symbol.to_owned(),
            name: name.to_owned(),
            slug: symbol.to_ascii_lowercase(),
            price: "1".to_owned(),
            change: "0".to_owned(),
            currency: String::new(),
            exchange: exchange.to_owned(),
            asset_type: String::new(),
            contracts,
        }
    }

    #[test]
    fn unnamed_dex_tokens_are_labelled_by_their_symbol() {
        let mut signal = token_signal();
        signal.pair.base_token.name = String::new();
        let candidate = super::market_token_candidate(&signal, Some("24h"), None);
        assert_eq!(candidate.name, "SYN");
        assert_eq!(candidate.symbol, "SYN");
        assert_eq!(
            candidate.id,
            "token:solana:solana:J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump"
        );
    }

    #[test]
    fn provider_candidates_replace_overlapping_dex_candidates() {
        let token = token_signal().token;
        let mut candidates = vec![
            market_candidate("token:solana:solana:x", "SYN", "", "", vec![token.clone()]),
            market_candidate("42", "SYN", "Synthetic", "", vec![token.clone()]),
            // A second DEX duplicate never replaces the provider identity.
            market_candidate("token:solana:solana:y", "SYN", "", "", vec![token]),
        ];
        deduplicate_market_candidates(&mut candidates);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, "42");
        assert_eq!(candidates[0].name, "Synthetic");
    }

    #[test]
    fn token_signals_match_symbol_name_and_slug_queries() {
        let signal = token_signal();
        let matches = |query: SignalQuery| super::token_signal_matches_query(&signal, &query);
        assert!(matches(SignalQuery::Symbol("$syn".to_owned())));
        assert!(matches(SignalQuery::Symbol("synthetic token".to_owned())));
        assert!(!matches(SignalQuery::Symbol("other".to_owned())));
        assert!(matches(SignalQuery::Slug("synthetic-token".to_owned())));
        assert!(matches(SignalQuery::Slug("syn".to_owned())));
        assert!(!matches(SignalQuery::Slug("other-token".to_owned())));
    }

    #[test]
    fn provider_requests_use_slugs_and_detected_contract_addresses() {
        let token = token_signal().token;
        assert_eq!(
            super::provider_request_text(
                "https://www.coingecko.com/en/coins/libra",
                &SignalQuery::Slug("libra".to_owned())
            ),
            "libra"
        );
        assert_eq!(
            super::provider_request_text(
                "https://dexscreener.com/solana/J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump",
                &SignalQuery::Address(token.clone())
            ),
            token.address
        );
        assert_eq!(
            super::provider_request_text(&token.address, &SignalQuery::Address(token.clone())),
            token.address
        );
        assert_eq!(
            super::provider_request_text("syn", &SignalQuery::Symbol("syn".to_owned())),
            "syn"
        );
    }

    #[test]
    fn market_selection_buttons_disambiguate_identical_labels() {
        let token = token_signal().token;
        let selection = bot_core::market_prices::MarketSelection {
            query: "syn".to_owned(),
            timeframe: None,
            target_symbol: "USD".to_owned(),
            target_parameter: "USD".to_owned(),
            conversion: None,
            candidates: vec![
                market_candidate("stock:SYN", "SYN", "Synthetic Inc", "", Vec::new()),
                market_candidate("stock:SYN", "SYN", "Synthetic Inc", "", Vec::new()),
                market_candidate("7", "SYN", "Synthetic", "", vec![token.clone()]),
                market_candidate("8", "SYN", "Synthetic", "", vec![token]),
            ],
        };
        let keyboard =
            super::market_selection_page("sel", &selection, bot_core::locale::Locale::En, 0);
        let labels = keyboard
            .inline_keyboard
            .iter()
            .take(4)
            .map(|row| row[0].text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            [
                // A stock without an exchange shows only its ticker, and
                // duplicates without contracts fall back to their ids.
                "Synthetic Inc, SYN [stock:SYN]",
                "Synthetic Inc, SYN [stock:SYN]",
                "Synthetic, SYN (solana) [J8PSdN…pump]",
                "Synthetic, SYN (solana) [J8PSdN…pump]",
            ]
        );
    }

    /// Message state storage that rejects every write.
    fn broken_state_dispatcher(
        config: ChatConfig,
    ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization> {
        NativeDispatcher::new(
            Config {
                value: Ok(config),
                chat_ids: Vec::new(),
            },
            Actions::default(),
            State::failing(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
    }

    /// Link replacement that only implements the required loader, so the AI
    /// turn falls back to the trait's "no preview" default. It never finds a
    /// better link.
    struct LoadOnlyLinks;

    impl LinkReplacementSource for LoadOnlyLinks {
        fn load(&mut self, text: &str, _now_unix: i64) -> LinkReplacementLoad {
            LinkReplacementLoad {
                replacement: LinkReplacement {
                    text: text.to_owned(),
                    changed: false,
                    original_links: Vec::new(),
                },
                context: None,
                oversized_video: None,
                diagnostics: Vec::new(),
            }
        }
    }

    #[test]
    fn link_replacement_guards_skip_incomplete_and_unreplaceable_messages() {
        let mut dispatcher = dispatcher().with_link_replacement_source(Box::new(links(true)));
        let config = ChatConfig::default();
        let mut anonymous = incoming_message("https://x.com/a/status/1", None);
        anonymous.sender_id = None;
        assert_eq!(
            dispatcher.dispatch_link_replacement(
                &anonymous,
                &config,
                bot_core::locale::Locale::Es,
                1_672_531_200,
                false,
            ),
            Ok(Some(DispatchOutcome::Unsupported))
        );
        let ordinary = incoming_message("mirá https://example.com/nota", None);
        assert_eq!(
            dispatcher.dispatch_link_replacement(
                &ordinary,
                &config,
                bot_core::locale::Locale::Es,
                1_672_531_200,
                false,
            ),
            Ok(None)
        );
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn link_replacement_state_failures_are_diagnostic() {
        let mut unchanged = broken_state_dispatcher(ChatConfig::default())
            .with_link_replacement_source(Box::new(links(false)));
        assert_eq!(
            unchanged.dispatch(update("https://x.com/a/status/1", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert!(unchanged.actions.0.is_empty());
        assert_eq!(
            unchanged.state_diagnostics(),
            ["unreplaced link state: synthetic incoming failure"]
        );

        let mut fixed = broken_state_dispatcher(ChatConfig::default())
            .with_link_replacement_source(Box::new(links(true)));
        assert_eq!(
            fixed.dispatch(update("https://x.com/a/status/1", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&fixed.actions.0),
            ["https://fixupx.com/a/status/1\n\ncompartido por @tester"]
        );
        assert_eq!(
            fixed.state_diagnostics(),
            ["fixed link state: synthetic outgoing failure"]
        );
    }

    #[test]
    fn addressed_link_fix_marker_failures_do_not_block_the_fix_or_the_answer() {
        let (source, (prepared, _ignored, _deliveries)) =
            ai_source(Ok(AiPreparation::reply("respuesta", None)));
        let mut dispatcher = dispatcher()
            .with_link_replacement_source(Box::new(links(true)))
            .with_market_price_source(Box::new(ScriptedTakeMarketPrices {
                load: Err("synthetic marker lookup failure".to_owned()),
                takes: RefCell::new(VecDeque::new()),
                saves: RefCell::new(VecDeque::from([Err(
                    "synthetic marker save failure".to_owned()
                )])),
                candidate: market_candidate_quote(),
            }))
            .with_ai_conversation_source(Box::new(source));
        assert_eq!(
            dispatcher.dispatch(group_link_update("@mybot mirá https://x.com/a/status/1")),
            Ok(DispatchOutcome::Handled)
        );
        let diagnostics = dispatcher.state_diagnostics();
        assert!(
            diagnostics
                .iter()
                .any(|entry| entry == "sent link fix lookup: synthetic marker lookup failure")
        );
        assert!(
            diagnostics
                .iter()
                .any(|entry| entry == "sent link fix marker: synthetic marker save failure")
        );
        assert_eq!(
            first_sent(&dispatcher.actions.0).text,
            "https://fixupx.com/a/status/1\n\ncompartido por @tester"
        );
        assert_eq!(prepared.borrow().len(), 1);
    }

    #[test]
    fn link_sources_without_previews_leave_the_ai_turn_without_link_context() {
        let (source, (prepared, _ignored, _deliveries)) =
            ai_source(Ok(AiPreparation::reply("respuesta", None)));
        let mut dispatcher = dispatcher()
            .with_link_replacement_source(Box::new(LoadOnlyLinks))
            .with_ai_conversation_source(Box::new(source));
        assert_eq!(
            dispatcher.dispatch(group_link_update("@mybot leé https://example.com/nota")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(prepared.borrow().len(), 1);
        assert_eq!(prepared.borrow()[0].link_context, None);
        // An unaddressed link it cannot improve is only remembered.
        let answered = dispatcher.actions.0.len();
        assert_eq!(
            dispatcher.dispatch(group_link_update("https://x.com/a/status/1")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(dispatcher.actions.0.len(), answered);
        assert_eq!(dispatcher.state.incoming.len(), 1);
        assert_eq!(prepared.borrow().len(), 1);
    }

    #[test]
    fn token_signal_source_defaults_expand_loads_and_offer_no_history() {
        let mut source = NoHistorySignals(Some(token_signal()));
        let candidates = source.load_candidates(&SignalQuery::Symbol("syn".to_owned()));
        assert_eq!(candidates.signals, [token_signal()]);
        assert!(candidates.diagnostics.is_empty());
        assert_eq!(
            source.render_period_photo(&token_signal(), "24h", 1_672_531_200),
            Err("requested token history unavailable".to_owned())
        );
        assert_eq!(
            source.period_candles(&token_signal(), "24h", 1_672_531_200),
            Ok(Vec::new())
        );
        assert_eq!(source.load_state("abc"), Ok(None));
        assert_eq!(source.save_state("abc", &requester_state(None)), Ok(()));
        assert_eq!(source.clear_state("abc"), Ok(()));
    }

    fn payment_message(chat: Value, from: Value, payment: Value) -> IncomingUpdate {
        IncomingUpdate {
            update_id: 103,
            event: IncomingEvent::SuccessfulPayment(Map::from_iter([
                ("chat".to_owned(), chat),
                ("from".to_owned(), from),
                ("successful_payment".to_owned(), payment),
            ])),
        }
    }

    fn valid_payment(payload: &str) -> Value {
        json!({
            "currency": "XTR",
            "invoice_payload": payload,
            "telegram_payment_charge_id": "charge-1",
            "total_amount": 25,
        })
    }

    #[test]
    fn malformed_and_anonymous_successful_payments_are_acknowledged_silently() {
        let records = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = dispatcher().with_payment_sink(Box::new(Payments {
            result: Ok(StarPaymentReceipt {
                inserted: true,
                user_balance: 5_000,
            }),
            records: Rc::clone(&records),
        }));
        assert_eq!(
            dispatcher.dispatch(payment_message(
                json!("not an object"),
                json!({"id": 42}),
                valid_payment("topup:p50:42:en"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.state_diagnostics(),
            [
                "invalid successful payment: Telegram payment message, chat, or payment payload is malformed"
            ]
        );
        for (chat, from) in [
            (json!({"type": "private"}), json!({"id": 42})),
            (json!({"id": 42, "type": "private"}), json!({})),
        ] {
            assert_eq!(
                dispatcher.dispatch(payment_message(
                    chat,
                    from,
                    valid_payment("topup:p50:42:en")
                )),
                Ok(DispatchOutcome::Handled)
            );
        }
        assert!(dispatcher.actions.0.is_empty());
        assert!(records.borrow().is_empty());
    }

    #[test]
    fn spanish_payment_failures_are_localized() {
        let records = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = dispatcher().with_payment_sink(Box::new(Payments {
            result: Err("synthetic ledger failure".to_owned()),
            records: Rc::clone(&records),
        }));
        let chat = json!({"id": 42, "type": "private"});
        assert_eq!(
            dispatcher.dispatch(payment_message(
                chat.clone(),
                json!({"id": 42}),
                json!({
                    "currency": "XTR",
                    "invoice_payload": "topup:p50:42:es",
                    "telegram_payment_charge_id": "charge-1",
                    "total_amount": 24,
                }),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.dispatch(payment_message(
                chat,
                json!({"id": 42}),
                valid_payment("topup:p50:42:es"),
            )),
            Err(DispatchError::Persistence(_))
        ));
        assert_eq!(
            sent_texts(&dispatcher.actions.0),
            ["Me cayó un pago raro y no lo pude validar. Avisale al admin"]
        );
        assert_eq!(records.borrow().len(), 1);
    }

    #[test]
    fn recorded_payment_in_a_non_numeric_chat_is_an_invariant_failure() {
        let records = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = dispatcher().with_payment_sink(Box::new(Payments {
            result: Ok(StarPaymentReceipt {
                inserted: true,
                user_balance: 5_000,
            }),
            records: Rc::clone(&records),
        }));
        assert_eq!(
            dispatcher.dispatch(payment_message(
                json!({"id": "channel", "type": "private"}),
                json!({"id": 42}),
                valid_payment("topup:p50:42:en"),
            )),
            Err(DispatchError::Invariant(
                "validated payment chat id was not numeric"
            ))
        );
        // The ledger write is idempotent, so a retry cannot double credit.
        assert_eq!(records.borrow().len(), 1);
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn command_menu_sync_failure_does_not_block_the_language_reply() {
        let mut dispatcher = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            failing_actions(
                ActionKind::SetCommands,
                usize::MAX,
                "synthetic menu failure",
                false,
            ),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        );
        assert_eq!(
            dispatcher.dispatch(update("/idioma en", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&dispatcher.actions.0),
            ["Done, I will speak English now"]
        );
        assert_eq!(
            dispatcher.state_diagnostics(),
            ["chat command menu update failed chat_id=-42"]
        );
    }

    #[test]
    fn price_delivery_state_failures_are_diagnostic() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut dispatcher = broken_state_dispatcher(ChatConfig::default())
            .with_market_price_source(Box::new(MarketPrices {
                result: MarketPriceLoad {
                    chart: None,
                    selection: None,
                    no_assets_found: false,
                    text: "BTC: 1 USD".to_owned(),
                    diagnostics: Vec::new(),
                },
                calls: Rc::clone(&calls),
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p btc", None)),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(sent_texts(&dispatcher.actions.0), ["BTC: 1 USD"]);
        assert_eq!(
            dispatcher.state_diagnostics(),
            [
                "incoming price state: synthetic incoming failure",
                "outgoing price state: synthetic outgoing failure",
            ]
        );
    }

    /// Token signals with scripted discovery, lookups, rendering and state.
    struct ScriptedSignals {
        candidates: Vec<TokenSignal>,
        token: Option<TokenSignal>,
        photo: Result<Vec<u8>, String>,
        state: Result<Option<SignalState>, String>,
        clear_error: Option<String>,
    }

    impl ScriptedSignals {
        fn new(candidates: Vec<TokenSignal>, token: Option<TokenSignal>) -> Self {
            Self {
                candidates,
                token,
                photo: Ok(b"scripted-card".to_vec()),
                state: Ok(None),
                clear_error: None,
            }
        }
    }

    impl TokenSignalSource for ScriptedSignals {
        fn load(&mut self, _query: &SignalQuery) -> TokenSignalLoad {
            TokenSignalLoad {
                signal: self.candidates.first().cloned(),
                diagnostics: Vec::new(),
            }
        }

        /// The best match first, followed by the other scripted pairs.
        fn load_candidates(&mut self, query: &SignalQuery) -> super::TokenSignalCandidates {
            let best = self.load(query);
            super::TokenSignalCandidates {
                signals: best
                    .signal
                    .into_iter()
                    .chain(self.candidates.iter().skip(1).cloned())
                    .collect(),
                diagnostics: vec!["scripted discovery".to_owned()],
            }
        }

        fn load_token(&mut self, _token: &TokenAddress) -> TokenSignalLoad {
            TokenSignalLoad {
                signal: self.token.clone(),
                diagnostics: Vec::new(),
            }
        }

        fn render_period_photo(
            &mut self,
            _: &TokenSignal,
            _: &str,
            _: i64,
        ) -> Result<Vec<u8>, String> {
            self.photo.clone()
        }

        fn load_state(&mut self, _signal_id: &str) -> Result<Option<SignalState>, String> {
            self.state.clone()
        }

        fn save_state(&mut self, _signal_id: &str, _state: &SignalState) -> Result<(), String> {
            Ok(())
        }

        fn clear_state(&mut self, _signal_id: &str) -> Result<(), String> {
            self.clear_error.clone().map_or(Ok(()), Err)
        }
    }

    /// Token signals that rely on the trait's defaults: no candle history
    /// and no period photo.
    struct NoHistorySignals(Option<TokenSignal>);

    impl TokenSignalSource for NoHistorySignals {
        fn load(&mut self, _query: &SignalQuery) -> TokenSignalLoad {
            TokenSignalLoad {
                signal: self.0.clone(),
                diagnostics: Vec::new(),
            }
        }

        fn load_token(&mut self, _token: &TokenAddress) -> TokenSignalLoad {
            self.load(&SignalQuery::Symbol(String::new()))
        }

        fn load_state(&mut self, _signal_id: &str) -> Result<Option<SignalState>, String> {
            Ok(None)
        }

        fn save_state(&mut self, _signal_id: &str, _state: &SignalState) -> Result<(), String> {
            Ok(())
        }
    }

    fn token_signal_at(address: &str) -> TokenSignal {
        let mut signal = token_signal();
        signal.token.address = address.to_owned();
        signal.pair.base_token.address = address.to_owned();
        signal
    }

    const SECOND_MINT: &str = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU";

    fn quote_load(text: &str, no_assets_found: bool) -> MarketPriceLoad {
        MarketPriceLoad {
            chart: None,
            selection: None,
            no_assets_found,
            text: text.to_owned(),
            diagnostics: Vec::new(),
        }
    }

    fn market_prices(result: MarketPriceLoad) -> Box<MarketPrices> {
        Box::new(MarketPrices {
            result,
            calls: Rc::new(RefCell::new(Vec::new())),
        })
    }

    fn selection_storage(
        initial: MarketPriceLoad,
        candidate: MarketPriceLoad,
        stored: &Rc<RefCell<HashMap<String, String>>>,
    ) -> Box<SelectionStorageMarketPrices> {
        Box::new(SelectionStorageMarketPrices {
            initial,
            candidate,
            stored: Rc::clone(stored),
            save_calls: Rc::new(RefCell::new(0)),
            fail_save_at: None,
            fail_load: None,
            fail_clear: None,
            render_success: false,
            render_caption: None,
        })
    }

    fn libra_chart(token: Option<TokenAddress>) -> bot_core::market_prices::MarketChart {
        bot_core::market_prices::MarketChart {
            timeframe: None,
            symbol: "LIBRA".to_owned(),
            name: "Libra Finance".to_owned(),
            yahoo_symbol: "LIBRA-USD".to_owned(),
            token,
            candidate: None,
        }
    }

    #[test]
    fn conversion_queries_with_several_assets_persist_a_selection_menu() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = dispatcher().with_market_price_source(selection_storage(
            market_selection_load(None),
            market_candidate_quote(),
            &stored,
        ));
        assert_eq!(
            dispatcher.dispatch(update("/p 2 libra in usd", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let menu = only_sent(&dispatcher.actions.0);
        assert_eq!(menu.text, "Choose an asset: libra");
        assert!(menu.reply_markup.is_some());
        assert_eq!(stored.borrow().len(), 1);
    }

    #[test]
    fn several_dex_matches_extend_the_provider_menu() {
        let mut dispatcher = dispatcher()
            .with_market_price_source(market_prices(market_selection_load(None)))
            .with_token_signal_source(Box::new(ScriptedSignals::new(
                vec![token_signal(), token_signal_at(SECOND_MINT)],
                None,
            )));
        assert_eq!(
            dispatcher.dispatch(update("/p $syn", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        // Without menu storage the choices are listed as text instead.
        let reply = only_sent(&dispatcher.actions.0);
        assert!(reply.reply_markup.is_none());
        assert!(reply.text.contains("Libra Finance"));
        assert_eq!(reply.text.matches("Synthetic Token").count(), 2);
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|entry| entry.starts_with("market selection storage unavailable"))
        );
    }

    #[test]
    fn address_queries_with_several_dex_pairs_list_every_distinct_contract() {
        let address = token_signal().token.address;
        let mut distinct = dispatcher().with_token_signal_source(Box::new(ScriptedSignals::new(
            vec![token_signal(), token_signal_at(SECOND_MINT)],
            None,
        )));
        assert_eq!(
            distinct.dispatch(update(&format!("/p {address}"), Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let reply = only_sent(&distinct.actions.0);
        assert_eq!(reply.text.matches("Synthetic Token").count(), 2);
        assert!(reply.reply_markup.is_none());

        // Two pairs of one contract collapse into a single candidate, which
        // is not a unique signal, so the reply says nothing was found.
        let mut duplicated = dispatcher().with_token_signal_source(Box::new(ScriptedSignals::new(
            vec![token_signal(), token_signal()],
            None,
        )));
        assert_eq!(
            duplicated.dispatch(update(&format!("/p {address}"), Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&duplicated.actions.0),
            [format!("I could not find data for {address}")]
        );
    }

    #[test]
    fn provider_chart_tokens_fall_back_to_the_dex_card() {
        let mut dispatcher = dispatcher()
            .with_market_price_source(market_prices(MarketPriceLoad {
                chart: Some(libra_chart(Some(token_signal().token))),
                ..quote_load("LIBRA: 0.007 USD", false)
            }))
            .with_token_signal_source(Box::new(ScriptedSignals::new(
                Vec::new(),
                Some(token_signal()),
            )));
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [TelegramAction::SendPhoto { photo, reply_to_message_id: Some(MessageId(7)), .. }]
                if photo.as_ref() == b"scripted-card"
        ));
        assert_eq!(dispatcher.state.outgoing.len(), 1);
    }

    #[test]
    fn unresolved_token_queries_report_the_missing_quote_in_each_locale() {
        for (language, expected) in [
            ("en", "I could not get a quote for $foo"),
            ("es", "No pude conseguir una cotización para $foo"),
        ] {
            let mut dispatcher = dispatcher()
                .with_market_price_source(market_prices(quote_load("", true)))
                .with_token_signal_source(Box::new(ScriptedSignals::new(Vec::new(), None)));
            assert_eq!(
                dispatcher.dispatch(update("/p $foo", Some(language))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(sent_texts(&dispatcher.actions.0), [expected]);
        }
        let address = token_signal().token.address;
        let mut missing =
            dispatcher().with_token_signal_source(Box::new(ScriptedSignals::new(Vec::new(), None)));
        assert_eq!(
            missing.dispatch(update(&format!("/p {address}"), Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&missing.actions.0),
            [format!("No encontré datos para {address}")]
        );
    }

    struct PollSink {
        stored: Result<bool, String>,
        calls: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
    }

    impl crate::dispatcher::PollUpdateSink for PollSink {
        fn record_answer(&mut self, answer: &bot_core::polls::PollAnswer) -> Result<bool, String> {
            self.calls
                .borrow_mut()
                .push(format!("answer:{}", answer.voter_id));
            self.stored.clone()
        }

        fn apply_state(&mut self, state: bot_core::polls::PollState) -> Result<bool, String> {
            self.calls
                .borrow_mut()
                .push(format!("state:{}", state.closed));
            self.stored.clone()
        }
    }

    #[test]
    fn poll_updates_are_stored_only_for_known_polls() {
        let poll_event = |answer: bool, payload: Value| {
            let payload = payload.as_object().cloned().unwrap_or_default();
            IncomingUpdate {
                update_id: 1,
                event: if answer {
                    IncomingEvent::PollAnswer(payload)
                } else {
                    IncomingEvent::Poll(payload)
                },
            }
        };
        let answer =
            json!({"poll_id": "p1", "user": {"id": 7, "first_name": "Ana"}, "option_ids": [0]});
        let state = json!({"id": "p1", "options": [], "is_closed": true});

        let mut without_sink = dispatcher();
        assert_eq!(
            without_sink.dispatch(poll_event(true, answer.clone())),
            Ok(DispatchOutcome::Unsupported)
        );

        for (stored, expected) in [
            (Ok(true), DispatchOutcome::Handled),
            (Ok(false), DispatchOutcome::Unsupported),
            (
                Err("synthetic Redis failure".to_owned()),
                DispatchOutcome::Unsupported,
            ),
        ] {
            let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let failed = stored.is_err();
            let mut dispatcher = dispatcher().with_poll_update_sink(Box::new(PollSink {
                stored,
                calls: std::rc::Rc::clone(&calls),
            }));
            assert_eq!(
                dispatcher.dispatch(poll_event(true, answer.clone())),
                Ok(expected)
            );
            assert_eq!(
                dispatcher.dispatch(poll_event(false, state.clone())),
                Ok(expected)
            );
            assert_eq!(*calls.borrow(), ["answer:7", "state:true"]);
            assert_eq!(
                dispatcher
                    .state_diagnostics
                    .iter()
                    .any(|line| line.contains("synthetic Redis failure")),
                failed
            );
            assert_eq!(
                dispatcher.dispatch(poll_event(true, json!({"poll_id": "p1"}))),
                Ok(DispatchOutcome::Unsupported)
            );
            assert_eq!(calls.borrow().len(), 2);
        }
    }

    #[test]
    fn long_laughs_are_not_treated_as_solana_addresses() {
        let mut dispatcher =
            dispatcher().with_token_signal_source(Box::new(ScriptedSignals::new(Vec::new(), None)));
        let _ = dispatcher.dispatch(update("JAKAJAJJAJAJAJAJAJAJJAJAJAJAJAJAJJAJA", Some("es")));
        let texts = sent_texts(&dispatcher.actions.0);
        assert_eq!(
            texts
                .iter()
                .find(|text| text.starts_with("No encontré datos para")),
            None
        );
    }

    #[test]
    fn empty_multi_asset_quotes_are_localized_per_request() {
        let mut dispatcher =
            dispatcher().with_market_price_source(market_prices(MarketPriceLoad {
                chart: Some(libra_chart(None)),
                ..quote_load(" ", false)
            }));
        assert_eq!(
            dispatcher.dispatch(update("/p libra, eth", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&dispatcher.actions.0),
            [
                "No pude conseguir una cotización para LIBRA\nNo pude conseguir una cotización para LIBRA"
            ]
        );
        let mut english = configured(ChatConfig::default(), Actions::default())
            .with_market_price_source(market_prices(MarketPriceLoad {
                chart: Some(libra_chart(None)),
                ..quote_load("", false)
            }));
        assert_eq!(
            english.dispatch(update("/p libra, eth", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&english.actions.0),
            ["I could not get a quote for LIBRA\nI could not get a quote for LIBRA"]
        );
    }

    #[test]
    fn voice_durations_reach_ai_and_summary_turns() {
        let (mut source, (prepared, _ignored, _deliveries)) =
            ai_source(Ok(AiPreparation::reply("respuesta", None)));
        source.summary_preparation = Some(Ok(AiPreparation::reply("resumen", None)));
        let mut dispatcher = dispatcher().with_ai_conversation_source(Box::new(source));
        for text in ["qué dice este audio", "/summary"] {
            let voice = message_update(text, None, |message| {
                message.audio_media_kind = Some("voice".to_owned());
                message.audio_duration_seconds = Some(3);
            });
            assert_eq!(dispatcher.dispatch(voice), Ok(DispatchOutcome::Handled));
        }
        let durations = prepared
            .borrow()
            .iter()
            .map(|input| input.audio_duration_seconds)
            .collect::<Vec<_>>();
        assert_eq!(durations, [Some(3.0), Some(3.0)]);
    }

    #[test]
    fn rejected_live_updates_still_deliver_the_reply() {
        for (text, diagnostic, reply) in [
            (
                "synthetic question",
                "AI Telegram live update failed; continuing so response delivery can retry",
                "respuesta",
            ),
            (
                "/summary",
                "summary Telegram live update failed; continuing so response delivery can retry",
                "resumen",
            ),
        ] {
            let (mut source, _observations) =
                ai_source(Ok(AiPreparation::reply("respuesta", None)));
            source.tokens = vec!["borrador".to_owned()];
            source.summary_preparation = Some(Ok(AiPreparation::reply("resumen", None)));
            // The thinking status is rejected; the reply is still sent once.
            let mut dispatcher = configured(
                ChatConfig::default(),
                failing_actions(ActionKind::SendMessage, 1, "synthetic draft failure", false),
            )
            .with_ai_conversation_source(Box::new(source));
            assert_eq!(
                dispatcher.dispatch(update(text, None)),
                Ok(DispatchOutcome::Handled)
            );
            assert!(
                dispatcher
                    .state_diagnostics()
                    .iter()
                    .any(|entry| entry == diagnostic)
            );
            assert_eq!(sent_texts(&dispatcher.actions.0), [reply]);
        }
    }

    #[test]
    fn token_cards_report_missing_history_and_failed_photos() {
        let no_sender = {
            let mut message = incoming_message("/p $syn", Some("en"));
            message.sender_id = None;
            message
        };
        let mut anonymous = dispatcher()
            .with_token_signal_source(Box::new(ScriptedSignals::new(vec![token_signal()], None)));
        assert_eq!(
            anonymous.dispatch_asset_prices(
                &no_sender,
                "$syn",
                bot_core::market_prices::MarketPriceCommand::Unified,
                bot_core::locale::Locale::En,
                1_672_531_200,
            ),
            Ok(Some(DispatchOutcome::Unsupported))
        );

        for (query, reported) in [
            ("$syn", "query=syn"),
            (
                "https://www.coingecko.com/en/coins/synthetic-token",
                "query=synthetic-token",
            ),
        ] {
            let mut no_history = dispatcher()
                .with_token_signal_source(Box::new(NoHistorySignals(Some(token_signal()))));
            assert_eq!(
                no_history.dispatch(update(&format!("/p {query}"), Some("es"))),
                Ok(DispatchOutcome::Handled)
            );
            let reply = only_sent(&no_history.actions.0);
            assert!(
                reply
                    .text
                    .ends_with("\nNo tengo historial de 24h; te dejo la cotización")
            );
            assert_eq!(
                reply.parse_mode,
                Some(bot_core::telegram_actions::ParseMode::Html)
            );
            assert!(
                no_history
                    .state_diagnostics()
                    .iter()
                    .any(|entry| entry
                        == &format!("token signal photo failed chat_id=-42 {reported}"))
            );
        }

        let mut failed_photo = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            attempt_actions(Attempt::Fail, ActionScript::photo),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_token_signal_source(Box::new(ScriptedSignals::new(vec![token_signal()], None)));
        assert_eq!(
            failed_photo.dispatch(update("/p $syn", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(sent_messages(&failed_photo.actions.0).len(), 1);
        assert!(failed_photo.state_diagnostics().iter().any(|entry| {
            entry.starts_with("token signal photo delivery failed chat_id=-42 signal_id=")
        }));
    }

    fn requester_state(last_refresh_at: Option<i64>) -> SignalState {
        let signal = token_signal();
        SignalState {
            chart_period: None,
            chat_id: "-42".to_owned(),
            message_id: 7,
            source_message_id: 6,
            requester_id: "88".to_owned(),
            chain_id: signal.token.chain_id,
            network: signal.token.network,
            tag: signal.token.tag,
            address: signal.token.address,
            last_refresh_at,
        }
    }

    fn silent_signal_callback(data: &str) -> IncomingUpdate {
        callback_update_with_context(data, json!(-42), "private", 7, Some(88), Some("en"), None)
    }

    #[test]
    fn token_signal_callbacks_without_an_id_complete_without_toasts() {
        let scripted = |state: Result<Option<SignalState>, String>| {
            let mut source = ScriptedSignals::new(Vec::new(), Some(token_signal()));
            source.state = state;
            source
        };
        let mut unreadable = dispatcher().with_token_signal_source(Box::new(scripted(Err(
            "synthetic state failure".to_owned(),
        ))));
        assert_eq!(
            unreadable.dispatch(silent_signal_callback("sig:ref:abc")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(unreadable.actions.0.is_empty());
        assert_eq!(
            unreadable.state_diagnostics(),
            ["token signal state read failed chat_id=-42 signal_id=abc: synthetic state failure"]
        );

        let mut other_owner = requester_state(None);
        other_owner.requester_id = "7".to_owned();
        let mut cooldown = requester_state(Some(1_672_531_195));
        cooldown.chart_period = Some("7d".to_owned());
        for state in [None, Some(other_owner), Some(cooldown)] {
            let mut dispatcher =
                dispatcher().with_token_signal_source(Box::new(scripted(Ok(state))));
            assert_eq!(
                dispatcher.dispatch(silent_signal_callback("sig:ref:abc")),
                Ok(DispatchOutcome::Handled)
            );
            assert!(dispatcher.actions.0.is_empty());
        }

        let mut no_data_source = scripted(Ok(Some(requester_state(None))));
        no_data_source.token = None;
        let mut render_failure = scripted(Ok(Some(requester_state(None))));
        render_failure.photo = Err("synthetic render failure".to_owned());
        for source in [no_data_source, render_failure] {
            let mut dispatcher = dispatcher().with_token_signal_source(Box::new(source));
            assert_eq!(
                dispatcher.dispatch(silent_signal_callback("sig:ref:abc")),
                Ok(DispatchOutcome::Handled)
            );
            assert!(dispatcher.actions.0.is_empty());
        }

        let mut refreshed = dispatcher()
            .with_token_signal_source(Box::new(scripted(Ok(Some(requester_state(None))))));
        assert_eq!(
            refreshed.dispatch(silent_signal_callback("sig:ref:abc")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            refreshed.actions.0.as_slice(),
            [TelegramAction::EditMessagePhoto {
                message_id: MessageId(7),
                ..
            }]
        ));

        let mut deleting = scripted(Ok(Some(requester_state(None))));
        deleting.clear_error = Some("synthetic clear failure".to_owned());
        let mut deleted = dispatcher().with_token_signal_source(Box::new(deleting));
        assert_eq!(
            deleted.dispatch(silent_signal_callback("sig:del:abc")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            deleted.actions.0,
            [TelegramAction::DeleteMessage {
                chat_id: ChatId(-42),
                message_id: MessageId(7),
            }]
        );
        assert_eq!(
            deleted.state_diagnostics(),
            ["token signal state cleanup failed: synthetic clear failure"]
        );
    }

    #[test]
    fn token_signal_callbacks_reject_malformed_ids_chats_and_failed_edits() {
        let owned = || {
            let mut source = ScriptedSignals::new(Vec::new(), Some(token_signal()));
            source.state = Ok(Some(requester_state(None)));
            source
        };
        let mut missing_id = dispatcher().with_token_signal_source(Box::new(owned()));
        assert_eq!(
            missing_id.dispatch(callback_update("sig:ref", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut foreign_chat = dispatcher().with_token_signal_source(Box::new(owned()));
        assert_eq!(
            foreign_chat.dispatch(callback_update_with_context(
                "sig:ref:abc",
                json!("channel"),
                "private",
                7,
                Some(88),
                Some("en"),
                Some("callback-1"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        for actions in [&missing_id.actions.0, &foreign_chat.actions.0] {
            assert_eq!(
                actions.as_slice(),
                [TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: None,
                    show_alert: false,
                }]
            );
        }

        let mut rejected_edit = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            attempt_actions(Attempt::Fail, ActionScript::edit),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_token_signal_source(Box::new(owned()));
        assert_eq!(
            rejected_edit.dispatch(callback_update("sig:ref:abc", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            rejected_edit.state_diagnostics(),
            ["token signal Telegram refresh failed chat_id=-42 signal_id=abc"]
        );
        assert!(matches!(
            rejected_edit.actions.0.as_slice(),
            [
                TelegramAction::EditMessagePhoto { .. },
                TelegramAction::AnswerCallback {
                    show_alert: true,
                    ..
                },
            ]
        ));

        assert_eq!(
            dispatcher().dispatch(callback_update("mkt:select:abc:0", "private", Some("en"))),
            Err(DispatchError::MissingService("market prices"))
        );
    }

    fn first_selection_callback() -> String {
        format!(
            "mkt:select:{}:0",
            market_selection_id(-42, 7, 88, 1_672_531_200, 0)
        )
    }

    fn token_chart_quote() -> MarketPriceLoad {
        MarketPriceLoad {
            chart: Some(libra_chart(Some(token_signal().token))),
            ..quote_load("LIBRA: 0.007 USD", false)
        }
    }

    #[test]
    fn selected_chart_tokens_fall_back_to_the_quote_when_no_card_is_sent() {
        let token_sources: [Option<Box<dyn TokenSignalSource>>; 3] = [
            Some(Box::new(NoHistorySignals(Some(token_signal())))),
            Some(Box::new(ScriptedSignals::new(Vec::new(), None))),
            None,
        ];
        for token_source in token_sources {
            let stored = Rc::new(RefCell::new(HashMap::new()));
            let mut dispatcher = dispatcher().with_market_price_source(selection_storage(
                market_selection_load(None),
                token_chart_quote(),
                &stored,
            ));
            if let Some(source) = token_source {
                dispatcher = dispatcher.with_token_signal_source(source);
            }
            assert_eq!(
                dispatcher.dispatch(update("/p libra", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher.dispatch(callback_update_for_message(
                    &first_selection_callback(),
                    "private",
                    Some("en"),
                    700,
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(sent_texts(&dispatcher.actions.0)[1..], ["LIBRA: 0.007 USD"]);
            assert!(stored.borrow().is_empty());
        }

        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut failed_photo = NativeDispatcher::new(
            Config {
                value: Ok(ChatConfig::default()),
                chat_ids: Vec::new(),
            },
            attempt_actions(Attempt::Fail, ActionScript::photo),
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
        .with_market_price_source(selection_storage(
            market_selection_load(None),
            token_chart_quote(),
            &stored,
        ))
        .with_token_signal_source(Box::new(ScriptedSignals::new(
            Vec::new(),
            Some(token_signal()),
        )));
        assert_eq!(
            failed_photo.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            failed_photo.dispatch(callback_update_for_message(
                &first_selection_callback(),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            failed_photo
                .state_diagnostics()
                .iter()
                .any(|entry| entry == "token chart photo delivery failed")
        );
        assert_eq!(
            sent_texts(&failed_photo.actions.0)[1..],
            ["LIBRA: 0.007 USD"]
        );
    }

    #[test]
    fn selected_quote_history_failures_are_diagnostic() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut dispatcher = broken_state_dispatcher(ChatConfig::default())
            .with_market_price_source(selection_storage(
                market_selection_load(None),
                market_candidate_quote(),
                &stored,
            ));
        assert_eq!(
            dispatcher.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.dispatch(callback_update_for_message(
                &first_selection_callback(),
                "private",
                Some("en"),
                700,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(
            dispatcher
                .state_diagnostics()
                .iter()
                .any(|entry| entry == "market callback history: synthetic outgoing failure")
        );
        assert_eq!(
            sent_texts(&dispatcher.actions.0)[1..],
            ["LIBRA: 0.007 USD (N/A 24h)"]
        );
    }

    #[test]
    fn selected_dex_candidate_without_data_is_localized() {
        let mut selection = market_selection_fixture(None);
        selection.candidates = vec![super::market_token_candidate(&token_signal(), None, None)];
        let stored = Rc::new(RefCell::new(HashMap::from([(
            "market_selection:dex-only".to_owned(),
            StoredMarketSelection {
                selection,
                chat_id: "-42".to_owned(),
                message_id: 7,
                source_message_id: Some(6),
                requester_id: 88,
                command: "unified".to_owned(),
            }
            .encode(),
        )])));
        let mut dispatcher = dispatcher()
            .with_market_price_source(selection_storage(
                market_selection_load(None),
                market_candidate_quote(),
                &stored,
            ))
            .with_token_signal_source(Box::new(ScriptedSignals::new(Vec::new(), None)));
        assert_eq!(
            dispatcher.dispatch(callback_update(
                "mkt:select:dex-only:0",
                "private",
                Some("es")
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&dispatcher.actions.0),
            ["No pude conseguir una cotización para SYN"]
        );
        // The menu is restored so the user can retry the same choice.
        assert_eq!(stored.borrow().len(), 1);
    }

    /// A sink that only implements `execute`, relying on every trait default.
    #[derive(Default)]
    struct PlainActions(Vec<TelegramAction>);

    impl ActionSink for PlainActions {
        type Error = &'static str;

        fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
            self.0.push(action);
            Ok(ActionReceipt {
                message_id: Some(MessageId(9)),
            })
        }
    }

    #[test]
    fn default_delivery_attempts_execute_the_action() {
        let mut sink = PlainActions::default();
        let message = || TelegramAction::SendMessage(SendMessage::new(ChatId(-42), "synthetic"));
        assert_eq!(sink.try_edit(message()), Ok(true));
        assert_eq!(sink.try_invoice(message()), Ok(true));
        assert_eq!(sink.try_animation(message()), Ok(true));
        let confirmed = Some(ActionReceipt {
            message_id: Some(MessageId(9)),
        });
        assert_eq!(sink.try_video(message()), Ok(confirmed));
        assert_eq!(sink.try_photo(message()), Ok(confirmed));
        assert_eq!(sink.0.len(), 5);
        assert!(!sink.is_permanent_failure(&"synthetic failure"));
    }

    fn configured(
        config: ChatConfig,
        actions: Actions,
    ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization> {
        NativeDispatcher::new(
            Config {
                value: Ok(config),
                chat_ids: Vec::new(),
            },
            actions,
            State::default(),
            values(),
            random(),
            authorization(),
            "@mybot",
        )
    }

    fn english() -> ChatConfig {
        ChatConfig {
            language: "en".to_owned(),
            ..ChatConfig::default()
        }
    }

    fn uncalled_back(data: &str, chat_id: Value, chat_type: &str) -> IncomingUpdate {
        callback_update_with_context(data, chat_id, chat_type, 7, Some(88), Some("en"), None)
    }

    #[test]
    fn trigger_words_sample_random_replies_and_propagate_random_failures() {
        let random_replies = ChatConfig {
            ai_random_replies: true,
            ..ChatConfig::default()
        };
        let (source, (prepared, ignored, _deliveries)) =
            ai_source(Ok(AiPreparation::reply("respuesta", None)));
        let mut dispatcher = configured(random_replies.clone(), Actions::default())
            .with_trigger_words(vec!["gordo".to_owned()])
            .with_ai_conversation_source(Box::new(source));
        dispatcher.random.choice_index = 0;
        assert_eq!(
            dispatcher.dispatch(group_link_update("che gordo, qué hacés")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(prepared.borrow().len(), 1);
        assert!(prepared.borrow()[0].spontaneous);
        assert!(ignored.borrow().is_empty());

        let (source, _observations) = ai_source(Ok(AiPreparation::silent()));
        let mut failing = configured(random_replies, Actions::default())
            .with_trigger_words(vec!["gordo".to_owned()])
            .with_ai_conversation_source(Box::new(source));
        failing.random.failing = true;
        assert_eq!(
            failing.dispatch(group_link_update("che gordo")),
            Err(DispatchError::Random("synthetic random failure"))
        );
        assert!(failing.actions.0.is_empty());
    }

    #[test]
    fn provider_chart_tokens_without_cards_fall_back_or_fail_loudly() {
        let chart_quote = || {
            market_prices(MarketPriceLoad {
                chart: Some(libra_chart(Some(token_signal().token))),
                ..quote_load("LIBRA: 0.007 USD", false)
            })
        };
        let mut unknown_token = dispatcher()
            .with_market_price_source(chart_quote())
            .with_token_signal_source(Box::new(ScriptedSignals::new(Vec::new(), None)));
        assert_eq!(
            unknown_token.dispatch(update("/p libra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&unknown_token.actions.0),
            ["LIBRA: 0.007 USD\nChart unavailable. Try again later"]
        );
        assert!(
            unknown_token
                .state_diagnostics()
                .iter()
                .any(|entry| entry == "market chart unavailable or undelivered: LIBRA")
        );

        let rejecting = || {
            failing_actions(
                ActionKind::SendMessage,
                usize::MAX,
                "synthetic send failure",
                false,
            )
        };
        let mut chart_card = configured(ChatConfig::default(), rejecting())
            .with_market_price_source(chart_quote())
            .with_token_signal_source(Box::new(NoHistorySignals(Some(token_signal()))));
        assert_eq!(
            chart_card.dispatch(update("/p libra", Some("en"))),
            Err(DispatchError::Action("synthetic send failure"))
        );
        let mut direct_card = configured(ChatConfig::default(), rejecting())
            .with_token_signal_source(Box::new(NoHistorySignals(Some(token_signal()))));
        assert_eq!(
            direct_card.dispatch(update("/p $syn", Some("en"))),
            Err(DispatchError::Action("synthetic send failure"))
        );
        let mut stock = configured(ChatConfig::default(), rejecting())
            .with_market_price_source(market_prices(quote_load("AAPL: 1 USD", false)));
        assert_eq!(
            stock.dispatch(update("/s aapl", Some("en"))),
            Err(DispatchError::Action("synthetic send failure"))
        );
    }

    fn task_dispatcher(
        source: Box<dyn super::ScheduledTaskSource>,
        is_admin: bool,
    ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Authorization> {
        let mut dispatcher = dispatcher().with_scheduled_task_source(source);
        dispatcher.authorization.is_admin = is_admin;
        dispatcher
    }

    fn tasks(owner: i64) -> Result<Box<Tasks>, TaskStateError> {
        Ok(Box::new(Tasks {
            lists: vec![vec![scheduled_task(owner)?]],
            cancellations: Rc::new(RefCell::new(Vec::new())),
        }))
    }

    #[test]
    fn task_callbacks_without_callback_ids_finish_silently() -> Result<(), TaskStateError> {
        let chat = || json!(-42);
        let cases: Vec<(Box<dyn super::ScheduledTaskSource>, &str, &str, bool)> = vec![
            (
                Box::new(FallibleTasks {
                    list_result: Err("synthetic list failure".to_owned()),
                    cancel_result: Ok(true),
                }),
                "task:del:task0001",
                "private",
                true,
            ),
            (tasks(88)?, "task:del:task0404", "private", true),
            (tasks(55)?, "task:del:task0001", "group", false),
            (
                Box::new(FallibleTasks {
                    list_result: Ok(vec![scheduled_task(88)?]),
                    cancel_result: Ok(false),
                }),
                "task:del:task0001",
                "private",
                true,
            ),
            (
                Box::new(FallibleTasks {
                    list_result: Ok(vec![scheduled_task(88)?]),
                    cancel_result: Err("synthetic cancel failure".to_owned()),
                }),
                "task:del:task0001",
                "private",
                true,
            ),
        ];
        for (source, data, chat_type, is_admin) in cases {
            let mut dispatcher = task_dispatcher(source, is_admin);
            assert_eq!(
                dispatcher.dispatch(uncalled_back(data, chat(), chat_type)),
                Ok(DispatchOutcome::Handled)
            );
            assert!(dispatcher.actions.0.is_empty(), "{data} in {chat_type}");
        }

        let mut deleted = task_dispatcher(tasks(88)?, true);
        assert_eq!(
            deleted.dispatch(uncalled_back("task:del:task0001", chat(), "private")),
            Ok(DispatchOutcome::Handled)
        );
        // The list is refreshed in place without a toast.
        assert!(matches!(
            deleted.actions.0.as_slice(),
            [TelegramAction::EditMessage {
                message_id: MessageId(7),
                ..
            }]
        ));
        Ok(())
    }

    #[test]
    fn task_callbacks_in_non_numeric_chats_only_acknowledge() -> Result<(), TaskStateError> {
        for data in ["task:close", "task:page:1", "task:view:task0001"] {
            let mut dispatcher = task_dispatcher(tasks(88)?, true);
            assert_eq!(
                dispatcher.dispatch(callback_update_with_context(
                    data,
                    json!("channel"),
                    "private",
                    7,
                    Some(88),
                    Some("en"),
                    Some("callback-1"),
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher.actions.0,
                [TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: None,
                    show_alert: false,
                }],
                "{data}"
            );
        }
        Ok(())
    }

    #[test]
    fn help_and_payment_callbacks_reject_non_numeric_chats() {
        for data in [
            "help:close",
            "topup:p50",
            "chg:88:2:o:29:-180",
            "cfg:page:home",
        ] {
            let mut dispatcher = dispatcher();
            assert_eq!(
                dispatcher.dispatch(callback_update_with_context(
                    data,
                    json!("channel"),
                    "private",
                    7,
                    Some(88),
                    Some("en"),
                    Some("callback-1"),
                )),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher.actions.0,
                [TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: None,
                    show_alert: false,
                }],
                "{data}"
            );
        }
        let mut topup = dispatcher();
        assert_eq!(
            topup.dispatch(callback_update_with_context(
                "topup:p50",
                json!("channel"),
                "private",
                7,
                Some(88),
                Some("en"),
                None,
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            topup.state_diagnostics(),
            ["invalid top-up callback chat id"]
        );
    }

    #[test]
    fn topup_callbacks_without_ids_and_repeated_taps_stay_quiet() {
        let mut unknown = dispatcher();
        assert_eq!(
            unknown.dispatch(uncalled_back("topup:missing", json!(-42), "private")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(unknown.actions.0.is_empty());

        let stored = Rc::new(RefCell::new(HashMap::new()));
        let mut repeated =
            dispatcher().with_market_price_source(Box::new(SelectableMarketPrices {
                initial: market_selection_load(None),
                candidate: market_candidate_quote(),
                stored: Rc::clone(&stored),
                selected: Rc::new(RefCell::new(Vec::new())),
            }));
        // Without a callback id the invoice is sent but nothing is answered.
        assert_eq!(
            repeated.dispatch(uncalled_back("topup:p50", json!(88), "private")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            repeated.actions.0.as_slice(),
            [TelegramAction::SendInvoice { .. }]
        ));
        assert_eq!(
            repeated.dispatch(uncalled_back("topup:p50", json!(88), "private")),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(repeated.actions.0.len(), 1);
        assert_eq!(
            repeated.dispatch(callback_update_with_context(
                "topup:p50",
                json!(88),
                "private",
                7,
                Some(88),
                Some("es"),
                Some("callback-2"),
            )),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            repeated.actions.0.last(),
            Some(TelegramAction::AnswerCallback { text: Some(text), show_alert: false, .. })
                if text == "Ya te dejé la factura más arriba"
        ));
    }

    #[test]
    fn refused_topup_invoice_reports_a_failed_claim_release() {
        let mut dispatcher = configured(
            english(),
            Actions::scripted(ActionScript {
                invoice: Attempt::Skip,
                ..ActionScript::default()
            }),
        )
        .with_market_price_source(Box::new(ScriptedTakeMarketPrices {
            load: Ok(None),
            takes: RefCell::new(VecDeque::from([
                Err("synthetic release failure".to_owned()),
            ])),
            saves: RefCell::new(VecDeque::new()),
            candidate: market_candidate_quote(),
        }));
        assert_eq!(
            dispatcher.dispatch(callback_update("topup:p50", "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(dispatcher.state_diagnostics().iter().any(|entry| entry
            == "topup invoice claim release failed chat_id=-42 key=topup_invoice:88:p50: synthetic release failure"));
        assert!(matches!(
            dispatcher.actions.0.as_slice(),
            [
                TelegramAction::SendInvoice { .. },
                TelegramAction::AnswerCallback {
                    show_alert: true,
                    ..
                },
            ]
        ));
    }

    fn charge_page(groups: bool) -> ChargeHistoryPage {
        ChargeHistoryPage {
            groups: if groups {
                vec![ChargeHistoryGroup {
                    cursor_id: 20,
                    created_at: "2026-08-26T17:00:00+00:00".to_owned(),
                    entries: vec![ChargeHistoryEntry {
                        id: 20,
                        event_type: "ai_settlement_result".to_owned(),
                        metadata: json!({"charged_credit_units_total":4}),
                    }],
                }]
            } else {
                Vec::new()
            },
            has_newer: false,
            has_older: false,
            newer_cursor: Some(20),
            older_cursor: Some(20),
        }
    }

    fn charge_histories(result: Result<ChargeHistoryPage, String>) -> Box<ChargeHistories> {
        Box::new(ChargeHistories {
            result,
            calls: Rc::new(RefCell::new(Vec::new())),
        })
    }

    #[test]
    fn charge_history_callbacks_without_ids_skip_every_toast() {
        let data = "chg:88:2:o:29:-180";
        let mut foreign =
            dispatcher().with_charge_history_source(charge_histories(Ok(charge_page(true))));
        assert_eq!(
            foreign.dispatch(uncalled_back("chg:55:2:o:29:-180", json!(-42), "private")),
            Ok(DispatchOutcome::Handled)
        );
        let mut failed = configured(english(), Actions::default())
            .with_charge_history_source(charge_histories(Err("synthetic read failure".to_owned())));
        assert_eq!(
            failed.dispatch(uncalled_back(data, json!(-42), "private")),
            Ok(DispatchOutcome::Handled)
        );
        let mut empty = configured(english(), Actions::default())
            .with_charge_history_source(charge_histories(Ok(charge_page(false))));
        assert_eq!(
            empty.dispatch(uncalled_back(data, json!(-42), "private")),
            Ok(DispatchOutcome::Handled)
        );
        let mut rejected = configured(
            ChatConfig::default(),
            attempt_actions(Attempt::Fail, ActionScript::edit),
        )
        .with_charge_history_source(charge_histories(Ok(charge_page(true))));
        assert_eq!(
            rejected.dispatch(uncalled_back(data, json!(-42), "private")),
            Ok(DispatchOutcome::Handled)
        );
        for dispatcher in [&foreign, &failed, &empty] {
            assert!(dispatcher.actions.0.is_empty());
        }
        assert!(matches!(
            rejected.actions.0.as_slice(),
            [TelegramAction::EditMessage { .. }]
        ));
        let mut edited =
            dispatcher().with_charge_history_source(charge_histories(Ok(charge_page(true))));
        assert_eq!(
            edited.dispatch(uncalled_back(data, json!(-42), "private")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            edited.actions.0.as_slice(),
            [TelegramAction::EditMessage { .. }]
        ));
    }

    #[test]
    fn charge_history_callback_toasts_are_localized() {
        let data = "chg:88:2:o:29:-180";
        let mut failed = configured(english(), Actions::default())
            .with_charge_history_source(charge_histories(Err("synthetic read failure".to_owned())));
        assert_eq!(
            failed.dispatch(callback_update(data, "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut empty = configured(english(), Actions::default())
            .with_charge_history_source(charge_histories(Ok(charge_page(false))));
        assert_eq!(
            empty.dispatch(callback_update(data, "private", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut rejected = configured(
            ChatConfig::default(),
            attempt_actions(Attempt::Fail, ActionScript::edit),
        )
        .with_charge_history_source(charge_histories(Ok(charge_page(true))));
        assert_eq!(
            rejected.dispatch(callback_update(data, "private", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        for (actions, expected) in [
            (
                &failed.actions.0,
                "I could not load your spending. Try again",
            ),
            (&empty.actions.0, "There is no more spending to show"),
            (
                &rejected.actions.0,
                "Se trabó leyendo tus gastos. Probá de nuevo",
            ),
        ] {
            assert!(matches!(
                actions.last(),
                Some(TelegramAction::AnswerCallback { text: Some(text), .. }) if text == expected
            ));
        }
        assert_eq!(
            dispatcher().dispatch(callback_update(data, "private", Some("en"))),
            Err(DispatchError::MissingService("charge history"))
        );
    }

    #[test]
    fn config_callbacks_report_denials_invalid_values_and_rejected_edits() {
        let mut denied = configured(english(), Actions::default());
        denied.authorization.is_admin = false;
        assert_eq!(
            denied.dispatch(callback_update("cfg:random:on", "group", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            denied.actions.0.as_slice(),
            [TelegramAction::AnswerCallback { text: Some(text), show_alert: true, .. }]
                if text == "Only group admins can use this command"
        ));
        let mut admin = configured(english(), Actions::default());
        assert_eq!(
            admin.dispatch(callback_update("cfg:random:on", "group", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            admin.authorization.checks,
            [("-42".to_owned(), "88".to_owned())]
        );
        assert!(matches!(
            admin.actions.0.as_slice(),
            [
                TelegramAction::EditMessage {
                    message_id: MessageId(7),
                    ..
                },
                TelegramAction::AnswerCallback { .. },
            ]
        ));
        assert!(admin.config.chat_ids.contains(&"set:-42".to_owned()));
        let mut quiet_denial = configured(english(), Actions::default());
        quiet_denial.authorization.is_admin = false;
        assert_eq!(
            quiet_denial.dispatch(uncalled_back("cfg:random:on", json!(-42), "group")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(quiet_denial.actions.0.is_empty());

        for (data, expected) in [
            (
                "cfg:timezone:abc",
                "Invalid timezone callback value chat_id=-42 value=abc",
            ),
            (
                "cfg:creditless:-5",
                "Invalid creditless callback value chat_id=-42 value=-5",
            ),
        ] {
            let mut invalid = dispatcher();
            assert_eq!(
                invalid.dispatch(callback_update(data, "private", Some("en"))),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(invalid.state_diagnostics(), [expected]);
            assert!(
                invalid
                    .config
                    .chat_ids
                    .iter()
                    .all(|id| !id.starts_with("set:"))
            );
        }

        let mut rejected = configured(
            ChatConfig::default(),
            attempt_actions(Attempt::Fail, ActionScript::edit),
        );
        assert_eq!(
            rejected.dispatch(callback_update("cfg:random:on", "private", Some("en"))),
            Err(DispatchError::Action("synthetic edit failure"))
        );
        assert!(matches!(
            rejected.actions.0.last(),
            Some(TelegramAction::AnswerCallback { text: None, .. })
        ));
    }

    #[test]
    fn command_handlers_guard_incomplete_messages_and_missing_ai() {
        let mut anonymous = incoming_message("/transcribe", None);
        anonymous.sender_id = None;
        let config = ChatConfig::default();
        let locale = bot_core::locale::Locale::Es;
        let mut dispatcher = dispatcher();
        assert_eq!(
            dispatcher.dispatch_ai_message(&anonymous, &config, locale, 1, "/ask", "hola"),
            Ok(DispatchOutcome::Unsupported)
        );
        assert_eq!(
            dispatcher.dispatch_media_command(&anonymous, &config, locale, 1, "/transcribe", ""),
            Ok(DispatchOutcome::Unsupported)
        );
        assert_eq!(
            dispatcher.dispatch_summary_command(&anonymous, &config, locale, 1, "/summary", ""),
            Ok(DispatchOutcome::Unsupported)
        );
        for command in ["/transcribe", "/summary"] {
            assert_eq!(
                dispatcher.dispatch(update(command, None)),
                Err(DispatchError::MissingService("AI conversation"))
            );
        }
        assert!(dispatcher.actions.0.is_empty());
    }

    #[test]
    fn spanish_summary_failure_is_localized() {
        let (mut source, _observations) = ai_source(Ok(AiPreparation::silent()));
        source.summary_preparation = Some(Err("synthetic summary failure".to_owned()));
        let mut dispatcher = dispatcher().with_ai_conversation_source(Box::new(source));
        assert_eq!(
            dispatcher.dispatch(update("/resumen", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&dispatcher.actions.0),
            ["Pensando.", "No pude generar el resumen. Probá de nuevo"]
        );
    }

    #[test]
    fn task_list_command_survives_a_failed_listing() {
        let mut dispatcher = dispatcher().with_scheduled_task_source(Box::new(FallibleTasks {
            list_result: Err("synthetic list failure".to_owned()),
            cancel_result: Ok(true),
        }));
        assert_eq!(
            dispatcher.dispatch(update("/tareas", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            dispatcher.state_diagnostics(),
            ["scheduled task list command chat_id=-42: synthetic list failure"]
        );
        assert_eq!(sent_messages(&dispatcher.actions.0).len(), 1);
    }

    #[test]
    fn billing_and_admin_commands_localize_failures_and_unavailability() {
        let mut unavailable = dispatcher().with_billing_available(false);
        assert_eq!(
            unavailable.dispatch(update("/balance", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(sent_messages(&unavailable.actions.0).len(), 1);

        let mut balance = dispatcher().with_balance_source(Box::new(Balances {
            result: Err("synthetic balance failure".to_owned()),
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            balance.dispatch(update("/balance", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut charges = configured(english(), Actions::default())
            .with_charge_history_source(charge_histories(Err("synthetic read failure".to_owned())));
        assert_eq!(
            charges.dispatch(update("/charges", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut transfer =
            configured(english(), Actions::default()).with_transfer_sink(Box::new(Transfers {
                result: Err("synthetic transfer failure".to_owned()),
                calls: Rc::new(RefCell::new(Vec::new())),
            }));
        let group_transfer = message_update("/transfer 1", Some("en"), |message| {
            message.chat_type = Some("group".to_owned());
        });
        assert_eq!(
            transfer.dispatch(group_transfer),
            Ok(DispatchOutcome::Handled)
        );
        let mut mint = configured(english(), Actions::default())
            .with_admin_user_id(Some(88))
            .with_admin_credit_sink(Box::new(AdminCredits {
                result: Err("synthetic mint failure".to_owned()),
                calls: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            mint.dispatch(update("/printcredits 1", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let creditlog = |result| {
            configured(english(), Actions::default())
                .with_admin_user_id(Some(88))
                .with_admin_creditlog_source(Box::new(AdminCreditLogs {
                    result,
                    calls: Rc::new(RefCell::new(Vec::new())),
                }))
        };
        let mut empty_log = creditlog(Ok(Vec::new()));
        assert_eq!(
            empty_log.dispatch(update("/creditlog", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut failed_log = creditlog(Err("synthetic log failure".to_owned()));
        assert_eq!(
            failed_log.dispatch(update("/creditlog", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut foreign_log = creditlog(Ok(Vec::new()));
        foreign_log.admin_user_id = Some(1);
        assert_eq!(
            foreign_log.dispatch(update("/creditlog", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let texts = [
            &balance,
            &charges,
            &transfer,
            &mint,
            &empty_log,
            &failed_log,
        ]
        .map(|dispatcher| last_sent(&dispatcher.actions.0).text.clone());
        assert_eq!(
            texts,
            [
                "Se trabó leyendo tu saldo. Probá de nuevo",
                "I could not load your spending. Try again",
                "The transfer failed. Try again",
                "I could not mint credits. Try again",
                "There are no recent AI settlements",
                "I could not load the credit log. Try again",
            ]
        );
        // Only the admin may read the log; others get a plain refusal.
        assert_eq!(sent_messages(&foreign_log.actions.0).len(), 1);
    }

    #[test]
    fn market_data_failures_are_localized_in_english() {
        let mut bcra =
            configured(english(), Actions::default()).with_bcra_source(Box::new(BcraVariables {
                result: BcraLoad {
                    text: None,
                    diagnostics: Vec::new(),
                },
                calls: Rc::new(RefCell::new(Vec::new())),
            }));
        assert_eq!(
            bcra.dispatch(update("/bcra", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut dollar = configured(english(), Actions::default()).with_dollar_market_source(
            Box::new(DollarMarket {
                result: DollarMarketLoad {
                    text: None,
                    diagnostics: Vec::new(),
                },
                calls: Rc::new(RefCell::new(Vec::new())),
            }),
        );
        assert_eq!(
            dollar.dispatch(update("/dolar", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            [
                only_sent(&bcra.actions.0).text.as_str(),
                only_sent(&dollar.actions.0).text.as_str(),
            ],
            [
                "I could not load the BCRA variables. Try again later",
                "I could not load dollar rates. Try again later",
            ]
        );
    }

    #[test]
    fn numeric_price_queries_use_the_provider_text_directly() {
        let mut missing = dispatcher();
        assert_eq!(
            missing.dispatch(update("/p 123", Some("en"))),
            Err(DispatchError::MissingService("market prices"))
        );
        let mut menu = configured(english(), Actions::default())
            .with_market_price_source(market_prices(market_selection_load(None)));
        assert_eq!(
            menu.dispatch(update("/p 123", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert!(only_sent(&menu.actions.0).text.contains("Libra Finance"));
        let mut empty = dispatcher().with_market_price_source(market_prices(quote_load("", true)));
        assert_eq!(
            empty.dispatch(update("/p 123", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut english_empty = configured(english(), Actions::default())
            .with_market_price_source(market_prices(quote_load("", true)));
        assert_eq!(
            english_empty.dispatch(update("/p 123", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        let mut quoted =
            dispatcher().with_market_price_source(market_prices(quote_load("123 quoted", false)));
        assert_eq!(
            quoted.dispatch(update("/p 123", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            [
                only_sent(&empty.actions.0).text.as_str(),
                only_sent(&english_empty.actions.0).text.as_str(),
                only_sent(&quoted.actions.0).text.as_str(),
            ],
            [
                "No pude conseguir una cotización. Probá más tarde",
                "I could not get a quote. Try again later",
                "123 quoted",
            ]
        );
    }

    #[test]
    fn out_of_range_random_choices_are_invariant_failures() {
        let mut greeting = dispatcher().with_greeting_pool_source(Box::new(GreetingPools {
            result: GreetingPoolLoad {
                urls: vec!["https://example.test/greeting.gif".to_owned()],
                diagnostics: Vec::new(),
            },
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        greeting.random.choice_index = 5;
        assert_eq!(
            greeting.dispatch(update("/gm", None)),
            Err(DispatchError::Invariant(
                "random greeting index out of bounds"
            ))
        );
        let mut choice = dispatcher();
        choice.random.choice_index = 5;
        assert_eq!(
            choice.dispatch(update("/random alpha, beta", None)),
            Err(DispatchError::Invariant("random reply index out of bounds"))
        );
        assert!(greeting.actions.0.is_empty() && choice.actions.0.is_empty());
    }

    #[test]
    fn missing_prices_and_malformed_random_requests_are_explained() {
        let mut satoshi = dispatcher().with_bitcoin_price_source(Box::new(BitcoinPrices {
            results: vec![Ok(None)],
            calls: Rc::new(RefCell::new(Vec::new())),
        }));
        assert_eq!(
            satoshi.dispatch(update("/sats", Some("en"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(sent_messages(&satoshi.actions.0).len(), 1);
        let mut random = dispatcher();
        assert_eq!(
            random.dispatch(update("/random", Some("es"))),
            Ok(DispatchOutcome::Handled)
        );
        assert_eq!(
            sent_texts(&random.actions.0),
            [
                "Mandate algo como 'pizza, carne, sushi' o '1-10', boludo, no me hagas laburar al pedo"
            ]
        );
    }

    fn malformed_pre_checkout(id: Value, payload: &str) -> IncomingUpdate {
        IncomingUpdate {
            update_id: 104,
            event: IncomingEvent::PreCheckoutQuery(Map::from_iter([
                ("id".to_owned(), id),
                ("from".to_owned(), json!("malformed")),
                ("invoice_payload".to_owned(), json!(payload)),
                ("currency".to_owned(), json!("XTR")),
                ("total_amount".to_owned(), json!(25)),
            ])),
        }
    }

    #[test]
    fn malformed_pre_checkout_answers_only_string_query_ids() {
        let mut numeric_id = dispatcher();
        assert_eq!(
            numeric_id.dispatch(malformed_pre_checkout(json!(5), "topup:p50:42:es")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(numeric_id.actions.0.is_empty());
        assert!(numeric_id.state_diagnostics()[0].starts_with("invalid pre-checkout query:"));
        let mut spanish = dispatcher();
        assert_eq!(
            spanish.dispatch(malformed_pre_checkout(json!("query-1"), "topup:p50:42:es")),
            Ok(DispatchOutcome::Handled)
        );
        assert!(matches!(
            spanish.actions.0.as_slice(),
            [TelegramAction::AnswerPreCheckout { ok: false, error_message: Some(text), .. }]
                if text == "Ese pago vino raro y no te lo pude validar"
        ));
    }

    /// Members a test group's bot has seen write, as stored in Redis, or a
    /// failed lookup.
    struct Members(Result<Vec<(String, String)>, String>);

    impl crate::chat_members_tool::ChatMemberSource for Members {
        fn members(&mut self, chat_id: &str) -> Result<Vec<(String, String)>, String> {
            assert_eq!(chat_id, "-42");
            self.0.clone()
        }
    }

    /// Lemon (77) wrote last with @Lemon, which 66 used before; @nameless
    /// (55) has no first name; one stored id isn't a number; the sender (88)
    /// is @tester, admin 99 is @boss, and anonymous admins show up as a bot.
    /// @helper (44) was stored as a bot and @the_abbot (33) is a person.
    fn known_members() -> Members {
        let member = |user_id: &str, first_name: &str, username: &str, last_seen: i64| {
            let payload = serde_json::json!({
                "schema_version": 1,
                "first_name": first_name,
                "username": username,
                "last_seen": last_seen,
            });
            (user_id.to_owned(), payload.to_string())
        };
        Members(Ok(vec![
            member("66", "Old", "lemon", 1),
            member("77", "Lemon", "Lemon", 2),
            member("55", "", "nameless", 3),
            member("x", "Broken", "broken", 4),
            member("88", "Synthetic", "tester", 5),
            member("99", "Boss", "boss", 6),
            member("1087968824", "Group", "GroupAnonymousBot", 7),
            member("33", "Abbot", "the_abbot", 8),
            (
                "44".to_owned(),
                r#"{"schema_version":1,"first_name":"Helper","username":"helper","last_seen":9,"is_bot":true}"#
                    .to_owned(),
            ),
        ]))
    }

    /// A member picked from Telegram's mention list, who has no username.
    fn picked(text: &str, user_id: i64, is_bot: bool) -> bot_core::telegram_input::TextMention {
        bot_core::telegram_input::TextMention {
            text: text.to_owned(),
            user_id,
            first_name: text.to_owned(),
            username: String::new(),
            is_bot,
        }
    }

    mod chat_bans {
        use super::*;
        use crate::dispatcher::ChatBanStore;
        use bot_core::chat_bans::BannedUser;

        type BanRows = Rc<RefCell<Vec<(i64, BannedUser, i64)>>>;

        /// In-memory bans keyed like the PostgreSQL table, with every lookup
        /// recorded and an optional failure for all calls.
        #[derive(Default)]
        struct Bans {
            rows: BanRows,
            checks: Rc<RefCell<Vec<(i64, i64)>>>,
            error: Option<String>,
        }

        impl Bans {
            fn fail(&self) -> Result<(), String> {
                self.error.clone().map_or(Ok(()), Err)
            }
        }

        impl ChatBanStore for Bans {
            fn is_banned(&mut self, chat_id: i64, user_id: i64) -> Result<bool, String> {
                self.checks.borrow_mut().push((chat_id, user_id));
                self.fail()?;
                Ok(self
                    .rows
                    .borrow()
                    .iter()
                    .any(|(chat, user, _)| *chat == chat_id && user.user_id == user_id))
            }

            fn ban(
                &mut self,
                chat_id: i64,
                user_id: i64,
                name: &str,
                banned_by: i64,
            ) -> Result<bool, String> {
                self.fail()?;
                if self.is_banned(chat_id, user_id)? {
                    return Ok(false);
                }
                let user = BannedUser {
                    user_id,
                    name: name.to_owned(),
                };
                self.rows.borrow_mut().push((chat_id, user, banned_by));
                Ok(true)
            }

            fn unban(&mut self, chat_id: i64, user_id: i64) -> Result<bool, String> {
                self.fail()?;
                let mut rows = self.rows.borrow_mut();
                let before = rows.len();
                rows.retain(|(chat, user, _)| !(*chat == chat_id && user.user_id == user_id));
                Ok(rows.len() < before)
            }

            fn list(&mut self, chat_id: i64) -> Result<Vec<BannedUser>, String> {
                self.fail()?;
                Ok(self
                    .rows
                    .borrow()
                    .iter()
                    .filter(|(chat, _, _)| *chat == chat_id)
                    .map(|(_, user, _)| user.clone())
                    .collect())
            }
        }

        struct Admins(Vec<i64>);

        impl GroupAuthorizer for Admins {
            fn authorize(&mut self, _chat_id: &str, user_id: &str) -> GroupAuthorizationDecision {
                GroupAuthorizationDecision {
                    is_admin: self.0.iter().any(|admin| admin.to_string() == user_id),
                    diagnostics: vec![format!("checked {user_id}")],
                }
            }
        }

        fn build(
            language: &str,
            admins: &[i64],
            bans: Option<Bans>,
            ai: Option<AiSource>,
        ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Admins> {
            let dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig {
                        language: language.to_owned(),
                        ..ChatConfig::default()
                    }),
                    chat_ids: Vec::new(),
                },
                Actions::default(),
                State::default(),
                values(),
                random(),
                Admins(admins.to_vec()),
                "@mybot",
            );
            let dispatcher = match bans {
                Some(bans) => dispatcher.with_ban_store(Box::new(bans)),
                None => dispatcher,
            };
            match ai {
                Some(ai) => dispatcher.with_ai_conversation_source(Box::new(ai)),
                None => dispatcher,
            }
        }

        fn group(text: &str, edit: impl FnOnce(&mut IncomingMessage)) -> IncomingUpdate {
            message_update(text, None, |message| {
                message.chat_type = Some("supergroup".to_owned());
                edit(message);
            })
        }

        /// A group message replying to member 77, named Ana.
        fn replying(text: &str) -> IncomingUpdate {
            group(text, |message| {
                message.has_reply = true;
                message.replied_sender_id = Some(UserId(77));
                message.replied_sender_first_name = Some("Ana".to_owned());
            })
        }

        fn silent_ai() -> AiSource {
            ai_source(Ok(AiPreparation::silent())).0
        }

        fn replies(
            dispatcher: &mut NativeDispatcher<Config, Actions, State, Values, Samples, Admins>,
            update: IncomingUpdate,
        ) -> String {
            dispatcher.actions.0.clear();
            assert_eq!(dispatcher.dispatch(update), Ok(DispatchOutcome::Handled));
            let message = first_sent(&dispatcher.actions.0);
            assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
            message.text.clone()
        }

        #[test]
        fn admins_ban_list_and_unban_replied_members() {
            let bans = Bans::default();
            let rows = Rc::clone(&bans.rows);
            let mut dispatcher = build("es", &[88], Some(bans), Some(silent_ai()));

            assert_eq!(
                replies(&mut dispatcher, group("/vetados", |_| {})),
                "No ignoro a nadie en este grupo"
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/vetar")),
                "Listo, Ana ya no puede usarme en este grupo"
            );
            assert_eq!(
                rows.borrow().as_slice(),
                [(
                    -42,
                    BannedUser {
                        user_id: 77,
                        name: "Ana".to_owned()
                    },
                    88
                )]
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/ban@mybot")),
                "Ana ya estaba ignorado en este grupo"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/ignored", |_| {})),
                "Ignorados en este grupo\n- Ana"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/ignorados", |_| {})),
                "Ignorados en este grupo\n- Ana"
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/desvetar")),
                "Listo, Ana puede volver a usarme"
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/unignore")),
                "Ana no estaba ignorado"
            );
            assert!(rows.borrow().is_empty());
            assert!(
                dispatcher
                    .state_diagnostics()
                    .contains(&"checked 88".to_owned())
            );
        }

        #[test]
        fn ban_commands_refuse_private_chats_non_admins_and_admin_targets() {
            for (language, group_only, admins_only, admin_target) in [
                (
                    "es",
                    "Esto funciona solo en grupos",
                    "Este comando es solo para admins del grupo",
                    "A los admins no los puedo ignorar",
                ),
                (
                    "en",
                    "This only works in groups",
                    "Only group admins can use this command",
                    "I can't ignore admins",
                ),
            ] {
                let bans = Bans::default();
                let rows = Rc::clone(&bans.rows);
                let mut dispatcher = build(language, &[88, 77], Some(bans), Some(silent_ai()));
                assert_eq!(replies(&mut dispatcher, update("/ban", None)), group_only);
                assert_eq!(replies(&mut dispatcher, replying("/ignore")), admin_target);
                assert!(rows.borrow().is_empty());

                let mut member = build(language, &[], Some(Bans::default()), Some(silent_ai()));
                for command in ["/ignore", "/unignore", "/ignored"] {
                    assert_eq!(replies(&mut member, replying(command)), admins_only);
                }
                assert!(
                    member
                        .state_diagnostics()
                        .contains(&"Unauthorized ban attempt chat_id=-42 user_id=88".to_owned())
                );
            }
        }

        #[test]
        fn bare_ban_commands_in_groups_are_left_to_moderation_bots() {
            let (ai, (prepared, ignored, _)) = ai_source(Ok(AiPreparation::silent()));
            let bans = Bans::default();
            let rows = Rc::clone(&bans.rows);
            let mut dispatcher = build("es", &[88], Some(bans), Some(ai));
            for text in ["/ban", "/BAN @lemon", "/unban", "/bans", "/banned"] {
                assert_eq!(
                    dispatcher.dispatch(replying(text)),
                    Ok(DispatchOutcome::Handled),
                    "{text}"
                );
            }
            assert!(dispatcher.actions.0.is_empty());
            assert!(rows.borrow().is_empty());
            assert!(prepared.borrow().is_empty());
            assert_eq!(ignored.borrow().len(), 5);
            assert!(!dispatcher.listen_only);

            assert_eq!(
                replies(&mut dispatcher, replying("/ban@MyBot")),
                "Listo, Ana ya no puede usarme en este grupo"
            );
            assert_eq!(rows.borrow().len(), 1);
        }

        #[test]
        fn ban_planner_replies_reach_the_chat() {
            let mut dispatcher = build("en", &[88], Some(Bans::default()), Some(silent_ai()));
            assert_eq!(
                replies(&mut dispatcher, group("/ignore", |_| {})),
                "Reply to someone's message with /ignore, or send /ignore @username, and I'll ignore them"
            );
            let bot = group("/ignore", |message| {
                message.has_reply = true;
                message.replied_sender_id = Some(UserId(5));
                message.replied_sender_is_bot = true;
            });
            assert_eq!(replies(&mut dispatcher, bot), "I can't ignore bots");
        }

        #[test]
        fn storage_failures_reply_and_leave_a_diagnostic() {
            for (language, saved, listed) in [
                (
                    "es",
                    "No pude guardar el cambio, probá de nuevo",
                    "No pude cargar la lista, probá de nuevo",
                ),
                (
                    "en",
                    "I couldn't save that, try again",
                    "I couldn't load the list, try again",
                ),
            ] {
                let bans = Bans {
                    error: Some("synthetic database failure".to_owned()),
                    ..Bans::default()
                };
                let mut dispatcher = build(language, &[88], Some(bans), Some(silent_ai()));
                for (command, expected, diagnostic) in [
                    ("/ignore", saved, "chat ban chat_id=-42 user_id=77"),
                    ("/unignore", saved, "chat unban chat_id=-42 user_id=77"),
                    ("/ignored", listed, "chat ban list chat_id=-42"),
                ] {
                    assert_eq!(replies(&mut dispatcher, replying(command)), expected);
                    assert!(
                        dispatcher
                            .state_diagnostics()
                            .contains(&format!("{diagnostic}: synthetic database failure")),
                        "{command}"
                    );
                }
            }
        }

        #[test]
        fn ban_commands_need_the_ban_store() {
            let mut dispatcher = build("es", &[88], None, Some(silent_ai()));
            assert_eq!(
                dispatcher.dispatch(group("/vetados", |_| {})),
                Err(DispatchError::MissingService("chat bans"))
            );
            assert!(dispatcher.actions.0.is_empty());
        }

        fn banned_88() -> Bans {
            let bans = Bans::default();
            bans.rows.borrow_mut().push((
                -42,
                BannedUser {
                    user_id: 88,
                    name: "Synthetic".to_owned(),
                },
                1,
            ));
            bans
        }

        #[test]
        fn banned_members_are_ignored_but_kept_in_chat_history() {
            let (ai, (prepared, ignored, _)) = ai_source(Ok(AiPreparation::silent()));
            let bans = banned_88();
            let checks = Rc::clone(&bans.checks);
            let mut dispatcher = build("es", &[], Some(bans), Some(ai));
            for text in ["@mybot hola", "/ask hola", "/time", "/ban", "$btc"] {
                assert_eq!(
                    dispatcher.dispatch(group(text, |_| {})),
                    Ok(DispatchOutcome::Handled),
                    "{text}"
                );
            }
            assert!(dispatcher.actions.0.is_empty());
            assert!(prepared.borrow().is_empty());
            let ignored = ignored.borrow();
            assert_eq!(ignored.len(), 5);
            assert_eq!(ignored[0].message_text, "@mybot hola");
            assert_eq!(ignored[1].message_text, "hola");
            assert_eq!(checks.borrow()[0], (-42, 88));
            assert!(!dispatcher.listen_only);

            // The same member is untouched in private chats and other groups.
            let other_group = group("/time", |message| message.chat_id = Some(ChatId(-43)));
            for update in [update("/time", None), other_group] {
                dispatcher.actions.0.clear();
                assert_eq!(dispatcher.dispatch(update), Ok(DispatchOutcome::Handled));
                assert_eq!(dispatcher.actions.0.len(), 1);
            }
        }

        #[test]
        fn banned_members_are_dropped_without_an_ai_service() {
            let mut dispatcher = build("es", &[], Some(banned_88()), None);
            assert_eq!(
                dispatcher.dispatch(group("/time", |_| {})),
                Ok(DispatchOutcome::Handled)
            );
            assert!(dispatcher.actions.0.is_empty());
        }

        #[test]
        fn admins_and_failed_lookups_are_never_silenced() {
            let mut admin = build("es", &[88], Some(banned_88()), Some(silent_ai()));
            assert_eq!(
                admin.dispatch(group("/time", |_| {})),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(admin.actions.0.len(), 1);
            assert!(admin.state_diagnostics().contains(&"checked 88".to_owned()));

            let failing = Bans {
                error: Some("synthetic database failure".to_owned()),
                ..banned_88()
            };
            let mut dispatcher = build("es", &[], Some(failing), Some(silent_ai()));
            assert_eq!(
                dispatcher.dispatch(group("/time", |_| {})),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(dispatcher.actions.0.len(), 1);
            assert!(dispatcher.state_diagnostics().contains(
                &"chat ban check chat_id=-42 user_id=88: synthetic database failure".to_owned()
            ));
        }

        #[test]
        fn banned_members_buttons_only_stop_the_spinner() {
            let mut dispatcher = build("es", &[], Some(banned_88()), Some(silent_ai()));
            assert_eq!(
                dispatcher.dispatch(callback_update("help:home", "supergroup", None)),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(
                dispatcher.actions.0,
                [TelegramAction::AnswerCallback {
                    callback_id: "callback-1".to_owned(),
                    text: None,
                    show_alert: false,
                }]
            );

            // Private buttons skip the lookup and work as usual.
            dispatcher.actions.0.clear();
            assert_eq!(
                dispatcher.dispatch(callback_update("help:home", "private", None)),
                Ok(DispatchOutcome::Handled)
            );
            assert_eq!(dispatcher.actions.0.len(), 2);
        }
        #[test]
        fn admins_ban_and_unban_members_by_username() {
            let bans = Bans::default();
            let rows = Rc::clone(&bans.rows);
            let mut dispatcher = build("es", &[88, 99], Some(bans), Some(silent_ai()))
                .with_member_source(Box::new(known_members()));
            assert_eq!(
                replies(&mut dispatcher, group("/ignore @LEMON", |_| {})),
                "Listo, Lemon ya no puede usarme en este grupo"
            );
            assert_eq!(rows.borrow()[0].1.user_id, 77);
            // The username wins over the replied member, and without a first
            // name the reply uses the username.
            assert_eq!(
                replies(&mut dispatcher, replying("/ban@mybot @nameless")),
                "Listo, @nameless ya no puede usarme en este grupo"
            );
            assert_eq!(rows.borrow()[1].1.user_id, 55);
            assert_eq!(
                replies(&mut dispatcher, group("/unignore @lemon", |_| {})),
                "Listo, Lemon puede volver a usarme"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/ignore @boss", |_| {})),
                "A los admins no los puedo ignorar"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/ignore @tester", |_| {})),
                "No podés ignorarte"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/ignore @groupanonymousbot", |_| {})),
                "A los bots no los puedo ignorar"
            );
            // A mistyped username while replying never bans the replied member.
            for username in ["nadie", "broken", "lemon."] {
                assert_eq!(
                    replies(&mut dispatcher, replying(&format!("/ignore @{username}"))),
                    format!(
                        "No sé quién es @{username}: tiene que haber escrito en el grupo, o respondé a un mensaje suyo"
                    )
                );
            }
            // Listing ignores a username.
            assert_eq!(
                replies(&mut dispatcher, group("/ignorados @nadie", |_| {})),
                "Ignorados en este grupo\n- @nameless"
            );
            // Bots go by the flag Telegram sent, not by how the username ends.
            assert_eq!(
                replies(&mut dispatcher, group("/ignore @helper", |_| {})),
                "A los bots no los puedo ignorar"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/ignore @the_abbot", |_| {})),
                "Listo, Abbot ya no puede usarme en este grupo"
            );
            assert_eq!(rows.borrow()[1].1.user_id, 33);
        }

        #[test]
        fn admins_ban_members_picked_from_the_mention_list() {
            let bans = Bans::default();
            let rows = Rc::clone(&bans.rows);
            // Picked members need no member lookup.
            let mut dispatcher = build("es", &[88, 99], Some(bans), Some(silent_ai()));
            let picking =
                |text: &str, mention| group(text, |message| message.text_mentions = vec![mention]);
            assert_eq!(
                replies(
                    &mut dispatcher,
                    picking("/ignore Lemon Pie", picked("Lemon Pie", 77, false))
                ),
                "Listo, Lemon Pie ya no puede usarme en este grupo"
            );
            assert_eq!(rows.borrow()[0].1.user_id, 77);
            // The picked member wins over the replied one.
            let mut reply = replying("/unignore Lemon Pie");
            if let IncomingEvent::Message(message) = &mut reply.event {
                message.text_mentions = vec![picked("Lemon Pie", 66, false)];
            }
            replies(&mut dispatcher, reply);
            assert_eq!(rows.borrow()[0].1.user_id, 77);
            assert_eq!(
                replies(
                    &mut dispatcher,
                    picking("/unignore Lemon Pie", picked("Lemon Pie", 77, false))
                ),
                "Listo, Lemon Pie puede volver a usarme"
            );
            assert!(rows.borrow().is_empty());
            for (mention, expected) in [
                (
                    picked("Helper", 44, true),
                    "A los bots no los puedo ignorar",
                ),
                (
                    picked("Boss", 99, false),
                    "A los admins no los puedo ignorar",
                ),
                (picked("Synthetic", 88, false), "No podés ignorarte"),
            ] {
                let text = format!("/ignore {}", mention.text);
                assert_eq!(replies(&mut dispatcher, picking(&text, mention)), expected);
            }
        }

        #[test]
        fn usernames_the_bot_cannot_look_up_get_a_reply_saying_so() {
            let unknown = "I don't know who @lemon is: they need to have written in the group, or reply to one of their messages";
            let mut failing = build("en", &[88], Some(Bans::default()), Some(silent_ai()))
                .with_member_source(Box::new(Members(Err("synthetic redis failure".to_owned()))));
            assert_eq!(
                replies(&mut failing, replying("/ignore @lemon")),
                "I couldn't look up @lemon. Try again, or reply to one of their messages"
            );
            assert_eq!(failing.actions.0.len(), 1);
            assert!(
                failing
                    .state_diagnostics()
                    .contains(&"chat members chat_id=-42: synthetic redis failure".to_owned())
            );
            let mut without = build("en", &[88], Some(Bans::default()), Some(silent_ai()));
            assert_eq!(
                replies(&mut without, group("/unignore @lemon", |_| {})),
                unknown
            );

            // Members still get the admins-only reply, and private chats the
            // groups-only one.
            let mut member = build("es", &[], Some(Bans::default()), Some(silent_ai()))
                .with_member_source(Box::new(known_members()));
            assert_eq!(
                replies(&mut member, group("/ignore @lemon", |_| {})),
                "Este comando es solo para admins del grupo"
            );
            assert_eq!(
                replies(&mut member, update("/ban @lemon", None)),
                "Esto funciona solo en grupos"
            );
        }
    }

    mod chat_limits {
        use super::*;
        use crate::dispatcher::ChatLimitStore;
        use bot_core::chat_limits::LimitedUser;

        type LimitRows = Rc<RefCell<Vec<(i64, LimitedUser, i64)>>>;

        /// In-memory limits keyed like the PostgreSQL table, as (chat, member,
        /// set by), with an optional failure for every call.
        #[derive(Default)]
        struct Limits {
            rows: LimitRows,
            error: Option<String>,
        }

        impl ChatLimitStore for Limits {
            fn hourly_limit(&mut self, chat_id: i64, user_id: i64) -> Result<Option<i64>, String> {
                self.error.clone().map_or(Ok(()), Err)?;
                Ok(self
                    .rows
                    .borrow()
                    .iter()
                    .find(|(chat, user, _)| *chat == chat_id && user.user_id == user_id)
                    .map(|(_, user, _)| user.hourly_limit))
            }

            fn set(
                &mut self,
                chat_id: i64,
                user_id: i64,
                name: &str,
                hourly_limit: i64,
                set_by: i64,
            ) -> Result<(), String> {
                self.clear(chat_id, user_id)?;
                let user = LimitedUser {
                    user_id,
                    name: name.to_owned(),
                    hourly_limit,
                };
                self.rows.borrow_mut().push((chat_id, user, set_by));
                Ok(())
            }

            fn clear(&mut self, chat_id: i64, user_id: i64) -> Result<bool, String> {
                self.error.clone().map_or(Ok(()), Err)?;
                let mut rows = self.rows.borrow_mut();
                let before = rows.len();
                rows.retain(|(chat, user, _)| !(*chat == chat_id && user.user_id == user_id));
                Ok(rows.len() < before)
            }

            fn list(&mut self, chat_id: i64) -> Result<Vec<LimitedUser>, String> {
                self.error.clone().map_or(Ok(()), Err)?;
                Ok(self
                    .rows
                    .borrow()
                    .iter()
                    .filter(|(chat, _, _)| *chat == chat_id)
                    .map(|(_, user, _)| user.clone())
                    .collect())
            }
        }

        fn limited(user_id: i64, hourly_limit: i64) -> Limits {
            let mut limits = Limits::default();
            assert_eq!(
                limits.set(-42, user_id, "Synthetic", hourly_limit, 1),
                Ok(())
            );
            limits
        }

        struct Admins(Vec<i64>);

        impl GroupAuthorizer for Admins {
            fn authorize(&mut self, _chat_id: &str, user_id: &str) -> GroupAuthorizationDecision {
                GroupAuthorizationDecision {
                    is_admin: self.0.iter().any(|admin| admin.to_string() == user_id),
                    diagnostics: vec![format!("checked {user_id}")],
                }
            }
        }

        /// The group's own limit is unlimited, as most groups leave it.
        fn build(
            language: &str,
            admins: &[i64],
            limits: Option<Limits>,
            ai: AiSource,
        ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Admins> {
            let dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig {
                        language: language.to_owned(),
                        creditless_user_hourly_limit: -1,
                        ..ChatConfig::default()
                    }),
                    chat_ids: Vec::new(),
                },
                Actions::default(),
                State::default(),
                values(),
                random(),
                Admins(admins.to_vec()),
                "@mybot",
            )
            .with_ai_conversation_source(Box::new(ai));
            match limits {
                Some(limits) => dispatcher.with_limit_store(Box::new(limits)),
                None => dispatcher,
            }
        }

        fn group(text: &str, edit: impl FnOnce(&mut IncomingMessage)) -> IncomingUpdate {
            message_update(text, None, |message| {
                message.chat_type = Some("supergroup".to_owned());
                edit(message);
            })
        }

        /// A group message replying to member 77, named Ana.
        fn replying(text: &str) -> IncomingUpdate {
            group(text, |message| {
                message.has_reply = true;
                message.replied_sender_id = Some(UserId(77));
                message.replied_sender_first_name = Some("Ana".to_owned());
            })
        }

        fn silent_ai() -> AiSource {
            ai_source(Ok(AiPreparation::silent())).0
        }

        fn replies(
            dispatcher: &mut NativeDispatcher<Config, Actions, State, Values, Samples, Admins>,
            update: IncomingUpdate,
        ) -> String {
            dispatcher.actions.0.clear();
            assert_eq!(dispatcher.dispatch(update), Ok(DispatchOutcome::Handled));
            let message = first_sent(&dispatcher.actions.0);
            assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
            message.text.clone()
        }

        #[test]
        fn admins_set_replace_and_clear_member_limits() {
            let limits = Limits::default();
            let rows = Rc::clone(&limits.rows);
            let mut dispatcher = build("es", &[88], Some(limits), silent_ai());

            let ana = |hourly_limit| {
                let user = LimitedUser {
                    user_id: 77,
                    name: "Ana".to_owned(),
                    hourly_limit,
                };
                vec![(-42, user, 88)]
            };

            assert_eq!(
                replies(&mut dispatcher, group("/limitados", |_| {})),
                "Nadie tiene un límite propio en este grupo"
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/limit 3")),
                "Listo, el grupo le paga a Ana hasta 3 mensajes por hora"
            );
            assert_eq!(*rows.borrow(), ana(3));
            assert!(
                dispatcher
                    .state_diagnostics()
                    .contains(&"checked 77".to_owned())
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/limitar@mybot 0")),
                "Listo, Ana ya no puede usar el saldo del grupo, solo el suyo"
            );
            assert_eq!(*rows.borrow(), ana(0));
            assert_eq!(
                replies(&mut dispatcher, group("/limited", |_| {})),
                "Límites propios en este grupo\n- Ana: solo sus créditos"
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/limit off")),
                "Listo, Ana vuelve al límite del grupo"
            );
            assert_eq!(
                replies(&mut dispatcher, replying("/limit off")),
                "Ana no tenía un límite propio"
            );
            assert!(rows.borrow().is_empty());
        }

        #[test]
        fn limit_command_refuses_private_chats_non_admins_and_admin_targets() {
            for (language, group_only, admins_only, admin_target) in [
                (
                    "es",
                    "Esto funciona solo en grupos",
                    "Este comando es solo para admins del grupo",
                    "A los admins no los puedo limitar",
                ),
                (
                    "en",
                    "This only works in groups",
                    "Only group admins can use this command",
                    "I can't limit admins",
                ),
            ] {
                let limits = Limits::default();
                let rows = Rc::clone(&limits.rows);
                let mut dispatcher = build(language, &[88, 77], Some(limits), silent_ai());
                assert_eq!(
                    replies(&mut dispatcher, update("/limit 3", None)),
                    group_only
                );
                assert_eq!(replies(&mut dispatcher, replying("/limit 3")), admin_target);
                assert!(rows.borrow().is_empty());

                let mut member = build(language, &[], Some(Limits::default()), silent_ai());
                for command in ["/limit 3", "/limit off", "/limitados"] {
                    assert_eq!(replies(&mut member, replying(command)), admins_only);
                }
                assert!(
                    member
                        .state_diagnostics()
                        .contains(&"Unauthorized limit attempt chat_id=-42 user_id=88".to_owned())
                );
            }
        }

        #[test]
        fn limit_planner_replies_reach_the_chat() {
            let mut dispatcher = build("en", &[88], Some(Limits::default()), silent_ai());
            let usage = "Reply to someone's message with /limit and how many messages per hour the group pays for, or send /limit @username and the number. Use off to remove it";
            assert_eq!(replies(&mut dispatcher, group("/limit 3", |_| {})), usage);
            assert_eq!(replies(&mut dispatcher, replying("/limit lots")), usage);
            let bot = group("/limit 3", |message| {
                message.has_reply = true;
                message.replied_sender_id = Some(UserId(5));
                message.replied_sender_is_bot = true;
            });
            assert_eq!(replies(&mut dispatcher, bot), "I can't limit bots");
            let own = group("/limit 3", |message| {
                message.has_reply = true;
                message.replied_sender_id = Some(UserId(88));
            });
            assert_eq!(replies(&mut dispatcher, own), "You can't limit yourself");
        }

        #[test]
        fn limit_storage_failures_reply_and_leave_a_diagnostic() {
            for (language, saved, listed) in [
                (
                    "es",
                    "No pude guardar el cambio, probá de nuevo",
                    "No pude cargar la lista, probá de nuevo",
                ),
                (
                    "en",
                    "I couldn't save that, try again",
                    "I couldn't load the list, try again",
                ),
            ] {
                let limits = Limits {
                    error: Some("synthetic database failure".to_owned()),
                    ..Limits::default()
                };
                let mut dispatcher = build(language, &[88], Some(limits), silent_ai());
                for (command, expected, diagnostic) in [
                    ("/limit 3", saved, "chat limit chat_id=-42 user_id=77"),
                    ("/limit off", saved, "chat unlimit chat_id=-42 user_id=77"),
                    ("/limited", listed, "chat limit list chat_id=-42"),
                ] {
                    assert_eq!(replies(&mut dispatcher, replying(command)), expected);
                    assert!(
                        dispatcher
                            .state_diagnostics()
                            .contains(&format!("{diagnostic}: synthetic database failure")),
                        "{command}"
                    );
                }
            }
        }

        #[test]
        fn limit_command_needs_the_limit_store() {
            let mut dispatcher = build("es", &[88], None, silent_ai());
            assert_eq!(
                dispatcher.dispatch(replying("/limit 3")),
                Err(DispatchError::MissingService("chat limits"))
            );
            assert!(dispatcher.actions.0.is_empty());
        }

        #[test]
        fn own_limit_replaces_the_group_limit_on_every_paid_ai_path() {
            let (mut ai, (prepared, _, _)) = ai_source(Ok(AiPreparation::silent()));
            ai.media_preparation = Some(Err("synthetic media failure".to_owned()));
            ai.summary_preparation = Some(Err("synthetic summary failure".to_owned()));
            let mut dispatcher = build("es", &[], Some(limited(88, 2)), ai);
            for text in ["@mybot hola", "/transcribe", "/resumen"] {
                assert_eq!(
                    dispatcher.dispatch(group(text, |_| {})),
                    Ok(DispatchOutcome::Handled),
                    "{text}"
                );
            }
            let received = prepared
                .borrow()
                .iter()
                .map(|input| input.creditless_limit)
                .collect::<Vec<_>>();
            assert_eq!(received, [CreditlessLimit::Member(2); 3]);
        }

        /// The limit the AI turn receives for one message from member 88,
        /// with the diagnostics it left.
        fn received_limit(
            admins: &[i64],
            limits: Option<Limits>,
            update: IncomingUpdate,
        ) -> (i64, Vec<String>) {
            let (ai, (prepared, _, _)) = ai_source(Ok(AiPreparation::silent()));
            let mut dispatcher = build("es", admins, limits, ai);
            assert_eq!(dispatcher.dispatch(update), Ok(DispatchOutcome::Handled));
            // Every case here keeps the group's limit, so none is the member's own.
            let limit = prepared.borrow()[0].creditless_limit;
            assert!(matches!(limit, CreditlessLimit::Group(_)));
            (limit.hourly(), dispatcher.state_diagnostics().to_vec())
        }

        #[test]
        fn group_limit_stays_for_admins_other_members_private_chats_and_failed_lookups() {
            let mention = || group("@mybot hola", |_| {});
            let (limit, diagnostics) = received_limit(&[88], Some(limited(88, 2)), mention());
            assert_eq!(limit, -1);
            assert!(diagnostics.contains(&"checked 88".to_owned()));

            assert_eq!(received_limit(&[], Some(limited(77, 2)), mention()).0, -1);
            assert_eq!(received_limit(&[], None, mention()).0, -1);
            let private = update("hola", None);
            assert_eq!(received_limit(&[], Some(limited(88, 2)), private).0, -1);

            let failing = Limits {
                error: Some("synthetic database failure".to_owned()),
                ..limited(88, 2)
            };
            let (limit, diagnostics) = received_limit(&[], Some(failing), mention());
            assert_eq!(limit, -1);
            assert!(diagnostics.contains(
                &"chat limit check chat_id=-42 user_id=88: synthetic database failure".to_owned()
            ));
        }

        #[test]
        fn admins_limit_members_by_username() {
            let limits = Limits::default();
            let rows = Rc::clone(&limits.rows);
            let mut dispatcher = build("es", &[88], Some(limits), silent_ai())
                .with_member_source(Box::new(known_members()));
            assert_eq!(
                replies(&mut dispatcher, group("/limit @lemon 0", |_| {})),
                "Listo, Lemon ya no puede usar el saldo del grupo, solo el suyo"
            );
            assert_eq!(rows.borrow()[0].1.user_id, 77);
            assert_eq!(
                replies(&mut dispatcher, group("/limitar @Lemon 3", |_| {})),
                "Listo, el grupo le paga a Lemon hasta 3 mensajes por hora"
            );
            assert_eq!(rows.borrow()[0].1.hourly_limit, 3);
            assert_eq!(
                replies(&mut dispatcher, group("/limit @lemon off", |_| {})),
                "Listo, Lemon vuelve al límite del grupo"
            );
            assert!(rows.borrow().is_empty());
            assert_eq!(
                replies(&mut dispatcher, group("/limit @lemon", |_| {})),
                "Respondé al mensaje de alguien con /limitar y cuántos mensajes por hora le paga el grupo, o mandá /limitar @usuario y el número. Con off le sacás el límite"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/limit @nadie 3", |_| {})),
                "No sé quién es @nadie: tiene que haber escrito en el grupo, o respondé a un mensaje suyo"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/limited", |_| {})),
                "Nadie tiene un límite propio en este grupo"
            );
        }

        #[test]
        fn admins_limit_members_picked_from_the_mention_list() {
            let limits = Limits::default();
            let rows = Rc::clone(&limits.rows);
            let mut dispatcher = build("en", &[88], Some(limits), silent_ai());
            let picking = |text: &str| {
                group(text, |message| {
                    message.text_mentions = vec![picked("Lemon Pie", 77, false)];
                })
            };
            assert_eq!(
                replies(&mut dispatcher, picking("/limit Lemon Pie 3")),
                "Done, the group pays for up to 3 messages per hour from Lemon Pie"
            );
            assert_eq!(rows.borrow()[0].1.user_id, 77);
            assert_eq!(rows.borrow()[0].1.hourly_limit, 3);
            assert_eq!(
                replies(&mut dispatcher, picking("/limit Lemon Pie off")),
                "Done, Lemon Pie is back to the group's limit"
            );
            assert!(rows.borrow().is_empty());
        }
    }

    mod group_charges {
        use super::*;
        use crate::dispatcher::GroupSpendingSource;
        use bot_core::group_charges::GroupSpender;

        type Loads = Rc<RefCell<Vec<(i64, i64, usize)>>>;

        /// Fixed spenders, every load recorded, and an optional failure.
        #[derive(Default)]
        struct Spending {
            loads: Loads,
            spenders: Vec<GroupSpender>,
            error: Option<String>,
            max_days: Option<i64>,
        }

        impl GroupSpendingSource for Spending {
            fn load(
                &mut self,
                chat_id: i64,
                days: i64,
                limit: usize,
            ) -> Result<Vec<GroupSpender>, String> {
                self.loads.borrow_mut().push((chat_id, days, limit));
                self.error
                    .clone()
                    .map_or_else(|| Ok(self.spenders.clone()), Err)
            }

            fn max_days(&self) -> i64 {
                self.max_days.unwrap_or(30)
            }
        }

        struct Admins(Vec<i64>);

        impl GroupAuthorizer for Admins {
            fn authorize(&mut self, _chat_id: &str, user_id: &str) -> GroupAuthorizationDecision {
                GroupAuthorizationDecision {
                    is_admin: self.0.iter().any(|admin| admin.to_string() == user_id),
                    diagnostics: vec![format!("checked {user_id}")],
                }
            }
        }

        fn build(
            language: &str,
            admins: &[i64],
            spending: Option<Spending>,
        ) -> NativeDispatcher<Config, Actions, State, Values, Samples, Admins> {
            let dispatcher = NativeDispatcher::new(
                Config {
                    value: Ok(ChatConfig {
                        language: language.to_owned(),
                        ..ChatConfig::default()
                    }),
                    chat_ids: Vec::new(),
                },
                Actions::default(),
                State::default(),
                values(),
                random(),
                Admins(admins.to_vec()),
                "@mybot",
            );
            match spending {
                Some(spending) => dispatcher.with_group_spending_source(Box::new(spending)),
                None => dispatcher,
            }
        }

        fn group(text: &str) -> IncomingUpdate {
            message_update(text, None, |message| {
                message.chat_type = Some("supergroup".to_owned());
            })
        }

        fn replies(
            dispatcher: &mut NativeDispatcher<Config, Actions, State, Values, Samples, Admins>,
            update: IncomingUpdate,
        ) -> String {
            dispatcher.actions.0.clear();
            assert_eq!(dispatcher.dispatch(update), Ok(DispatchOutcome::Handled));
            let message = first_sent(&dispatcher.actions.0);
            assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
            message.text.clone()
        }

        #[test]
        fn admins_see_who_spent_the_groups_credits() {
            let spending = Spending {
                spenders: vec![GroupSpender {
                    user_id: 77,
                    name: "Ana".to_owned(),
                    credit_units: 123_450,
                    messages: 8,
                }],
                ..Spending::default()
            };
            let loads = Rc::clone(&spending.loads);
            let mut dispatcher = build("es", &[88], Some(spending));
            assert_eq!(
                replies(&mut dispatcher, group("/groupcharges")),
                "Créditos del grupo gastados en las últimas 24 horas\n- Ana: 1,234.50 (8 mensajes)"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/groupcharges@mybot 7")),
                "Créditos del grupo gastados en los últimos 7 días\n- Ana: 1,234.50 (8 mensajes)"
            );
            assert_eq!(
                replies(&mut dispatcher, group("/gastosgrupo 3")),
                "Créditos del grupo gastados en los últimos 3 días\n- Ana: 1,234.50 (8 mensajes)"
            );
            assert_eq!(
                loads.borrow().as_slice(),
                [(-42, 1, 10), (-42, 7, 10), (-42, 3, 10)]
            );

            // A shorter ledger retention caps the days.
            let short = Spending {
                max_days: Some(7),
                ..Spending::default()
            };
            let short_loads = Rc::clone(&short.loads);
            let mut short = build("en", &[88], Some(short));
            assert_eq!(
                replies(&mut short, group("/groupcharges 8")),
                "Send /groupcharges for the last day, or /groupcharges and a number of days, up to 7"
            );
            assert!(short_loads.borrow().is_empty());

            let mut empty = build("en", &[88], Some(Spending::default()));
            assert_eq!(
                replies(&mut empty, group("/groupcharges 30")),
                "Nobody spent the group's credits in the last 30 days"
            );
        }

        #[test]
        fn group_charges_refuse_private_chats_non_admins_and_bad_days() {
            for (language, group_only, admins_only, usage) in [
                (
                    "es",
                    "Esto funciona solo en grupos",
                    "Este comando es solo para admins del grupo",
                    "Mandá /gastosgrupo para el último día, o /gastosgrupo y una cantidad de días, hasta 30",
                ),
                (
                    "en",
                    "This only works in groups",
                    "Only group admins can use this command",
                    "Send /groupcharges for the last day, or /groupcharges and a number of days, up to 30",
                ),
            ] {
                let spending = Spending::default();
                let loads = Rc::clone(&spending.loads);
                let mut dispatcher = build(language, &[88], Some(spending));
                assert_eq!(
                    replies(&mut dispatcher, update("/groupcharges", None)),
                    group_only
                );
                assert_eq!(replies(&mut dispatcher, group("/groupcharges 31")), usage);
                assert!(loads.borrow().is_empty());

                let mut member = build(language, &[], Some(Spending::default()));
                assert_eq!(replies(&mut member, group("/groupcharges")), admins_only);
                assert!(member.state_diagnostics().contains(
                    &"Unauthorized group charges attempt chat_id=-42 user_id=88".to_owned()
                ));
            }
        }

        #[test]
        fn group_charges_failure_replies_and_leaves_a_diagnostic() {
            let spending = Spending {
                error: Some("synthetic database failure".to_owned()),
                ..Spending::default()
            };
            let mut dispatcher = build("en", &[88], Some(spending));
            assert_eq!(
                replies(&mut dispatcher, group("/groupcharges")),
                "I couldn't load the spending, try again"
            );
            assert!(
                dispatcher
                    .state_diagnostics()
                    .contains(&"group charges chat_id=-42: synthetic database failure".to_owned())
            );
        }

        #[test]
        fn group_charges_need_the_spending_source() {
            let mut dispatcher = build("es", &[88], None);
            assert_eq!(
                dispatcher.dispatch(group("/groupcharges")),
                Err(DispatchError::MissingService("group charges"))
            );
            assert!(dispatcher.actions.0.is_empty());
        }
    }
}
