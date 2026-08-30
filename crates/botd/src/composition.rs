//! Concrete adapter composition for the native Telegram runtime.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bot_adapters::billing_read::{BillingRepository, ChargeHistoryRow};
use bot_adapters::chat_config::{ChatConfigRepository, ChatConfigRepositoryError};
use bot_adapters::coinmarketcap::{
    BitcoinPriceOutcome, CoinMarketCapTransport, ReqwestCoinMarketCapTransport,
    TransportFailureKind as CoinMarketCapTransportFailureKind, fetch_bitcoin_price,
};
use bot_adapters::criptoya::{
    CriptoYaTransport, DollarQuotesOutcome, ExchangeQuotesOutcome, ExchangeSide,
    ReqwestCriptoYaTransport, RuloMarketOutcome,
    TransportFailureKind as CriptoYaTransportFailureKind, fetch_dollar_quotes,
    fetch_exchange_quotes, fetch_rulo_market,
};
use bot_adapters::giphy::{
    GiphyTransport, ReqwestGiphyTransport, TransportFailureKind as GiphyTransportFailureKind,
};
use bot_adapters::giphy_pool::{GiphyPoolCache, load_giphy_pool};
use bot_adapters::redis_chat_admin::{cache_chat_admin, get_cached_chat_admin};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};
use bot_adapters::redis_message_state::{RedisMessageState, RedisMessageStateError};
use bot_adapters::telegram_actions::{ActionError, ActionOutcome, execute_with};
use bot_adapters::telegram_chat_admin::lookup_chat_admin_with;
use bot_adapters::telegram_http::{
    ReqwestTelegramTransport, TelegramTransport, TransportFailureKind,
};
use bot_adapters::telegram_polling::{PollOutcome, PollingError, poll_once_with};
use bot_core::charge_history::{ChargeHistoryEntry, ChargeHistoryGroup, ChargeHistoryPage};
use bot_core::chat_config::ChatConfig;
use bot_core::command_state::{
    BOT_MESSAGE_METADATA_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT, CHAT_STATE_TTL_SECONDS,
    IncomingCommandWritePlan, OutgoingCommandWritePlan,
};
use bot_core::telegram_actions::TelegramAction;
use bot_core::telegram_commands::command_publication_actions;
use bot_core::telegram_payments::StarPaymentRecord;
use num_bigint::{BigInt, BigUint};
use thiserror::Error;

use crate::dispatcher::{
    ActionReceipt, ActionSink, AdminCreditLogSource, AdminCreditSink, BillingBalanceSource,
    BillingBalances, BillingTransferSink, BitcoinPriceSource, ChargeHistorySource,
    ChatConfigSource, DollarQuotesSource, GreetingPoolLoad, GreetingPoolSource,
    GroupAuthorizationDecision, GroupAuthorizer, MessageStateSink, NativeDispatcher, RandomSource,
    RuloInputLoad, RuloSource, RuntimeValues, StarPaymentReceipt, StarPaymentSink,
};
use crate::runtime::{PollingRuntime, UpdateSource};

impl AdminCreditSink for BillingRepository {
    fn mint(&mut self, user_id: i64, amount: i64) -> Result<i64, String> {
        let amount = i32::try_from(amount)
            .map_err(|_| "admin credit amount exceeds the persistent range".to_owned())?;
        self.mint_user_credits(user_id, amount, Some(user_id))
            .map_err(|error| error.to_string())
    }
}

impl AdminCreditLogSource for BillingRepository {
    fn load(
        &mut self,
        limit: usize,
    ) -> Result<Vec<bot_core::admin_commands::CreditLogEntry>, String> {
        let limit = i64::try_from(limit).map_err(|_| "creditlog limit is too large".to_owned())?;
        self.list_recent_ai_settlement_results(limit)
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|entry| bot_core::admin_commands::CreditLogEntry {
                        user_id: entry.user_id,
                        chat_id: entry.chat_id,
                        metadata: entry.metadata,
                        created_at: entry.created_at,
                    })
                    .collect()
            })
            .map_err(|error| error.to_string())
    }
}

struct CoinMarketCapBitcoinPriceSource<T> {
    transport: T,
    api_key: String,
}

impl<T: CoinMarketCapTransport> BitcoinPriceSource for CoinMarketCapBitcoinPriceSource<T> {
    fn price(&mut self, currency: &str) -> Result<Option<f64>, String> {
        match fetch_bitcoin_price(&self.transport, &self.api_key, currency) {
            BitcoinPriceOutcome::Price(price) => Ok(Some(price)),
            BitcoinPriceOutcome::Missing => Ok(None),
            BitcoinPriceOutcome::HttpError { status_code } => {
                Err(format!("CoinMarketCap returned HTTP {status_code}"))
            }
            BitcoinPriceOutcome::InvalidJson => {
                Err("CoinMarketCap returned invalid JSON".to_owned())
            }
            BitcoinPriceOutcome::TransportError(kind) => {
                Err(format!("CoinMarketCap transport failed: {kind:?}"))
            }
        }
    }
}

struct CriptoYaDollarQuotesSource<T> {
    transport: T,
}

impl<T: CriptoYaTransport> DollarQuotesSource for CriptoYaDollarQuotesSource<T> {
    fn devo_quotes(&mut self) -> Result<Option<bot_core::devo::DevoQuotes>, String> {
        match fetch_dollar_quotes(&self.transport) {
            DollarQuotesOutcome::Quotes(quotes) => Ok(Some(quotes)),
            DollarQuotesOutcome::Missing => Ok(None),
            DollarQuotesOutcome::HttpError { status_code } => {
                Err(format!("CriptoYa returned HTTP {status_code}"))
            }
            DollarQuotesOutcome::InvalidJson => Err("CriptoYa returned invalid JSON".to_owned()),
            DollarQuotesOutcome::TransportError(kind) => {
                Err(format!("CriptoYa transport failed: {kind:?}"))
            }
        }
    }
}

struct CriptoYaRuloSource<T> {
    transport: T,
}

fn exchange_failure(label: &str, outcome: ExchangeQuotesOutcome) -> String {
    match outcome {
        ExchangeQuotesOutcome::Quotes(_) => format!("{label} unexpectedly succeeded"),
        ExchangeQuotesOutcome::InvalidJson => format!("CriptoYa {label} returned invalid JSON"),
        ExchangeQuotesOutcome::HttpError { status_code } => {
            format!("CriptoYa {label} returned HTTP {status_code}")
        }
        ExchangeQuotesOutcome::TransportError(kind) => {
            format!("CriptoYa {label} transport failed: {kind:?}")
        }
    }
}

impl<T: CriptoYaTransport> RuloSource for CriptoYaRuloSource<T> {
    fn rulo_input(&mut self) -> Result<RuloInputLoad, String> {
        let mut input = match fetch_rulo_market(&self.transport) {
            RuloMarketOutcome::Input(input) => input,
            RuloMarketOutcome::InvalidJson => {
                return Err("CriptoYa dollar market returned invalid JSON".to_owned());
            }
            RuloMarketOutcome::HttpError { status_code } => {
                return Err(format!(
                    "CriptoYa dollar market returned HTTP {status_code}"
                ));
            }
            RuloMarketOutcome::TransportError(kind) => {
                return Err(format!("CriptoYa dollar market transport failed: {kind:?}"));
            }
        };
        let mut diagnostics = Vec::new();
        match fetch_exchange_quotes(&self.transport, "USD", ExchangeSide::Ask) {
            ExchangeQuotesOutcome::Quotes(quotes) => input.usd_to_usdt = quotes,
            failure => diagnostics.push(exchange_failure("USDT/USD", failure)),
        }
        match fetch_exchange_quotes(&self.transport, "ARS", ExchangeSide::Bid) {
            ExchangeQuotesOutcome::Quotes(quotes) => input.usdt_to_ars = quotes,
            failure => diagnostics.push(exchange_failure("USDT/ARS", failure)),
        }
        Ok(RuloInputLoad { input, diagnostics })
    }
}

struct GiphyGreetingPoolSource<T, C> {
    transport: T,
    cache: C,
    api_key: Option<String>,
}

impl<T: GiphyTransport, C: GiphyPoolCache> GreetingPoolSource for GiphyGreetingPoolSource<T, C> {
    fn pool(
        &mut self,
        category: bot_core::greeting_commands::GreetingCategory,
    ) -> GreetingPoolLoad {
        let load = load_giphy_pool(
            &self.transport,
            &mut self.cache,
            self.api_key.as_deref(),
            category,
            || rand::random_range(0..=100_u16),
        );
        GreetingPoolLoad {
            urls: load.urls,
            diagnostics: load.diagnostics,
        }
    }
}

impl ChatConfigSource for ChatConfigRepository {
    type Error = ChatConfigRepositoryError;

    fn get(&mut self, chat_id: &str) -> Result<ChatConfig, Self::Error> {
        ChatConfigRepository::get(self, chat_id).map(|config| config.unwrap_or_default())
    }

    fn set(&mut self, chat_id: &str, config: &ChatConfig) -> Result<ChatConfig, Self::Error> {
        ChatConfigRepository::set(self, chat_id, config)
    }
}

impl StarPaymentSink for BillingRepository {
    fn record(&mut self, payment: &StarPaymentRecord) -> Result<StarPaymentReceipt, String> {
        let xtr_amount = i32::try_from(payment.xtr_amount)
            .map_err(|_| "Stars amount exceeds the PostgreSQL integer range".to_owned())?;
        let credits_awarded = i32::try_from(payment.credits_awarded)
            .map_err(|_| "credit amount exceeds the PostgreSQL integer range".to_owned())?;
        BillingRepository::record_star_payment(
            self,
            &payment.charge_id,
            payment.user_id,
            &payment.pack_id,
            xtr_amount,
            credits_awarded,
            Some(&payment.payload),
        )
        .map(|result| StarPaymentReceipt {
            inserted: result.inserted,
            user_balance: result.user_balance,
        })
        .map_err(|error| error.to_string())
    }
}

impl BillingBalanceSource for BillingRepository {
    fn load(&mut self, user_id: i64, chat_id: Option<i64>) -> Result<BillingBalances, String> {
        let mut diagnostics = Vec::new();
        if let Err(error) = self.grant_onboarding_if_needed(user_id, 300) {
            diagnostics.push(format!(
                "billing onboarding grant user_id={user_id}: {error}"
            ));
        }
        let user_balance = self
            .get_balance("user", user_id)
            .map_err(|error| error.to_string())?;
        let chat_balance = chat_id
            .map(|chat_id| self.get_balance("chat", chat_id))
            .transpose()
            .map_err(|error| error.to_string())?;
        Ok(BillingBalances {
            user_balance,
            chat_balance,
            diagnostics,
        })
    }
}

impl BillingTransferSink for BillingRepository {
    fn transfer(
        &mut self,
        user_id: i64,
        chat_id: i64,
        amount: i64,
    ) -> Result<bot_core::billing_commands::TransferResult, String> {
        let amount = i32::try_from(amount)
            .map_err(|_| "credit transfer amount exceeds the persistent range".to_owned())?;
        let result = self
            .transfer_user_to_chat(user_id, chat_id, amount)
            .map_err(|error| error.to_string())?;
        Ok(bot_core::billing_commands::TransferResult {
            transferred: result.transferred,
            user_balance: result.user_balance,
            chat_balance: result.chat_balance,
        })
    }
}

fn build_charge_history_page(
    rows: Vec<ChargeHistoryRow>,
    limit: usize,
    cursor_id: Option<i64>,
    direction: &str,
) -> ChargeHistoryPage {
    let mut groups = Vec::<ChargeHistoryGroup>::new();
    for row in rows {
        let entry = ChargeHistoryEntry {
            id: row.id,
            event_type: row.event_type,
            metadata: row.metadata,
        };
        if let Some(group) = groups
            .last_mut()
            .filter(|group| group.cursor_id == row.group_cursor)
        {
            group.entries.push(entry);
        } else {
            groups.push(ChargeHistoryGroup {
                cursor_id: row.group_cursor,
                created_at: row.group_created_at,
                entries: vec![entry],
            });
        }
    }
    let has_extra = groups.len() > limit;
    groups.truncate(limit);
    if direction == "newer" {
        groups.reverse();
    }
    ChargeHistoryPage {
        has_newer: if direction == "newer" {
            has_extra
        } else {
            cursor_id.is_some()
        },
        has_older: if direction == "newer" {
            cursor_id.is_some()
        } else {
            has_extra
        },
        newer_cursor: groups.first().map(|group| group.cursor_id),
        older_cursor: groups.last().map(|group| group.cursor_id),
        groups,
    }
}

impl ChargeHistorySource for BillingRepository {
    fn load(
        &mut self,
        user_id: i64,
        limit: usize,
        cursor_id: Option<i64>,
        direction: &str,
    ) -> Result<ChargeHistoryPage, String> {
        let query_limit = i64::try_from(limit.saturating_add(1))
            .map_err(|_| "charge history limit exceeds the query range".to_owned())?;
        let rows = self
            .list_user_ai_charge_rows(user_id, cursor_id, direction, query_limit)
            .map_err(|error| error.to_string())?;
        Ok(build_charge_history_page(rows, limit, cursor_id, direction))
    }
}

pub struct TelegramUpdateSource<Transport> {
    transport: Transport,
    token: String,
    long_poll_seconds: u64,
}

impl<Transport> TelegramUpdateSource<Transport> {
    #[must_use]
    pub fn new(transport: Transport, token: &str, long_poll_timeout: Duration) -> Self {
        Self {
            transport,
            token: token.to_owned(),
            long_poll_seconds: long_poll_timeout.as_secs(),
        }
    }
}

impl<Transport: TelegramTransport> UpdateSource for TelegramUpdateSource<Transport> {
    fn poll(&mut self, offset: Option<i64>) -> Result<PollOutcome, PollingError> {
        poll_once_with(&self.transport, &self.token, offset, self.long_poll_seconds)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TelegramActionSinkError {
    #[error(transparent)]
    Adapter(#[from] ActionError),
    #[error("Telegram action was rate limited")]
    RateLimited { retry_after_seconds: Option<u64> },
    #[error("Telegram action failed with status {status_code:?}: {description}")]
    Failed {
        status_code: Option<u16>,
        description: String,
    },
    #[error("Telegram action transport failed: {0:?}")]
    Transport(TransportFailureKind),
}

pub struct TelegramActionSink<Transport> {
    transport: Transport,
    token: String,
}

pub struct SystemRuntimeValues {
    instance_name: Option<String>,
}

impl SystemRuntimeValues {
    #[must_use]
    pub const fn new(instance_name: Option<String>) -> Self {
        Self { instance_name }
    }
}

impl RuntimeValues for SystemRuntimeValues {
    fn unix_timestamp(&mut self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64)
    }

    fn instance_name(&self) -> Option<&str> {
        self.instance_name.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SystemRandomError {
    #[error("random range must contain at least one value")]
    EmptyRange,
    #[error("random range is too large for this platform")]
    RangeTooLarge,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemRandomSource;

fn random_biguint_below(upper_exclusive: &BigUint) -> Result<BigUint, SystemRandomError> {
    if upper_exclusive == &BigUint::from(0_u8) {
        return Err(SystemRandomError::EmptyRange);
    }
    let bit_count = upper_exclusive.bits();
    let byte_count =
        usize::try_from(bit_count.div_ceil(8)).map_err(|_| SystemRandomError::RangeTooLarge)?;
    let retained_bits = bit_count % 8;
    loop {
        let mut bytes = vec![0_u8; byte_count];
        rand::fill(&mut bytes);
        if retained_bits != 0 {
            let mask = (1_u8 << retained_bits) - 1;
            if let Some(last) = bytes.last_mut() {
                *last &= mask;
            }
        }
        let candidate = BigUint::from_bytes_le(&bytes);
        if &candidate < upper_exclusive {
            return Ok(candidate);
        }
    }
}

impl RandomSource for SystemRandomSource {
    type Error = SystemRandomError;

    fn choice_index(&mut self, upper_exclusive: usize) -> Result<usize, Self::Error> {
        if upper_exclusive == 0 {
            return Err(SystemRandomError::EmptyRange);
        }
        Ok(rand::random_range(0..upper_exclusive))
    }

    fn inclusive_integer(&mut self, start: &BigInt, end: &BigInt) -> Result<BigInt, Self::Error> {
        let width = end - start + BigInt::from(1_u8);
        let Some(upper_exclusive) = width.to_biguint() else {
            return Err(SystemRandomError::EmptyRange);
        };
        let offset = random_biguint_below(&upper_exclusive)?;
        Ok(start + BigInt::from(offset))
    }
}

pub struct RedisCommandState {
    state: RedisMessageState,
}

pub struct TelegramGroupAuthorizer<Transport> {
    transport: Transport,
    token: String,
    redis_endpoint: RedisEndpoint,
}

impl<Transport> TelegramGroupAuthorizer<Transport> {
    #[must_use]
    pub fn new(transport: Transport, token: &str, redis_endpoint: &RedisEndpoint) -> Self {
        Self {
            transport,
            token: token.to_owned(),
            redis_endpoint: redis_endpoint.clone(),
        }
    }
}

impl<Transport: TelegramTransport> GroupAuthorizer for TelegramGroupAuthorizer<Transport> {
    fn authorize(&mut self, chat_id: &str, user_id: &str) -> GroupAuthorizationDecision {
        let mut diagnostics = Vec::new();
        match get_cached_chat_admin(&self.redis_endpoint, chat_id, user_id) {
            Ok(Some(is_admin)) => {
                return GroupAuthorizationDecision {
                    is_admin,
                    diagnostics,
                };
            }
            Ok(None) => {}
            Err(error) => diagnostics.push(format!("chat-admin cache read: {error}")),
        }
        let lookup = lookup_chat_admin_with(&self.transport, &self.token, chat_id, user_id);
        let is_admin = match lookup {
            Ok(lookup) => {
                if let Some(diagnostic) = lookup.diagnostic {
                    diagnostics.push(diagnostic);
                }
                lookup.is_admin
            }
            Err(error) => {
                diagnostics.push(format!("chat-admin lookup: {error}"));
                false
            }
        };
        if let Err(error) = cache_chat_admin(&self.redis_endpoint, chat_id, user_id, is_admin, 300)
        {
            diagnostics.push(format!("chat-admin cache write: {error}"));
        }
        GroupAuthorizationDecision {
            is_admin,
            diagnostics,
        }
    }
}

impl RedisCommandState {
    pub fn new(endpoint: &RedisEndpoint) -> Result<Self, RedisMessageStateError> {
        RedisMessageState::new(endpoint).map(|state| Self { state })
    }
}

impl MessageStateSink for RedisCommandState {
    type Error = RedisMessageStateError;

    fn record_incoming(&mut self, plan: &IncomingCommandWritePlan) -> Result<(), Self::Error> {
        let _stored = self.state.save_message(
            &plan.message,
            CHAT_STATE_TTL_SECONDS,
            CHAT_HISTORY_WRITE_LIMIT,
        )?;
        if let Some(member) = &plan.member {
            self.state.save_chat_member(
                &member.key,
                &member.user_id,
                &member.payload,
                CHAT_STATE_TTL_SECONDS,
            )?;
        }
        Ok(())
    }

    fn record_outgoing(&mut self, plan: &OutgoingCommandWritePlan) -> Result<(), Self::Error> {
        let _stored = self.state.save_message(
            &plan.message,
            CHAT_STATE_TTL_SECONDS,
            CHAT_HISTORY_WRITE_LIMIT,
        )?;
        if let Some(metadata) = &plan.metadata {
            self.state.set_value(
                &metadata.key,
                &metadata.payload,
                BOT_MESSAGE_METADATA_TTL_SECONDS,
            )?;
        }
        Ok(())
    }
}

impl<Transport> TelegramActionSink<Transport> {
    #[must_use]
    pub fn new(transport: Transport, token: &str) -> Self {
        Self {
            transport,
            token: token.to_owned(),
        }
    }
}

impl<Transport: TelegramTransport> ActionSink for TelegramActionSink<Transport> {
    type Error = TelegramActionSinkError;

    fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
        match execute_with(&self.transport, &self.token, action)? {
            ActionOutcome::Completed { message_id } => Ok(ActionReceipt {
                message_id: message_id.map(bot_core::telegram_input::MessageId),
            }),
            ActionOutcome::RateLimited {
                retry_after_seconds,
            } => Err(TelegramActionSinkError::RateLimited {
                retry_after_seconds,
            }),
            ActionOutcome::Failed {
                status_code,
                description,
            } => Err(TelegramActionSinkError::Failed {
                status_code,
                description,
            }),
            ActionOutcome::TransportFailed(failure) => {
                Err(TelegramActionSinkError::Transport(failure))
            }
        }
    }

    fn try_edit(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        Ok(matches!(
            execute_with(&self.transport, &self.token, action)?,
            ActionOutcome::Completed { .. }
        ))
    }

    fn try_invoice(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        Ok(matches!(
            execute_with(&self.transport, &self.token, action)?,
            ActionOutcome::Completed { .. }
        ))
    }

    fn try_animation(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        Ok(matches!(
            execute_with(&self.transport, &self.token, action)?,
            ActionOutcome::Completed { .. }
        ))
    }
}

pub fn publish_telegram_commands<Actions: ActionSink>(
    actions: &mut Actions,
) -> Result<(), Actions::Error> {
    for action in command_publication_actions() {
        let _receipt = actions.execute(action)?;
    }
    Ok(())
}

pub type ConcreteNativeRuntime = PollingRuntime<
    TelegramUpdateSource<ReqwestTelegramTransport>,
    NativeDispatcher<
        ChatConfigRepository,
        TelegramActionSink<ReqwestTelegramTransport>,
        RedisCommandState,
        SystemRuntimeValues,
        SystemRandomSource,
        TelegramGroupAuthorizer<ReqwestTelegramTransport>,
    >,
>;

#[derive(Debug, Error)]
pub enum CompositionError {
    #[error("could not construct Telegram polling transport: {0:?}")]
    PollingTransport(TransportFailureKind),
    #[error("could not construct Telegram action transport: {0:?}")]
    ActionTransport(TransportFailureKind),
    #[error("could not construct Telegram chat-admin transport: {0:?}")]
    AdminTransport(TransportFailureKind),
    #[error("could not construct CoinMarketCap transport: {0:?}")]
    CoinMarketCapTransport(CoinMarketCapTransportFailureKind),
    #[error("could not construct CriptoYa transport: {0:?}")]
    CriptoYaTransport(CriptoYaTransportFailureKind),
    #[error("could not construct Giphy transport: {0:?}")]
    GiphyTransport(GiphyTransportFailureKind),
    #[error("could not construct Giphy Redis cache: {0}")]
    GiphyCache(RedisJsonCacheError),
    #[error("could not construct Redis command state: {0}")]
    RedisState(#[from] RedisMessageStateError),
}

pub struct NativeRuntimeOptions<'a> {
    pub token: &'a str,
    pub database_url: &'a str,
    pub bot_name: &'a str,
    pub instance_name: Option<String>,
    pub redis_endpoint: &'a RedisEndpoint,
    pub long_poll_timeout: Duration,
    pub admin_user_id: Option<i64>,
    pub coinmarketcap_key: Option<String>,
    pub giphy_api_key: Option<String>,
}

pub fn build_native_runtime(
    options: NativeRuntimeOptions<'_>,
) -> Result<ConcreteNativeRuntime, CompositionError> {
    let polling_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::PollingTransport)?;
    let action_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::ActionTransport)?;
    let admin_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::AdminTransport)?;
    let criptoya_transport =
        ReqwestCriptoYaTransport::new().map_err(CompositionError::CriptoYaTransport)?;
    let rulo_transport =
        ReqwestCriptoYaTransport::new().map_err(CompositionError::CriptoYaTransport)?;
    let giphy_transport = ReqwestGiphyTransport::new().map_err(CompositionError::GiphyTransport)?;
    let giphy_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::GiphyCache)?;
    let source =
        TelegramUpdateSource::new(polling_transport, options.token, options.long_poll_timeout);
    let config = ChatConfigRepository::new(options.database_url);
    let actions = TelegramActionSink::new(action_transport, options.token);
    let state = RedisCommandState::new(options.redis_endpoint)?;
    let authorization =
        TelegramGroupAuthorizer::new(admin_transport, options.token, options.redis_endpoint);
    let dispatcher = NativeDispatcher::new(
        config,
        actions,
        state,
        SystemRuntimeValues::new(options.instance_name),
        SystemRandomSource,
        authorization,
        options.bot_name,
    )
    .with_payment_sink(Box::new(BillingRepository::new(options.database_url)))
    .with_balance_source(Box::new(BillingRepository::new(options.database_url)))
    .with_transfer_sink(Box::new(BillingRepository::new(options.database_url)))
    .with_charge_history_source(Box::new(BillingRepository::new(options.database_url)))
    .with_admin_user_id(options.admin_user_id)
    .with_admin_credit_sink(Box::new(BillingRepository::new(options.database_url)))
    .with_admin_creditlog_source(Box::new(BillingRepository::new(options.database_url)))
    .with_dollar_quotes_source(Box::new(CriptoYaDollarQuotesSource {
        transport: criptoya_transport,
    }))
    .with_rulo_source(Box::new(CriptoYaRuloSource {
        transport: rulo_transport,
    }))
    .with_greeting_pool_source(Box::new(GiphyGreetingPoolSource {
        transport: giphy_transport,
        cache: giphy_cache,
        api_key: options.giphy_api_key,
    }));
    let dispatcher = if let Some(api_key) = options.coinmarketcap_key.filter(|key| !key.is_empty())
    {
        let transport = ReqwestCoinMarketCapTransport::new()
            .map_err(CompositionError::CoinMarketCapTransport)?;
        dispatcher.with_bitcoin_price_source(Box::new(CoinMarketCapBitcoinPriceSource {
            transport,
            api_key,
        }))
    } else {
        dispatcher
    };
    Ok(PollingRuntime::new(source, dispatcher))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use bot_adapters::billing_read::{BillingRepository, ChargeHistoryRow};
    use bot_adapters::criptoya::{
        CriptoYaRequest, CriptoYaTransport, HttpResponse as CriptoYaHttpResponse,
        TransportFailureKind as CriptoYaFailure,
    };
    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_adapters::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
    };
    use bot_adapters::telegram_polling::{PollFailure, PollOutcome};
    use bot_core::telegram_actions::{LabeledPrice, SendMessage, TelegramAction};
    use bot_core::telegram_input::ChatId;
    use bot_core::telegram_payments::StarPaymentRecord;
    use bot_core::{
        command_state::{
            IncomingCommandState, OutgoingCommandState, prepare_incoming_command_state,
            prepare_outgoing_command_state,
        },
        telegram_input::{MessageId, UserId},
    };
    use num_bigint::BigInt;

    use crate::dispatcher::{
        ActionReceipt, ActionSink, BillingTransferSink, GroupAuthorizer, MessageStateSink,
        RandomSource, RuntimeValues, StarPaymentSink,
    };
    use crate::runtime::UpdateSource;

    use super::build_charge_history_page;

    use super::{
        CriptoYaRuloSource, NativeRuntimeOptions, RedisCommandState, SystemRandomError,
        SystemRandomSource, SystemRuntimeValues, TelegramActionSink, TelegramActionSinkError,
        TelegramGroupAuthorizer, TelegramUpdateSource, build_native_runtime,
        publish_telegram_commands,
    };
    use crate::dispatcher::RuloSource;

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    struct CriptoTransport {
        results: RefCell<Vec<Result<CriptoYaHttpResponse, CriptoYaFailure>>>,
        requests: RefCell<Vec<CriptoYaRequest>>,
    }

    impl CriptoYaTransport for CriptoTransport {
        fn get(&self, request: &CriptoYaRequest) -> Result<CriptoYaHttpResponse, CriptoYaFailure> {
            self.requests.borrow_mut().push(request.clone());
            if self.results.borrow().is_empty() {
                return Err(CriptoYaFailure::Request);
            }
            self.results.borrow_mut().remove(0)
        }
    }

    impl TelegramTransport for Transport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn transport(status_code: u16, body: &str) -> Transport {
        Transport {
            response: RefCell::new(Some(Ok(HttpResponse {
                status_code,
                body: body.to_owned(),
            }))),
            requests: RefCell::new(Vec::new()),
        }
    }

    fn read_command(reader: &mut BufReader<TcpStream>) -> std::io::Result<Vec<String>> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let count = line
            .trim_end()
            .strip_prefix('*')
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            line.clear();
            reader.read_line(&mut line)?;
            let length = line
                .trim_end()
                .strip_prefix('$')
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let mut bytes = vec![0_u8; length + 2];
            std::io::Read::read_exact(reader, &mut bytes)?;
            values.push(String::from_utf8_lossy(&bytes[..length]).into_owned());
        }
        Ok(values)
    }

    #[test]
    fn polling_source_uses_configured_timeout_token_and_offset() {
        let transport = transport(200, r#"{"ok":true,"result":[]}"#);
        let mut source =
            TelegramUpdateSource::new(transport, "synthetic-token", Duration::from_secs(17));
        assert_eq!(source.poll(Some(42)), Ok(PollOutcome::Updates(Vec::new())));
        let requests = source.transport.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].token, "synthetic-token");
        assert_eq!(requests[0].timeout, Duration::from_secs(22));
        assert_eq!(
            requests[0]
                .params
                .as_ref()
                .and_then(|params| params.get("offset")),
            Some(&serde_json::json!(42))
        );
    }

    #[test]
    fn polling_transport_failure_is_retryable_not_an_invalid_update() {
        let transport = Transport {
            response: RefCell::new(Some(Err(TransportFailureKind::Timeout))),
            requests: RefCell::new(Vec::new()),
        };
        let mut source = TelegramUpdateSource::new(transport, "token", Duration::from_secs(30));
        assert_eq!(
            source.poll(None),
            Ok(PollOutcome::Retry(PollFailure::Transport {
                failure: TransportFailureKind::Timeout
            }))
        );
    }

    #[test]
    fn action_sink_accepts_only_confirmed_delivery() {
        let mut sink = TelegramActionSink::new(
            transport(200, r#"{"ok":true,"result":{"message_id":9}}"#),
            "token",
        );
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Ok(ActionReceipt {
                message_id: Some(bot_core::telegram_input::MessageId(9))
            })
        );

        let mut sink = TelegramActionSink::new(
            transport(
                429,
                r#"{"ok":false,"error_code":429,"parameters":{"retry_after":4}}"#,
            ),
            "token",
        );
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Err(TelegramActionSinkError::RateLimited {
                retry_after_seconds: Some(4)
            })
        );
    }

    #[test]
    fn payment_sink_rejects_values_that_cannot_fit_the_persistent_schema() {
        let mut repository = BillingRepository::new("postgresql://unused");
        let base = StarPaymentRecord {
            charge_id: "charge-1".to_owned(),
            user_id: 42,
            pack_id: "p50".to_owned(),
            xtr_amount: i64::from(i32::MAX) + 1,
            credits_awarded: 5_000,
            payload: "topup:p50:42:en".to_owned(),
        };
        assert_eq!(
            repository.record(&base),
            Err("Stars amount exceeds the PostgreSQL integer range".to_owned())
        );
        assert_eq!(
            repository.record(&StarPaymentRecord {
                xtr_amount: 25,
                credits_awarded: i64::from(i32::MAX) + 1,
                ..base
            }),
            Err("credit amount exceeds the PostgreSQL integer range".to_owned())
        );
    }

    #[test]
    fn transfer_sink_rejects_values_that_cannot_fit_the_persistent_schema() {
        let mut repository = BillingRepository::new("postgresql://unused");
        assert_eq!(
            repository.transfer(42, -202, i64::from(i32::MAX) + 1),
            Err("credit transfer amount exceeds the persistent range".to_owned())
        );
    }

    #[test]
    fn charge_history_page_groups_rows_and_preserves_bidirectional_cursors() {
        let row = |id: i64, cursor: i64, group: &str| ChargeHistoryRow {
            id,
            event_type: "ai_settlement_result".to_owned(),
            actor_user_id: Some(42),
            user_id: Some(42),
            chat_id: Some(-202),
            amount: -1,
            metadata: serde_json::json!({"charged_credit_units_total":1}),
            created_at: "2026-08-26 17:00:00+00".to_owned(),
            group_key: group.to_owned(),
            group_cursor: cursor,
            group_created_at: "2026-08-26 17:00:00+00".to_owned(),
        };
        let older = build_charge_history_page(
            vec![row(30, 29, "a"), row(29, 29, "a"), row(28, 28, "b")],
            1,
            None,
            "older",
        );
        assert_eq!(older.groups.len(), 1);
        assert_eq!(older.groups[0].entries.len(), 2);
        assert!(!older.has_newer);
        assert!(older.has_older);
        assert_eq!(older.newer_cursor, Some(29));
        assert_eq!(older.older_cursor, Some(29));

        let newer = build_charge_history_page(
            vec![row(28, 28, "b"), row(30, 29, "a")],
            2,
            Some(27),
            "newer",
        );
        assert_eq!(
            newer
                .groups
                .iter()
                .map(|group| group.cursor_id)
                .collect::<Vec<_>>(),
            [29, 28]
        );
        assert!(!newer.has_newer);
        assert!(newer.has_older);
    }

    #[test]
    fn action_sink_preserves_api_and_transport_failures() {
        let mut sink = TelegramActionSink::new(
            transport(400, r#"{"ok":false,"description":"synthetic rejection"}"#),
            "token",
        );
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Err(TelegramActionSinkError::Failed {
                status_code: Some(400),
                description: "synthetic rejection".to_owned()
            })
        );

        let transport = Transport {
            response: RefCell::new(Some(Err(TransportFailureKind::Connection))),
            requests: RefCell::new(Vec::new()),
        };
        let mut sink = TelegramActionSink::new(transport, "token");
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Err(TelegramActionSinkError::Transport(
                TransportFailureKind::Connection
            ))
        );
    }

    #[test]
    fn optional_edit_reports_api_failure_for_new_message_fallback() {
        let mut sink = TelegramActionSink::new(
            transport(
                400,
                r#"{"ok":false,"description":"message cannot be edited"}"#,
            ),
            "token",
        );
        assert_eq!(
            sink.try_edit(TelegramAction::EditMessage {
                chat_id: ChatId(1),
                message_id: MessageId(2),
                text: "updated settings".to_owned(),
                reply_markup: None,
            }),
            Ok(false)
        );

        let mut sink = TelegramActionSink::new(
            transport(200, r#"{"ok":true,"result":{"message_id":2}}"#),
            "token",
        );
        assert_eq!(
            sink.try_edit(TelegramAction::EditMessage {
                chat_id: ChatId(1),
                message_id: MessageId(2),
                text: "updated settings".to_owned(),
                reply_markup: None,
            }),
            Ok(true)
        );
    }

    #[test]
    fn optional_invoice_reports_delivery_success_for_callback_feedback() {
        let invoice = || TelegramAction::SendInvoice {
            chat_id: ChatId(1),
            title: "50.00 AI credit pack".to_owned(),
            description: "Add credits".to_owned(),
            payload: "topup:p50:42:en".to_owned(),
            currency: "XTR".to_owned(),
            prices: vec![LabeledPrice {
                label: "50.00 AI credits".to_owned(),
                amount: 25,
            }],
        };
        let mut failed = TelegramActionSink::new(
            transport(400, r#"{"ok":false,"description":"invoice rejected"}"#),
            "token",
        );
        assert_eq!(failed.try_invoice(invoice()), Ok(false));

        let mut delivered = TelegramActionSink::new(
            transport(200, r#"{"ok":true,"result":{"message_id":3}}"#),
            "token",
        );
        assert_eq!(delivered.try_invoice(invoice()), Ok(true));
    }

    #[test]
    fn command_publication_executes_default_spanish_and_english_in_order() {
        #[derive(Default)]
        struct Published(Vec<TelegramAction>);

        impl ActionSink for Published {
            type Error = std::convert::Infallible;

            fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
                self.0.push(action);
                Ok(ActionReceipt { message_id: None })
            }
        }

        let mut published = Published::default();
        assert_eq!(publish_telegram_commands(&mut published), Ok(()));
        let languages = published
            .0
            .iter()
            .filter_map(|action| match action {
                TelegramAction::SetCommands { language_code, .. } => language_code.as_deref(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(languages, ["es", "en"]);
        assert_eq!(published.0.len(), 3);
    }

    #[test]
    fn system_runtime_values_preserve_instance_and_current_epoch_seconds() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs());
        let mut values = SystemRuntimeValues::new(Some("synthetic".to_owned()));
        let actual = values.unix_timestamp();
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs());
        assert!(actual >= before as i64);
        assert!(actual <= after as i64);
        assert_eq!(values.instance_name(), Some("synthetic"));
    }

    #[test]
    fn system_random_source_samples_choices_and_arbitrary_precision_ranges() {
        let mut random = SystemRandomSource;
        for _ in 0..32 {
            let index = random.choice_index(3);
            assert!(index.is_ok_and(|value| value < 3));
        }
        assert_eq!(random.choice_index(0), Err(SystemRandomError::EmptyRange));

        let start = BigInt::parse_bytes(b"100000000000000000000", 10);
        let end = BigInt::parse_bytes(b"100000000000000000002", 10);
        let (Some(start), Some(end)) = (start, end) else {
            return;
        };
        for _ in 0..32 {
            let sampled = random.inclusive_integer(&start, &end);
            assert!(sampled.is_ok_and(|value| value >= start && value <= end));
        }
        assert_eq!(
            random.inclusive_integer(&BigInt::from(2_u8), &BigInt::from(1_u8)),
            Err(SystemRandomError::EmptyRange)
        );
    }

    #[test]
    fn group_authorizer_uses_cached_compatible_boolean_without_telegram() {
        let listener = TcpListener::bind(("127.0.0.1", 0));
        assert!(listener.is_ok());
        let Ok(listener) = listener else { return };
        let port = listener.local_addr().map(|address| address.port());
        assert!(port.is_ok());
        let Ok(port) = port else { return };
        let server = thread::spawn(move || {
            let accepted = listener.accept();
            assert!(accepted.is_ok());
            let Ok((stream, _)) = accepted else { return };
            let mut reader = BufReader::new(stream);
            assert_eq!(
                read_command(&mut reader).ok(),
                Some(vec!["GET".to_owned(), "chat_admin:-42:7".to_owned()])
            );
            assert!(reader.get_mut().write_all(b"$4\r\ntrue\r\n").is_ok());
        });
        let transport = Transport {
            response: RefCell::new(None),
            requests: RefCell::new(Vec::new()),
        };
        let mut authorizer = TelegramGroupAuthorizer::new(
            transport,
            "token",
            &RedisEndpoint {
                host: "127.0.0.1".to_owned(),
                port,
                password: None,
            },
        );
        let decision = authorizer.authorize("-42", "7");
        assert!(decision.is_admin);
        assert!(decision.diagnostics.is_empty());
        assert!(authorizer.transport.requests.borrow().is_empty());
        assert!(server.join().is_ok());
    }

    #[test]
    fn group_authorizer_queries_telegram_on_cache_miss_and_caches_result() {
        let listener = TcpListener::bind(("127.0.0.1", 0));
        assert!(listener.is_ok());
        let Ok(listener) = listener else { return };
        let port = listener.local_addr().map(|address| address.port());
        assert!(port.is_ok());
        let Ok(port) = port else { return };
        let server = thread::spawn(move || {
            let accepted = listener.accept();
            assert!(accepted.is_ok());
            let Ok((stream, _)) = accepted else { return };
            let mut reader = BufReader::new(stream);
            assert_eq!(
                read_command(&mut reader)
                    .ok()
                    .and_then(|command| command.first().cloned()),
                Some("GET".to_owned())
            );
            assert!(reader.get_mut().write_all(b"$-1\r\n").is_ok());

            let accepted = listener.accept();
            assert!(accepted.is_ok());
            let Ok((stream, _)) = accepted else { return };
            let mut reader = BufReader::new(stream);
            let command = read_command(&mut reader);
            assert!(command.is_ok());
            let Ok(command) = command else { return };
            assert_eq!(command.first().map(String::as_str), Some("SETEX"));
            assert_eq!(command.get(1).map(String::as_str), Some("chat_admin:-42:7"));
            assert_eq!(command.get(2).map(String::as_str), Some("300"));
            assert_eq!(
                command.get(3).map(String::as_str),
                Some(r#"{"is_admin":true}"#)
            );
            assert!(reader.get_mut().write_all(b"+OK\r\n").is_ok());
        });
        let transport = transport(200, r#"{"ok":true,"result":{"status":"administrator"}}"#);
        let mut authorizer = TelegramGroupAuthorizer::new(
            transport,
            "token",
            &RedisEndpoint {
                host: "127.0.0.1".to_owned(),
                port,
                password: None,
            },
        );
        let decision = authorizer.authorize("-42", "7");
        assert!(decision.is_admin);
        assert!(decision.diagnostics.is_empty());
        assert_eq!(authorizer.transport.requests.borrow().len(), 1);
        assert!(server.join().is_ok());
    }

    #[test]
    fn redis_command_state_writes_history_member_and_metadata_contracts()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(
            move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let (index, _) = listener.accept()?;
                let mut reader = BufReader::new(index);
                assert_eq!(
                    read_command(&mut reader)?.first().map(String::as_str),
                    Some("FT.CREATE")
                );
                reader.get_mut().write_all(b"-Index already exists\r\n")?;

                let (incoming, _) = listener.accept()?;
                let mut reader = BufReader::new(incoming);
                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("EVAL"));
                assert!(command.iter().any(|value| value == "chat_history:-42"));
                reader.get_mut().write_all(b":1\r\n")?;

                let (member, _) = listener.accept()?;
                let mut reader = BufReader::new(member);
                assert_eq!(read_command(&mut reader)?, ["MULTI"]);
                reader.get_mut().write_all(b"+OK\r\n")?;
                assert_eq!(
                    read_command(&mut reader)?.first().map(String::as_str),
                    Some("HSET")
                );
                reader.get_mut().write_all(b"+QUEUED\r\n")?;
                assert_eq!(
                    read_command(&mut reader)?.first().map(String::as_str),
                    Some("EXPIRE")
                );
                reader.get_mut().write_all(b"+QUEUED\r\n")?;
                assert_eq!(read_command(&mut reader)?, ["EXEC"]);
                reader.get_mut().write_all(b"*2\r\n:1\r\n:1\r\n")?;

                let (outgoing, _) = listener.accept()?;
                let mut reader = BufReader::new(outgoing);
                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("EVAL"));
                assert!(command.iter().any(|value| value == "bot_99"));
                reader.get_mut().write_all(b":1\r\n")?;

                let (metadata, _) = listener.accept()?;
                let mut reader = BufReader::new(metadata);
                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("SETEX"));
                assert_eq!(
                    command.get(1).map(String::as_str),
                    Some("bot_message_meta:-42:99")
                );
                reader.get_mut().write_all(b"+OK\r\n")?;
                Ok(())
            },
        );

        let mut state = RedisCommandState::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;
        let incoming = prepare_incoming_command_state(IncomingCommandState {
            chat_id: ChatId(-42),
            message_id: MessageId(7),
            user_id: UserId(88),
            first_name: Some("Synthetic"),
            username: Some("tester"),
            text: "/time",
            is_group: true,
            timestamp: 1_672_531_200,
        })?;
        state.record_incoming(&incoming)?;
        let outgoing = prepare_outgoing_command_state(OutgoingCommandState {
            chat_id: ChatId(-42),
            incoming_message_id: MessageId(7),
            sent_message_id: Some(MessageId(99)),
            text: "1672531200",
            command: "/time",
            timestamp: 1_672_531_200,
        })?;
        state.record_outgoing(&outgoing)?;

        match server.join() {
            Ok(result) => result?,
            Err(_) => return Err("synthetic Redis server panicked".into()),
        }
        Ok(())
    }

    #[test]
    fn rulo_source_keeps_exchange_failures_nonfatal_and_primary_failures_explicit() {
        let transport = CriptoTransport {
            results: RefCell::new(vec![
                Ok(CriptoYaHttpResponse {
                    status_code: 200,
                    body: r#"{"oficial":{"price":1440},"blue":{"bid":1430}}"#.to_owned(),
                }),
                Ok(CriptoYaHttpResponse {
                    status_code: 200,
                    body: "bad".to_owned(),
                }),
                Ok(CriptoYaHttpResponse {
                    status_code: 200,
                    body: r#"{"buenbit":{"totalBid":1458.44}}"#.to_owned(),
                }),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        let mut source = CriptoYaRuloSource { transport };
        let load = source.rulo_input().unwrap_or_else(|_| unreachable!());
        assert_eq!(load.input.official, Some(1440.0));
        assert!(load.input.usd_to_usdt.is_empty());
        assert_eq!(load.input.usdt_to_ars[0].exchange, "buenbit");
        assert_eq!(load.diagnostics.len(), 1);
        assert!(load.diagnostics[0].contains("USDT/USD"));
        assert_eq!(source.transport.requests.borrow().len(), 3);

        let transport = CriptoTransport {
            results: RefCell::new(vec![Ok(CriptoYaHttpResponse {
                status_code: 503,
                body: String::new(),
            })]),
            requests: RefCell::new(Vec::new()),
        };
        let mut source = CriptoYaRuloSource { transport };
        assert!(
            source
                .rulo_input()
                .is_err_and(|error| error.contains("HTTP 503"))
        );
    }

    #[test]
    fn concrete_runtime_composes_without_contacting_external_services() {
        let endpoint = RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port: 1,
            password: None,
        };
        let result = build_native_runtime(NativeRuntimeOptions {
            token: "synthetic-token",
            database_url: "postgresql://synthetic.invalid/database",
            bot_name: "@synthetic_bot",
            instance_name: Some("synthetic-instance".to_owned()),
            redis_endpoint: &endpoint,
            long_poll_timeout: Duration::from_secs(30),
            admin_user_id: Some(99),
            coinmarketcap_key: None,
            giphy_api_key: None,
        });
        assert!(result.is_ok());
    }
}
