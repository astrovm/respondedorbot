//! Concrete adapter composition for the native Telegram runtime.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bot_adapters::bcra::{
    BcraTransport, ReqwestBcraTransport, TransportFailureKind as BcraTransportFailureKind,
    load_bcra, load_dollar_references,
};
use bot_adapters::billing_read::{BillingRepository, ChargeHistoryRow};
use bot_adapters::chat_config::{ChatConfigRepository, ChatConfigRepositoryError};
use bot_adapters::coinmarketcap::{
    BitcoinPriceOutcome, CoinMarketCapMarketTransport, CoinMarketCapTransport, MarketRequest,
    MarketRequestKind, ReqwestCoinMarketCapTransport,
    TransportFailureKind as CoinMarketCapTransportFailureKind, fetch_bitcoin_price,
    load_market_assets,
};
use bot_adapters::criptoya::{
    CriptoYaTransport, DollarQuotesOutcome, ExchangeQuotesOutcome, ExchangeSide,
    ReqwestCriptoYaTransport, RuloMarketOutcome,
    TransportFailureKind as CriptoYaTransportFailureKind, fetch_dollar_quotes,
    fetch_exchange_quotes, fetch_rulo_market,
};
use bot_adapters::dollar::{
    DollarCache, DollarTransport, ReqwestDollarTransport,
    TransportFailureKind as DollarTransportFailureKind, load_dollar_market,
};
use bot_adapters::finviz::{
    FinvizTransport, ReqwestFinvizTransport, TransportFailureKind as FinvizTransportFailureKind,
};
use bot_adapters::firecrawl::ReqwestFirecrawlTransport;
use bot_adapters::giphy::{
    GiphyTransport, ReqwestGiphyTransport, TransportFailureKind as GiphyTransportFailureKind,
};
use bot_adapters::giphy_pool::{GiphyPoolCache, load_giphy_pool};
use bot_adapters::hacker_news::ReqwestHackerNewsTransport;
use bot_adapters::link_preview::{
    LinkPreviewTransport, ReqwestLinkPreviewTransport, download_oversized_video,
    inspect_with as inspect_link_preview,
};
use bot_adapters::media_provider::ReqwestGroqTranscriptionTransport;
use bot_adapters::openrouter_chat::{
    DEFAULT_OPENROUTER_BASE_URL, OpenRouterChatError, ReqwestOpenRouterTransport,
};
use bot_adapters::polymarket::{
    PolymarketTransport, ReqwestPolymarketTransport,
    TransportFailureKind as PolymarketTransportFailureKind, load_elections,
};
use bot_adapters::redis_chat_admin::{cache_chat_admin, get_cached_chat_admin};
use bot_adapters::redis_compaction_queue::RedisCompactionQueue;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_creditless_cap::RedisCreditlessCap;
use bot_adapters::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};
use bot_adapters::redis_message_state::{RedisMessageState, RedisMessageStateError};
use bot_adapters::redis_task_store::{RedisTaskStore, RedisTaskStoreError};
use bot_adapters::redis_update_queue::RedisUpdateQueue;
use bot_adapters::request_cache::RequestCache;
use bot_adapters::stock_pool::{StockPoolCache, load_stock_pool};
use bot_adapters::telegram_actions::{ActionError, ActionOutcome, execute_with};
use bot_adapters::telegram_chat_admin::lookup_chat_admin_with;
use bot_adapters::telegram_http::{
    ReqwestTelegramTransport, TelegramTransport, TransportFailureKind,
};
use bot_adapters::telegram_polling::{PollOutcome, PollingError, poll_once_with};
use bot_adapters::token_signal::{
    ReqwestTokenSignalTransport, TokenSignalAdapter, TokenSignalCache, TokenSignalTransport,
};
use bot_adapters::weather::{
    ReqwestWeatherTransport, TransportFailureKind as WeatherTransportFailureKind, WeatherTransport,
    load_weather,
};
use bot_adapters::web_fetch::{ReqwestWebFetchTransport, SystemHostResolver};
use bot_adapters::yahoo_finance::{
    ReqwestYahooFinanceTransport, TransportFailureKind as YahooTransportFailureKind,
    YahooFinanceTransport, load_quote as load_yahoo_quote, load_symbol as load_yahoo_symbol,
};
use bot_core::ai_reserve::VISION_OUTPUT_TOKEN_LIMIT;
use bot_core::charge_history::{ChargeHistoryEntry, ChargeHistoryGroup, ChargeHistoryPage};
use bot_core::chat_config::ChatConfig;
use bot_core::command_state::{
    BOT_MESSAGE_METADATA_TTL_SECONDS, CHAT_HISTORY_WRITE_LIMIT, CHAT_STATE_TTL_SECONDS,
    IncomingCommandWritePlan, OutgoingCommandWritePlan,
};
use bot_core::links::replace_social_links;
use bot_core::market_prices::{
    CryptoAsset, CryptoMarketProvider, MarketPriceCommand, UnifiedStockProvider,
    execute_market_price_command,
};
use bot_core::stocks::{StockQuery, StockQuote, plan_stock_query};
use bot_core::telegram_actions::TelegramAction;
use bot_core::telegram_commands::command_publication_actions;
use bot_core::telegram_payments::StarPaymentRecord;
use num_bigint::{BigInt, BigUint};
use thiserror::Error;

use crate::chat_members_tool::ChatMembersTool;
use crate::chat_provider::OpenRouterChatStreamer;
use crate::chat_tool_loop::DEFAULT_MAX_TOOL_ROUNDS;
use crate::compaction_scheduler::production_compaction_scheduler;
use crate::conversation::ConversationToolFactory;
use crate::conversation::NativeConversation;
use crate::conversation_adapters::{PostgresConversationBilling, RedisConversationState};
use crate::dispatcher::{
    ActionReceipt, ActionSink, AdminCreditLogSource, AdminCreditSink, BcraLoad, BcraSource,
    BillingBalanceSource, BillingBalances, BillingTransferSink, BitcoinPriceSource,
    ChargeHistorySource, ChatConfigSource, DollarMarketLoad, DollarMarketSource,
    DollarQuotesSource, ElectionLoad, ElectionSource, GreetingPoolLoad, GreetingPoolSource,
    GroupAuthorizationDecision, GroupAuthorizer, LinkReplacementLoad, LinkReplacementSource,
    MarketPriceLoad, MarketPriceSource, MessageStateSink, NativeDispatcher, OilPriceSource,
    OilQuoteLoad, RandomSource, RuloInputLoad, RuloSource, RuntimeValues, ScheduledTaskSource,
    StarPaymentReceipt, StarPaymentSink, StockPriceSource, StockQuotesLoad, TokenSignalLoad,
    TokenSignalSource, WeatherObservationLoad, WeatherSource,
};
use crate::firecrawl_tool::FirecrawlTool;
use crate::hacker_news_tool::HackerNewsTool;
use crate::market_tools::{CryptoPricesTool, DollarRatesTool, StockPricesTool, WeatherTool};
use crate::media::NativeMedia;
use crate::media_adapters::{
    FallbackTranscriptionProvider, FfmpegMediaProcessor, OpenRouterVisionProvider, RedisMediaCache,
    TelegramMediaFiles, TranscriptionProviderConfig,
};
use crate::native_tools::{NativeTool, NativeToolRegistry, StandardNativeToolBackend};
use crate::random_tool::RandomChoiceTool;
use crate::reconciliation::ActiveOperationRegistry;
use crate::runtime::{DurableParallelUpdateHandler, PollingRuntime, UpdateSource};
use crate::task_tools::{
    RandomTaskIdSource, TaskCancelTool, TaskListTool, TaskSetTool, TaskToolContext,
};
use crate::tool_requests::{ExternalToolbox, ValidatedNativeToolPorts};
use crate::web_fetch_tool::WebFetchTool;

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

struct CachedCoinMarketCap<'a, T, C> {
    transport: &'a T,
    cache: &'a mut C,
    api_key: &'a str,
    now_unix: i64,
    diagnostics: Vec<String>,
}

impl<T: CoinMarketCapMarketTransport, C: RequestCache> CryptoMarketProvider
    for CachedCoinMarketCap<'_, T, C>
{
    fn listings(&mut self, currency: &str) -> Result<Vec<CryptoAsset>, String> {
        self.load(currency, MarketRequestKind::Listings)
    }

    fn quotes(
        &mut self,
        identifiers: &[String],
        currency: &str,
        by_slug: bool,
    ) -> Result<Vec<CryptoAsset>, String> {
        self.load(
            currency,
            MarketRequestKind::Quotes {
                identifiers: identifiers.to_vec(),
                by_slug,
            },
        )
    }
}

impl<T: CoinMarketCapMarketTransport, C: RequestCache> CachedCoinMarketCap<'_, T, C> {
    fn load(
        &mut self,
        currency: &str,
        kind: MarketRequestKind,
    ) -> Result<Vec<CryptoAsset>, String> {
        let load = load_market_assets(
            self.transport,
            self.cache,
            &MarketRequest {
                api_key: self.api_key.to_owned(),
                currency: currency.to_owned(),
                kind,
            },
            self.now_unix,
        );
        self.diagnostics.extend(load.diagnostics);
        load.assets
            .ok_or_else(|| "provider returned no usable market data".to_owned())
    }
}

struct UnifiedStocks<'a, T, F, C> {
    source: &'a mut YahooStockPriceSource<T, F, C>,
    now_unix: i64,
    diagnostics: Vec<String>,
}

impl<T, F, C> UnifiedStockProvider for UnifiedStocks<'_, T, F, C>
where
    T: YahooFinanceTransport,
    F: FinvizTransport,
    C: RequestCache + StockPoolCache,
{
    fn lookup(&mut self, query: &str) -> Result<Option<Vec<(String, Option<StockQuote>)>>, String> {
        let load = self.source.load(query, self.now_unix);
        self.diagnostics.extend(load.diagnostics);
        Ok(load.quotes)
    }
}

struct NativeMarketPriceSource<T, C, Y, F, S> {
    transport: T,
    cache: C,
    api_key: String,
    stocks: YahooStockPriceSource<Y, F, S>,
}

struct NativeLinkReplacementSource<T> {
    transport: T,
}

fn bounded_link_text(value: Option<&str>, limit: usize) -> Option<String> {
    let normalized = value?.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return None;
    }
    if normalized.chars().count() <= limit {
        return Some(normalized);
    }
    let mut truncated = normalized
        .chars()
        .take(limit.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    Some(truncated)
}

impl<T: LinkPreviewTransport> LinkReplacementSource for NativeLinkReplacementSource<T> {
    fn load(&mut self, text: &str, now_unix: i64) -> LinkReplacementLoad {
        let mut previews = Vec::new();
        let replacement = replace_social_links(text, now_unix, |candidate| {
            let preview = inspect_link_preview(&self.transport, candidate);
            if preview.embeddable {
                previews.push(preview);
                true
            } else {
                false
            }
        });
        let context = (!previews.is_empty()).then(|| {
            let mut lines = vec!["LINKS DEL MENSAJE:".to_owned()];
            for (index, preview) in previews.iter().enumerate() {
                lines.push(format!("{}. {}", index + 1, preview.final_url));
                if let Some(title) = bounded_link_text(preview.metadata.title.as_deref(), 160) {
                    lines.push(format!("titulo: {title}"));
                }
                if let Some(description) =
                    bounded_link_text(preview.metadata.description.as_deref(), 280)
                {
                    lines.push(format!("descripcion: {description}"));
                }
            }
            lines.join("\n")
        });
        let oversized_video = previews
            .iter()
            .find_map(|preview| download_oversized_video(&self.transport, preview));
        LinkReplacementLoad {
            replacement,
            context,
            oversized_video,
            diagnostics: Vec::new(),
        }
    }
}

impl<T, C, Y, F, S> MarketPriceSource for NativeMarketPriceSource<T, C, Y, F, S>
where
    T: CoinMarketCapMarketTransport,
    C: RequestCache,
    Y: YahooFinanceTransport,
    F: FinvizTransport,
    S: RequestCache + StockPoolCache,
{
    fn load(
        &mut self,
        query: &str,
        command: MarketPriceCommand,
        locale: bot_core::locale::Locale,
        now_unix: i64,
    ) -> MarketPriceLoad {
        let mut crypto = CachedCoinMarketCap {
            transport: &self.transport,
            cache: &mut self.cache,
            api_key: &self.api_key,
            now_unix,
            diagnostics: Vec::new(),
        };
        let mut stocks = UnifiedStocks {
            source: &mut self.stocks,
            now_unix,
            diagnostics: Vec::new(),
        };
        let mut execution =
            execute_market_price_command(query, command, locale, &mut crypto, &mut stocks);
        execution.diagnostics.extend(crypto.diagnostics);
        execution.diagnostics.extend(stocks.diagnostics);
        MarketPriceLoad {
            text: execution.text,
            diagnostics: execution.diagnostics,
        }
    }
}

struct CriptoYaDollarQuotesSource<T> {
    transport: T,
}

struct CriptoYaDollarMarketSource<T, B, C> {
    transport: T,
    bcra_transport: B,
    cache: C,
}

struct NativeBcraSource<T, C> {
    transport: T,
    cache: C,
}

impl<T: BcraTransport, C: RequestCache> BcraSource for NativeBcraSource<T, C> {
    fn load(&mut self, locale: bot_core::locale::Locale, now_unix: i64) -> BcraLoad {
        let load = load_bcra(&self.transport, &mut self.cache, locale, now_unix);
        BcraLoad {
            text: load.text,
            diagnostics: load.diagnostics,
        }
    }
}

impl<T: DollarTransport, B: BcraTransport, C: DollarCache> DollarMarketSource
    for CriptoYaDollarMarketSource<T, B, C>
{
    fn load(
        &mut self,
        hours_ago: i64,
        locale: bot_core::locale::Locale,
        now_unix: i64,
    ) -> DollarMarketLoad {
        let references = load_dollar_references(&self.bcra_transport, &mut self.cache, now_unix);
        let mut load = load_dollar_market(
            &self.transport,
            &mut self.cache,
            hours_ago,
            locale,
            now_unix,
        );
        load.diagnostics.extend(references.diagnostics);
        DollarMarketLoad {
            text: load.text,
            diagnostics: load.diagnostics,
        }
    }
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

struct OpenMeteoWeatherSource<T, C> {
    transport: T,
    cache: C,
}

impl<T: WeatherTransport, C: RequestCache> WeatherSource for OpenMeteoWeatherSource<T, C> {
    fn load(&mut self, location: &str, now_unix: i64) -> WeatherObservationLoad {
        let load = load_weather(&self.transport, &mut self.cache, location, now_unix);
        WeatherObservationLoad {
            observation: load.observation,
            diagnostics: load.diagnostics,
        }
    }
}

struct YahooOilPriceSource<T, C> {
    transport: T,
    cache: C,
}

impl<T: YahooFinanceTransport, C: RequestCache> OilPriceSource for YahooOilPriceSource<T, C> {
    fn load(&mut self, now_unix: i64) -> OilQuoteLoad {
        let brent = load_yahoo_quote(&self.transport, &mut self.cache, "BZ=F", now_unix);
        let wti = load_yahoo_quote(&self.transport, &mut self.cache, "CL=F", now_unix);
        let mut diagnostics = brent.diagnostics;
        diagnostics.extend(wti.diagnostics);
        OilQuoteLoad {
            brent: brent.quote,
            wti: wti.quote,
            diagnostics,
        }
    }
}

struct YahooStockPriceSource<T, F, C> {
    yahoo_transport: T,
    finviz_transport: F,
    cache: C,
}

impl<T, F, C> YahooStockPriceSource<T, F, C>
where
    T: YahooFinanceTransport,
    F: FinvizTransport,
    C: RequestCache + StockPoolCache,
{
    fn quote(
        &mut self,
        symbol: &str,
        now_unix: i64,
        diagnostics: &mut Vec<String>,
    ) -> Option<StockQuote> {
        let load = load_yahoo_quote(&self.yahoo_transport, &mut self.cache, symbol, now_unix);
        diagnostics.extend(load.diagnostics);
        load.quote
    }

    fn resolve(
        &mut self,
        query: &str,
        now_unix: i64,
        diagnostics: &mut Vec<String>,
    ) -> Option<String> {
        let load = load_yahoo_symbol(&self.yahoo_transport, &mut self.cache, query, now_unix);
        diagnostics.extend(load.diagnostics);
        load.symbol
    }

    fn resolve_missing(
        &mut self,
        quotes: Vec<(String, Option<StockQuote>)>,
        now_unix: i64,
        diagnostics: &mut Vec<String>,
    ) -> Vec<(String, Option<StockQuote>)> {
        quotes
            .into_iter()
            .map(|(query, quote)| {
                let quote = quote.or_else(|| {
                    let symbol = self.resolve(&query, now_unix, diagnostics)?;
                    self.quote(&symbol, now_unix, diagnostics)
                });
                (query, quote)
            })
            .collect()
    }
}

impl<T, F, C> StockPriceSource for YahooStockPriceSource<T, F, C>
where
    T: YahooFinanceTransport,
    F: FinvizTransport,
    C: RequestCache + StockPoolCache,
{
    fn load(&mut self, query: &str, now_unix: i64) -> StockQuotesLoad {
        let plan = plan_stock_query(query);
        let mut diagnostics = Vec::new();
        let queries = if plan.needs_top_stocks {
            let pool = load_stock_pool(&self.finviz_transport, &mut self.cache);
            diagnostics.extend(pool.diagnostics);
            if pool.symbols.is_empty() {
                return StockQuotesLoad {
                    quotes: None,
                    diagnostics,
                };
            }
            pool.symbols
                .into_iter()
                .take(20)
                .map(|symbol| StockQuery {
                    original: symbol.clone(),
                    normalized: symbol.to_uppercase().trim_start_matches('$').to_owned(),
                    is_symbol: true,
                })
                .collect::<Vec<_>>()
        } else {
            plan.queries
        };

        let mut quotes = Vec::with_capacity(queries.len());
        for item in queries {
            let quote = item
                .is_symbol
                .then(|| self.quote(&item.normalized, now_unix, &mut diagnostics))
                .flatten();
            quotes.push((item.original, quote));
        }
        let direct_quotes = quotes
            .iter()
            .filter_map(|(_, quote)| quote.clone())
            .collect::<Vec<_>>();
        if !plan.full_query_fallback || direct_quotes.len() == quotes.len() {
            return StockQuotesLoad {
                quotes: Some(self.resolve_missing(quotes, now_unix, &mut diagnostics)),
                diagnostics,
            };
        }

        let resolved = self.resolve(&plan.raw_query, now_unix, &mut diagnostics);
        let mut full_quote = resolved.as_ref().and_then(|symbol| {
            direct_quotes
                .iter()
                .find(|quote| quote.symbol.eq_ignore_ascii_case(symbol))
                .cloned()
        });
        if full_quote.is_none()
            && let Some(symbol) = resolved
        {
            full_quote = self.quote(&symbol, now_unix, &mut diagnostics);
        }
        if let Some(full_quote) = full_quote
            && (direct_quotes.is_empty()
                || !direct_quotes
                    .iter()
                    .any(|quote| quote.symbol.eq_ignore_ascii_case(&full_quote.symbol)))
        {
            return StockQuotesLoad {
                quotes: Some(vec![(plan.raw_query, Some(full_quote))]),
                diagnostics,
            };
        }
        if direct_quotes.is_empty() {
            return StockQuotesLoad {
                quotes: Some(vec![(plan.raw_query, None)]),
                diagnostics,
            };
        }
        StockQuotesLoad {
            quotes: Some(self.resolve_missing(quotes, now_unix, &mut diagnostics)),
            diagnostics,
        }
    }
}

struct PolymarketElectionSource<T, C> {
    transport: T,
    cache: C,
}

impl<T: PolymarketTransport, C: RequestCache> ElectionSource for PolymarketElectionSource<T, C> {
    fn load(&mut self, now_unix: i64) -> ElectionLoad {
        let load = load_elections(&self.transport, &mut self.cache, now_unix);
        ElectionLoad {
            events: load.events,
            live_prices: load.live_prices,
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

    fn set_changed(
        &mut self,
        chat_id: &str,
        previous: &ChatConfig,
        config: &ChatConfig,
    ) -> Result<ChatConfig, Self::Error> {
        ChatConfigRepository::set_changed(self, chat_id, previous, config)
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
    wait: Box<dyn FnMut(Duration) + Send>,
    delivery: TelegramDeliveryCoordinator,
}

const TELEGRAM_ACTION_MAX_ATTEMPTS: usize = 3;
const TELEGRAM_ACTION_DEFAULT_RETRY_SECONDS: u64 = 1;
const TELEGRAM_DELIVERY_LOCK_CLEANUP_THRESHOLD: usize = 1_024;

#[derive(Clone, Default)]
pub struct TelegramDeliveryCoordinator {
    locks: Arc<Mutex<HashMap<i64, Weak<Mutex<()>>>>>,
}

impl TelegramDeliveryCoordinator {
    fn delivery_lock(&self, chat_id: i64) -> Arc<Mutex<()>> {
        let mut locks = lock_unpoisoned(&self.locks);
        if let Some(lock) = locks.get(&chat_id).and_then(Weak::upgrade) {
            return lock;
        }
        if locks.len() >= TELEGRAM_DELIVERY_LOCK_CLEANUP_THRESHOLD {
            locks.retain(|_, lock| lock.strong_count() > 0);
        }
        let lock = Arc::new(Mutex::new(()));
        locks.insert(chat_id, Arc::downgrade(&lock));
        lock
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn telegram_action_chat_id(action: &TelegramAction) -> Option<i64> {
    match action {
        TelegramAction::SendMessage(message) => Some(message.chat_id.0),
        TelegramAction::SendAnimation { chat_id, .. }
        | TelegramAction::SendVideo { chat_id, .. }
        | TelegramAction::SendPhoto { chat_id, .. }
        | TelegramAction::EditMessagePhoto { chat_id, .. }
        | TelegramAction::SendInvoice { chat_id, .. }
        | TelegramAction::SendTyping { chat_id }
        | TelegramAction::EditMessage { chat_id, .. }
        | TelegramAction::DeleteMessage { chat_id, .. } => Some(chat_id.0),
        TelegramAction::SetCommands { .. }
        | TelegramAction::AnswerCallback { .. }
        | TelegramAction::AnswerPreCheckout { .. } => None,
    }
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

struct RedisScheduledTaskSource {
    store: RedisTaskStore,
}

impl ScheduledTaskSource for RedisScheduledTaskSource {
    fn list(
        &mut self,
        chat_id: &str,
    ) -> Result<Vec<bot_core::scheduled_tasks::ScheduledTask>, String> {
        self.store
            .list_chat_tasks(chat_id)
            .map(|documents| {
                documents
                    .into_iter()
                    .map(|document| document.task)
                    .collect()
            })
            .map_err(|error| error.to_string())
    }

    fn cancel(
        &mut self,
        task_id: &bot_core::scheduled_tasks::TaskId,
        chat_id: &str,
    ) -> Result<bool, String> {
        self.store
            .cancel_task(task_id.as_str(), chat_id)
            .map_err(|error| error.to_string())
    }
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
            wait: Box::new(std::thread::sleep),
            delivery: TelegramDeliveryCoordinator::default(),
        }
    }

    #[must_use]
    pub fn with_delivery_coordinator(mut self, delivery: TelegramDeliveryCoordinator) -> Self {
        self.delivery = delivery;
        self
    }

    #[cfg(test)]
    fn with_wait<Wait>(mut self, wait: Wait) -> Self
    where
        Wait: FnMut(Duration) + Send + 'static,
    {
        self.wait = Box::new(wait);
        self
    }
}

impl<Transport: TelegramTransport> TelegramActionSink<Transport> {
    fn execute_with_retry(&mut self, action: TelegramAction) -> Result<ActionOutcome, ActionError> {
        let delivery_lock =
            telegram_action_chat_id(&action).map(|chat_id| self.delivery.delivery_lock(chat_id));
        let _delivery_guard = delivery_lock.as_deref().map(lock_unpoisoned);
        let mut attempts = 0;
        loop {
            attempts += 1;
            let outcome = execute_with(&self.transport, &self.token, action.clone())?;
            match outcome {
                ActionOutcome::RateLimited {
                    retry_after_seconds,
                } if attempts < TELEGRAM_ACTION_MAX_ATTEMPTS => {
                    let seconds = retry_after_seconds
                        .unwrap_or(TELEGRAM_ACTION_DEFAULT_RETRY_SECONDS)
                        .max(1);
                    (self.wait)(Duration::from_secs(seconds));
                }
                outcome => return Ok(outcome),
            }
        }
    }
}

impl<Transport: TelegramTransport> ActionSink for TelegramActionSink<Transport> {
    type Error = TelegramActionSinkError;

    fn execute(&mut self, action: TelegramAction) -> Result<ActionReceipt, Self::Error> {
        match self.execute_with_retry(action)? {
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
            self.execute_with_retry(action)?,
            ActionOutcome::Completed { .. }
        ))
    }

    fn try_invoice(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        Ok(matches!(
            self.execute_with_retry(action)?,
            ActionOutcome::Completed { .. }
        ))
    }

    fn try_animation(&mut self, action: TelegramAction) -> Result<bool, Self::Error> {
        Ok(matches!(
            self.execute_with_retry(action)?,
            ActionOutcome::Completed { .. }
        ))
    }

    fn try_video(&mut self, action: TelegramAction) -> Result<Option<ActionReceipt>, Self::Error> {
        Ok(match self.execute_with_retry(action)? {
            ActionOutcome::Completed { message_id } => Some(ActionReceipt {
                message_id: message_id.map(bot_core::telegram_input::MessageId),
            }),
            ActionOutcome::RateLimited { .. }
            | ActionOutcome::Failed { .. }
            | ActionOutcome::TransportFailed(_) => None,
        })
    }

    fn try_photo(&mut self, action: TelegramAction) -> Result<Option<ActionReceipt>, Self::Error> {
        Ok(match self.execute_with_retry(action)? {
            ActionOutcome::Completed { message_id } => Some(ActionReceipt {
                message_id: message_id.map(bot_core::telegram_input::MessageId),
            }),
            ActionOutcome::RateLimited { .. }
            | ActionOutcome::Failed { .. }
            | ActionOutcome::TransportFailed(_) => None,
        })
    }
}

impl<Transport, Cache> TokenSignalSource for TokenSignalAdapter<Transport, Cache>
where
    Transport: TokenSignalTransport,
    Cache: TokenSignalCache,
{
    fn load(&mut self, query: &bot_core::token_signals::SignalQuery) -> TokenSignalLoad {
        let load = self.load_query(query);
        TokenSignalLoad {
            signal: load.signal,
            diagnostics: load.diagnostics,
        }
    }

    fn load_token(&mut self, token: &bot_core::token_signals::TokenAddress) -> TokenSignalLoad {
        let load = TokenSignalAdapter::load_token(self, token);
        TokenSignalLoad {
            signal: load.signal,
            diagnostics: load.diagnostics,
        }
    }

    fn render_photo(
        &mut self,
        signal: &bot_core::token_signals::TokenSignal,
    ) -> Result<Vec<u8>, String> {
        TokenSignalAdapter::render_photo(self, signal)
    }

    fn load_state(
        &mut self,
        signal_id: &str,
    ) -> Result<Option<bot_core::token_signals::SignalState>, String> {
        TokenSignalAdapter::load_state(self, signal_id)
    }

    fn save_state(
        &mut self,
        signal_id: &str,
        state: &bot_core::token_signals::SignalState,
    ) -> Result<(), String> {
        TokenSignalAdapter::save_state(self, signal_id, state)
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

pub const UPDATE_WORKER_COUNT: usize = 8;
pub const UPDATE_QUEUE_CAPACITY: usize = 100;

pub type ConcreteNativeDispatcher = NativeDispatcher<
    ChatConfigRepository,
    TelegramActionSink<ReqwestTelegramTransport>,
    RedisCommandState,
    SystemRuntimeValues,
    SystemRandomSource,
    TelegramGroupAuthorizer<ReqwestTelegramTransport>,
>;

pub type ConcreteNativeRuntime = PollingRuntime<
    TelegramUpdateSource<ReqwestTelegramTransport>,
    DurableParallelUpdateHandler<ConcreteNativeDispatcher, RedisUpdateQueue>,
>;

#[derive(Debug, Error)]
pub enum CompositionError {
    #[error("could not construct Telegram polling transport: {0:?}")]
    PollingTransport(TransportFailureKind),
    #[error("could not construct Telegram action transport: {0:?}")]
    ActionTransport(TransportFailureKind),
    #[error("could not construct Telegram chat-admin transport: {0:?}")]
    AdminTransport(TransportFailureKind),
    #[error("could not construct Telegram media transport: {0:?}")]
    MediaTransport(TransportFailureKind),
    #[error("could not construct CoinMarketCap transport: {0:?}")]
    CoinMarketCapTransport(CoinMarketCapTransportFailureKind),
    #[error("could not construct CoinMarketCap Redis cache: {0}")]
    CoinMarketCapCache(RedisJsonCacheError),
    #[error("could not construct CriptoYa transport: {0:?}")]
    CriptoYaTransport(CriptoYaTransportFailureKind),
    #[error("could not construct dollar-market transport: {0:?}")]
    DollarTransport(DollarTransportFailureKind),
    #[error("could not construct dollar-market Redis cache: {0}")]
    DollarCache(RedisJsonCacheError),
    #[error("could not construct BCRA transport: {0:?}")]
    BcraTransport(BcraTransportFailureKind),
    #[error("could not construct BCRA Redis cache: {0}")]
    BcraCache(RedisJsonCacheError),
    #[error("could not construct Giphy transport: {0:?}")]
    GiphyTransport(GiphyTransportFailureKind),
    #[error("could not construct Giphy Redis cache: {0}")]
    GiphyCache(RedisJsonCacheError),
    #[error("could not construct social-link preview transport: {0:?}")]
    LinkPreviewTransport(bot_adapters::link_preview::PreviewFailure),
    #[error("could not construct Open-Meteo transport: {0:?}")]
    WeatherTransport(WeatherTransportFailureKind),
    #[error("could not construct weather Redis cache: {0}")]
    WeatherCache(RedisJsonCacheError),
    #[error("could not construct Yahoo Finance transport: {0:?}")]
    YahooTransport(YahooTransportFailureKind),
    #[error("could not construct Yahoo Finance Redis cache: {0}")]
    YahooCache(RedisJsonCacheError),
    #[error("could not construct stock Yahoo Finance transport: {0:?}")]
    StockYahooTransport(YahooTransportFailureKind),
    #[error("could not construct Finviz transport: {0:?}")]
    FinvizTransport(FinvizTransportFailureKind),
    #[error("could not construct stock Redis cache: {0}")]
    StockCache(RedisJsonCacheError),
    #[error("could not construct Polymarket transport: {0:?}")]
    PolymarketTransport(PolymarketTransportFailureKind),
    #[error("could not construct Polymarket Redis cache: {0}")]
    PolymarketCache(RedisJsonCacheError),
    #[error("could not construct token-signal transport: {0}")]
    TokenSignalTransport(String),
    #[error("could not construct token-signal Redis cache: {0}")]
    TokenSignalCache(RedisJsonCacheError),
    #[error("could not construct Redis command state: {0}")]
    RedisState(#[from] RedisMessageStateError),
    #[error("could not construct Redis scheduled-task state: {0}")]
    RedisTasks(#[from] RedisTaskStoreError),
    #[error("could not construct OpenRouter chat transport: {0}")]
    OpenRouterChatTransport(OpenRouterChatError),
    #[error("could not construct media provider transport: {0}")]
    MediaProviderTransport(String),
    #[error("could not construct Redis AI conversation state: {0}")]
    ConversationState(String),
    #[error("could not construct Redis AI credit policy: {0}")]
    ConversationBillingPolicy(String),
    #[error("could not start parallel update workers: {0}")]
    UpdateWorkers(String),
}

#[derive(Clone)]
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
    pub openrouter_api_key: Option<String>,
    pub openrouter_base_url: Option<String>,
    pub groq_free_api_key: Option<String>,
    pub groq_api_key: Option<String>,
    pub firecrawl_api_key: Option<String>,
    pub system_prompt: Option<String>,
    pub trigger_words: Option<Vec<String>>,
    pub active_operations: ActiveOperationRegistry,
    pub telegram_delivery: TelegramDeliveryCoordinator,
}

#[derive(Clone)]
struct OwnedNativeRuntimeOptions {
    token: String,
    database_url: String,
    bot_name: String,
    instance_name: Option<String>,
    redis_endpoint: RedisEndpoint,
    long_poll_timeout: Duration,
    admin_user_id: Option<i64>,
    coinmarketcap_key: Option<String>,
    giphy_api_key: Option<String>,
    openrouter_api_key: Option<String>,
    openrouter_base_url: Option<String>,
    groq_free_api_key: Option<String>,
    groq_api_key: Option<String>,
    firecrawl_api_key: Option<String>,
    system_prompt: Option<String>,
    trigger_words: Option<Vec<String>>,
    active_operations: ActiveOperationRegistry,
    telegram_delivery: TelegramDeliveryCoordinator,
}

impl OwnedNativeRuntimeOptions {
    fn new(options: NativeRuntimeOptions<'_>) -> Self {
        Self {
            token: options.token.to_owned(),
            database_url: options.database_url.to_owned(),
            bot_name: options.bot_name.to_owned(),
            instance_name: options.instance_name,
            redis_endpoint: options.redis_endpoint.clone(),
            long_poll_timeout: options.long_poll_timeout,
            admin_user_id: options.admin_user_id,
            coinmarketcap_key: options.coinmarketcap_key,
            giphy_api_key: options.giphy_api_key,
            openrouter_api_key: options.openrouter_api_key,
            openrouter_base_url: options.openrouter_base_url,
            groq_free_api_key: options.groq_free_api_key,
            groq_api_key: options.groq_api_key,
            firecrawl_api_key: options.firecrawl_api_key,
            system_prompt: options.system_prompt,
            trigger_words: options.trigger_words,
            active_operations: options.active_operations,
            telegram_delivery: options.telegram_delivery,
        }
    }

    fn borrowed(&self) -> NativeRuntimeOptions<'_> {
        NativeRuntimeOptions {
            token: &self.token,
            database_url: &self.database_url,
            bot_name: &self.bot_name,
            instance_name: self.instance_name.clone(),
            redis_endpoint: &self.redis_endpoint,
            long_poll_timeout: self.long_poll_timeout,
            admin_user_id: self.admin_user_id,
            coinmarketcap_key: self.coinmarketcap_key.clone(),
            giphy_api_key: self.giphy_api_key.clone(),
            openrouter_api_key: self.openrouter_api_key.clone(),
            openrouter_base_url: self.openrouter_base_url.clone(),
            groq_free_api_key: self.groq_free_api_key.clone(),
            groq_api_key: self.groq_api_key.clone(),
            firecrawl_api_key: self.firecrawl_api_key.clone(),
            system_prompt: self.system_prompt.clone(),
            trigger_words: self.trigger_words.clone(),
            active_operations: self.active_operations.clone(),
            telegram_delivery: self.telegram_delivery.clone(),
        }
    }
}

pub type ProductionConversationTools =
    NativeToolRegistry<StandardNativeToolBackend<ValidatedNativeToolPorts<ExternalToolbox>>>;

pub struct ProductionToolFactory {
    redis_endpoint: RedisEndpoint,
    database_url: String,
    coinmarketcap_key: Option<String>,
    firecrawl_key: Option<String>,
}

impl ProductionToolFactory {
    #[must_use]
    pub fn new(
        redis_endpoint: &RedisEndpoint,
        database_url: &str,
        coinmarketcap_key: Option<String>,
        firecrawl_key: Option<String>,
    ) -> Self {
        Self {
            redis_endpoint: redis_endpoint.clone(),
            database_url: database_url.to_owned(),
            coinmarketcap_key,
            firecrawl_key,
        }
    }
}

impl ConversationToolFactory for ProductionToolFactory {
    type Tools = ProductionConversationTools;

    fn create(
        &mut self,
        input: &crate::ai_dispatch::AiConversationInput,
    ) -> Result<Self::Tools, String> {
        let locale = input.locale;
        let chat_id = input.chat_id.0.to_string();
        let mut toolbox = ExternalToolbox::new(locale);

        if let Some(api_key) = self.coinmarketcap_key.clone().filter(|key| !key.is_empty()) {
            let market = NativeMarketPriceSource {
                transport: ReqwestCoinMarketCapTransport::new()
                    .map_err(|error| format!("CoinMarketCap tool transport: {error:?}"))?,
                cache: RedisJsonCache::new(&self.redis_endpoint)
                    .map_err(|error| error.to_string())?,
                api_key,
                stocks: YahooStockPriceSource {
                    yahoo_transport: ReqwestYahooFinanceTransport::new()
                        .map_err(|error| format!("Yahoo tool transport: {error:?}"))?,
                    finviz_transport: ReqwestFinvizTransport::new()
                        .map_err(|error| format!("Finviz tool transport: {error:?}"))?,
                    cache: RedisJsonCache::new(&self.redis_endpoint)
                        .map_err(|error| error.to_string())?,
                },
            };
            toolbox = toolbox.with_executor(
                NativeTool::CryptoPrices,
                Box::new(CryptoPricesTool::new(
                    market,
                    current_unix_timestamp,
                    locale,
                )),
            );
        }

        toolbox = toolbox
            .with_executor(
                NativeTool::StockPrices,
                Box::new(StockPricesTool::new(
                    YahooStockPriceSource {
                        yahoo_transport: ReqwestYahooFinanceTransport::new()
                            .map_err(|error| format!("Yahoo tool transport: {error:?}"))?,
                        finviz_transport: ReqwestFinvizTransport::new()
                            .map_err(|error| format!("Finviz tool transport: {error:?}"))?,
                        cache: RedisJsonCache::new(&self.redis_endpoint)
                            .map_err(|error| error.to_string())?,
                    },
                    current_unix_timestamp,
                    locale,
                )),
            )
            .with_executor(
                NativeTool::DollarRates,
                Box::new(DollarRatesTool::new(
                    CriptoYaDollarMarketSource {
                        transport: ReqwestDollarTransport::new()
                            .map_err(|error| format!("dollar tool transport: {error:?}"))?,
                        bcra_transport: ReqwestBcraTransport::new()
                            .map_err(|error| format!("BCRA tool transport: {error:?}"))?,
                        cache: RedisJsonCache::new(&self.redis_endpoint)
                            .map_err(|error| error.to_string())?,
                    },
                    current_unix_timestamp,
                    locale,
                )),
            )
            .with_executor(
                NativeTool::Weather,
                Box::new(WeatherTool::new(
                    OpenMeteoWeatherSource {
                        transport: ReqwestWeatherTransport::new()
                            .map_err(|error| format!("weather tool transport: {error:?}"))?,
                        cache: RedisJsonCache::new(&self.redis_endpoint)
                            .map_err(|error| error.to_string())?,
                    },
                    current_unix_timestamp,
                    locale,
                )),
            )
            .with_executor(
                NativeTool::WebFetch,
                Box::new(WebFetchTool::new(
                    ReqwestWebFetchTransport::new().map_err(|error| error.to_string())?,
                    SystemHostResolver,
                    locale,
                )),
            )
            .with_executor(
                NativeTool::RandomChoice,
                Box::new(RandomChoiceTool::new(SystemRandomSource, locale)),
            )
            .with_executor(
                NativeTool::TaskSet,
                Box::new(TaskSetTool::new(
                    RedisTaskStore::new(&self.redis_endpoint).map_err(|error| error.to_string())?,
                    BillingRepository::new(&self.database_url),
                    RandomTaskIdSource,
                    current_unix_timestamp,
                    TaskToolContext {
                        chat_id: chat_id.clone(),
                        user_name: if input.sender_username.is_empty() {
                            input.sender_first_name.clone()
                        } else {
                            input.sender_username.clone()
                        },
                        user_id: Some(input.sender_id.0),
                        timezone_offset: i32::try_from(input.timezone_offset_hours)
                            .unwrap_or_default(),
                        locale,
                    },
                )),
            )
            .with_executor(
                NativeTool::TaskList,
                Box::new(TaskListTool::new(
                    RedisTaskStore::new(&self.redis_endpoint).map_err(|error| error.to_string())?,
                    &chat_id,
                    locale,
                )),
            )
            .with_executor(
                NativeTool::TaskCancel,
                Box::new(TaskCancelTool::new(
                    RedisTaskStore::new(&self.redis_endpoint).map_err(|error| error.to_string())?,
                    &chat_id,
                    locale,
                )),
            )
            .with_executor(
                NativeTool::GetChatMembers,
                Box::new(ChatMembersTool::new(
                    RedisMessageState::new(&self.redis_endpoint)
                        .map_err(|error| error.to_string())?,
                    current_unix_timestamp,
                    &chat_id,
                    locale,
                )),
            )
            .with_executor(
                NativeTool::HackerNews,
                Box::new(HackerNewsTool::new(
                    ReqwestHackerNewsTransport::new().map_err(|error| error.to_string())?,
                    RedisJsonCache::new(&self.redis_endpoint).map_err(|error| error.to_string())?,
                    locale,
                )),
            );

        if let Some(api_key) = self.firecrawl_key.clone().filter(|key| !key.is_empty()) {
            toolbox = toolbox.with_executor(
                NativeTool::WebSearch,
                Box::new(FirecrawlTool::new(
                    ReqwestFirecrawlTransport::new().map_err(|error| error.to_string())?,
                    std::thread::sleep,
                    &api_key,
                    locale,
                )),
            );
        }

        Ok(NativeToolRegistry::new(
            StandardNativeToolBackend::new(ValidatedNativeToolPorts::new(toolbox, locale), locale),
            locale,
        ))
    }
}

fn current_unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64)
}

fn build_native_dispatcher(
    options: NativeRuntimeOptions<'_>,
) -> Result<ConcreteNativeDispatcher, CompositionError> {
    let conversation_coinmarketcap_key = options.coinmarketcap_key.clone();
    let conversation_firecrawl_key = options.firecrawl_api_key.clone();
    let groq_free_api_key = options.groq_free_api_key.clone();
    let groq_api_key = options.groq_api_key.clone();
    let action_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::ActionTransport)?;
    let admin_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::AdminTransport)?;
    let criptoya_transport =
        ReqwestCriptoYaTransport::new().map_err(CompositionError::CriptoYaTransport)?;
    let dollar_transport =
        ReqwestDollarTransport::new().map_err(CompositionError::DollarTransport)?;
    let dollar_bcra_transport =
        ReqwestBcraTransport::new().map_err(CompositionError::BcraTransport)?;
    let dollar_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::DollarCache)?;
    let bcra_transport = ReqwestBcraTransport::new().map_err(CompositionError::BcraTransport)?;
    let bcra_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::BcraCache)?;
    let rulo_transport =
        ReqwestCriptoYaTransport::new().map_err(CompositionError::CriptoYaTransport)?;
    let giphy_transport = ReqwestGiphyTransport::new().map_err(CompositionError::GiphyTransport)?;
    let giphy_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::GiphyCache)?;
    let link_preview_transport =
        ReqwestLinkPreviewTransport::new().map_err(CompositionError::LinkPreviewTransport)?;
    let weather_transport =
        ReqwestWeatherTransport::new().map_err(CompositionError::WeatherTransport)?;
    let weather_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::WeatherCache)?;
    let yahoo_transport =
        ReqwestYahooFinanceTransport::new().map_err(CompositionError::YahooTransport)?;
    let yahoo_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::YahooCache)?;
    let stock_yahoo_transport =
        ReqwestYahooFinanceTransport::new().map_err(CompositionError::StockYahooTransport)?;
    let finviz_transport =
        ReqwestFinvizTransport::new().map_err(CompositionError::FinvizTransport)?;
    let stock_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::StockCache)?;
    let polymarket_transport =
        ReqwestPolymarketTransport::new().map_err(CompositionError::PolymarketTransport)?;
    let polymarket_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::PolymarketCache)?;
    let token_signal_transport =
        ReqwestTokenSignalTransport::new().map_err(CompositionError::TokenSignalTransport)?;
    let token_signal_cache =
        RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::TokenSignalCache)?;
    let config = ChatConfigRepository::new(options.database_url);
    let actions = TelegramActionSink::new(action_transport, options.token)
        .with_delivery_coordinator(options.telegram_delivery);
    let state = RedisCommandState::new(options.redis_endpoint)?;
    let task_source = RedisScheduledTaskSource {
        store: RedisTaskStore::new(options.redis_endpoint)?,
    };
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
    .with_dollar_market_source(Box::new(CriptoYaDollarMarketSource {
        transport: dollar_transport,
        bcra_transport: dollar_bcra_transport,
        cache: dollar_cache,
    }))
    .with_bcra_source(Box::new(NativeBcraSource {
        transport: bcra_transport,
        cache: bcra_cache,
    }))
    .with_rulo_source(Box::new(CriptoYaRuloSource {
        transport: rulo_transport,
    }))
    .with_greeting_pool_source(Box::new(GiphyGreetingPoolSource {
        transport: giphy_transport,
        cache: giphy_cache,
        api_key: options.giphy_api_key,
    }))
    .with_weather_source(Box::new(OpenMeteoWeatherSource {
        transport: weather_transport,
        cache: weather_cache,
    }))
    .with_oil_price_source(Box::new(YahooOilPriceSource {
        transport: yahoo_transport,
        cache: yahoo_cache,
    }))
    .with_stock_price_source(Box::new(YahooStockPriceSource {
        yahoo_transport: stock_yahoo_transport,
        finviz_transport,
        cache: stock_cache,
    }))
    .with_election_source(Box::new(PolymarketElectionSource {
        transport: polymarket_transport,
        cache: polymarket_cache,
    }))
    .with_link_replacement_source(Box::new(NativeLinkReplacementSource {
        transport: link_preview_transport,
    }))
    .with_scheduled_task_source(Box::new(task_source))
    .with_token_signal_source(Box::new(TokenSignalAdapter::new(
        token_signal_transport,
        token_signal_cache,
    )));
    let dispatcher = if let Some(words) = options.trigger_words.filter(|words| !words.is_empty()) {
        dispatcher.with_trigger_words(words)
    } else {
        dispatcher
    };
    let dispatcher = match (
        options.openrouter_api_key.filter(|key| !key.is_empty()),
        options.system_prompt.filter(|prompt| !prompt.is_empty()),
    ) {
        (Some(api_key), Some(system_prompt)) => {
            let openrouter_base_url = options
                .openrouter_base_url
                .as_deref()
                .filter(|url| !url.is_empty())
                .unwrap_or(DEFAULT_OPENROUTER_BASE_URL)
                .to_owned();
            let provider = OpenRouterChatStreamer::new(
                ReqwestOpenRouterTransport::new()
                    .map_err(CompositionError::OpenRouterChatTransport)?,
                &api_key,
                &openrouter_base_url,
                crate::native_ai::PRIMARY_CHAT_MODEL,
            );
            let groq_accounts = [("free", groq_free_api_key), ("paid", groq_api_key)]
                .into_iter()
                .filter_map(|(account, api_key)| {
                    api_key
                        .filter(|key| !key.is_empty())
                        .map(|key| (account.to_owned(), key))
                })
                .collect();
            let media = NativeMedia::new(
                TelegramMediaFiles::new(
                    ReqwestTelegramTransport::new().map_err(CompositionError::MediaTransport)?,
                    options.token,
                ),
                RedisMediaCache::new(options.redis_endpoint.clone()),
                FfmpegMediaProcessor::default(),
                OpenRouterVisionProvider::new(
                    ReqwestOpenRouterTransport::new()
                        .map_err(CompositionError::OpenRouterChatTransport)?,
                    &api_key,
                    &openrouter_base_url,
                    crate::native_ai::VISION_MODEL,
                    u64::try_from(VISION_OUTPUT_TOKEN_LIMIT).unwrap_or(512),
                ),
                FallbackTranscriptionProvider::new(
                    ReqwestGroqTranscriptionTransport::new().map_err(|error| {
                        CompositionError::MediaProviderTransport(error.to_string())
                    })?,
                    ReqwestOpenRouterTransport::new()
                        .map_err(CompositionError::OpenRouterChatTransport)?,
                    TranscriptionProviderConfig {
                        groq_accounts,
                        openrouter_api_key: Some(api_key.clone()),
                        openrouter_base_url: openrouter_base_url.clone(),
                        groq_model: crate::native_ai::GROQ_TRANSCRIPTION_MODEL.to_owned(),
                        openrouter_model: crate::native_ai::OPENROUTER_TRANSCRIPTION_MODEL
                            .to_owned(),
                        default_backoff_seconds: 60,
                    },
                ),
                crate::native_ai::VISION_MODEL,
            );
            let compaction_scheduler = production_compaction_scheduler(
                RedisCompactionQueue::new(options.redis_endpoint)
                    .map_err(|error| CompositionError::ConversationState(error.to_string()))?,
                options.database_url,
                &system_prompt,
            );
            let conversation = NativeConversation::new(
                provider,
                ProductionToolFactory::new(
                    options.redis_endpoint,
                    options.database_url,
                    conversation_coinmarketcap_key,
                    conversation_firecrawl_key,
                ),
                RedisConversationState::new(options.redis_endpoint)
                    .map_err(CompositionError::ConversationState)?,
                PostgresConversationBilling::new(options.database_url)
                    .with_creditless_cap(RedisCreditlessCap::new(options.redis_endpoint).map_err(
                        |error| CompositionError::ConversationBillingPolicy(error.to_string()),
                    )?)
                    .with_active_operations(options.active_operations.clone()),
                &system_prompt,
                crate::native_ai::PRIMARY_CHAT_MODEL,
                DEFAULT_MAX_TOOL_ROUNDS,
            )
            .with_media(Box::new(media))
            .with_compaction_scheduler(Box::new(compaction_scheduler));
            dispatcher.with_ai_conversation_source(Box::new(conversation))
        }
        _ => dispatcher,
    };
    let dispatcher = if let Some(api_key) = options.coinmarketcap_key.filter(|key| !key.is_empty())
    {
        let transport = ReqwestCoinMarketCapTransport::new()
            .map_err(CompositionError::CoinMarketCapTransport)?;
        let market_transport = ReqwestCoinMarketCapTransport::new()
            .map_err(CompositionError::CoinMarketCapTransport)?;
        let market_cache = RedisJsonCache::new(options.redis_endpoint)
            .map_err(CompositionError::CoinMarketCapCache)?;
        let market_yahoo_transport =
            ReqwestYahooFinanceTransport::new().map_err(CompositionError::StockYahooTransport)?;
        let market_finviz_transport =
            ReqwestFinvizTransport::new().map_err(CompositionError::FinvizTransport)?;
        let market_stock_cache =
            RedisJsonCache::new(options.redis_endpoint).map_err(CompositionError::StockCache)?;
        dispatcher
            .with_bitcoin_price_source(Box::new(CoinMarketCapBitcoinPriceSource {
                transport,
                api_key: api_key.clone(),
            }))
            .with_market_price_source(Box::new(NativeMarketPriceSource {
                transport: market_transport,
                cache: market_cache,
                api_key,
                stocks: YahooStockPriceSource {
                    yahoo_transport: market_yahoo_transport,
                    finviz_transport: market_finviz_transport,
                    cache: market_stock_cache,
                },
            }))
    } else {
        dispatcher
    };
    Ok(dispatcher)
}

pub fn build_native_runtime(
    options: NativeRuntimeOptions<'_>,
) -> Result<ConcreteNativeRuntime, CompositionError> {
    let options = OwnedNativeRuntimeOptions::new(options);
    let polling_transport =
        ReqwestTelegramTransport::new().map_err(CompositionError::PollingTransport)?;
    let source =
        TelegramUpdateSource::new(polling_transport, &options.token, options.long_poll_timeout);
    let worker_options = options.clone();
    let update_queue = RedisUpdateQueue::new(&options.redis_endpoint)
        .map_err(|error| CompositionError::UpdateWorkers(error.to_string()))?;
    let handler = DurableParallelUpdateHandler::start(
        UPDATE_WORKER_COUNT,
        UPDATE_QUEUE_CAPACITY,
        update_queue,
        move || build_native_dispatcher(worker_options.borrowed()),
    )
    .map_err(|error| CompositionError::UpdateWorkers(error.to_string()))?;
    Ok(PollingRuntime::new(source, handler))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use bot_adapters::billing_read::{BillingRepository, ChargeHistoryRow};
    use bot_adapters::chat_config::ChatConfigRepository;
    use bot_adapters::coinmarketcap::{
        BitcoinPriceRequest, CoinMarketCapTransport, HttpResponse as CoinMarketCapHttpResponse,
        TransportFailureKind as CoinMarketCapFailure,
    };
    use bot_adapters::criptoya::{
        CriptoYaRequest, CriptoYaTransport, HttpResponse as CriptoYaHttpResponse,
        TransportFailureKind as CriptoYaFailure,
    };
    use bot_adapters::finviz::{
        FinvizTransport, HttpResponse as FinvizHttpResponse, TransportFailureKind as FinvizFailure,
    };
    use bot_adapters::giphy::{
        GiphyTransport, HttpResponse as GiphyHttpResponse, SearchRequest,
        TransportFailureKind as GiphyFailure,
    };
    use bot_adapters::giphy_pool::GiphyPoolCache;
    use bot_adapters::link_preview::{
        LinkPreviewTransport, PreviewFailure, PreviewRequest, PreviewResponse,
    };
    use bot_adapters::polymarket::{
        HttpResponse as PolymarketHttpResponse, MidpointsRequest, PolymarketTransport,
        TransportFailureKind as PolymarketFailure,
    };
    use bot_adapters::redis_connection::RedisEndpoint;
    use bot_adapters::request_cache::RequestCache;
    use bot_adapters::stock_pool::StockPoolCache;
    use bot_adapters::telegram_http::{
        HttpResponse, TelegramRequest, TelegramTransport, TransportFailureKind,
    };
    use bot_adapters::telegram_polling::{PollFailure, PollOutcome};
    use bot_adapters::weather::{
        HttpResponse as WeatherHttpResponse, TransportFailureKind as WeatherFailure,
        WeatherRequest, WeatherTransport,
    };
    use bot_adapters::yahoo_finance::{
        HttpResponse as YahooHttpResponse, TransportFailureKind as YahooFailure, YahooChartRequest,
        YahooFinanceTransport, YahooSearchRequest,
    };
    use bot_core::greeting_commands::GreetingCategory;
    use bot_core::locale::Locale;
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
        ActionReceipt, ActionSink, BillingTransferSink, ElectionSource, GroupAuthorizer,
        MessageStateSink, OilPriceSource, RandomSource, RuntimeValues, StarPaymentSink,
        StockPriceSource, WeatherSource,
    };
    use crate::reconciliation::ActiveOperationRegistry;
    use crate::runtime::UpdateSource;

    use super::build_charge_history_page;

    use super::{
        CriptoYaRuloSource, NativeRuntimeOptions, OpenMeteoWeatherSource, PolymarketElectionSource,
        RedisCommandState, SystemRandomError, SystemRandomSource, SystemRuntimeValues,
        TelegramActionSink, TelegramActionSinkError, TelegramDeliveryCoordinator,
        TelegramGroupAuthorizer, TelegramUpdateSource, YahooOilPriceSource, YahooStockPriceSource,
        build_native_runtime, publish_telegram_commands,
    };
    use crate::ai_dispatch::AiConversationInput;
    use crate::conversation::ConversationToolFactory;
    use crate::dispatcher::RuloSource;

    fn integration_redis_endpoint() -> Option<RedisEndpoint> {
        let port = std::env::var("TEST_REDIS_PORT").ok()?.parse().ok()?;
        Some(RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
            port,
            password: std::env::var("TEST_REDIS_PASSWORD")
                .ok()
                .filter(|value| !value.is_empty()),
        })
    }

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    struct SequenceTransport {
        responses: RefCell<Vec<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<TelegramRequest>>,
    }

    #[derive(Clone)]
    struct BlockingTelegramTransport {
        active: Arc<AtomicUsize>,
        maximum: Arc<AtomicUsize>,
        entered: mpsc::Sender<()>,
        release: Arc<(Mutex<bool>, Condvar)>,
    }

    struct CriptoTransport {
        results: RefCell<Vec<Result<CriptoYaHttpResponse, CriptoYaFailure>>>,
        requests: RefCell<Vec<CriptoYaRequest>>,
    }

    struct WeatherTransportStub {
        response: RefCell<Option<Result<WeatherHttpResponse, WeatherFailure>>>,
    }

    struct YahooTransportStub {
        responses: RefCell<Vec<Result<YahooHttpResponse, YahooFailure>>>,
        requests: RefCell<Vec<YahooChartRequest>>,
    }

    struct StockYahooTransportStub {
        chart_responses: RefCell<Vec<Result<YahooHttpResponse, YahooFailure>>>,
        search_responses: RefCell<Vec<Result<YahooHttpResponse, YahooFailure>>>,
        charts: RefCell<Vec<YahooChartRequest>>,
        searches: RefCell<Vec<YahooSearchRequest>>,
    }

    impl YahooFinanceTransport for StockYahooTransportStub {
        fn chart(&self, request: &YahooChartRequest) -> Result<YahooHttpResponse, YahooFailure> {
            self.charts.borrow_mut().push(request.clone());
            if self.chart_responses.borrow().is_empty() {
                return Err(YahooFailure::Request);
            }
            self.chart_responses.borrow_mut().remove(0)
        }

        fn search(&self, request: &YahooSearchRequest) -> Result<YahooHttpResponse, YahooFailure> {
            self.searches.borrow_mut().push(request.clone());
            if self.search_responses.borrow().is_empty() {
                return Err(YahooFailure::Request);
            }
            self.search_responses.borrow_mut().remove(0)
        }
    }

    struct FinvizTransportStub {
        response: RefCell<Option<Result<FinvizHttpResponse, FinvizFailure>>>,
    }

    struct PolymarketTransportStub {
        events: RefCell<Option<Result<PolymarketHttpResponse, PolymarketFailure>>>,
        midpoints: RefCell<Option<Result<PolymarketHttpResponse, PolymarketFailure>>>,
        requests: RefCell<Vec<MidpointsRequest>>,
    }

    struct BitcoinTransportStub {
        response: RefCell<Option<Result<CoinMarketCapHttpResponse, CoinMarketCapFailure>>>,
    }

    impl CoinMarketCapTransport for BitcoinTransportStub {
        fn get(
            &self,
            _: &BitcoinPriceRequest,
        ) -> Result<CoinMarketCapHttpResponse, CoinMarketCapFailure> {
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(CoinMarketCapFailure::Request))
        }
    }

    struct GiphyTransportStub;

    impl GiphyTransport for GiphyTransportStub {
        fn search(&self, request: &SearchRequest) -> Result<GiphyHttpResponse, GiphyFailure> {
            Ok(GiphyHttpResponse {
                status_code: 200,
                body: serde_json::json!({
                    "data":[{"images":{"original":{"url":format!("https://images.example.test/{}.gif", request.term)}}}]
                })
                .to_string(),
            })
        }
    }

    struct LinkPreviewTransportStub;

    impl LinkPreviewTransport for LinkPreviewTransportStub {
        fn request(&self, request: &PreviewRequest) -> Result<PreviewResponse, PreviewFailure> {
            if request.url.contains("cdn.example.test") {
                return Ok(PreviewResponse {
                    status_code: 200,
                    final_url: request.url.clone(),
                    content_type: "video/mp4".to_owned(),
                    content_length: Some(25_000_000),
                    location: None,
                    body: String::new(),
                });
            }
            Ok(PreviewResponse {
                status_code: 200,
                final_url: request.url.clone(),
                content_type: "text/html".to_owned(),
                content_length: None,
                location: None,
                body: format!(
                    "<meta property='og:title' content='{}'><meta property='og:description' content='{}'><meta property='og:video' content='https://cdn.example.test/video.mp4'>",
                    "title ".repeat(40),
                    "description ".repeat(40)
                ),
            })
        }

        fn download_video(&self, _: &str, _: u64) -> Result<Vec<u8>, PreviewFailure> {
            Ok(vec![1, 2, 3])
        }
    }

    impl PolymarketTransport for PolymarketTransportStub {
        fn events(&self) -> Result<PolymarketHttpResponse, PolymarketFailure> {
            self.events
                .borrow_mut()
                .take()
                .unwrap_or(Err(PolymarketFailure::Request))
        }

        fn midpoints(
            &self,
            request: &MidpointsRequest,
        ) -> Result<PolymarketHttpResponse, PolymarketFailure> {
            self.requests.borrow_mut().push(request.clone());
            self.midpoints
                .borrow_mut()
                .take()
                .unwrap_or(Err(PolymarketFailure::Request))
        }
    }

    impl FinvizTransport for FinvizTransportStub {
        fn fetch(&self) -> Result<FinvizHttpResponse, FinvizFailure> {
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(FinvizFailure::Request))
        }
    }

    impl YahooFinanceTransport for YahooTransportStub {
        fn chart(&self, request: &YahooChartRequest) -> Result<YahooHttpResponse, YahooFailure> {
            self.requests.borrow_mut().push(request.clone());
            if self.responses.borrow().is_empty() {
                return Err(YahooFailure::Request);
            }
            self.responses.borrow_mut().remove(0)
        }

        fn search(&self, _request: &YahooSearchRequest) -> Result<YahooHttpResponse, YahooFailure> {
            Err(YahooFailure::Request)
        }
    }

    impl WeatherTransport for WeatherTransportStub {
        fn get(&self, _request: &WeatherRequest) -> Result<WeatherHttpResponse, WeatherFailure> {
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(WeatherFailure::Request))
        }
    }

    #[derive(Default)]
    struct WeatherCacheStub;

    impl RequestCache for WeatherCacheStub {
        type Error = std::convert::Infallible;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(None)
        }

        fn set(&mut self, _key: &str, _value: &str, _ttl_seconds: i64) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl StockPoolCache for WeatherCacheStub {
        type Error = std::convert::Infallible;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(None)
        }

        fn set(&mut self, _key: &str, _value: &str, _ttl_seconds: i64) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl GiphyPoolCache for WeatherCacheStub {
        type Error = std::convert::Infallible;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(None)
        }

        fn set(&mut self, _key: &str, _value: &str, _ttl_seconds: i64) -> Result<(), Self::Error> {
            Ok(())
        }
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

    impl TelegramTransport for SequenceTransport {
        fn send(&self, request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            if self.responses.borrow().is_empty() {
                return Err(TransportFailureKind::Request);
            }
            self.responses.borrow_mut().remove(0)
        }
    }

    impl TelegramTransport for BlockingTelegramTransport {
        fn send(&self, _request: &TelegramRequest) -> Result<HttpResponse, TransportFailureKind> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.maximum.fetch_max(active, Ordering::SeqCst);
            let _ = self.entered.send(());
            let (released, wake) = &*self.release;
            if let Ok(guard) = released.lock() {
                let _guard = wake.wait_while(guard, |released| !*released);
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(HttpResponse {
                status_code: 200,
                body: r#"{"ok":true,"result":{"message_id":9}}"#.to_owned(),
            })
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

    fn telegram_response(
        status_code: u16,
        body: &str,
    ) -> Result<HttpResponse, TransportFailureKind> {
        Ok(HttpResponse {
            status_code,
            body: body.to_owned(),
        })
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
                .json_payload
                .as_ref()
                .and_then(|payload| payload.get("offset")),
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

        let rate_limit = r#"{"ok":false,"error_code":429,"parameters":{"retry_after":4}}"#;
        let transport = SequenceTransport {
            responses: RefCell::new(vec![
                telegram_response(429, rate_limit),
                telegram_response(429, rate_limit),
                telegram_response(429, rate_limit),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        let waits = Arc::new(Mutex::new(Vec::new()));
        let recorded_waits = waits.clone();
        let mut sink = TelegramActionSink::new(transport, "token").with_wait(move |duration| {
            if let Ok(mut waits) = recorded_waits.lock() {
                waits.push(duration);
            }
        });
        assert_eq!(
            sink.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(1),
                "hello"
            ))),
            Err(TelegramActionSinkError::RateLimited {
                retry_after_seconds: Some(4)
            })
        );
        assert_eq!(sink.transport.requests.borrow().len(), 3);
        assert!(waits.lock().is_ok_and(|waits| {
            waits.as_slice() == [Duration::from_secs(4), Duration::from_secs(4)]
        }));
    }

    #[test]
    fn action_sink_retries_rate_limits_and_delivers_the_original_action() {
        let transport = SequenceTransport {
            responses: RefCell::new(vec![
                telegram_response(
                    429,
                    r#"{"ok":false,"error_code":429,"parameters":{"retry_after":2}}"#,
                ),
                telegram_response(200, r#"{"ok":true,"result":{"message_id":9}}"#),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        let waits = Arc::new(Mutex::new(Vec::new()));
        let recorded_waits = waits.clone();
        let mut sink = TelegramActionSink::new(transport, "token").with_wait(move |duration| {
            if let Ok(mut waits) = recorded_waits.lock() {
                waits.push(duration);
            }
        });
        let action = TelegramAction::SendMessage(SendMessage::new(ChatId(1), "hello"));

        assert_eq!(
            sink.execute(action),
            Ok(ActionReceipt {
                message_id: Some(MessageId(9))
            })
        );
        assert_eq!(sink.transport.requests.borrow().len(), 2);
        assert!(
            waits
                .lock()
                .is_ok_and(|waits| waits.as_slice() == [Duration::from_secs(2)])
        );
    }

    #[test]
    fn action_sinks_coordinate_delivery_to_the_same_chat() {
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let (entered_sender, entered_receiver) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let transport = BlockingTelegramTransport {
            active: active.clone(),
            maximum: maximum.clone(),
            entered: entered_sender,
            release: release.clone(),
        };
        let delivery = TelegramDeliveryCoordinator::default();
        let mut first = TelegramActionSink::new(transport.clone(), "token")
            .with_delivery_coordinator(delivery.clone());
        let mut second =
            TelegramActionSink::new(transport, "token").with_delivery_coordinator(delivery);

        let first_send = thread::spawn(move || {
            first.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(777),
                "first",
            )))
        });
        assert_eq!(
            entered_receiver.recv_timeout(Duration::from_secs(1)),
            Ok(())
        );
        let second_send = thread::spawn(move || {
            second.execute(TelegramAction::SendMessage(SendMessage::new(
                ChatId(777),
                "second",
            )))
        });
        assert!(
            entered_receiver
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );

        if let Ok(mut released) = release.0.lock() {
            *released = true;
            release.1.notify_all();
        }
        assert!(first_send.join().is_ok_and(|result| result.is_ok()));
        assert!(second_send.join().is_ok_and(|result| result.is_ok()));
        assert_eq!(maximum.load(Ordering::SeqCst), 1);
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
                let (stream, _) = listener.accept()?;
                let mut reader = BufReader::new(stream);
                assert_eq!(
                    read_command(&mut reader)?.first().map(String::as_str),
                    Some("FT.CREATE")
                );
                reader.get_mut().write_all(b"-Index already exists\r\n")?;

                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("EVAL"));
                assert!(command.iter().any(|value| value == "chat_history:-42"));
                reader.get_mut().write_all(b":1\r\n")?;

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

                let command = read_command(&mut reader)?;
                assert_eq!(command.first().map(String::as_str), Some("EVAL"));
                assert!(command.iter().any(|value| value == "bot_99"));
                reader.get_mut().write_all(b":1\r\n")?;

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
            openrouter_api_key: None,
            openrouter_base_url: None,
            groq_free_api_key: None,
            groq_api_key: None,
            firecrawl_api_key: None,
            system_prompt: None,
            trigger_words: None,
            active_operations: ActiveOperationRegistry::default(),
            telegram_delivery: TelegramDeliveryCoordinator::default(),
        });
        assert!(result.is_ok());
        let result = build_native_runtime(NativeRuntimeOptions {
            token: "synthetic-token",
            database_url: "postgresql://synthetic.invalid/database",
            bot_name: "@synthetic_bot",
            instance_name: Some("synthetic-instance".to_owned()),
            redis_endpoint: &endpoint,
            long_poll_timeout: Duration::from_secs(30),
            admin_user_id: Some(99),
            coinmarketcap_key: Some("synthetic-cmc-key".to_owned()),
            giphy_api_key: None,
            openrouter_api_key: None,
            openrouter_base_url: None,
            groq_free_api_key: None,
            groq_api_key: None,
            firecrawl_api_key: None,
            system_prompt: None,
            trigger_words: None,
            active_operations: ActiveOperationRegistry::default(),
            telegram_delivery: TelegramDeliveryCoordinator::default(),
        });
        assert!(result.is_ok());
        let result = build_native_runtime(NativeRuntimeOptions {
            token: "synthetic-token",
            database_url: "postgresql://synthetic.invalid/database",
            bot_name: "@synthetic_bot",
            instance_name: Some("synthetic-instance".to_owned()),
            redis_endpoint: &endpoint,
            long_poll_timeout: Duration::from_secs(30),
            admin_user_id: Some(99),
            coinmarketcap_key: Some("synthetic-cmc-key".to_owned()),
            giphy_api_key: None,
            openrouter_api_key: Some("synthetic-openrouter-key".to_owned()),
            openrouter_base_url: Some("https://openrouter.example.test/v1".to_owned()),
            groq_free_api_key: Some("synthetic-groq-free-key".to_owned()),
            groq_api_key: Some("synthetic-groq-paid-key".to_owned()),
            firecrawl_api_key: Some("synthetic-firecrawl-key".to_owned()),
            system_prompt: Some("synthetic persona".to_owned()),
            trigger_words: Some(vec!["synthetic".to_owned()]),
            active_operations: ActiveOperationRegistry::default(),
            telegram_delivery: TelegramDeliveryCoordinator::default(),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn production_dispatcher_composes_every_optional_feature_with_real_local_stores() {
        let Some(endpoint) = integration_redis_endpoint() else {
            return;
        };
        let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
            return;
        };
        let result = super::build_native_dispatcher(NativeRuntimeOptions {
            token: "synthetic-token",
            database_url: &database_url,
            bot_name: "@synthetic_bot",
            instance_name: Some("synthetic-instance".to_owned()),
            redis_endpoint: &endpoint,
            long_poll_timeout: Duration::from_secs(30),
            admin_user_id: Some(99),
            coinmarketcap_key: Some("synthetic-market-key".to_owned()),
            giphy_api_key: Some("synthetic-image-key".to_owned()),
            openrouter_api_key: Some("synthetic-ai-key".to_owned()),
            openrouter_base_url: Some("https://provider.example.test/v1".to_owned()),
            groq_free_api_key: Some("synthetic-audio-free-key".to_owned()),
            groq_api_key: Some("synthetic-audio-paid-key".to_owned()),
            firecrawl_api_key: Some("synthetic-search-key".to_owned()),
            system_prompt: Some("synthetic system prompt".to_owned()),
            trigger_words: Some(vec!["synthetic".to_owned()]),
            active_operations: ActiveOperationRegistry::default(),
            telegram_delivery: TelegramDeliveryCoordinator::default(),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn production_tool_factory_exposes_all_configured_tools_with_local_stores() {
        let Some(endpoint) = integration_redis_endpoint() else {
            return;
        };
        let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
            return;
        };
        let mut factory = super::ProductionToolFactory::new(
            &endpoint,
            &database_url,
            Some("synthetic-market-key".to_owned()),
            Some("synthetic-search-key".to_owned()),
        );
        let input = AiConversationInput {
            chat_id: ChatId(-42),
            message_id: MessageId(7),
            chat_type: "supergroup".to_owned(),
            chat_title: "Synthetic chat".to_owned(),
            sender_id: UserId(9),
            sender_first_name: "Synthetic".to_owned(),
            sender_username: "synthetic_user".to_owned(),
            message_text: "synthetic question".to_owned(),
            command: String::new(),
            reply_to_message_id: None,
            reply_context: None,
            has_reply: false,
            visual_media_kind: None,
            audio_media_kind: None,
            photo_file_id: None,
            audio_file_id: None,
            audio_duration_seconds: None,
            locale: Locale::En,
            timezone_offset_hours: i64::from(i32::MAX) + 1,
            creditless_user_hourly_limit: 10,
            timestamp: 1_700_000_000,
            spontaneous: false,
        };
        assert!(factory.create(&input).is_ok());

        let mut minimal = super::ProductionToolFactory::new(&endpoint, &database_url, None, None);
        assert!(minimal.create(&input).is_ok());
    }

    #[test]
    fn link_replacement_includes_bounded_preview_context_and_large_video() {
        let mut source = super::NativeLinkReplacementSource {
            transport: LinkPreviewTransportStub,
        };
        let load = crate::dispatcher::LinkReplacementSource::load(
            &mut source,
            "see https://www.instagram.com/p/synthetic/",
            1_700_000_000,
        );
        assert!(load.replacement.changed);
        assert!(load.replacement.text.contains("eeinstagram"));
        assert!(load.context.as_deref().is_some_and(|context| {
            context.contains("LINKS DEL MENSAJE:")
                && context.contains("titulo:")
                && context.contains("descripcion:")
        }));
        assert_eq!(load.oversized_video, Some(vec![1, 2, 3]));
        assert!(load.diagnostics.is_empty());
        assert_eq!(super::bounded_link_text(Some("   "), 10), None);
        assert_eq!(super::bounded_link_text(None, 10), None);
    }

    #[test]
    fn provider_source_wrappers_preserve_success_missing_and_failure_semantics() {
        use crate::dispatcher::{BitcoinPriceSource, DollarQuotesSource, GreetingPoolSource};

        for (response, expected) in [
            (
                Ok(CoinMarketCapHttpResponse {
                    status_code: 200,
                    body: serde_json::json!({"data":[{"quote":{"USD":{"price":42.5}}}]})
                        .to_string(),
                }),
                Ok(Some(42.5)),
            ),
            (
                Ok(CoinMarketCapHttpResponse {
                    status_code: 404,
                    body: String::new(),
                }),
                Err("CoinMarketCap returned HTTP 404".to_owned()),
            ),
            (
                Err(CoinMarketCapFailure::Timeout),
                Err("CoinMarketCap transport failed: Timeout".to_owned()),
            ),
        ] {
            let mut source = super::CoinMarketCapBitcoinPriceSource {
                transport: BitcoinTransportStub {
                    response: RefCell::new(Some(response)),
                },
                api_key: "synthetic-key".to_owned(),
            };
            assert_eq!(source.price("USD"), expected);
        }

        let mut dollar = super::CriptoYaDollarQuotesSource {
            transport: CriptoTransport {
                results: RefCell::new(vec![Ok(CriptoYaHttpResponse {
                    status_code: 200,
                    body: serde_json::json!({
                        "oficial":{"price":100},
                        "tarjeta":{"price":150},
                        "cripto":{"usdt":{"ask":200,"bid":190}}
                    })
                    .to_string(),
                })]),
                requests: RefCell::new(Vec::new()),
            },
        };
        assert!(dollar.devo_quotes().is_ok_and(|quotes| quotes.is_some()));

        let mut greetings = super::GiphyGreetingPoolSource {
            transport: GiphyTransportStub,
            cache: WeatherCacheStub,
            api_key: Some("synthetic-key".to_owned()),
        };
        let load = greetings.pool(GreetingCategory::Morning);
        assert!(!load.urls.is_empty());
        assert!(load.diagnostics.is_empty());

        for (response, expected) in [
            (
                Ok(CoinMarketCapHttpResponse {
                    status_code: 200,
                    body: serde_json::json!({"data":[]}).to_string(),
                }),
                Ok(None),
            ),
            (
                Ok(CoinMarketCapHttpResponse {
                    status_code: 200,
                    body: "not-json".to_owned(),
                }),
                Err("CoinMarketCap returned invalid JSON".to_owned()),
            ),
        ] {
            let mut source = super::CoinMarketCapBitcoinPriceSource {
                transport: BitcoinTransportStub {
                    response: RefCell::new(Some(response)),
                },
                api_key: "synthetic-key".to_owned(),
            };
            assert_eq!(source.price("USD"), expected);
        }

        for (result, expected) in [
            (
                Ok(CriptoYaHttpResponse {
                    status_code: 200,
                    body: serde_json::json!({
                        "oficial":{"price":"NaN"},
                        "tarjeta":{"price":150},
                        "cripto":{"usdt":{"ask":200,"bid":190}}
                    })
                    .to_string(),
                }),
                Ok(None),
            ),
            (
                Ok(CriptoYaHttpResponse {
                    status_code: 503,
                    body: String::new(),
                }),
                Err("CriptoYa returned HTTP 503".to_owned()),
            ),
            (
                Ok(CriptoYaHttpResponse {
                    status_code: 200,
                    body: "not-json".to_owned(),
                }),
                Err("CriptoYa returned invalid JSON".to_owned()),
            ),
            (
                Err(CriptoYaFailure::Timeout),
                Err("CriptoYa transport failed: Timeout".to_owned()),
            ),
        ] {
            let mut source = super::CriptoYaDollarQuotesSource {
                transport: CriptoTransport {
                    results: RefCell::new(vec![result]),
                    requests: RefCell::new(Vec::new()),
                },
            };
            assert_eq!(source.devo_quotes(), expected);
        }

        let exchange = |body: &str| {
            Ok(CriptoYaHttpResponse {
                status_code: 200,
                body: body.to_owned(),
            })
        };
        let mut rulo = CriptoYaRuloSource {
            transport: CriptoTransport {
                results: RefCell::new(vec![
                    exchange(r#"{"oficial":{"price":100},"blue":{"bid":110}}"#),
                    exchange(r#"{"synthetic_exchange":{"totalAsk":1.1}}"#),
                    exchange(r#"{"synthetic_exchange":{"totalBid":120}}"#),
                ]),
                requests: RefCell::new(Vec::new()),
            },
        };
        let input = rulo.rulo_input().unwrap_or_else(|_| unreachable!());
        assert_eq!(input.input.official, Some(100.0));
        assert_eq!(input.input.usd_to_usdt.len(), 1);
        assert_eq!(input.input.usdt_to_ars.len(), 1);
        assert!(input.diagnostics.is_empty());

        for (result, expected) in [
            (
                Ok(CriptoYaHttpResponse {
                    status_code: 200,
                    body: "not-json".to_owned(),
                }),
                "CriptoYa dollar market returned invalid JSON",
            ),
            (
                Ok(CriptoYaHttpResponse {
                    status_code: 429,
                    body: String::new(),
                }),
                "CriptoYa dollar market returned HTTP 429",
            ),
            (
                Err(CriptoYaFailure::Connection),
                "CriptoYa dollar market transport failed: Connection",
            ),
        ] {
            let mut source = CriptoYaRuloSource {
                transport: CriptoTransport {
                    results: RefCell::new(vec![result]),
                    requests: RefCell::new(Vec::new()),
                },
            };
            assert_eq!(source.rulo_input(), Err(expected.to_owned()));
        }

        let mut partial_rulo = CriptoYaRuloSource {
            transport: CriptoTransport {
                results: RefCell::new(vec![
                    exchange("{}"),
                    Err(CriptoYaFailure::Timeout),
                    Ok(CriptoYaHttpResponse {
                        status_code: 503,
                        body: String::new(),
                    }),
                ]),
                requests: RefCell::new(Vec::new()),
            },
        };
        let input = partial_rulo.rulo_input().unwrap_or_else(|_| unreachable!());
        assert_eq!(input.diagnostics.len(), 2);
    }

    #[test]
    fn postgres_dispatcher_ports_round_trip_balances_payments_history_and_chat_config()
    -> Result<(), String> {
        use crate::dispatcher::{
            AdminCreditLogSource, AdminCreditSink, BillingBalanceSource, BillingTransferSink,
            ChargeHistorySource, ChatConfigSource, StarPaymentSink,
        };

        let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
            return Ok(());
        };
        bot_adapters::billing_schema::BillingSchemaRepository::new(&database_url)
            .ensure_schema()
            .map_err(|error| error.to_string())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let suffix = i64::try_from(nonce % 100_000_000).map_err(|error| error.to_string())?;
        let user_id = 7_300_000_000_000_i64 + suffix;
        let chat_id = -7_400_000_000_000_i64 - suffix;
        let mut billing = BillingRepository::new(&database_url);
        assert_eq!(AdminCreditSink::mint(&mut billing, user_id, 100)?, 100);
        let balances = BillingBalanceSource::load(&mut billing, user_id, Some(chat_id))?;
        assert!(matches!(balances.user_balance, 100 | 400));
        assert_eq!(balances.chat_balance, Some(0));
        let initial_user_balance = balances.user_balance;
        let transfer = BillingTransferSink::transfer(&mut billing, user_id, chat_id, 25)?;
        assert!(transfer.transferred);
        assert_eq!(transfer.user_balance, initial_user_balance - 25);
        assert_eq!(transfer.chat_balance, 25);

        let payment = StarPaymentRecord {
            charge_id: format!("synthetic-charge-{nonce}"),
            user_id,
            pack_id: "synthetic-pack".to_owned(),
            xtr_amount: 5,
            credits_awarded: 50,
            payload: format!("synthetic:{nonce}"),
        };
        let receipt = StarPaymentSink::record(&mut billing, &payment)?;
        assert!(receipt.inserted);
        assert_eq!(receipt.user_balance, initial_user_balance + 25);
        let replay = StarPaymentSink::record(&mut billing, &payment)?;
        assert!(!replay.inserted);
        assert_eq!(replay.user_balance, initial_user_balance + 25);

        let operation_id = format!("synthetic-history-{nonce}");
        let metadata = serde_json::Map::from_iter([
            ("operation_id".to_owned(), serde_json::json!(&operation_id)),
            ("credit_scale".to_owned(), serde_json::json!(100)),
        ]);
        let reserve = billing
            .charge_ai_credits(
                user_id,
                None,
                10,
                "ai_reserve",
                &metadata,
                Some("user"),
                Some(&operation_id),
                &operation_id,
            )
            .map_err(|error| error.to_string())?;
        assert!(reserve.ok);
        billing
            .settle_ai_operation_once(user_id, None, &operation_id, 2, &metadata)
            .map_err(|error| error.to_string())?;

        let history = ChargeHistorySource::load(&mut billing, user_id, 10, None, "older")?;
        assert!(!history.groups.is_empty());
        assert!(!AdminCreditLogSource::load(&mut billing, 10)?.is_empty());

        let mut configs = ChatConfigRepository::new(&database_url);
        let previous = ChatConfigSource::get(&mut configs, &chat_id.to_string())
            .map_err(|error| error.to_string())?;
        let mut changed = previous.clone();
        changed.language = "en".to_owned();
        let stored =
            ChatConfigSource::set_changed(&mut configs, &chat_id.to_string(), &previous, &changed)
                .map_err(|error| error.to_string())?;
        assert_eq!(stored.language, "en");
        assert_eq!(
            ChatConfigSource::get(&mut configs, &chat_id.to_string())
                .map_err(|error| error.to_string())?
                .language,
            "en"
        );
        Ok(())
    }

    #[test]
    fn weather_source_maps_the_typed_adapter_load() {
        let transport = WeatherTransportStub {
            response: RefCell::new(Some(Ok(WeatherHttpResponse {
                status_code: 200,
                body: r#"{"current":{"time":"2026-01-02T10:00"},"hourly":{"time":["2026-01-02T10:00"],"apparent_temperature":[19.5],"precipitation_probability":[20],"weather_code":[1],"cloud_cover":[30],"visibility":[15000]}}"#.to_owned(),
            }))),
        };
        let mut source = OpenMeteoWeatherSource {
            transport,
            cache: WeatherCacheStub,
        };
        let load = source.load("CABA", 1_767_345_000);
        assert_eq!(
            load.observation.map(|observation| observation.location),
            Some("Buenos Aires, Argentina".to_owned())
        );
        assert!(load.diagnostics.is_empty());
    }

    #[test]
    fn oil_source_loads_both_yahoo_contract_symbols() {
        let chart = |symbol: &str, price: f64| YahooHttpResponse {
            status_code: 200,
            body: format!(
                r#"{{"chart":{{"result":[{{"meta":{{"symbol":"{symbol}","regularMarketPrice":{price},"chartPreviousClose":100,"currency":"USD"}}}}]}}}}"#
            ),
        };
        let transport = YahooTransportStub {
            responses: RefCell::new(vec![Ok(chart("BZ=F", 98.15)), Ok(chart("CL=F", 95.45))]),
            requests: RefCell::default(),
        };
        let mut source = YahooOilPriceSource {
            transport,
            cache: WeatherCacheStub,
        };
        let load = source.load(100);
        assert_eq!(load.brent.map(|quote| quote.price), Some(98.15));
        assert_eq!(load.wti.map(|quote| quote.price), Some(95.45));
        assert!(load.diagnostics.is_empty());
        assert_eq!(
            source
                .transport
                .requests
                .borrow()
                .iter()
                .map(|request| request.symbol.as_str())
                .collect::<Vec<_>>(),
            vec!["BZ=F", "CL=F"]
        );
    }

    #[test]
    fn election_source_maps_cached_events_and_batch_live_prices() {
        let transport = PolymarketTransportStub {
            events: RefCell::new(Some(Ok(PolymarketHttpResponse {
                status_code: 200,
                body: r#"[{"title":"Election","slug":"election","liquidity":1000,"markets":[{"groupItemTitle":"A","outcomes":["Yes","No"],"outcomePrices":[0.4,0.6],"clobTokenIds":["a","a-no"]}]}]"#.to_owned(),
            }))),
            midpoints: RefCell::new(Some(Ok(PolymarketHttpResponse {
                status_code: 200,
                body: r#"{"a":"0.72"}"#.to_owned(),
            }))),
            requests: RefCell::default(),
        };
        let mut source = PolymarketElectionSource {
            transport,
            cache: WeatherCacheStub,
        };
        let load = source.load(100);
        assert_eq!(load.events.len(), 1);
        assert_eq!(load.live_prices.get("a"), Some(&0.72));
        assert!(load.diagnostics.is_empty());
        assert_eq!(source.transport.requests.borrow()[0].token_ids, ["a"]);
    }

    #[test]
    fn stock_source_resolves_a_multiword_company_as_one_quote() {
        let empty_chart = || {
            Ok(YahooHttpResponse {
                status_code: 200,
                body: r#"{"chart":{"result":[]}}"#.to_owned(),
            })
        };
        let apple_chart = Ok(YahooHttpResponse {
            status_code: 200,
            body: r#"{"chart":{"result":[{"meta":{"symbol":"AAPL","regularMarketPrice":205.5,"chartPreviousClose":200,"currency":"USD"}}]}}"#.to_owned(),
        });
        let transport = StockYahooTransportStub {
            chart_responses: RefCell::new(vec![empty_chart(), empty_chart(), apple_chart]),
            search_responses: RefCell::new(vec![Ok(YahooHttpResponse {
                status_code: 200,
                body: r#"{"quotes":[{"quoteType":"EQUITY","symbol":"AAPL"}]}"#.to_owned(),
            })]),
            charts: RefCell::default(),
            searches: RefCell::default(),
        };
        let mut source = YahooStockPriceSource {
            yahoo_transport: transport,
            finviz_transport: FinvizTransportStub {
                response: RefCell::new(None),
            },
            cache: WeatherCacheStub,
        };
        let load = source.load("Apple Inc", 100);
        let quotes = load.quotes.unwrap_or_default();
        assert_eq!(quotes.len(), 1);
        assert_eq!(quotes[0].0, "Apple Inc");
        assert_eq!(
            quotes[0].1.as_ref().map(|quote| quote.symbol.as_str()),
            Some("AAPL")
        );
        assert_eq!(
            source
                .yahoo_transport
                .charts
                .borrow()
                .iter()
                .map(|request| request.symbol.as_str())
                .collect::<Vec<_>>(),
            ["APPLE", "INC", "AAPL"]
        );
        assert_eq!(
            source.yahoo_transport.searches.borrow()[0].query,
            "Apple Inc"
        );
    }

    #[test]
    fn stock_source_uses_finviz_pool_for_an_empty_query() {
        let chart = |symbol: &str, price: f64| {
            Ok(YahooHttpResponse {
                status_code: 200,
                body: format!(
                    r#"{{"chart":{{"result":[{{"meta":{{"symbol":"{symbol}","regularMarketPrice":{price},"chartPreviousClose":100,"currency":"USD"}}}}]}}}}"#
                ),
            })
        };
        let transport = StockYahooTransportStub {
            chart_responses: RefCell::new(vec![chart("AAPL", 205.5), chart("MSFT", 150.0)]),
            search_responses: RefCell::default(),
            charts: RefCell::default(),
            searches: RefCell::default(),
        };
        let mut source = YahooStockPriceSource {
            yahoo_transport: transport,
            finviz_transport: FinvizTransportStub {
                response: RefCell::new(Some(Ok(FinvizHttpResponse {
                    status_code: 200,
                    body: concat!(
                        r#"data-boxover-ticker="AAPL" data-boxover-company="Apple" "#,
                        r#"data-boxover-ticker="MSFT" data-boxover-company="Microsoft""#
                    )
                    .to_owned(),
                }))),
            },
            cache: WeatherCacheStub,
        };
        let load = source.load("", 100);
        let quotes = load.quotes.unwrap_or_default();
        assert_eq!(quotes.len(), 2);
        assert!(quotes.iter().all(|(_, quote)| quote.is_some()));
        assert!(source.yahoo_transport.searches.borrow().is_empty());
    }
}
