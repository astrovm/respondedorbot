//! Unified cryptocurrency and stock price command behavior.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::locale::Locale;
use crate::price_queries::{
    AmountConversion, ChartPeriod, PriceQuery, ProviderScope, parse_price_query,
    price_query_parameter,
};
use crate::stocks::StockQuote;
use crate::token_signals::TokenAddress;

const TIMEFRAMES: [&str; 4] = ["1h", "24h", "7d", "30d"];
const STABLECOINS: [&str; 26] = [
    "BUSD", "DAI", "DOC", "EURT", "FDUSD", "FRAX", "GHO", "GUSD", "LUSD", "MAI", "MIM", "MIMATIC",
    "NUARS", "PAXG", "PYUSD", "RAI", "SUSD", "TUSD", "USDC", "USDD", "USDM", "USDP", "USDT", "UXD",
    "XAUT", "XSGD",
];
const UNAMBIGUOUS_CRYPTO_SYMBOLS: [&str; 12] = [
    "BTC", "BITCOIN", "ETH", "ETHEREUM", "SATS", "XMR", "MONERO", "SOL", "SOLANA", "BNB", "XRP",
    "DOGE",
];
const SUPPORTED_CURRENCIES: [&str; 35] = [
    "ARS", "AUD", "BRL", "BTC", "BUSD", "CAD", "CHF", "CLP", "CNY", "COP", "CZK", "DAI", "DKK",
    "ETH", "EUR", "GBP", "HKD", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "NZD", "PEN", "SATS",
    "SEK", "SGD", "TWD", "USD", "USDC", "USDT", "UYU", "XAU", "XMR",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketPriceCommand {
    Unified,
    CryptoOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CryptoQuote {
    pub price: f64,
    pub market_cap: Option<f64>,
    pub volume_24h: Option<f64>,
    pub percent_change_1h: Option<f64>,
    pub percent_change_24h: Option<f64>,
    pub percent_change_7d: Option<f64>,
    pub percent_change_30d: Option<f64>,
}

impl CryptoQuote {
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.price.is_finite() && self.price > 0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CryptoAsset {
    pub id: String,
    pub symbol: String,
    pub name: String,
    pub slug: String,
    pub quotes: HashMap<String, CryptoQuote>,
    pub contracts: Vec<TokenAddress>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketCandidate {
    pub id: String,
    pub symbol: String,
    pub name: String,
    pub slug: String,
    pub price: String,
    pub change: String,
    pub contracts: Vec<TokenAddress>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketConversion {
    /// Serialized so callback state remains compatible with Eq and avoids
    /// reformatting a user-supplied decimal amount after selection.
    pub amount: String,
    pub source_symbol: String,
    pub target_symbol: String,
    /// The selected asset is the quote currency in a reverse conversion.
    pub reverse: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketSelection {
    pub query: String,
    pub timeframe: Option<String>,
    pub target_symbol: String,
    pub target_parameter: String,
    #[serde(default)]
    pub conversion: Option<MarketConversion>,
    pub candidates: Vec<MarketCandidate>,
}

#[must_use]
pub fn format_market_selection(selection: &MarketSelection, locale: Locale) -> String {
    let heading = match locale {
        Locale::Es => "Encontré varias monedas con ese ticker:",
        Locale::En => "I found several coins with that ticker:",
    };
    let conversion = selection
        .conversion
        .as_ref()
        .map_or_else(String::new, |conversion| {
            format!(
                "\n{} {} in {}:",
                conversion.amount, conversion.source_symbol, conversion.target_symbol
            )
        });
    let target = selection
        .conversion
        .as_ref()
        .filter(|conversion| conversion.reverse)
        .map_or_else(
            || {
                if selection.target_symbol.is_empty() {
                    "USD".to_owned()
                } else {
                    selection.target_symbol.clone()
                }
            },
            |conversion| conversion.source_symbol.clone(),
        );
    let lines = selection
        .candidates
        .iter()
        .take(10)
        .enumerate()
        .map(|(index, candidate)| {
            let name = if candidate.name.is_empty() {
                candidate.slug.as_str()
            } else {
                candidate.name.as_str()
            };
            let identity = candidate_identity(candidate);
            format!(
                "{}. {} ({}) — {} {} ({}; {})",
                index + 1,
                shorten(name, 100),
                shorten(&candidate.symbol, 24),
                shorten(&candidate.price, 32),
                target,
                shorten(&candidate.change, 16),
                identity,
            )
        })
        .collect::<Vec<_>>();
    let text = format!("{heading}{conversion}\n{}", lines.join("\n"));
    shorten(&text, crate::telegram_actions::MAX_TELEGRAM_TEXT_LENGTH)
}

fn candidate_identity(candidate: &MarketCandidate) -> String {
    let mut parts = Vec::new();
    if !candidate.id.trim().is_empty() {
        parts.push(format!("CMC #{}", shorten(&candidate.id, 24)));
    }
    for contract in &candidate.contracts {
        let address = shorten_address(&contract.address);
        parts.push(format!(
            "{}:{} {}",
            contract.chain_id, contract.tag, address
        ));
    }
    if parts.is_empty() {
        shorten(&candidate.slug, 48)
    } else {
        parts.join(" / ")
    }
}

fn shorten(value: &str, limit: usize) -> String {
    let value = value.trim();
    if value.chars().count() <= limit {
        return value.to_owned();
    }
    let mut result = value
        .chars()
        .take(limit.saturating_sub(3))
        .collect::<String>();
    result.push_str("...");
    result
}

fn shorten_address(value: &str) -> String {
    let value = value.trim();
    if value.chars().count() <= 18 {
        return value.to_owned();
    }
    let start = value.chars().take(8).collect::<String>();
    let end = value
        .chars()
        .rev()
        .take(6)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{start}…{end}")
}

pub trait CryptoMarketProvider {
    fn listings(&mut self, currency: &str) -> Result<Vec<CryptoAsset>, String>;
    fn quotes(
        &mut self,
        identifiers: &[String],
        currency: &str,
        by_slug: bool,
    ) -> Result<Vec<CryptoAsset>, String>;

    fn quotes_by_id(
        &mut self,
        identifiers: &[String],
        currency: &str,
    ) -> Result<Vec<CryptoAsset>, String> {
        self.quotes(identifiers, currency, false)
    }
}

pub type StockLookupRows = Vec<(String, Option<StockQuote>)>;

pub trait UnifiedStockProvider {
    fn lookup(&mut self, query: &str) -> Result<Option<StockLookupRows>, String>;
}

/// Identity selected by market resolution, used to fetch a chart without searching again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketChart {
    pub timeframe: Option<String>,
    pub symbol: String,
    pub name: String,
    pub yahoo_symbol: String,
    pub token: Option<TokenAddress>,
}

fn stock_chart(quote: &StockQuote) -> MarketChart {
    MarketChart {
        timeframe: None,
        symbol: quote.symbol.clone(),
        name: quote.name.clone(),
        yahoo_symbol: quote.symbol.clone(),
        token: None,
    }
}

fn company_matches(query: &str, quote: &StockQuote) -> bool {
    let normalized = |value: &str| {
        value
            .to_ascii_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| {
                !word.is_empty()
                    && !matches!(
                        *word,
                        "inc"
                            | "incorporated"
                            | "corp"
                            | "corporation"
                            | "ltd"
                            | "limited"
                            | "plc"
                            | "company"
                            | "co"
                    )
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    quote
        .symbol
        .eq_ignore_ascii_case(query.trim().trim_start_matches('$'))
        || normalized(&quote.name) == normalized(query)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketPriceExecution {
    pub chart: Option<MarketChart>,
    pub selection: Option<MarketSelection>,
    pub no_assets_found: bool,
    pub text: String,
    pub diagnostics: Vec<String>,
}

#[must_use]
pub fn classify_market_price_command(command: &str) -> Option<MarketPriceCommand> {
    match command {
        "/c" | "/cripto" | "/criptos" | "/crypto" | "/cryptos" => {
            Some(MarketPriceCommand::CryptoOnly)
        }
        "/prices" | "/price" | "/precios" | "/precio" | "/presios" | "/presio" | "/bresio"
        | "/bresios" | "/brecio" | "/brecios" | "/p" => Some(MarketPriceCommand::Unified),
        _ => None,
    }
}

pub fn execute_market_price_command<C: CryptoMarketProvider, S: UnifiedStockProvider>(
    text: &str,
    command: MarketPriceCommand,
    locale: Locale,
    crypto: &mut C,
    stocks: &mut S,
) -> MarketPriceExecution {
    let mut valid = TIMEFRAMES.map(str::to_owned).to_vec();
    if let Some(period) = text.split_whitespace().last()
        && ChartPeriod::parse(period).is_some()
    {
        valid.push(period.to_ascii_lowercase());
    }
    let query = parse_price_query(text, &valid);
    let mut diagnostics = Vec::new();
    let mut no_assets_found = false;
    let mut chart = None;
    let mut selection = None;
    let rendered = match query {
        PriceQuery::UnsupportedTimeframe { timeframe } => invalid_timeframe(&timeframe, locale),
        PriceQuery::AmountConversion(request) => convert_amount(
            &request,
            locale,
            crypto,
            &mut diagnostics,
            &mut no_assets_found,
            &mut selection,
        ),
        PriceQuery::Assets {
            query,
            timeframe,
            target_symbol,
            target_parameter,
            conversion_requested,
            provider_scope,
        } => {
            if !supported_currency(&target_symbol) {
                unsupported_currency(&target_symbol, locale)
            } else if provider_scope == Some(ProviderScope::Stock) {
                if modifiers_unsupported(timeframe.as_deref(), conversion_requested) {
                    stock_modifier_error(&query, locale)
                } else {
                    stock_only(&query, locale, stocks, &mut diagnostics, true, &mut chart)
                }
            } else {
                assets(
                    &query,
                    &target_symbol,
                    &target_parameter,
                    timeframe.as_deref(),
                    conversion_requested,
                    command == MarketPriceCommand::CryptoOnly
                        || provider_scope == Some(ProviderScope::Crypto),
                    locale,
                    crypto,
                    stocks,
                    &mut diagnostics,
                    &mut no_assets_found,
                    &mut chart,
                    &mut selection,
                )
            }
        }
    };
    MarketPriceExecution {
        chart,
        selection,
        no_assets_found,
        text: rendered,
        diagnostics,
    }
}

pub fn execute_market_price_candidate<C: CryptoMarketProvider>(
    candidate: &MarketCandidate,
    timeframe: Option<&str>,
    target_symbol: &str,
    target_parameter: &str,
    conversion: Option<&MarketConversion>,
    locale: Locale,
    crypto: &mut C,
) -> MarketPriceExecution {
    let mut diagnostics = Vec::new();
    let assets = match crypto.quotes_by_id(std::slice::from_ref(&candidate.id), target_parameter) {
        Ok(assets) => assets,
        Err(error) => {
            diagnostics.push(format!("CoinMarketCap identity quote: {error}"));
            return MarketPriceExecution {
                chart: None,
                selection: None,
                no_assets_found: true,
                text: quote_unavailable(&candidate.symbol, locale),
                diagnostics,
            };
        }
    };
    let Some(mut asset) = assets.into_iter().find(|asset| asset.id == candidate.id) else {
        return MarketPriceExecution {
            chart: None,
            selection: None,
            no_assets_found: true,
            text: quote_unavailable(&candidate.symbol, locale),
            diagnostics,
        };
    };
    if asset.symbol.is_empty() {
        asset.symbol.clone_from(&candidate.symbol);
    }
    if asset.name.is_empty() {
        asset.name.clone_from(&candidate.name);
    }
    if asset.slug.is_empty() {
        asset.slug.clone_from(&candidate.slug);
    }
    for contract in &candidate.contracts {
        if !asset.contracts.iter().any(|known| {
            known.chain_id == contract.chain_id
                && known.network == contract.network
                && known.address.eq_ignore_ascii_case(&contract.address)
        }) {
            asset.contracts.push(contract.clone());
        }
    }
    let Some(_quote) = asset
        .quotes
        .get(target_parameter)
        .filter(|quote| quote.is_usable())
    else {
        return MarketPriceExecution {
            chart: None,
            selection: None,
            no_assets_found: true,
            text: quote_unavailable(&asset.symbol, locale),
            diagnostics,
        };
    };
    let text = conversion.map_or_else(
        || {
            format_assets_with_timeframe(
                std::slice::from_ref(&asset),
                target_symbol,
                target_parameter,
                timeframe,
            )
        },
        |conversion| format_selected_conversion(&asset, target_parameter, conversion),
    );
    let chart = conversion.is_none().then(|| MarketChart {
        timeframe: timeframe.map(str::to_owned),
        symbol: asset.symbol.clone(),
        name: asset.name.clone(),
        yahoo_symbol: verified_yahoo_symbol(&asset).unwrap_or_default(),
        token: asset.contracts.first().cloned(),
    });
    MarketPriceExecution {
        chart,
        selection: None,
        no_assets_found: false,
        text,
        diagnostics,
    }
}

#[allow(clippy::too_many_arguments)]
fn assets<C: CryptoMarketProvider, S: UnifiedStockProvider>(
    raw_query: &str,
    target_symbol: &str,
    target_parameter: &str,
    timeframe: Option<&str>,
    conversion_requested: bool,
    crypto_only: bool,
    locale: Locale,
    crypto: &mut C,
    stocks: &mut S,
    diagnostics: &mut Vec<String>,
    no_assets_found: &mut bool,
    chart: &mut Option<MarketChart>,
    selection_result: &mut Option<MarketSelection>,
) -> String {
    let listed = match crypto.listings(target_parameter) {
        Ok(rows) => rows,
        Err(error) => {
            diagnostics.push(format!("CoinMarketCap listings: {error}"));
            if !crypto_only && !contains_unambiguous_crypto_symbol(raw_query) {
                let stock = stock_only(raw_query, locale, stocks, diagnostics, false, chart);
                if !stock.is_empty() {
                    if modifiers_unsupported(timeframe, conversion_requested) {
                        *chart = None;
                        return stock_modifier_error(raw_query, locale);
                    }
                    return stock;
                }
            }
            // A provider outage must not substitute a DEX namesake for a known native coin.
            *no_assets_found = !contains_unambiguous_crypto_symbol(raw_query);
            return load_error(locale);
        }
    };
    let mut selection = select_assets(raw_query, &listed);
    canonicalize_rows(&mut selection.rows, &selection.requested);
    retain_usable_quotes(&mut selection.rows, target_parameter, diagnostics);
    selection.count = selection.count.min(selection.rows.len());
    let canonical_ids = missing_canonical_ids(&selection.rows, &selection.requested);
    if !canonical_ids.is_empty() {
        let mut fetched = crypto
            .quotes_by_id(&canonical_ids, target_parameter)
            .unwrap_or_else(|error| {
                diagnostics.push(format!("CoinMarketCap canonical identity quotes: {error}"));
                Vec::new()
            });
        fetched.retain(|asset| canonical_ids.iter().any(|id| id == &asset.id));
        merge_assets(&mut selection.rows, fetched);
        canonicalize_rows(&mut selection.rows, &selection.requested);
        retain_usable_quotes(&mut selection.rows, target_parameter, diagnostics);
    }
    let missing = missing_tokens(&selection.rows, &selection.requested);
    let mut symbol_requests = missing
        .iter()
        .filter(|token| canonical_id_for_token(token).is_none())
        .cloned()
        .collect::<Vec<_>>();
    symbol_requests.extend(explicit_symbol_requests(raw_query, &listed));
    symbol_requests.sort();
    symbol_requests.dedup();
    if !symbol_requests.is_empty() {
        let mut fetched = crypto
            .quotes(&symbol_requests, target_parameter, false)
            .unwrap_or_else(|error| {
                diagnostics.push(format!("CoinMarketCap symbol quotes: {error}"));
                Vec::new()
            });
        retain_requested_symbols(&mut fetched, &symbol_requests);
        merge_assets(&mut selection.rows, fetched);
    }
    let still_missing =
        missing_usable_tokens(&selection.rows, &selection.requested, target_parameter);
    let single_token = raw_query
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.is_empty())
        .count()
        == 1;
    let slug_requests = explicit_slug_requests(raw_query, &listed)
        .into_iter()
        .collect::<HashSet<_>>();
    if single_token {
        let slugs = still_missing
            .iter()
            .filter(|token| {
                token.len() > 3 && canonical_id_for_token(token).is_none() && {
                    let listed_identity = listed
                        .iter()
                        .any(|asset| asset_tokens(asset).iter().any(|known| known == *token));
                    slug_requests.contains(*token) || !listed_identity
                }
            })
            .map(|token| token.to_lowercase())
            .collect::<Vec<_>>();
        if !slugs.is_empty() {
            let mut slug_fetched = crypto
                .quotes(&slugs, target_parameter, true)
                .unwrap_or_else(|error| {
                    diagnostics.push(format!("CoinMarketCap slug quotes: {error}"));
                    Vec::new()
                });
            retain_requested_slugs(&mut slug_fetched, &still_missing);
            merge_assets(&mut selection.rows, slug_fetched);
        }
    }
    canonicalize_rows(&mut selection.rows, &selection.requested);
    unique_assets(&mut selection.rows);
    retain_usable_quotes(&mut selection.rows, target_parameter, diagnostics);
    selection.count = if selection.explicit_requested.is_empty() {
        selection.count.min(selection.rows.len())
    } else {
        selection.rows.len()
    };
    // A company-name match must not be shadowed by a similarly named cryptocurrency.
    if !crypto_only
        && selection.requested.len() == 1
        && !selection.rows.is_empty()
        && !contains_unambiguous_crypto_symbol(raw_query)
        && let Ok(Some(rows)) = stocks.lookup(raw_query)
        && let Some(quote) = rows
            .iter()
            .filter_map(|(_, quote)| quote.as_ref())
            .find(|quote| company_matches(raw_query, quote))
    {
        if modifiers_unsupported(timeframe, conversion_requested) {
            return stock_modifier_error(raw_query, locale);
        }
        *chart = Some(stock_chart(quote));
        return format_stocks(std::slice::from_ref(quote));
    }
    let mut unresolved = missing_tokens(&selection.rows, &selection.explicit_requested);
    let mut stock_quotes = Vec::new();
    if !crypto_only && !unresolved.is_empty() {
        let stock_query = stock_fallback_query(raw_query, &selection, &unresolved);
        match stocks.lookup(&stock_query) {
            Ok(Some(resolved)) => {
                let quotes = resolved
                    .iter()
                    .filter_map(|(_, quote)| quote.clone())
                    .collect::<Vec<_>>();
                if !quotes.is_empty() {
                    unresolved = resolved
                        .iter()
                        .filter(|(_, quote)| quote.is_none())
                        .map(|(query, _)| query.to_uppercase())
                        .collect();
                    stock_quotes = quotes;
                }
            }
            Ok(None) => {}
            Err(error) => diagnostics.push(format!("stock fallback: {error}")),
        }
    }
    if !raw_query.contains(',')
        && selection.explicit_requested.len() == 1
        && selection.rows.len() > 1
    {
        let mut ranked_assets = selection.rows.iter().collect::<Vec<_>>();
        ranked_assets.sort_by(|left, right| compare_market_assets(left, right, target_parameter));
        let mut candidates = ranked_assets
            .into_iter()
            .map(|asset| market_candidate(asset, target_symbol, target_parameter, timeframe))
            .collect::<Vec<_>>();
        candidates.truncate(10);
        *selection_result = Some(MarketSelection {
            query: raw_query.to_owned(),
            timeframe: timeframe.map(str::to_owned),
            target_symbol: target_symbol.to_owned(),
            target_parameter: target_parameter.to_owned(),
            conversion: None,
            candidates,
        });
        return String::new();
    }
    if !conversion_requested
        && selection.rows.len() + stock_quotes.len() == 1
        && unresolved.is_empty()
    {
        *chart = if let Some(asset) = selection.rows.first() {
            Some(MarketChart {
                timeframe: timeframe.map(str::to_owned),
                symbol: asset.symbol.clone(),
                name: asset.name.clone(),
                yahoo_symbol: verified_yahoo_symbol(asset).unwrap_or_default(),
                token: asset.contracts.first().cloned(),
            })
        } else {
            stock_quotes.first().map(stock_chart)
        };
    }
    if !stock_quotes.is_empty() && modifiers_unsupported(timeframe, conversion_requested) {
        *chart = None;
        let error = stock_modifier_error(
            &stock_quotes
                .iter()
                .map(|quote| quote.symbol.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            locale,
        );
        let crypto_text = format_assets_with_timeframe(
            &selection.rows[..selection.rows.len().min(selection.count)],
            target_symbol,
            target_parameter,
            timeframe,
        );
        return if crypto_text.is_empty() {
            error
        } else {
            format!("{crypto_text}\n{error}")
        };
    }
    if unresolved.is_empty() && selection.rows.is_empty() && stock_quotes.is_empty() {
        return String::new();
    }
    if !unresolved.is_empty() && selection.rows.is_empty() && stock_quotes.is_empty() {
        *no_assets_found = true;
        return missing_assets(&unresolved, locale);
    }
    let mut parts = Vec::new();
    let crypto_text = format_assets_with_timeframe(
        &selection.rows[..selection.rows.len().min(selection.count)],
        target_symbol,
        target_parameter,
        timeframe,
    );
    if !crypto_text.is_empty() {
        parts.push(crypto_text);
    }
    let stock_text = format_stocks(&stock_quotes);
    if !stock_text.is_empty() {
        parts.push(stock_text);
    }
    if !unresolved.is_empty() {
        parts.push(missing_assets(&unresolved, locale));
    }
    parts.join("\n")
}

fn contains_unambiguous_crypto_symbol(query: &str) -> bool {
    query
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.is_empty())
        .map(normalized)
        .any(|token| {
            UNAMBIGUOUS_CRYPTO_SYMBOLS.contains(&token.as_str())
                || canonical_id_for_token(&token).is_some()
                || STABLECOINS.contains(&token.as_str())
                || matches!(token.as_str(), "STABLES" | "STABLECOINS")
        })
}

fn canonical_id_for_token(token: &str) -> Option<&'static str> {
    match normalized(token).as_str() {
        "BTC" | "BITCOIN" | "SATS" => Some("1"),
        "LTC" | "LITECOIN" => Some("2"),
        "XRP" => Some("52"),
        "DOGE" | "DOGECOIN" => Some("74"),
        "XMR" | "MONERO" => Some("328"),
        "ETH" | "ETHEREUM" => Some("1027"),
        "BNB" => Some("1839"),
        "SOL" | "SOLANA" => Some("5426"),
        "ADA" | "CARDANO" => Some("2010"),
        "TRX" | "TRON" => Some("1958"),
        "BCH" => Some("1831"),
        "LINK" | "CHAINLINK" => Some("1975"),
        "MATIC" | "POLYGON" => Some("3890"),
        "DOT" | "POLKADOT" => Some("6636"),
        "UNI" | "UNISWAP" => Some("7083"),
        "AVAX" | "AVALANCHE" => Some("5805"),
        "SHIB" | "SHIBAINU" => Some("5994"),
        _ => None,
    }
}

fn canonicalize_rows(rows: &mut Vec<CryptoAsset>, requested: &[String]) {
    let canonical = requested
        .iter()
        .filter_map(|token| canonical_id_for_token(token).map(|id| (token, id)))
        .collect::<HashMap<_, _>>();
    if canonical.is_empty() {
        return;
    }
    rows.retain(|asset| {
        !asset_tokens(asset)
            .iter()
            .any(|token| canonical.contains_key(token))
            || canonical.values().any(|id| *id == asset.id.as_str())
    });
}

fn missing_canonical_ids(rows: &[CryptoAsset], requested: &[String]) -> Vec<String> {
    let present = rows
        .iter()
        .filter_map(|asset| (!asset.id.is_empty()).then_some(asset.id.as_str()))
        .collect::<HashSet<_>>();
    let mut ids = requested
        .iter()
        .filter_map(|token| canonical_id_for_token(token))
        .filter(|id| !present.contains(id))
        .map(str::to_owned)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn explicit_symbol_requests(raw_query: &str, listed: &[CryptoAsset]) -> Vec<String> {
    let mut seen = HashSet::new();
    raw_query
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.is_empty())
        .filter_map(|raw| {
            let token = normalized(raw);
            if token.is_empty() || canonical_id_for_token(&token).is_some() {
                return None;
            }
            let exact_symbol = listed
                .iter()
                .any(|asset| normalized(&asset.symbol) == token);
            if raw.trim_start().starts_with('$') || exact_symbol {
                seen.insert(token.clone()).then_some(token)
            } else {
                None
            }
        })
        .collect()
}

fn explicit_slug_requests(raw_query: &str, listed: &[CryptoAsset]) -> Vec<String> {
    let mut seen = HashSet::new();
    raw_query
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.is_empty())
        .filter_map(|raw| {
            let token = normalized(raw);
            if token.len() <= 3
                || token.is_empty()
                || raw.trim_start().starts_with('$')
                || canonical_id_for_token(&token).is_some()
                || listed
                    .iter()
                    .any(|asset| normalized(&asset.symbol) == token)
            {
                return None;
            }
            let exact_identity = listed
                .iter()
                .any(|asset| normalized(&asset.name) == token || normalized(&asset.slug) == token);
            exact_identity.then(|| seen.insert(token.clone()).then_some(token))?
        })
        .collect()
}

struct Selection {
    rows: Vec<CryptoAsset>,
    count: usize,
    requested: Vec<String>,
    explicit_requested: Vec<String>,
}

fn select_assets(text: &str, listed: &[CryptoAsset]) -> Selection {
    let top_n = requested_count(text);
    if !text.chars().any(char::is_alphabetic) {
        return Selection {
            rows: listed.to_vec(),
            count: if top_n == 0 { 10 } else { top_n },
            requested: Vec::new(),
            explicit_requested: Vec::new(),
        };
    }
    let single = normalized(text);
    if !text.contains(',') {
        let exact_matches = listed
            .iter()
            .filter(|asset| {
                normalized(&asset.symbol) == single
                    || normalized(&asset.name) == single
                    || normalized(&asset.slug) == single
            })
            .collect::<Vec<_>>();
        let exact_symbol_matches = listed
            .iter()
            .filter(|asset| normalized(&asset.symbol) == single)
            .collect::<Vec<_>>();
        // Canonical native symbols have an established interpretation. Keep
        // the existing preference for those names while exposing genuinely
        // ambiguous tickers (such as LIBRA) to the selection UI.
        if contains_unambiguous_crypto_symbol(text)
            && let Some(asset) = canonical_id_for_token(&single)
                .and_then(|id| exact_matches.iter().find(|asset| asset.id == id))
                .or_else(|| exact_matches.first())
        {
            return Selection {
                rows: vec![(*asset).clone()],
                count: 1,
                requested: vec![single.clone()],
                explicit_requested: vec![single],
            };
        }
        // A single top-list symbol is not proof that the ticker is unique:
        // CMC's symbol endpoint can contain lower-ranked namesakes.  Let the
        // caller perform that discovery before deciding whether to show one
        // quote or an identity menu.  Exact names/slugs remain direct when
        // they do not also collide with a ticker.
        if exact_symbol_matches.is_empty()
            && exact_matches.len() == 1
            && let Some(asset) = exact_matches.first()
        {
            return Selection {
                rows: vec![(*asset).clone()],
                count: 1,
                requested: vec![single.clone()],
                explicit_requested: vec![single],
            };
        }
    }
    let raw_tokens = text
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let coins = expand_tokens(&raw_tokens);
    let explicit_requested = fallback_tokens(&coins[..raw_tokens.len().min(coins.len())]);
    let requested = fallback_tokens(&coins);
    let requested_set = coins.iter().cloned().collect::<HashSet<_>>();
    let mut rows = Vec::new();
    let mut identities = HashSet::new();
    for (index, coin) in listed.iter().enumerate() {
        let symbol = normalized(&coin.symbol);
        let exact_symbol = requested_set.contains(&symbol);
        if !exact_symbol && index >= top_n {
            continue;
        }
        let identity = if coin.id.is_empty() {
            symbol.clone()
        } else {
            coin.id.clone()
        };
        if identities.insert(identity) {
            rows.push(coin.clone());
        }
    }
    let count = if rows.is_empty() && explicit_requested.is_empty() {
        0
    } else {
        rows.len()
    };
    Selection {
        rows,
        count,
        requested,
        explicit_requested,
    }
}

fn expand_tokens(tokens: &[&str]) -> Vec<String> {
    let mut result = tokens
        .iter()
        .map(|token| normalized(token))
        .collect::<Vec<_>>();
    if result
        .iter()
        .any(|token| token == "STABLES" || token == "STABLECOINS")
    {
        result.extend(STABLECOINS.iter().map(|token| (*token).to_owned()));
    }
    result
}

fn fallback_tokens(tokens: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    tokens
        .iter()
        .filter(|token| token.as_str() != "STABLES" && token.as_str() != "STABLECOINS")
        .filter(|token| token.parse::<u64>().is_err())
        .filter(|token| seen.insert((*token).clone()))
        .cloned()
        .collect()
}

fn asset_tokens(asset: &CryptoAsset) -> [String; 3] {
    [
        normalized(&asset.symbol),
        normalized(&asset.name),
        normalized(&asset.slug),
    ]
}

fn missing_tokens(rows: &[CryptoAsset], requested: &[String]) -> Vec<String> {
    let matched = rows.iter().flat_map(asset_tokens).collect::<HashSet<_>>();
    requested
        .iter()
        .filter(|token| {
            !matched.contains(*token)
                && canonical_id_for_token(token)
                    .is_none_or(|id| !rows.iter().any(|asset| asset.id == id))
        })
        .cloned()
        .collect()
}

fn missing_usable_tokens(
    rows: &[CryptoAsset],
    requested: &[String],
    parameter: &str,
) -> Vec<String> {
    let matched = rows
        .iter()
        .filter(|asset| {
            asset
                .quotes
                .get(parameter)
                .is_some_and(CryptoQuote::is_usable)
        })
        .flat_map(asset_tokens)
        .collect::<HashSet<_>>();
    requested
        .iter()
        .filter(|token| {
            !matched.contains(*token)
                && canonical_id_for_token(token).is_none_or(|id| {
                    !rows.iter().any(|asset| {
                        asset.id == id
                            && asset
                                .quotes
                                .get(parameter)
                                .is_some_and(CryptoQuote::is_usable)
                    })
                })
        })
        .cloned()
        .collect()
}

fn retain_requested_symbols(rows: &mut Vec<CryptoAsset>, requested: &[String]) {
    let requested = requested.iter().collect::<HashSet<_>>();
    rows.retain(|asset| requested.contains(&normalized(&asset.symbol)));
}

fn retain_requested_slugs(rows: &mut Vec<CryptoAsset>, requested: &[String]) {
    let requested = requested.iter().collect::<HashSet<_>>();
    rows.retain(|asset| requested.contains(&normalized(&asset.slug)));
}

fn unique_assets(rows: &mut Vec<CryptoAsset>) {
    let mut unique = Vec::with_capacity(rows.len());
    for asset in rows.drain(..) {
        if let Some(existing) = unique.iter_mut().find(|existing: &&mut CryptoAsset| {
            !asset.id.is_empty() && existing.id == asset.id
                || asset.id.is_empty()
                    && existing.id.is_empty()
                    && existing.symbol == asset.symbol
                    && existing.slug == asset.slug
        }) {
            merge_asset(existing, asset);
        } else {
            unique.push(asset);
        }
    }
    *rows = unique;
}

fn merge_asset(existing: &mut CryptoAsset, incoming: CryptoAsset) {
    if existing.symbol.is_empty() {
        existing.symbol = incoming.symbol.clone();
    }
    if existing.name.is_empty() {
        existing.name = incoming.name.clone();
    }
    if existing.slug.is_empty() {
        existing.slug = incoming.slug.clone();
    }
    for (currency, incoming_quote) in incoming.quotes {
        let Some(existing_quote) = existing.quotes.get_mut(&currency) else {
            existing.quotes.insert(currency, incoming_quote);
            continue;
        };
        if !existing_quote.is_usable() && incoming_quote.is_usable() {
            existing_quote.price = incoming_quote.price;
        }
        if incoming_quote.market_cap.is_some() {
            existing_quote.market_cap = incoming_quote.market_cap;
        }
        if incoming_quote.volume_24h.is_some() {
            existing_quote.volume_24h = incoming_quote.volume_24h;
        }
        if incoming_quote.percent_change_1h.is_some() {
            existing_quote.percent_change_1h = incoming_quote.percent_change_1h;
        }
        if incoming_quote.percent_change_24h.is_some() {
            existing_quote.percent_change_24h = incoming_quote.percent_change_24h;
        }
        if incoming_quote.percent_change_7d.is_some() {
            existing_quote.percent_change_7d = incoming_quote.percent_change_7d;
        }
        if incoming_quote.percent_change_30d.is_some() {
            existing_quote.percent_change_30d = incoming_quote.percent_change_30d;
        }
    }
    for contract in incoming.contracts {
        if !existing.contracts.iter().any(|known| {
            known.chain_id == contract.chain_id
                && known.network == contract.network
                && known.address.eq_ignore_ascii_case(&contract.address)
        }) {
            existing.contracts.push(contract);
        }
    }
}

fn merge_assets(rows: &mut Vec<CryptoAsset>, incoming: Vec<CryptoAsset>) {
    for asset in incoming {
        if let Some(existing) = rows.iter_mut().find(|existing| {
            !asset.id.is_empty() && existing.id == asset.id
                || asset.id.is_empty()
                    && existing.id.is_empty()
                    && existing.symbol == asset.symbol
                    && existing.slug == asset.slug
        }) {
            merge_asset(existing, asset);
        } else {
            rows.push(asset);
        }
    }
}

fn retain_usable_quotes(
    rows: &mut Vec<CryptoAsset>,
    parameter: &str,
    diagnostics: &mut Vec<String>,
) {
    let unusable = rows
        .iter()
        .filter(|asset| {
            !asset
                .quotes
                .get(parameter)
                .is_some_and(CryptoQuote::is_usable)
        })
        .map(|asset| asset.symbol.clone())
        .collect::<Vec<_>>();
    if unusable.is_empty() {
        return;
    }
    diagnostics.push(format!(
        "CoinMarketCap rows without a usable {parameter} quote: {}",
        unusable.join(", ")
    ));
    rows.retain(|asset| {
        asset
            .quotes
            .get(parameter)
            .is_some_and(CryptoQuote::is_usable)
    });
}

fn stock_fallback_query(raw: &str, selection: &Selection, unresolved: &[String]) -> String {
    if selection.rows.is_empty() && !raw.contains(',') && unresolved.len() > 1 {
        return raw.to_owned();
    }
    if !raw.contains(',') {
        return unresolved.join(",");
    }
    let unresolved = unresolved.iter().cloned().collect::<HashSet<_>>();
    let segments = raw
        .split(',')
        .map(str::trim)
        .filter(|segment| {
            let tokens = expand_tokens(&segment.split_whitespace().collect::<Vec<_>>());
            tokens.iter().any(|token| unresolved.contains(token))
        })
        .collect::<Vec<_>>();
    if segments.is_empty() {
        unresolved.into_iter().collect::<Vec<_>>().join(",")
    } else {
        segments.join(",")
    }
}

fn convert_amount<C: CryptoMarketProvider>(
    request: &AmountConversion,
    locale: Locale,
    crypto: &mut C,
    diagnostics: &mut Vec<String>,
    no_assets_found: &mut bool,
    selection_result: &mut Option<MarketSelection>,
) -> String {
    if !supported_currency(&request.target_symbol) {
        return unsupported_currency(&request.target_symbol, locale);
    }
    let listed = match crypto.listings(&request.target_parameter) {
        Ok(rows) => rows,
        Err(error) => {
            diagnostics.push(format!("CoinMarketCap conversion listings: {error}"));
            *no_assets_found = true;
            return load_error(locale);
        }
    };
    let direct = conversion_candidates(
        listed,
        &request.source_symbol,
        &request.target_parameter,
        crypto,
        diagnostics,
    );
    if !direct.is_empty() {
        return format_or_select_conversion(
            request,
            &direct,
            false,
            &request.target_symbol,
            &request.target_parameter,
            locale,
            selection_result,
        );
    }
    let source_parameter = price_query_parameter(&request.source_symbol);
    let reversed = match crypto.listings(&source_parameter) {
        Ok(rows) => rows,
        Err(error) => {
            diagnostics.push(format!(
                "CoinMarketCap reverse conversion listings: {error}"
            ));
            *no_assets_found = true;
            return load_error(locale);
        }
    };
    let reversed = conversion_candidates(
        reversed,
        &request.target_symbol,
        &source_parameter,
        crypto,
        diagnostics,
    );
    if reversed.is_empty() {
        *no_assets_found = true;
        unsupported_pair(locale)
    } else {
        format_or_select_conversion(
            request,
            &reversed,
            true,
            &request.source_symbol,
            &source_parameter,
            locale,
            selection_result,
        )
    }
}

fn conversion_candidates<C: CryptoMarketProvider>(
    listed: Vec<CryptoAsset>,
    token: &str,
    currency: &str,
    crypto: &mut C,
    diagnostics: &mut Vec<String>,
) -> Vec<CryptoAsset> {
    let token = normalized(token);
    let canonical_id = canonical_id_for_token(&token);
    let has_exact_symbol = listed
        .iter()
        .any(|asset| normalized(&asset.symbol) == token);
    let mut candidates = listed
        .into_iter()
        .filter(|asset| {
            if let Some(id) = canonical_id {
                return asset.id == id;
            }
            if has_exact_symbol {
                normalized(&asset.symbol) == token
            } else {
                normalized(&asset.name) == token || normalized(&asset.slug) == token
            }
        })
        .collect::<Vec<_>>();
    if canonical_id.is_some() || has_exact_symbol || candidates.is_empty() {
        let mut fetched = if let Some(id) = canonical_id {
            crypto
                .quotes_by_id(&[id.to_owned()], currency)
                .unwrap_or_else(|error| {
                    diagnostics.push(format!("CoinMarketCap conversion identity quotes: {error}"));
                    Vec::new()
                })
        } else {
            crypto
                .quotes(std::slice::from_ref(&token), currency, false)
                .unwrap_or_else(|error| {
                    diagnostics.push(format!("CoinMarketCap conversion symbol quotes: {error}"));
                    Vec::new()
                })
        };
        if canonical_id.is_some() {
            fetched.retain(|asset| canonical_id == Some(asset.id.as_str()));
        } else {
            retain_requested_symbols(&mut fetched, std::slice::from_ref(&token));
        }
        merge_assets(&mut candidates, fetched);
    }
    candidates.retain(|asset| {
        asset
            .quotes
            .get(currency)
            .is_some_and(CryptoQuote::is_usable)
    });
    unique_assets(&mut candidates);
    candidates
}

#[allow(clippy::too_many_arguments)]
fn format_or_select_conversion(
    request: &AmountConversion,
    candidates: &[CryptoAsset],
    reverse: bool,
    display_currency: &str,
    quote_parameter: &str,
    locale: Locale,
    selection_result: &mut Option<MarketSelection>,
) -> String {
    if candidates.len() > 1 {
        let mut market_candidates = candidates
            .iter()
            .map(|asset| market_candidate(asset, display_currency, quote_parameter, None))
            .collect::<Vec<_>>();
        market_candidates.sort_by(|left, right| {
            left.name
                .to_ascii_lowercase()
                .cmp(&right.name.to_ascii_lowercase())
                .then_with(|| left.id.cmp(&right.id))
        });
        market_candidates.truncate(10);
        *selection_result = Some(MarketSelection {
            query: format!(
                "{} {} {}",
                request.amount, request.source_symbol, request.target_symbol
            ),
            timeframe: None,
            target_symbol: request.target_symbol.clone(),
            target_parameter: quote_parameter.to_owned(),
            conversion: Some(MarketConversion {
                amount: trimmed(request.amount, 8),
                source_symbol: request.source_symbol.clone(),
                target_symbol: request.target_symbol.clone(),
                reverse,
            }),
            candidates: market_candidates,
        });
        return String::new();
    }
    candidates.first().map_or_else(
        || unsupported_pair(locale),
        |asset| {
            let conversion = MarketConversion {
                amount: trimmed(request.amount, 8),
                source_symbol: request.source_symbol.clone(),
                target_symbol: request.target_symbol.clone(),
                reverse,
            };
            format_selected_conversion(asset, quote_parameter, &conversion)
        },
    )
}

fn quote_change_for_timeframe<'a>(
    quote: &CryptoQuote,
    timeframe: Option<&'a str>,
) -> (Option<f64>, &'a str) {
    match timeframe {
        Some("1h") => (quote.percent_change_1h, "1h"),
        Some("24h") => (quote.percent_change_24h, "24h"),
        Some("7d") => (quote.percent_change_7d, "7d"),
        Some("30d") => (quote.percent_change_30d, "30d"),
        Some(period) => (None, period),
        None => (quote.percent_change_24h, "24h"),
    }
}

fn format_assets_with_timeframe(
    rows: &[CryptoAsset],
    display: &str,
    parameter: &str,
    timeframe: Option<&str>,
) -> String {
    rows.iter()
        .filter_map(|asset| {
            let quote = asset
                .quotes
                .get(parameter)
                .filter(|quote| quote.is_usable())?;
            let price = quote.price
                * if display == "SATS" {
                    100_000_000.0
                } else {
                    1.0
                };
            let (change, period) = quote_change_for_timeframe(quote, timeframe);
            let fixed = format!("{price:.12}");
            let decimals = fixed.split('.').nth(1).unwrap_or("");
            let zeros = decimals
                .chars()
                .take_while(|character| *character == '0')
                .count();
            let change = change.map_or_else(
                || "N/A".to_owned(),
                |change| format!("{}%", signed_trimmed(change, 2)),
            );
            Some(format!(
                "{}: {} {} ({} {})",
                asset.symbol,
                trimmed(price, zeros + 4),
                display,
                change,
                period
            ))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn compare_market_assets(
    left: &CryptoAsset,
    right: &CryptoAsset,
    quote_parameter: &str,
) -> Ordering {
    // Prefer established assets, then use trading activity when market-cap
    // data is tied or unavailable. Names and IDs only provide deterministic
    // ordering after the market signals are exhausted.
    let left_quote = left.quotes.get(quote_parameter);
    let right_quote = right.quotes.get(quote_parameter);
    compare_market_metric(
        left_quote.and_then(|quote| quote.market_cap),
        right_quote.and_then(|quote| quote.market_cap),
    )
    .then_with(|| {
        compare_market_metric(
            left_quote.and_then(|quote| quote.volume_24h),
            right_quote.and_then(|quote| quote.volume_24h),
        )
    })
    .then_with(|| {
        left.name
            .to_ascii_lowercase()
            .cmp(&right.name.to_ascii_lowercase())
    })
    .then_with(|| left.id.cmp(&right.id))
}

fn compare_market_metric(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (
        left.filter(|value| value.is_finite() && *value > 0.0),
        right.filter(|value| value.is_finite() && *value > 0.0),
    ) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn market_candidate(
    asset: &CryptoAsset,
    display: &str,
    parameter: &str,
    timeframe: Option<&str>,
) -> MarketCandidate {
    let (price, change) = asset
        .quotes
        .get(parameter)
        .filter(|quote| quote.is_usable())
        .map(|quote| {
            let multiplier = if display == "SATS" {
                100_000_000.0
            } else {
                1.0
            };
            let (value, period) = quote_change_for_timeframe(quote, timeframe);
            let change = value.map_or_else(
                || format!("N/A {period}"),
                |value| format!("{}% {period}", signed_trimmed(value, 2)),
            );
            (trimmed(quote.price * multiplier, 12), change)
        })
        .unwrap_or_else(|| {
            let period = timeframe.unwrap_or("24h");
            ("N/A".to_owned(), format!("N/A {period}"))
        });
    MarketCandidate {
        id: asset.id.clone(),
        symbol: asset.symbol.clone(),
        name: asset.name.clone(),
        slug: asset.slug.clone(),
        price,
        change,
        contracts: asset.contracts.clone(),
    }
}

fn format_selected_conversion(
    asset: &CryptoAsset,
    target_parameter: &str,
    conversion: &MarketConversion,
) -> String {
    let Some(amount) = conversion
        .amount
        .parse::<f64>()
        .ok()
        .filter(|amount| amount.is_finite() && *amount >= 0.0)
    else {
        return String::new();
    };
    let Some(quote) = asset
        .quotes
        .get(target_parameter)
        .filter(|quote| quote.is_usable())
    else {
        return String::new();
    };
    let source_amount = if conversion.source_symbol == "SATS" {
        amount / 100_000_000.0
    } else {
        amount
    };
    let target_amount = if conversion.reverse {
        source_amount / quote.price
    } else {
        source_amount * quote.price
    } * if conversion.target_symbol == "SATS" {
        100_000_000.0
    } else {
        1.0
    };
    if !target_amount.is_finite() {
        return String::new();
    }
    format!(
        "{} {} = {} {}",
        conversion.amount,
        conversion.source_symbol,
        trimmed(target_amount, 8),
        conversion.target_symbol
    )
}

fn verified_yahoo_symbol(asset: &CryptoAsset) -> Option<String> {
    let symbol = match asset.id.as_str() {
        "1" => "BTC-USD",
        "2" => "LTC-USD",
        "52" => "XRP-USD",
        "74" => "DOGE-USD",
        "328" => "XMR-USD",
        "1027" => "ETH-USD",
        "1831" => "BCH-USD",
        "1839" => "BNB-USD",
        "1958" => "TRX-USD",
        "1975" => "LINK-USD",
        "2010" => "ADA-USD",
        "3890" => "MATIC-USD",
        "5426" => "SOL-USD",
        "5805" => "AVAX-USD",
        "5994" => "SHIB-USD",
        "6636" => "DOT-USD",
        "7083" => "UNI-USD",
        _ => return None,
    };
    Some(symbol.to_owned())
}

fn stock_only<S: UnifiedStockProvider>(
    query: &str,
    locale: Locale,
    stocks: &mut S,
    diagnostics: &mut Vec<String>,
    missing_error: bool,
    chart: &mut Option<MarketChart>,
) -> String {
    if query.trim().is_empty() {
        return if missing_error {
            missing_assets(&[query.to_uppercase()], locale)
        } else {
            String::new()
        };
    }
    let resolved = match stocks.lookup(query) {
        Ok(Some(resolved)) => resolved,
        Ok(None) => Vec::new(),
        Err(error) => {
            diagnostics.push(format!("stock lookup: {error}"));
            Vec::new()
        }
    };
    let quotes = resolved
        .iter()
        .filter_map(|(_, quote)| quote.clone())
        .collect::<Vec<_>>();
    if quotes.len() == 1 && resolved.len() == 1 {
        *chart = quotes.first().map(stock_chart);
    }
    let mut parts = Vec::new();
    let text = format_stocks(&quotes);
    if !text.is_empty() {
        parts.push(text);
    }
    let missing = resolved
        .iter()
        .filter(|(_, quote)| quote.is_none())
        .map(|(query, _)| query.to_uppercase())
        .collect::<Vec<_>>();
    if missing_error && (!missing.is_empty() || quotes.is_empty()) {
        let missing = if missing.is_empty() {
            vec![query.to_uppercase()]
        } else {
            missing
        };
        parts.push(missing_assets(&missing, locale));
    }
    parts.join("\n")
}

fn format_stocks(quotes: &[StockQuote]) -> String {
    quotes
        .iter()
        .map(|quote| {
            format!(
                "{}: {:.2} {} ({:+.2}% 24h)",
                quote.symbol, quote.price, quote.currency, quote.variation
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn requested_count(text: &str) -> usize {
    text.to_uppercase()
        .replace(' ', "")
        .split(',')
        .filter_map(|token| token.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| value.trunc() as usize)
        .max()
        .unwrap_or(0)
}

fn normalized(value: &str) -> String {
    value
        .to_uppercase()
        .replace(' ', "")
        .trim_start_matches('$')
        .to_owned()
}
fn supported_currency(value: &str) -> bool {
    SUPPORTED_CURRENCIES.contains(&value)
}
fn modifiers_unsupported(timeframe: Option<&str>, conversion: bool) -> bool {
    conversion || !matches!(timeframe, None | Some("24h"))
}
fn trimmed(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}
fn signed_trimmed(value: f64, decimals: usize) -> String {
    format!("{value:+.decimals$}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn invalid_timeframe(value: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => format!(
            "timeframe '{value}' no soportado, uso: {}",
            TIMEFRAMES.join(", ")
        ),
        Locale::En => format!(
            "unsupported timeframe '{value}', use: {}",
            TIMEFRAMES.join(", ")
        ),
    }
}
fn unsupported_currency(value: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("no laburo con {value} gordo"),
        Locale::En => format!("I do not support {value}"),
    }
}
fn load_error(locale: Locale) -> String {
    match locale {
        Locale::Es => "no pude traer precios de crypto boludo",
        Locale::En => "I could not load crypto prices",
    }
    .to_owned()
}
fn unsupported_pair(locale: Locale) -> String {
    match locale {
        Locale::Es => "no laburo con esos ponzis boludo",
        Locale::En => "I do not support that asset pair",
    }
    .to_owned()
}
fn missing_assets(values: &[String], locale: Locale) -> String {
    let values = values.join(", ");
    match locale {
        Locale::Es => format!("no encontré estos activos: {values}"),
        Locale::En => format!("I could not find these assets: {values}"),
    }
}

fn quote_unavailable(symbol: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("no pude obtener una cotización usable para {symbol}"),
        Locale::En => format!("I could not obtain a usable quote for {symbol}"),
    }
}
fn stock_modifier_error(value: &str, locale: Locale) -> String {
    let value = value.to_uppercase();
    match locale {
        Locale::Es => format!("{value}: las acciones solo soportan moneda nativa y variación 24h"),
        Locale::En => format!("{value}: stocks only support native currency and 24h change"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Crypto {
        listings: Vec<Vec<CryptoAsset>>,
        quotes: Vec<Vec<CryptoAsset>>,
    }
    impl CryptoMarketProvider for Crypto {
        fn listings(&mut self, _: &str) -> Result<Vec<CryptoAsset>, String> {
            Ok(if self.listings.is_empty() {
                Vec::new()
            } else {
                self.listings.remove(0)
            })
        }
        fn quotes(&mut self, _: &[String], _: &str, _: bool) -> Result<Vec<CryptoAsset>, String> {
            Ok(if self.quotes.is_empty() {
                Vec::new()
            } else {
                self.quotes.remove(0)
            })
        }
    }
    #[derive(Default)]
    struct Stocks(Vec<(String, Option<StockQuote>)>);
    impl UnifiedStockProvider for Stocks {
        fn lookup(&mut self, _: &str) -> Result<Option<Vec<(String, Option<StockQuote>)>>, String> {
            Ok(Some(self.0.clone()))
        }
    }
    fn coin(symbol: &str, price: f64) -> CryptoAsset {
        let id = match symbol {
            "BTC" => "1",
            "LTC" => "2",
            "XRP" => "52",
            "DOGE" => "74",
            "XMR" => "328",
            "ETH" => "1027",
            "BNB" => "1839",
            "SOL" => "5426",
            "ADA" => "2010",
            _ => symbol,
        };
        CryptoAsset {
            id: id.to_owned(),
            symbol: symbol.to_owned(),
            name: symbol.to_owned(),
            slug: symbol.to_lowercase(),
            quotes: HashMap::from([(
                "USD".to_owned(),
                CryptoQuote {
                    price,
                    market_cap: None,
                    volume_24h: None,
                    percent_change_1h: Some(1.0),
                    percent_change_24h: Some(2.5),
                    percent_change_7d: Some(7.0),
                    percent_change_30d: Some(30.0),
                },
            )]),
            contracts: Vec::new(),
        }
    }

    fn stock(symbol: &str) -> StockQuote {
        StockQuote {
            symbol: symbol.to_owned(),
            name: symbol.to_owned(),
            price: 123.45,
            currency: "USD".to_owned(),
            exchange: "Synthetic".to_owned(),
            variation: 1.25,
        }
    }

    #[test]
    fn canonical_bitcoin_and_apple_beat_namesake_tokens() {
        let mut bitcoin = coin("BTC", 79_000.0);
        bitcoin.name = "Bitcoin".to_owned();
        bitcoin.slug = "bitcoin".to_owned();
        let result = execute_market_price_command(
            "bitcoin",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![bitcoin, coin("BITCOIN", 0.001)]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        assert!(result.text.starts_with("BTC:"));
        assert!(!result.text.contains("BITCOIN:"));
        assert_eq!(
            result
                .chart
                .as_ref()
                .map(|chart| chart.yahoo_symbol.as_str()),
            Some("BTC-USD")
        );
        let mut apple = stock("AAPL");
        apple.name = "Apple Inc.".to_owned();
        let result = execute_market_price_command(
            "apple",
            MarketPriceCommand::Unified,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![]],
                quotes: vec![vec![coin("APPLE", 0.00008273)]],
            },
            &mut Stocks(vec![("apple".to_owned(), Some(apple))]),
        );
        assert!(result.text.starts_with("AAPL:"));
        assert!(!result.text.contains("APPLE:"));
        assert_eq!(
            result
                .chart
                .as_ref()
                .map(|chart| chart.yahoo_symbol.as_str()),
            Some("AAPL")
        );
    }

    #[test]
    fn empty_crypto_queries_return_the_provider_top_list_without_a_bitcoin_default() {
        let result = execute_market_price_command(
            "",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![coin("BTC", 50_000.0), coin("ETH", 2_500.0)]],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert!(result.text.contains("BTC: 50000 USD"));
        assert!(result.text.contains("ETH: 2500 USD"));
        assert!(result.chart.is_none());
        assert!(!result.no_assets_found);
    }

    #[test]
    fn every_chart_period_uses_the_requested_change_period() {
        for period in ["1h", "2h", "7d", "1m", "1mo", "1w", "1y", "5y"] {
            let result = execute_market_price_command(
                &format!("btc {period}"),
                MarketPriceCommand::CryptoOnly,
                Locale::En,
                &mut Crypto {
                    listings: vec![vec![coin("BTC", 50_000.0)]],
                    quotes: Vec::new(),
                },
                &mut Stocks::default(),
            );
            let expected = match period {
                "1h" => "BTC: 50000 USD (+1% 1h)".to_owned(),
                "7d" => "BTC: 50000 USD (+7% 7d)".to_owned(),
                _ => format!("BTC: 50000 USD (N/A {period})"),
            };
            assert_eq!(result.text, expected, "{period}");
            assert_eq!(
                result
                    .chart
                    .as_ref()
                    .and_then(|chart| chart.timeframe.as_deref()),
                Some(period),
                "{period}"
            );
        }
    }

    #[test]
    fn ambiguous_tickers_return_identity_candidates_instead_of_combined_quotes()
    -> Result<(), String> {
        let mut first = coin("LIBRA", 0.007);
        first.id = "1001".to_owned();
        first.name = "Libra Finance".to_owned();
        let mut second = coin("LIBRA", 0.00009);
        second.id = "1002".to_owned();
        second.name = "Libra Protocol".to_owned();
        let result = execute_market_price_command(
            "$libra 1h",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![first, second]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        assert!(result.text.is_empty());
        assert!(!result.no_assets_found);
        let selection = result
            .selection
            .ok_or_else(|| "ambiguous ticker selection".to_owned())?;
        assert_eq!(selection.timeframe.as_deref(), Some("1h"));
        assert_eq!(
            selection
                .candidates
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            vec!["1001", "1002"]
        );
        Ok(())
    }

    #[test]
    fn ambiguous_tickers_prioritize_market_cap_then_volume() -> Result<(), String> {
        let set_metrics = |asset: &mut CryptoAsset, market_cap, volume_24h| {
            let quote = asset
                .quotes
                .get_mut("USD")
                .ok_or_else(|| "missing USD quote".to_owned())?;
            quote.market_cap = market_cap;
            quote.volume_24h = volume_24h;
            Ok::<_, String>(())
        };
        let mut cap_winner = coin("TRUMP", 2.0);
        cap_winner.id = "1001".to_owned();
        cap_winner.name = "Zeta Trump".to_owned();
        set_metrics(&mut cap_winner, Some(200.0), Some(1.0))?;
        let mut volume_tiebreaker = coin("TRUMP", 2.0);
        volume_tiebreaker.id = "1002".to_owned();
        volume_tiebreaker.name = "Alpha Trump".to_owned();
        set_metrics(&mut volume_tiebreaker, Some(200.0), Some(50.0))?;
        let mut lower_cap = coin("TRUMP", 2.0);
        lower_cap.id = "1003".to_owned();
        lower_cap.name = "Omega Trump".to_owned();
        set_metrics(&mut lower_cap, Some(100.0), Some(999.0))?;
        let mut volume_only = coin("TRUMP", 2.0);
        volume_only.id = "1004".to_owned();
        volume_only.name = "Beta Trump".to_owned();
        set_metrics(&mut volume_only, None, Some(10.0))?;
        let mut no_metrics = coin("TRUMP", 2.0);
        no_metrics.id = "1005".to_owned();
        no_metrics.name = "Aardvark Trump".to_owned();

        let result = execute_market_price_command(
            "$trump",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![
                    cap_winner,
                    volume_tiebreaker,
                    lower_cap,
                    volume_only,
                    no_metrics,
                ]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        let selection = result
            .selection
            .ok_or_else(|| "ambiguous ticker selection".to_owned())?;
        assert_eq!(
            selection
                .candidates
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            vec!["1002", "1001", "1003", "1004", "1005"]
        );
        Ok(())
    }

    #[test]
    fn a_top_list_ticker_is_checked_for_lower_ranked_namesakes() -> Result<(), String> {
        let mut listed = coin("LIBRA", 0.007);
        listed.id = "1001".to_owned();
        listed.name = "Libra Finance".to_owned();
        let mut namesake = coin("LIBRA", 0.00009);
        namesake.id = "1002".to_owned();
        namesake.name = "Libra Protocol".to_owned();
        let result = execute_market_price_command(
            "libra",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![listed]],
                quotes: vec![vec![namesake]],
            },
            &mut Stocks::default(),
        );
        let selection = result
            .selection
            .ok_or_else(|| "single top-list ticker was not disambiguated".to_owned())?;
        assert_eq!(
            selection
                .candidates
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            vec!["1001", "1002"]
        );
        assert!(result.text.is_empty());
        assert!(result.chart.is_none());
        Ok(())
    }

    #[test]
    fn exact_names_and_unlisted_slugs_keep_their_identity() -> Result<(), String> {
        let mut named = coin("LIBRA", 0.007);
        named.name = "Libra Finance".to_owned();
        named.slug = "libra-finance".to_owned();
        let named_result = execute_market_price_command(
            "libra finance",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![named]],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert_eq!(named_result.text, "LIBRA: 0.007 USD (+2.5% 24h)");

        let mut fetched = coin("MYST", 0.25);
        fetched.id = "777".to_owned();
        fetched.name = "Mysterious Token".to_owned();
        fetched.slug = "mysterious-token".to_owned();
        let slug_result = execute_market_price_command(
            "mysterious-token",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![Vec::new()],
                quotes: vec![Vec::new(), vec![fetched]],
            },
            &mut Stocks::default(),
        );
        assert_eq!(slug_result.text, "MYST: 0.25 USD (+2.5% 24h)");
        assert_eq!(
            slug_result
                .chart
                .as_ref()
                .map(|chart| chart.symbol.as_str()),
            Some("MYST")
        );
        Ok(())
    }

    #[test]
    fn unusable_quotes_are_reported_without_an_empty_or_fake_price_line() {
        let result = execute_market_price_command(
            "magaiba",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![vec![coin("MAGAIBA", 0.0)]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        assert!(result.no_assets_found);
        assert!(!result.text.is_empty());
        assert!(!result.text.contains("N/A USD"));
    }

    #[test]
    fn missing_timeframe_change_stays_explicit_without_a_spurious_percent_sign()
    -> Result<(), String> {
        let mut asset = coin("BTC", 50_000.0);
        let quote = asset
            .quotes
            .get_mut("USD")
            .ok_or_else(|| "synthetic USD quote".to_owned())?;
        quote.percent_change_24h = None;
        let result = execute_market_price_command(
            "btc",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![asset]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "BTC: 50000 USD (N/A 24h)");
        Ok(())
    }

    #[test]
    fn unknown_crypto_symbols_do_not_become_guessed_yahoo_targets() {
        let mut asset = coin("TRUMP", 2.2);
        asset.id = "99999".to_owned();
        let result = execute_market_price_command(
            "trump",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![asset]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        assert_eq!(
            result
                .chart
                .as_ref()
                .map(|chart| chart.yahoo_symbol.as_str()),
            Some("")
        );
    }

    #[test]
    fn failed_market_provider_keeps_token_fallback_available() {
        struct Failed;
        impl CryptoMarketProvider for Failed {
            fn listings(&mut self, _: &str) -> Result<Vec<CryptoAsset>, String> {
                Err("outage".to_owned())
            }
            fn quotes(
                &mut self,
                _: &[String],
                _: &str,
                _: bool,
            ) -> Result<Vec<CryptoAsset>, String> {
                Ok(vec![])
            }
        }
        let result = execute_market_price_command(
            "timba",
            MarketPriceCommand::Unified,
            Locale::En,
            &mut Failed,
            &mut Stocks::default(),
        );
        assert!(result.no_assets_found);
        assert!(result.chart.is_none());
    }

    #[test]
    fn recognizes_every_public_alias() {
        for alias in [
            "/prices", "/price", "/precios", "/precio", "/presios", "/presio", "/bresio",
            "/bresios", "/brecio", "/brecios", "/p",
        ] {
            assert_eq!(
                classify_market_price_command(alias),
                Some(MarketPriceCommand::Unified)
            );
        }
        for alias in ["/c", "/cripto", "/criptos", "/crypto", "/cryptos"] {
            assert_eq!(
                classify_market_price_command(alias),
                Some(MarketPriceCommand::CryptoOnly)
            );
        }
    }

    #[test]
    fn formats_timeframes_stables_missing_and_stock_fallback() {
        let mut crypto = Crypto {
            listings: vec![vec![coin("BTC", 50_000.0), coin("USDT", 1.0)]],
            quotes: vec![Vec::new(), Vec::new()],
        };
        let mut stocks = Stocks(vec![(
            "NVDA".to_owned(),
            Some(StockQuote {
                symbol: "NVDA".to_owned(),
                name: String::new(),
                price: 123.45,
                currency: "USD".to_owned(),
                exchange: String::new(),
                variation: 1.25,
            }),
        )]);
        let result = execute_market_price_command(
            "btc nvda 7d",
            MarketPriceCommand::Unified,
            Locale::Es,
            &mut crypto,
            &mut stocks,
        );
        assert_eq!(
            result.text,
            "BTC: 50000 USD (+7% 7d)\nNVDA: las acciones solo soportan moneda nativa y variación 24h"
        );
        let result = execute_market_price_command(
            "btc 2h",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![coin("BTC", 50_000.0)]],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "BTC: 50000 USD (N/A 2h)");
        assert_eq!(
            result
                .chart
                .as_ref()
                .and_then(|chart| chart.timeframe.as_deref()),
            Some("2h")
        );
    }

    #[test]
    fn converts_direct_and_reverse_amounts() {
        let mut direct = coin("USDT", 7.8);
        let Some(quote) = direct.quotes.remove("USD") else {
            return;
        };
        direct.quotes.insert("HKD".to_owned(), quote);
        let result = execute_market_price_command(
            "2000 usdt in hkd",
            MarketPriceCommand::Unified,
            Locale::Es,
            &mut Crypto {
                listings: vec![vec![direct]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "2000 USDT = 15600 HKD");
        let mut target = coin("USDT", 7.8);
        let Some(quote) = target.quotes.remove("USD") else {
            return;
        };
        target.quotes.insert("HKD".to_owned(), quote);
        let result = execute_market_price_command(
            "2000 hkd in usdt",
            MarketPriceCommand::Unified,
            Locale::Es,
            &mut Crypto {
                listings: vec![Vec::new(), vec![target]],
                quotes: vec![],
            },
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "2000 HKD = 256.41025641 USDT");
    }

    #[test]
    fn converts_satoshi_amounts_in_both_directions() {
        let result = execute_market_price_command(
            "1 sats in usd",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![vec![coin("BTC", 50_000.0)]],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "1 SATS = 0.0005 USD");

        let result = execute_market_price_command(
            "1 usd in sats",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut Crypto {
                listings: vec![Vec::new(), vec![coin("BTC", 50_000.0)]],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "1 USD = 2000 SATS");
    }

    #[test]
    fn expands_stables_and_respects_top_n_and_explicit_lists() {
        let listed = vec![
            coin("BTC", 50_000.0),
            coin("USDT", 1.0),
            coin("USDC", 0.999),
        ];
        let result = execute_market_price_command(
            "stables",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![listed.clone()],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert!(!result.text.contains("BTC:"));
        assert!(result.text.contains("USDT:"));
        assert!(result.text.contains("USDC:"));

        let result = execute_market_price_command(
            "2",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![listed.clone()],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert!(result.text.contains("BTC:"));
        assert!(result.text.contains("USDT:"));
        assert!(!result.text.contains("USDC:"));

        let result = execute_market_price_command(
            "btc,usdc",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![listed],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert!(result.text.contains("BTC:"));
        assert!(!result.text.contains("USDT:"));
        assert!(result.text.contains("USDC:"));
    }

    #[test]
    fn reports_unresolved_market_queries_without_parsing_localized_text() {
        for (query, listed, missing) in [
            ("timba", Vec::new(), true),
            ("btc", vec![coin("BTC", 50_000.0)], false),
            ("btc timba", vec![coin("BTC", 50_000.0)], false),
        ] {
            for locale in [Locale::Es, Locale::En] {
                let result = execute_market_price_command(
                    query,
                    MarketPriceCommand::Unified,
                    locale,
                    &mut Crypto {
                        listings: vec![listed.clone()],
                        quotes: Vec::new(),
                    },
                    &mut Stocks::default(),
                );
                assert_eq!(result.no_assets_found, missing);
            }
        }
    }

    #[test]
    fn fetches_missing_symbols_then_slugs_and_reports_only_unresolved_assets() {
        let result = execute_market_price_command(
            "btc zzz yyy",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![vec![coin("BTC", 50_000.0)]],
                quotes: vec![vec![coin("ZZZ", 2.0)], Vec::new()],
            },
            &mut Stocks::default(),
        );
        assert!(result.text.contains("BTC:"));
        assert!(result.text.contains("ZZZ:"));
        assert!(result.text.ends_with("no encontré estos activos: YYY"));
    }

    #[test]
    fn rejects_provider_results_that_do_not_exactly_match_the_requested_asset() {
        let mut misleading_usd = coin("xUSD", 1.0);
        misleading_usd.name = "USD".to_owned();
        misleading_usd.slug = "usd".to_owned();
        let result = execute_market_price_command(
            "btc usd 24h",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![vec![coin("BTC", 50_000.0), misleading_usd.clone()]],
                quotes: vec![vec![misleading_usd], Vec::new()],
            },
            &mut Stocks::default(),
        );

        assert_eq!(
            result.text,
            "BTC: 50000 USD (+2.5% 24h)\nno encontré estos activos: USD"
        );
    }

    #[test]
    fn accepts_unambiguous_long_asset_slugs() {
        let mut bitcoin = coin("BTC", 50_000.0);
        bitcoin.name = "Bitcoin".to_owned();
        bitcoin.slug = "bitcoin".to_owned();

        let result = execute_market_price_command(
            "bitcoin",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![vec![bitcoin]],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );

        assert_eq!(result.text, "BTC: 50000 USD (+2.5% 24h)");
    }

    #[test]
    fn preserves_company_names_and_combines_crypto_with_stock_fallback() {
        let result = execute_market_price_command(
            "btc, Mercado Libre",
            MarketPriceCommand::Unified,
            Locale::Es,
            &mut Crypto {
                listings: vec![vec![coin("BTC", 50_000.0)]],
                quotes: vec![Vec::new(), Vec::new()],
            },
            &mut Stocks(vec![("Mercado Libre".to_owned(), Some(stock("MELI")))]),
        );
        assert_eq!(
            result.text,
            "BTC: 50000 USD (+2.5% 24h)\nMELI: 123.45 USD (+1.25% 24h)"
        );
    }

    #[test]
    fn provider_scopes_control_collision_and_stock_modifier_behavior() {
        let result = execute_market_price_command(
            "stock:META",
            MarketPriceCommand::Unified,
            Locale::En,
            &mut Crypto::default(),
            &mut Stocks(vec![("META".to_owned(), Some(stock("META")))]),
        );
        assert_eq!(result.text, "META: 123.45 USD (+1.25% 24h)");

        let result = execute_market_price_command(
            "stock:NVDA in EUR",
            MarketPriceCommand::Unified,
            Locale::En,
            &mut Crypto::default(),
            &mut Stocks::default(),
        );
        assert_eq!(
            result.text,
            "NVDA: stocks only support native currency and 24h change"
        );

        let result = execute_market_price_command(
            "crypto:META",
            MarketPriceCommand::Unified,
            Locale::En,
            &mut Crypto {
                listings: vec![Vec::new()],
                quotes: vec![Vec::new(), Vec::new()],
            },
            &mut Stocks(vec![("META".to_owned(), Some(stock("META")))]),
        );
        assert_eq!(result.text, "I could not find these assets: META");
    }

    struct FailedCrypto;
    impl CryptoMarketProvider for FailedCrypto {
        fn listings(&mut self, _: &str) -> Result<Vec<CryptoAsset>, String> {
            Err("synthetic failure".to_owned())
        }
        fn quotes(&mut self, _: &[String], _: &str, _: bool) -> Result<Vec<CryptoAsset>, String> {
            Err("synthetic failure".to_owned())
        }
    }

    #[test]
    fn candidate_resolution_is_identity_bound_and_preserves_contracts() -> Result<(), String> {
        let contract = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: "F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump".to_owned(),
        };
        let candidate = MarketCandidate {
            id: "42".to_owned(),
            symbol: "LIBRA".to_owned(),
            name: "Libra Finance".to_owned(),
            slug: "libra-finance".to_owned(),
            price: "1.5".to_owned(),
            change: "+2.5%".to_owned(),
            contracts: vec![contract.clone()],
        };
        let mut asset = coin("LIBRA", 1.5);
        asset.id = candidate.id.clone();
        asset.symbol.clear();
        asset.name.clear();
        asset.slug.clear();
        let result = execute_market_price_candidate(
            &candidate,
            Some("7d"),
            "USD",
            "USD",
            None,
            Locale::En,
            &mut Crypto {
                listings: Vec::new(),
                quotes: vec![vec![asset.clone()]],
            },
        );
        assert_eq!(result.text, "LIBRA: 1.5 USD (+7% 7d)");
        let chart = result
            .chart
            .as_ref()
            .ok_or_else(|| "candidate quote did not retain chart identity".to_owned())?;
        assert_eq!(chart.timeframe.as_deref(), Some("7d"));
        assert_eq!(chart.token.as_ref(), Some(&contract));

        let conversion = MarketConversion {
            amount: "2".to_owned(),
            source_symbol: "LIBRA".to_owned(),
            target_symbol: "USD".to_owned(),
            reverse: false,
        };
        let converted = execute_market_price_candidate(
            &candidate,
            None,
            "USD",
            "USD",
            Some(&conversion),
            Locale::En,
            &mut Crypto {
                listings: Vec::new(),
                quotes: vec![vec![asset]],
            },
        );
        assert_eq!(converted.text, "2 LIBRA = 3 USD");
        assert!(converted.chart.is_none());
        Ok(())
    }

    #[test]
    fn candidate_resolution_rejects_wrong_id_unusable_quote_and_provider_failure() {
        let candidate = MarketCandidate {
            id: "42".to_owned(),
            symbol: "LIBRA".to_owned(),
            name: "Libra Finance".to_owned(),
            slug: "libra-finance".to_owned(),
            price: "1".to_owned(),
            change: "N/A".to_owned(),
            contracts: Vec::new(),
        };
        let mut wrong_id = coin("LIBRA", 1.0);
        wrong_id.id = "99".to_owned();
        let wrong = execute_market_price_candidate(
            &candidate,
            None,
            "USD",
            "USD",
            None,
            Locale::En,
            &mut Crypto {
                listings: Vec::new(),
                quotes: vec![vec![wrong_id]],
            },
        );
        assert!(wrong.no_assets_found);
        assert!(wrong.text.contains("usable quote"));

        let mut unusable_asset = coin("LIBRA", 0.0);
        unusable_asset.id = candidate.id.clone();
        let unusable = execute_market_price_candidate(
            &candidate,
            None,
            "USD",
            "USD",
            None,
            Locale::En,
            &mut Crypto {
                listings: Vec::new(),
                quotes: vec![vec![unusable_asset]],
            },
        );
        assert!(unusable.no_assets_found);
        assert!(unusable.text.contains("LIBRA"));

        let failed = execute_market_price_candidate(
            &candidate,
            None,
            "USD",
            "USD",
            None,
            Locale::En,
            &mut FailedCrypto,
        );
        assert!(failed.no_assets_found);
        assert_eq!(
            failed.diagnostics,
            vec!["CoinMarketCap identity quote: synthetic failure"]
        );
    }

    #[test]
    fn market_selection_text_is_bounded_and_includes_candidate_identity() {
        let contract = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: "123456789012345678901234567890123456789012".to_owned(),
        };
        let mut candidates = vec![MarketCandidate {
            id: "".to_owned(),
            symbol: "LIBRA".to_owned(),
            name: String::new(),
            slug: "libra-finance".to_owned(),
            price: "0.007".to_owned(),
            change: "N/A".to_owned(),
            contracts: Vec::new(),
        }];
        candidates.extend((1..=10).map(|index| MarketCandidate {
            id: index.to_string(),
            symbol: "LIBRA".to_owned(),
            name: format!("Libra {index}"),
            slug: format!("libra-{index}"),
            price: "0.007".to_owned(),
            change: "N/A".to_owned(),
            contracts: if index == 1 {
                vec![contract.clone()]
            } else {
                Vec::new()
            },
        }));
        let selection = MarketSelection {
            query: "libra".to_owned(),
            timeframe: None,
            target_symbol: "USD".to_owned(),
            target_parameter: "USD".to_owned(),
            conversion: None,
            candidates,
        };
        let text = format_market_selection(&selection, Locale::En);
        assert!(text.chars().count() <= crate::telegram_actions::MAX_TELEGRAM_TEXT_LENGTH);
        assert!(text.contains("libra-finance"));
        assert!(text.contains("solana:SOL"));
        assert!(!text.contains("11. Libra"));
    }

    #[test]
    fn provider_failure_keeps_unified_stock_fallback_but_crypto_only_fails() {
        let result = execute_market_price_command(
            "nvda",
            MarketPriceCommand::Unified,
            Locale::En,
            &mut FailedCrypto,
            &mut Stocks(vec![("nvda".to_owned(), Some(stock("NVDA")))]),
        );
        assert_eq!(result.text, "NVDA: 123.45 USD (+1.25% 24h)");
        assert_eq!(
            result.diagnostics,
            vec!["CoinMarketCap listings: synthetic failure"]
        );

        let result = execute_market_price_command(
            "btc",
            MarketPriceCommand::CryptoOnly,
            Locale::En,
            &mut FailedCrypto,
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "I could not load crypto prices");

        let result = execute_market_price_command(
            "btc",
            MarketPriceCommand::Unified,
            Locale::En,
            &mut FailedCrypto,
            &mut Stocks(vec![("btc".to_owned(), Some(stock("BTC")))]),
        );
        assert_eq!(result.text, "I could not load crypto prices");
    }

    #[test]
    fn supports_satoshi_display_without_mutating_provider_data() {
        let mut btc = coin("BTC", 1.0);
        let Some(quote) = btc.quotes.remove("USD") else {
            return;
        };
        btc.quotes.insert("BTC".to_owned(), quote);
        let original = btc.clone();
        let result = execute_market_price_command(
            "btc in sats",
            MarketPriceCommand::CryptoOnly,
            Locale::Es,
            &mut Crypto {
                listings: vec![vec![btc]],
                quotes: Vec::new(),
            },
            &mut Stocks::default(),
        );
        assert_eq!(result.text, "BTC: 100000000 SATS (+2.5% 24h)");
        assert_eq!(original.quotes["BTC"].price, 1.0);
    }
}
