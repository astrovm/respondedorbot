//! Unified cryptocurrency and stock price command behavior.

use std::collections::{HashMap, HashSet};

use crate::locale::Locale;
use crate::price_queries::{
    AmountConversion, PriceQuery, ProviderScope, parse_price_query, price_query_parameter,
};
use crate::stocks::StockQuote;

const TIMEFRAMES: [&str; 4] = ["1h", "24h", "7d", "30d"];
const STABLECOINS: [&str; 26] = [
    "BUSD", "DAI", "DOC", "EURT", "FDUSD", "FRAX", "GHO", "GUSD", "LUSD", "MAI", "MIM", "MIMATIC",
    "NUARS", "PAXG", "PYUSD", "RAI", "SUSD", "TUSD", "USDC", "USDD", "USDM", "USDP", "USDT", "UXD",
    "XAUT", "XSGD",
];
const UNAMBIGUOUS_CRYPTO_SYMBOLS: [&str; 4] = ["BTC", "ETH", "SATS", "XMR"];
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
    pub percent_change_1h: Option<f64>,
    pub percent_change_24h: Option<f64>,
    pub percent_change_7d: Option<f64>,
    pub percent_change_30d: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CryptoAsset {
    pub id: String,
    pub symbol: String,
    pub name: String,
    pub slug: String,
    pub quotes: HashMap<String, CryptoQuote>,
}

pub trait CryptoMarketProvider {
    fn listings(&mut self, currency: &str) -> Result<Vec<CryptoAsset>, String>;
    fn quotes(
        &mut self,
        identifiers: &[String],
        currency: &str,
        by_slug: bool,
    ) -> Result<Vec<CryptoAsset>, String>;
}

pub type StockLookupRows = Vec<(String, Option<StockQuote>)>;

pub trait UnifiedStockProvider {
    fn lookup(&mut self, query: &str) -> Result<Option<StockLookupRows>, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketPriceExecution {
    pub text: String,
    pub diagnostics: Vec<String>,
}

#[must_use]
pub fn classify_market_price_command(command: &str) -> Option<MarketPriceCommand> {
    match command {
        "/crypto" | "/criptos" => Some(MarketPriceCommand::CryptoOnly),
        "/prices" | "/price" | "/precios" | "/precio" | "/presios" | "/presio" | "/bresio"
        | "/bresios" | "/brecio" | "/brecios" | "/c" => Some(MarketPriceCommand::Unified),
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
    let valid = TIMEFRAMES.map(str::to_owned);
    let query = parse_price_query(text, &valid);
    let mut diagnostics = Vec::new();
    let rendered = match query {
        PriceQuery::UnsupportedTimeframe { timeframe } => invalid_timeframe(&timeframe, locale),
        PriceQuery::AmountConversion(request) => {
            convert_amount(&request, locale, crypto, &mut diagnostics)
        }
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
                    stock_only(&query, locale, stocks, &mut diagnostics, true)
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
                )
            }
        }
    };
    MarketPriceExecution {
        text: rendered,
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
) -> String {
    let listed = match crypto.listings(target_parameter) {
        Ok(rows) => rows,
        Err(error) => {
            diagnostics.push(format!("CoinMarketCap listings: {error}"));
            if !crypto_only && !contains_unambiguous_crypto_symbol(raw_query) {
                let stock = stock_only(raw_query, locale, stocks, diagnostics, false);
                if !stock.is_empty() {
                    if modifiers_unsupported(timeframe, conversion_requested) {
                        return stock_modifier_error(raw_query, locale);
                    }
                    return stock;
                }
            }
            return load_error(locale);
        }
    };
    let mut selection = select_assets(raw_query, &listed);
    let missing = missing_tokens(&selection.rows, &selection.requested);
    if !missing.is_empty() {
        let mut fetched = crypto
            .quotes(&missing, target_parameter, false)
            .unwrap_or_else(|error| {
                diagnostics.push(format!("CoinMarketCap symbol quotes: {error}"));
                Vec::new()
            });
        retain_requested_symbols(&mut fetched, &missing);
        let still_missing = missing_tokens(&fetched, &missing);
        if !still_missing.is_empty() {
            let slugs = still_missing
                .iter()
                .filter(|token| token.len() > 3)
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
                fetched.extend(slug_fetched);
            }
        }
        selection.rows.extend(fetched);
        unique_assets(&mut selection.rows);
        selection.count = selection.rows.len();
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
    if !stock_quotes.is_empty() && modifiers_unsupported(timeframe, conversion_requested) {
        let error = stock_modifier_error(
            &stock_quotes
                .iter()
                .map(|quote| quote.symbol.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            locale,
        );
        let crypto_text = format_assets(
            &selection.rows[..selection.rows.len().min(selection.count)],
            target_symbol,
            target_parameter,
            timeframe.unwrap_or("24h"),
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
        return missing_assets(&unresolved, locale);
    }
    let mut parts = Vec::new();
    let crypto_text = format_assets(
        &selection.rows[..selection.rows.len().min(selection.count)],
        target_symbol,
        target_parameter,
        timeframe.unwrap_or("24h"),
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
                || STABLECOINS.contains(&token.as_str())
                || matches!(token.as_str(), "STABLES" | "STABLECOINS")
        })
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
        let slug = normalized(&coin.slug);
        let exact_symbol = requested_set.contains(&symbol);
        let exact_slug = slug.len() > 3 && requested_set.contains(&slug);
        if !exact_symbol && !exact_slug && index >= top_n {
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
        .filter(|token| !matched.contains(*token))
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
    let mut seen = HashSet::new();
    rows.retain(|coin| {
        let identity = if !coin.id.is_empty() {
            coin.id.clone()
        } else if !coin.symbol.is_empty() {
            coin.symbol.clone()
        } else {
            coin.slug.clone()
        };
        !identity.is_empty() && seen.insert(identity)
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
) -> String {
    if !supported_currency(&request.target_symbol) {
        return unsupported_currency(&request.target_symbol, locale);
    }
    let listed = match crypto.listings(&request.target_parameter) {
        Ok(rows) => rows,
        Err(error) => {
            diagnostics.push(format!("CoinMarketCap conversion listings: {error}"));
            return load_error(locale);
        }
    };
    if let Some(asset) = find_asset(&listed, &request.source_symbol)
        && let Some(quote) = asset.quotes.get(&request.target_parameter)
    {
        let multiplier = if request.target_symbol == "SATS" {
            100_000_000.0
        } else {
            1.0
        };
        return format!(
            "{} {} = {} {}",
            trimmed(request.amount, 8),
            asset.symbol.to_uppercase(),
            trimmed(request.amount * quote.price * multiplier, 8),
            request.target_symbol
        );
    }
    let source_parameter = price_query_parameter(&request.source_symbol);
    let reversed = match crypto.listings(&source_parameter) {
        Ok(rows) => rows,
        Err(error) => {
            diagnostics.push(format!(
                "CoinMarketCap reverse conversion listings: {error}"
            ));
            return load_error(locale);
        }
    };
    let Some(asset) = find_asset(&reversed, &request.target_symbol) else {
        return unsupported_pair(locale);
    };
    let Some(quote) = asset
        .quotes
        .get(&source_parameter)
        .filter(|quote| quote.price != 0.0)
    else {
        return unsupported_pair(locale);
    };
    let amount = if request.source_symbol == "SATS" {
        request.amount / 100_000_000.0
    } else {
        request.amount
    };
    format!(
        "{} {} = {} {}",
        trimmed(request.amount, 8),
        request.source_symbol,
        trimmed(amount / quote.price, 8),
        asset.symbol.to_uppercase()
    )
}

fn find_asset<'a>(assets: &'a [CryptoAsset], token: &str) -> Option<&'a CryptoAsset> {
    let token = normalized(token);
    assets
        .iter()
        .find(|asset| normalized(&asset.symbol) == token || normalized(&asset.name) == token)
}

fn format_assets(rows: &[CryptoAsset], display: &str, parameter: &str, timeframe: &str) -> String {
    rows.iter()
        .filter_map(|asset| {
            let quote = asset.quotes.get(parameter)?;
            let price = quote.price
                * if display == "SATS" {
                    100_000_000.0
                } else {
                    1.0
                };
            let change = match timeframe {
                "1h" => quote.percent_change_1h,
                "7d" => quote.percent_change_7d,
                "30d" => quote.percent_change_30d,
                _ => quote.percent_change_24h,
            }
            .unwrap_or(0.0);
            let fixed = format!("{price:.12}");
            let decimals = fixed.split('.').nth(1).unwrap_or("");
            let zeros = decimals
                .chars()
                .take_while(|character| *character == '0')
                .count();
            Some(format!(
                "{}: {} {} ({}% {})",
                asset.symbol,
                trimmed(price, zeros + 4),
                display,
                signed_trimmed(change, 2),
                timeframe
            ))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn stock_only<S: UnifiedStockProvider>(
    query: &str,
    locale: Locale,
    stocks: &mut S,
    diagnostics: &mut Vec<String>,
    missing_error: bool,
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
        CryptoAsset {
            id: symbol.to_owned(),
            symbol: symbol.to_owned(),
            name: symbol.to_owned(),
            slug: symbol.to_lowercase(),
            quotes: HashMap::from([(
                "USD".to_owned(),
                CryptoQuote {
                    price,
                    percent_change_1h: Some(1.0),
                    percent_change_24h: Some(2.5),
                    percent_change_7d: Some(7.0),
                    percent_change_30d: Some(30.0),
                },
            )]),
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
    fn recognizes_every_public_alias() {
        for alias in [
            "/prices", "/price", "/precios", "/precio", "/presios", "/presio", "/bresio",
            "/bresios", "/brecio", "/brecios", "/c",
        ] {
            assert_eq!(
                classify_market_price_command(alias),
                Some(MarketPriceCommand::Unified)
            );
        }
        for alias in ["/crypto", "/criptos"] {
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
            &mut Crypto::default(),
            &mut Stocks::default(),
        );
        assert_eq!(
            result.text,
            "unsupported timeframe '2h', use: 1h, 24h, 7d, 30d"
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
