//! Parsing for the unified crypto and stock price command.

use regex::{Regex, RegexBuilder};
use std::sync::OnceLock;

const AMOUNT_CONVERSION_PATTERN: &str =
    r"^\s*([0-9]+(?:[\.,][0-9]+)?)\s+(\$?[a-zA-Z0-9]+)\s+(?:in|to|a|en)\s+(\$?[a-zA-Z0-9]+)\s*$";
const CONVERSION_ONLY_PATTERN: &str = r"^\s*(?:in|to|a|en)\s+(\$?[a-zA-Z0-9]+)\s*$";
const CONVERSION_SPLIT_PATTERN: &str = r"\s+(?:in|to|a|en)\s+";
const PROVIDER_SCOPE_PATTERN: &str = r"^\s*(crypto|stock)\s*:\s*(.*?)\s*$";
const CONVERSION_MODIFIER_PATTERN: &str = r"(?:^|\s)(?:in|to|a|en)\s+\$?[a-zA-Z0-9]+\s*$";
const UNSUPPORTED_TIMEFRAME_PATTERN: &str = r"^\d+[hd]$";

static AMOUNT_CONVERSION_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();
static CONVERSION_ONLY_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();
static CONVERSION_SPLIT_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();
static PROVIDER_SCOPE_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();
static CONVERSION_MODIFIER_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();
static UNSUPPORTED_TIMEFRAME_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();

/// A provider explicitly selected by the user.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderScope {
    Crypto,
    Stock,
}

/// A request to convert a fixed amount between two symbols.
#[derive(Clone, Debug, PartialEq)]
pub struct AmountConversion {
    pub amount: f64,
    pub source_symbol: String,
    pub target_symbol: String,
    pub target_parameter: String,
}

/// A normalized unified-price query.
#[derive(Clone, Debug, PartialEq)]
pub enum PriceQuery {
    /// A timeframe-shaped suffix was supplied but is unsupported.
    UnsupportedTimeframe { timeframe: String },
    /// A fixed-amount conversion such as `2 BTC in USD`.
    AmountConversion(AmountConversion),
    /// A list, limit, crypto, or stock lookup.
    Assets {
        query: String,
        timeframe: Option<String>,
        target_symbol: String,
        target_parameter: String,
        conversion_requested: bool,
        provider_scope: Option<ProviderScope>,
    },
}

/// Normalize a symbol exactly once before matching or provider use.
#[must_use]
pub fn normalize_price_symbol(value: &str) -> String {
    value
        .to_uppercase()
        .replace(' ', "")
        .trim_start_matches('$')
        .to_owned()
}

/// Translate display-only SATS into the provider's BTC query parameter.
#[must_use]
pub fn price_query_parameter(symbol: &str) -> String {
    if symbol == "SATS" {
        "BTC".to_owned()
    } else {
        symbol.to_owned()
    }
}

/// Parse the full pure-logic portion of a unified price query.
#[must_use]
pub fn parse_price_query(text: &str, valid_timeframes: &[String]) -> PriceQuery {
    let (text, timeframe) = parse_timeframe(text, valid_timeframes);
    if timeframe.is_none()
        && let Some(candidate) = unsupported_timeframe(&text)
    {
        return PriceQuery::UnsupportedTimeframe {
            timeframe: candidate,
        };
    }

    if let Some(conversion) = parse_amount_conversion(&text) {
        return PriceQuery::AmountConversion(conversion);
    }

    let conversion_requested = has_conversion_modifier(&text);
    let (query, target_symbol, target_parameter) = parse_conversion_only(&text);
    let (provider_scope, query) = parse_provider_scope(&query);
    PriceQuery::Assets {
        query,
        timeframe,
        target_symbol,
        target_parameter,
        conversion_requested,
        provider_scope,
    }
}

fn cached_regex(
    cell: &'static OnceLock<Result<Regex, regex::Error>>,
    pattern: &str,
) -> Option<&'static Regex> {
    cell.get_or_init(|| RegexBuilder::new(pattern).case_insensitive(true).build())
        .as_ref()
        .ok()
}

fn parse_amount_conversion(text: &str) -> Option<AmountConversion> {
    let regex = cached_regex(&AMOUNT_CONVERSION_REGEX, AMOUNT_CONVERSION_PATTERN)?;
    let captures = regex.captures(text)?;
    let amount_text = captures.get(1)?.as_str().replace(',', ".");
    let source_symbol = normalize_price_symbol(captures.get(2)?.as_str());
    let target_symbol = normalize_price_symbol(captures.get(3)?.as_str());
    Some(AmountConversion {
        amount: amount_text.parse().ok()?,
        source_symbol,
        target_parameter: price_query_parameter(&target_symbol),
        target_symbol,
    })
}

fn parse_conversion_only(text: &str) -> (String, String, String) {
    if let Some(regex) = cached_regex(&CONVERSION_ONLY_REGEX, CONVERSION_ONLY_PATTERN)
        && let Some(captures) = regex.captures(text)
        && let Some(target_match) = captures.get(1)
    {
        let target = normalize_price_symbol(target_match.as_str());
        let parameter = price_query_parameter(&target);
        return (String::new(), target, parameter);
    }

    if let Some(regex) = cached_regex(&CONVERSION_SPLIT_REGEX, CONVERSION_SPLIT_PATTERN) {
        let mut parts = regex.splitn(text, 2);
        if let (Some(query), Some(target)) = (parts.next(), parts.next()) {
            let target = normalize_price_symbol(target.trim());
            let parameter = price_query_parameter(&target);
            return (query.trim().to_owned(), target, parameter);
        }
    }

    (text.to_owned(), "USD".to_owned(), "USD".to_owned())
}

fn parse_provider_scope(text: &str) -> (Option<ProviderScope>, String) {
    let Some(regex) = cached_regex(&PROVIDER_SCOPE_REGEX, PROVIDER_SCOPE_PATTERN) else {
        return (None, text.to_owned());
    };
    let Some(captures) = regex.captures(text) else {
        return (None, text.to_owned());
    };
    let scope = match captures
        .get(1)
        .map(|value| value.as_str().to_ascii_lowercase())
    {
        Some(value) if value == "crypto" => ProviderScope::Crypto,
        Some(_) => ProviderScope::Stock,
        None => return (None, text.to_owned()),
    };
    let query = captures
        .get(2)
        .map_or_else(String::new, |value| value.as_str().to_owned());
    (Some(scope), query)
}

fn has_conversion_modifier(text: &str) -> bool {
    cached_regex(&CONVERSION_MODIFIER_REGEX, CONVERSION_MODIFIER_PATTERN)
        .is_some_and(|regex| regex.is_match(text))
}

fn parse_timeframe(text: &str, valid: &[String]) -> (String, Option<String>) {
    let trimmed = text.trim();
    let (remaining, candidate) = trimmed.rfind(char::is_whitespace).map_or_else(
        || ("", trimmed),
        |position| (&trimmed[..position], trimmed[position..].trim_start()),
    );
    let normalized = candidate.to_lowercase();
    if valid.iter().any(|item| item == &normalized) {
        return (remaining.trim().to_owned(), Some(normalized));
    }
    (trimmed.to_owned(), None)
}

fn unsupported_timeframe(text: &str) -> Option<String> {
    let candidate = text.split_whitespace().next_back()?.to_lowercase();
    if cached_regex(&UNSUPPORTED_TIMEFRAME_REGEX, UNSUPPORTED_TIMEFRAME_PATTERN)
        .is_some_and(|regex| regex.is_match(&candidate))
    {
        Some(candidate)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{
        AmountConversion, PriceQuery, ProviderScope, normalize_price_symbol, parse_price_query,
        price_query_parameter,
    };

    fn timeframes() -> Vec<String> {
        ["1h", "24h", "7d", "30d"].map(str::to_owned).to_vec()
    }

    #[test]
    fn normalizes_symbols_and_provider_parameters() {
        assert_eq!(normalize_price_symbol(" $$bt c "), "BTC");
        assert_eq!(price_query_parameter("SATS"), "BTC");
        assert_eq!(price_query_parameter("USD"), "USD");
    }

    #[test]
    fn parses_amount_conversions_with_all_supported_prepositions() {
        for preposition in ["in", "to", "a", "en"] {
            assert_eq!(
                parse_price_query(&format!(" 2,5 $btc {preposition} $sats "), &timeframes()),
                PriceQuery::AmountConversion(AmountConversion {
                    amount: 2.5,
                    source_symbol: "BTC".to_owned(),
                    target_symbol: "SATS".to_owned(),
                    target_parameter: "BTC".to_owned(),
                })
            );
        }
    }

    #[test]
    fn rejects_malformed_amount_conversions_as_asset_queries() {
        assert!(matches!(
            parse_price_query("2. BTC in USD", &timeframes()),
            PriceQuery::Assets { .. }
        ));
    }

    #[test]
    fn parses_conversion_only_and_provider_scope() {
        assert_eq!(
            parse_price_query(" stock: Mercado Libre in EUR ", &timeframes()),
            PriceQuery::Assets {
                query: "Mercado Libre".to_owned(),
                timeframe: None,
                target_symbol: "EUR".to_owned(),
                target_parameter: "EUR".to_owned(),
                conversion_requested: true,
                provider_scope: Some(ProviderScope::Stock),
            }
        );
        assert_eq!(
            parse_price_query("in sats", &timeframes()),
            PriceQuery::Assets {
                query: String::new(),
                timeframe: None,
                target_symbol: "SATS".to_owned(),
                target_parameter: "BTC".to_owned(),
                conversion_requested: true,
                provider_scope: None,
            }
        );
    }

    #[test]
    fn extracts_valid_timeframes_before_other_parsing() {
        assert_eq!(
            parse_price_query("crypto:BTC 7D", &timeframes()),
            PriceQuery::Assets {
                query: "BTC".to_owned(),
                timeframe: Some("7d".to_owned()),
                target_symbol: "USD".to_owned(),
                target_parameter: "USD".to_owned(),
                conversion_requested: false,
                provider_scope: Some(ProviderScope::Crypto),
            }
        );
    }

    #[test]
    fn identifies_only_timeframe_shaped_unsupported_suffixes() {
        assert_eq!(
            parse_price_query("btc 2h", &timeframes()),
            PriceQuery::UnsupportedTimeframe {
                timeframe: "2h".to_owned()
            }
        );
        assert!(matches!(
            parse_price_query("btc soon", &timeframes()),
            PriceQuery::Assets { .. }
        ));
        assert!(matches!(
            parse_price_query("btc 2é", &timeframes()),
            PriceQuery::Assets { .. }
        ));
        assert_eq!(
            parse_price_query("btc ٢h", &timeframes()),
            PriceQuery::UnsupportedTimeframe {
                timeframe: "٢h".to_owned()
            }
        );
    }

    proptest! {
        #[test]
        fn arbitrary_unicode_queries_never_panic(input in ".{0,512}") {
            let _query = parse_price_query(&input, &timeframes());
        }
    }
}

/// Explicit chart duration. `m` means minutes; `mo` means 30 days.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChartPeriod {
    pub seconds: i64,
}

impl ChartPeriod {
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.to_ascii_lowercase();
        let split = value.find(|c: char| !c.is_ascii_digit())?;
        let count = value[..split].parse::<i64>().ok()?;
        let unit = match &value[split..] {
            "m" => 60,
            "h" => 3600,
            "d" => 86400,
            "w" => 7 * 86400,
            "mo" => 30 * 86400,
            "y" => 365 * 86400,
            _ => return None,
        };
        let seconds = count.checked_mul(unit)?;
        (seconds > 0 && seconds <= 100 * 365 * 86400).then_some(Self { seconds })
    }

    #[must_use]
    pub fn yahoo_interval(self) -> &'static str {
        match self.seconds {
            0..=86400 => "1m",
            86401..=604800 => "15m",
            604801..=5184000 => "1h",
            5184001..=63072000 => "1d",
            _ => "1wk",
        }
    }
}

#[cfg(test)]
mod chart_period_tests {
    use super::ChartPeriod;
    #[test]
    fn ranges_are_unambiguous_bounded_and_choose_usable_candles() {
        for (input, seconds, interval) in [
            ("1m", 60, "1m"),
            ("2h", 7200, "1m"),
            ("2d", 172800, "15m"),
            ("1w", 604800, "15m"),
            ("1mo", 2592000, "1h"),
            ("1y", 31536000, "1d"),
            ("5y", 157680000, "1wk"),
        ] {
            let period = ChartPeriod::parse(input);
            assert_eq!(period.map(|p| p.seconds), Some(seconds));
            assert_eq!(period.map(ChartPeriod::yahoo_interval), Some(interval));
        }
        for input in ["0m", "-1d", "9999999999999999999999y", "101y", "1q", "btc"] {
            assert!(ChartPeriod::parse(input).is_none(), "{input}");
        }
    }
}
