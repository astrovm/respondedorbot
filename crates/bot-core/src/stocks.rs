//! Typed parsing and request planning for stock-market commands.

use std::sync::OnceLock;

use regex::Regex;
use serde::Serialize;
use serde_json::Value;

use crate::locale::Locale;

static SYMBOL_REGEX: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StockQuote {
    pub symbol: String,
    pub name: String,
    pub price: f64,
    pub currency: String,
    pub exchange: String,
    pub asset_type: String,
    pub variation: f64,
}

/// A provider search result before its current quote is loaded.
///
/// Yahoo returns several listings for a short ticker. Keeping the provider's
/// full symbol and display metadata here lets callers rank and present those
/// listings without guessing which exchange the user meant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StockSearchCandidate {
    pub symbol: String,
    pub name: String,
    pub exchange: String,
    pub asset_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StockQuery {
    pub original: String,
    pub normalized: String,
    pub is_symbol: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StockQueryPlan {
    pub raw_query: String,
    pub queries: Vec<StockQuery>,
    pub full_query_fallback: bool,
    pub needs_top_stocks: bool,
}

#[must_use]
pub fn classify_oil_command(command: &str) -> bool {
    matches!(command, "/petroleo" | "/oil")
}

#[must_use]
pub fn classify_stock_command(command: &str) -> bool {
    matches!(
        command,
        "/accion" | "/acciones" | "/s" | "/stock" | "/stocks"
    )
}

#[must_use]
pub fn render_stock_quotes(
    quotes: Option<&[(String, Option<StockQuote>)]>,
    locale: Locale,
) -> String {
    let Some(quotes) = quotes else {
        return match locale {
            Locale::Es => "no pude traer el top de acciones, probá de nuevo".to_owned(),
            Locale::En => "I could not load the top stocks, try again".to_owned(),
        };
    };
    let lines = quotes
        .iter()
        .map(|(query, quote)| match quote {
            Some(quote) => {
                let sign = if quote.variation >= 0.0 { "+" } else { "" };
                format!(
                    "{}: {:.2} {} ({sign}{:.2}% 24h)",
                    quote.symbol, quote.price, quote.currency, quote.variation
                )
            }
            None => match locale {
                Locale::Es => format!("{query}: no se pudo encontrar"),
                Locale::En => format!("{query}: not found"),
            },
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        match locale {
            Locale::Es => "no se pudo obtener ninguna cotización".to_owned(),
            Locale::En => "I could not load any quote".to_owned(),
        }
    } else {
        lines.join("\n")
    }
}

fn trimmed_decimal(value: f64) -> String {
    let formatted = format!("{value:.2}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

#[must_use]
pub fn render_oil_quotes(
    brent: Option<&StockQuote>,
    wti: Option<&StockQuote>,
    locale: Locale,
) -> String {
    let mut lines = Vec::new();
    for (name, quote) in [("Brent", brent), ("WTI", wti)] {
        let Some(quote) = quote else {
            continue;
        };
        let sign = if quote.variation >= 0.0 { "+" } else { "" };
        lines.push(format!(
            "{name}: {} USD ({sign}{}% 24hs)",
            trimmed_decimal(quote.price),
            trimmed_decimal(quote.variation)
        ));
    }
    if lines.is_empty() {
        match locale {
            Locale::Es => "no pude traer el precio del petróleo boludo".to_owned(),
            Locale::En => "I could not load the oil price".to_owned(),
        }
    } else {
        lines.join("\n")
    }
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn text(value: Option<&Value>) -> Option<&str> {
    value?.as_str()
}

fn closing_values(result: &serde_json::Map<String, Value>) -> Option<Vec<&Value>> {
    let indicators = match result.get("indicators") {
        Some(Value::Object(indicators)) => Some(indicators),
        Some(_) => return None,
        None => None,
    };
    let quote = match indicators.and_then(|value| value.get("quote")) {
        Some(Value::Array(quotes)) => quotes.first()?.as_object()?,
        Some(_) => return None,
        None => return Some(Vec::new()),
    };
    match quote.get("close") {
        Some(Value::Array(values)) => {
            Some(values.iter().filter(|value| !value.is_null()).collect())
        }
        Some(_) => None,
        None => Some(Vec::new()),
    }
}

#[must_use]
pub fn parse_yahoo_quote(response: &Value, fallback_symbol: &str) -> Option<StockQuote> {
    let result = response
        .get("data")?
        .get("chart")?
        .get("result")?
        .as_array()?
        .first()?
        .as_object()?;
    let meta = match result.get("meta") {
        Some(Value::Object(meta)) => Some(meta),
        Some(_) => return None,
        None => None,
    };
    let closes = closing_values(result)?;

    let current = number(meta.and_then(|value| value.get("regularMarketPrice")))
        .or_else(|| number(closes.last().copied()))?;
    let previous_close =
        number(meta.and_then(|value| value.get("chartPreviousClose"))).or_else(|| {
            closes
                .len()
                .checked_sub(2)
                .and_then(|index| closes.get(index))
                .and_then(|value| number(Some(value)))
        })?;
    if previous_close == 0.0 {
        return None;
    }
    let symbol = text(meta.and_then(|value| value.get("symbol")))
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback_symbol)
        .to_uppercase();
    let name = text(meta.and_then(|value| value.get("shortName")))
        .filter(|value| !value.is_empty())
        .or_else(|| text(meta.and_then(|value| value.get("longName"))))
        .unwrap_or_default()
        .to_owned();
    let currency = text(meta.and_then(|value| value.get("currency")))
        .filter(|value| !value.is_empty())
        .map_or_else(|| "USD".to_owned(), normalize_currency);
    let exchange = text(meta.and_then(|value| value.get("exchangeName")))
        .unwrap_or_default()
        .to_owned();
    Some(StockQuote {
        symbol,
        name,
        price: current,
        currency,
        exchange,
        asset_type: String::new(),
        variation: ((current - previous_close) / previous_close) * 100.0,
    })
}

fn normalize_currency(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() == 3
        && bytes[0].eq_ignore_ascii_case(&b'g')
        && bytes[1].eq_ignore_ascii_case(&b'b')
        && bytes[2] == b'p'
    {
        "GBp".to_owned()
    } else {
        value.to_uppercase()
    }
}

#[must_use]
pub fn select_yahoo_candidates(response: &Value) -> Vec<StockSearchCandidate> {
    let Some(quotes) = response
        .get("data")
        .and_then(|data| data.get("quotes"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    for quote in quotes {
        let Some(quote) = quote.as_object() else {
            continue;
        };
        let Some(asset_type) = quote
            .get("quoteType")
            .and_then(Value::as_str)
            .and_then(yahoo_asset_type)
        else {
            continue;
        };
        let Some(symbol) = quote
            .get("symbol")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|symbol| !symbol.is_empty())
        else {
            continue;
        };
        let name = quote
            .get("longname")
            .and_then(Value::as_str)
            .or_else(|| quote.get("shortname").and_then(Value::as_str))
            .map_or_else(String::new, |name| name.trim().to_owned());
        let exchange = quote
            .get("exchDisp")
            .and_then(Value::as_str)
            .or_else(|| quote.get("exchange").and_then(Value::as_str))
            .map_or_else(String::new, |exchange| exchange.trim().to_owned());
        if let Some(existing) =
            candidates
                .iter_mut()
                .find(|candidate: &&mut StockSearchCandidate| {
                    candidate.symbol.eq_ignore_ascii_case(symbol)
                        && candidate.exchange.eq_ignore_ascii_case(&exchange)
                        && candidate.asset_type.eq_ignore_ascii_case(asset_type)
                })
        {
            if existing.name.is_empty() && !name.is_empty() {
                existing.name.clone_from(&name);
            }
            if existing.exchange.is_empty() && !exchange.is_empty() {
                existing.exchange.clone_from(&exchange);
            }
            continue;
        }
        candidates.push(StockSearchCandidate {
            symbol: symbol.to_owned(),
            name,
            exchange,
            asset_type: asset_type.to_owned(),
        });
    }
    candidates
}

fn yahoo_asset_type(value: &str) -> Option<&'static str> {
    match value {
        "EQUITY" => Some("Equity"),
        "ETF" => Some("ETF"),
        "MUTUALFUND" => Some("Mutual fund"),
        "INDEX" => Some("Index"),
        "FUTURE" => Some("Future"),
        _ => None,
    }
}

fn normalized_search_text(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('$')
        .to_ascii_lowercase()
        .split(|character: char| !character.is_alphanumeric())
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
}

fn normalized_ticker(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('$')
        .to_ascii_uppercase()
        .replace(' ', "")
}

fn yahoo_symbol_base(symbol: &str) -> Option<&str> {
    let (base, suffix) = symbol.rsplit_once('.')?;
    (!base.is_empty() && !suffix.is_empty()).then_some(base)
}

fn yahoo_candidate_relevance(query: &str, candidate: &StockSearchCandidate) -> u8 {
    let ticker = normalized_ticker(query);
    let symbol = normalized_ticker(&candidate.symbol);
    if symbol == ticker {
        return 0;
    }
    if yahoo_symbol_base(&symbol).is_some_and(|base| base == ticker) {
        return 1;
    }
    let words = normalized_search_text(query);
    let name = normalized_search_text(&candidate.name);
    if !words.is_empty() && name == words {
        return 2;
    }
    if !ticker.is_empty() && symbol.starts_with(&ticker) {
        return 3;
    }
    if !words.is_empty() && name.starts_with(&words) {
        return 4;
    }
    if !words.is_empty() && name.contains(&words) {
        return 5;
    }
    6
}

/// Rank search results without choosing a single fuzzy winner.
pub fn rank_yahoo_candidates(query: &str, candidates: &mut [StockSearchCandidate]) {
    candidates.sort_by(|left, right| {
        yahoo_candidate_relevance(query, left)
            .cmp(&yahoo_candidate_relevance(query, right))
            .then_with(|| normalized_ticker(&left.symbol).cmp(&normalized_ticker(&right.symbol)))
            .then_with(|| {
                left.exchange
                    .to_ascii_lowercase()
                    .cmp(&right.exchange.to_ascii_lowercase())
            })
            .then_with(|| {
                left.name
                    .to_ascii_lowercase()
                    .cmp(&right.name.to_ascii_lowercase())
            })
            .then_with(|| left.asset_type.cmp(&right.asset_type))
    });
}

#[must_use]
pub fn select_yahoo_symbol(response: &Value) -> Option<String> {
    select_yahoo_candidates(response)
        .first()
        .map(|candidate| candidate.symbol.clone())
}

#[must_use]
pub fn plan_stock_query(message: &str) -> StockQueryPlan {
    let raw_query = message.trim().to_owned();
    let (originals, full_query_fallback) = if raw_query.contains(',') {
        (
            raw_query
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            false,
        )
    } else {
        let parts = raw_query
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let fallback = parts.len() > 1;
        (parts, fallback)
    };
    let symbol_regex = SYMBOL_REGEX.get_or_init(|| Regex::new(r"^[A-Z0-9.\^=\-]{1,30}$"));
    let queries = originals
        .into_iter()
        .take(20)
        .map(|original| {
            let normalized = original.to_uppercase().trim_start_matches('$').to_owned();
            let is_symbol = symbol_regex
                .as_ref()
                .is_ok_and(|regex| regex.is_match(&normalized));
            StockQuery {
                original,
                normalized,
                is_symbol,
            }
        })
        .collect::<Vec<_>>();
    StockQueryPlan {
        raw_query,
        needs_top_stocks: queries.is_empty(),
        queries,
        full_query_fallback,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        StockQuery, StockQueryPlan, classify_oil_command, classify_stock_command,
        parse_yahoo_quote, plan_stock_query, rank_yahoo_candidates, render_oil_quotes,
        render_stock_quotes, select_yahoo_candidates, select_yahoo_symbol,
    };
    use crate::locale::Locale;

    #[test]
    fn quote_parser_uses_metadata_and_calculates_variation() {
        assert_eq!(
            parse_yahoo_quote(
                &json!({"data":{"chart":{"result":[{"meta":{"symbol":"exm.ba","shortName":"Example","regularMarketPrice":123.45,"chartPreviousClose":120,"currency":"ars","exchangeName":"Synthetic"},"indicators":{"quote":[{"close":[118,120,123.45]}]}}]}}}),
                "fallback",
            ),
            Some(super::StockQuote {
                symbol: "EXM.BA".to_owned(),
                name: "Example".to_owned(),
                price: 123.45,
                currency: "ARS".to_owned(),
                exchange: "Synthetic".to_owned(),
                asset_type: String::new(),
                variation: 2.875000000000002,
            })
        );
    }

    #[test]
    fn quote_parser_falls_back_to_closes_and_defaults() {
        assert_eq!(
            parse_yahoo_quote(
                &json!({"data":{"chart":{"result":[{"meta":{},"indicators":{"quote":[{"close":[null,"10","12"]}]}}]}}}),
                "alt",
            ),
            Some(super::StockQuote {
                symbol: "ALT".to_owned(),
                name: String::new(),
                price: 12.0,
                currency: "USD".to_owned(),
                exchange: String::new(),
                asset_type: String::new(),
                variation: 20.0,
            })
        );
    }

    #[test]
    fn quote_parser_rejects_missing_results_and_zero_previous_close() {
        assert_eq!(parse_yahoo_quote(&json!({}), "EXM"), None);
        assert_eq!(
            parse_yahoo_quote(
                &json!({"data":{"chart":{"result":[{"meta":{"regularMarketPrice":10,"chartPreviousClose":0}}]}}}),
                "EXM",
            ),
            None
        );
        assert_eq!(
            parse_yahoo_quote(
                &json!({"data":{"chart":{"result":[{"meta":[],"indicators":{"quote":[{"close":[10,12]}]}}]}}}),
                "EXM",
            ),
            None
        );
        assert_eq!(
            parse_yahoo_quote(
                &json!({"data":{"chart":{"result":[{"meta":{"regularMarketPrice":12,"chartPreviousClose":10},"indicators":{"quote":[]}}]}}}),
                "EXM",
            ),
            None
        );
    }

    #[test]
    fn symbol_selection_filters_provider_types_and_empty_symbols() {
        let response = json!({"data":{"quotes":[
            {"quoteType":"CRYPTOCURRENCY","symbol":"BTC-USD"},
            {"quoteType":"ETF","symbol":""},
            {"quoteType":"EQUITY","symbol":"EXM"}
        ]}});
        assert_eq!(select_yahoo_symbol(&response), Some("EXM".to_owned()));
        assert_eq!(select_yahoo_symbol(&json!([])), None);
    }

    #[test]
    fn yahoo_search_preserves_supported_candidates_and_ranks_exchange_matches() {
        let response = json!({"data":{"quotes":[
            {"quoteType":"CRYPTOCURRENCY","symbol":"RKH-USD"},
            {"quoteType":"EQUITY","symbol":"RKHNF","longname":"Rockhaven Resources Ltd.","exchange":"PNK","exchDisp":"OTC Markets"},
            {"quoteType":"EQUITY","symbol":"RKH.L","longname":"Rockhopper Exploration plc","exchange":"LSE","exchDisp":"London"},
            {"quoteType":"EQUITY","symbol":"RKHL.XC","longname":"Rockhopper Exploration plc","exchange":"CXE","exchDisp":"CXE"},
            {"quoteType":"ETF","symbol":"RKHX","shortname":"RKH ETF","exchange":"NMS"}
        ]}});
        let mut candidates = select_yahoo_candidates(&response);
        rank_yahoo_candidates("rkh", &mut candidates);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.symbol.as_str())
                .collect::<Vec<_>>(),
            ["RKH.L", "RKHL.XC", "RKHNF", "RKHX"]
        );
        assert_eq!(candidates[0].name, "Rockhopper Exploration plc");
        assert_eq!(candidates[0].exchange, "London");
        assert_eq!(candidates[0].asset_type, "Equity");
    }

    #[test]
    fn quote_parser_preserves_penny_sterling_units() {
        let quote = parse_yahoo_quote(
            &json!({"data":{"chart":{"result":[{"meta":{"symbol":"RKH.L","regularMarketPrice":12.5,"chartPreviousClose":10,"currency":"GBp","exchangeName":"London"},"indicators":{"quote":[{"close":[10,12.5]}]}}]}}}),
            "RKH.L",
        );
        assert_eq!(
            quote.as_ref().map(|quote| quote.currency.as_str()),
            Some("GBp")
        );
    }

    #[test]
    fn query_plan_preserves_commas_spaces_symbols_and_limit() {
        assert_eq!(
            plan_stock_query("  $exm.ba, Example Holdings ,, "),
            StockQueryPlan {
                raw_query: "$exm.ba, Example Holdings ,,".to_owned(),
                queries: vec![
                    StockQuery {
                        original: "$exm.ba".to_owned(),
                        normalized: "EXM.BA".to_owned(),
                        is_symbol: true,
                    },
                    StockQuery {
                        original: "Example Holdings".to_owned(),
                        normalized: "EXAMPLE HOLDINGS".to_owned(),
                        is_symbol: false,
                    },
                ],
                full_query_fallback: false,
                needs_top_stocks: false,
            }
        );
        let spaced = plan_stock_query("Example Holdings");
        assert!(spaced.full_query_fallback);
        assert_eq!(spaced.queries.len(), 2);
        assert!(plan_stock_query(" ").needs_top_stocks);
        assert_eq!(
            plan_stock_query(
                &(0..25)
                    .map(|index| format!("S{index}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .queries
            .len(),
            20
        );
    }

    #[test]
    fn oil_commands_render_partial_full_and_localized_failure_results() {
        assert!(classify_oil_command("/petroleo"));
        assert!(classify_oil_command("/oil"));
        assert!(!classify_oil_command("/oily"));
        let brent = super::StockQuote {
            symbol: "BZ=F".to_owned(),
            name: String::new(),
            price: 98.15,
            currency: "USD".to_owned(),
            exchange: String::new(),
            asset_type: String::new(),
            variation: -8.782_527_881_040_9,
        };
        let wti = super::StockQuote {
            symbol: "CL=F".to_owned(),
            name: String::new(),
            price: 95.0,
            currency: "USD".to_owned(),
            exchange: String::new(),
            asset_type: String::new(),
            variation: 0.0,
        };
        assert_eq!(
            render_oil_quotes(Some(&brent), Some(&wti), Locale::Es),
            "Brent: 98.15 USD (-8.78% 24hs)\nWTI: 95 USD (+0% 24hs)"
        );
        assert_eq!(
            render_oil_quotes(Some(&brent), None, Locale::En),
            "Brent: 98.15 USD (-8.78% 24hs)"
        );
        assert_eq!(
            render_oil_quotes(None, None, Locale::Es),
            "no pude traer el precio del petróleo boludo"
        );
        assert_eq!(
            render_oil_quotes(None, None, Locale::En),
            "I could not load the oil price"
        );
    }

    #[test]
    fn stock_commands_render_quotes_missing_entries_and_localized_failures() {
        assert!(classify_stock_command("/acciones"));
        assert!(classify_stock_command("/stocks"));
        assert!(classify_stock_command("/accion"));
        assert!(classify_stock_command("/s"));
        assert!(classify_stock_command("/stock"));
        assert!(!classify_stock_command("/shares"));
        let quote = super::StockQuote {
            symbol: "EXM".to_owned(),
            name: "Example".to_owned(),
            price: 12.5,
            currency: "USD".to_owned(),
            exchange: "Synthetic".to_owned(),
            asset_type: String::new(),
            variation: -1.25,
        };
        let entries = vec![
            ("EXM".to_owned(), Some(quote)),
            ("Missing Corp".to_owned(), None),
        ];
        assert_eq!(
            render_stock_quotes(Some(&entries), Locale::Es),
            "EXM: 12.50 USD (-1.25% 24h)\nMissing Corp: no se pudo encontrar"
        );
        assert_eq!(
            render_stock_quotes(Some(&[("Unknown".to_owned(), None)]), Locale::En),
            "Unknown: not found"
        );
        assert_eq!(
            render_stock_quotes(None, Locale::Es),
            "no pude traer el top de acciones, probá de nuevo"
        );
        assert_eq!(
            render_stock_quotes(None, Locale::En),
            "I could not load the top stocks, try again"
        );
        assert_eq!(
            render_stock_quotes(Some(&[]), Locale::En),
            "I could not load any quote"
        );
    }
}
