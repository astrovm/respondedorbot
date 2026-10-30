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
    /// Percent change of `price` over `change_window`.
    pub variation: f64,
    pub change_window: ChangeWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ChangeWindow {
    /// Since the previous regular-session close.
    PreviousClose,
    /// Over the last 24 hours, pre-market and after-hours trades included.
    Hours24,
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

/// Label a quote's change by its window, matching crypto's `24h`.
#[must_use]
pub fn stock_change_period(quote: &StockQuote) -> &'static str {
    match quote.change_window {
        ChangeWindow::Hours24 => "24h",
        ChangeWindow::PreviousClose => "1d",
    }
}

/// Render one quote line. A requested timeframe keeps its own label.
#[must_use]
pub fn render_stock_quote(label: &str, quote: &StockQuote, timeframe: Option<&str>) -> String {
    crate::output_format::quote(
        label,
        quote.price,
        &quote.currency,
        Some(quote.variation),
        timeframe.unwrap_or_else(|| stock_change_period(quote)),
    )
}

#[must_use]
pub fn render_stock_quotes(
    quotes: Option<&[(String, Option<StockQuote>)]>,
    locale: Locale,
) -> String {
    let Some(quotes) = quotes else {
        return match locale {
            Locale::Es => "No pude traer el top de acciones. Probá de nuevo".to_owned(),
            Locale::En => "I could not load the top stocks. Try again".to_owned(),
        };
    };
    let lines = quotes
        .iter()
        .map(|(query, quote)| match quote {
            Some(quote) => render_stock_quote(&quote.symbol, quote, None),
            None => match locale {
                Locale::Es => format!("{query}: no lo encontré"),
                Locale::En => format!("{query}: not found"),
            },
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        match locale {
            Locale::Es => "No pude conseguir ninguna cotización. Probá más tarde".to_owned(),
            Locale::En => "I could not load any quote. Try again later".to_owned(),
        }
    } else {
        lines.join("\n")
    }
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
        lines.push(render_stock_quote(name, quote, None));
    }
    if lines.is_empty() {
        match locale {
            Locale::Es => {
                "No pude traer el precio del petróleo, boludo. Probá más tarde".to_owned()
            }
            Locale::En => "I could not load the oil price. Try again later".to_owned(),
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

fn integer(value: Option<&Value>) -> Option<i64> {
    value?.as_i64()
}

fn chart_result(chart: &Value) -> Option<&serde_json::Map<String, Value>> {
    chart
        .get("chart")?
        .get("result")?
        .as_array()?
        .first()?
        .as_object()
}

/// Pair each bar's timestamp with its close, skipping bars without a trade.
fn timed_closes(result: &serde_json::Map<String, Value>) -> Vec<(i64, f64)> {
    let timestamps = result.get("timestamp").and_then(Value::as_array);
    let closes = result
        .get("indicators")
        .and_then(|indicators| indicators.get("quote"))
        .and_then(Value::as_array)
        .and_then(|quotes| quotes.first())
        .and_then(|quote| quote.get("close"))
        .and_then(Value::as_array);
    let (Some(timestamps), Some(closes)) = (timestamps, closes) else {
        return Vec::new();
    };
    timestamps
        .iter()
        .zip(closes)
        .filter_map(|(time, close)| Some((time.as_i64()?, number(Some(close))?)))
        .filter(|(_, close)| close.is_finite() && *close > 0.0)
        .collect()
}

/// Yahoo's `chartPreviousClose` is the close before the whole requested range,
/// which for the default five-day chart is days old. The previous session's
/// close is the last daily bar from an earlier exchange-local day than the
/// latest regular trade.
fn previous_session_close(
    result: &serde_json::Map<String, Value>,
    meta: &serde_json::Map<String, Value>,
) -> Option<f64> {
    let regular_time = integer(meta.get("regularMarketTime"))?;
    let offset = integer(meta.get("gmtoffset")).unwrap_or(0);
    let day = |time: i64| (time + offset).div_euclid(86_400);
    timed_closes(result)
        .into_iter()
        .filter(|(time, _)| day(*time) < day(regular_time))
        .max_by_key(|(time, _)| *time)
        .map(|(_, close)| close)
}

#[must_use]
pub fn parse_yahoo_quote(response: &Value, fallback_symbol: &str) -> Option<StockQuote> {
    let result = chart_result(response.get("data")?)?;
    let meta = match result.get("meta") {
        Some(Value::Object(meta)) => Some(meta),
        Some(_) => return None,
        None => None,
    };
    let closes = closing_values(result)?;

    let regular_price = number(meta.and_then(|value| value.get("regularMarketPrice")));
    let current = regular_price.or_else(|| number(closes.last().copied()))?;
    // Yahoo's own daily change also covers futures rolling to a new contract.
    let reported_variation = regular_price
        .and(number(
            meta.and_then(|value| value.get("regularMarketChangePercent")),
        ))
        .filter(|variation| variation.is_finite());
    let variation = match reported_variation {
        Some(variation) => variation,
        None => {
            let previous_close = meta
                .and_then(|meta| previous_session_close(result, meta))
                .or_else(|| number(meta.and_then(|value| value.get("previousClose"))))
                .or_else(|| number(meta.and_then(|value| value.get("chartPreviousClose"))))
                .or_else(|| {
                    closes
                        .len()
                        .checked_sub(2)
                        .and_then(|index| closes.get(index))
                        .and_then(|value| number(Some(value)))
                })?;
            if previous_close == 0.0 {
                return None;
            }
            ((current - previous_close) / previous_close) * 100.0
        }
    };
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
        variation,
        change_window: ChangeWindow::PreviousClose,
    })
}

/// Whether the chart is a continuous futures series. Its bars jump when the
/// front contract rolls, so only Yahoo's own daily change is roll-safe.
#[must_use]
pub fn yahoo_is_futures(chart: &Value) -> bool {
    chart_result(chart)
        .and_then(|result| result.get("meta"))
        .and_then(|meta| text(meta.get("instrumentType")))
        == Some("FUTURE")
}

/// The latest price and its change over the last 24 hours, read from an
/// intraday chart with pre-market and after-hours bars. The base is the last
/// trade at least 24 hours before `now_unix`.
#[must_use]
pub fn yahoo_24h_change(chart: &Value, now_unix: i64) -> Option<(f64, f64)> {
    let result = chart_result(chart)?;
    let meta = result.get("meta")?.as_object()?;
    let bars = timed_closes(result);
    let (latest_time, latest_bar) = bars.iter().copied().max_by_key(|(time, _)| *time)?;
    let (_, base) = bars
        .into_iter()
        .filter(|(time, _)| *time <= now_unix - 86_400)
        .max_by_key(|(time, _)| *time)?;
    // Bars lag the regular session's last trade by up to one interval.
    let latest = number(meta.get("regularMarketPrice"))
        .filter(|price| price.is_finite() && *price > 0.0)
        .filter(|_| integer(meta.get("regularMarketTime")).is_some_and(|time| time > latest_time))
        .unwrap_or(latest_bar);
    Some((latest, (latest / base - 1.0) * 100.0))
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
            // Duplicates share the exchange (case-insensitively), so only a
            // missing display name can be filled from a later row.
            if existing.name.is_empty() && !name.is_empty() {
                existing.name.clone_from(&name);
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

fn yahoo_relevance(query: &str, symbol: &str, name: &str) -> u8 {
    let ticker = normalized_ticker(query);
    let symbol = normalized_ticker(symbol);
    if symbol == ticker {
        return 0;
    }
    if yahoo_symbol_base(&symbol).is_some_and(|base| base == ticker) {
        return 1;
    }
    let words = normalized_search_text(query);
    let name = normalized_search_text(name);
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

fn yahoo_candidate_relevance(query: &str, candidate: &StockSearchCandidate) -> u8 {
    yahoo_relevance(query, &candidate.symbol, &candidate.name)
}

/// Whether a provider candidate's display name identifies `query` as a
/// company-name search rather than a ticker-shaped query.
#[must_use]
pub fn yahoo_candidate_name_matches_query(query: &str, candidate: &StockSearchCandidate) -> bool {
    let query_words = normalized_search_text(query);
    let name_words = normalized_search_text(&candidate.name);
    !query_words.is_empty()
        && (name_words == query_words
            || name_words
                .strip_prefix(&query_words)
                .is_some_and(|remainder| remainder.starts_with(' ')))
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

/// Rank loaded stock quotes using the same identity relevance as Yahoo search.
pub fn rank_stock_quotes(query: &str, quotes: &mut [StockQuote]) {
    quotes.sort_by(|left, right| {
        yahoo_relevance(query, &left.symbol, &left.name)
            .cmp(&yahoo_relevance(query, &right.symbol, &right.name))
            .then_with(|| normalized_ticker(&left.symbol).cmp(&normalized_ticker(&right.symbol)))
            .then_with(|| {
                left.exchange
                    .to_ascii_lowercase()
                    .cmp(&right.exchange.to_ascii_lowercase())
            })
            .then_with(|| left.asset_type.cmp(&right.asset_type))
            .then_with(|| left.name.cmp(&right.name))
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
        ChangeWindow, StockQuery, StockQueryPlan, StockQuote, StockSearchCandidate,
        classify_oil_command, classify_stock_command, parse_yahoo_quote, plan_stock_query,
        rank_stock_quotes, rank_yahoo_candidates, render_oil_quotes, render_stock_quote,
        render_stock_quotes, select_yahoo_candidates, select_yahoo_symbol, yahoo_24h_change,
        yahoo_candidate_name_matches_query, yahoo_is_futures,
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
                change_window: ChangeWindow::PreviousClose,
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
                change_window: ChangeWindow::PreviousClose,
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
    fn yahoo_candidate_names_distinguish_company_punctuation_from_tickers() {
        let company = StockSearchCandidate {
            symbol: "RR.L".to_owned(),
            name: "Rolls-Royce Holdings plc".to_owned(),
            exchange: "London".to_owned(),
            asset_type: "Equity".to_owned(),
        };
        assert!(yahoo_candidate_name_matches_query("Rolls-Royce", &company));
        assert!(!yahoo_candidate_name_matches_query("BRK-B", &company));
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
            change_window: ChangeWindow::PreviousClose,
        };
        let wti = super::StockQuote {
            symbol: "CL=F".to_owned(),
            name: String::new(),
            price: 95.0,
            currency: "USD".to_owned(),
            exchange: String::new(),
            asset_type: String::new(),
            variation: 0.0,
            change_window: ChangeWindow::PreviousClose,
        };
        assert_eq!(
            render_oil_quotes(Some(&brent), Some(&wti), Locale::Es),
            "Brent: 98.15 USD (-8.78% 1d)\nWTI: 95 USD (+0% 1d)"
        );
        assert_eq!(
            render_oil_quotes(Some(&brent), None, Locale::En),
            "Brent: 98.15 USD (-8.78% 1d)"
        );
        assert_eq!(
            render_oil_quotes(None, None, Locale::Es),
            "No pude traer el precio del petróleo, boludo. Probá más tarde"
        );
        assert_eq!(
            render_oil_quotes(None, None, Locale::En),
            "I could not load the oil price. Try again later"
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
            change_window: ChangeWindow::PreviousClose,
        };
        let entries = vec![
            ("EXM".to_owned(), Some(quote)),
            ("Missing Corp".to_owned(), None),
        ];
        assert_eq!(
            render_stock_quotes(Some(&entries), Locale::Es),
            "EXM: 12.5 USD (-1.25% 1d)\nMissing Corp: no lo encontré"
        );
        assert_eq!(
            render_stock_quotes(Some(&[("Unknown".to_owned(), None)]), Locale::En),
            "Unknown: not found"
        );
        assert_eq!(
            render_stock_quotes(None, Locale::Es),
            "No pude traer el top de acciones. Probá de nuevo"
        );
        assert_eq!(
            render_stock_quotes(None, Locale::En),
            "I could not load the top stocks. Try again"
        );
        assert_eq!(
            render_stock_quotes(Some(&[]), Locale::En),
            "I could not load any quote. Try again later"
        );
    }

    fn chart(result: serde_json::Value) -> serde_json::Value {
        json!({"data": {"chart": {"result": [result]}}})
    }

    #[test]
    fn quote_parser_handles_malformed_and_partial_chart_sections() {
        // Non-object indicators, quote lists and close series are malformed payloads.
        for malformed in [
            json!({"meta": {"regularMarketPrice": 12, "chartPreviousClose": 10}, "indicators": []}),
            json!({"meta": {"regularMarketPrice": 12, "chartPreviousClose": 10}, "indicators": {"quote": {}}}),
            json!({"meta": {"regularMarketPrice": 12, "chartPreviousClose": 10}, "indicators": {"quote": [{"close": 5}]}}),
        ] {
            assert_eq!(parse_yahoo_quote(&chart(malformed), "EXM"), None);
        }
        // Metadata alone is enough when the close series is absent.
        let quote = parse_yahoo_quote(
            &chart(json!({
                "meta": {"regularMarketPrice": 12, "chartPreviousClose": 10, "longName": "Example Long"},
                "indicators": {"quote": [{}]}
            })),
            "exm",
        );
        assert_eq!(
            quote.map(|quote| (quote.symbol, quote.name, quote.variation)),
            Some(("EXM".to_owned(), "Example Long".to_owned(), 20.0))
        );
        // Without metadata, or with non-numeric prices, the closes are used.
        for result in [
            json!({"indicators": {"quote": [{"close": [8, 10]}]}}),
            json!({"meta": {"regularMarketPrice": true}, "indicators": {"quote": [{"close": [8, 10]}]}}),
        ] {
            let quote = parse_yahoo_quote(&chart(result), "exm");
            assert_eq!(
                quote.map(|quote| (quote.price, quote.variation, quote.currency)),
                Some((10.0, 25.0, "USD".to_owned()))
            );
        }
    }

    #[test]
    fn yahoo_search_skips_non_objects_and_merges_duplicate_listings() {
        let response = json!({"data":{"quotes":[
            "not an object",
            {"quoteType":"EQUITY","symbol":"EXM","exchDisp":"NYSE"},
            {"quoteType":"EQUITY","symbol":"exm","longname":"Example Corp","exchange":"nyse"},
            {"quoteType":"EQUITY","symbol":"EXM","longname":"Ignored Later Name","exchDisp":"NYSE"}
        ]}});
        assert_eq!(
            select_yahoo_candidates(&response),
            vec![StockSearchCandidate {
                symbol: "EXM".to_owned(),
                name: "Example Corp".to_owned(),
                exchange: "NYSE".to_owned(),
                asset_type: "Equity".to_owned(),
            }]
        );
    }

    fn listing(symbol: &str, name: &str, exchange: &str) -> StockSearchCandidate {
        StockSearchCandidate {
            symbol: symbol.to_owned(),
            name: name.to_owned(),
            exchange: exchange.to_owned(),
            asset_type: "Equity".to_owned(),
        }
    }

    #[test]
    fn ranking_orders_every_relevance_tier_then_exchange_and_name() {
        let mut candidates = vec![
            listing("QQQ", "Unrelated", "X"),
            listing("ABC", "The Rock Collective", "X"),
            listing("XYZ", "Rockets Unlimited", "X"),
            listing("ROCKY", "Rocky Mountain", "X"),
            listing("ZZZ", "Rock Inc", "X"),
            listing("ROCK.L", "Rock London", "X"),
            listing("ROCK", "Rock Holdings", "nyse"),
            listing("ROCK", "Rock Holdings", "LSE"),
            listing("ROCK", "B Rock", "LSE"),
        ];
        rank_yahoo_candidates("rock", &mut candidates);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| format!(
                    "{}@{}:{}",
                    candidate.symbol, candidate.exchange, candidate.name
                ))
                .collect::<Vec<_>>(),
            vec![
                "ROCK@LSE:B Rock",
                "ROCK@LSE:Rock Holdings",
                "ROCK@nyse:Rock Holdings",
                "ROCK.L@X:Rock London",
                "ZZZ@X:Rock Inc",
                "ROCKY@X:Rocky Mountain",
                "XYZ@X:Rockets Unlimited",
                "ABC@X:The Rock Collective",
                "QQQ@X:Unrelated",
            ]
        );

        let quote = |symbol: &str, name: &str, exchange: &str, asset_type: &str| StockQuote {
            symbol: symbol.to_owned(),
            name: name.to_owned(),
            price: 1.0,
            currency: "USD".to_owned(),
            exchange: exchange.to_owned(),
            asset_type: asset_type.to_owned(),
            variation: 0.0,
            change_window: ChangeWindow::PreviousClose,
        };
        let mut quotes = vec![
            quote("ROCKY", "Rocky Mountain", "X", "Equity"),
            quote("ROCK", "Rock B", "nyse", "Equity"),
            quote("ROCK", "Rock A", "NYSE", "Equity"),
            quote("ROCK", "Rock Fund", "LSE", "ETF"),
            quote("ROCK", "Rock Equity", "LSE", "Equity"),
        ];
        rank_stock_quotes("$rock", &mut quotes);
        assert_eq!(
            quotes
                .iter()
                .map(|quote| format!("{}@{}:{}", quote.symbol, quote.exchange, quote.name))
                .collect::<Vec<_>>(),
            vec![
                "ROCK@LSE:Rock Fund",
                "ROCK@LSE:Rock Equity",
                "ROCK@NYSE:Rock A",
                "ROCK@nyse:Rock B",
                "ROCKY@X:Rocky Mountain",
            ]
        );
    }

    #[test]
    fn an_empty_quote_list_is_a_localized_failure() {
        assert_eq!(
            render_stock_quotes(Some(&[]), Locale::Es),
            "No pude conseguir ninguna cotización. Probá más tarde"
        );
    }

    #[test]
    fn quotes_without_any_previous_close_are_rejected() {
        assert_eq!(
            parse_yahoo_quote(
                &chart(
                    json!({"meta": {"regularMarketPrice": 10}, "indicators": {"quote": [{"close": [10]}]}})
                ),
                "EXM",
            ),
            None
        );
    }

    // A synthetic US listing whose last regular trade was Friday's close.
    // `chartPreviousClose` is the close before the five-day window, as Yahoo
    // reports it for `range=5d`.
    const DAY: i64 = 86_400;
    const OFFSET: i64 = -4 * 3_600;
    const FRIDAY_CLOSE: i64 = 4 * DAY + 20 * 3_600;

    fn five_day_chart(extra_meta: serde_json::Value) -> serde_json::Value {
        let mut meta = json!({
            "symbol": "EXM",
            "regularMarketPrice": 20.0,
            "regularMarketTime": FRIDAY_CLOSE,
            "chartPreviousClose": 15.0,
            "gmtoffset": OFFSET
        });
        if let (Some(meta), Some(extra)) = (meta.as_object_mut(), extra_meta.as_object()) {
            meta.extend(extra.clone());
        }
        let opens = [0, 1, 2, 3, 4].map(|day| day * DAY + 13 * 3_600 + 1_800);
        chart(json!({
            "meta": meta,
            "timestamp": opens,
            "indicators": {"quote": [{"close": [15.5, 16.0, 17.0, 16.0, 20.0]}]}
        }))
    }

    #[test]
    fn daily_change_uses_the_previous_session_not_the_five_day_window() {
        // Thursday closed at 16, so Friday's 20 is +25%. The old window base
        // of 15 would have reported +33.33% as the daily move.
        let quote = parse_yahoo_quote(&five_day_chart(json!({})), "EXM");
        assert_eq!(quote.map(|quote| quote.variation), Some(25.0));
    }

    #[test]
    fn daily_change_prefers_the_change_yahoo_reports() {
        // A futures roll makes the previous daily bar a different contract,
        // so Yahoo's own daily change wins over the bars.
        let quote = parse_yahoo_quote(
            &five_day_chart(json!({"regularMarketChangePercent": -0.5})),
            "EXM",
        );
        assert_eq!(quote.map(|quote| quote.variation), Some(-0.5));
        let quote = parse_yahoo_quote(
            &five_day_chart(json!({"regularMarketChangePercent": "NaN"})),
            "EXM",
        );
        assert_eq!(quote.map(|quote| quote.variation), Some(25.0));
        // A change without a regular price is not trusted.
        let quote = parse_yahoo_quote(
            &chart(json!({
                "meta": {"regularMarketChangePercent": -0.5},
                "indicators": {"quote": [{"close": [8, 10]}]}
            })),
            "EXM",
        );
        assert_eq!(quote.map(|quote| quote.variation), Some(25.0));
    }

    fn intraday_chart(
        extra_meta: serde_json::Value,
        timestamps: &[i64],
        closes: serde_json::Value,
    ) -> serde_json::Value {
        let mut meta = json!({
            "symbol": "EXM",
            "regularMarketPrice": 20.0,
            "regularMarketTime": FRIDAY_CLOSE,
        });
        if let (Some(meta), Some(extra)) = (meta.as_object_mut(), extra_meta.as_object()) {
            meta.extend(extra.clone());
        }
        json!({"chart": {"result": [{
            "meta": meta,
            "timestamp": timestamps,
            "indicators": {"quote": [{"close": closes}]}
        }]}})
    }

    #[test]
    fn day_change_compares_the_latest_trade_with_the_one_24_hours_earlier() {
        // Thursday's trades, Friday's last regular bar, then after-hours.
        let timestamps = [
            FRIDAY_CLOSE - DAY - 600,
            FRIDAY_CLOSE - DAY,
            FRIDAY_CLOSE - 300,
            FRIDAY_CLOSE + 3_600,
        ];
        let chart = intraday_chart(json!({}), &timestamps, json!([15.0, 16.0, 19.0, 24.0]));
        assert_eq!(
            yahoo_24h_change(&chart, FRIDAY_CLOSE + 3_660),
            Some((24.0, 50.0))
        );
        // A day without trades is flat.
        assert_eq!(
            yahoo_24h_change(&chart, FRIDAY_CLOSE + 2 * DAY),
            Some((24.0, 0.0))
        );
        // The regular session's last trade is newer than the last bar.
        let chart = intraday_chart(json!({}), &timestamps[..3], json!([15.0, 16.0, 19.0]));
        assert_eq!(
            yahoo_24h_change(&chart, FRIDAY_CLOSE + 60),
            Some((20.0, 25.0))
        );
        // An unusable regular price keeps the last bar.
        let chart = intraday_chart(
            json!({"regularMarketPrice": 0.0}),
            &timestamps[..3],
            json!([15.0, 16.0, 19.0]),
        );
        assert_eq!(
            yahoo_24h_change(&chart, FRIDAY_CLOSE + 60),
            Some((19.0, 18.75))
        );
    }

    #[test]
    fn day_change_needs_a_trade_from_24_hours_ago() {
        let recent = intraday_chart(json!({}), &[FRIDAY_CLOSE - 300], json!([19.0]));
        assert_eq!(yahoo_24h_change(&recent, FRIDAY_CLOSE), None);
        let empty = intraday_chart(json!({}), &[], json!([]));
        assert_eq!(yahoo_24h_change(&empty, FRIDAY_CLOSE), None);
        assert_eq!(yahoo_24h_change(&json!({}), FRIDAY_CLOSE), None);
        assert_eq!(
            yahoo_24h_change(&chart(json!({"meta": []})), FRIDAY_CLOSE),
            None
        );
    }

    #[test]
    fn only_futures_series_count_as_futures() {
        let futures = intraday_chart(json!({"instrumentType": "FUTURE"}), &[], json!([]));
        assert!(yahoo_is_futures(&futures));
        let equity = intraday_chart(json!({"instrumentType": "EQUITY"}), &[], json!([]));
        assert!(!yahoo_is_futures(&equity));
        assert!(!yahoo_is_futures(&json!({})));
    }

    #[test]
    fn quote_lines_label_the_change_window() {
        let mut quote = StockQuote {
            symbol: "EXM".to_owned(),
            name: "Example".to_owned(),
            price: 20.0,
            currency: "USD".to_owned(),
            exchange: "NYSE".to_owned(),
            asset_type: "Equity".to_owned(),
            variation: 2.5,
            change_window: ChangeWindow::Hours24,
        };
        assert_eq!(
            render_stock_quotes(Some(&[("EXM".to_owned(), Some(quote.clone()))]), Locale::Es),
            "EXM: 20 USD (+2.5% 24h)"
        );
        assert_eq!(
            render_stock_quote("EXM", &quote, Some("7d")),
            "EXM: 20 USD (+2.5% 7d)"
        );
        quote.change_window = ChangeWindow::PreviousClose;
        assert_eq!(
            render_stock_quote("EXM", &quote, None),
            "EXM: 20 USD (+2.5% 1d)"
        );
    }

    #[test]
    fn same_named_listings_on_one_exchange_order_by_asset_type() {
        let mut candidates = vec![
            StockSearchCandidate {
                asset_type: "ETF".to_owned(),
                ..listing("ROCK", "Rock", "NYSE")
            },
            listing("ROCK", "Rock", "nyse"),
        ];
        rank_yahoo_candidates("rock", &mut candidates);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.asset_type.as_str())
                .collect::<Vec<_>>(),
            vec!["ETF", "Equity"]
        );
    }

    #[test]
    fn only_lowercase_pence_keep_the_sterling_pence_spelling() {
        assert_eq!(super::normalize_currency("gbp"), "GBp");
        assert_eq!(super::normalize_currency("GBP"), "GBP");
        assert_eq!(super::normalize_currency("gxp"), "GXP");
        assert_eq!(super::normalize_currency("usdt"), "USDT");
    }

    #[test]
    fn yahoo_search_keeps_listings_on_other_exchanges_apart() {
        let response = json!({"data":{"quotes":[
            {"quoteType":"EQUITY","symbol":"EXM","exchDisp":"NYSE"},
            {"quoteType":"EQUITY","symbol":"EXM","exchDisp":"NYSE"},
            {"quoteType":"EQUITY","symbol":"EXM","longname":"Example Corp","exchDisp":"LSE"}
        ]}});
        assert_eq!(
            select_yahoo_candidates(&response),
            vec![
                listing("EXM", "", "NYSE"),
                listing("EXM", "Example Corp", "LSE")
            ]
        );
    }

    #[test]
    fn blank_queries_and_dotted_symbols_without_a_base_rank_last() {
        assert_eq!(super::yahoo_symbol_base(".L"), None);
        assert_eq!(super::yahoo_symbol_base("ABC."), None);
        assert_eq!(super::yahoo_relevance("  ", "ABC", "Example Corp"), 6);
        assert!(!yahoo_candidate_name_matches_query(
            "  ",
            &listing("ABC", "Example Corp", "NYSE")
        ));
        assert!(yahoo_candidate_name_matches_query(
            "example corp",
            &listing("ABC", "Example Corp", "NYSE")
        ));
    }
}
