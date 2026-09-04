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
    pub variation: f64,
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
    matches!(command, "/accion" | "/acciones" | "/stock" | "/stocks")
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
        .unwrap_or("USD")
        .to_uppercase();
    let exchange = text(meta.and_then(|value| value.get("exchangeName")))
        .unwrap_or_default()
        .to_owned();
    Some(StockQuote {
        symbol,
        name,
        price: current,
        currency,
        exchange,
        variation: ((current - previous_close) / previous_close) * 100.0,
    })
}

#[must_use]
pub fn select_yahoo_symbol(response: &Value) -> Option<String> {
    const ALLOWED_TYPES: [&str; 5] = ["EQUITY", "ETF", "MUTUALFUND", "INDEX", "FUTURE"];
    response
        .get("data")?
        .get("quotes")?
        .as_array()?
        .iter()
        .filter_map(Value::as_object)
        .find_map(|quote| {
            let quote_type = quote.get("quoteType")?.as_str()?;
            if !ALLOWED_TYPES.contains(&quote_type) {
                return None;
            }
            quote
                .get("symbol")?
                .as_str()
                .filter(|symbol| !symbol.is_empty())
                .map(str::to_owned)
        })
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
        parse_yahoo_quote, plan_stock_query, render_oil_quotes, render_stock_quotes,
        select_yahoo_symbol,
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
            variation: -8.782_527_881_040_9,
        };
        let wti = super::StockQuote {
            symbol: "CL=F".to_owned(),
            name: String::new(),
            price: 95.0,
            currency: "USD".to_owned(),
            exchange: String::new(),
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
        assert!(classify_stock_command("/stock"));
        assert!(!classify_stock_command("/shares"));
        let quote = super::StockQuote {
            symbol: "EXM".to_owned(),
            name: "Example".to_owned(),
            price: 12.5,
            currency: "USD".to_owned(),
            exchange: "Synthetic".to_owned(),
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
