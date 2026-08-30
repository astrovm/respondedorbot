//! Deterministic Polymarket outcome reconciliation and ranking.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value;

use crate::locale::Locale;

#[derive(Clone, Debug, PartialEq)]
pub struct MarketOutcome {
    pub title: String,
    pub cached_probability: f64,
    pub live_probability: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedOutcome {
    pub title: String,
    pub percentage: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ElectionQuote {
    pub title: String,
    pub probability: f64,
    pub token_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ElectionEvent {
    pub title: String,
    pub slug: String,
    pub liquidity: f64,
    pub end_date: String,
    pub tags: Vec<String>,
    pub quotes: Vec<ElectionQuote>,
}

#[must_use]
pub fn classify_election_command(command: &str) -> bool {
    matches!(
        command,
        "/eleccion" | "/elecciones" | "/election" | "/elections"
    )
}

fn list(value: Option<&Value>) -> Vec<Value> {
    match value {
        Some(Value::Array(values)) => values.clone(),
        Some(Value::String(value)) => serde_json::from_str::<Value>(value)
            .ok()
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(value) if !value.is_empty() => Some(value.clone()),
        Value::Number(value) if value.as_f64().is_some_and(|value| value != 0.0) => {
            Some(value.to_string())
        }
        Value::Bool(true) => Some("True".to_owned()),
        _ => None,
    }
}

fn quotes(event: &serde_json::Map<String, Value>) -> Vec<ElectionQuote> {
    let Some(markets) = event.get("markets").and_then(Value::as_array) else {
        return Vec::new();
    };
    markets
        .iter()
        .filter_map(Value::as_object)
        .filter(|market| market.get("active") != Some(&Value::Bool(false)))
        .filter(|market| market.get("closed") != Some(&Value::Bool(true)))
        .filter_map(|market| {
            let outcomes = list(market.get("outcomes"));
            let yes_index = outcomes
                .iter()
                .position(|outcome| outcome.as_str() == Some("Yes"))?;
            let prices = list(market.get("outcomePrices"));
            let probability = number(prices.get(yes_index))?.clamp(0.0, 1.0);
            let title = text(
                market
                    .get("groupItemTitle")
                    .or_else(|| market.get("question"))
                    .or_else(|| market.get("slug")),
            )?;
            let token_ids = list(market.get("clobTokenIds"));
            let token_id = text(token_ids.get(yes_index));
            Some(ElectionQuote {
                title,
                probability,
                token_id,
            })
        })
        .collect()
}

#[must_use]
pub fn parse_election_events(value: &Value) -> Vec<ElectionEvent> {
    let Some(values) = value.as_array() else {
        return Vec::new();
    };
    let mut events = values
        .iter()
        .filter_map(Value::as_object)
        .map(|event| {
            let end_date = text(event.get("endDate"))
                .unwrap_or_default()
                .chars()
                .take(10)
                .collect();
            let tags = event
                .get("tags")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_object)
                .filter_map(|tag| text(tag.get("slug")))
                .collect();
            ElectionEvent {
                title: text(event.get("title")).unwrap_or_default(),
                slug: text(event.get("slug")).unwrap_or_default(),
                liquidity: number(event.get("liquidity")).unwrap_or(0.0),
                end_date,
                tags,
                quotes: quotes(event),
            }
        })
        .collect::<Vec<_>>();
    events.sort_by(|left, right| {
        right
            .liquidity
            .partial_cmp(&left.liquidity)
            .unwrap_or(Ordering::Equal)
    });
    events.truncate(10);
    events
}

fn normalized_country_name(value: &str) -> String {
    value
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn country_code(name: &str) -> Option<String> {
    let normalized = normalized_country_name(name);
    let alias = match normalized.as_str() {
        "bosnia-herzegovina" => Some("BA"),
        "cabo verde" | "cape verde" => Some("CV"),
        "congo dr" | "dr congo" => Some("CD"),
        "cote d'ivoire" | "ivory coast" => Some("CI"),
        "england" | "scotland" | "uk" | "wales" => Some("GB"),
        "ir iran" => Some("IR"),
        "korea republic" => Some("KR"),
        "turkey" | "turkiye" => Some("TR"),
        _ => None,
    };
    if let Some(alias) = alias {
        return Some(alias.to_owned());
    }
    static COUNTRY_CODES: OnceLock<HashMap<String, String>> = OnceLock::new();
    COUNTRY_CODES
        .get_or_init(|| {
            serde_json::from_str(include_str!("../data/iso_country_codes.json")).unwrap_or_default()
        })
        .get(&normalized.replace('-', " "))
        .cloned()
}

fn country_flag(code: &str) -> String {
    if code.len() != 2 || !code.bytes().all(|value| value.is_ascii_alphabetic()) {
        return String::new();
    }
    code.to_uppercase()
        .bytes()
        .filter_map(|value| char::from_u32(127_397 + u32::from(value)))
        .collect()
}

#[must_use]
pub fn country_flag_from_name(name: &str) -> String {
    match normalized_country_name(name).as_str() {
        "england" => "🏴󠁧󠁢󠁥󠁮󠁧󠁿".to_owned(),
        "scotland" => "🏴󠁧󠁢󠁳󠁣󠁴󠁿".to_owned(),
        "wales" => "🏴󠁧󠁢󠁷󠁬󠁳󠁿".to_owned(),
        _ => country_code(name).map_or_else(String::new, |code| country_flag(&code)),
    }
}

fn event_flag(event: &ElectionEvent) -> String {
    event
        .tags
        .iter()
        .map(|tag| country_flag_from_name(tag))
        .find(|flag| !flag.is_empty())
        .unwrap_or_default()
}

fn trimmed(value: f64, decimals: usize) -> String {
    let formatted = format!("{value:.decimals$}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

fn usd_compact(value: f64) -> String {
    for (divisor, suffix) in [(1_000_000_000.0, "B"), (1_000_000.0, "M"), (1_000.0, "K")] {
        if value >= divisor {
            return format!("US${}{}", trimmed(value / divisor, 1), suffix);
        }
    }
    format!("US${}", trimmed(value, 0))
}

#[must_use]
pub fn render_elections(
    events: &[ElectionEvent],
    live_prices: &HashMap<String, f64>,
    locale: Locale,
) -> String {
    let (title, liquidity_label, closes_label, error) = match locale {
        Locale::Es => (
            "Polymarket - Elecciones globales por liquidez",
            "Liquidez",
            "Cierra",
            "No pude traer las elecciones desde Polymarket",
        ),
        Locale::En => (
            "Polymarket - Global elections by liquidity",
            "Liquidity",
            "Closes",
            "I could not load the elections from Polymarket",
        ),
    };
    if events.is_empty() {
        return error.to_owned();
    }
    let mut lines = vec![title.to_owned()];
    for event in events {
        if event.title.is_empty() || event.slug.is_empty() {
            continue;
        }
        let outcomes = event
            .quotes
            .iter()
            .map(|quote| MarketOutcome {
                title: quote.title.clone(),
                cached_probability: quote.probability,
                live_probability: quote
                    .token_id
                    .as_ref()
                    .and_then(|token_id| live_prices.get(token_id).copied()),
            })
            .collect::<Vec<_>>();
        let outcomes = rank_outcomes(&outcomes, 2)
            .into_iter()
            .map(|outcome| {
                let decimals = if outcome.percentage < 10.0 { 2 } else { 1 };
                format!(
                    "{} {}%",
                    html_escape(&outcome.title),
                    trimmed(outcome.percentage, decimals)
                )
            })
            .collect::<Vec<_>>();
        let mut details = vec![format!(
            "{liquidity_label} {}",
            usd_compact(event.liquidity)
        )];
        if !event.end_date.is_empty() {
            details.push(format!("{closes_label} {}", event.end_date));
        }
        let flag = event_flag(event);
        let display_title = if flag.is_empty() {
            event.title.clone()
        } else {
            format!("{flag} {}", event.title)
        };
        lines.push(String::new());
        lines.push(format!(
            "<a href=\"https://polymarket.com/event/{}\">{}</a>",
            html_escape(&event.slug),
            html_escape(&display_title)
        ));
        if !outcomes.is_empty() {
            lines.push(outcomes.join(" | "));
        }
        lines.push(details.join(" | "));
    }
    if lines.len() > 1 {
        lines.join("\n")
    } else {
        error.to_owned()
    }
}

/// Reconcile optional live prices, clamp probabilities, and return the highest
/// outcomes. Sorting is stable so provider order breaks equal-price ties.
#[must_use]
pub fn rank_outcomes(outcomes: &[MarketOutcome], limit: usize) -> Vec<RankedOutcome> {
    let mut ranked = outcomes
        .iter()
        .map(|outcome| RankedOutcome {
            title: outcome.title.clone(),
            percentage: outcome
                .live_probability
                .unwrap_or(outcome.cached_probability)
                .clamp(0.0, 1.0)
                * 100.0,
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .percentage
            .partial_cmp(&left.percentage)
            .unwrap_or(Ordering::Equal)
    });
    ranked.truncate(limit);
    ranked
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::{
        MarketOutcome, RankedOutcome, classify_election_command, country_flag_from_name,
        parse_election_events, rank_outcomes, render_elections,
    };
    use crate::locale::Locale;

    fn outcome(title: &str, cached: f64, live: Option<f64>) -> MarketOutcome {
        MarketOutcome {
            title: title.to_owned(),
            cached_probability: cached,
            live_probability: live,
        }
    }

    #[test]
    fn live_prices_override_cached_prices_before_ranking() {
        let ranked = rank_outcomes(
            &[
                outcome("cached leader", 0.9, Some(0.2)),
                outcome("live leader", 0.4, Some(0.8)),
                outcome("cached only", 0.6, None),
            ],
            2,
        );
        assert_eq!(
            ranked,
            vec![
                RankedOutcome {
                    title: "live leader".to_owned(),
                    percentage: 80.0,
                },
                RankedOutcome {
                    title: "cached only".to_owned(),
                    percentage: 60.0,
                },
            ]
        );
    }

    #[test]
    fn probabilities_are_clamped_and_equal_values_keep_provider_order() {
        assert_eq!(
            rank_outcomes(
                &[
                    outcome("first", 2.0, None),
                    outcome("second", 1.0, Some(3.0)),
                    outcome("last", -1.0, None),
                ],
                10,
            ),
            vec![
                RankedOutcome {
                    title: "first".to_owned(),
                    percentage: 100.0,
                },
                RankedOutcome {
                    title: "second".to_owned(),
                    percentage: 100.0,
                },
                RankedOutcome {
                    title: "last".to_owned(),
                    percentage: 0.0,
                },
            ]
        );
    }

    #[test]
    fn zero_limit_and_empty_inputs_are_supported() {
        assert!(rank_outcomes(&[outcome("ignored", 0.5, None)], 0).is_empty());
        assert!(rank_outcomes(&[], 3).is_empty());
    }

    #[test]
    fn election_parser_filters_markets_sorts_liquidity_and_decodes_lists() {
        let events = parse_election_events(&json!([
            {"title":"Lower","slug":"lower","liquidity":"1000","markets":[]},
            {"title":"Higher","slug":"higher","liquidity":2500000,"endDate":"2027-04-30T00:00:00Z","tags":[{"slug":"united-states"}],"markets":[
                {"groupItemTitle":"Candidate A","outcomes":"[\"Yes\",\"No\"]","outcomePrices":"[\"0.42\",\"0.58\"]","clobTokenIds":"[\"a\",\"a-no\"]"},
                {"groupItemTitle":"Closed","outcomes":["Yes","No"],"outcomePrices":[0.99,0.01],"closed":true},
                {"groupItemTitle":"No market","outcomes":["Up","Down"],"outcomePrices":[0.5,0.5]}
            ]}
        ]));
        assert_eq!(events[0].title, "Higher");
        assert_eq!(events[0].end_date, "2027-04-30");
        assert_eq!(events[0].quotes.len(), 1);
        assert_eq!(events[0].quotes[0].token_id.as_deref(), Some("a"));
    }

    #[test]
    fn election_rendering_uses_live_prices_flags_html_and_localization() {
        assert!(classify_election_command("/eleccion"));
        assert!(classify_election_command("/elections"));
        assert!(!classify_election_command("/electionary"));
        assert_eq!(country_flag_from_name("armenia"), "🇦🇲");
        assert_eq!(country_flag_from_name("uk"), "🇬🇧");
        assert_eq!(country_flag_from_name("england"), "🏴󠁧󠁢󠁥󠁮󠁧󠁿");
        let events = parse_election_events(&json!([{
            "title":"US election & runoff","slug":"us-election","tags":[{"slug":"united-states"}],"liquidity":2500000,"endDate":"2027-04-30T00:00:00Z","markets":[
                {"groupItemTitle":"Candidate A","outcomes":["Yes","No"],"outcomePrices":[0.42,0.58],"clobTokenIds":["a","a-no"]},
                {"groupItemTitle":"Candidate B","outcomes":["Yes","No"],"outcomePrices":[0.61,0.39],"clobTokenIds":["b","b-no"]},
                {"groupItemTitle":"Candidate C","outcomes":["Yes","No"],"outcomePrices":[0.20,0.80]}
            ]
        }]));
        let live = HashMap::from([("a".to_owned(), 0.72), ("b".to_owned(), 0.55)]);
        assert_eq!(
            render_elections(&events, &live, Locale::Es),
            concat!(
                "Polymarket - Elecciones globales por liquidez\n\n",
                "<a href=\"https://polymarket.com/event/us-election\">🇺🇸 US election &amp; runoff</a>\n",
                "Candidate A 72% | Candidate B 55%\n",
                "Liquidez US$2.5M | Cierra 2027-04-30"
            )
        );
        assert_eq!(
            render_elections(&[], &HashMap::new(), Locale::En),
            "I could not load the elections from Polymarket"
        );
    }
}
