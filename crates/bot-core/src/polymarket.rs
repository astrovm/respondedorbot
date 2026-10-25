//! Deterministic Polymarket outcome reconciliation and ranking.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::locale::Locale;

/// Most markets shown per searched event. Sports events can carry hundreds of
/// props, so only the most relevant ones reach the model.
pub const SEARCH_MARKETS_PER_EVENT: usize = 8;

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchOutcome {
    pub title: String,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchMarket {
    pub title: String,
    pub outcomes: Vec<SearchOutcome>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchEvent {
    pub title: String,
    pub slug: String,
    pub liquidity: f64,
    pub volume: f64,
    pub end_date: String,
    pub markets: Vec<SearchMarket>,
    pub hidden_markets: usize,
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

/// Polymarket sends numbers as JSON numbers or strings. Strings like "NaN" or
/// "inf" parse as floats but are not prices, so only finite values count.
#[must_use]
pub fn provider_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
    .filter(|value: &f64| value.is_finite())
}

fn number(value: Option<&Value>) -> Option<f64> {
    provider_number(value?)
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

fn is_open(item: &Map<String, Value>) -> bool {
    item.get("active") != Some(&Value::Bool(false))
        && item.get("closed") != Some(&Value::Bool(true))
}

/// Single-market events send an empty `groupItemTitle` and sports moneylines
/// send null, so fall through to the question instead of dropping the market.
fn market_title(market: &Map<String, Value>) -> Option<String> {
    ["groupItemTitle", "question", "slug"]
        .iter()
        .find_map(|key| text(market.get(*key)))
}

fn open_markets(event: &Map<String, Value>) -> impl Iterator<Item = &Map<String, Value>> {
    event
        .get("markets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter(|market| is_open(market))
}

fn quotes(event: &Map<String, Value>) -> Vec<ElectionQuote> {
    open_markets(event)
        .filter_map(|market| {
            let outcomes = list(market.get("outcomes"));
            let yes_index = outcomes
                .iter()
                .position(|outcome| outcome.as_str() == Some("Yes"))?;
            let prices = list(market.get("outcomePrices"));
            let probability = number(prices.get(yes_index))?.clamp(0.0, 1.0);
            let title = market_title(market)?;
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
    // Whole numbers have no fractional zeros to drop: US$500 must stay 500.
    if !formatted.contains('.') {
        return formatted;
    }
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

fn percent(percentage: f64) -> String {
    let decimals = if percentage < 10.0 { 2 } else { 1 };
    format!("{}%", trimmed(percentage, decimals))
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
            "Elecciones en Polymarket por liquidez",
            "Liquidez",
            "Cierra",
            "No pude traer las elecciones de Polymarket. Probá más tarde",
        ),
        Locale::En => (
            "Polymarket elections by liquidity",
            "Liquidity",
            "Closes",
            "I could not load the elections from Polymarket. Try again later",
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
                format!(
                    "{} {}",
                    html_escape(&outcome.title),
                    percent(outcome.percentage)
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

fn search_market(market: &Map<String, Value>) -> Option<(SearchMarket, f64)> {
    let names = list(market.get("outcomes"));
    let prices = list(market.get("outcomePrices"));
    if names.is_empty() || names.len() != prices.len() {
        return None;
    }
    let outcomes = names
        .iter()
        .zip(&prices)
        .map(|(name, price)| {
            Some(SearchOutcome {
                title: text(Some(name))?,
                probability: number(Some(price))?.clamp(0.0, 1.0),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let volume = ["volumeNum", "volume"]
        .iter()
        .find_map(|key| number(market.get(*key)))
        .unwrap_or(0.0);
    Some((
        SearchMarket {
            title: market_title(market)?,
            outcomes,
        },
        volume,
    ))
}

fn yes_probability(market: &SearchMarket) -> f64 {
    market
        .outcomes
        .iter()
        .find(|outcome| outcome.title == "Yes")
        .map_or(0.0, |outcome| outcome.probability)
}

fn search_event(event: &Map<String, Value>) -> Option<SearchEvent> {
    if !is_open(event) {
        return None;
    }
    let mut markets = open_markets(event)
        .filter_map(search_market)
        .collect::<Vec<_>>();
    if markets.is_empty() {
        return None;
    }
    // Mutually exclusive outcomes (candidates, rate decisions) read best as a
    // leaderboard. Anything else, like sports props, keeps its busiest markets.
    if event.get("negRisk") == Some(&Value::Bool(true)) {
        markets.sort_by(|left, right| {
            yes_probability(&right.0)
                .partial_cmp(&yes_probability(&left.0))
                .unwrap_or(Ordering::Equal)
        });
    } else {
        markets.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    }
    let hidden_markets = markets.len().saturating_sub(SEARCH_MARKETS_PER_EVENT);
    markets.truncate(SEARCH_MARKETS_PER_EVENT);
    Some(SearchEvent {
        title: text(event.get("title"))?,
        slug: text(event.get("slug"))?,
        liquidity: number(event.get("liquidity")).unwrap_or(0.0),
        volume: number(event.get("volume")).unwrap_or(0.0),
        end_date: text(event.get("endDate"))
            .unwrap_or_default()
            .chars()
            .take(10)
            .collect(),
        markets: markets.into_iter().map(|(market, _)| market).collect(),
        hidden_markets,
    })
}

/// Parse a Gamma `public-search` response into open events in relevance order.
#[must_use]
pub fn parse_search_events(value: &Value) -> Vec<SearchEvent> {
    value
        .get("events")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter_map(search_event)
        .collect()
}

#[must_use]
pub fn market_search_error(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "No pude buscar en Polymarket. Probá más tarde",
        Locale::En => "I could not search Polymarket. Try again later",
    }
}

fn search_market_line(event: &SearchEvent, market: &SearchMarket) -> String {
    let outcomes = match market.outcomes.as_slice() {
        [yes, no] if yes.title == "Yes" && no.title == "No" => std::slice::from_ref(yes),
        outcomes => outcomes,
    }
    .iter()
    .map(|outcome| format!("{} {}", outcome.title, percent(outcome.probability * 100.0)))
    .collect::<Vec<_>>()
    .join(" / ");
    if market.title == event.title {
        format!("- {outcomes}")
    } else {
        format!("- {}: {outcomes}", market.title)
    }
}

/// Plain-text search results for the AI tool. Prices are implied probabilities.
#[must_use]
pub fn render_market_search(query: &str, events: &[SearchEvent], locale: Locale) -> String {
    let (header, empty, liquidity_label, volume_label, closes_label, more_label) = match locale {
        Locale::Es => (
            "Mercados de Polymarket para",
            "No encontré mercados abiertos en Polymarket para",
            "Liquidez",
            "Volumen",
            "Cierra",
            "mercados más",
        ),
        Locale::En => (
            "Polymarket markets for",
            "No open Polymarket markets found for",
            "Liquidity",
            "Volume",
            "Closes",
            "more markets",
        ),
    };
    if events.is_empty() {
        return format!("{empty} \"{query}\"");
    }
    let mut lines = vec![format!("{header} \"{query}\" (%: implied probability)")];
    for (index, event) in events.iter().enumerate() {
        let mut details = vec![
            format!("{liquidity_label} {}", usd_compact(event.liquidity)),
            format!("{volume_label} {}", usd_compact(event.volume)),
        ];
        if !event.end_date.is_empty() {
            details.push(format!("{closes_label} {}", event.end_date));
        }
        lines.push(String::new());
        lines.push(format!("{}. {}", index + 1, event.title));
        lines.push(format!("https://polymarket.com/event/{}", event.slug));
        lines.push(details.join(" | "));
        lines.extend(
            event
                .markets
                .iter()
                .map(|market| search_market_line(event, market)),
        );
        if event.hidden_markets > 0 {
            lines.push(format!("- +{} {more_label}", event.hidden_markets));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use proptest::prelude::*;
    use serde_json::json;

    use super::{
        MarketOutcome, RankedOutcome, SEARCH_MARKETS_PER_EVENT, SearchEvent, SearchMarket,
        SearchOutcome, classify_election_command, country_flag_from_name, market_search_error,
        parse_election_events, parse_search_events, rank_outcomes, render_elections,
        render_market_search,
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
                "Elecciones en Polymarket por liquidez\n\n",
                "<a href=\"https://polymarket.com/event/us-election\">🇺🇸 US election &amp; runoff</a>\n",
                "Candidate A 72% | Candidate B 55%\n",
                "Liquidez US$2.5M | Cierra 2027-04-30"
            )
        );
        assert_eq!(
            render_elections(&[], &HashMap::new(), Locale::En),
            "I could not load the elections from Polymarket. Try again later"
        );
    }

    #[test]
    fn parser_falls_back_through_titles_and_skips_malformed_markets() {
        assert!(parse_election_events(&json!({"events": []})).is_empty());
        let events = parse_election_events(&json!([{
            "title": true,
            "slug": 7,
            "liquidity": "n/a",
            "markets": [
                {"question": "Question title", "outcomes": ["Yes"], "outcomePrices": ["0.3"], "clobTokenIds": [123]},
                {"slug": "slug-title", "outcomes": ["Yes"], "outcomePrices": [0.2], "clobTokenIds": [0]},
                {"outcomes": ["Yes"], "outcomePrices": [0.1]},
                {"groupItemTitle": "Bad price", "outcomes": ["Yes"], "outcomePrices": [true]},
                {"groupItemTitle": "Inactive", "active": false, "outcomes": ["Yes"], "outcomePrices": [0.9]}
            ]
        }, {"title": "No markets", "slug": "none", "markets": "not a list"}]));
        assert_eq!(events[0].title, "True");
        assert_eq!(events[0].slug, "7");
        assert_eq!(events[0].liquidity, 0.0);
        assert_eq!(
            events[0]
                .quotes
                .iter()
                .map(|quote| (quote.title.as_str(), quote.token_id.as_deref()))
                .collect::<Vec<_>>(),
            vec![("Question title", Some("123")), ("slug-title", None)]
        );
        assert!(events[1].quotes.is_empty());
    }

    #[test]
    fn english_rendering_handles_small_markets_and_skips_untitled_events() {
        let events = parse_election_events(&json!([
            {"title": "Local vote", "slug": "local-vote", "liquidity": 500, "markets": [
                {"groupItemTitle": "Long shot", "outcomes": ["Yes"], "outcomePrices": [0.055]}
            ]},
            {"title": "Quiet vote", "slug": "quiet", "liquidity": 1500, "markets": []},
            {"title": "", "slug": "untitled", "liquidity": 9_000_000_000_i64, "markets": []}
        ]));
        assert_eq!(
            render_elections(&events, &HashMap::new(), Locale::En),
            concat!(
                "Polymarket elections by liquidity\n\n",
                "<a href=\"https://polymarket.com/event/quiet\">Quiet vote</a>\n",
                "Liquidity US$1.5K\n\n",
                "<a href=\"https://polymarket.com/event/local-vote\">Local vote</a>\n",
                "Long shot 5.5%\n",
                "Liquidity US$500"
            )
        );
        let untitled = parse_election_events(&json!([
            {"title": "", "slug": "untitled", "liquidity": 2_000_000_000_i64}
        ]));
        assert_eq!(
            render_elections(&untitled, &HashMap::new(), Locale::Es),
            "No pude traer las elecciones de Polymarket. Probá más tarde"
        );
        assert_eq!(super::usd_compact(2_000_000_000.0), "US$2B");
        assert_eq!(super::country_flag("ABC"), "");
        assert_eq!(super::country_flag("A1"), "");
        assert_eq!(country_flag_from_name("atlantis"), "");
    }

    #[test]
    fn events_without_a_slug_are_skipped() {
        let events = parse_election_events(&json!([
            {"title": "No link", "slug": "", "liquidity": 9_000, "markets": []},
            {"title": "Linked", "slug": "linked", "liquidity": 1_000, "markets": []}
        ]));
        let text = render_elections(&events, &HashMap::new(), Locale::En);
        assert!(text.contains("Linked"));
        assert!(!text.contains("No link"));
    }

    #[test]
    fn election_titles_fall_back_past_empty_and_null_group_titles() {
        let events = parse_election_events(&json!([{
            "title": "Vote", "slug": "vote", "markets": [
                {"groupItemTitle": "", "question": "Will A win?", "outcomes": ["Yes", "No"], "outcomePrices": [0.3, 0.7]},
                {"groupItemTitle": null, "question": "Will B win?", "outcomes": ["Yes", "No"], "outcomePrices": [0.2, 0.8]}
            ]
        }]));
        assert_eq!(
            events[0]
                .quotes
                .iter()
                .map(|quote| quote.title.as_str())
                .collect::<Vec<_>>(),
            ["Will A win?", "Will B win?"]
        );
    }

    fn binary(title: &str, yes: f64) -> serde_json::Value {
        json!({"groupItemTitle": title, "outcomes": "[\"Yes\", \"No\"]", "outcomePrices": format!("[\"{yes}\", \"{}\"]", 1.0 - yes)})
    }

    #[test]
    fn search_ranks_exclusive_outcomes_by_probability_and_caps_markets() {
        let mut markets = (0..10)
            .map(|index| binary(&format!("Candidate {index}"), f64::from(index) / 100.0))
            .collect::<Vec<_>>();
        markets.push(json!({"groupItemTitle": "Closed", "closed": true, "outcomes": ["Yes", "No"], "outcomePrices": [0.99, 0.01]}));
        markets.push(json!({"groupItemTitle": "Inactive", "active": false, "outcomes": ["Yes", "No"], "outcomePrices": [0.98, 0.02]}));
        let events = parse_search_events(&json!({"events": [{
            "title": "Who wins?", "slug": "who-wins", "negRisk": true, "liquidity": "2500", "volume": 1_200_000,
            "endDate": "2027-10-25T03:59:00Z", "markets": markets
        }]}));
        let event = &events[0];
        assert_eq!(event.markets.len(), SEARCH_MARKETS_PER_EVENT);
        assert_eq!(event.hidden_markets, 2);
        assert_eq!(event.markets[0].title, "Candidate 9");
        assert_eq!(event.markets[7].title, "Candidate 2");
        assert_eq!(event.liquidity, 2500.0);
        assert_eq!(event.end_date, "2027-10-25");
    }

    #[test]
    fn search_keeps_busiest_props_and_non_binary_outcomes() {
        let events = parse_search_events(&json!({"events": [{
            "title": "Home vs. Away", "slug": "home-away", "markets": [
                {"groupItemTitle": "Overtime?", "outcomes": ["Yes", "No"], "outcomePrices": ["0.5", "0.5"], "volumeNum": 10},
                {"groupItemTitle": null, "question": "Home vs. Away", "outcomes": ["Home", "Away"], "outcomePrices": ["0.685", "0.315"], "volumeNum": 9000},
                {"groupItemTitle": "O/U 210.5", "outcomes": ["Over", "Under"], "outcomePrices": ["0.52", "0.48"], "volume": "500"},
                {"groupItemTitle": "Mismatched", "outcomes": ["Over", "Under"], "outcomePrices": ["0.52"]},
                {"groupItemTitle": "Bad price", "outcomes": ["Yes", "No"], "outcomePrices": [true, 0.5]}
            ]
        }]}));
        assert_eq!(
            events[0]
                .markets
                .iter()
                .map(|market| market.title.as_str())
                .collect::<Vec<_>>(),
            ["Home vs. Away", "O/U 210.5", "Overtime?"]
        );
        assert_eq!(events[0].hidden_markets, 0);
        assert_eq!(events[0].markets[0].outcomes[1].title, "Away");
    }

    #[test]
    fn search_skips_closed_untitled_and_empty_events_and_keeps_relevance_order() {
        let events = parse_search_events(&json!({"events": [
            {"title": "Low liquidity", "slug": "first", "liquidity": 1, "markets": [binary("A", 0.5)]},
            {"title": "Closed", "slug": "closed", "closed": true, "markets": [binary("A", 0.5)]},
            {"title": "", "slug": "untitled", "markets": [binary("A", 0.5)]},
            {"title": "No slug", "markets": [binary("A", 0.5)]},
            {"title": "All closed", "slug": "all-closed", "markets": [{"groupItemTitle": "A", "closed": true, "outcomes": ["Yes", "No"], "outcomePrices": [0.5, 0.5]}]},
            {"title": "No markets", "slug": "none", "markets": "not a list"},
            "not an object",
            {"title": "High liquidity", "slug": "second", "liquidity": 9_000_000, "markets": [binary("A", 0.5)]}
        ]}));
        assert_eq!(
            events
                .iter()
                .map(|event| event.slug.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert!(parse_search_events(&json!([])).is_empty());
        assert!(parse_search_events(&json!({"events": {}})).is_empty());
    }

    fn search_event(markets: Vec<SearchMarket>, hidden_markets: usize) -> SearchEvent {
        SearchEvent {
            title: "Who wins?".to_owned(),
            slug: "who-wins".to_owned(),
            liquidity: 1_100_000.0,
            volume: 2_500.0,
            end_date: "2027-10-25".to_owned(),
            markets,
            hidden_markets,
        }
    }

    fn search_market(title: &str, outcomes: &[(&str, f64)]) -> SearchMarket {
        SearchMarket {
            title: title.to_owned(),
            outcomes: outcomes
                .iter()
                .map(|(title, probability)| SearchOutcome {
                    title: (*title).to_owned(),
                    probability: *probability,
                })
                .collect(),
        }
    }

    #[test]
    fn search_rendering_shows_links_details_and_compact_binary_odds() {
        let events = [
            search_event(
                vec![
                    search_market("Candidate A", &[("Yes", 0.485), ("No", 0.515)]),
                    search_market("Candidate B", &[("Yes", 0.0005), ("No", 0.9995)]),
                    search_market("Who wins?", &[("Home", 0.685), ("Away", 0.315)]),
                    search_market("Yes only", &[("Yes", 1.0)]),
                ],
                56,
            ),
            SearchEvent {
                end_date: String::new(),
                ..search_event(Vec::new(), 0)
            },
        ];
        assert_eq!(
            render_market_search("synthetic", &events, Locale::En),
            concat!(
                "Polymarket markets for \"synthetic\" (%: implied probability)\n\n",
                "1. Who wins?\n",
                "https://polymarket.com/event/who-wins\n",
                "Liquidity US$1.1M | Volume US$2.5K | Closes 2027-10-25\n",
                "- Candidate A: Yes 48.5%\n",
                "- Candidate B: Yes 0.05%\n",
                "- Home 68.5% / Away 31.5%\n",
                "- Yes only: Yes 100%\n",
                "- +56 more markets\n\n",
                "2. Who wins?\n",
                "https://polymarket.com/event/who-wins\n",
                "Liquidity US$1.1M | Volume US$2.5K"
            )
        );
        assert!(
            render_market_search("x", &events[..1], Locale::Es)
                .contains("Liquidez US$1.1M | Volumen US$2.5K | Cierra 2027-10-25")
        );
        assert!(render_market_search("x", &events[..1], Locale::Es).contains("- +56 mercados más"));
    }

    #[test]
    fn search_rendering_reports_empty_results_and_errors_per_locale() {
        assert_eq!(
            render_market_search("nada", &[], Locale::Es),
            "No encontré mercados abiertos en Polymarket para \"nada\""
        );
        assert_eq!(
            render_market_search("none", &[], Locale::En),
            "No open Polymarket markets found for \"none\""
        );
        assert_eq!(
            market_search_error(Locale::Es),
            "No pude buscar en Polymarket. Probá más tarde"
        );
        assert_eq!(
            market_search_error(Locale::En),
            "I could not search Polymarket. Try again later"
        );
    }

    #[test]
    fn search_events_round_trip_through_the_cache_encoding() {
        let events = vec![search_event(
            vec![search_market("A", &[("Yes", 0.25), ("No", 0.75)])],
            3,
        )];
        let encoded = serde_json::to_value(&events).unwrap_or_default();
        assert_eq!(
            serde_json::from_value::<Vec<SearchEvent>>(encoded).ok(),
            Some(events)
        );
    }

    #[test]
    fn non_finite_provider_numbers_are_not_prices() {
        assert_eq!(super::provider_number(&json!("0.25")), Some(0.25));
        for value in [json!("NaN"), json!("inf"), json!("-infinity"), json!(null)] {
            assert_eq!(super::provider_number(&value), None, "{value}");
        }
        let events = parse_search_events(&json!({"events": [{
            "title": "Odd numbers", "slug": "odd", "liquidity": "inf", "volume": "NaN", "markets": [
                {"groupItemTitle": "NaN price", "outcomes": ["Yes", "No"], "outcomePrices": ["NaN", "0.5"]},
                {"groupItemTitle": "Fine", "outcomes": ["Yes", "No"], "outcomePrices": ["0.5", "0.5"], "volumeNum": "inf"}
            ]
        }]}));
        assert_eq!(events[0].liquidity, 0.0);
        assert_eq!(events[0].volume, 0.0);
        assert_eq!(events[0].markets.len(), 1);
        let elections = parse_election_events(&json!([{
            "title": "Vote", "slug": "vote", "liquidity": "inf",
            "markets": [{"groupItemTitle": "A", "outcomes": ["Yes"], "outcomePrices": ["NaN"]}]
        }]));
        assert_eq!(elections[0].liquidity, 0.0);
        assert!(elections[0].quotes.is_empty());
    }

    fn arbitrary_number() -> impl Strategy<Value = serde_json::Value> {
        prop_oneof![
            any::<f64>().prop_map(|value| json!(value)),
            any::<f64>().prop_map(|value| json!(value.to_string())),
            Just(json!("NaN")),
            Just(json!(null)),
        ]
    }

    fn arbitrary_market() -> impl Strategy<Value = serde_json::Value> {
        (
            prop::option::of("[A-Za-z ]{0,8}"),
            prop::collection::vec(
                prop::sample::select(vec!["Yes", "No", "Home", "Over"]),
                0..4,
            ),
            prop::collection::vec(arbitrary_number(), 0..4),
            arbitrary_number(),
            any::<(bool, bool)>(),
        )
            .prop_map(|(title, outcomes, prices, volume, (active, closed))| {
                json!({
                    "groupItemTitle": title, "question": "Question?", "outcomes": outcomes,
                    "outcomePrices": prices, "volumeNum": volume, "active": active, "closed": closed
                })
            })
    }

    proptest! {
        #[test]
        fn searched_events_stay_bounded_finite_and_cacheable(
            neg_risk in any::<bool>(),
            liquidity in arbitrary_number(),
            markets in prop::collection::vec(arbitrary_market(), 0..20),
        ) {
            let open = markets
                .iter()
                .filter(|market| market["active"] == json!(true) && market["closed"] == json!(false))
                .count();
            let value = json!({"events": [{
                "title": "Event", "slug": "event", "negRisk": neg_risk,
                "liquidity": liquidity, "markets": markets
            }]});
            let events = parse_search_events(&value);
            for event in &events {
                prop_assert!(event.markets.len() <= SEARCH_MARKETS_PER_EVENT);
                prop_assert!(event.markets.len() + event.hidden_markets <= open);
                prop_assert!(event.liquidity.is_finite());
                for outcome in event.markets.iter().flat_map(|market| &market.outcomes) {
                    prop_assert!((0.0..=1.0).contains(&outcome.probability));
                }
                if neg_risk {
                    let leaders = event.markets.iter().map(super::yes_probability).collect::<Vec<_>>();
                    prop_assert!(leaders.windows(2).all(|pair| pair[0] >= pair[1]));
                }
            }
            let encoded = serde_json::to_value(&events).unwrap_or_default();
            prop_assert_eq!(serde_json::from_value::<Vec<SearchEvent>>(encoded).ok(), Some(events));
        }
    }
}
