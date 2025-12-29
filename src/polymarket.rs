use serde_json::Value;

use crate::http::HttpClient;
use crate::http_cache::cached_get_json;
use crate::storage::Storage;

const TTL_POLYMARKET: u64 = 5;
const EVENTS_URL: &str = "https://gamma-api.polymarket.com/events";

const SLUG_ELECTION: &str = "which-party-wins-most-seats-in-argentina-deputies-election";
const SLUG_SEATS_AFTER: &str = "which-party-holds-the-most-seats-after-argentina-deputies-election";

pub async fn get_polymarket_argentina_election(http: &HttpClient, storage: &Storage) -> String {
    let mut sections = Vec::new();

    for (slug, header, url) in [
        (
            SLUG_ELECTION,
            "Polymarket - ¿Quién gana más bancas en Diputados 2025?",
            "https://polymarket.com/event/which-party-wins-most-seats-in-argentina-deputies-election",
        ),
        (
            SLUG_SEATS_AFTER,
            "Polymarket - ¿Quién queda con más bancas después de Diputados 2025?",
            "https://polymarket.com/event/which-party-holds-the-most-seats-after-argentina-deputies-election",
        ),
    ] {
        if let Some(section) = fetch_event_section(http, storage, slug, header, url).await {
            sections.push(section);
        }
    }

    if sections.is_empty() {
        "No pude traer las probabilidades desde Polymarket".to_string()
    } else {
        sections.join("\n\n")
    }
}

async fn fetch_event_section(
    http: &HttpClient,
    storage: &Storage,
    slug: &str,
    header: &str,
    url: &str,
) -> Option<String> {
    let data = cached_get_json(
        http,
        storage,
        EVENTS_URL,
        Some(&[("slug", slug.to_string())][..]),
        None,
        TTL_POLYMARKET,
    )
    .await?;

    let events = data.get("data").and_then(|v| v.as_array())?;
    if events.is_empty() {
        return None;
    }
    let event = events.first()?;

    let markets = event.get("markets").and_then(|v| v.as_array())?;
    if markets.is_empty() {
        return None;
    }

    let mut odds = Vec::new();
    for market in markets {
        let raw_outcomes = market.get("outcomes").and_then(|v| v.as_str());
        let raw_prices = market.get("outcomePrices").and_then(|v| v.as_str());
        if raw_outcomes.is_none() || raw_prices.is_none() {
            continue;
        }
        let outcomes: Vec<String> = serde_json::from_str(raw_outcomes.unwrap()).ok()?;
        let prices: Vec<f64> = serde_json::from_str(raw_prices.unwrap()).ok()?;
        if outcomes.is_empty() || prices.is_empty() {
            continue;
        }
        let yes_index = outcomes
            .iter()
            .position(|v| v.eq_ignore_ascii_case("yes"))
            .unwrap_or(0);
        if yes_index >= prices.len() {
            continue;
        }
        let price = prices[yes_index];
        let probability = price.clamp(0.0, 1.0) * 100.0;
        let title = market
            .get("groupItemTitle")
            .and_then(|v| v.as_str())
            .or_else(|| market.get("question").and_then(|v| v.as_str()))
            .or_else(|| market.get("slug").and_then(|v| v.as_str()))
            .unwrap_or("?")
            .to_string();
        odds.push((title, probability));
    }

    if odds.is_empty() {
        return None;
    }

    odds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let filtered: Vec<(String, f64)> = odds
        .iter()
        .filter(|(title, _)| {
            let upper = title.trim().to_uppercase();
            upper.starts_with("LLA") || upper.starts_with("UP")
        })
        .cloned()
        .collect();

    let display = if filtered.is_empty() { odds } else { filtered };

    let mut lines = Vec::new();
    lines.push(header.to_string());
    lines.push("".to_string());
    for (title, probability) in display {
        let decimals = if probability < 10.0 { 2 } else { 1 };
        lines.push(format!("- {}: {:.*}%", title, decimals, probability));
    }

    if let Some(updated) = extract_updated_at(event) {
        lines.push("".to_string());
        lines.push(format!("Actualizado: {}", updated));
    }
    lines.push(url.to_string());

    Some(lines.join("\n"))
}

fn extract_updated_at(event: &Value) -> Option<String> {
    let timestamp = event
        .get("updatedAt")
        .and_then(|v| v.as_str())
        .or_else(|| event.get("endDate").and_then(|v| v.as_str()));
    let ts = timestamp?;
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts) {
        let ba = dt.with_timezone(&chrono::FixedOffset::west_opt(3 * 3600)?);
        return Some(ba.format("%Y-%m-%d %H:%M UTC-3").to_string());
    }
    None
}
