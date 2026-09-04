//! Pure token-signal detection, selection, formatting, and callback state.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use url::form_urlencoded;

use crate::locale::Locale;
use crate::telegram_actions::{CopyTextButton, InlineKeyboardButton, InlineKeyboardMarkup};

pub const SIGNAL_STATE_TTL_SECONDS: i64 = 3_600;
pub const SIGNAL_REFRESH_COOLDOWN_SECONDS: i64 = 15;
const PUMP_INITIAL_REAL_TOKENS: f64 = 793_100_000_000_000.0;

static SOLANA_ADDRESS: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"^[1-9A-HJ-NP-Za-km-z]{32,44}(?:pump)?$").ok());
static EVM_ADDRESS: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"^0x[a-fA-F0-9]{40}$").ok());
static TOKEN_SYMBOL: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"^\$([A-Za-z][A-Za-z0-9]{1,31})$").ok());

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenAddress {
    pub chain_id: String,
    pub network: String,
    pub tag: String,
    pub address: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PairToken {
    pub address: String,
    pub name: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairTransactions {
    pub buys: Value,
    pub sells: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairTransactionWindows {
    pub h1: PairTransactions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairPriceChange {
    pub h1: Value,
    pub h24: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairVolume {
    pub h24: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairLiquidity {
    pub usd: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairWebsite {
    pub label: String,
    pub url: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairSocial {
    #[serde(rename = "type")]
    pub kind: String,
    pub url: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PairInfo {
    pub header: String,
    pub image_url: String,
    pub open_graph: String,
    pub websites: Vec<PairWebsite>,
    pub socials: Vec<PairSocial>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TokenPair {
    pub chain_id: String,
    pub url: String,
    pub pair_address: String,
    pub base_token: PairToken,
    pub price_usd: Value,
    pub price_change: PairPriceChange,
    pub market_cap: Value,
    pub fdv: Value,
    pub volume: PairVolume,
    pub liquidity: PairLiquidity,
    pub txns: PairTransactionWindows,
    pub pair_created_at: Value,
    pub info: PairInfo,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PumpMetadata {
    pub image_uri: String,
    pub twitter: String,
    pub telegram: String,
    pub website: String,
    pub created_timestamp: Value,
    pub real_token_reserves: Value,
    pub total_supply: Value,
    pub ath_market_cap: Value,
    pub ath_market_cap_timestamp: Value,
    pub complete: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenSignal {
    pub token: TokenAddress,
    pub pair: TokenPair,
    pub candles: Vec<Vec<f64>>,
    pub supply: Option<f64>,
    pub token_image_url: Option<String>,
    pub socials: BTreeMap<String, String>,
    pub pump: Option<PumpMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalState {
    pub chat_id: String,
    pub message_id: i64,
    pub source_message_id: i64,
    pub requester_id: String,
    pub chain_id: String,
    pub network: String,
    pub tag: String,
    pub address: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_refresh_at: Option<i64>,
}

impl SignalState {
    #[must_use]
    pub fn token(&self) -> TokenAddress {
        TokenAddress {
            chain_id: self.chain_id.clone(),
            network: self.network.clone(),
            tag: self.tag.clone(),
            address: self.address.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalQuery {
    Address(TokenAddress),
    Symbol(String),
}

#[must_use]
pub fn detect_signal_query(text: &str) -> Option<SignalQuery> {
    let candidate = text.trim();
    if candidate.is_empty() || candidate.chars().any(char::is_whitespace) {
        return None;
    }
    if EVM_ADDRESS
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(candidate))
    {
        return Some(SignalQuery::Address(TokenAddress {
            chain_id: "ethereum".to_owned(),
            network: "eth".to_owned(),
            tag: "ETH".to_owned(),
            address: candidate.to_ascii_lowercase(),
        }));
    }
    if SOLANA_ADDRESS
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(candidate))
    {
        return Some(SignalQuery::Address(TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: candidate.to_owned(),
        }));
    }
    TOKEN_SYMBOL
        .as_ref()
        .and_then(|pattern| pattern.captures(candidate))
        .and_then(|captures| captures.get(1))
        .map(|symbol| SignalQuery::Symbol(symbol.as_str().to_ascii_lowercase()))
}

#[must_use]
pub fn signal_state_key(signal_id: &str) -> String {
    format!("token_signal:{signal_id}")
}

#[must_use]
pub fn stable_signal_id(chat_id: i64, message_id: i64, requester_id: i64, now_unix: i64) -> String {
    let digest =
        Sha256::digest(format!("{chat_id}:{message_id}:{requester_id}:{now_unix}").as_bytes());
    digest
        .iter()
        .take(6)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn number(value: &Value) -> f64 {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse::<f64>().ok()))
        .unwrap_or(0.0)
}

#[must_use]
pub fn pair_rank(pair: &TokenPair) -> (f64, f64) {
    (number(&pair.liquidity.usd), number(&pair.volume.h24))
}

fn compare_rank(left: &TokenPair, right: &TokenPair) -> Ordering {
    let (left_liquidity, left_volume) = pair_rank(left);
    let (right_liquidity, right_volume) = pair_rank(right);
    left_liquidity
        .total_cmp(&right_liquidity)
        .then_with(|| left_volume.total_cmp(&right_volume))
}

#[must_use]
pub fn choose_best_pair(pairs: &[TokenPair]) -> Option<TokenPair> {
    pairs
        .iter()
        .max_by(|left, right| compare_rank(left, right))
        .cloned()
}

#[must_use]
pub fn token_from_pair(pair: &TokenPair) -> Option<TokenAddress> {
    let (network, tag) = match pair.chain_id.as_str() {
        "solana" => ("solana", "SOL"),
        "ethereum" => ("eth", "ETH"),
        chain
            if !chain.is_empty()
                && EVM_ADDRESS.as_ref().is_some_and(|pattern| {
                    pattern.is_match(&pair.base_token.address.to_ascii_lowercase())
                }) =>
        {
            (chain, "")
        }
        _ => return None,
    };
    if pair.base_token.address.is_empty() {
        return None;
    }
    Some(TokenAddress {
        chain_id: pair.chain_id.clone(),
        network: network.to_owned(),
        tag: if tag.is_empty() {
            pair.chain_id.to_ascii_uppercase()
        } else {
            tag.to_owned()
        },
        address: if pair.chain_id != "solana" {
            pair.base_token.address.to_ascii_lowercase()
        } else {
            pair.base_token.address.clone()
        },
    })
}

#[must_use]
pub fn choose_symbol_pair(pairs: &[TokenPair], symbol: &str) -> Option<TokenPair> {
    let normalized = symbol.trim_start_matches('$').to_ascii_lowercase();
    let supported = pairs
        .iter()
        .filter(|pair| token_from_pair(pair).is_some())
        .cloned()
        .collect::<Vec<_>>();
    let exact = supported
        .iter()
        .filter(|pair| pair.base_token.symbol.to_ascii_lowercase() == normalized)
        .cloned()
        .collect::<Vec<_>>();
    choose_best_pair(if exact.is_empty() { &supported } else { &exact })
}

#[must_use]
pub fn token_image_url(pair: &TokenPair, pump: Option<&PumpMetadata>) -> Option<String> {
    if let Some(pump) = pump
        && !pump.image_uri.is_empty()
    {
        return Some(pump.image_uri.clone());
    }
    [
        &pair.info.header,
        &pair.info.image_url,
        &pair.info.open_graph,
    ]
    .into_iter()
    .find(|value| !value.is_empty())
    .cloned()
}

#[must_use]
pub fn token_socials(pair: &TokenPair, pump: Option<&PumpMetadata>) -> BTreeMap<String, String> {
    let mut socials = BTreeMap::new();
    if let Some(pump) = pump {
        for (label, value) in [
            ("X", &pump.twitter),
            ("TG", &pump.telegram),
            ("Web", &pump.website),
        ] {
            if !value.trim().is_empty() {
                socials.insert(label.to_owned(), value.trim().to_owned());
            }
        }
    }
    for website in &pair.info.websites {
        if website.url.is_empty() {
            continue;
        }
        if !socials.contains_key("Web") {
            socials.insert("Web".to_owned(), website.url.clone());
        } else if !website.label.is_empty() {
            socials
                .entry(website.label.chars().take(12).collect())
                .or_insert_with(|| website.url.clone());
        }
    }
    for social in &pair.info.socials {
        if social.url.is_empty() {
            continue;
        }
        let kind = social.kind.to_ascii_lowercase();
        let label = match kind.as_str() {
            "twitter" | "x" => "X".to_owned(),
            "telegram" => "TG".to_owned(),
            "tiktok" => "TikTok".to_owned(),
            "discord" => "Discord".to_owned(),
            "" => "Link".to_owned(),
            _ => kind.chars().take(12).collect(),
        };
        socials.entry(label).or_insert_with(|| social.url.clone());
    }
    socials
}

fn trim_decimal(mut value: String) -> String {
    if value.contains('.') {
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
    }
    value
}

fn trim_exact_suffix(mut value: String, suffix: &str) -> String {
    if value.ends_with(suffix) {
        value.truncate(value.len().saturating_sub(suffix.len()));
    }
    value
}

fn grouped_integer(value: f64) -> String {
    let rounded = format!("{value:.0}");
    let (sign, digits) = rounded
        .strip_prefix('-')
        .map_or(("", rounded.as_str()), |digits| ("-", digits));
    let mut output = String::new();
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    format!("{sign}{output}")
}

#[must_use]
pub fn format_money(value: f64, price: bool) -> String {
    if price {
        let decimals = if value >= 1.0 {
            3
        } else if value >= 0.01 {
            4
        } else {
            8
        };
        return format!("${}", trim_decimal(format!("{value:.decimals$}")));
    }
    let absolute = value.abs();
    if absolute >= 1_000_000_000.0 {
        return format!(
            "${}B",
            trim_exact_suffix(format!("{:.2}", value / 1e9), ".00")
        );
    }
    if absolute >= 1_000_000.0 {
        return format!(
            "${}M",
            trim_exact_suffix(format!("{:.2}", value / 1e6), ".00")
        );
    }
    if absolute >= 1_000.0 {
        return format!(
            "${}K",
            trim_exact_suffix(format!("{:.1}", value / 1e3), ".0")
        );
    }
    format!("${}", grouped_integer(value))
}

fn format_amount(value: f64) -> String {
    let absolute = value.abs();
    if absolute >= 1_000_000_000.0 {
        return format!("{}B", trim_decimal(format!("{:.1}", value / 1e9)));
    }
    if absolute >= 1_000_000.0 {
        return format!("{}M", trim_decimal(format!("{:.1}", value / 1e6)));
    }
    if absolute >= 1_000.0 {
        return format!("{}K", trim_decimal(format!("{:.1}", value / 1e3)));
    }
    grouped_integer(value)
}

fn format_percentage(value: f64) -> String {
    if value.abs() < 0.05 {
        return "+0%".to_owned();
    }
    let prefix = if value >= 0.0 { "+" } else { "" };
    format!("{prefix}{}%", trim_decimal(format!("{value:.1}")))
}

fn age_text(seconds: i64) -> String {
    let seconds = seconds.max(0);
    let days = seconds / 86_400;
    if days >= 365 {
        return format!("{}y", days / 365);
    }
    if days >= 1 {
        return format!("{days}d");
    }
    let hours = seconds / 3_600;
    if hours >= 1 {
        return format!("{hours}h");
    }
    format!("{}m", (seconds / 60).max(1))
}

fn age_from_milliseconds(value: &Value, now_unix: i64) -> Option<String> {
    let milliseconds = number(value);
    (milliseconds > 0.0).then(|| age_text(now_unix.saturating_sub((milliseconds / 1_000.0) as i64)))
}

fn age_from_candles(candles: &[Vec<f64>], now_unix: i64) -> Option<String> {
    candles
        .iter()
        .filter_map(|candle| candle.first().copied())
        .map(|timestamp| timestamp as i64)
        .min()
        .map(|timestamp| age_text(now_unix.saturating_sub(timestamp)))
}

fn pump_progress(pump: Option<&PumpMetadata>) -> Option<f64> {
    let pump = pump.filter(|pump| !pump.complete)?;
    let reserves = number(&pump.real_token_reserves);
    (reserves >= 0.0)
        .then(|| (100.0 - reserves * 100.0 / PUMP_INITIAL_REAL_TOKENS).clamp(0.0, 100.0))
}

fn compact_address(token: &TokenAddress) -> String {
    let prefix = token.address.chars().take(3).collect::<String>();
    let suffix = token
        .address
        .chars()
        .rev()
        .take(4)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{prefix}...{suffix}")
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn html_link(label: &str, url: &str) -> String {
    format!(
        "<a href=\"{}\">{}</a>",
        html_escape(url),
        html_escape(label)
    )
}

fn social_rows(socials: &BTreeMap<String, String>) -> Vec<String> {
    let mut links = Vec::new();
    let mut seen = HashSet::new();
    for label in ["X", "TG", "Web", "TikTok", "Discord"] {
        if let Some(url) = socials.get(label).filter(|url| !url.is_empty()) {
            links.push(html_link(label, url));
            seen.insert(label.to_owned());
        }
    }
    for (label, url) in socials {
        if !seen.contains(label) && !url.is_empty() {
            links.push(html_link(label, url));
        }
    }
    if links.is_empty() {
        Vec::new()
    } else {
        vec![
            String::new(),
            "🔗 <b>Socials</b>".to_owned(),
            format!("└ {}", links.join(" • ")),
        ]
    }
}

fn encoded_query(value: &str) -> String {
    form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

fn link_rows(signal: &TokenSignal, symbol: &str) -> [String; 2] {
    let token = &signal.token;
    let pair = &signal.pair;
    if !matches!(token.chain_id.as_str(), "ethereum" | "solana") {
        let dex = format!(
            "https://dexscreener.com/{}/{}",
            token.chain_id,
            if pair.pair_address.is_empty() {
                &token.address
            } else {
                &pair.pair_address
            }
        );
        let search = format!(
            "https://x.com/search?f=live&q={}&src=typed_query",
            encoded_query(&format!("(${symbol} OR {})", token.address))
        );
        return [
            [html_link("DS", &dex), html_link("Xs", &search)].join("•"),
            String::new(),
        ];
    }
    let explorer = if token.chain_id == "ethereum" {
        format!("https://etherscan.io/address/{}", token.address)
    } else {
        format!("https://solscan.io/token/{}", token.address)
    };
    let defined = format!("https://www.defined.fi/{}/{}", token.network, token.address);
    let pair_address = if pair.pair_address.is_empty() {
        &token.address
    } else {
        &pair.pair_address
    };
    let gecko = format!(
        "https://www.geckoterminal.com/{}/pools/{pair_address}",
        token.network
    );
    let mut search = format!("(${symbol} OR {}", token.address);
    if !pair.pair_address.is_empty() {
        search.push_str(&format!(" OR url:{}", pair.pair_address));
    }
    search.push(')');
    let x_search = format!(
        "https://x.com/search?f=live&q={}&src=typed_query",
        encoded_query(&search)
    );
    let dexscreener = if pair.url.is_empty() {
        "DS".to_owned()
    } else {
        html_link("DS", &pair.url)
    };
    let primary = if pair.pair_address.is_empty() && pair.url.starts_with("https://pump.fun/coin/")
    {
        [
            html_link("PF", &pair.url),
            html_link("EXP", &explorer),
            html_link("Xs", &x_search),
        ]
        .join("•")
    } else {
        [
            html_link("DEF", &defined),
            dexscreener,
            html_link("GT", &gecko),
            html_link("EXP", &explorer),
            html_link("Xs", &x_search),
        ]
        .join("•")
    };
    let trade = if token.chain_id == "ethereum" {
        [
            html_link(
                "GM",
                &format!("https://gmgn.ai/eth/token/{}", token.address),
            ),
            html_link(
                "OKX",
                &format!("https://web3.okx.com/token/ethereum/{}", token.address),
            ),
            html_link(
                "PHO",
                &format!(
                    "https://photon.tinyastro.io/en/r/@respondedor/{}",
                    token.address
                ),
            ),
        ]
        .join("•")
    } else {
        [
            html_link(
                "GM",
                &format!("https://gmgn.ai/sol/token/{}", token.address),
            ),
            html_link("AXI", &format!("https://axiom.trade/t/{}", token.address)),
            html_link(
                "TRO",
                &format!("https://t.me/menelaus_trojanbot?start={}", token.address),
            ),
            html_link(
                "BLO",
                &format!("https://t.me/BloomSolana_bot?start=ca_{}", token.address),
            ),
            html_link(
                "PHO",
                &format!(
                    "https://photon-sol.tinyastro.io/en/r/@respondedor/{}",
                    token.address
                ),
            ),
        ]
        .join("•")
    };
    [primary, trade]
}

fn ath(candles: &[Vec<f64>]) -> Option<(f64, i64)> {
    candles
        .iter()
        .filter(|candle| candle.len() >= 4)
        .map(|candle| (candle[2], candle[0] as i64))
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

#[must_use]
pub fn format_signal_caption(signal: &TokenSignal, now_unix: i64) -> String {
    let pair = &signal.pair;
    let name = if pair.base_token.name.is_empty() {
        "Token"
    } else {
        &pair.base_token.name
    };
    let symbol = if pair.base_token.symbol.is_empty() {
        "TOKEN".to_owned()
    } else {
        pair.base_token.symbol.to_ascii_uppercase()
    };
    let price = number(&pair.price_usd);
    let market_cap = {
        let market_cap = number(&pair.market_cap);
        if market_cap == 0.0 {
            number(&pair.fdv)
        } else {
            market_cap
        }
    };
    let liquidity = number(&pair.liquidity.usd);
    let displayed_liquidity = if signal.token.chain_id == "solana" {
        liquidity / 2.0
    } else {
        liquidity
    };
    let buys = number(&pair.txns.h1.buys) as i64;
    let sells = number(&pair.txns.h1.sells) as i64;
    let pump_ath = signal
        .pump
        .as_ref()
        .map_or(0.0, |pump| number(&pump.ath_market_cap));
    let (ath_value, ath_timestamp) = if pump_ath > 0.0 {
        let timestamp = signal
            .pump
            .as_ref()
            .map(|pump| number(&pump.ath_market_cap_timestamp) as i64 / 1_000)
            .filter(|timestamp| *timestamp > 0);
        (pump_ath, timestamp)
    } else if let Some((ath_price, timestamp)) = ath(&signal.candles) {
        let value = if price > 0.0 && market_cap > 0.0 {
            market_cap * ath_price / price
        } else {
            ath_price
        };
        (value, Some(timestamp))
    } else {
        (0.0, None)
    };
    let ath_line = if ath_value > 0.0 {
        let drawdown_base = if market_cap > 0.0 { market_cap } else { price };
        let drawdown = if drawdown_base == 0.0 {
            0.0
        } else {
            (drawdown_base - ath_value) / ath_value * 100.0
        };
        let age = ath_timestamp.map_or_else(String::new, |timestamp| {
            format!(
                " / {}d",
                (now_unix.saturating_sub(timestamp) / 86_400).max(1)
            )
        });
        format!(
            "{} ({}{age})",
            format_money(ath_value, false),
            format_percentage(drawdown)
        )
    } else {
        "?".to_owned()
    };
    let age = age_from_milliseconds(&pair.pair_created_at, now_unix)
        .or_else(|| {
            signal
                .pump
                .as_ref()
                .and_then(|pump| age_from_milliseconds(&pump.created_timestamp, now_unix))
        })
        .or_else(|| age_from_candles(&signal.candles, now_unix))
        .unwrap_or_else(|| "?".to_owned());
    let chain = pump_progress(signal.pump.as_ref()).map_or_else(
        || format!("#{}", signal.token.tag),
        |progress| format!("#{} (Pump @ {progress:.0}%)", signal.token.tag),
    );
    let mut stats = vec![
        format!(
            "├ USD   <b>{}</b> ({})",
            format_money(price, true),
            format_percentage(number(&pair.price_change.h24))
        ),
        format!("├ MC    <b>{}</b>", format_money(market_cap, false)),
        format!(
            "├ Vol   <b>{}</b>",
            format_money(number(&pair.volume.h24), false)
        ),
        format!(
            "├ LP    <b>{}</b>",
            format_money(displayed_liquidity, false)
        ),
    ];
    if let Some(supply) = signal.supply.filter(|supply| *supply > 0.0) {
        let formatted = format_amount(supply);
        stats.push(format!("├ Sup   <b>{formatted}/{formatted}</b>"));
    }
    stats.push(format!(
        "├ 1H    <b>{}</b> 🟩 {buys} 🟥 {sells}",
        format_percentage(number(&pair.price_change.h1))
    ));
    stats.push(format!("└ ATH   <b>{ath_line}</b>"));

    let mut rows = vec![
        format!("💊 <b>{}</b> (${symbol})", html_escape(name)),
        format!(
            "├ <code>{}</code>",
            html_escape(&compact_address(&signal.token))
        ),
        format!("└ {chain} | <i>{age}</i>"),
        String::new(),
        "📊 <b>Stats</b>".to_owned(),
    ];
    rows.extend(stats);
    rows.extend(social_rows(&signal.socials));
    rows.push(String::new());
    rows.extend(link_rows(signal, &symbol));
    rows.join("\n")
}

#[must_use]
pub fn has_usable_chart(signal: &TokenSignal) -> bool {
    let candles = signal
        .candles
        .iter()
        .filter(|candle| candle.len() >= 5)
        .collect::<Vec<_>>();
    if candles.len() < 5 {
        return false;
    }
    let high = candles
        .iter()
        .map(|candle| candle[2])
        .fold(0.0_f64, f64::max);
    let low = candles
        .iter()
        .map(|candle| candle[3])
        .fold(f64::INFINITY, f64::min);
    high > 0.0 && (high - low).abs() > f64::EPSILON
}

#[must_use]
pub fn build_signal_keyboard(
    signal_id: &str,
    token: &TokenAddress,
    pair: &TokenPair,
) -> InlineKeyboardMarkup {
    let dexscreener = if pair.url.is_empty() {
        format!("https://www.defined.fi/{}/{}", token.network, token.address)
    } else {
        pair.url.clone()
    };
    InlineKeyboardMarkup {
        inline_keyboard: vec![vec![
            InlineKeyboardButton {
                text: "🗑".to_owned(),
                url: None,
                callback_data: Some(format!("sig:del:{signal_id}")),
                copy_text: None,
            },
            InlineKeyboardButton {
                text: "🔄".to_owned(),
                url: None,
                callback_data: Some(format!("sig:ref:{signal_id}")),
                copy_text: None,
            },
            InlineKeyboardButton {
                text: "📋".to_owned(),
                url: None,
                callback_data: None,
                copy_text: Some(CopyTextButton {
                    text: token.address.clone(),
                }),
            },
            InlineKeyboardButton {
                text: "DS".to_owned(),
                url: Some(dexscreener),
                callback_data: None,
                copy_text: None,
            },
        ]],
    }
}

#[must_use]
pub fn callback_text(key: &str, locale: Locale) -> &'static str {
    match (key, locale) {
        ("expired", Locale::Es) => "card vencida",
        ("expired", Locale::En) => "card expired",
        ("owner_only", Locale::Es) => "solo quien pidió la tarjeta o un admin puede hacer eso",
        ("owner_only", Locale::En) => "only the requester or an admin can do that",
        ("deleted", Locale::Es) => "tarjeta borrada",
        ("deleted", Locale::En) => "card deleted",
        ("cooldown", Locale::Es) => "Podés actualizar cada 15s",
        ("cooldown", Locale::En) => "You can refresh every 15s",
        ("no_data", Locale::Es) => "no encontré datos nuevos",
        ("no_data", Locale::En) => "I could not find new data",
        ("refresh_failed", Locale::Es) => "no pude actualizar la tarjeta",
        ("refresh_failed", Locale::En) => "I could not refresh the card",
        ("refreshed", Locale::Es) => "tarjeta actualizada",
        ("refreshed", Locale::En) => "card refreshed",
        _ => "card expired",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::{
        PairInfo, PairLiquidity, PairPriceChange, PairSocial, PairToken, PairTransactionWindows,
        PairTransactions, PairVolume, PairWebsite, PumpMetadata, SignalQuery, SignalState,
        TokenAddress, TokenPair, TokenSignal, age_text, build_signal_keyboard, callback_text,
        choose_best_pair, choose_symbol_pair, detect_signal_query, format_money,
        format_signal_caption, has_usable_chart, pair_rank, signal_state_key, stable_signal_id,
        token_from_pair, token_image_url, token_socials,
    };
    use crate::locale::Locale;

    const SOL_MINT: &str = "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump";
    const EVM: &str = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48";

    fn pair() -> TokenPair {
        TokenPair {
            chain_id: "solana".to_owned(),
            url: "https://dexscreener.com/solana/pair".to_owned(),
            pair_address: "pair1".to_owned(),
            base_token: PairToken {
                address: SOL_MINT.to_owned(),
                name: "Tung Tung".to_owned(),
                symbol: "TRIPLET".to_owned(),
            },
            price_usd: json!("0.0106"),
            price_change: PairPriceChange {
                h1: json!(5.1),
                h24: json!(2.3),
            },
            market_cap: json!(10_580_000),
            volume: PairVolume {
                h24: json!(851_700),
            },
            liquidity: PairLiquidity {
                usd: json!(246_700),
            },
            txns: PairTransactionWindows {
                h1: PairTransactions {
                    buys: json!(122),
                    sells: json!(148),
                },
            },
            pair_created_at: json!(1_710_000_000_000_i64),
            ..TokenPair::default()
        }
    }

    fn candles() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 1.0, 2.0, 0.8, 1.5, 1_000.0],
            vec![2.0, 1.5, 2.5, 1.2, 1.3, 1_000.0],
            vec![3.0, 1.3, 1.8, 1.0, 1.7, 1_000.0],
        ]
    }

    fn signal() -> TokenSignal {
        TokenSignal {
            token: TokenAddress {
                chain_id: "solana".to_owned(),
                network: "solana".to_owned(),
                tag: "SOL".to_owned(),
                address: SOL_MINT.to_owned(),
            },
            pair: pair(),
            candles: candles(),
            supply: Some(999_900_000.0),
            token_image_url: None,
            socials: BTreeMap::new(),
            pump: None,
        }
    }

    #[test]
    fn detects_only_complete_solana_evm_and_cashtag_queries() {
        assert!(matches!(
            detect_signal_query(SOL_MINT),
            Some(SignalQuery::Address(TokenAddress { chain_id, .. })) if chain_id == "solana"
        ));
        assert!(matches!(
            detect_signal_query(EVM),
            Some(SignalQuery::Address(TokenAddress { chain_id, address, .. }))
                if chain_id == "ethereum" && address == EVM
        ));
        assert_eq!(
            detect_signal_query("$GLORP"),
            Some(SignalQuery::Symbol("glorp".to_owned()))
        );
        assert_eq!(detect_signal_query(&format!("buy {SOL_MINT}")), None);
        assert_eq!(detect_signal_query("buy $glorp"), None);
    }

    #[test]
    fn pair_selection_prefers_liquidity_and_exact_supported_symbol() {
        let mut low = pair();
        low.liquidity.usd = json!(10);
        low.volume.h24 = json!(999);
        let mut high = pair();
        high.liquidity.usd = json!(100);
        high.volume.h24 = json!(1);
        assert_eq!(choose_best_pair(&[low, high.clone()]), Some(high));

        let mut unrelated = pair();
        unrelated.base_token.symbol = "NOPE".to_owned();
        unrelated.liquidity.usd = json!(1_000_000);
        let mut exact = pair();
        exact.base_token.symbol = "glorp".to_owned();
        exact.liquidity.usd = json!(1_000);
        assert_eq!(
            choose_symbol_pair(&[unrelated, exact.clone()], "glorp"),
            Some(exact)
        );
    }

    #[test]
    fn caption_and_keyboard_preserve_the_observable_card_contract() {
        let signal = signal();
        let caption = format_signal_caption(&signal, 1_720_000_000);
        assert!(caption.contains("Tung Tung"));
        assert!(caption.contains("$TRIPLET"));
        assert!(caption.contains("J8P...pump"));
        assert!(!caption.contains(&format!("<code>{SOL_MINT}</code>")));
        assert!(caption.contains("├ LP    <b>$123.3K</b>"));
        assert!(caption.contains("├ Sup   <b>999.9M/999.9M</b>"));
        assert!(caption.contains("ATH   <b>$2.50B"));
        let keyboard = build_signal_keyboard("abc", &signal.token, &signal.pair);
        assert_eq!(
            keyboard.inline_keyboard[0][2]
                .copy_text
                .as_ref()
                .map(|copy| copy.text.as_str()),
            Some(SOL_MINT)
        );
    }

    #[test]
    fn pump_metadata_adds_progress_socials_and_published_ath() {
        let mut signal = signal();
        signal.socials = BTreeMap::from([
            ("X".to_owned(), "https://x.com/token".to_owned()),
            ("TG".to_owned(), "https://t.me/token".to_owned()),
            ("Web".to_owned(), "https://token.test".to_owned()),
        ]);
        signal.pump = Some(PumpMetadata {
            real_token_reserves: json!(489_800_000_000_000_i64),
            ath_market_cap: json!(51_600),
            ath_market_cap_timestamp: json!(1_710_100_000_000_i64),
            ..PumpMetadata::default()
        });
        let caption = format_signal_caption(&signal, 1_720_000_000);
        assert!(caption.contains("#SOL (Pump @ 38%)"));
        assert!(caption.contains("🔗 <b>Socials</b>"));
        assert!(caption.contains("ATH   <b>$51.6K"));
    }

    #[test]
    fn usable_chart_requires_five_nonflat_candles() {
        let mut signal = signal();
        signal.candles.clear();
        assert!(!has_usable_chart(&signal));
        signal.candles = vec![vec![1.0, 1.0, 1.0, 1.0, 1.0]; 5];
        assert!(!has_usable_chart(&signal));
        signal.candles = [candles(), candles()].concat();
        assert!(has_usable_chart(&signal));
    }

    #[test]
    fn signal_identity_state_and_pair_boundaries_are_stable() {
        let state = SignalState {
            chat_id: "-1001".to_owned(),
            message_id: 20,
            source_message_id: 19,
            requester_id: "42".to_owned(),
            chain_id: "ethereum".to_owned(),
            network: "eth".to_owned(),
            tag: "ETH".to_owned(),
            address: EVM.to_ascii_uppercase(),
            last_refresh_at: Some(100),
        };
        assert_eq!(state.token().chain_id, "ethereum");
        assert_eq!(signal_state_key("abc123"), "token_signal:abc123");
        let first = stable_signal_id(-1001, 20, 42, 1_720_000_000);
        assert_eq!(first.len(), 12);
        assert_eq!(first, stable_signal_id(-1001, 20, 42, 1_720_000_000));
        assert_ne!(first, stable_signal_id(-1001, 21, 42, 1_720_000_000));

        assert_eq!(choose_best_pair(&[]), None);
        let mut liquid = pair();
        liquid.liquidity.usd = json!("100");
        liquid.volume.h24 = json!(10);
        assert_eq!(pair_rank(&liquid), (100.0, 10.0));
        let mut volume_winner = liquid.clone();
        volume_winner.volume.h24 = json!(11);
        assert_eq!(
            choose_best_pair(&[liquid, volume_winner.clone()]),
            Some(volume_winner)
        );

        let mut ethereum = pair();
        ethereum.chain_id = "ethereum".to_owned();
        ethereum.base_token.address = EVM.to_ascii_uppercase();
        assert_eq!(
            token_from_pair(&ethereum),
            Some(TokenAddress {
                chain_id: "ethereum".to_owned(),
                network: "eth".to_owned(),
                tag: "ETH".to_owned(),
                address: EVM.to_owned(),
            })
        );
        ethereum.base_token.address.clear();
        assert_eq!(token_from_pair(&ethereum), None);
        ethereum.chain_id = "unsupported".to_owned();
        ethereum.base_token.address = "value".to_owned();
        assert_eq!(token_from_pair(&ethereum), None);
        assert_eq!(choose_symbol_pair(&[ethereum], "$unknown"), None);
    }

    #[test]
    fn image_and_social_selection_cover_provider_precedence_and_labels() {
        let mut pair = pair();
        pair.info = PairInfo {
            header: "https://img.test/header.png".to_owned(),
            image_url: "https://img.test/image.png".to_owned(),
            open_graph: "https://img.test/og.png".to_owned(),
            websites: vec![
                PairWebsite {
                    label: String::new(),
                    url: String::new(),
                },
                PairWebsite {
                    label: "Official website with long label".to_owned(),
                    url: "https://pair.test".to_owned(),
                },
                PairWebsite {
                    label: "Docs and more".to_owned(),
                    url: "https://docs.test".to_owned(),
                },
            ],
            socials: vec![
                PairSocial {
                    kind: "twitter".to_owned(),
                    url: "https://x.com/pair".to_owned(),
                },
                PairSocial {
                    kind: "telegram".to_owned(),
                    url: "https://t.me/pair".to_owned(),
                },
                PairSocial {
                    kind: "tiktok".to_owned(),
                    url: "https://tiktok.test/pair".to_owned(),
                },
                PairSocial {
                    kind: "discord".to_owned(),
                    url: "https://discord.test/pair".to_owned(),
                },
                PairSocial {
                    kind: String::new(),
                    url: "https://link.test".to_owned(),
                },
                PairSocial {
                    kind: "VeryLongCommunityName".to_owned(),
                    url: "https://other.test".to_owned(),
                },
                PairSocial {
                    kind: "ignored".to_owned(),
                    url: String::new(),
                },
            ],
        };
        assert_eq!(
            token_image_url(&pair, None).as_deref(),
            Some("https://img.test/header.png")
        );
        pair.info.header.clear();
        assert_eq!(
            token_image_url(&pair, None).as_deref(),
            Some("https://img.test/image.png")
        );
        pair.info.image_url.clear();
        assert_eq!(
            token_image_url(&pair, None).as_deref(),
            Some("https://img.test/og.png")
        );

        let pump = PumpMetadata {
            image_uri: "https://pump.test/image.png".to_owned(),
            twitter: " https://x.com/pump ".to_owned(),
            telegram: "https://t.me/pump".to_owned(),
            website: "https://pump.test".to_owned(),
            ..PumpMetadata::default()
        };
        assert_eq!(
            token_image_url(&pair, Some(&pump)).as_deref(),
            Some("https://pump.test/image.png")
        );
        let socials = token_socials(&pair, Some(&pump));
        assert_eq!(
            socials.get("X").map(String::as_str),
            Some("https://x.com/pump")
        );
        assert_eq!(
            socials.get("TG").map(String::as_str),
            Some("https://t.me/pump")
        );
        assert_eq!(
            socials.get("Web").map(String::as_str),
            Some("https://pump.test")
        );
        assert!(socials.contains_key("Docs and mor"));
        assert!(socials.contains_key("TikTok"));
        assert!(socials.contains_key("Discord"));
        assert!(socials.contains_key("Link"));
        assert!(socials.contains_key("verylongcomm"));
    }

    #[test]
    fn caption_formats_empty_evm_data_and_all_numeric_bands_safely() {
        assert_eq!(format_money(1.2345, true), "$1.234");
        assert_eq!(format_money(0.125, true), "$0.125");
        assert_eq!(format_money(0.000_012_34, true), "$0.00001234");
        assert_eq!(format_money(2_000_000_000.0, false), "$2B");
        assert_eq!(format_money(2_500_000.0, false), "$2.50M");
        assert_eq!(format_money(-1_500.0, false), "$-1.5K");
        assert_eq!(format_money(-999.0, false), "$-999");
        assert_eq!(age_text(400 * 86_400), "1y");
        assert_eq!(age_text(2 * 86_400), "2d");
        assert_eq!(age_text(7_200), "2h");
        assert_eq!(age_text(-1), "1m");

        let token = TokenAddress {
            chain_id: "ethereum".to_owned(),
            network: "eth".to_owned(),
            tag: "ETH".to_owned(),
            address: EVM.to_owned(),
        };
        let pair = TokenPair {
            chain_id: "ethereum".to_owned(),
            base_token: PairToken {
                address: EVM.to_owned(),
                ..PairToken::default()
            },
            price_usd: json!(2),
            market_cap: json!(0),
            fdv: json!(900),
            liquidity: PairLiquidity { usd: json!(1_000) },
            price_change: PairPriceChange {
                h1: json!(0.01),
                h24: json!(-3.25),
            },
            pair_created_at: json!(0),
            ..TokenPair::default()
        };
        let signal = TokenSignal {
            token,
            pair,
            candles: Vec::new(),
            supply: Some(0.0),
            token_image_url: None,
            socials: BTreeMap::from([(
                "Custom<&".to_owned(),
                "https://social.test/?a=1&b=2".to_owned(),
            )]),
            pump: Some(PumpMetadata {
                complete: true,
                ..PumpMetadata::default()
            }),
        };
        let caption = format_signal_caption(&signal, 1_720_000_000);
        assert!(caption.contains("💊 <b>Token</b> ($TOKEN)"));
        assert!(caption.contains("#ETH | <i>?</i>"));
        assert!(caption.contains("MC    <b>$900</b>"));
        assert!(caption.contains("ATH   <b>?</b>"));
        assert!(caption.contains("etherscan.io"));
        assert!(caption.contains("gmgn.ai/eth"));
        assert!(caption.contains("Custom&lt;&amp;"));
        let keyboard = build_signal_keyboard("fallback", &signal.token, &signal.pair);
        assert_eq!(
            keyboard.inline_keyboard[0][3].url.as_deref(),
            Some("https://www.defined.fi/eth/0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")
        );
    }

    #[test]
    fn callback_copy_covers_every_localized_result() {
        for (key, spanish, english) in [
            ("expired", "card vencida", "card expired"),
            (
                "owner_only",
                "solo quien pidió la tarjeta o un admin puede hacer eso",
                "only the requester or an admin can do that",
            ),
            ("deleted", "tarjeta borrada", "card deleted"),
            (
                "cooldown",
                "Podés actualizar cada 15s",
                "You can refresh every 15s",
            ),
            (
                "no_data",
                "no encontré datos nuevos",
                "I could not find new data",
            ),
            (
                "refresh_failed",
                "no pude actualizar la tarjeta",
                "I could not refresh the card",
            ),
            ("refreshed", "tarjeta actualizada", "card refreshed"),
        ] {
            assert_eq!(callback_text(key, Locale::Es), spanish);
            assert_eq!(callback_text(key, Locale::En), english);
        }
        assert_eq!(callback_text("unknown", Locale::Es), "card expired");
    }
}
