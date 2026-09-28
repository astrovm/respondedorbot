//! DexScreener/GeckoTerminal token-card adapter and PNG renderer.

use std::io::Cursor;
use std::sync::{LazyLock, OnceLock};
use std::time::Duration;

use ab_glyph::FontArc;
use bot_core::token_signals::{
    PumpMetadata, SIGNAL_STATE_TTL_SECONDS, SignalQuery, SignalState, TokenAddress, TokenPair,
    TokenSignal, TokenSignalCandidates, format_money, is_usable_chart_candle, normalize_token_name,
    pair_rank, signal_state_key, token_from_pair, token_image_url, token_socials,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};

const HTTP_TIMEOUT_SECONDS: u64 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryResponse {
    pub status_code: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

pub trait TokenSignalTransport {
    fn get_json(&self, url: &str, query: &[(&str, String)]) -> Result<JsonResponse, String>;

    fn post_json(&self, url: &str, body: &Value) -> Result<JsonResponse, String>;

    fn get_binary(&self, url: &str) -> Result<BinaryResponse, String>;
}

pub struct ReqwestTokenSignalTransport {
    client: Client,
}

impl ReqwestTokenSignalTransport {
    pub fn new() -> Result<Self, String> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder()
                .timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
                .build()
        })
        .map(|client| Self { client })
        .map_err(failure("could not build token-signal HTTP client"))
    }
}

impl TokenSignalTransport for ReqwestTokenSignalTransport {
    fn get_json(&self, url: &str, query: &[(&str, String)]) -> Result<JsonResponse, String> {
        let response = self
            .client
            .get(url)
            .query(query)
            .send()
            .map_err(failure("token-signal GET failed"))?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| JsonResponse { status_code, body })
            .map_err(failure("token-signal response read failed"))
    }

    fn post_json(&self, url: &str, body: &Value) -> Result<JsonResponse, String> {
        let response = self
            .client
            .post(url)
            .json(body)
            .send()
            .map_err(failure("token-signal POST failed"))?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| JsonResponse { status_code, body })
            .map_err(failure("token-signal response read failed"))
    }

    fn get_binary(&self, url: &str) -> Result<BinaryResponse, String> {
        let response = self
            .client
            .get(url)
            .send()
            .map_err(failure("token image GET failed"))?;
        let status_code = response.status().as_u16();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();
        response
            .bytes()
            .map(|body| BinaryResponse {
                status_code,
                content_type,
                body: body.to_vec(),
            })
            .map_err(failure("token image response read failed"))
    }
}

/// Prefix an error with the operation that failed.
fn failure<E: std::fmt::Display>(context: &'static str) -> impl FnOnce(E) -> String {
    move |error| format!("{context}: {error}")
}

pub trait TokenSignalCache {
    type Error: std::fmt::Display;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error>;

    /// A zero TTL stores the value without expiration. Positive TTLs expire normally.
    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error>;
}

impl TokenSignalCache for RedisJsonCache {
    type Error = RedisJsonCacheError;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        RedisJsonCache::get(self, key)
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
        RedisJsonCache::set(self, key, value, (ttl_seconds != 0).then_some(ttl_seconds))
            .map(|_stored| ())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenSignalLoad {
    pub signal: Option<TokenSignal>,
    pub diagnostics: Vec<String>,
}

pub struct TokenSignalAdapter<Transport, Cache> {
    transport: Transport,
    cache: Cache,
}

/// Activity is a better discriminator for a bare ticker than pool depth:
/// namesake pools often advertise a large LP while receiving little real
/// trading. Liquidity remains the tie-breaker for pools with equal volume.
fn compare_symbol_pairs(left: &TokenPair, right: &TokenPair) -> std::cmp::Ordering {
    let (left_liquidity, left_volume) = pair_rank(left);
    let (right_liquidity, right_volume) = pair_rank(right);
    right_volume
        .total_cmp(&left_volume)
        .then_with(|| right_liquidity.total_cmp(&left_liquidity))
}

fn same_token_identity(left: &TokenAddress, right: &TokenAddress) -> bool {
    left.chain_id.eq_ignore_ascii_case(&right.chain_id)
        && left.network.eq_ignore_ascii_case(&right.network)
        && if left.address.starts_with("0x") && right.address.starts_with("0x") {
            left.address.eq_ignore_ascii_case(&right.address)
        } else {
            left.address == right.address
        }
}

fn identified_pair(pair: &TokenPair) -> Option<(TokenAddress, TokenPair)> {
    token_from_pair(pair).map(|token| (token, pair.clone()))
}

/// Keep the first (best-ranked) pair of every distinct token identity from
/// pairs already sorted by [`compare_symbol_pairs`].
fn unique_identities(pairs: Vec<TokenPair>) -> Vec<(TokenAddress, TokenPair)> {
    let mut unique = Vec::<(TokenAddress, TokenPair)>::new();
    for (token, pair) in pairs.iter().filter_map(identified_pair) {
        if !unique
            .iter()
            .any(|(known, _)| same_token_identity(known, &token))
        {
            unique.push((token, pair));
        }
    }
    unique
}

fn preview_signal(token: TokenAddress, pair: TokenPair) -> TokenSignal {
    let token_image_url = token_image_url(&pair, None);
    let socials = token_socials(&pair, None);
    TokenSignal {
        token,
        pair,
        candles: Vec::new(),
        supply: None,
        token_image_url,
        socials,
        pump: None,
    }
}

impl<Transport, Cache> TokenSignalAdapter<Transport, Cache> {
    #[must_use]
    pub fn new(transport: Transport, cache: Cache) -> Self {
        Self { transport, cache }
    }
}

impl<Transport, Cache> TokenSignalAdapter<Transport, Cache>
where
    Transport: TokenSignalTransport,
    Cache: TokenSignalCache,
{
    fn cached_json(
        &mut self,
        key: &str,
        ttl_seconds: i64,
        label: &str,
        fetch: &dyn Fn(&Transport) -> Result<JsonResponse, String>,
        extract: &dyn Fn(Value) -> Option<Value>,
        diagnostics: &mut Vec<String>,
    ) -> Option<Value> {
        match self.cache.get(key) {
            Ok(Some(value)) => match serde_json::from_str(&value) {
                Ok(value) => return Some(value),
                Err(error) => diagnostics.push(format!("invalid {label} cache {key}: {error}")),
            },
            Ok(None) => {}
            Err(error) => diagnostics.push(format!("could not read {label} cache {key}: {error}")),
        }
        let response = match fetch(&self.transport) {
            Ok(response) if (200..300).contains(&response.status_code) => response,
            Ok(response) => {
                diagnostics.push(format!("{label} HTTP {}", response.status_code));
                return None;
            }
            Err(error) => {
                diagnostics.push(format!("{label}: {error}"));
                return None;
            }
        };
        let provider_value = match serde_json::from_str::<Value>(&response.body) {
            Ok(value) => value,
            Err(error) => {
                diagnostics.push(format!("invalid {label} response: {error}"));
                return None;
            }
        };
        let Some(value) = extract(provider_value) else {
            diagnostics.push(format!(
                "{label} response did not contain the expected value"
            ));
            return None;
        };
        // A JSON value always encodes; only the cache write can fail.
        if let Err(error) = self.cache.set(key, &value.to_string(), ttl_seconds) {
            diagnostics.push(format!("could not write {label} cache {key}: {error}"));
        }
        Some(value)
    }

    fn pairs(&mut self, token: &TokenAddress, diagnostics: &mut Vec<String>) -> Vec<TokenPair> {
        let key = format!("token_signal:pairs:{}:{}", token.chain_id, token.address);
        let url = format!(
            "https://api.dexscreener.com/token-pairs/v1/{}/{}",
            token.chain_id, token.address
        );
        self.cached_json(
            &key,
            30,
            "DexScreener pairs",
            &|transport| transport.get_json(&url, &[]),
            &Some,
            diagnostics,
        )
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
    }

    fn search_pairs(&mut self, symbol: &str, diagnostics: &mut Vec<String>) -> Vec<TokenPair> {
        let normalized = symbol.trim_start_matches('$').to_ascii_lowercase();
        let key = format!("token_signal:search:{normalized}");
        self.cached_json(
            &key,
            30,
            "DexScreener search",
            &|transport| {
                transport.get_json(
                    "https://api.dexscreener.com/latest/dex/search",
                    &[("q", normalized.clone())],
                )
            },
            &|value| value.get("pairs").cloned(),
            diagnostics,
        )
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
    }

    fn candles(
        &mut self,
        token: &TokenAddress,
        pair_address: &str,
        diagnostics: &mut Vec<String>,
    ) -> Vec<Vec<f64>> {
        let key = format!("token_signal:ohlcv:{}:{pair_address}:hour", token.network);
        let url = format!(
            "https://api.geckoterminal.com/api/v2/networks/{}/pools/{pair_address}/ohlcv/hour",
            token.network
        );
        let raw = self.cached_json(
            &key,
            60,
            "GeckoTerminal OHLCV",
            &|transport| {
                transport.get_json(
                    &url,
                    &[
                        ("aggregate", "4".to_owned()),
                        ("limit", "60".to_owned()),
                        ("currency", "usd".to_owned()),
                    ],
                )
            },
            &|value| value.pointer("/data/attributes/ohlcv_list").cloned(),
            diagnostics,
        );
        raw.and_then(|candles| candles.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|candle| {
                candle.as_array().map(|values| {
                    values
                        .iter()
                        .filter_map(flexible_number)
                        .collect::<Vec<_>>()
                })
            })
            .filter(|candle| !candle.is_empty())
            .collect()
    }

    fn pump(
        &mut self,
        token: &TokenAddress,
        diagnostics: &mut Vec<String>,
    ) -> Option<PumpMetadata> {
        self.pump_json(token, diagnostics)
            .and_then(|value| serde_json::from_value(value).ok())
    }

    fn pump_json(&mut self, token: &TokenAddress, diagnostics: &mut Vec<String>) -> Option<Value> {
        if token.chain_id != "solana" || !token.address.ends_with("pump") {
            return None;
        }
        let key = format!("token_signal:pump:{}", token.address);
        let url = format!("https://frontend-api-v3.pump.fun/coins/{}", token.address);
        self.cached_json(
            &key,
            60,
            "pump.fun metadata",
            &|transport| transport.get_json(&url, &[]),
            &Some,
            diagnostics,
        )
    }

    fn pump_signal(&mut self, value: Value, diagnostics: &mut Vec<String>) -> Option<TokenSignal> {
        let mint = value.get("mint")?.as_str()?;
        let SignalQuery::Address(token) = bot_core::token_signals::detect_signal_query(mint)?
        else {
            return None;
        };
        if token.chain_id != "solana" || value.get("symbol")?.as_str()?.trim().is_empty() {
            return None;
        }
        let pump: PumpMetadata = serde_json::from_value(value.clone()).ok()?;
        let decimals = value
            .get("base_decimals")
            .and_then(Value::as_u64)
            .unwrap_or(6);
        let scale = 10_f64.powi(i32::try_from(decimals).ok()?.min(18));
        let supply = self
            .supply(&token, diagnostics)
            .or_else(|| flexible_number(&pump.total_supply).map(|supply| supply / scale))
            .filter(|supply| *supply > 0.0);
        let market_cap = value
            .get("usd_market_cap")
            .or_else(|| value.get("market_cap_usd"))
            .and_then(flexible_number);
        let pair: TokenPair = serde_json::from_value(json!({
            "chainId": "solana",
            "url": format!("https://pump.fun/coin/{mint}"),
            "baseToken": {"address": mint, "name": value.get("name"), "symbol": value.get("symbol")},
            "priceUsd": market_cap.zip(supply).map(|(cap, supply)| cap / supply),
            "marketCap": market_cap,
            "pairCreatedAt": pump.created_timestamp,
        })).ok()?;
        Some(TokenSignal {
            token,
            token_image_url: token_image_url(&pair, Some(&pump)),
            socials: token_socials(&pair, Some(&pump)),
            pair,
            candles: Vec::new(),
            supply,
            pump: Some(pump),
        })
    }

    fn search_pump_signal(
        &mut self,
        symbol: &str,
        diagnostics: &mut Vec<String>,
    ) -> Option<TokenSignal> {
        let normalized = symbol.trim_start_matches('$').to_ascii_lowercase();
        let key = format!("token_signal:pump_search:{normalized}");
        let value = self.cached_json(
            &key,
            30,
            "pump.fun search",
            &|transport| {
                transport.get_json(
                    "https://frontend-api-v3.pump.fun/coins/search-unrestricted",
                    &[
                        ("searchTerm", normalized.clone()),
                        ("limit", "20".to_owned()),
                        ("offset", "0".to_owned()),
                        ("includeNsfw", "true".to_owned()),
                        ("sort", "market_cap".to_owned()),
                        ("order", "DESC".to_owned()),
                    ],
                )
            },
            &Some,
            diagnostics,
        )?;
        let mut matches = value
            .as_array()?
            .iter()
            .filter(|coin| {
                coin.get("symbol")
                    .and_then(Value::as_str)
                    .is_some_and(|symbol| symbol.trim().eq_ignore_ascii_case(&normalized))
            })
            .collect::<Vec<_>>();
        let cap = |coin: &Value| {
            coin.get("usd_market_cap")
                .or_else(|| coin.get("market_cap_usd"))
                .and_then(flexible_number)
                .unwrap_or(0.0)
        };
        matches.sort_by(|left, right| cap(right).total_cmp(&cap(left)));
        for coin in matches {
            if let Some(signal) = self.pump_signal(coin.clone(), diagnostics) {
                return Some(signal);
            }
        }
        None
    }

    fn supply(&mut self, token: &TokenAddress, diagnostics: &mut Vec<String>) -> Option<f64> {
        if token.chain_id != "solana" {
            return None;
        }
        let key = format!("token_signal:supply:{}", token.address);
        let body = json!({
            "jsonrpc":"2.0",
            "id":1,
            "method":"getTokenSupply",
            "params":[token.address],
        });
        self.cached_json(
            &key,
            300,
            "Solana token supply",
            &|transport| transport.post_json("https://api.mainnet-beta.solana.com", &body),
            &|value| {
                value
                    .pointer("/result/value/uiAmountString")
                    .or_else(|| value.pointer("/result/value/uiAmount"))
                    .and_then(flexible_number)
                    .map(|supply| json!(supply))
            },
            diagnostics,
        )
        .as_ref()
        .and_then(flexible_number)
        .filter(|supply| *supply >= 0.0)
    }

    fn enrich(
        &mut self,
        token: TokenAddress,
        pair: TokenPair,
        candles: Vec<Vec<f64>>,
        diagnostics: &mut Vec<String>,
    ) -> TokenSignal {
        let pump = self.pump(&token, diagnostics);
        let supply = self.supply(&token, diagnostics).or_else(|| {
            pump.as_ref()
                .and_then(|pump| flexible_number(&pump.total_supply))
                .map(|supply| supply / 1_000_000.0)
        });
        let token_image_url = token_image_url(&pair, pump.as_ref());
        let socials = token_socials(&pair, pump.as_ref());
        TokenSignal {
            token,
            pair,
            candles,
            supply,
            token_image_url,
            socials,
            pump,
        }
    }

    pub fn load_query(&mut self, query: &SignalQuery) -> TokenSignalLoad {
        match query {
            SignalQuery::Address(token) if token.chain_id == "ethereum" => {
                let mut diagnostics = Vec::new();
                let key = format!("token_signal:address:{}", token.address);
                let url = format!(
                    "https://api.dexscreener.com/latest/dex/tokens/{}",
                    token.address
                );
                let pairs: Vec<TokenPair> = self
                    .cached_json(
                        &key,
                        30,
                        "DexScreener address",
                        &|transport| transport.get_json(&url, &[]),
                        &|value| value.get("pairs").cloned(),
                        &mut diagnostics,
                    )
                    .and_then(|value| serde_json::from_value(value).ok())
                    .unwrap_or_default();
                let pairs = pairs
                    .iter()
                    .filter(|pair| pair.base_token.address.eq_ignore_ascii_case(&token.address))
                    .filter_map(identified_pair)
                    .collect();
                self.load_pairs(pairs, diagnostics)
            }
            SignalQuery::Address(token) => self.load_token(token),
            SignalQuery::Symbol(symbol) => self.load_symbol(symbol),
            SignalQuery::Slug(slug) => self.load_slug(slug),
        }
    }

    pub fn load_token(&mut self, token: &TokenAddress) -> TokenSignalLoad {
        let mut diagnostics = Vec::new();
        let pairs = self
            .pairs(token, &mut diagnostics)
            .iter()
            .filter_map(identified_pair)
            .filter(|(resolved, _)| same_token_identity(resolved, token))
            .collect::<Vec<_>>();
        if pairs.is_empty() {
            let signal = self
                .pump_json(token, &mut diagnostics)
                .filter(|value| {
                    value.get("mint").and_then(Value::as_str) == Some(token.address.as_str())
                })
                .and_then(|value| self.pump_signal(value, &mut diagnostics));
            return TokenSignalLoad {
                signal,
                diagnostics,
            };
        }
        self.load_pairs(pairs, diagnostics)
    }

    fn pair_candles(
        &mut self,
        token: &TokenAddress,
        pair: &TokenPair,
        diagnostics: &mut Vec<String>,
    ) -> Vec<Vec<f64>> {
        let reference_price = pair_reference_price(pair);
        filter_chart_candles(
            self.candles(token, &pair.pair_address, diagnostics),
            reference_price,
        )
    }

    fn load_pairs(
        &mut self,
        mut pairs: Vec<(TokenAddress, TokenPair)>,
        mut diagnostics: Vec<String>,
    ) -> TokenSignalLoad {
        pairs.sort_by(|(_, left), (_, right)| {
            let left = pair_rank(left);
            let right = pair_rank(right);
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| right.1.total_cmp(&left.1))
        });
        let fallback = pairs.first().cloned();
        for (resolved, pair) in pairs {
            if pair.pair_address.is_empty() {
                continue;
            }
            let candles = self.pair_candles(&resolved, &pair, &mut diagnostics);
            if !candles.is_empty() {
                return TokenSignalLoad {
                    signal: Some(self.enrich(resolved, pair, candles, &mut diagnostics)),
                    diagnostics,
                };
            }
        }
        TokenSignalLoad {
            signal: fallback
                .map(|(resolved, pair)| self.enrich(resolved, pair, Vec::new(), &mut diagnostics)),
            diagnostics,
        }
    }

    fn load_ranked_pairs(
        &mut self,
        mut pairs: Vec<TokenPair>,
        initial_pair: TokenPair,
        initial_token: TokenAddress,
        mut diagnostics: Vec<String>,
    ) -> TokenSignalLoad {
        if !initial_pair.pair_address.is_empty() {
            let candles = self.pair_candles(&initial_token, &initial_pair, &mut diagnostics);
            if !candles.is_empty() {
                return TokenSignalLoad {
                    signal: Some(self.enrich(
                        initial_token,
                        initial_pair,
                        candles,
                        &mut diagnostics,
                    )),
                    diagnostics,
                };
            }
        }
        pairs.sort_by(compare_symbol_pairs);
        for pair in pairs {
            let Some(token) = token_from_pair(&pair) else {
                continue;
            };
            if token != initial_token || pair.pair_address == initial_pair.pair_address {
                continue;
            }
            if pair.pair_address.is_empty() {
                continue;
            }
            let candles = self.pair_candles(&token, &pair, &mut diagnostics);
            if !candles.is_empty() {
                return TokenSignalLoad {
                    signal: Some(self.enrich(token, pair, candles, &mut diagnostics)),
                    diagnostics,
                };
            }
        }
        TokenSignalLoad {
            signal: Some(self.enrich(initial_token, initial_pair, Vec::new(), &mut diagnostics)),
            diagnostics,
        }
    }

    fn load_symbol_pairs(
        &mut self,
        mut pairs: Vec<TokenPair>,
        symbol: &str,
        mut diagnostics: Vec<String>,
    ) -> TokenSignalLoad {
        let normalized = symbol.trim_start_matches('$').to_ascii_lowercase();
        pairs.sort_by(|left, right| {
            let left_exact = left.base_token.symbol.to_ascii_lowercase() == normalized;
            let right_exact = right.base_token.symbol.to_ascii_lowercase() == normalized;
            right_exact
                .cmp(&left_exact)
                .then_with(|| compare_symbol_pairs(left, right))
        });
        let Some((initial_token, initial_pair)) = pairs
            .iter()
            .filter(|pair| pair.base_token.symbol.eq_ignore_ascii_case(&normalized))
            .find_map(identified_pair)
        else {
            return TokenSignalLoad {
                signal: self.search_pump_signal(symbol, &mut diagnostics),
                diagnostics,
            };
        };
        self.load_ranked_pairs(pairs, initial_pair, initial_token, diagnostics)
    }

    pub fn load_symbol(&mut self, symbol: &str) -> TokenSignalLoad {
        let mut diagnostics = Vec::new();
        let pairs = self.search_pairs(symbol, &mut diagnostics);
        self.load_symbol_pairs(pairs, symbol, diagnostics)
    }

    pub fn load_slug(&mut self, slug: &str) -> TokenSignalLoad {
        let mut diagnostics = Vec::new();
        let search = slug.replace(['-', '_'], " ");
        let normalized = normalize_token_name(slug);
        let mut pairs = self
            .search_pairs(&search, &mut diagnostics)
            .into_iter()
            .filter(|pair| {
                token_from_pair(pair).is_some()
                    && (normalize_token_name(&pair.base_token.name) == normalized
                        || normalize_token_name(&pair.base_token.symbol) == normalized)
            })
            .collect::<Vec<_>>();
        pairs.sort_by(compare_symbol_pairs);
        let Some((initial_token, initial_pair)) = pairs.first().and_then(identified_pair) else {
            diagnostics.push(format!("no DexScreener token matched slug {slug}"));
            return TokenSignalLoad {
                signal: None,
                diagnostics,
            };
        };
        self.load_ranked_pairs(pairs, initial_pair, initial_token, diagnostics)
    }

    fn symbol_candidates(
        &mut self,
        mut pairs: Vec<TokenPair>,
        symbol: &str,
        mut diagnostics: Vec<String>,
    ) -> TokenSignalCandidates {
        let normalized = symbol.trim_start_matches('$').to_ascii_lowercase();
        pairs.retain(|pair| {
            token_from_pair(pair).is_some()
                && pair.base_token.symbol.eq_ignore_ascii_case(&normalized)
        });
        pairs.sort_by(compare_symbol_pairs);
        let exact_pairs = pairs.clone();
        let unique = unique_identities(pairs);
        if unique.len() <= 1 {
            let load = self.load_symbol_pairs(exact_pairs, symbol, diagnostics);
            return TokenSignalCandidates {
                signals: load.signal.into_iter().collect(),
                diagnostics: load.diagnostics,
            };
        }
        diagnostics.push(format!(
            "ambiguous token symbol {symbol}: {} identities",
            unique.len()
        ));
        let signals = unique
            .into_iter()
            .take(10)
            .map(|(token, pair)| preview_signal(token, pair))
            .collect();
        TokenSignalCandidates {
            signals,
            diagnostics,
        }
    }

    fn slug_candidates(
        &mut self,
        pairs: Vec<TokenPair>,
        slug: &str,
        diagnostics: Vec<String>,
    ) -> TokenSignalCandidates {
        let normalized = normalize_token_name(slug);
        let pairs = pairs
            .into_iter()
            .filter(|pair| {
                token_from_pair(pair).is_some()
                    && (normalize_token_name(&pair.base_token.name) == normalized
                        || normalize_token_name(&pair.base_token.symbol) == normalized)
            })
            .collect::<Vec<_>>();
        let mut pairs = pairs;
        pairs.sort_by(compare_symbol_pairs);
        let exact_pairs = pairs.clone();
        let unique = unique_identities(pairs);
        if unique.len() <= 1 {
            let Some((token, pair)) = unique.first().cloned() else {
                return TokenSignalCandidates {
                    signals: Vec::new(),
                    diagnostics,
                };
            };
            let load = self.load_ranked_pairs(exact_pairs, pair, token, diagnostics);
            return TokenSignalCandidates {
                signals: load.signal.into_iter().collect(),
                diagnostics: load.diagnostics,
            };
        }
        let signals = unique
            .into_iter()
            .take(10)
            .map(|(token, pair)| preview_signal(token, pair))
            .collect();
        TokenSignalCandidates {
            signals,
            diagnostics,
        }
    }

    pub fn load_candidates(&mut self, query: &SignalQuery) -> TokenSignalCandidates {
        match query {
            SignalQuery::Address(_) => {
                let load = self.load_query(query);
                TokenSignalCandidates {
                    signals: load.signal.into_iter().collect(),
                    diagnostics: load.diagnostics,
                }
            }
            SignalQuery::Symbol(symbol) => {
                let mut diagnostics = Vec::new();
                let pairs = self.search_pairs(symbol, &mut diagnostics);
                self.symbol_candidates(pairs, symbol, diagnostics)
            }
            SignalQuery::Slug(slug) => {
                let mut diagnostics = Vec::new();
                let search = slug.replace(['-', '_'], " ");
                let pairs = self.search_pairs(&search, &mut diagnostics);
                self.slug_candidates(pairs, slug, diagnostics)
            }
        }
    }

    fn pump_period_candles(
        &mut self,
        signal: &TokenSignal,
        interval: &str,
        limit: i64,
        period: &str,
        now: i64,
    ) -> Vec<Vec<f64>> {
        let url = format!(
            "https://swap-api.pump.fun/v2/coins/{}/candles",
            signal.token.address
        );
        let key = format!(
            "token_signal:pump-history:{}:{period}",
            signal.token.address
        );
        let created = signal
            .pump
            .as_ref()
            .and_then(|pump| flexible_number(&pump.created_timestamp))
            .unwrap_or(0.0) as i64;
        self.cached_json(
            &key,
            60,
            "pump.fun chart history",
            &|transport| {
                transport.get_json(
                    &url,
                    &[
                        ("interval", interval.to_owned()),
                        ("limit", limit.to_string()),
                        ("currency", "USD".to_owned()),
                        ("createdTs", created.to_string()),
                        ("beforeTs", now.to_string()),
                    ],
                )
            },
            &Some,
            &mut Vec::new(),
        )
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|value| {
            let mut row = vec![flexible_number(value.get("timestamp")?)? / 1000.0];
            for key in ["open", "high", "low", "close"] {
                let price = flexible_number(value.get(key)?)?;
                if !price.is_finite() || price <= 0.0 {
                    return None;
                }
                row.push(price);
            }
            row.push(value.get("volume").and_then(flexible_number).unwrap_or(0.0));
            Some(row)
        })
        .collect()
    }

    pub fn period_candles(
        &mut self,
        signal: &TokenSignal,
        period: &str,
        now: i64,
    ) -> Result<Vec<Vec<f64>>, String> {
        let history = self.load_period_history(signal, period, now)?;
        if history.windowed.is_empty() {
            Ok(history.idle_pump_candles(now))
        } else {
            Ok(history.windowed)
        }
    }

    pub fn render_period_photo(
        &mut self,
        signal: &TokenSignal,
        period: &str,
        now: i64,
    ) -> Result<Vec<u8>, String> {
        let history = self.load_period_history(signal, period, now)?;
        let shown = history
            .shown_period(period)
            .unwrap_or_else(|| period.to_owned());
        if history.windowed.is_empty() {
            let flat = history.idle_pump_candles(now);
            if !flat.is_empty() {
                let title = format!(
                    "{} ({shown})\nLast available trade price",
                    signal.pair.base_token.symbol
                );
                let mut pair = signal.pair.clone();
                pair.price_usd = json!(flat[1][4]);
                return render_price_chart(&pair, &flat, Some(&title), None, 1280, 900);
            }
            return Err("requested token history unavailable".into());
        }
        let title = format!(
            "{} ({shown})\nAvailable history",
            signal.pair.base_token.symbol
        );
        render_price_chart(
            &signal.pair,
            &history.windowed,
            Some(&title),
            None,
            1280,
            900,
        )
    }

    fn load_period_history(
        &mut self,
        signal: &TokenSignal,
        period: &str,
        now: i64,
    ) -> Result<PeriodHistory, String> {
        let range =
            bot_core::price_queries::ChartPeriod::parse(period).ok_or("invalid chart period")?;
        let (unit, aggregate, step) = match range.seconds {
            0..=3600 => ("minute", 1, 60),
            3601..=86400 => ("minute", 5, 300),
            86401..=604800 => ("hour", 1, 3600),
            604801..=5184000 => ("hour", 4, 14400),
            _ => ("day", 1, 86400),
        };
        let limit = ((range.seconds + step - 1) / step + 1).min(1000);
        let pump_history = signal.token.chain_id == "solana"
            && signal.pump.is_some()
            && signal.pair.pair_address.is_empty();
        let candles = if pump_history {
            let interval = match unit {
                "minute" => format!("{aggregate}m"),
                "hour" => format!("{aggregate}h"),
                _ => "24h".to_owned(),
            };
            self.pump_period_candles(signal, &interval, limit, period, now)
        } else {
            let url = format!(
                "https://api.geckoterminal.com/api/v2/networks/{}/pools/{}/ohlcv/{unit}",
                signal.token.network, signal.pair.pair_address
            );
            let key = format!(
                "token_signal:history:{}:{}:{period}",
                signal.token.network, signal.pair.pair_address
            );
            let raw = self.cached_json(
                &key,
                60,
                "token chart history",
                &|transport| {
                    transport.get_json(
                        &url,
                        &[
                            ("aggregate", aggregate.to_string()),
                            ("limit", limit.to_string()),
                            ("currency", "usd".into()),
                            ("before_timestamp", now.to_string()),
                        ],
                    )
                },
                &|value| value.pointer("/data/attributes/ohlcv_list").cloned(),
                &mut Vec::new(),
            );
            raw.and_then(|value| serde_json::from_value::<Vec<Vec<f64>>>(value).ok())
                .unwrap_or_default()
        };
        let reference_price = pair_reference_price(&signal.pair);
        let candles = filter_chart_candles(candles, reference_price);
        let last_known = candles
            .iter()
            .filter(|row| row[0] < (now - range.seconds) as f64)
            .max_by(|a, b| a[0].total_cmp(&b[0]))
            .map(|row| row[4]);
        let windowed = candles
            .into_iter()
            .filter(|row| row[0] >= (now - range.seconds) as f64 && row[0] <= now as f64)
            .collect();
        Ok(PeriodHistory {
            windowed,
            last_known,
            pump_history,
            range_seconds: range.seconds,
        })
    }

    pub fn load_state(&mut self, signal_id: &str) -> Result<Option<SignalState>, String> {
        let key = signal_state_key(signal_id);
        self.cache
            .get(&key)
            .map_err(|error| error.to_string())?
            .map(|encoded| {
                serde_json::from_str::<Option<SignalState>>(&encoded)
                    .map_err(|error| format!("invalid token-signal state {key}: {error}"))
            })
            .transpose()
            .map(Option::flatten)
    }

    pub fn clear_state(&mut self, signal_id: &str) -> Result<(), String> {
        self.cache
            .set(&signal_state_key(signal_id), "null", 1)
            .map_err(|error| error.to_string())
    }

    pub fn save_state(&mut self, signal_id: &str, state: &SignalState) -> Result<(), String> {
        let key = signal_state_key(signal_id);
        let encoded =
            serde_json::to_string(state).map_err(failure("could not encode token-signal state"))?;
        self.cache
            .set(&key, &encoded, SIGNAL_STATE_TTL_SECONDS)
            .map_err(|error| error.to_string())
    }
}

struct PeriodHistory {
    windowed: Vec<Vec<f64>>,
    last_known: Option<f64>,
    pump_history: bool,
    range_seconds: i64,
}

impl PeriodHistory {
    #[cfg(test)]
    fn for_test(windowed: Vec<Vec<f64>>, range_seconds: i64) -> Self {
        Self {
            windowed,
            last_known: None,
            pump_history: false,
            range_seconds,
        }
    }

    fn shown_period(&self, requested: &str) -> Option<String> {
        if self.windowed.is_empty() {
            return None;
        }
        let mut oldest = f64::INFINITY;
        let mut latest = f64::NEG_INFINITY;
        for row in &self.windowed {
            if let Some(timestamp) = row.first().copied().filter(|value| value.is_finite()) {
                oldest = oldest.min(timestamp);
                latest = latest.max(timestamp);
            }
        }
        if !oldest.is_finite() || !latest.is_finite() {
            return None;
        }
        bot_core::token_signals::span_period_label(
            (latest - oldest) as i64,
            self.range_seconds,
            requested,
        )
    }

    fn idle_pump_candles(&self, now: i64) -> Vec<Vec<f64>> {
        match (self.pump_history, self.last_known) {
            (true, Some(price)) => vec![
                vec![
                    (now - self.range_seconds) as f64,
                    price,
                    price,
                    price,
                    price,
                    0.0,
                ],
                vec![now as f64, price, price, price, price, 0.0],
            ],
            _ => Vec::new(),
        }
    }
}

fn flexible_number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn pair_reference_price(pair: &TokenPair) -> Option<f64> {
    flexible_number(&pair.price_usd).filter(|price| *price > 0.0)
}

fn filter_chart_candles(candles: Vec<Vec<f64>>, reference_price: Option<f64>) -> Vec<Vec<f64>> {
    candles
        .into_iter()
        .filter(|candle| is_usable_chart_candle(candle, reference_price))
        .collect()
}

/// Fonts are read and parsed once per process; `FontArc` clones are cheap.
static CHART_FONT_REGULAR: LazyLock<Option<FontArc>> = LazyLock::new(|| load_chart_font(false));
static CHART_FONT_BOLD: LazyLock<Option<FontArc>> = LazyLock::new(|| load_chart_font(true));

fn chart_font(bold: bool) -> Option<FontArc> {
    if bold {
        CHART_FONT_BOLD.clone()
    } else {
        CHART_FONT_REGULAR.clone()
    }
}

fn load_chart_font(bold: bool) -> Option<FontArc> {
    let paths = if bold {
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    } else {
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    };
    paths.into_iter().find_map(|path| {
        std::fs::read(path)
            .ok()
            .and_then(|bytes| FontArc::try_from_vec(bytes).ok())
    })
}

fn fill_rectangle(
    image: &mut RgbImage,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    color: Rgb<u8>,
) {
    let width = i32::try_from(image.width()).unwrap_or(i32::MAX);
    let height = i32::try_from(image.height()).unwrap_or(i32::MAX);
    for y in top.max(0)..=bottom.min(height.saturating_sub(1)) {
        for x in left.max(0)..=right.min(width.saturating_sub(1)) {
            image.put_pixel(x as u32, y as u32, color);
        }
    }
}

fn draw_line(image: &mut RgbImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        if let (Ok(x), Ok(y)) = (u32::try_from(x0), u32::try_from(y0))
            && x < image.width()
            && y < image.height()
        {
            image.put_pixel(x, y, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = error.saturating_mul(2);
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

pub fn render_signal_chart(
    signal: &TokenSignal,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    render_price_chart(&signal.pair, &signal.candles, None, None, width, height)
}

/// Render daily market candles using the same drawing engine as token cards.
pub fn render_market_chart(
    quote: &bot_core::stocks::StockQuote,
    candles: &[Vec<f64>],
) -> Result<Vec<u8>, String> {
    render_market_chart_for_period(quote, candles, "5d")
}

/// Keep chart captions aligned with the requested chart period.
#[must_use]
pub fn market_chart_caption(
    quote: &bot_core::stocks::StockQuote,
    candles: &[Vec<f64>],
    period: &str,
) -> String {
    let opening_price = candles
        .iter()
        .filter_map(|candle| {
            let timestamp = candle.first().copied()?;
            let opening_price = candle.get(1).copied()?;
            (timestamp.is_finite() && opening_price.is_finite() && opening_price > 0.0)
                .then_some((timestamp, opening_price))
        })
        .min_by(|(left, _), (right, _)| left.total_cmp(right))
        .map(|(_, opening_price)| opening_price);
    let change = opening_price.and_then(|opening_price| {
        let change = (quote.price / opening_price - 1.0) * 100.0;
        change.is_finite().then_some(change)
    });
    bot_core::output_format::quote(&quote.symbol, quote.price, &quote.currency, change, period)
}

pub fn render_market_chart_for_period(
    quote: &bot_core::stocks::StockQuote,
    candles: &[Vec<f64>],
    period: &str,
) -> Result<Vec<u8>, String> {
    if candles.is_empty() {
        return Err("historical chart data unavailable".to_owned());
    }
    let pair = TokenPair {
        price_usd: json!(quote.price),
        ..TokenPair::default()
    };
    let title = format!("{} ({period})\n{}", quote.symbol, quote.name);
    render_price_chart(
        &pair,
        candles,
        Some(&title),
        Some(&quote.currency),
        1_280,
        900,
    )
}

// Fit the complete numeric value and currency inside the right margin.
fn chart_label_size(font: &FontArc, text: &str, preferred: f32, available: u32) -> f32 {
    let mut size = preferred;
    while imageproc::drawing::text_size(size, font, text).0 > available {
        size *= 0.95;
    }
    size
}

fn render_price_chart(
    pair: &TokenPair,
    candles: &[Vec<f64>],
    heading: Option<&str>,
    currency: Option<&str>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let fonts = ChartFonts {
        bold: chart_font(true),
        regular: chart_font(false),
    };
    render_price_chart_with_fonts(pair, candles, heading, currency, (width, height), &fonts)
}

/// Fonts are optional: without system fonts the chart still renders, only
/// without text labels.
struct ChartFonts {
    bold: Option<FontArc>,
    regular: Option<FontArc>,
}

fn render_price_chart_with_fonts(
    pair: &TokenPair,
    candles: &[Vec<f64>],
    heading: Option<&str>,
    currency: Option<&str>,
    (width, height): (u32, u32),
    fonts: &ChartFonts,
) -> Result<Vec<u8>, String> {
    if width < 320 || height < 240 {
        return Err("token chart dimensions are too small".to_owned());
    }
    let mut image = RgbImage::from_pixel(width, height, Rgb([7, 9, 18]));
    let left = 56_i32;
    let right = i32::try_from(width).map_err(|_| "chart width is too large")? - 200;
    let top = 130_i32;
    let bottom = i32::try_from(height).map_err(|_| "chart height is too large")? - 82;
    let price = flexible_number(&pair.price_usd).unwrap_or(0.0);
    let symbol = if pair.base_token.symbol.is_empty() {
        "TOKEN".to_owned()
    } else {
        pair.base_token.symbol.to_ascii_uppercase()
    };
    let price_label = |value| match currency {
        Some(currency) => format!(
            "{} {currency}",
            format_money(value, true).trim_start_matches('$')
        ),
        None => format_money(value, true),
    };
    let price_text = price_label(price);
    if let Some(font) = &fonts.bold {
        let title = format!("{symbol}\n{price_text}");
        for (index, line) in heading.unwrap_or(&title).lines().take(2).enumerate() {
            let font_size = if index == 0 { 38.0 } else { 28.0 };
            let mut label = line.to_owned();
            while imageproc::drawing::text_size(font_size, &font, &label).0
                > width.saturating_sub(48)
                && label.chars().count() > 1
            {
                label.pop();
            }
            draw_text_mut(
                &mut image,
                Rgb([220, 231, 244]),
                24,
                18 + index as i32 * 46,
                font_size,
                &font,
                &label,
            );
        }
    }
    for index in 0..6 {
        let y = top + (bottom - top) * index / 5;
        draw_line(&mut image, left, y, right, y, Rgb([17, 24, 39]));
    }
    for index in 0..7 {
        let x = left + (right - left) * index / 6;
        draw_line(&mut image, x, top, x, bottom, Rgb([17, 24, 39]));
    }
    let reference_price = pair_reference_price(pair);
    let mut candles = filter_chart_candles(candles.to_vec(), reference_price);
    if candles
        .first()
        .zip(candles.last())
        .is_some_and(|(first, last)| first[0] > last[0])
    {
        candles.reverse();
    }
    if candles.is_empty() {
        return Err("token chart history unavailable".to_owned());
    } else {
        let low = candles
            .iter()
            .map(|candle| candle[3])
            .fold(f64::INFINITY, f64::min);
        let high = candles
            .iter()
            .map(|candle| candle[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut range = high - low;
        if !range.is_finite() || range.abs() <= f64::EPSILON {
            range = high.abs().max(1e-12) * 0.02;
        }
        let minimum = low - range * 0.08;
        let maximum = high + range * 0.08;
        let span = (maximum - minimum).max(f64::EPSILON);
        let y_for =
            |value: f64| top + (((maximum - value) / span) * f64::from(bottom - top)) as i32;
        let chart_width = (right - left).max(1);
        let count = i32::try_from(candles.len()).unwrap_or(i32::MAX).max(1);
        let step = f64::from(chart_width) / f64::from(count);
        let body_width = ((step * 0.58) as i32).max(4);
        for (index, candle) in candles.iter().enumerate() {
            let index = i32::try_from(index).unwrap_or(i32::MAX);
            let x = left + (f64::from(index) * step + step / 2.0) as i32;
            let color = if candle[4] >= candle[1] {
                Rgb([18, 184, 166])
            } else {
                Rgb([255, 67, 86])
            };
            draw_line(&mut image, x, y_for(candle[3]), x, y_for(candle[2]), color);
            let body_top = y_for(candle[1].max(candle[4]));
            let body_bottom = y_for(candle[1].min(candle[4])).max(body_top + 2);
            fill_rectangle(
                &mut image,
                x - body_width / 2,
                body_top,
                x + body_width / 2,
                body_bottom,
                color,
            );
        }
        let current = if price != 0.0 {
            price
        } else {
            candles.last().map_or(0.0, |candle| candle[4])
        };
        let current_y = y_for(current).clamp(top, bottom);
        draw_line(
            &mut image,
            left,
            current_y,
            right,
            current_y,
            Rgb([0, 184, 148]),
        );
        if let Some(font) = &fonts.regular {
            for index in 0..6 {
                let value = maximum - span * f64::from(index) / 5.0;
                let y = top + (bottom - top) * index / 5;
                if (y - current_y).abs() > 32 {
                    draw_text_mut(
                        &mut image,
                        Rgb([170, 185, 205]),
                        right + 12,
                        y - 12,
                        chart_label_size(
                            font,
                            &price_label(value),
                            24.0,
                            width.saturating_sub((right + 24) as u32),
                        ),
                        &font,
                        &price_label(value),
                    );
                }
            }
            draw_text_mut(
                &mut image,
                Rgb([54, 224, 195]),
                right + 12,
                current_y - 12,
                chart_label_size(
                    font,
                    &price_text,
                    26.0,
                    width.saturating_sub((right + 24) as u32),
                ),
                &font,
                &price_text,
            );
        }
    }
    if let Some(font) = &fonts.regular {
        for (candle, x) in [
            (candles.first(), left),
            (candles.last(), right.saturating_sub(250)),
        ] {
            if let Some(candle) = candle
                && let Some(date) = chrono::DateTime::from_timestamp(candle[0] as i64, 0)
            {
                draw_text_mut(
                    &mut image,
                    Rgb([170, 185, 205]),
                    x,
                    bottom + 24,
                    24.0,
                    &font,
                    &date.format("%Y-%m-%d %H:%M UTC").to_string(),
                );
            }
        }
    }
    let border = Rgb([42, 52, 66]);
    draw_line(&mut image, left, top, right, top, border);
    draw_line(&mut image, right, top, right, bottom, border);
    draw_line(&mut image, right, bottom, left, bottom, border);
    draw_line(&mut image, left, bottom, left, top, border);
    encode_png(image)
}

fn encode_png(image: RgbImage) -> Result<Vec<u8>, String> {
    let mut output = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut output, ImageFormat::Png)
        .map_err(failure("token chart PNG encode failed"))?;
    Ok(output.into_inner())
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    #[test]
    fn chart_price_labels_fit_without_losing_currency_or_precision() {
        // Label sizing only applies when a system font is installed.
        super::chart_font(false).into_iter().for_each(|font| {
            for label in ["123456.789 USD", "1500000.123 ARS", "0.00000679 USD"] {
                for preferred in [24.0, 26.0] {
                    let size = super::chart_label_size(&font, label, preferred, 176);
                    assert!(imageproc::drawing::text_size(size, &font, label).0 <= 176);
                    assert!(size <= preferred);
                }
            }
            assert_eq!(super::chart_label_size(&font, "$1", 26.0, 176), 26.0);
        });
    }

    #[test]
    fn market_caption_uses_requested_period() {
        let mut quote = bot_core::stocks::StockQuote {
            symbol: "BTC".into(),
            name: "Bitcoin".into(),
            price: 120.0,
            currency: "USD".into(),
            exchange: String::new(),
            asset_type: String::new(),
            variation: -1.53,
        };
        let candles = vec![
            vec![1.0, 100.0, 110.0, 90.0, 105.0],
            vec![2.0, 105.0, 125.0, 100.0, 120.0],
        ];
        for period in ["1m", "7d", "2h", "1y"] {
            assert_eq!(
                super::market_chart_caption(&quote, &candles, period),
                format!("BTC: 120 USD (+20% {period})")
            );
        }
        quote.price = 80.0;
        assert_eq!(
            super::market_chart_caption(&quote, &candles, "1m"),
            "BTC: 80 USD (-20% 1m)"
        );
        for missing in [
            vec![],
            vec![vec![]],
            vec![vec![1.0, 0.0]],
            vec![vec![1.0, f64::NAN]],
        ] {
            assert_eq!(
                super::market_chart_caption(&quote, &missing, "1m"),
                "BTC: 80 USD (N/A 1m)"
            );
        }
    }

    #[test]
    fn market_chart_renders_real_candles_and_rejects_missing_history() -> Result<(), String> {
        let quote = bot_core::stocks::StockQuote {
            symbol: "AAPL".into(),
            name: "Apple".into(),
            price: 100.0,
            currency: "JPY".into(),
            exchange: "TEST".into(),
            asset_type: String::new(),
            variation: 1.0,
        };
        assert!(super::render_market_chart(&quote, &[]).is_err());
        let png =
            super::render_market_chart(&quote, &[vec![1.0, 99.0, 102.0, 98.0, 100.0, 1000.0]])?;
        assert!(png.starts_with(b"\x89PNG"));
        Ok(())
    }

    #[test]
    fn chart_history_drops_extreme_provider_candles() {
        let candles = vec![
            vec![1.0, 0.9, 1.1, 0.8, 1.0],
            vec![2.0, 1.0, 2_000_000.0, 0.9, 1.1],
        ];
        let filtered = super::filter_chart_candles(candles, Some(1.0));
        assert_eq!(filtered, vec![vec![1.0, 0.9, 1.1, 0.8, 1.0]]);
    }

    use std::collections::{BTreeMap, VecDeque};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use bot_core::token_signals::{SignalQuery, TokenAddress, TokenSignal};
    use serde_json::json;

    use super::{
        BinaryResponse, JsonResponse, PeriodHistory, ReqwestTokenSignalTransport,
        TokenSignalAdapter, TokenSignalCache, TokenSignalTransport, render_signal_chart,
    };

    #[test]
    fn missing_candles_never_switch_to_a_different_mint() -> Result<(), String> {
        let a = "0x0000000000000000000000000000000000000001";
        let b = "0x0000000000000000000000000000000000000002";
        let pair = |mint: &str, pool: &str, liquidity: u64| {
            json!({
                "chainId":"ethereum", "pairAddress":pool,
                "baseToken":{"address":mint,"symbol":"TIMBA"}, "liquidity":{"usd":liquidity}
            })
        };
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse { status_code:200, body:json!({"pairs":[pair(a,"primary",1000),pair(b,"other",500),pair(a,"alternate",100)]}).to_string() },
                JsonResponse { status_code:200, body:json!({"data":{"attributes":{"ohlcv_list":[]}}}).to_string() },
                JsonResponse { status_code:200, body:json!({"data":{"attributes":{"ohlcv_list":[[1,1,2,0.5,1.5]]}}}).to_string() },
            ])), post: Default::default(), binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let signal = adapter
            .load_symbol("timba")
            .signal
            .ok_or("missing signal")?;
        assert_eq!(signal.token.address, a);
        assert_eq!(signal.pair.pair_address, "alternate");
        assert!(
            !adapter
                .cache
                .writes
                .iter()
                .any(|(key, _)| key.contains(":other:"))
        );
        Ok(())
    }

    #[test]
    fn reqwest_transport_supports_json_get_post_and_binary_downloads() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            for (content_type, body) in [
                ("application/json", br#"{"method":"get"}"#.as_slice()),
                ("application/json", br#"{"method":"post"}"#.as_slice()),
                ("image/png", &[1_u8, 2, 3][..]),
            ] {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 8_192];
                let _ = stream.read(&mut request);
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(headers.as_bytes())?;
                stream.write_all(body)?;
            }
            Ok(())
        });
        let transport = ReqwestTokenSignalTransport::new()?;
        let base_url = format!("http://{address}");
        let get = transport.get_json(&base_url, &[("query", "synthetic".to_owned())])?;
        assert!(get.body.contains("get"));
        let post = transport.post_json(&base_url, &json!({"value":"synthetic"}))?;
        assert!(post.body.contains("post"));
        let binary = transport.get_binary(&base_url)?;
        assert_eq!(binary.content_type, "image/png");
        assert_eq!(binary.body, [1, 2, 3]);
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[derive(Default)]
    struct Cache {
        values: BTreeMap<String, String>,
        writes: Vec<(String, i64)>,
        fail_get: bool,
        fail_set: bool,
    }

    impl TokenSignalCache for Cache {
        type Error = &'static str;

        fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
            if self.fail_get {
                return Err("synthetic cache read failure");
            }
            Ok(self.values.get(key).cloned())
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            if self.fail_set {
                return Err("synthetic cache write failure");
            }
            self.values.insert(key.to_owned(), value.to_owned());
            self.writes.push((key.to_owned(), ttl_seconds));
            Ok(())
        }
    }

    struct Transport {
        json: std::cell::RefCell<VecDeque<JsonResponse>>,
        post: std::cell::RefCell<VecDeque<JsonResponse>>,
        binary: std::cell::RefCell<VecDeque<BinaryResponse>>,
    }

    impl TokenSignalTransport for Transport {
        fn get_json(&self, _url: &str, _query: &[(&str, String)]) -> Result<JsonResponse, String> {
            self.json
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| "unexpected request".to_owned())
        }

        fn post_json(&self, _url: &str, _body: &serde_json::Value) -> Result<JsonResponse, String> {
            self.post
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| "synthetic supply unavailable".to_owned())
        }

        fn get_binary(&self, _url: &str) -> Result<BinaryResponse, String> {
            self.binary
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| "synthetic image unavailable".to_owned())
        }
    }

    /// A transport whose JSON GETs are answered by a closure; POST and binary
    /// downloads are not part of the scenarios that use it.
    struct JsonOnly<F>(F);

    impl<F> TokenSignalTransport for JsonOnly<F>
    where
        F: Fn(&str, &[(&str, String)]) -> Result<JsonResponse, String>,
    {
        fn get_json(&self, url: &str, query: &[(&str, String)]) -> Result<JsonResponse, String> {
            (self.0)(url, query)
        }

        fn post_json(&self, _: &str, _: &serde_json::Value) -> Result<JsonResponse, String> {
            Err("unused".into())
        }

        fn get_binary(&self, _: &str) -> Result<BinaryResponse, String> {
            Err("unused".into())
        }
    }

    type JsonHandler = Box<dyn Fn(&str, &[(&str, String)]) -> Result<JsonResponse, String>>;

    fn pump_history(body: serde_json::Value, interval: &'static str) -> JsonOnly<JsonHandler> {
        JsonOnly(Box::new(move |url: &str, query: &[(&str, String)]| {
            assert_eq!(url, "https://swap-api.pump.fun/v2/coins/timba-mint/candles");
            assert!(query.contains(&("interval", interval.to_owned())));
            assert!(query.contains(&("currency", "USD".to_owned())));
            assert!(query.contains(&("createdTs", "1700000000000".to_owned())));
            assert!(query.contains(&("beforeTs", "1800000000".to_owned())));
            Ok(JsonResponse {
                status_code: 200,
                body: body.to_string(),
            })
        }))
    }

    #[test]
    fn scripted_transports_reject_requests_outside_their_scenario() {
        let json_only: JsonOnly<JsonHandler> = JsonOnly(Box::new(|_, _| Err("no json".into())));
        assert_eq!(
            json_only
                .get_json("https://example.test", &[])
                .err()
                .as_deref(),
            Some("no json")
        );
        assert_eq!(
            json_only
                .post_json("https://example.test", &json!({}))
                .err()
                .as_deref(),
            Some("unused")
        );
        assert_eq!(
            json_only
                .get_binary("https://example.test")
                .err()
                .as_deref(),
            Some("unused")
        );
        let transport = Transport {
            json: Default::default(),
            post: Default::default(),
            binary: std::cell::RefCell::new(VecDeque::from([BinaryResponse {
                status_code: 200,
                content_type: "image/png".to_owned(),
                body: vec![1],
            }])),
        };
        assert_eq!(
            transport
                .get_binary("https://example.test")
                .map(|image| image.body),
            Ok(vec![1])
        );
        assert_eq!(
            transport
                .get_binary("https://example.test")
                .err()
                .as_deref(),
            Some("synthetic image unavailable")
        );
    }

    #[test]
    fn pump_history_renders_recent_and_idle_tokens_without_a_dex_pool() -> Result<(), String> {
        let signal = TokenSignal {
            token: TokenAddress {
                chain_id: "solana".into(),
                network: "solana".into(),
                tag: "SOL".into(),
                address: "timba-mint".into(),
            },
            pair: Default::default(),
            candles: vec![],
            supply: None,
            token_image_url: None,
            socials: BTreeMap::new(),
            pump: Some(
                serde_json::from_value(json!({"created_timestamp":1700000000000_i64}))
                    .ok()
                    .ok_or("pump metadata")?,
            ),
        };
        let candle = |timestamp| json!({"timestamp": timestamp, "open":"0.000005", "high":"0.000006", "low":"0.000004", "close":"0.00000525", "volume":"3.97"});
        for timestamp in [1799999970000_i64, 1799900000000] {
            let mut adapter = TokenSignalAdapter::new(
                pump_history(json!([candle(timestamp)]), "1m"),
                Cache::default(),
            );
            let parsed = adapter.pump_period_candles(&signal, "1m", 61, "1h", 1800000000);
            assert_eq!(
                parsed,
                vec![vec![
                    timestamp as f64 / 1000.0,
                    0.000005,
                    0.000006,
                    0.000004,
                    0.00000525,
                    3.97
                ]]
            );
            assert!(
                adapter
                    .render_period_photo(&signal, "1h", 1800000000)?
                    .starts_with(b"\x89PNG")
            );
        }
        let mut idle = TokenSignalAdapter::new(
            pump_history(json!([candle(1_799_900_000_000_i64)]), "1m"),
            Cache::default(),
        );
        let idle_candles = idle.period_candles(&signal, "1h", 1_800_000_000)?;
        assert_eq!(idle_candles.len(), 2);
        assert_eq!(idle_candles[0][4], idle_candles[1][4]);
        let covered = PeriodHistory::for_test(
            vec![
                vec![1_702_419_200.0, 1.0, 1.0, 1.0, 100.0],
                vec![1_702_592_000.0, 2.0, 2.0, 2.0, 125.0],
            ],
            604_800,
        );
        assert_eq!(covered.shown_period("7d"), Some("2d".to_owned()));
        let near_complete = PeriodHistory::for_test(
            vec![
                vec![1_700_057_600.0, 1.0, 1.0, 1.0, 100.0],
                vec![1_702_592_000.0, 2.0, 2.0, 2.0, 156.0],
            ],
            2_592_000,
        );
        assert_eq!(near_complete.shown_period("30d"), Some("30d".to_owned()));
        let empty = PeriodHistory::for_test(Vec::new(), 86_400);
        assert_eq!(empty.shown_period("24h"), None);
        let undated = PeriodHistory::for_test(vec![vec![f64::NAN, 1.0, 1.0, 1.0, 1.0]], 86_400);
        assert_eq!(undated.shown_period("24h"), None);
        for (period, interval) in [
            ("1h", "1m"),
            ("1d", "5m"),
            ("24h", "5m"),
            ("7d", "1h"),
            ("60d", "4h"),
            ("61d", "24h"),
            ("1y", "24h"),
            ("5y", "24h"),
        ] {
            let mut adapter = TokenSignalAdapter::new(
                pump_history(json!([candle(1799999970000_i64)]), interval),
                Cache::default(),
            );
            assert!(
                adapter
                    .render_period_photo(&signal, period, 1800000000)?
                    .starts_with(b"\x89PNG")
            );
        }
        for missing in [
            json!([]),
            json!([candle(1800000060000_i64)]),
            json!([{"timestamp":1799999970000_i64,"open":"bad"}]),
            json!([{"timestamp":1799999970000_i64,"open":"0","high":"1","low":"1","close":"1"}]),
        ] {
            let mut adapter =
                TokenSignalAdapter::new(pump_history(missing, "1m"), Cache::default());
            assert!(
                adapter
                    .render_period_photo(&signal, "1h", 1800000000)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn token_history_ranges_choose_granularity_and_keep_identity() -> Result<(), String> {
        let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::<String>::new()));
        let recorded = calls.clone();
        let history: JsonHandler = Box::new(move |url: &str, query: &[(&str, String)]| {
            recorded.borrow_mut().push(format!("{url} {query:?}"));
            Ok(JsonResponse { status_code: 200, body: json!({"data":{"attributes":{"ohlcv_list":[[1799999970,1,2,0.5,1.5,100],[1,1,2,0.5,1.5,100]]}}}).to_string() })
        });
        let mut adapter = TokenSignalAdapter::new(JsonOnly(history), Cache::default());
        let signal = TokenSignal {
            token: TokenAddress {
                chain_id: "solana".into(),
                network: "solana".into(),
                tag: "SOL".into(),
                address: "mint".into(),
            },
            pair: bot_core::token_signals::TokenPair {
                pair_address: "fixed-pool".into(),
                ..Default::default()
            },
            candles: vec![],
            supply: None,
            token_image_url: None,
            socials: BTreeMap::new(),
            pump: None,
        };
        for period in ["1h", "1d", "7d", "1m", "1y", "5y"] {
            assert!(
                adapter
                    .render_period_photo(&signal, period, 1_800_000_000)?
                    .starts_with(b"\x89PNG")
            );
            assert!(
                !adapter
                    .period_candles(&signal, period, 1_800_000_000)?
                    .is_empty(),
                "{period}"
            );
        }
        let calls = calls.borrow();
        assert!(calls[0].contains("fixed-pool/ohlcv/minute"));
        assert!(calls[2].contains("fixed-pool/ohlcv/hour"));
        assert!(calls[4].contains("fixed-pool/ohlcv/day"));
        drop(calls);
        assert!(
            adapter
                .render_period_photo(&signal, "bad", 1_800_000_000)
                .is_err()
        );
        assert!(
            adapter
                .period_candles(&signal, "bad", 1_800_000_000)
                .is_err()
        );
        assert!(
            adapter
                .render_period_photo(&signal, "2m", 1_900_000_000)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn timba_ticker_and_mint_use_pump_when_dexscreener_has_no_listing() -> Result<(), String> {
        let mint = "F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump";
        let coin = json!({
            "mint": mint, "name": "TIMBA", "symbol": "TIMBA",
            "usd_market_cap": 5140.0, "total_supply": 1000000000000000_u64,
            "real_token_reserves": 510180253241868_u64,
            "twitter": "https://x.com/timbatoken", "complete": false,
            "image_uri": "https://image.test/timba.png"
        });
        for query in ["$timba", mint] {
            let search = query.starts_with('$');
            let mut other = coin.clone();
            other["symbol"] = json!("TIMBAX");
            other["usd_market_cap"] = json!(999999);
            let mut smaller = coin.clone();
            smaller["mint"] = json!("J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump");
            smaller["usd_market_cap"] = json!(100);
            let transport = Transport {
                json: std::cell::RefCell::new(VecDeque::from([
                    JsonResponse {
                        status_code: 200,
                        body: if search {
                            json!({"pairs":[]})
                        } else {
                            json!([])
                        }
                        .to_string(),
                    },
                    JsonResponse {
                        status_code: 200,
                        body: if search {
                            json!([other, smaller, coin])
                        } else {
                            coin.clone()
                        }
                        .to_string(),
                    },
                ])),
                post: std::cell::RefCell::new(VecDeque::new()),
                binary: std::cell::RefCell::new(VecDeque::new()),
            };
            let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
            let query = bot_core::token_signals::detect_signal_query(query).ok_or("query")?;
            let signal = adapter.load_query(&query).signal.ok_or("signal")?;
            assert_eq!(signal.token.address, mint);
            assert_eq!(signal.pair.base_token.symbol, "TIMBA");
            assert_eq!(signal.supply, Some(1_000_000_000.0));
            assert_eq!(signal.pair.market_cap, json!(5140.0));
            assert_eq!(signal.pair.price_usd, json!(0.00000514));
            assert_eq!(
                signal.token_image_url.as_deref(),
                Some("https://image.test/timba.png")
            );
            assert_eq!(
                signal.socials.get("X").map(String::as_str),
                Some("https://x.com/timbatoken")
            );
            let caption = bot_core::token_signals::format_signal_caption(&signal, 0);
            assert!(caption.contains(&format!("https://pump.fun/coin/{mint}")));
            assert!(!caption.contains("geckoterminal.com"));
            assert!(
                adapter
                    .render_period_photo(&signal, "24h", 1_800_000_000)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn unavailable_pump_search_leaves_symbols_unresolved() {
        let transport = scripted(vec![
            json(json!({"pairs": []})),
            JsonResponse {
                status_code: 500,
                body: String::new(),
            },
        ]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let load = adapter.load_symbol("timba");
        assert!(load.signal.is_none());
        assert_eq!(load.diagnostics, ["pump.fun search HTTP 500"]);
    }

    #[test]
    fn pump_search_failures_and_inexact_symbols_preserve_market_fallback() {
        for response in [
            json!([]),
            json!({"error":"unavailable"}),
            json!([{
                "symbol":"TIMBAX", "mint":"F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump"
            }]),
            json!([{"symbol":"TIMBA", "mint":"invalid"}]),
        ] {
            let transport = Transport {
                json: std::cell::RefCell::new(VecDeque::from([
                    JsonResponse {
                        status_code: 200,
                        body: json!({"pairs":[]}).to_string(),
                    },
                    JsonResponse {
                        status_code: 200,
                        body: response.to_string(),
                    },
                ])),
                post: std::cell::RefCell::new(VecDeque::new()),
                binary: std::cell::RefCell::new(VecDeque::new()),
            };
            let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
            assert!(adapter.load_symbol("timba").signal.is_none());
        }
    }

    #[test]
    fn evm_address_discovers_chain_and_keeps_card_without_candles() -> Result<(), String> {
        let address = "0x26449b21EaF982D252956e34E675634b8b15f990";
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: serde_json::json!({"pairs":[
                        {"chainId":"ethereum", "pairAddress":"unrelated",
                         "baseToken":{"address":"0x0000000000000000000000000000000000000001"},
                         "liquidity":{"usd":999999}},
                        {"chainId":"robinhood", "pairAddress":"pool",
                         "baseToken":{"address":address,"name":"Netanyahu","symbol":"BIBI"},
                         "priceUsd":"0.000003256", "liquidity":{"usd":5174}}
                    ]})
                    .to_string(),
                },
                JsonResponse {
                    status_code: 404,
                    body: "{}".to_owned(),
                },
            ])),
            post: std::cell::RefCell::new(VecDeque::new()),
            binary: std::cell::RefCell::new(VecDeque::new()),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let query = bot_core::token_signals::detect_signal_query(address).ok_or("missing query")?;
        let signal = adapter.load_query(&query).signal.ok_or("missing signal")?;
        assert_eq!(signal.token.chain_id, "robinhood");
        assert_eq!(signal.token.address, address.to_ascii_lowercase());
        assert_eq!(signal.pair.base_token.symbol, "BIBI");
        assert!(signal.candles.is_empty());
        assert!(
            adapter
                .render_period_photo(&signal, "24h", 1_800_000_000)
                .is_err()
        );
        let caption = bot_core::token_signals::format_signal_caption(&signal, 0);
        assert!(caption.contains("#ROBINHOOD"));
        assert!(caption.contains("https://dexscreener.com/robinhood/pool"));
        assert!(!caption.contains("solscan.io"));
        assert!(!caption.contains("etherscan.io"));
        assert!(adapter.cache.writes.iter().any(
            |(key, _)| key == &format!("token_signal:address:{}", address.to_ascii_lowercase())
        ));
        Ok(())
    }

    #[test]
    fn symbol_resolution_prefers_real_activity_over_a_high_lp_namesake() -> Result<(), String> {
        let official = "0xb095274743941e953c746f9c228da9c18bb6ec29";
        let namesake = "0x0000000000000000000000000000000000000001";
        let pair = |address: &str, chain: &str, liquidity: u64, volume: u64| {
            json!({
                "chainId": chain,
                "pairAddress": format!("pool-{address}"),
                "baseToken": {"address": address, "name": "Laptop", "symbol": "LAPTOP"},
                "liquidity": {"usd": liquidity},
                "volume": {"h24": volume},
                "priceUsd": "1"
            })
        };
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: json!({
                        "pairs": [
                            pair(namesake, "robinhood", 2_000_000, 1),
                            pair(official, "base", 100_000, 100_000)
                        ]
                    })
                    .to_string(),
                },
                JsonResponse {
                    status_code: 404,
                    body: "{}".to_owned(),
                },
            ])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let signal = adapter
            .load_symbol("$LAPTOP")
            .signal
            .ok_or("missing signal")?;
        assert_eq!(signal.token.chain_id, "base");
        assert_eq!(signal.token.address, official);
        Ok(())
    }

    #[test]
    fn ambiguous_symbol_candidates_keep_each_contract_for_selection() {
        let first = "0x0000000000000000000000000000000000000001";
        let second = "0x0000000000000000000000000000000000000002";
        let pair = |address: &str, chain: &str, volume: u64| {
            json!({
                "chainId": chain,
                "pairAddress": format!("pool-{address}"),
                "baseToken": {"address": address, "name": "Synthetic Laptop", "symbol": "LAPTOP"},
                "liquidity": {"usd": 1000},
                "volume": {"h24": volume},
                "priceUsd": "1"
            })
        };
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([JsonResponse {
                status_code: 200,
                body: json!({"pairs":[pair(first, "base", 20), pair(second, "robinhood", 10)]})
                    .to_string(),
            }])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let candidates = adapter.load_candidates(&SignalQuery::Symbol("laptop".to_owned()));
        assert_eq!(candidates.signals.len(), 2);
        assert_eq!(candidates.signals[0].token.address, first);
        assert_eq!(candidates.signals[1].token.address, second);
        assert!(
            candidates
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("ambiguous token symbol"))
        );
    }

    #[test]
    fn provider_slug_resolution_matches_token_name_and_keeps_chain() -> Result<(), String> {
        let address = "0xb095274743941e953c746f9c228da9c18bb6ec29";
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: json!({"pairs":[{
                        "chainId":"base",
                        "pairAddress":"pool",
                        "baseToken":{"address":address,"name":"Hunter Biden's Laptop","symbol":"LAPTOP"},
                        "liquidity":{"usd":1000},
                        "volume":{"h24":10},
                        "priceUsd":"1"
                    }]}).to_string(),
                },
                JsonResponse {
                    status_code: 404,
                    body: "{}".to_owned(),
                },
            ])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let load = adapter.load_query(&SignalQuery::Slug("hunter-biden-s-laptop".to_owned()));
        let signal = load.signal.ok_or("missing signal")?;
        assert_eq!(signal.token.chain_id, "base");
        assert_eq!(signal.pair.base_token.name, "Hunter Biden's Laptop");
        Ok(())
    }

    #[test]
    fn slug_candidates_preserve_distinct_contract_identities() {
        let first = "0x0000000000000000000000000000000000000001";
        let second = "0x0000000000000000000000000000000000000002";
        let pair = |address: &str, chain: &str, volume: u64| {
            json!({
                "chainId": chain,
                "pairAddress": format!("pool-{chain}"),
                "baseToken": {
                    "address": address,
                    "name": "Hunter Biden's Laptop",
                    "symbol": "LAPTOP"
                },
                "liquidity": {"usd": 1000},
                "volume": {"h24": volume},
                "priceUsd": "1"
            })
        };
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([JsonResponse {
                status_code: 200,
                body: json!({
                    "pairs": [
                        pair(first, "base", 20),
                        pair(second, "robinhood", 10)
                    ]
                })
                .to_string(),
            }])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let candidates =
            adapter.load_candidates(&SignalQuery::Slug("hunter-bidens-laptop".to_owned()));
        assert_eq!(candidates.signals.len(), 2);
        assert_eq!(candidates.signals[0].token.address, first);
        assert_eq!(candidates.signals[1].token.address, second);
        assert_eq!(candidates.signals[0].token.chain_id, "base");
        assert_eq!(candidates.signals[1].token.chain_id, "robinhood");
    }

    #[test]
    fn address_candidates_keep_the_resolved_contract() {
        let address = "0x0000000000000000000000000000000000000001";
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: json!([{
                        "chainId": "base",
                        "pairAddress": "pool",
                        "baseToken": {"address": address, "symbol": "LAPTOP"},
                        "liquidity": {"usd": 1000},
                        "priceUsd": "1"
                    }])
                    .to_string(),
                },
                JsonResponse {
                    status_code: 404,
                    body: "{}".to_owned(),
                },
            ])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let token = TokenAddress {
            chain_id: "base".to_owned(),
            network: "base".to_owned(),
            tag: "BASE".to_owned(),
            address: address.to_owned(),
        };
        let candidates = adapter.load_candidates(&SignalQuery::Address(token));
        assert_eq!(candidates.signals.len(), 1);
        assert_eq!(candidates.signals[0].token.address, address);
    }

    #[test]
    fn single_symbol_candidate_loads_history_and_duplicate_pairs_fall_back() {
        let address = "0x0000000000000000000000000000000000000001";
        let pair = |pool: &str, volume: u64| {
            json!({
                "chainId": "base",
                "pairAddress": pool,
                "baseToken": {"address": address, "symbol": "SYN"},
                "liquidity": {"usd": 1000},
                "volume": {"h24": volume},
                "priceUsd": "1"
            })
        };
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: json!({"pairs": [pair("history", 20)]}).to_string(),
                },
                JsonResponse {
                    status_code: 200,
                    body: json!({
                        "data": {"attributes": {"ohlcv_list": [[1, 1, 2, 0.8, 1.5]]}}
                    })
                    .to_string(),
                },
            ])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let candidates = adapter.load_candidates(&SignalQuery::Symbol("syn".to_owned()));
        assert_eq!(candidates.signals.len(), 1);
        assert_eq!(candidates.signals[0].pair.pair_address, "history");
        assert!(!candidates.signals[0].candles.is_empty());

        let pair = |pool: &str, volume: u64| {
            json!({
                "chainId": "solana",
                "pairAddress": pool,
                "baseToken": {"address": "synthetic-mint", "symbol": "SYN"},
                "liquidity": {"usd": 1000},
                "volume": {"h24": volume},
                "priceUsd": "1"
            })
        };
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: json!({"pairs": [pair("primary", 20), pair("alternate", 10)]})
                        .to_string(),
                },
                JsonResponse {
                    status_code: 404,
                    body: "{}".to_owned(),
                },
                JsonResponse {
                    status_code: 404,
                    body: "{}".to_owned(),
                },
            ])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let candidates = adapter.load_candidates(&SignalQuery::Symbol("syn".to_owned()));
        assert_eq!(candidates.signals.len(), 1);
        assert_eq!(candidates.signals[0].pair.pair_address, "primary");
        assert_eq!(candidates.signals[0].token.address, "synthetic-mint");
    }

    #[test]
    fn slug_load_reports_when_dex_has_no_exact_name_match() {
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([JsonResponse {
                status_code: 200,
                body: json!({"pairs": [{
                    "chainId": "base",
                    "pairAddress": "pool",
                    "baseToken": {"address": "0x0000000000000000000000000000000000000001", "name": "Other", "symbol": "OTHER"}
                }]}).to_string(),
            }])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let load = adapter.load_slug("missing-token");
        assert!(load.signal.is_none());
        assert!(
            load.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("no DexScreener token matched slug"))
        );
    }

    #[test]
    fn chart_truncates_long_titles_without_losing_the_quote() {
        let signal = TokenSignal {
            token: TokenAddress {
                chain_id: "base".to_owned(),
                network: "base".to_owned(),
                tag: "BASE".to_owned(),
                address: "0x0000000000000000000000000000000000000001".to_owned(),
            },
            pair: bot_core::token_signals::TokenPair {
                base_token: bot_core::token_signals::PairToken {
                    symbol: "SYNTHETIC-LONG-TOKEN-SYMBOL".to_owned(),
                    ..Default::default()
                },
                price_usd: json!(1.2345),
                ..Default::default()
            },
            candles: vec![vec![1.0, 1.0, 2.0, 0.8, 1.5]],
            supply: None,
            token_image_url: None,
            socials: BTreeMap::new(),
            pump: None,
        };
        let chart = render_signal_chart(&signal, 420, 300);
        assert!(chart.is_ok_and(|png| png.starts_with(b"\x89PNG")));
    }

    #[test]
    fn address_load_uses_first_ranked_pair_with_candles_and_compatible_cache_keys() {
        let pair = |address: &str, liquidity: i64| {
            serde_json::json!({
                "chainId":"ethereum",
                "pairAddress":address,
                "baseToken":{"address":"0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48","symbol":"USDC"},
                "liquidity":{"usd":liquidity},
                "volume":{"h24":1}
            })
        };
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: serde_json::json!([pair("bad", 1_000), pair("good", 100)]).to_string(),
                },
                JsonResponse {
                    status_code: 200,
                    body: serde_json::json!({"data":{"attributes":{"ohlcv_list":[]}}})
                        .to_string(),
                },
                JsonResponse {
                    status_code: 200,
                    body: serde_json::json!({"data":{"attributes":{"ohlcv_list":[[1,1,2,0.8,1.5,1000]]}}})
                        .to_string(),
                },
            ])),
            post: std::cell::RefCell::new(VecDeque::new()),
            binary: std::cell::RefCell::new(VecDeque::new()),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let token = TokenAddress {
            chain_id: "ethereum".to_owned(),
            network: "eth".to_owned(),
            tag: "ETH".to_owned(),
            address: "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".to_owned(),
        };
        let load = adapter.load_token(&token);
        assert_eq!(
            load.signal
                .as_ref()
                .map(|signal| signal.pair.pair_address.as_str()),
            Some("good")
        );
        assert!(
            adapter.cache.writes.iter().any(|(key, ttl)| {
                key.starts_with("token_signal:pairs:ethereum:") && *ttl == 30
            })
        );
        assert!(
            adapter
                .cache
                .writes
                .iter()
                .any(|(key, ttl)| key == "token_signal:ohlcv:eth:good:hour" && *ttl == 60)
        );
    }

    #[test]
    fn address_load_ignores_provider_tag_aliases_when_contract_matches() {
        let address = "0xb095274743941e953c746f9c228da9c18bb6ec29";
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: json!([{
                        "chainId":"base",
                        "pairAddress":"pool",
                        "baseToken":{"address":address,"symbol":"LAPTOP"},
                        "liquidity":{"usd":1000},
                        "priceUsd":"1"
                    }])
                    .to_string(),
                },
                JsonResponse {
                    status_code: 404,
                    body: "{}".to_owned(),
                },
            ])),
            post: Default::default(),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let token = TokenAddress {
            chain_id: "base".to_owned(),
            network: "base".to_owned(),
            tag: "ETH".to_owned(),
            address: address.to_owned(),
        };
        assert!(adapter.load_token(&token).signal.is_some());
    }

    #[test]
    fn symbol_enrichment_writes_python_readable_extracted_cache_values() {
        let mint = "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump";
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                JsonResponse {
                    status_code: 200,
                    body: serde_json::json!({
                        "pairs":[{
                            "chainId":"solana",
                            "pairAddress":"pair1",
                            "baseToken":{"address":mint,"symbol":"SYN"},
                            "liquidity":{"usd":1000},
                            "volume":{"h24":10}
                        }]
                    })
                    .to_string(),
                },
                JsonResponse {
                    status_code: 200,
                    body: serde_json::json!({
                        "data":{"attributes":{"ohlcv_list":[[1,1,2,0.8,1.5,1000]]}}
                    })
                    .to_string(),
                },
                JsonResponse {
                    status_code: 200,
                    body: serde_json::json!({
                        "image_uri":"https://example.test/token.png",
                        "total_supply":999000000
                    })
                    .to_string(),
                },
            ])),
            post: std::cell::RefCell::new(VecDeque::from([JsonResponse {
                status_code: 200,
                body: serde_json::json!({
                    "result":{"value":{"uiAmountString":"999.5"}}
                })
                .to_string(),
            }])),
            binary: std::cell::RefCell::new(VecDeque::new()),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let load = adapter.load_query(&SignalQuery::Symbol("syn".to_owned()));
        assert!(load.signal.is_some());

        let decode = |key: &str| {
            adapter
                .cache
                .values
                .get(key)
                .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        };
        assert!(decode("token_signal:search:syn").is_some_and(|value| value.is_array()));
        assert!(
            decode("token_signal:ohlcv:solana:pair1:hour").is_some_and(|value| value.is_array())
        );
        assert_eq!(
            decode(&format!("token_signal:supply:{mint}")).and_then(|value| value.as_f64()),
            Some(999.5)
        );
        assert!(
            decode(&format!("token_signal:pump:{mint}")).is_some_and(|value| value.is_object())
        );
    }

    #[test]
    fn missing_history_does_not_substitute_a_token_image() {
        let mut jpeg = std::io::Cursor::new(Vec::new());
        let source = image::RgbImage::from_pixel(4, 4, image::Rgb([255, 0, 0]));
        let encoded =
            image::DynamicImage::ImageRgb8(source).write_to(&mut jpeg, image::ImageFormat::Jpeg);
        assert!(encoded.is_ok());
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::new()),
            post: std::cell::RefCell::new(VecDeque::new()),
            binary: std::cell::RefCell::new(VecDeque::from([BinaryResponse {
                status_code: 200,
                content_type: "image/jpeg".to_owned(),
                body: jpeg.into_inner(),
            }])),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let signal = TokenSignal {
            token: TokenAddress {
                chain_id: "solana".to_owned(),
                network: "solana".to_owned(),
                tag: "SOL".to_owned(),
                address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            },
            pair: bot_core::token_signals::TokenPair::default(),
            candles: Vec::new(),
            supply: None,
            token_image_url: Some("https://example.test/token.jpg".to_owned()),
            socials: BTreeMap::new(),
            pump: None,
        };
        let photo = adapter.render_period_photo(&signal, "24h", 1_800_000_000);
        assert!(photo.is_err());
        assert_eq!(adapter.transport.binary.borrow().len(), 1);
    }

    #[test]
    fn chart_renderer_rejects_missing_data_and_renders_usable_data() {
        let mut signal = TokenSignal {
            token: TokenAddress {
                chain_id: "solana".to_owned(),
                network: "solana".to_owned(),
                tag: "SOL".to_owned(),
                address: "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump".to_owned(),
            },
            pair: bot_core::token_signals::TokenPair {
                base_token: bot_core::token_signals::PairToken {
                    symbol: "TEST".to_owned(),
                    ..bot_core::token_signals::PairToken::default()
                },
                ..bot_core::token_signals::TokenPair::default()
            },
            candles: Vec::new(),
            supply: None,
            token_image_url: None,
            socials: BTreeMap::new(),
            pump: None,
        };
        let blank = render_signal_chart(&signal, 420, 300);
        assert!(blank.is_err());
        signal.candles = vec![
            vec![1.0, 1.0, 2.0, 0.8, 1.5],
            vec![2.0, 1.5, 2.5, 1.2, 1.3],
            vec![3.0, 1.3, 1.8, 1.0, 1.7],
            vec![4.0, 1.7, 2.2, 1.4, 2.0],
            vec![5.0, 2.0, 2.4, 1.8, 2.1],
        ];
        let chart = render_signal_chart(&signal, 420, 300);
        assert!(chart.as_ref().is_ok_and(|png| png.starts_with(b"\x89PNG")));
        assert_ne!(blank, chart);
    }

    fn json(body: serde_json::Value) -> JsonResponse {
        JsonResponse {
            status_code: 200,
            body: body.to_string(),
        }
    }

    fn scripted(responses: Vec<JsonResponse>) -> Transport {
        Transport {
            json: std::cell::RefCell::new(VecDeque::from(responses)),
            post: Default::default(),
            binary: Default::default(),
        }
    }

    #[test]
    fn evm_identities_ignore_address_case_but_other_chains_do_not() {
        let token = |chain: &str, address: &str| TokenAddress {
            chain_id: chain.to_owned(),
            network: chain.to_owned(),
            tag: String::new(),
            address: address.to_owned(),
        };
        assert!(super::same_token_identity(
            &token("base", "0xAbC0000000000000000000000000000000000001"),
            &token("BASE", "0xabc0000000000000000000000000000000000001"),
        ));
        assert!(!super::same_token_identity(
            &token("solana", "MintAbc"),
            &token("solana", "mintabc"),
        ));
        assert!(!super::same_token_identity(
            &token("base", "0xabc0000000000000000000000000000000000001"),
            &token("robinhood", "0xabc0000000000000000000000000000000000001"),
        ));
    }

    #[test]
    fn solana_address_loads_match_the_exact_case_sensitive_mint() -> Result<(), String> {
        let mint = "So1anaSyntheticMint11111111111111111111111";
        let pair = |address: &str, pool: &str| {
            json!({
                "chainId": "solana", "pairAddress": pool,
                "baseToken": {"address": address, "symbol": "SYN"},
                "liquidity": {"usd": 1000}, "priceUsd": "1"
            })
        };
        let transport = scripted(vec![
            json(json!([
                pair(&mint.to_ascii_lowercase(), "lowercase"),
                pair(mint, "exact")
            ])),
            JsonResponse {
                status_code: 404,
                body: "{}".to_owned(),
            },
        ]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let token = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: mint.to_owned(),
        };
        let load = adapter.load_token(&token);
        let signal = load.signal.ok_or("missing signal")?;
        assert_eq!(signal.token.address, mint);
        assert_eq!(signal.pair.pair_address, "exact");
        assert!(signal.candles.is_empty());
        Ok(())
    }

    #[test]
    fn pump_metadata_without_a_solana_mint_or_symbol_is_not_a_signal() {
        let mint = "F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump";
        let token = TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: mint.to_owned(),
        };
        let transport = scripted(vec![
            json(json!([])),
            json(json!({"mint": mint, "symbol": "  ", "usd_market_cap": 5140.0})),
        ]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        assert!(adapter.load_token(&token).signal.is_none());

        for coin_mint in ["$TIMBA", "0x0000000000000000000000000000000000000001"] {
            let transport = scripted(vec![
                json(json!({"pairs": []})),
                json(json!([{"symbol": "TIMBA", "mint": coin_mint, "usd_market_cap": 1}])),
            ]);
            let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
            assert!(adapter.load_symbol("timba").signal.is_none(), "{coin_mint}");
        }
    }

    #[test]
    fn ranked_fallbacks_skip_same_token_pairs_without_a_pool_address() -> Result<(), String> {
        let pair = |pool: &str, volume: u64| {
            json!({
                "chainId": "solana", "pairAddress": pool,
                "baseToken": {"address": "synthetic-mint", "symbol": "SYN"},
                "liquidity": {"usd": 1000}, "volume": {"h24": volume}, "priceUsd": "1"
            })
        };
        let transport = scripted(vec![
            json(json!({"pairs": [pair("primary", 20), pair("", 10)]})),
            JsonResponse {
                status_code: 404,
                body: "{}".to_owned(),
            },
        ]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let signal = adapter.load_symbol("syn").signal.ok_or("missing signal")?;
        assert_eq!(signal.pair.pair_address, "primary");
        assert!(signal.candles.is_empty());
        assert!(adapter.transport.json.borrow().is_empty());
        Ok(())
    }

    #[test]
    fn slug_candidates_match_symbols_and_report_no_match_as_empty() {
        let address = "0x0000000000000000000000000000000000000001";
        let transport = scripted(vec![
            json(json!({"pairs": [{
                "chainId": "base", "pairAddress": "pool",
                "baseToken": {"address": address, "name": "Unrelated Name", "symbol": "LAPTOP"},
                "liquidity": {"usd": 1000}, "volume": {"h24": 10}, "priceUsd": "1"
            }]})),
            json(json!({"data": {"attributes": {"ohlcv_list": [[1, 1, 2, 0.8, 1.5]]}}})),
        ]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let candidates = adapter.load_candidates(&SignalQuery::Slug("laptop".to_owned()));
        assert_eq!(candidates.signals.len(), 1);
        assert_eq!(candidates.signals[0].token.address, address);
        assert!(!candidates.signals[0].candles.is_empty());

        let transport = scripted(vec![json(json!({"pairs": [{
            "chainId": "base", "pairAddress": "pool",
            "baseToken": {"address": address, "name": "Other", "symbol": "OTHER"}
        }]}))]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let candidates = adapter.load_candidates(&SignalQuery::Slug("laptop".to_owned()));
        assert!(candidates.signals.is_empty());
    }

    #[test]
    fn charts_render_without_text_when_no_system_font_is_available() {
        let pair = bot_core::token_signals::TokenPair {
            price_usd: json!("1.5"),
            ..bot_core::token_signals::TokenPair::default()
        };
        let candles = vec![vec![1.0, 1.0, 2.0, 0.8, 1.5], vec![2.0, 1.5, 2.5, 1.2, 1.3]];
        let no_fonts = super::ChartFonts {
            bold: None,
            regular: None,
        };
        let plain = super::render_price_chart_with_fonts(
            &pair,
            &candles,
            Some("SYN (1d)"),
            Some("USD"),
            (640, 480),
            &no_fonts,
        );
        assert!(plain.as_ref().is_ok_and(|png| png.starts_with(b"\x89PNG")));
        let decoded = plain
            .as_deref()
            .ok()
            .and_then(|png| image::load_from_memory(png).ok())
            .map(|image| (image.width(), image.height()));
        assert_eq!(decoded, Some((640, 480)));
    }

    #[test]
    fn redis_json_cache_reads_token_signal_values()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::redis_connection::{RedisEndpoint, test_support::read_command};
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(
            move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
                assert_eq!(
                    read_command(&mut stream)?,
                    ["GET", "token_signal:state:synthetic"]
                );
                stream.write_all(b"$4\r\ncard\r\n")?;
                assert_eq!(
                    read_command(&mut stream)?,
                    ["GET", "token_signal:state:missing"]
                );
                stream.write_all(b"$-1\r\n")?;
                Ok(())
            },
        );
        let mut cache = crate::redis_json_cache::RedisJsonCache::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        })?;
        assert_eq!(
            TokenSignalCache::get(&mut cache, "token_signal:state:synthetic")?,
            Some("card".to_owned())
        );
        assert_eq!(
            TokenSignalCache::get(&mut cache, "token_signal:state:missing")?,
            None
        );
        assert!(server.join().is_ok_and(|result| result.is_ok()));
        Ok(())
    }

    fn solana(address: &str) -> TokenAddress {
        TokenAddress {
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: address.to_owned(),
        }
    }

    fn not_found() -> JsonResponse {
        JsonResponse {
            status_code: 404,
            body: "{}".to_owned(),
        }
    }

    #[test]
    fn provider_cache_and_response_failures_are_diagnostic() {
        let token = solana("So1anaSyntheticMint11111111111111111111111");
        let key = format!("token_signal:pairs:solana:{}", token.address);
        let scenarios: Vec<(Cache, Vec<JsonResponse>, &str)> = vec![
            (
                Cache {
                    values: BTreeMap::from([(key.clone(), "not json".to_owned())]),
                    ..Cache::default()
                },
                vec![json(json!([]))],
                "invalid DexScreener pairs cache",
            ),
            (
                Cache {
                    fail_get: true,
                    ..Cache::default()
                },
                vec![json(json!([]))],
                "could not read DexScreener pairs cache",
            ),
            (
                Cache {
                    fail_set: true,
                    ..Cache::default()
                },
                vec![json(json!([]))],
                "could not write DexScreener pairs cache",
            ),
            (
                Cache::default(),
                vec![JsonResponse {
                    status_code: 500,
                    body: String::new(),
                }],
                "DexScreener pairs HTTP 500",
            ),
            (
                Cache::default(),
                Vec::new(),
                "DexScreener pairs: unexpected request",
            ),
            (
                Cache::default(),
                vec![JsonResponse {
                    status_code: 200,
                    body: "not json".to_owned(),
                }],
                "invalid DexScreener pairs response",
            ),
        ];
        for (cache, responses, expected) in scenarios {
            let mut adapter = TokenSignalAdapter::new(scripted(responses), cache);
            let load = adapter.load_token(&token);
            assert!(load.signal.is_none());
            assert!(
                load.diagnostics
                    .iter()
                    .any(|entry| entry.starts_with(expected)),
                "{expected}: {:?}",
                load.diagnostics
            );
        }

        let mut adapter =
            TokenSignalAdapter::new(scripted(vec![json(json!({}))]), Cache::default());
        let query = bot_core::token_signals::detect_signal_query(
            "0x0000000000000000000000000000000000000001",
        );
        let load = adapter.load_query(&query.unwrap_or(SignalQuery::Symbol(String::new())));
        assert!(load.signal.is_none());
        assert_eq!(
            load.diagnostics,
            ["DexScreener address response did not contain the expected value"]
        );
    }

    #[test]
    fn pump_market_caps_and_supply_accept_alternate_provider_fields() -> Result<(), String> {
        let mint = "F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump";
        let other = "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump";
        let coin = |mint: &str, cap: u64| json!({"mint": mint, "name": "TIMBA", "symbol": "TIMBA", "market_cap_usd": cap});
        let transport = Transport {
            json: std::cell::RefCell::new(VecDeque::from([
                json(json!({"pairs": []})),
                json(json!([coin(other, 100), coin(mint, 5000)])),
            ])),
            post: std::cell::RefCell::new(VecDeque::from([json(
                json!({"result": {"value": {"uiAmount": 1000}}}),
            )])),
            binary: Default::default(),
        };
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let signal = adapter.load_symbol("timba").signal.ok_or("pump signal")?;
        assert_eq!(signal.token.address, mint);
        assert_eq!(signal.pair.market_cap, json!(5000.0));
        assert_eq!(signal.supply, Some(1000.0));
        assert_eq!(signal.pair.price_usd, json!(5.0));
        Ok(())
    }

    #[test]
    fn listed_pump_tokens_fall_back_to_pump_supply_when_rpc_fails() -> Result<(), String> {
        let mint = "F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump";
        let transport = scripted(vec![
            json(json!([{
                "chainId": "solana", "pairAddress": "pool",
                "baseToken": {"address": mint, "symbol": "TIMBA"},
                "liquidity": {"usd": 1000}, "priceUsd": "1"
            }])),
            not_found(),
            json(json!({"mint": mint, "symbol": "TIMBA", "total_supply": 2_000_000_000_u64})),
        ]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let load = adapter.load_token(&solana(mint));
        let signal = load.signal.ok_or("listed signal")?;
        assert_eq!(signal.supply, Some(2_000.0));
        assert!(signal.pump.is_some());
        assert!(
            load.diagnostics
                .iter()
                .any(|entry| entry.contains("synthetic supply unavailable"))
        );
        Ok(())
    }

    #[test]
    fn address_pairs_rank_by_liquidity_then_volume_and_skip_poolless_pairs() -> Result<(), String> {
        let mint = "So1anaSyntheticMint11111111111111111111111";
        let pair = |pool: &str, volume: u64| {
            json!({
                "chainId": "solana", "pairAddress": pool,
                "baseToken": {"address": mint, "symbol": "SYN"},
                "liquidity": {"usd": 1000}, "volume": {"h24": volume}, "priceUsd": "1"
            })
        };
        let transport = scripted(vec![
            json(json!([pair("", 50), pair("quiet", 10), pair("busy", 30)])),
            not_found(),
            json(json!({"data": {"attributes": {"ohlcv_list": [[1, 1, 2, 0.5, 1.5]]}}})),
        ]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let signal = adapter.load_token(&solana(mint)).signal.ok_or("signal")?;
        assert_eq!(signal.pair.pair_address, "quiet");
        assert!(!signal.candles.is_empty());
        assert!(
            adapter
                .cache
                .writes
                .iter()
                .any(|(key, _)| key.contains(":quiet:"))
        );
        // The busier pool was tried first and had no history.
        assert!(adapter.transport.json.borrow().is_empty());
        Ok(())
    }

    #[test]
    fn symbol_ranking_skips_unsupported_chains_and_poolless_initial_pairs() -> Result<(), String> {
        let transport = scripted(vec![json(json!({"pairs": [
            {"chainId": "solana", "pairAddress": "",
             "baseToken": {"address": "synthetic-mint", "symbol": "SYN"},
             "liquidity": {"usd": 1000}, "volume": {"h24": 20}, "priceUsd": "1"},
            {"chainId": "unsupported", "pairAddress": "elsewhere",
             "baseToken": {"address": "not-evm", "symbol": "SYN"},
             "liquidity": {"usd": 1000}, "volume": {"h24": 10}, "priceUsd": "1"}
        ]}))]);
        let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
        let signal = adapter.load_symbol("syn").signal.ok_or("signal")?;
        assert_eq!(signal.token.address, "synthetic-mint");
        assert!(signal.pair.pair_address.is_empty());
        assert!(signal.candles.is_empty());
        Ok(())
    }

    #[test]
    fn signal_state_round_trips_clears_and_reports_cache_failures() {
        let state = bot_core::token_signals::SignalState {
            chart_period: None,
            chat_id: "synthetic-chat".to_owned(),
            message_id: 2,
            source_message_id: 1,
            requester_id: "synthetic-user".to_owned(),
            chain_id: "solana".to_owned(),
            network: "solana".to_owned(),
            tag: "SOL".to_owned(),
            address: "synthetic-mint".to_owned(),
            last_refresh_at: Some(1_800_000_000),
        };
        let mut adapter = TokenSignalAdapter::new(scripted(Vec::new()), Cache::default());
        assert_eq!(adapter.save_state("synthetic", &state), Ok(()));
        assert_eq!(adapter.load_state("synthetic"), Ok(Some(state.clone())));
        assert_eq!(adapter.clear_state("synthetic"), Ok(()));
        assert_eq!(adapter.load_state("synthetic"), Ok(None));
        assert_eq!(adapter.cache.writes.last().map(|write| write.1), Some(1));

        adapter.cache.fail_set = true;
        assert_eq!(
            adapter.clear_state("synthetic"),
            Err("synthetic cache write failure".to_owned())
        );
        assert_eq!(
            adapter.save_state("synthetic", &state),
            Err("synthetic cache write failure".to_owned())
        );
    }

    #[test]
    fn charts_reject_tiny_canvases_and_sort_reversed_candles() -> Result<(), String> {
        let mut signal = TokenSignal {
            token: solana("synthetic-mint"),
            pair: Default::default(),
            candles: vec![
                vec![3.0, 1.3, 1.8, 1.0, 1.7],
                vec![2.0, 1.5, 2.5, 1.2, 1.3],
                vec![1.0, 1.0, 2.0, 0.8, 1.5],
            ],
            supply: None,
            token_image_url: None,
            socials: BTreeMap::new(),
            pump: None,
        };
        assert_eq!(
            render_signal_chart(&signal, 100, 100),
            Err("token chart dimensions are too small".to_owned())
        );
        let reversed = render_signal_chart(&signal, 420, 300)?;
        signal.candles.reverse();
        assert_eq!(render_signal_chart(&signal, 420, 300)?, reversed);
        Ok(())
    }

    #[test]
    fn reqwest_transport_prefixes_connection_failures_with_the_operation() -> Result<(), String> {
        let closed = TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .ok()
            .ok_or("closed port")?;
        let transport = ReqwestTokenSignalTransport::new()?;
        let url = format!("http://{closed}/token");
        assert!(
            transport
                .get_json(&url, &[])
                .is_err_and(|error| error.starts_with("token-signal GET failed: "))
        );
        assert!(
            transport
                .post_json(&url, &json!({}))
                .is_err_and(|error| error.starts_with("token-signal POST failed: "))
        );
        assert!(
            transport
                .get_binary(&url)
                .is_err_and(|error| error.starts_with("token image GET failed: "))
        );
        Ok(())
    }
}
