//! DexScreener/GeckoTerminal token-card adapter and PNG renderer.

use std::io::Cursor;
use std::time::Duration;

use ab_glyph::FontArc;
use bot_core::token_signals::{
    PumpMetadata, SIGNAL_STATE_TTL_SECONDS, SignalQuery, SignalState, TokenAddress, TokenPair,
    TokenSignal, choose_symbol_pair, format_money, has_usable_chart, pair_rank, signal_state_key,
    token_from_pair, token_image_url, token_socials,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage, imageops::FilterType};
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
        Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
            .build()
            .map(|client| Self { client })
            .map_err(|error| format!("could not build token-signal HTTP client: {error}"))
    }
}

impl TokenSignalTransport for ReqwestTokenSignalTransport {
    fn get_json(&self, url: &str, query: &[(&str, String)]) -> Result<JsonResponse, String> {
        let response = self
            .client
            .get(url)
            .query(query)
            .send()
            .map_err(|error| format!("token-signal GET failed: {error}"))?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| JsonResponse { status_code, body })
            .map_err(|error| format!("token-signal response read failed: {error}"))
    }

    fn post_json(&self, url: &str, body: &Value) -> Result<JsonResponse, String> {
        let response = self
            .client
            .post(url)
            .json(body)
            .send()
            .map_err(|error| format!("token-signal POST failed: {error}"))?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| JsonResponse { status_code, body })
            .map_err(|error| format!("token-signal response read failed: {error}"))
    }

    fn get_binary(&self, url: &str) -> Result<BinaryResponse, String> {
        let response = self
            .client
            .get(url)
            .send()
            .map_err(|error| format!("token image GET failed: {error}"))?;
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
            .map_err(|error| format!("token image response read failed: {error}"))
    }
}

pub trait TokenSignalCache {
    type Error: std::fmt::Display;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error>;

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error>;
}

impl TokenSignalCache for RedisJsonCache {
    type Error = RedisJsonCacheError;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        RedisJsonCache::get(self, key)
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
        RedisJsonCache::set(self, key, value, Some(ttl_seconds)).map(|_stored| ())
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
        fetch: impl FnOnce(&Transport) -> Result<JsonResponse, String>,
        extract: impl FnOnce(Value) -> Option<Value>,
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
        match serde_json::to_string(&value) {
            Ok(encoded) => {
                if let Err(error) = self.cache.set(key, &encoded, ttl_seconds) {
                    diagnostics.push(format!("could not write {label} cache {key}: {error}"));
                }
            }
            Err(error) => diagnostics.push(format!("could not encode {label}: {error}")),
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
            |transport| transport.get_json(&url, &[]),
            Some,
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
            |transport| {
                transport.get_json(
                    "https://api.dexscreener.com/latest/dex/search",
                    &[("q", normalized.clone())],
                )
            },
            |value| value.get("pairs").cloned(),
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
            |transport| {
                transport.get_json(
                    &url,
                    &[
                        ("aggregate", "4".to_owned()),
                        ("limit", "60".to_owned()),
                        ("currency", "usd".to_owned()),
                    ],
                )
            },
            |value| value.pointer("/data/attributes/ohlcv_list").cloned(),
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
        if token.chain_id != "solana" || !token.address.ends_with("pump") {
            return None;
        }
        let key = format!("token_signal:pump:{}", token.address);
        let url = format!("https://frontend-api-v3.pump.fun/coins/{}", token.address);
        self.cached_json(
            &key,
            60,
            "pump.fun metadata",
            |transport| transport.get_json(&url, &[]),
            Some,
            diagnostics,
        )
        .and_then(|value| serde_json::from_value(value).ok())
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
            |transport| transport.post_json("https://api.mainnet-beta.solana.com", &body),
            |value| {
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
            SignalQuery::Address(token) => self.load_token(token),
            SignalQuery::Symbol(symbol) => self.load_symbol(symbol),
        }
    }

    pub fn load_token(&mut self, token: &TokenAddress) -> TokenSignalLoad {
        let mut diagnostics = Vec::new();
        let mut pairs = self.pairs(token, &mut diagnostics);
        pairs.sort_by(|left, right| {
            let left = pair_rank(left);
            let right = pair_rank(right);
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| right.1.total_cmp(&left.1))
        });
        let fallback = pairs.first().cloned();
        for pair in pairs {
            if pair.pair_address.is_empty() {
                continue;
            }
            let candles = self.candles(token, &pair.pair_address, &mut diagnostics);
            if !candles.is_empty() {
                return TokenSignalLoad {
                    signal: Some(self.enrich(token.clone(), pair, candles, &mut diagnostics)),
                    diagnostics,
                };
            }
        }
        TokenSignalLoad {
            signal: fallback
                .map(|pair| self.enrich(token.clone(), pair, Vec::new(), &mut diagnostics)),
            diagnostics,
        }
    }

    pub fn load_symbol(&mut self, symbol: &str) -> TokenSignalLoad {
        let mut diagnostics = Vec::new();
        let mut pairs = self.search_pairs(symbol, &mut diagnostics);
        let normalized = symbol.trim_start_matches('$').to_ascii_lowercase();
        pairs.sort_by(|left, right| {
            let left_exact = left.base_token.symbol.to_ascii_lowercase() == normalized;
            let right_exact = right.base_token.symbol.to_ascii_lowercase() == normalized;
            right_exact.cmp(&left_exact).then_with(|| {
                let left = pair_rank(left);
                let right = pair_rank(right);
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| right.1.total_cmp(&left.1))
            })
        });
        let Some(initial_pair) = choose_symbol_pair(&pairs, symbol) else {
            return TokenSignalLoad {
                signal: None,
                diagnostics,
            };
        };
        let Some(initial_token) = token_from_pair(&initial_pair) else {
            return TokenSignalLoad {
                signal: None,
                diagnostics,
            };
        };
        if !initial_pair.pair_address.is_empty() {
            let candles =
                self.candles(&initial_token, &initial_pair.pair_address, &mut diagnostics);
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
        for pair in pairs {
            let Some(token) = token_from_pair(&pair) else {
                continue;
            };
            if pair.pair_address.is_empty() {
                continue;
            }
            let candles = self.candles(&token, &pair.pair_address, &mut diagnostics);
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

    pub fn render_photo(&self, signal: &TokenSignal) -> Result<Vec<u8>, String> {
        if has_usable_chart(signal) {
            return render_signal_chart(signal, 1_280, 900);
        }
        if let Some(url) = signal.token_image_url.as_deref()
            && let Ok(image) = self.download_image(url)
        {
            return Ok(image);
        }
        render_signal_chart(signal, 1_280, 900)
    }

    fn download_image(&self, url: &str) -> Result<Vec<u8>, String> {
        let response = self.transport.get_binary(url)?;
        if !(200..300).contains(&response.status_code)
            || !response.content_type.starts_with("image/")
            || response.body.is_empty()
        {
            return Err("token image response was not a usable image".to_owned());
        }
        let image = image::load_from_memory(&response.body)
            .map_err(|error| format!("token image decode failed: {error}"))?;
        let image = image.resize(1_280, 900, FilterType::Lanczos3).to_rgb8();
        encode_png(image)
    }

    pub fn load_state(&mut self, signal_id: &str) -> Result<Option<SignalState>, String> {
        let key = signal_state_key(signal_id);
        self.cache
            .get(&key)
            .map_err(|error| error.to_string())?
            .map(|encoded| {
                serde_json::from_str(&encoded)
                    .map_err(|error| format!("invalid token-signal state {key}: {error}"))
            })
            .transpose()
    }

    pub fn save_state(&mut self, signal_id: &str, state: &SignalState) -> Result<(), String> {
        let key = signal_state_key(signal_id);
        let encoded = serde_json::to_string(state)
            .map_err(|error| format!("could not encode token-signal state: {error}"))?;
        self.cache
            .set(&key, &encoded, SIGNAL_STATE_TTL_SECONDS)
            .map_err(|error| error.to_string())
    }
}

fn flexible_number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn chart_font(bold: bool) -> Option<FontArc> {
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
    if width < 120 || height < 120 {
        return Err("token chart dimensions are too small".to_owned());
    }
    let mut image = RgbImage::from_pixel(width, height, Rgb([7, 9, 18]));
    let left = 56_i32;
    let right = i32::try_from(width).map_err(|_| "chart width is too large")? - 92;
    let top = 74_i32;
    let bottom = i32::try_from(height).map_err(|_| "chart height is too large")? - 82;
    let price = flexible_number(&signal.pair.price_usd).unwrap_or(0.0);
    let market_cap = flexible_number(&signal.pair.market_cap)
        .filter(|market_cap| *market_cap != 0.0)
        .or_else(|| flexible_number(&signal.pair.fdv))
        .unwrap_or(0.0);
    let symbol = if signal.pair.base_token.symbol.is_empty() {
        "TOKEN".to_owned()
    } else {
        signal.pair.base_token.symbol.to_ascii_uppercase()
    };
    let price_text = format_money(price, true);
    if let Some(font) = chart_font(true) {
        let change = flexible_number(&signal.pair.price_change.h24).unwrap_or(0.0);
        let sign = if change >= 0.0 { "+" } else { "" };
        let title = format!(
            "{symbol} (4H) Price: {price_text} ({sign}{change:.1}%) • MC: {}",
            format_money(market_cap, false)
        );
        draw_text_mut(
            &mut image,
            Rgb([220, 231, 244]),
            24,
            22,
            24.0,
            &font,
            &title,
        );
    }
    for index in 0..6 {
        let y = top + (bottom - top) * index / 5;
        draw_line(&mut image, left, y, right, y, Rgb([17, 24, 39]));
    }
    for index in 0..7 {
        let x = left + (right - left) * index / 6;
        draw_line(&mut image, x, top, x, bottom, Rgb([17, 24, 39]));
    }
    let mut candles = signal
        .candles
        .iter()
        .filter(|candle| candle.len() >= 5)
        .cloned()
        .collect::<Vec<_>>();
    if candles
        .first()
        .zip(candles.last())
        .is_some_and(|(first, last)| first[0] > last[0])
    {
        candles.reverse();
    }
    if candles.is_empty() {
        if let Some(font) = chart_font(true) {
            draw_text_mut(
                &mut image,
                Rgb([141, 161, 182]),
                i32::try_from(width / 2).unwrap_or(0).saturating_sub(120),
                i32::try_from(height / 2).unwrap_or(0),
                32.0,
                &font,
                "no chart data",
            );
        }
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
            range = high.abs().max(1.0) * 0.02;
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
        let current_y = y_for(current);
        draw_line(
            &mut image,
            left,
            current_y,
            right,
            current_y,
            Rgb([0, 184, 148]),
        );
        fill_rectangle(
            &mut image,
            right.saturating_sub(190),
            current_y.saturating_sub(24),
            right,
            current_y.saturating_add(24),
            Rgb([14, 143, 125]),
        );
        if let Some(font) = chart_font(true) {
            draw_text_mut(
                &mut image,
                Rgb([234, 255, 249]),
                right.saturating_sub(184),
                current_y.saturating_sub(18),
                18.0,
                &font,
                &price_text,
            );
            if let Some(ath) = candles
                .iter()
                .map(|candle| candle[2])
                .max_by(f64::total_cmp)
            {
                draw_text_mut(
                    &mut image,
                    Rgb([54, 224, 195]),
                    right.saturating_sub(220),
                    y_for(ath).saturating_sub(28).max(top),
                    20.0,
                    &font,
                    &format!("{} ATH", format_money(ath, true)),
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
        .map_err(|error| format!("token chart PNG encode failed: {error}"))?;
    Ok(output.into_inner())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use bot_core::token_signals::{SignalQuery, TokenAddress, TokenSignal};
    use serde_json::json;

    use super::{
        BinaryResponse, JsonResponse, ReqwestTokenSignalTransport, TokenSignalAdapter,
        TokenSignalCache, TokenSignalTransport, render_signal_chart,
    };

    #[test]
    fn reqwest_transport_supports_json_get_post_and_binary_downloads() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            for (content_type, body) in [
                ("application/json", br#"{"method":"get"}"#.as_slice()),
                ("application/json", br#"{"method":"post"}"#.as_slice()),
                ("image/png", &[1_u8, 2, 3][..]),
            ] {
                let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
                let mut request = [0_u8; 8_192];
                let _ = stream.read(&mut request);
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream
                    .write_all(headers.as_bytes())
                    .unwrap_or_else(|_| unreachable!());
                stream.write_all(body).unwrap_or_else(|_| unreachable!());
            }
        });
        let transport = ReqwestTokenSignalTransport::new().unwrap_or_else(|_| unreachable!());
        let base_url = format!("http://{address}");
        let get = transport
            .get_json(&base_url, &[("query", "synthetic".to_owned())])
            .unwrap_or_else(|_| unreachable!());
        assert!(get.body.contains("get"));
        let post = transport
            .post_json(&base_url, &json!({"value":"synthetic"}))
            .unwrap_or_else(|_| unreachable!());
        assert!(post.body.contains("post"));
        let binary = transport
            .get_binary(&base_url)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(binary.content_type, "image/png");
        assert_eq!(binary.body, [1, 2, 3]);
        assert!(server.join().is_ok());
    }

    #[derive(Default)]
    struct Cache {
        values: BTreeMap<String, String>,
        writes: Vec<(String, i64)>,
    }

    impl TokenSignalCache for Cache {
        type Error = &'static str;

        fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
            Ok(self.values.get(key).cloned())
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
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
        let load = adapter.load_query(&SignalQuery::Address(token));
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
    fn missing_chart_downloads_and_normalizes_a_real_token_image_to_png() {
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
        let adapter = TokenSignalAdapter::new(transport, Cache::default());
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
        let photo = adapter.render_photo(&signal);
        assert!(photo.as_ref().is_ok_and(|png| png.starts_with(b"\x89PNG")));
    }

    #[test]
    fn chart_renderer_returns_a_real_png_for_missing_and_usable_data() {
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
        assert!(blank.as_ref().is_ok_and(|png| png.starts_with(b"\x89PNG")));
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
}
