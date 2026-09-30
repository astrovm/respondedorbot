//! Typed blocking OpenRouter chat-completion boundary.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use bot_core::provider_pricing::{OPENROUTER_TRANSCRIPTION_MODEL, TokenPricing};
use bot_core::provider_stream_policy::StreamToolCallFragment;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const DEFAULT_OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

const OPENROUTER_PRICING_TTL: Duration = Duration::from_secs(300);
const OPENROUTER_PRICING_RETRY_INITIAL: Duration = Duration::from_secs(30);
const OPENROUTER_PRICING_RETRY_MAX: Duration = Duration::from_secs(15 * 60);
const USD_MICROS_PER_MILLION_TOKENS: i128 = 1_000_000_000_000;
const OPENROUTER_MAX_RESPONSE_BYTES: u64 = 1_048_576;

#[derive(Default)]
struct OpenRouterPricingState {
    fetched_at: Option<Instant>,
    models: BTreeMap<String, TokenPricing>,
    transcription_models: BTreeMap<String, TranscriptionPricing>,
    // Highest price among providers OpenRouter can still route to. The models
    // catalog publishes only the cheapest provider, which is too low to use as
    // both the routing cap and the credit reserve.
    endpoint_ceilings: BTreeMap<String, TokenPricing>,
    endpoint_checked_at: BTreeMap<String, Instant>,
    endpoint_retry_at: BTreeMap<String, Instant>,
    refresh_retry_at: Option<Instant>,
    refresh_retry_delay: Duration,
    last_refresh_error: Option<OpenRouterChatError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptionPricing {
    pub usd_micros_per_hour: i128,
}

#[derive(Clone)]
pub struct OpenRouterPricingCache {
    client: Client,
    api_key: String,
    base_url: String,
    state: Arc<Mutex<OpenRouterPricingState>>,
    refreshing: Arc<AtomicBool>,
    endpoint_refreshing: Arc<AtomicBool>,
}

impl OpenRouterPricingCache {
    pub fn new(api_key: &str, base_url: &str) -> Result<Self, OpenRouterChatError> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(transport_error)?;
        Ok(Self {
            client,
            api_key: api_key.to_owned(),
            base_url: base_url.to_owned(),
            state: Arc::new(Mutex::new(OpenRouterPricingState::default())),
            refreshing: Arc::new(AtomicBool::new(false)),
            endpoint_refreshing: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn pricing(&self, model: &str) -> Result<Option<TokenPricing>, OpenRouterChatError> {
        let model = model.trim();
        // A catalog floor alone cannot cover fallback providers. Every model's
        // first endpoint lookup waits, even when another model already loaded
        // the catalog. Later lookups use cached prices and refresh off-thread.
        let had_endpoint_check = {
            let state = self.lock_state()?;
            let base_model = catalog_base_model(model);
            state.endpoint_checked_at.contains_key(base_model)
                || state.endpoint_retry_at.contains_key(base_model)
        };
        let floor = self.lookup(model, |state, model| {
            cached_model_pricing(&state.models, model)
        })?;
        if floor.is_some() {
            if had_endpoint_check {
                self.spawn_endpoint_refresh(model);
            } else {
                let _ = self.refresh_endpoint_ceiling(model);
            }
        }
        let state = self.lock_state()?;
        Ok(effective_model_pricing(&state, model))
    }

    pub fn transcription_pricing(
        &self,
        model: &str,
    ) -> Result<Option<TranscriptionPricing>, OpenRouterChatError> {
        self.lookup(model, |state, model| {
            cached_model_pricing(&state.transcription_models, model)
        })
    }

    pub fn refresh(&self) -> Result<(), OpenRouterChatError> {
        let result = self.refresh_catalog();
        match result {
            Ok(()) => {
                self.record_refresh_success()?;
                Ok(())
            }
            Err(error) => {
                self.record_refresh_failure(&error)?;
                Err(error)
            }
        }
    }

    fn refresh_catalog(&self) -> Result<(), OpenRouterChatError> {
        let api_key = self.api_key.trim();
        if api_key.is_empty() {
            return Err(OpenRouterChatError::MissingApiKey);
        }
        let mut response = self
            .client
            .get(models_url(&self.base_url)?)
            .bearer_auth(api_key)
            .send()
            .map_err(transport_error)?;
        let status_code = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_ascii_lowercase(), value.to_owned()))
            })
            .collect::<BTreeMap<_, _>>();
        let body = read_bounded_body(&mut response)?;
        if status_code >= 400 {
            return Err(http_error(status_code, &body, &headers));
        }
        let payload = serde_json::from_str::<Value>(&body).map_err(invalid_json)?;
        let models = payload
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                OpenRouterChatError::InvalidJson("models response has no data array".to_owned())
            })?;
        let mut prices = BTreeMap::new();
        let mut transcription_prices = BTreeMap::new();
        for model in models {
            let Some((id, pricing, transcription_pricing)) = parse_catalog_model(model) else {
                continue;
            };
            prices.insert(id.clone(), pricing);
            prices
                .entry(catalog_base_model(&id).to_owned())
                .or_insert(pricing);
            if let Some(transcription_pricing) = transcription_pricing {
                transcription_prices.insert(id.clone(), transcription_pricing);
                transcription_prices
                    .entry(catalog_base_model(&id).to_owned())
                    .or_insert(transcription_pricing);
            }
        }
        let mut state = self.lock_state()?;
        if prices.is_empty() && transcription_prices.is_empty() {
            if state.models.is_empty() && state.transcription_models.is_empty() {
                return Err(OpenRouterChatError::InvalidJson(
                    "models response has no usable pricing entries".to_owned(),
                ));
            }
            // Keep the last usable catalog when the API returns a valid but
            // empty or partially unusable response.
        } else {
            // Treat catalog responses as patches. This preserves the last
            // usable price for an entry omitted by a transient partial reply,
            // while refreshing every entry that is present in the new reply.
            state.models.extend(prices);
            state.transcription_models.extend(transcription_prices);
        }
        Ok(())
    }

    pub fn price_ceiling(&self, model: &str) -> Result<(f64, f64), OpenRouterChatError> {
        let pricing =
            self.pricing(model)?
                .ok_or_else(|| OpenRouterChatError::MissingModelPricing {
                    model: model.to_owned(),
                })?;
        Ok((
            pricing.input_per_million as f64 / 1_000_000.0,
            pricing.output_per_million as f64 / 1_000_000.0,
        ))
    }

    pub fn apply_to_request(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> Result<(), OpenRouterChatError> {
        let (prompt, completion) = self.price_ceiling(&request.model)?;
        request.set_price_ceiling(prompt, completion);
        Ok(())
    }

    fn spawn_endpoint_refresh(&self, model: &str) {
        let base_model = catalog_base_model(model.trim()).to_owned();
        let Ok(true) = self.endpoint_refresh_due(&base_model) else {
            return;
        };
        if self
            .endpoint_refreshing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let cache = self.clone();
        let spawned = std::thread::Builder::new()
            .name("openrouter-endpoints".to_owned())
            .spawn(move || {
                let _ = cache.refresh_endpoint_ceiling(&base_model);
                cache.endpoint_refreshing.store(false, Ordering::Release);
            });
        self.endpoint_refreshing
            .fetch_and(spawned.is_ok(), Ordering::AcqRel);
    }

    fn refresh_endpoint_ceiling(&self, model: &str) -> Result<(), OpenRouterChatError> {
        let base_model = catalog_base_model(model.trim());
        if !self.endpoint_refresh_due(base_model)? {
            return Ok(());
        }
        match self.load_endpoint_ceiling(base_model) {
            Ok(ceiling) => {
                let mut state = self.lock_state()?;
                match ceiling {
                    Some(pricing) => {
                        state
                            .endpoint_ceilings
                            .insert(base_model.to_owned(), pricing);
                    }
                    None => {
                        state.endpoint_ceilings.remove(base_model);
                    }
                }
                state
                    .endpoint_checked_at
                    .insert(base_model.to_owned(), Instant::now());
                state.endpoint_retry_at.remove(base_model);
            }
            Err(error) => {
                eprintln!("could not load OpenRouter endpoint prices for {base_model}: {error}");
                let mut state = self.lock_state()?;
                state.endpoint_retry_at.insert(
                    base_model.to_owned(),
                    Instant::now() + OPENROUTER_PRICING_RETRY_INITIAL,
                );
            }
        }
        Ok(())
    }

    fn endpoint_refresh_due(&self, model: &str) -> Result<bool, OpenRouterChatError> {
        let state = self.lock_state()?;
        if cached_model_pricing(&state.models, model).is_none() {
            return Ok(false);
        }
        if state
            .endpoint_retry_at
            .get(model)
            .is_some_and(|retry_at| *retry_at > Instant::now())
        {
            return Ok(false);
        }
        Ok(state
            .endpoint_checked_at
            .get(model)
            .is_none_or(|checked_at| checked_at.elapsed() >= OPENROUTER_PRICING_TTL))
    }

    fn load_endpoint_ceiling(
        &self,
        model: &str,
    ) -> Result<Option<TokenPricing>, OpenRouterChatError> {
        let api_key = self.api_key.trim();
        if api_key.is_empty() {
            return Err(OpenRouterChatError::MissingApiKey);
        }
        let mut response = self
            .client
            .get(endpoints_url(&self.base_url, model)?)
            .bearer_auth(api_key)
            .send()
            .map_err(transport_error)?;
        let status_code = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_ascii_lowercase(), value.to_owned()))
            })
            .collect::<BTreeMap<_, _>>();
        let body = read_bounded_body(&mut response)?;
        if status_code >= 400 {
            return Err(http_error(status_code, &body, &headers));
        }
        let payload = serde_json::from_str::<Value>(&body).map_err(invalid_json)?;
        endpoint_ceiling_from_payload(&payload)
    }

    fn refresh_is_due(&self) -> Result<bool, OpenRouterChatError> {
        let state = self.lock_state()?;
        let stale = state
            .fetched_at
            .is_none_or(|fetched_at| fetched_at.elapsed() >= OPENROUTER_PRICING_TTL);
        Ok(stale
            && state
                .refresh_retry_at
                .is_none_or(|retry_at| retry_at <= Instant::now()))
    }

    fn lookup<T, F>(&self, model: &str, lookup: F) -> Result<Option<T>, OpenRouterChatError>
    where
        T: Copy,
        F: Fn(&OpenRouterPricingState, &str) -> Option<T>,
    {
        let model = model.trim();
        if model.is_empty() {
            return Ok(None);
        }
        let cached = {
            let state = self.lock_state()?;
            lookup(&state, model)
        };
        if cached.is_some() {
            // A known price is good enough for this request; refresh the
            // catalog off the request path so no reply waits on it.
            if self.refresh_is_due()? {
                self.refresh_in_background();
            }
            return Ok(cached);
        }
        if self.refresh_is_due()?
            && let Err(error) = self.refresh()
        {
            return Err(error);
        }
        let state = self.lock_state()?;
        if let Some(current) = lookup(&state, model) {
            return Ok(Some(current));
        }
        state.last_refresh_error.clone().map_or(Ok(None), Err)
    }

    fn refresh_in_background(&self) {
        if self
            .refreshing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let cache = self.clone();
        let spawned = std::thread::Builder::new()
            .name("openrouter-pricing".to_owned())
            .spawn(move || {
                if let Err(error) = cache.refresh() {
                    eprintln!("could not refresh OpenRouter pricing: {error}");
                }
                cache.refreshing.store(false, Ordering::Release);
            });
        // Without a worker thread nothing will clear the flag, so release it
        // here; a running worker keeps (or has already cleared) its own flag.
        self.refreshing.fetch_and(spawned.is_ok(), Ordering::AcqRel);
    }

    fn record_refresh_success(&self) -> Result<(), OpenRouterChatError> {
        let mut state = self.lock_state()?;
        state.fetched_at = Some(Instant::now());
        state.refresh_retry_at = None;
        state.refresh_retry_delay = Duration::ZERO;
        state.last_refresh_error = None;
        Ok(())
    }

    fn record_refresh_failure(
        &self,
        error: &OpenRouterChatError,
    ) -> Result<(), OpenRouterChatError> {
        let mut state = self.lock_state()?;
        let delay = if state.refresh_retry_delay.is_zero() {
            OPENROUTER_PRICING_RETRY_INITIAL
        } else {
            state
                .refresh_retry_delay
                .checked_mul(2)
                .unwrap_or(OPENROUTER_PRICING_RETRY_MAX)
                .min(OPENROUTER_PRICING_RETRY_MAX)
        };
        state.refresh_retry_delay = delay;
        state.refresh_retry_at = Some(Instant::now() + delay);
        state.last_refresh_error = Some(error.clone());
        Ok(())
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, OpenRouterPricingState>, OpenRouterChatError> {
        self.state.lock().map_err(|_| {
            OpenRouterChatError::Transport("OpenRouter pricing cache was poisoned".to_owned())
        })
    }
}

fn openrouter_base_url(base_url: &str) -> Result<&str, OpenRouterChatError> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let parsed = reqwest::Url::parse(trimmed).map_err(|_| OpenRouterChatError::InvalidBaseUrl)?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(OpenRouterChatError::InvalidBaseUrl);
    }
    Ok(trimmed)
}

fn models_url(base_url: &str) -> Result<String, OpenRouterChatError> {
    Ok(format!(
        "{}/models?output_modalities=all",
        openrouter_base_url(base_url)?
    ))
}

fn endpoints_url(base_url: &str, model: &str) -> Result<String, OpenRouterChatError> {
    let model = catalog_base_model(model.trim());
    if model.is_empty() {
        return Err(OpenRouterChatError::MissingModelPricing {
            model: model.to_owned(),
        });
    }
    Ok(format!(
        "{}/models/{model}/endpoints",
        openrouter_base_url(base_url)?
    ))
}

fn catalog_base_model(model: &str) -> &str {
    model.split(':').next().unwrap_or(model)
}

fn cached_model_pricing<T: Copy>(models: &BTreeMap<String, T>, model: &str) -> Option<T> {
    let base_model = catalog_base_model(model);
    models
        .get(model)
        .copied()
        .or_else(|| models.get(base_model).copied())
}

fn parse_catalog_model(
    value: &Value,
) -> Option<(String, TokenPricing, Option<TranscriptionPricing>)> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())?;
    let pricing = value.get("pricing").and_then(Value::as_object)?;
    let pricing = token_pricing_from_object(pricing)?;
    let is_transcription = value
        .get("architecture")
        .and_then(Value::as_object)
        .is_some_and(|architecture| {
            architecture
                .get("modality")
                .and_then(Value::as_str)
                .is_some_and(|modality| modality == "audio->transcription")
                || architecture
                    .get("output_modalities")
                    .and_then(Value::as_array)
                    .is_some_and(|modalities| {
                        modalities
                            .iter()
                            .any(|modality| modality.as_str() == Some("transcription"))
                    })
        });
    let transcription_pricing = is_transcription
        .then(|| transcription_rate_from_input(id, pricing.input_per_million))
        .flatten();
    Some((id.to_owned(), pricing, transcription_pricing))
}

fn token_pricing_from_object(pricing: &Map<String, Value>) -> Option<TokenPricing> {
    let mut input = pricing.get("prompt").and_then(parse_catalog_rate);
    let mut cached_input = pricing.get("input_cache_read").and_then(parse_catalog_rate);
    let mut cache_write = pricing
        .get("input_cache_write")
        .and_then(parse_catalog_rate);
    let mut audio_input = pricing.get("audio").and_then(parse_catalog_rate);
    let mut output = pricing.get("completion").and_then(parse_catalog_rate);
    if let Some(overrides) = pricing.get("overrides").and_then(Value::as_array) {
        for override_value in overrides {
            let Some(override_pricing) = override_value.as_object() else {
                continue;
            };
            update_max(&mut input, override_pricing.get("prompt"));
            update_max(&mut cached_input, override_pricing.get("input_cache_read"));
            update_max(&mut cache_write, override_pricing.get("input_cache_write"));
            update_max(&mut audio_input, override_pricing.get("audio"));
            update_max(&mut output, override_pricing.get("completion"));
        }
    }
    Some(TokenPricing {
        input_per_million: input?,
        cached_input_per_million: cached_input,
        cache_write_per_million: cache_write,
        audio_input_per_million: audio_input,
        output_per_million: output?,
    })
}

fn max_optional_rate(left: Option<i128>, right: Option<i128>) -> Option<i128> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(rate), None) | (None, Some(rate)) => Some(rate),
        (None, None) => None,
    }
}

fn max_token_pricing(left: TokenPricing, right: TokenPricing) -> TokenPricing {
    TokenPricing {
        input_per_million: left.input_per_million.max(right.input_per_million),
        cached_input_per_million: max_optional_rate(
            left.cached_input_per_million,
            right.cached_input_per_million,
        ),
        cache_write_per_million: max_optional_rate(
            left.cache_write_per_million,
            right.cache_write_per_million,
        ),
        audio_input_per_million: max_optional_rate(
            left.audio_input_per_million,
            right.audio_input_per_million,
        ),
        output_per_million: left.output_per_million.max(right.output_per_million),
    }
}

fn endpoint_is_routable(endpoint: &Value) -> bool {
    let Some(endpoint) = endpoint.as_object() else {
        return false;
    };
    match endpoint.get("status") {
        None => true,
        Some(status) => status.as_i64() == Some(0),
    }
}

fn endpoint_ceiling_from_payload(
    payload: &Value,
) -> Result<Option<TokenPricing>, OpenRouterChatError> {
    let endpoints = payload
        .get("data")
        .and_then(|data| data.get("endpoints"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            OpenRouterChatError::InvalidJson("endpoint response has no endpoints array".to_owned())
        })?;
    let mut ceiling = None;
    for endpoint in endpoints {
        if !endpoint_is_routable(endpoint) {
            continue;
        }
        let Some(pricing) = endpoint
            .get("pricing")
            .and_then(Value::as_object)
            .and_then(token_pricing_from_object)
        else {
            continue;
        };
        ceiling = Some(match ceiling {
            Some(current) => max_token_pricing(current, pricing),
            None => pricing,
        });
    }
    Ok(ceiling)
}

fn effective_model_pricing(state: &OpenRouterPricingState, model: &str) -> Option<TokenPricing> {
    let catalog = cached_model_pricing(&state.models, model)?;
    let base_model = catalog_base_model(model);
    match state.endpoint_ceilings.get(base_model).copied() {
        Some(endpoint) => Some(max_token_pricing(catalog, endpoint)),
        None => Some(catalog),
    }
}

fn transcription_rate_from_input(
    model: &str,
    input_per_million: i128,
) -> Option<TranscriptionPricing> {
    // The general catalog does not expose the billing unit for transcription
    // models. Only the configured MAI model is known to publish this value as
    // USD per hour; unsupported units must fail closed instead of being
    // mistaken for hourly pricing.
    if catalog_base_model(model) != OPENROUTER_TRANSCRIPTION_MODEL {
        return None;
    }
    Some(TranscriptionPricing {
        usd_micros_per_hour: input_per_million
            .checked_add(999_999)?
            .checked_div(1_000_000)?,
    })
}

fn update_max(target: &mut Option<i128>, value: Option<&Value>) {
    let Some(rate) = value.and_then(parse_catalog_rate) else {
        return;
    };
    if target.is_none_or(|current| rate > current) {
        *target = Some(rate);
    }
}

fn parse_catalog_rate(value: &Value) -> Option<i128> {
    let text = value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_number().map(ToString::to_string))?;
    decimal_rate_to_usd_micros_per_million(&text)
}

fn decimal_rate_to_usd_micros_per_million(value: &str) -> Option<i128> {
    let value = value.trim().strip_prefix('+').unwrap_or(value);
    if value.is_empty() || value.starts_with('-') {
        return None;
    }
    let (mantissa, exponent) = match value.find('e').or_else(|| value.find('E')) {
        Some(position) => {
            let (mantissa, exponent) = value.split_at(position);
            (mantissa, exponent[1..].parse::<i32>().ok()?)
        }
        None => (value, 0),
    };
    if !mantissa.contains('.') && mantissa.chars().all(|character| character.is_ascii_digit()) {
        let digits = mantissa.parse::<i128>().ok()?;
        return scale_decimal_rate(digits, 0, exponent);
    }
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if (!whole.is_empty() && !whole.chars().all(|character| character.is_ascii_digit()))
        || !fraction.chars().all(|character| character.is_ascii_digit())
        || whole.is_empty() && fraction.is_empty()
    {
        return None;
    }
    let digits = format!("{}{}", if whole.is_empty() { "0" } else { whole }, fraction)
        .parse::<i128>()
        .ok()?;
    let fraction_digits = i32::try_from(fraction.len()).ok()?;
    scale_decimal_rate(digits, fraction_digits, exponent)
}

fn scale_decimal_rate(digits: i128, fraction_digits: i32, exponent: i32) -> Option<i128> {
    let scale = fraction_digits.checked_sub(exponent)?;
    if scale <= 0 {
        return digits
            .checked_mul(10_i128.checked_pow(scale.unsigned_abs())?)?
            .checked_mul(USD_MICROS_PER_MILLION_TOKENS);
    }
    let denominator = 10_i128.checked_pow(u32::try_from(scale).ok()?)?;
    let numerator = digits.checked_mul(USD_MICROS_PER_MILLION_TOKENS)?;
    numerator
        .checked_add(denominator.checked_sub(1)?)
        .and_then(|value| value.checked_div(denominator))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reasoning_details: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    #[must_use]
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(Value::String(content.into())),
            reasoning: None,
            reasoning_details: Vec::new(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    #[must_use]
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: Some(Value::String(content.into())),
            reasoning: None,
            reasoning_details: Vec::new(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
        }
    }

    #[must_use]
    pub fn assistant_tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: None,
            reasoning: None,
            reasoning_details: Vec::new(),
            name: None,
            tool_call_id: None,
            tool_calls: calls,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderMaxPrice {
    pub prompt: f64,
    pub completion: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderPreferences {
    pub max_price: ProviderMaxPrice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReasoningConfig {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    pub stream: bool,
}

impl ChatCompletionRequest {
    pub fn set_price_ceiling(&mut self, prompt: f64, completion: f64) {
        self.provider = Some(ProviderPreferences {
            max_price: ProviderMaxPrice { prompt, completion },
        });
    }

    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        let model = model.into();
        Self {
            model,
            messages,
            tools: Vec::new(),
            max_tokens: None,
            temperature: None,
            provider: None,
            reasoning: None,
            stream: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    pub generation_id: Option<String>,
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub model: String,
    pub upstream_provider: Option<String>,
    pub service_tier: Option<String>,
    pub annotations: Vec<Value>,
    pub usage: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub url: String,
    pub bearer_token: String,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatStreamChunk {
    pub generation_id: Option<String>,
    pub text: String,
    pub reasoning: String,
    pub reasoning_details: Vec<Value>,
    pub tool_call_fragments: Vec<StreamToolCallFragment>,
    pub finish_reason: Option<String>,
    pub model: Option<String>,
    pub upstream_provider: Option<String>,
    pub service_tier: Option<String>,
    pub annotations: Vec<Value>,
    pub usage: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamEvent {
    Chunk(Box<ChatStreamChunk>),
    Done,
}

pub trait OpenRouterTransport {
    fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError>;
}

pub trait OpenRouterStreamTransport {
    fn post_stream(
        &self,
        request: &HttpRequest,
        on_bytes: &mut dyn FnMut(&[u8]) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError>;
}

pub struct ReqwestOpenRouterTransport {
    client: Client,
}

impl ReqwestOpenRouterTransport {
    pub fn new() -> Result<Self, OpenRouterChatError> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(90))
                .build()
        })
        .map(|client| Self { client })
        .map_err(transport_error)
    }
}

impl OpenRouterTransport for ReqwestOpenRouterTransport {
    fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
        let mut response = self
            .client
            .post(&request.url)
            .bearer_auth(&request.bearer_token)
            .header("Content-Type", "application/json")
            .body(request.body.clone())
            .send()
            .map_err(transport_error)?;
        let status_code = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_ascii_lowercase(), value.to_owned()))
            })
            .collect();
        let body = read_bounded_body(&mut response)?;
        Ok(HttpResponse {
            status_code,
            body,
            headers,
        })
    }
}

impl OpenRouterStreamTransport for ReqwestOpenRouterTransport {
    fn post_stream(
        &self,
        request: &HttpRequest,
        on_bytes: &mut dyn FnMut(&[u8]) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError> {
        let mut response = self
            .client
            .post(&request.url)
            .bearer_auth(&request.bearer_token)
            .header("Content-Type", "application/json")
            .body(request.body.clone())
            .send()
            .map_err(transport_error)?;
        let status_code = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_ascii_lowercase(), value.to_owned()))
            })
            .collect::<BTreeMap<_, _>>();
        if status_code >= 400 {
            let body = read_bounded_body(&mut response)?;
            return Err(http_error(status_code, &body, &headers));
        }
        let mut buffer = [0_u8; 8_192];
        loop {
            let count = response.read(&mut buffer).map_err(transport_error)?;
            if count == 0 {
                return Ok(());
            }
            on_bytes(&buffer[..count])?;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OpenRouterChatError {
    #[error("OpenRouter API key is missing")]
    MissingApiKey,
    #[error("OpenRouter model is missing")]
    MissingModel,
    #[error("OpenRouter model pricing is unavailable: {model}")]
    MissingModelPricing { model: String },
    #[error("OpenRouter base URL is invalid")]
    InvalidBaseUrl,
    #[error("OpenRouter request could not be serialized: {0}")]
    RequestJson(String),
    #[error("OpenRouter transport failed: {0}")]
    Transport(String),
    #[error("OpenRouter rate limited the request")]
    RateLimited {
        retry_after_seconds: Option<u64>,
        message: String,
    },
    #[error("OpenRouter returned HTTP {status_code}: {message}")]
    Http { status_code: u16, message: String },
    #[error("OpenRouter returned malformed JSON: {0}")]
    InvalidJson(String),
    #[error("OpenRouter response exceeded the safe size limit")]
    ResponseTooLarge,
    #[error("OpenRouter response did not contain a valid completion")]
    MalformedResponse,
    #[error("OpenRouter stream ended with an incomplete UTF-8 or SSE frame")]
    IncompleteStream,
    #[error("OpenRouter stream returned an error: {0}")]
    Stream(String),
}

#[derive(Deserialize)]
struct RawEnvelope {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    choices: Vec<RawChoice>,
    #[serde(default)]
    usage: Map<String, Value>,
}

#[derive(Deserialize)]
struct RawChoice {
    #[serde(default)]
    message: Option<RawMessage>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
    #[serde(default)]
    annotations: Vec<Value>,
}

#[derive(Deserialize)]
struct RawStreamEnvelope {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    choices: Vec<RawStreamChoice>,
    #[serde(default)]
    usage: Map<String, Value>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct RawStreamChoice {
    #[serde(default)]
    delta: RawStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Default, Deserialize)]
struct RawStreamDelta {
    #[serde(default)]
    content: Value,
    #[serde(default)]
    reasoning: Value,
    #[serde(default)]
    reasoning_content: Value,
    #[serde(default)]
    reasoning_details: Option<Vec<Value>>,
    #[serde(default)]
    tool_calls: Vec<RawStreamToolCall>,
    #[serde(default)]
    annotations: Vec<Value>,
}

#[derive(Deserialize)]
struct RawStreamToolCall {
    #[serde(default)]
    index: Value,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    call_type: Option<String>,
    #[serde(default)]
    function: Option<RawStreamFunction>,
}

#[derive(Deserialize)]
struct RawStreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

fn completion_url(base_url: &str) -> Result<String, OpenRouterChatError> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let parsed = reqwest::Url::parse(trimmed).map_err(|_| OpenRouterChatError::InvalidBaseUrl)?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(OpenRouterChatError::InvalidBaseUrl);
    }
    Ok(format!("{trimmed}/chat/completions"))
}

fn response_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| "upstream request failed".to_owned())
}

fn retry_after_seconds(headers: &BTreeMap<String, String>) -> Option<u64> {
    headers
        .get("retry-after")
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn transport_error(error: impl std::fmt::Display) -> OpenRouterChatError {
    OpenRouterChatError::Transport(error.to_string())
}

fn invalid_json(error: serde_json::Error) -> OpenRouterChatError {
    OpenRouterChatError::InvalidJson(error.to_string())
}

fn request_json_error(error: serde_json::Error) -> OpenRouterChatError {
    OpenRouterChatError::RequestJson(error.to_string())
}

fn http_error(
    status_code: u16,
    body: &str,
    headers: &BTreeMap<String, String>,
) -> OpenRouterChatError {
    if status_code == 429 {
        OpenRouterChatError::RateLimited {
            retry_after_seconds: retry_after_seconds(headers),
            message: response_message(body),
        }
    } else {
        OpenRouterChatError::Http {
            status_code,
            message: response_message(body),
        }
    }
}

fn read_bounded_body(
    response: &mut reqwest::blocking::Response,
) -> Result<String, OpenRouterChatError> {
    let mut body = Vec::new();
    response
        .by_ref()
        .take(OPENROUTER_MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut body)
        .map_err(transport_error)?;
    if body.len() as u64 > OPENROUTER_MAX_RESPONSE_BYTES {
        return Err(OpenRouterChatError::ResponseTooLarge);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

fn content_text(content: &Value) -> Result<String, OpenRouterChatError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => Ok(parts
            .iter()
            .filter_map(|part| {
                part.as_object()
                    .and_then(|part| part.get("text"))
                    .and_then(Value::as_str)
            })
            .collect::<String>()),
        Value::Bool(_) | Value::Number(_) | Value::Object(_) => {
            Err(OpenRouterChatError::MalformedResponse)
        }
    }
}

pub fn parse_chat_completion(
    response: HttpResponse,
    requested_model: &str,
) -> Result<ChatCompletion, OpenRouterChatError> {
    if response.status_code >= 400 {
        return Err(http_error(
            response.status_code,
            &response.body,
            &response.headers,
        ));
    }
    let envelope = serde_json::from_str::<RawEnvelope>(&response.body).map_err(invalid_json)?;
    let choice = envelope
        .choices
        .into_iter()
        .next()
        .ok_or(OpenRouterChatError::MalformedResponse)?;
    let message = choice
        .message
        .ok_or(OpenRouterChatError::MalformedResponse)?;
    let text = content_text(&message.content)?;
    if text.is_empty() && message.tool_calls.is_empty() {
        return Err(OpenRouterChatError::MalformedResponse);
    }
    Ok(ChatCompletion {
        generation_id: envelope.id.filter(|value| !value.is_empty()),
        text,
        tool_calls: message.tool_calls,
        finish_reason: choice.finish_reason.filter(|value| !value.is_empty()),
        model: envelope
            .model
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| requested_model.to_owned()),
        upstream_provider: envelope.provider.filter(|value| !value.is_empty()),
        service_tier: envelope.service_tier.filter(|value| !value.is_empty()),
        annotations: message.annotations,
        usage: envelope.usage,
    })
}

pub fn complete_with<T: OpenRouterTransport>(
    transport: &T,
    api_key: &str,
    base_url: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletion, OpenRouterChatError> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(OpenRouterChatError::MissingApiKey);
    }
    if request.model.trim().is_empty() {
        return Err(OpenRouterChatError::MissingModel);
    }
    if request.stream {
        return Err(OpenRouterChatError::MalformedResponse);
    }
    let body = serde_json::to_string(request).map_err(request_json_error)?;
    parse_chat_completion(
        transport.post(&HttpRequest {
            url: completion_url(base_url)?,
            bearer_token: api_key.to_owned(),
            body,
        })?,
        &request.model,
    )
}

pub fn complete(
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletion, OpenRouterChatError> {
    complete_with(
        &ReqwestOpenRouterTransport::new()?,
        api_key,
        DEFAULT_OPENROUTER_BASE_URL,
        request,
    )
}

#[derive(Debug, Default)]
struct SseDecoder {
    pending: Vec<u8>,
    saw_done: bool,
}

impl SseDecoder {
    fn feed(
        &mut self,
        bytes: &[u8],
        on_event: &mut dyn FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError> {
        self.pending.extend_from_slice(bytes);
        while let Some((frame_end, delimiter_length)) = next_sse_frame(&self.pending) {
            let frame = self.pending[..frame_end].to_vec();
            self.pending.drain(..frame_end + delimiter_length);
            if let Some(event) = parse_sse_frame(&frame)? {
                if event == ChatStreamEvent::Done {
                    self.saw_done = true;
                }
                on_event(event)?;
            }
        }
        Ok(())
    }

    fn finish(
        &mut self,
        on_event: &mut dyn FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError> {
        if !self.pending.iter().all(u8::is_ascii_whitespace) {
            let frame = std::mem::take(&mut self.pending);
            if let Some(event) = parse_sse_frame(&frame)? {
                if event == ChatStreamEvent::Done {
                    self.saw_done = true;
                }
                on_event(event)?;
            }
        }
        if self.pending.iter().all(u8::is_ascii_whitespace) {
            self.pending.clear();
        }
        if self.saw_done {
            Ok(())
        } else {
            Err(OpenRouterChatError::IncompleteStream)
        }
    }
}

fn next_sse_frame(bytes: &[u8]) -> Option<(usize, usize)> {
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4))
        .or_else(|| {
            bytes
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|position| (position, 2))
        })
}

fn parse_sse_frame(frame: &[u8]) -> Result<Option<ChatStreamEvent>, OpenRouterChatError> {
    let frame = std::str::from_utf8(frame).map_err(|_| OpenRouterChatError::IncompleteStream)?;
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        return Ok(None);
    }
    if data.trim() == "[DONE]" {
        return Ok(Some(ChatStreamEvent::Done));
    }
    let envelope = serde_json::from_str::<RawStreamEnvelope>(&data).map_err(invalid_json)?;
    if let Some(error) = envelope.error.as_ref() {
        return Err(OpenRouterChatError::Stream(stream_error_message(error)));
    }
    let choice = envelope.choices.into_iter().next();
    if let Some(error) = choice.as_ref().and_then(|choice| choice.error.as_ref()) {
        return Err(OpenRouterChatError::Stream(stream_error_message(error)));
    }
    let (text, reasoning, reasoning_details, fragments, finish_reason, annotations) = choice
        .map_or_else(
            || {
                Ok::<_, OpenRouterChatError>((
                    String::new(),
                    String::new(),
                    Vec::new(),
                    Vec::new(),
                    None,
                    Vec::new(),
                ))
            },
            |choice| {
                let text = content_text(&choice.delta.content)?;
                let reasoning =
                    reasoning_text(&choice.delta.reasoning, &choice.delta.reasoning_content)?;
                let fragments = choice
                    .delta
                    .tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(position, fragment)| {
                        let function = fragment.function;
                        StreamToolCallFragment {
                            position: i64::try_from(position).unwrap_or(i64::MAX),
                            index: fragment.index,
                            id: fragment.id,
                            call_type: fragment.call_type,
                            name: function.as_ref().and_then(|value| value.name.clone()),
                            arguments: function.and_then(|value| value.arguments),
                        }
                    })
                    .collect();
                Ok::<_, OpenRouterChatError>((
                    text,
                    reasoning,
                    choice.delta.reasoning_details.unwrap_or_default(),
                    fragments,
                    choice.finish_reason.filter(|value| !value.is_empty()),
                    choice.delta.annotations,
                ))
            },
        )?;
    Ok(Some(ChatStreamEvent::Chunk(Box::new(ChatStreamChunk {
        generation_id: envelope.id.filter(|value| !value.is_empty()),
        text,
        reasoning,
        reasoning_details,
        tool_call_fragments: fragments,
        finish_reason,
        model: envelope.model.filter(|value| !value.is_empty()),
        upstream_provider: envelope.provider.filter(|value| !value.is_empty()),
        service_tier: envelope.service_tier.filter(|value| !value.is_empty()),
        annotations,
        usage: envelope.usage,
    }))))
}

fn reasoning_text(
    reasoning: &Value,
    reasoning_content: &Value,
) -> Result<String, OpenRouterChatError> {
    if !reasoning.is_null() {
        return content_text(reasoning);
    }
    content_text(reasoning_content)
}

fn stream_error_message(error: &Value) -> String {
    error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .filter(|message| !message.is_empty())
        .unwrap_or("unknown provider error")
        .to_owned()
}

pub fn stream_with<T, F>(
    transport: &T,
    api_key: &str,
    base_url: &str,
    request: &ChatCompletionRequest,
    mut on_event: F,
) -> Result<(), OpenRouterChatError>
where
    T: OpenRouterStreamTransport,
    F: FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
{
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(OpenRouterChatError::MissingApiKey);
    }
    if request.model.trim().is_empty() {
        return Err(OpenRouterChatError::MissingModel);
    }
    if !request.stream {
        return Err(OpenRouterChatError::MalformedResponse);
    }
    let body = serde_json::to_string(request).map_err(request_json_error)?;
    let mut decoder = SseDecoder::default();
    transport.post_stream(
        &HttpRequest {
            url: completion_url(base_url)?,
            bearer_token: api_key.to_owned(),
            body,
        },
        &mut |bytes| decoder.feed(bytes, &mut on_event),
    )?;
    decoder.finish(&mut on_event)
}

pub fn stream<F>(
    api_key: &str,
    request: &ChatCompletionRequest,
    on_event: F,
) -> Result<(), OpenRouterChatError>
where
    F: FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError>,
{
    stream_with(
        &ReqwestOpenRouterTransport::new()?,
        api_key,
        DEFAULT_OPENROUTER_BASE_URL,
        request,
        on_event,
    )
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::{Duration, Instant};

    use bot_core::provider_pricing::{DEEPSEEK_MODEL, OPENROUTER_TRANSCRIPTION_MODEL};
    use serde_json::{Value, json};

    use super::{
        ChatCompletionRequest, ChatMessage, ChatRole, ChatStreamChunk, ChatStreamEvent,
        HttpRequest, HttpResponse, OPENROUTER_MAX_RESPONSE_BYTES, OpenRouterChatError,
        OpenRouterPricingCache, OpenRouterStreamTransport, OpenRouterTransport,
        ReqwestOpenRouterTransport, ToolCall, ToolFunctionCall, complete_with, parse_catalog_model,
        parse_chat_completion, stream_with,
    };

    fn serve_once(
        status: &str,
        content_type: &str,
        body: &str,
    ) -> Result<(String, thread::JoinHandle<TestResult>), Box<dyn std::error::Error + Send + Sync>>
    {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let status = status.to_owned();
        let content_type = content_type.to_owned();
        let body = body.to_owned();
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 8_192];
            let _ = stream.read(&mut request);
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nX-Synthetic: yes\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes())?;
            Ok(())
        });
        Ok((format!("http://{address}"), server))
    }

    fn serve_sequence(
        responses: Vec<(String, String, String)>,
    ) -> Result<(String, thread::JoinHandle<TestResult>), Box<dyn std::error::Error + Send + Sync>>
    {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            for (status, content_type, body) in responses {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 8_192];
                let _ = stream.read(&mut request);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes())?;
            }
            Ok(())
        });
        Ok((format!("http://{address}"), server))
    }

    struct Transport {
        response: RefCell<Option<Result<HttpResponse, OpenRouterChatError>>>,
        requests: RefCell<Vec<HttpRequest>>,
    }

    impl OpenRouterTransport for Transport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            self.response
                .borrow_mut()
                .take()
                .unwrap_or(Err(OpenRouterChatError::Transport(
                    "missing response".to_owned(),
                )))
        }
    }

    struct StreamTransport {
        chunks: Vec<Vec<u8>>,
        failure: Option<OpenRouterChatError>,
        requests: RefCell<Vec<HttpRequest>>,
    }

    impl OpenRouterStreamTransport for StreamTransport {
        fn post_stream(
            &self,
            request: &HttpRequest,
            on_bytes: &mut dyn FnMut(&[u8]) -> Result<(), OpenRouterChatError>,
        ) -> Result<(), OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            for chunk in &self.chunks {
                on_bytes(chunk)?;
            }
            self.failure.clone().map_or(Ok(()), Err)
        }
    }

    fn response(status_code: u16, body: Value) -> HttpResponse {
        HttpResponse {
            status_code,
            body: body.to_string(),
            headers: BTreeMap::new(),
        }
    }

    /// Every streaming scenario shares one `stream_with` instantiation per
    /// transport type: events are collected, and the consumer can be told to
    /// fail on the first event.
    fn stream_events<T: OpenRouterStreamTransport>(
        transport: &T,
        api_key: &str,
        base_url: &str,
        request: &ChatCompletionRequest,
        consumer_failure: Option<OpenRouterChatError>,
    ) -> (Result<(), OpenRouterChatError>, Vec<ChatStreamEvent>) {
        let mut events = Vec::new();
        let mut collect = |event| {
            events.push(event);
            consumer_failure.clone().map_or(Ok(()), Err)
        };
        let on_event: &mut dyn FnMut(ChatStreamEvent) -> Result<(), OpenRouterChatError> =
            &mut collect;
        let result = stream_with(transport, api_key, base_url, request, on_event);
        (result, events)
    }

    fn stream_chunk(event: &ChatStreamEvent) -> Option<&ChatStreamChunk> {
        match event {
            ChatStreamEvent::Chunk(chunk) => Some(chunk),
            ChatStreamEvent::Done => None,
        }
    }

    fn request() -> ChatCompletionRequest {
        ChatCompletionRequest::new(
            "synthetic/model",
            vec![
                ChatMessage::text(ChatRole::System, "synthetic system"),
                ChatMessage::text(ChatRole::User, "synthetic question"),
            ],
        )
    }

    #[test]
    fn dynamic_price_ceilings_are_applied_without_a_local_deepseek_table() {
        let mut request = ChatCompletionRequest::new(DEEPSEEK_MODEL, Vec::new());
        assert!(
            serde_json::to_value(&request)
                .unwrap_or(Value::Null)
                .get("provider")
                .is_none()
        );
        request.set_price_ceiling(0.3, 1.2);
        let body = serde_json::to_value(request).unwrap_or(Value::Null);
        assert_eq!(body["provider"]["max_price"]["prompt"], 0.3);
        assert_eq!(body["provider"]["max_price"]["completion"], 1.2);

        let unknown =
            serde_json::to_value(ChatCompletionRequest::new("synthetic/model", Vec::new()))
                .unwrap_or(Value::Null);
        assert!(unknown.get("provider").is_none());
    }

    #[test]
    fn loads_catalog_pricing_and_uses_the_highest_override_rate() -> TestResult {
        let body = json!({
            "data": [{
                "id": DEEPSEEK_MODEL,
                "pricing": {
                    "prompt": "0.00000015",
                    "completion": "0.0000006",
                    "input_cache_read": "0.000000003",
                    "overrides": [{
                        "utc_days": ["monday"],
                        "utc_start": 100,
                        "utc_end": 400,
                        "prompt": "0.0000003",
                        "completion": "0.0000012",
                        "input_cache_read": "0.000000006"
                    }]
                }
            }]
        })
        .to_string();
        let served = serve_once("200", "application/json", &body);
        let (base_url, server) = served?;
        let cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        let pricing = cache
            .pricing(&format!("{DEEPSEEK_MODEL}:free"))?
            .ok_or("unexpected missing value")?;
        assert_eq!(pricing.input_per_million, 300_000);
        assert_eq!(pricing.cached_input_per_million, Some(6_000));
        assert_eq!(pricing.output_per_million, 1_200_000);
        assert_eq!(
            cache.price_ceiling("synthetic/missing"),
            Err(OpenRouterChatError::MissingModelPricing {
                model: "synthetic/missing".to_owned(),
            })
        );

        let mut request = ChatCompletionRequest::new(format!("{DEEPSEEK_MODEL}:free"), Vec::new());
        cache.apply_to_request(&mut request)?;
        let request_body = serde_json::to_value(request).unwrap_or(Value::Null);
        assert_eq!(request_body["provider"]["max_price"]["prompt"], 0.3);
        assert_eq!(request_body["provider"]["max_price"]["completion"], 1.2);

        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn price_ceiling_uses_the_highest_routable_endpoint_and_keeps_the_catalog_when_lookup_fails()
    -> TestResult {
        assert_eq!(
            super::endpoints_url("not-a-url", DEEPSEEK_MODEL),
            Err(OpenRouterChatError::InvalidBaseUrl)
        );
        assert_eq!(
            super::endpoints_url("ftp://openrouter.example.test/api/v1", DEEPSEEK_MODEL),
            Err(OpenRouterChatError::InvalidBaseUrl)
        );
        assert_eq!(
            super::endpoints_url("https://openrouter.example.test/api/v1/", " "),
            Err(OpenRouterChatError::MissingModelPricing {
                model: String::new(),
            })
        );
        assert_eq!(
            super::endpoints_url(
                "https://openrouter.example.test/api/v1/",
                &format!("{DEEPSEEK_MODEL}:free")
            ),
            Ok(format!(
                "https://openrouter.example.test/api/v1/models/{DEEPSEEK_MODEL}/endpoints"
            ))
        );
        assert_eq!(
            super::endpoint_ceiling_from_payload(&json!({"data": {}})),
            Err(OpenRouterChatError::InvalidJson(
                "endpoint response has no endpoints array".to_owned()
            ))
        );

        let catalog = json!({
            "data": [{
                "id": DEEPSEEK_MODEL,
                "pricing": {
                    "prompt": "0.0000000198",
                    "completion": "0.000000396",
                    "input_cache_read": "0.000000003",
                    "overrides": [{
                        "prompt": "0.0000004",
                        "completion": "0.0000005"
                    }]
                }
            }]
        })
        .to_string();
        let endpoints = json!({
            "data": {
                "endpoints": [
                    "malformed",
                    {"status": 0, "pricing": {"completion": "0.000001"}},
                    {"status": "up", "pricing": {"prompt": "0.0000009", "completion": "0.000009"}},
                    {"status": -2, "pricing": {"prompt": "0.0000009", "completion": "0.000009"}},
                    {
                        "status": 0,
                        "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}
                    },
                    {
                        "status": 0,
                        "pricing": {
                            "prompt": "0.0000002",
                            "completion": "0.0000008",
                            "input_cache_read": "0.000000006"
                        }
                    },
                    {
                        "pricing": {
                            "prompt": "0.0000002",
                            "completion": "0.0000008",
                            "input_cache_read": "0.000000006",
                            "input_cache_write": "0.000000007",
                            "audio": "0.000000008"
                        }
                    },
                    {
                        "status": 0,
                        "pricing": {
                            "prompt": "0.0000003",
                            "completion": "0.0000012",
                            "overrides": [{
                                "prompt": "0.0000003",
                                "completion": "0.0000015",
                                "input_cache_read": "0.000000004"
                            }]
                        }
                    }
                ]
            }
        })
        .to_string();
        let cleared = json!({"data": {"endpoints": [{"status": -1}]}}).to_string();
        let served = serve_sequence(vec![
            ("200 OK".to_owned(), "application/json".to_owned(), catalog),
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                endpoints,
            ),
            ("200 OK".to_owned(), "application/json".to_owned(), cleared),
        ]);
        let (base_url, server) = served?;
        let cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        let priced = cache
            .pricing(&format!("{DEEPSEEK_MODEL}:free"))?
            .ok_or("unexpected missing value")?;
        assert_eq!(priced.input_per_million, 400_000);
        assert_eq!(priced.output_per_million, 1_500_000);
        assert_eq!(priced.cached_input_per_million, Some(6_000));
        assert_eq!(priced.cache_write_per_million, Some(7_000));
        assert_eq!(priced.audio_input_per_million, Some(8_000));
        assert_eq!(
            cache.price_ceiling(&format!("{DEEPSEEK_MODEL}:free"))?,
            (400_000.0 / 1_000_000.0, 1_500_000.0 / 1_000_000.0)
        );
        let repeated = cache
            .pricing(DEEPSEEK_MODEL)?
            .ok_or("unexpected missing value")?;
        assert_eq!(repeated, priced);

        cache
            .state
            .lock()
            .ok()
            .ok_or("pricing state")?
            .endpoint_checked_at
            .insert(
                DEEPSEEK_MODEL.to_owned(),
                Instant::now() - Duration::from_secs(301),
            );
        // The stale ceiling is replaced off the request path. Wait until that
        // refresh lands, then the catalog floor is the price again.
        let deadline = Instant::now() + Duration::from_secs(2);
        let cleared_price = loop {
            let current = cache
                .pricing(DEEPSEEK_MODEL)?
                .ok_or("unexpected missing value")?;
            if current.output_per_million == 500_000 || Instant::now() >= deadline {
                break current;
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(cleared_price.input_per_million, 400_000);
        assert_eq!(cleared_price.output_per_million, 500_000);
        assert_eq!(cleared_price.cached_input_per_million, Some(3_000));
        assert_eq!(cleared_price.cache_write_per_million, None);
        assert_eq!(cleared_price.audio_input_per_million, None);

        let unavailable = serve_sequence(vec![
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                json!({
                    "data": [{
                        "id": DEEPSEEK_MODEL,
                        "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}
                    }]
                })
                .to_string(),
            ),
            (
                "503 Service Unavailable".to_owned(),
                "application/json".to_owned(),
                "{\"error\":{\"message\":\"synthetic outage\"}}".to_owned(),
            ),
        ]);
        let (base_url, unavailable_server) = unavailable?;
        let unavailable_cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        let fallback = unavailable_cache
            .pricing(DEEPSEEK_MODEL)?
            .ok_or("unexpected missing value")?;
        assert_eq!(fallback.input_per_million, 150_000);
        assert_eq!(fallback.output_per_million, 600_000);
        assert_eq!(
            unavailable_cache
                .pricing(DEEPSEEK_MODEL)?
                .ok_or("unexpected missing value")?,
            fallback
        );

        let cached_only =
            OpenRouterPricingCache::new("", "https://openrouter.example.test/api/v1")?;
        cached_only
            .state
            .lock()
            .ok()
            .ok_or("pricing state")?
            .models
            .insert(DEEPSEEK_MODEL.to_owned(), fallback);
        assert_eq!(cached_only.pricing(DEEPSEEK_MODEL)?, Some(fallback));
        cached_only.refresh_endpoint_ceiling("synthetic/missing")?;

        let malformed = serve_sequence(vec![
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                json!({
                    "data": [{
                        "id": DEEPSEEK_MODEL,
                        "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}
                    }]
                })
                .to_string(),
            ),
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                "not-json".to_owned(),
            ),
        ]);
        let (base_url, malformed_server) = malformed?;
        let malformed_cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        assert_eq!(
            malformed_cache
                .pricing(DEEPSEEK_MODEL)?
                .ok_or("unexpected missing value")?,
            fallback
        );

        server.join().ok().ok_or("server thread panicked")??;
        unavailable_server
            .join()
            .ok()
            .ok_or("server thread panicked")??;
        malformed_server
            .join()
            .ok()
            .ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn warmed_catalog_waits_for_the_first_endpoint_ceiling_and_keeps_it_after_an_outage()
    -> TestResult {
        let catalog = json!({"data": [{
            "id": "synthetic/chat",
            "pricing": {"prompt": "0.0000000198", "completion": "0.000000396"}
        }]})
        .to_string();
        let endpoints = json!({"data": {"endpoints": [
            {"status": -5, "pricing": {"prompt": "0.0000000198", "completion": "0.000000396"}},
            {"status": 0, "pricing": {"prompt": "0.0000003", "completion": "0.0000012"}}
        ]}})
        .to_string();
        let (base_url, server) = serve_sequence(vec![
            ("200 OK".to_owned(), "application/json".to_owned(), catalog),
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                endpoints,
            ),
            (
                "503 Service Unavailable".to_owned(),
                "application/json".to_owned(),
                "{}".to_owned(),
            ),
        ])?;
        let cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        // Transcription or a different model may have warmed the shared catalog.
        cache.refresh()?;
        let pricing = cache.pricing("synthetic/chat")?.ok_or("missing price")?;
        assert_eq!(pricing.input_per_million, 300_000);
        assert_eq!(pricing.output_per_million, 1_200_000);
        assert_eq!(cache.pricing("  synthetic/chat  ")?, Some(pricing));
        let mut request = ChatCompletionRequest::new("synthetic/chat", Vec::new());
        cache.apply_to_request(&mut request)?;
        let body = serde_json::to_value(request)?;
        assert_eq!(body["provider"]["max_price"]["prompt"], 0.3);
        assert_eq!(body["provider"]["max_price"]["completion"], 1.2);
        cache
            .state
            .lock()
            .ok()
            .ok_or("pricing state")?
            .endpoint_checked_at
            .insert(
                "synthetic/chat".to_owned(),
                Instant::now() - Duration::from_secs(301),
            );
        assert_eq!(cache.pricing("synthetic/chat")?, Some(pricing));
        let deadline = Instant::now() + Duration::from_secs(2);
        while !cache
            .state
            .lock()
            .ok()
            .ok_or("pricing state")?
            .endpoint_retry_at
            .contains_key("synthetic/chat")
        {
            assert!(Instant::now() < deadline, "endpoint refresh did not finish");
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(cache.pricing("synthetic/chat")?, Some(pricing));
        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn transcription_pricing_uses_the_catalog_and_keeps_the_last_cache_on_refresh_failure()
    -> TestResult {
        let catalog = json!({
            "data": [{
                "id": OPENROUTER_TRANSCRIPTION_MODEL,
                "architecture": {
                    "modality": "audio->transcription",
                    "output_modalities": ["transcription"]
                },
                "pricing": {"prompt": "0.1", "completion": "0"}
            }]
        })
        .to_string();
        let served = serve_sequence(vec![
            ("200 OK".to_owned(), "application/json".to_owned(), catalog),
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                r#"{"data":[]}"#.to_owned(),
            ),
            (
                "503 Service Unavailable".to_owned(),
                "application/json".to_owned(),
                "{\"error\":{\"message\":\"synthetic outage\"}}".to_owned(),
            ),
            (
                "503 Service Unavailable".to_owned(),
                "application/json".to_owned(),
                "{\"error\":{\"message\":\"synthetic outage\"}}".to_owned(),
            ),
        ]);
        let (base_url, server) = served?;
        let cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        let first = cache
            .transcription_pricing(OPENROUTER_TRANSCRIPTION_MODEL)?
            .ok_or("unexpected missing value")?;
        assert_eq!(first.usd_micros_per_hour, 100_000);

        cache
            .state
            .lock()
            .ok()
            .ok_or("unexpected error")?
            .fetched_at = Some(Instant::now() - Duration::from_secs(301));
        let cached = cache
            .transcription_pricing(OPENROUTER_TRANSCRIPTION_MODEL)?
            .ok_or("unexpected missing value")?;
        assert_eq!(cached, first);
        wait_for_background_refresh(&cache);
        cache
            .state
            .lock()
            .ok()
            .ok_or("unexpected error")?
            .fetched_at = Some(Instant::now() - Duration::from_secs(301));
        let outage_fallback = cache
            .transcription_pricing(OPENROUTER_TRANSCRIPTION_MODEL)?
            .ok_or("unexpected missing value")?;
        wait_for_background_refresh(&cache);
        assert_eq!(outage_fallback, first);
        cache
            .state
            .lock()
            .ok()
            .ok_or("unexpected error")?
            .fetched_at = Some(Instant::now() - Duration::from_secs(301));
        cache
            .state
            .lock()
            .ok()
            .ok_or("unexpected error")?
            .refresh_retry_at = Some(Instant::now() - Duration::from_secs(1));
        let second_outage_fallback = cache
            .transcription_pricing(OPENROUTER_TRANSCRIPTION_MODEL)?
            .ok_or("unexpected missing value")?;
        wait_for_background_refresh(&cache);
        assert_eq!(second_outage_fallback, first);
        assert_eq!(
            cache
                .state
                .lock()
                .ok()
                .ok_or("unexpected error")?
                .refresh_retry_delay,
            Duration::from_secs(60)
        );
        let retry_at = cache
            .state
            .lock()
            .ok()
            .ok_or("unexpected error")?
            .refresh_retry_at
            .ok_or("unexpected missing value")?;
        assert!(retry_at > Instant::now());
        let repeated_fallback = cache
            .transcription_pricing(OPENROUTER_TRANSCRIPTION_MODEL)?
            .ok_or("unexpected missing value")?;
        assert_eq!(repeated_fallback, first);
        assert_eq!(
            cache
                .state
                .lock()
                .ok()
                .ok_or("unexpected error")?
                .refresh_retry_at,
            Some(retry_at)
        );
        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    fn wait_for_background_refresh(cache: &OpenRouterPricingCache) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while cache.refreshing.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!cache.refreshing.load(Ordering::Acquire));
    }

    #[test]
    fn fails_closed_when_catalog_has_no_model_pricing() -> TestResult {
        let served = serve_once("200", "application/json", r#"{"data":[]}"#);
        let (base_url, server) = served?;
        let cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        let mut request = ChatCompletionRequest::new(DEEPSEEK_MODEL, Vec::new());
        assert!(matches!(
            cache.apply_to_request(&mut request),
            Err(OpenRouterChatError::InvalidJson(message))
                if message == "models response has no usable pricing entries"
        ));
        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn rejects_missing_keys_and_invalid_catalog_urls() -> TestResult {
        let missing_key = OpenRouterPricingCache::new("", "https://openrouter.ai/api/v1")?;
        assert_eq!(
            missing_key.pricing(DEEPSEEK_MODEL),
            Err(OpenRouterChatError::MissingApiKey)
        );

        let invalid_url = OpenRouterPricingCache::new("synthetic-key", "not-a-url")?;
        assert_eq!(
            invalid_url.pricing(DEEPSEEK_MODEL),
            Err(OpenRouterChatError::InvalidBaseUrl)
        );
        assert_eq!(invalid_url.pricing(" "), Ok(None));

        let invalid_scheme =
            OpenRouterPricingCache::new("synthetic-key", "ftp://openrouter.example.test/api/v1")?;
        assert_eq!(
            invalid_scheme.pricing(DEEPSEEK_MODEL),
            Err(OpenRouterChatError::InvalidBaseUrl)
        );
        Ok(())
    }

    #[test]
    fn refresh_failures_without_a_cached_catalog_are_backed_off_and_remain_fail_closed()
    -> TestResult {
        let served = serve_once(
            "503 Service Unavailable",
            "application/json",
            "{\"error\":{\"message\":\"synthetic outage\"}}",
        );
        let (base_url, server) = served?;
        let cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        assert_eq!(
            cache.pricing(DEEPSEEK_MODEL),
            Err(OpenRouterChatError::Http {
                status_code: 503,
                message: "synthetic outage".to_owned(),
            })
        );
        assert_eq!(
            cache.pricing(DEEPSEEK_MODEL),
            Err(OpenRouterChatError::Http {
                status_code: 503,
                message: "synthetic outage".to_owned(),
            })
        );
        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn refresh_rejects_malformed_and_unusable_catalog_entries() -> TestResult {
        let served = serve_sequence(vec![
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                "{}".to_owned(),
            ),
            (
                "200 OK".to_owned(),
                "application/json".to_owned(),
                r#"{"data":[{"id":"synthetic/model"}]}"#.to_owned(),
            ),
        ]);
        let (base_url, server) = served?;
        let cache = OpenRouterPricingCache::new("synthetic-key", &base_url)?;
        assert_eq!(
            cache.refresh(),
            Err(OpenRouterChatError::InvalidJson(
                "models response has no data array".to_owned()
            ))
        );
        assert_eq!(
            cache.refresh(),
            Err(OpenRouterChatError::InvalidJson(
                "models response has no usable pricing entries".to_owned()
            ))
        );
        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn scales_decimal_catalog_rates_without_float_rounding() {
        assert_eq!(
            super::decimal_rate_to_usd_micros_per_million("1"),
            Some(1_000_000_000_000)
        );
        assert_eq!(
            super::decimal_rate_to_usd_micros_per_million("1e-6"),
            Some(1_000_000)
        );
        assert_eq!(
            super::decimal_rate_to_usd_micros_per_million("1e1"),
            Some(10_000_000_000_000)
        );
        assert_eq!(
            super::decimal_rate_to_usd_micros_per_million("+0.00000015"),
            Some(150_000)
        );
        assert_eq!(super::decimal_rate_to_usd_micros_per_million("-1"), None);
        assert_eq!(
            super::decimal_rate_to_usd_micros_per_million("not-a-rate"),
            None
        );
    }

    #[test]
    fn only_converts_the_known_hourly_transcription_model() -> TestResult {
        let mai = parse_catalog_model(&json!({
            "id": OPENROUTER_TRANSCRIPTION_MODEL,
            "architecture": {"modality": "audio->transcription"},
            "pricing": {"prompt": "0.1", "completion": "0"}
        }))
        .ok_or("unexpected missing value")?;
        assert_eq!(
            mai.2.map(|pricing| pricing.usd_micros_per_hour),
            Some(100_000)
        );

        let token_priced = parse_catalog_model(&json!({
            "id": "openai/gpt-transcribe",
            "architecture": {"modality": "audio->transcription"},
            "pricing": {"prompt": "0.0045", "completion": "0"}
        }))
        .ok_or("unexpected missing value")?;
        assert_eq!(token_priced.2, None);
        Ok(())
    }

    #[test]
    fn rejects_oversized_buffered_responses() -> TestResult {
        let body = "x".repeat((OPENROUTER_MAX_RESPONSE_BYTES + 1) as usize);
        let served = serve_once("200", "text/plain", &body);
        let (base_url, server) = served?;
        let transport = ReqwestOpenRouterTransport::new()?;
        assert_eq!(
            complete_with(&transport, "synthetic-key", &base_url, &request()),
            Err(OpenRouterChatError::ResponseTooLarge)
        );
        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn sends_authenticated_typed_request_and_normalizes_usage() {
        let transport = Transport {
            response: RefCell::new(Some(Ok(response(
                200,
                json!({
                    "id": "generation-1",
                    "model": "resolved/model",
                    "provider": "SyntheticProvider",
                    "service_tier": "priority",
                    "choices": [{
                        "message": {
                            "content": "synthetic answer",
                            "annotations": [{"type": "url_citation"}]
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4}
                }),
            )))),
            requests: RefCell::new(Vec::new()),
        };
        let actual = complete_with(
            &transport,
            " synthetic-key ",
            "https://openrouter.example/api/v1/",
            &request(),
        );
        assert_eq!(
            actual.as_ref().map(|completion| completion.text.as_str()),
            Ok("synthetic answer")
        );
        assert_eq!(
            actual.as_ref().map(|completion| completion.model.as_str()),
            Ok("resolved/model")
        );
        assert_eq!(
            actual
                .as_ref()
                .map(|completion| completion.upstream_provider.as_deref()),
            Ok(Some("SyntheticProvider"))
        );
        assert_eq!(
            actual
                .as_ref()
                .map(|completion| completion.service_tier.as_deref()),
            Ok(Some("priority"))
        );
        assert_eq!(
            actual
                .as_ref()
                .map(|completion| completion.usage["prompt_tokens"].clone()),
            Ok(json!(10))
        );
        let requests = transport.requests.borrow();
        assert_eq!(
            requests[0].url,
            "https://openrouter.example/api/v1/chat/completions"
        );
        assert_eq!(requests[0].bearer_token, "synthetic-key");
        let body: Value = serde_json::from_str(&requests[0].body).unwrap_or(Value::Null);
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn reqwest_transport_reads_blocking_and_streaming_http_responses() -> TestResult {
        let completion_body = json!({
            "choices":[{"message":{"content":"synthetic response"}}]
        })
        .to_string();
        let served = serve_once("200 OK", "application/json", &completion_body);
        let (base_url, server) = served?;
        let transport = ReqwestOpenRouterTransport::new()?;
        let completion = complete_with(&transport, "synthetic-key", &base_url, &request());
        assert!(matches!(completion, Ok(ref value) if value.text == "synthetic response"));
        assert!(matches!(server.join(), Ok(Ok(()))));

        let stream_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let served = serve_once("200 OK", "text/event-stream", stream_body);
        let (base_url, server) = served?;
        let mut streaming_request = request();
        streaming_request.stream = true;
        let (streamed, events) = stream_events(
            &transport,
            "synthetic-key",
            &base_url,
            &streaming_request,
            None,
        );
        assert!(streamed.is_ok());
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ChatStreamEvent::Done))
        );
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn preserves_function_tool_calls_and_builds_followup_messages() {
        let call = ToolCall {
            id: "call-1".to_owned(),
            call_type: "function".to_owned(),
            function: ToolFunctionCall {
                name: "weather".to_owned(),
                arguments: "{\"location\":\"Synthetic City\"}".to_owned(),
            },
        };
        let completion = parse_chat_completion(
            response(
                200,
                json!({
                    "choices": [{"message": {"content": null, "tool_calls": [call.clone()]}}]
                }),
            ),
            "synthetic/model",
        );
        assert!(completion.is_ok());
        assert_eq!(
            completion.map(|value| value.tool_calls),
            Ok(vec![call.clone()])
        );
        let assistant = serde_json::to_value(ChatMessage::assistant_tool_calls(vec![call]));
        assert_eq!(
            assistant.unwrap_or(Value::Null)["tool_calls"][0]["function"]["name"],
            "weather"
        );
        let result = serde_json::to_value(ChatMessage::tool_result("call-1", "sunny"));
        assert_eq!(result.unwrap_or(Value::Null)["tool_call_id"], "call-1");
    }

    #[test]
    fn joins_text_parts_and_uses_requested_model_when_response_omits_it() {
        let actual = parse_chat_completion(
            response(
                200,
                json!({
                    "choices": [{"message": {"content": [
                        {"type": "text", "text": "hello "},
                        {"type": "image", "url": "ignored"},
                        {"type": "text", "text": "world"}
                    ]}}]
                }),
            ),
            "requested/model",
        );
        assert!(actual.is_ok());
        assert_eq!(
            actual.map(|value| (value.text, value.model)),
            Ok(("hello world".into(), "requested/model".into()))
        );
    }

    #[test]
    fn classifies_rate_limits_with_retry_after_and_safe_error_message() {
        let mut headers = BTreeMap::new();
        headers.insert("retry-after".to_owned(), "17".to_owned());
        assert_eq!(
            parse_chat_completion(
                HttpResponse {
                    status_code: 429,
                    body: json!({"error": {"message": "capacity exhausted"}}).to_string(),
                    headers,
                },
                "model",
            ),
            Err(OpenRouterChatError::RateLimited {
                retry_after_seconds: Some(17),
                message: "capacity exhausted".to_owned(),
            })
        );
        assert_eq!(
            parse_chat_completion(response(503, json!({})), "model"),
            Err(OpenRouterChatError::Http {
                status_code: 503,
                message: "upstream request failed".to_owned(),
            })
        );
    }

    #[test]
    fn malformed_json_choices_content_and_empty_outputs_are_distinct() {
        assert!(matches!(
            parse_chat_completion(
                HttpResponse {
                    status_code: 200,
                    body: "not-json".to_owned(),
                    headers: BTreeMap::new(),
                },
                "model"
            ),
            Err(OpenRouterChatError::InvalidJson(_))
        ));
        for payload in [
            json!({"choices": []}),
            json!({"choices": [{"message": {"content": {"bad": true}}}]}),
            json!({"choices": [{"message": {"content": ""}}]}),
        ] {
            assert_eq!(
                parse_chat_completion(response(200, payload), "model"),
                Err(OpenRouterChatError::MalformedResponse)
            );
        }
    }

    #[test]
    fn rejects_credentials_model_url_and_stream_mode_before_transport() {
        for (key, base_url, mut request, expected) in [
            (
                "",
                "https://example.com",
                request(),
                OpenRouterChatError::MissingApiKey,
            ),
            (
                "key",
                "https://example.com",
                ChatCompletionRequest::new("", Vec::new()),
                OpenRouterChatError::MissingModel,
            ),
            (
                "key",
                "file:///tmp/provider",
                request(),
                OpenRouterChatError::InvalidBaseUrl,
            ),
        ] {
            let transport = Transport {
                response: RefCell::new(None),
                requests: RefCell::new(Vec::new()),
            };
            assert_eq!(
                complete_with(&transport, key, base_url, &request),
                Err(expected)
            );
            assert!(transport.requests.borrow().is_empty());
            request.stream = false;
        }
        let mut streaming = request();
        streaming.stream = true;
        let transport = Transport {
            response: RefCell::new(None),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            complete_with(&transport, "key", "https://example.com", &streaming),
            Err(OpenRouterChatError::MalformedResponse)
        );
    }

    #[test]
    fn transport_failures_propagate_without_leaking_the_api_key() {
        let transport = Transport {
            response: RefCell::new(Some(Err(OpenRouterChatError::Transport(
                "synthetic timeout".to_owned(),
            )))),
            requests: RefCell::new(Vec::new()),
        };
        let actual = complete_with(
            &transport,
            "synthetic-secret",
            "https://example.com/api/v1",
            &request(),
        );
        assert_eq!(
            actual,
            Err(OpenRouterChatError::Transport(
                "synthetic timeout".to_owned()
            ))
        );
        assert!(!format!("{actual:?}").contains("synthetic-secret"));
    }

    #[test]
    fn incremental_sse_preserves_text_tool_fragments_usage_and_metadata() -> TestResult {
        let body = [
            ": keepalive\r\n\r\n".to_owned(),
            format!(
                "data: {}\r\n\r\n",
                json!({
                    "id": "gen-1",
                    "model": "resolved/model",
                    "provider": "Synthetic",
                    "service_tier": "paid",
                    "choices": [{"delta": {
                        "content": "holá ",
                        "reasoning": "thinking"
                    }}]
                })
            ),
            format!(
                "data: {}\n\n",
                json!({
                    "choices": [{"delta": {
                        "content": "mundo",
                        "reasoning_content": "legacy",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "wea", "arguments": "{\"city\":"}
                        }]
                    }}]
                })
            ),
            format!(
                "data: {}\n\n",
                json!({
                    "choices": [{
                        "delta": {"tool_calls": [{
                            "index": "0",
                            "function": {"name": "ther", "arguments": "\"Synthetic\"}"}
                        }]},
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "cost": "0.001"
                    }
                })
            ),
            format!("data: {}\n\n", json!({"id": "heartbeat"})),
            "data: [DONE]\n\n".to_owned(),
        ]
        .concat();
        let chunks = body
            .as_bytes()
            .chunks(7)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let transport = StreamTransport {
            chunks,
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let mut request = request();
        request.stream = true;
        let (result, events) = stream_events(
            &transport,
            " synthetic-key ",
            "https://openrouter.example/api/v1/",
            &request,
            None,
        );
        assert_eq!(result, Ok(()));
        assert_eq!(events.len(), 5);
        let first = stream_chunk(&events[0]).ok_or("unexpected missing value")?;
        assert_eq!(first.text, "holá ");
        assert_eq!(first.reasoning, "thinking");
        assert_eq!(first.generation_id.as_deref(), Some("gen-1"));
        assert_eq!(first.model.as_deref(), Some("resolved/model"));
        let second = stream_chunk(&events[1]).ok_or("unexpected missing value")?;
        assert_eq!(second.text, "mundo");
        assert_eq!(second.reasoning, "legacy");
        assert_eq!(second.tool_call_fragments[0].name.as_deref(), Some("wea"));
        let final_chunk = stream_chunk(&events[2]).ok_or("unexpected missing value")?;
        assert_eq!(
            final_chunk.tool_call_fragments[0].arguments.as_deref(),
            Some("\"Synthetic\"}")
        );
        assert_eq!(final_chunk.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(final_chunk.usage["cost"], "0.001");
        assert!(matches!(
            &events[3],
            ChatStreamEvent::Chunk(chunk)
                if chunk.text.is_empty()
                    && chunk.reasoning.is_empty()
                    && chunk.reasoning_details.is_empty()
        ));
        assert_eq!(events[4], ChatStreamEvent::Done);
        let requests = transport.requests.borrow();
        assert_eq!(
            requests[0].url,
            "https://openrouter.example/api/v1/chat/completions"
        );
        assert_eq!(requests[0].bearer_token, "synthetic-key");
        let request_body: Value = serde_json::from_str(&requests[0].body).unwrap_or(Value::Null);
        assert_eq!(request_body["stream"], true);
        Ok(())
    }

    #[test]
    fn stream_reports_provider_errors_interruption_and_consumer_failure() {
        let mut request = request();
        request.stream = true;
        for (body, expected) in [
            (
                "data: {\"error\":{\"message\":\"provider exploded\"}}\n\n",
                OpenRouterChatError::Stream("provider exploded".to_owned()),
            ),
            (
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
                OpenRouterChatError::IncompleteStream,
            ),
        ] {
            let transport = StreamTransport {
                chunks: vec![body.as_bytes().to_vec()],
                failure: None,
                requests: RefCell::new(Vec::new()),
            };
            assert_eq!(
                stream_events(&transport, "key", "https://example.com", &request, None).0,
                Err(expected)
            );
        }

        let transport = StreamTransport {
            chunks: vec![b"data: [DONE]\n\n".to_vec()],
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let (result, events) = stream_events(
            &transport,
            "key",
            "https://example.com",
            &request,
            Some(OpenRouterChatError::Stream("consumer stopped".to_owned())),
        );
        assert_eq!(
            result,
            Err(OpenRouterChatError::Stream("consumer stopped".to_owned()))
        );
        assert_eq!(events, [ChatStreamEvent::Done]);
    }

    #[test]
    fn stream_validates_mode_and_propagates_transport_failure() {
        let transport = StreamTransport {
            chunks: Vec::new(),
            failure: Some(OpenRouterChatError::Transport("timeout".to_owned())),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            stream_events(&transport, "key", "https://example.com", &request(), None).0,
            Err(OpenRouterChatError::MalformedResponse)
        );
        let mut streaming = request();
        streaming.stream = true;
        assert_eq!(
            stream_events(&transport, "key", "https://example.com", &streaming, None).0,
            Err(OpenRouterChatError::Transport("timeout".to_owned()))
        );
        let mut unnamed = streaming.clone();
        unnamed.model = "  ".to_owned();
        assert_eq!(
            stream_events(&transport, "key", "https://example.com", &unnamed, None).0,
            Err(OpenRouterChatError::MissingModel)
        );
        assert_eq!(
            stream_events(&transport, "  ", "https://example.com", &streaming, None).0,
            Err(OpenRouterChatError::MissingApiKey)
        );
        assert_eq!(transport.requests.borrow().len(), 1);
        assert_eq!(stream_chunk(&ChatStreamEvent::Done), None);
    }

    #[test]
    fn stream_parses_an_unterminated_final_frame_and_rejects_choice_errors() {
        let mut streaming = request();
        streaming.stream = true;
        let transport = StreamTransport {
            chunks: vec![
                b"data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}\n\n".to_vec(),
                b"data: [DONE]".to_vec(),
            ],
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let (result, events) =
            stream_events(&transport, "key", "https://example.com", &streaming, None);
        assert_eq!(result, Ok(()));
        assert_eq!(events.len(), 2);
        assert_eq!(
            stream_chunk(&events[0]).map(|chunk| chunk.text.as_str()),
            Some("tail")
        );
        assert_eq!(events[1], ChatStreamEvent::Done);

        for (body, expected) in [
            (
                "data: {\"choices\":[{\"error\":{\"message\":\"choice exploded\"}}]}\n\n",
                OpenRouterChatError::Stream("choice exploded".to_owned()),
            ),
            (
                "data: {\"choices\":[{\"delta\":{\"content\":5}}]}\n\n",
                OpenRouterChatError::MalformedResponse,
            ),
            (
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {\"error\":\"late failure\"}",
                OpenRouterChatError::Stream("late failure".to_owned()),
            ),
        ] {
            let transport = StreamTransport {
                chunks: vec![body.as_bytes().to_vec()],
                failure: None,
                requests: RefCell::new(Vec::new()),
            };
            let (result, events) =
                stream_events(&transport, "key", "https://example.com", &streaming, None);
            assert_eq!(result, Err(expected));
            assert!(events.iter().all(|event| event != &ChatStreamEvent::Done));
        }

        // A trailing frame after [DONE] without a terminating blank line is
        // still delivered to the consumer.
        let transport = StreamTransport {
            chunks: vec![
                b"data: [DONE]\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}"
                    .to_vec(),
            ],
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let (result, events) =
            stream_events(&transport, "key", "https://example.com", &streaming, None);
        assert_eq!(result, Ok(()));
        assert_eq!(events[0], ChatStreamEvent::Done);
        assert_eq!(
            events
                .get(1)
                .and_then(stream_chunk)
                .map(|chunk| chunk.text.as_str()),
            Some("late")
        );

        // An unterminated trailing comment carries no event.
        let transport = StreamTransport {
            chunks: vec![b"data: [DONE]\n\n: keepalive".to_vec()],
            failure: None,
            requests: RefCell::new(Vec::new()),
        };
        let (result, events) =
            stream_events(&transport, "key", "https://example.com", &streaming, None);
        assert_eq!(result, Ok(()));
        assert_eq!(events, [ChatStreamEvent::Done]);
    }

    #[test]
    fn reqwest_stream_transport_reports_http_failures_with_the_provider_message() -> TestResult {
        let served = serve_once(
            "503 Service Unavailable",
            "application/json",
            r#"{"error":{"message":"synthetic overload"}}"#,
        );
        let (base_url, server) = served?;
        let transport = ReqwestOpenRouterTransport::new()?;
        let mut streaming = request();
        streaming.stream = true;
        let (result, events) =
            stream_events(&transport, "synthetic-key", &base_url, &streaming, None);
        assert!(
            matches!(
                &result,
                Err(OpenRouterChatError::Http { status_code: 503, message })
                    if message.contains("synthetic overload")
            ),
            "{result:?}"
        );
        assert!(events.is_empty());
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn catalog_parsing_skips_malformed_overrides_and_detects_transcription_outputs() -> TestResult {
        let model = json!({
            "id": OPENROUTER_TRANSCRIPTION_MODEL,
            "architecture": {"output_modalities": ["text", "transcription"]},
            "pricing": {
                "prompt": "0.1",
                "completion": "0",
                "overrides": ["malformed", {"prompt": "0.2"}]
            }
        });
        let (id, pricing, transcription) =
            parse_catalog_model(&model).ok_or("unexpected missing value")?;
        assert_eq!(id, OPENROUTER_TRANSCRIPTION_MODEL);
        assert_eq!(pricing.input_per_million, 200_000_000_000);
        assert!(transcription.is_some());

        let text_only = json!({
            "id": OPENROUTER_TRANSCRIPTION_MODEL,
            "architecture": {"output_modalities": ["text"]},
            "pricing": {"prompt": "0.1", "completion": "0"}
        });
        assert!(
            parse_catalog_model(&text_only)
                .is_some_and(|(_, _, transcription)| transcription.is_none())
        );
        for invalid in ["1.2.3", "abc", ".", "1a.5"] {
            assert_eq!(
                super::decimal_rate_to_usd_micros_per_million(invalid),
                None,
                "{invalid}"
            );
        }
        Ok(())
    }

    #[test]
    fn cached_prices_answer_immediately_and_refresh_in_the_background() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            thread::sleep(Duration::from_millis(80));
            let _ = stream.read(&mut [0_u8; 1_024]);
            Ok(())
        });
        let cache = OpenRouterPricingCache::new("synthetic-key", &format!("http://{address}"))?;
        let cached = super::TokenPricing {
            input_per_million: 1_000_000,
            cached_input_per_million: None,
            cache_write_per_million: None,
            audio_input_per_million: None,
            output_per_million: 2_000_000,
        };
        {
            let mut state = cache.state.lock().ok().ok_or("pricing state")?;
            state.models.insert("synthetic/model".to_owned(), cached);
            // The catalog refresh owns the one accepted connection. A fresh
            // endpoint check would race it and finish the refresh too soon.
            state
                .endpoint_checked_at
                .insert("synthetic/model".to_owned(), Instant::now());
        }
        assert_eq!(cache.pricing("synthetic/model")?, Some(cached));
        let deadline = Instant::now() + Duration::from_secs(10);
        while cache.refreshing.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!cache.refreshing.load(Ordering::Acquire));
        let state = cache.state.lock().ok().ok_or("pricing state")?;
        assert!(matches!(
            state.last_refresh_error,
            Some(OpenRouterChatError::Transport(_))
        ));
        assert!(state.refresh_retry_at.is_some());
        assert_eq!(state.models.get("synthetic/model"), Some(&cached));
        drop(state);
        server.join().ok().ok_or("server thread panicked")??;
        Ok(())
    }

    #[test]
    fn numeric_catalog_rates_and_request_encoding_errors_are_typed() {
        assert_eq!(
            super::parse_catalog_rate(&json!(0.000_000_3)),
            Some(300_000)
        );
        assert_eq!(super::parse_catalog_rate(&json!(true)), None);
        let encoding = serde_json::from_str::<Value>("{")
            .err()
            .map(super::request_json_error);
        assert!(matches!(
            encoding,
            Some(OpenRouterChatError::RequestJson(detail)) if detail.contains("EOF")
        ));
    }

    #[test]
    fn reqwest_transports_report_unreachable_providers_as_transport_errors() -> TestResult {
        let closed = TcpListener::bind("127.0.0.1:0")?.local_addr()?;
        let transport = ReqwestOpenRouterTransport::new()?;
        assert!(matches!(
            complete_with(
                &transport,
                "synthetic-key",
                &format!("http://{closed}"),
                &request()
            ),
            Err(OpenRouterChatError::Transport(_))
        ));
        let mut streaming = request();
        streaming.stream = true;
        let (result, events) = stream_events(
            &transport,
            "synthetic-key",
            &format!("http://{closed}"),
            &streaming,
            None,
        );
        assert!(matches!(result, Err(OpenRouterChatError::Transport(_))));
        assert!(events.is_empty());
        Ok(())
    }

    #[test]
    fn background_refresh_is_single_flight_and_a_poisoned_cache_fails_closed() -> TestResult {
        let cache = OpenRouterPricingCache::new("synthetic-key", "http://127.0.0.1:1")?;
        cache.refreshing.store(true, Ordering::Release);
        cache.refresh_in_background();
        assert!(cache.refreshing.load(Ordering::Acquire));
        assert!(
            cache
                .state
                .lock()
                .is_ok_and(|state| state.last_refresh_error.is_none() && state.fetched_at.is_none())
        );
        cache
            .state
            .lock()
            .ok()
            .ok_or("pricing state")?
            .models
            .insert(
                "synthetic/model".to_owned(),
                super::TokenPricing {
                    input_per_million: 1,
                    cached_input_per_million: None,
                    cache_write_per_million: None,
                    audio_input_per_million: None,
                    output_per_million: 1,
                },
            );
        cache.endpoint_refreshing.store(true, Ordering::Release);
        cache.spawn_endpoint_refresh("synthetic/model");
        assert!(cache.endpoint_refreshing.load(Ordering::Acquire));

        let poisoner = cache.clone();
        let poisoned = thread::spawn(move || {
            let _guard = poisoner.state.lock();
            unreachable!("synthetic failure while holding the pricing lock");
        })
        .join();
        assert!(poisoned.is_err());
        assert_eq!(
            cache.pricing("synthetic/model"),
            Err(OpenRouterChatError::Transport(
                "OpenRouter pricing cache was poisoned".to_owned()
            ))
        );
        Ok(())
    }
}
