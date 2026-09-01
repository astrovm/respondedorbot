//! Exact AI usage pricing over adapter-normalized JSON segments.

use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::provider_pricing::{
    CREDIT_UNIT_USD_MICROS, FIRECRAWL_STANDARD_USD_MICROS_PER_CREDIT,
    GROQ_TRANSCRIPTION_MIN_SECONDS, GROQ_TRANSCRIPTION_USD_MICROS_PER_HOUR, PRICING_VERSION,
    published_token_pricing,
};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AiPricingError {
    #[error("AI pricing input must be an array of objects or null values")]
    InvalidSegments,
    #[error("AI pricing input contains a value with incompatible type")]
    InvalidValue,
    #[error("AI pricing calculation exceeds the supported numeric range")]
    Overflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactDecimal {
    coefficient: i128,
    scale: u32,
}

impl ExactDecimal {
    const ZERO: Self = Self {
        coefficient: 0,
        scale: 0,
    };

    fn parse(value: &Value) -> Option<Self> {
        match value {
            Value::String(value) => Self::parse_str(value),
            Value::Number(value) => Self::parse_str(&value.to_string()),
            _ => None,
        }
    }

    fn parse_str(raw: &str) -> Option<Self> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        let (negative, unsigned) = raw
            .strip_prefix('-')
            .map_or((false, raw), |value| (true, value));
        let unsigned = unsigned.strip_prefix('+').unwrap_or(unsigned);
        let (mantissa, exponent) =
            if let Some((mantissa, exponent)) = unsigned.split_once(['e', 'E']) {
                (mantissa, exponent.parse::<i32>().ok()?)
            } else {
                (unsigned, 0_i32)
            };
        let (whole, fractional) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if whole.is_empty() && fractional.is_empty() {
            return None;
        }
        if !whole.bytes().all(|value| value.is_ascii_digit())
            || !fractional.bytes().all(|value| value.is_ascii_digit())
        {
            return None;
        }
        let digits = format!("{whole}{fractional}");
        let mut coefficient = if digits.is_empty() {
            0
        } else {
            digits.parse::<i128>().ok()?
        };
        if negative {
            coefficient = coefficient.checked_neg()?;
        }
        let fractional_digits = i32::try_from(fractional.len()).ok()?;
        let resulting_scale = fractional_digits.checked_sub(exponent)?;
        if resulting_scale >= 0 {
            Some(Self {
                coefficient,
                scale: u32::try_from(resulting_scale).ok()?,
            })
        } else {
            Some(Self {
                coefficient: coefficient
                    .checked_mul(pow10(u32::try_from(resulting_scale.checked_neg()?).ok()?)?)?,
                scale: 0,
            })
        }
    }

    fn from_ratio(numerator: i128, denominator_scale: u32) -> Self {
        let mut coefficient = numerator;
        let mut scale = denominator_scale;
        while scale > 0 && coefficient % 10 == 0 {
            coefficient /= 10;
            scale -= 1;
        }
        Self { coefficient, scale }
    }

    fn multiply_integer(self, multiplier: i128) -> Result<Self, AiPricingError> {
        Ok(Self {
            coefficient: self
                .coefficient
                .checked_mul(multiplier)
                .ok_or(AiPricingError::Overflow)?,
            scale: self.scale,
        })
    }

    fn add(self, other: Self) -> Result<Self, AiPricingError> {
        let scale = self.scale.max(other.scale);
        let left = self
            .coefficient
            .checked_mul(pow10(scale - self.scale).ok_or(AiPricingError::Overflow)?)
            .ok_or(AiPricingError::Overflow)?;
        let right = other
            .coefficient
            .checked_mul(pow10(scale - other.scale).ok_or(AiPricingError::Overflow)?)
            .ok_or(AiPricingError::Overflow)?;
        Ok(Self {
            coefficient: left.checked_add(right).ok_or(AiPricingError::Overflow)?,
            scale,
        })
    }

    fn is_positive(self) -> bool {
        self.coefficient > 0
    }

    fn floor_i64(self) -> Result<i64, AiPricingError> {
        let divisor = pow10(self.scale).ok_or(AiPricingError::Overflow)?;
        let quotient = self.coefficient.div_euclid(divisor);
        i64::try_from(quotient).map_err(|_| AiPricingError::Overflow)
    }

    fn ceil_credit_units(self) -> Result<i64, AiPricingError> {
        if !self.is_positive() {
            return Ok(0);
        }
        let divisor = pow10(self.scale)
            .and_then(|value| value.checked_mul(CREDIT_UNIT_USD_MICROS))
            .ok_or(AiPricingError::Overflow)?;
        let units = self
            .coefficient
            .checked_add(divisor - 1)
            .ok_or(AiPricingError::Overflow)?
            / divisor;
        i64::try_from(units).map_err(|_| AiPricingError::Overflow)
    }

    fn fixed_string(self) -> Result<String, AiPricingError> {
        if self.scale == 0 {
            return Ok(self.coefficient.to_string());
        }
        let divisor = pow10(self.scale).ok_or(AiPricingError::Overflow)?;
        let negative = self.coefficient < 0;
        let magnitude = self.coefficient.unsigned_abs();
        let divisor = u128::try_from(divisor).map_err(|_| AiPricingError::Overflow)?;
        let whole = magnitude / divisor;
        let fraction = magnitude % divisor;
        let width = usize::try_from(self.scale).map_err(|_| AiPricingError::Overflow)?;
        Ok(format!(
            "{}{whole}.{fraction:0width$}",
            if negative { "-" } else { "" }
        ))
    }
}

fn pow10(exponent: u32) -> Option<i128> {
    10_i128.checked_pow(exponent)
}

#[derive(Debug, Clone, Copy)]
struct TokenUsage {
    input_tokens: i64,
    input_cached_tokens: i64,
    input_non_cached_tokens: i64,
    output_tokens: i64,
}

impl TokenUsage {
    fn json_fields(self) -> Map<String, Value> {
        Map::from_iter([
            ("input_tokens".to_owned(), json!(self.input_tokens)),
            (
                "input_cached_tokens".to_owned(),
                json!(self.input_cached_tokens),
            ),
            (
                "input_non_cached_tokens".to_owned(),
                json!(self.input_non_cached_tokens),
            ),
            ("output_tokens".to_owned(), json!(self.output_tokens)),
        ])
    }

    fn has_tokens(self) -> bool {
        self.input_tokens != 0 || self.output_tokens != 0
    }
}

struct ModelCost {
    usd_micros: i64,
    exact: ExactDecimal,
    pricing_basis: &'static str,
    tokens: TokenUsage,
}

fn python_string(value: Option<&Value>) -> Result<String, AiPricingError> {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::Bool(true)) => Ok("True".to_owned()),
        Some(Value::Number(value)) => Ok(value.to_string()),
        _ => Err(AiPricingError::InvalidValue),
    }
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Number(number)) => number.as_f64() != Some(0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
        Some(Value::Bool(true)) => true,
    }
}

fn python_int(value: Option<&Value>) -> Result<i64, AiPricingError> {
    if !truthy(value) {
        return Ok(0);
    }
    match value {
        Some(Value::Bool(true)) => Ok(1),
        Some(Value::Number(value)) => value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| value.as_f64().map(|value| value.trunc() as i64))
            .ok_or(AiPricingError::InvalidValue),
        Some(Value::String(value)) => value
            .trim()
            .parse::<i64>()
            .map_err(|_| AiPricingError::InvalidValue),
        _ => Err(AiPricingError::InvalidValue),
    }
}

fn python_float(value: Option<&Value>) -> Result<f64, AiPricingError> {
    if !truthy(value) {
        return Ok(0.0);
    }
    let result = match value {
        Some(Value::Bool(true)) => 1.0,
        Some(Value::Number(value)) => value.as_f64().ok_or(AiPricingError::InvalidValue)?,
        Some(Value::String(value)) => value
            .trim()
            .parse::<f64>()
            .map_err(|_| AiPricingError::InvalidValue)?,
        _ => return Err(AiPricingError::InvalidValue),
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(AiPricingError::InvalidValue)
    }
}

fn object(value: Option<&Value>) -> Option<&Map<String, Value>> {
    value.and_then(Value::as_object)
}

fn token_usage(usage: &Map<String, Value>) -> Result<TokenUsage, AiPricingError> {
    let details = object(usage.get("prompt_tokens_details"));
    let input_tokens = python_int(
        usage
            .get("input_tokens")
            .filter(|value| truthy(Some(value)))
            .or_else(|| usage.get("prompt_tokens")),
    )?;
    let cached = python_int(
        usage
            .get("input_cached_tokens")
            .filter(|value| truthy(Some(value)))
            .or_else(|| details.and_then(|value| value.get("cached_tokens"))),
    )?;
    let output_tokens = python_int(
        usage
            .get("output_tokens")
            .filter(|value| truthy(Some(value)))
            .or_else(|| usage.get("completion_tokens")),
    )?;
    let cached = cached.clamp(0, input_tokens.max(0));
    Ok(TokenUsage {
        input_tokens: input_tokens.max(0),
        input_cached_tokens: cached,
        input_non_cached_tokens: input_tokens.saturating_sub(cached).max(0),
        output_tokens: output_tokens.max(0),
    })
}

fn reported_cost(usage: &Map<String, Value>) -> Result<Option<ExactDecimal>, AiPricingError> {
    let gateway_cost = usage
        .get("cost")
        .and_then(ExactDecimal::parse)
        .map(|cost| cost.multiply_integer(1_000_000))
        .transpose()?
        .filter(|cost| cost.is_positive());
    if gateway_cost.is_some() {
        return Ok(gateway_cost);
    }
    let upstream_cost = if let Some(cost_details) = object(usage.get("cost_details"))
        && let Some(raw_cost) = cost_details.get("upstream_inference_cost")
        && let Some(cost) = ExactDecimal::parse(raw_cost)
    {
        let cost = cost.multiply_integer(1_000_000)?;
        cost.is_positive().then_some(cost)
    } else {
        None
    };
    Ok(upstream_cost)
}

/// Return the local input and cached-input rates used by admin cache reports.
#[must_use]
pub fn model_cache_input_rates(model: &str) -> Option<(i64, i64)> {
    let pricing = published_token_pricing("", model)?;
    let input = i64::try_from(pricing.input_per_million).ok()?;
    let cached = i64::try_from(
        pricing
            .cached_input_per_million
            .unwrap_or(pricing.input_per_million),
    )
    .ok()?;
    Some((input, cached))
}

fn model_cost(
    model: &str,
    usage: &Map<String, Value>,
    provider: &str,
) -> Result<ModelCost, AiPricingError> {
    let tokens = token_usage(usage)?;
    if usage.is_empty() {
        return Ok(ModelCost {
            usd_micros: 0,
            exact: ExactDecimal::ZERO,
            pricing_basis: "missing",
            tokens,
        });
    }
    if let Some(exact) = reported_cost(usage)? {
        return Ok(ModelCost {
            usd_micros: exact.floor_i64()?,
            exact,
            pricing_basis: "provider_reported",
            tokens,
        });
    }
    let local_pricing = (provider != "openrouter")
        .then(|| published_token_pricing(provider, model))
        .flatten();
    let Some(pricing) = local_pricing else {
        return Ok(ModelCost {
            usd_micros: 0,
            exact: ExactDecimal::ZERO,
            pricing_basis: "missing",
            tokens,
        });
    };
    let details = object(usage.get("prompt_tokens_details"));
    let audio = python_int(details.and_then(|value| value.get("audio_tokens")))?.max(0);
    let cache_write = python_int(details.and_then(|value| value.get("cache_write_tokens")))?.max(0);
    let audio = audio.min(tokens.input_non_cached_tokens);
    let cache_write = cache_write.min(tokens.input_non_cached_tokens.saturating_sub(audio));
    let regular = tokens
        .input_non_cached_tokens
        .saturating_sub(audio)
        .saturating_sub(cache_write)
        .max(0);
    let cached_rate = pricing
        .cached_input_per_million
        .unwrap_or(pricing.input_per_million);
    let audio_rate = pricing
        .audio_input_per_million
        .unwrap_or(pricing.input_per_million);
    let cache_write_rate = pricing
        .cache_write_per_million
        .unwrap_or(pricing.input_per_million);
    let numerator = i128::from(regular)
        .checked_mul(pricing.input_per_million)
        .and_then(|value| {
            i128::from(tokens.input_cached_tokens)
                .checked_mul(cached_rate)
                .and_then(|cost| value.checked_add(cost))
        })
        .and_then(|value| {
            i128::from(audio)
                .checked_mul(audio_rate)
                .and_then(|cost| value.checked_add(cost))
        })
        .and_then(|value| {
            i128::from(cache_write)
                .checked_mul(cache_write_rate)
                .and_then(|cost| value.checked_add(cost))
        })
        .and_then(|value| {
            i128::from(tokens.output_tokens)
                .checked_mul(pricing.output_per_million)
                .and_then(|cost| value.checked_add(cost))
        })
        .ok_or(AiPricingError::Overflow)?;
    let exact = ExactDecimal::from_ratio(numerator, 6);
    Ok(ModelCost {
        usd_micros: exact.floor_i64()?,
        exact,
        pricing_basis: "published_rate",
        tokens,
    })
}

fn firecrawl_cost(metadata: &Map<String, Value>) -> Result<(i64, Option<Value>), AiPricingError> {
    let requests = python_int(metadata.get("web_search_requests")).unwrap_or(0);
    let credits = python_int(metadata.get("firecrawl_credits_used"))
        .unwrap_or(0)
        .max(0);
    if credits <= 0 {
        return Ok((0, None));
    }
    let usd_micros = i128::from(credits)
        .checked_mul(FIRECRAWL_STANDARD_USD_MICROS_PER_CREDIT)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(AiPricingError::Overflow)?;
    Ok((
        usd_micros,
        Some(json!({
            "tool": "web_search",
            "count": requests,
            "usd_micros": usd_micros,
        })),
    ))
}

fn transcription_cost(audio_seconds: f64) -> Result<i64, AiPricingError> {
    let seconds = audio_seconds.max(0.0);
    if seconds <= 0.0 {
        return Ok(0);
    }
    let value = (seconds.max(GROQ_TRANSCRIPTION_MIN_SECONDS)
        * GROQ_TRANSCRIPTION_USD_MICROS_PER_HOUR
        / 3_600.0)
        .ceil();
    if value > i64::MAX as f64 {
        return Err(AiPricingError::Overflow);
    }
    Ok(value as i64)
}

fn format_credit_units(units: i64) -> String {
    format!("{}.{:02}", units / 100, units % 100)
}

pub fn calculate_billing_for_segments(segments: &Value) -> Result<Value, AiPricingError> {
    let segments = segments.as_array().ok_or(AiPricingError::InvalidSegments)?;
    let mut total = ExactDecimal::ZERO;
    let mut model_breakdown = Vec::new();
    let mut tool_breakdown = Vec::new();
    let mut segment_breakdown = Vec::new();
    let mut unsupported_notes = Vec::new();

    for (segment_index, raw_segment) in segments
        .iter()
        .filter(|segment| !segment.is_null())
        .enumerate()
    {
        let segment = raw_segment
            .as_object()
            .ok_or(AiPricingError::InvalidSegments)?;
        let source = python_string(segment.get("source"))?;
        let normalized_source = source.trim().to_lowercase();
        let kind = python_string(segment.get("kind"))?;
        let model = python_string(segment.get("model"))?;
        if normalized_source == "cache" {
            segment_breakdown.push(json!({
                "segment_index": segment_index,
                "kind": kind,
                "model": model,
                "provider": "internal",
                "pricing_basis": "internal_cache",
                "cost_complete": true,
                "usd_micros_exact": "0",
            }));
            continue;
        }
        let empty_usage = Map::new();
        let usage = object(segment.get("usage")).unwrap_or(&empty_usage);
        let audio_seconds = python_float(segment.get("audio_seconds"))?;
        let empty_metadata = Map::new();
        let metadata = object(segment.get("metadata")).unwrap_or(&empty_metadata);
        let provider_source = metadata.get("provider").filter(|value| truthy(Some(value)));
        let provider = python_string(provider_source.or(segment.get("source")))?
            .trim()
            .to_lowercase();
        let upstream_provider = python_string(metadata.get("upstream_provider"))?
            .trim()
            .to_lowercase();
        let reported = reported_cost(usage)?;
        let (search_cost, tool_cost) = firecrawl_cost(metadata)?;

        if kind == "web_search"
            && let Some(tool_cost) = tool_cost
        {
            tool_breakdown.push(tool_cost);
            total = total.add(ExactDecimal::from_ratio(i128::from(search_cost), 0))?;
            segment_breakdown.push(json!({
                "segment_index": segment_index,
                "kind": kind,
                "model": model,
                "provider": if provider.is_empty() { "firecrawl" } else { &provider },
                "pricing_basis": "firecrawl_standard",
                "cost_complete": true,
                "usd_micros_exact": search_cost.to_string(),
            }));
            continue;
        }

        let has_audio_pricing =
            matches!(model.as_str(), "whisper-large-v3" | "groq/whisper-large-v3");
        if kind == "transcribe"
            && has_audio_pricing
            && !(provider == "openrouter" && reported.is_some_and(ExactDecimal::is_positive))
        {
            let usd_micros = transcription_cost(audio_seconds)?;
            total = total.add(ExactDecimal::from_ratio(i128::from(usd_micros), 0))?;
            model_breakdown.push(json!({
                "kind": kind,
                "model": if model.is_empty() { "whisper-large-v3" } else { &model },
                "usd_micros": usd_micros,
                "audio_seconds": audio_seconds,
            }));
            segment_breakdown.push(json!({
                "segment_index": segment_index,
                "kind": kind,
                "model": if model.is_empty() { "whisper-large-v3" } else { &model },
                "provider": if provider.is_empty() { "groq" } else { &provider },
                "pricing_basis": "published_rate",
                "cost_complete": audio_seconds > 0.0,
                "usd_micros_exact": usd_micros.to_string(),
            }));
            if audio_seconds <= 0.0 {
                unsupported_notes.push(format!(
                    "missing_usage_or_cost:segment={segment_index}:provider={}:model={model}",
                    if provider.is_empty() {
                        "groq"
                    } else {
                        &provider
                    }
                ));
            }
            continue;
        }

        let model_cost = model_cost(&model, usage, &provider)?;
        total = total.add(model_cost.exact)?;
        let mut model_item = model_cost.tokens.json_fields();
        model_item.insert("model".to_owned(), json!(model));
        model_item.insert("usd_micros".to_owned(), json!(model_cost.usd_micros));
        model_item.insert("kind".to_owned(), json!(kind));
        model_breakdown.push(Value::Object(model_item));

        total = total.add(ExactDecimal::from_ratio(i128::from(search_cost), 0))?;
        if let Some(tool_cost) = tool_cost {
            tool_breakdown.push(tool_cost);
        }
        let complete = reported.is_some_and(ExactDecimal::is_positive)
            || (model_cost.tokens.has_tokens() && model_cost.pricing_basis == "published_rate");
        if !complete {
            unsupported_notes.push(format!(
                "missing_usage_or_cost:segment={segment_index}:provider={}:model={}",
                if provider.is_empty() {
                    "unknown"
                } else {
                    &provider
                },
                if model.is_empty() { "unknown" } else { &model }
            ));
        }
        segment_breakdown.push(json!({
            "segment_index": segment_index,
            "kind": kind,
            "model": model,
            "provider": if provider.is_empty() { "unknown" } else { &provider },
            "upstream_provider": if upstream_provider.is_empty() { Value::Null } else { json!(upstream_provider) },
            "provider_request_id": metadata.get("provider_request_id").cloned().unwrap_or(Value::Null),
            "provider_generation_id": metadata.get("provider_generation_id").cloned().unwrap_or(Value::Null),
            "pricing_basis": model_cost.pricing_basis,
            "tool_pricing_basis": if search_cost > 0 { json!("firecrawl_standard") } else { Value::Null },
            "cost_complete": complete,
            "usd_micros_exact": model_cost.exact.fixed_string()?,
        }));
    }

    let charged_credit_units = total.ceil_credit_units()?;
    Ok(json!({
        "pricing_version": PRICING_VERSION,
        "markup_multiplier": 2.0,
        "raw_usd_micros": total.floor_i64()?,
        "raw_usd_micros_exact": total.fixed_string()?,
        "charged_credit_units": charged_credit_units,
        "charged_credits_display": format_credit_units(charged_credit_units),
        "model_breakdown": model_breakdown,
        "tool_breakdown": tool_breakdown,
        "segment_breakdown": segment_breakdown,
        "pricing_complete": !segment_breakdown.is_empty() && unsupported_notes.is_empty(),
        "unsupported_notes": unsupported_notes,
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{AiPricingError, ExactDecimal, calculate_billing_for_segments};

    #[test]
    fn prices_external_provider_payload_fixtures() -> Result<(), Box<dyn std::error::Error>> {
        let contract: Value =
            serde_json::from_str(include_str!("../tests/fixtures/ai_pricing.json"))?;
        let cases = contract["cases"]
            .as_array()
            .ok_or_else(|| std::io::Error::other("pricing contract cases must be an array"))?;
        for case in cases {
            let output = calculate_billing_for_segments(&case["segments"])?;
            let expected = &case["expected"];
            for key in [
                "raw_usd_micros",
                "raw_usd_micros_exact",
                "charged_credit_units",
                "pricing_complete",
            ] {
                assert_eq!(output[key], expected[key], "{}: {key}", case["name"]);
            }
            if let Some(expected_basis) = expected.get("pricing_basis") {
                assert_eq!(
                    output["segment_breakdown"][0]["pricing_basis"], *expected_basis,
                    "{}: pricing_basis",
                    case["name"]
                );
            }
            if let Some(expected_model_cost) = expected.get("model_usd_micros") {
                assert_eq!(
                    output["model_breakdown"][0]["usd_micros"], *expected_model_cost,
                    "{}: model_usd_micros",
                    case["name"]
                );
            }
            if let Some(expected_tool_cost) = expected.get("tool_usd_micros") {
                assert_eq!(
                    output["tool_breakdown"][0]["usd_micros"], *expected_tool_cost,
                    "{}: tool_usd_micros",
                    case["name"]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn decimal_arithmetic_preserves_provider_scale_and_rounds_only_at_the_end()
    -> Result<(), AiPricingError> {
        let first = ExactDecimal::parse(&json!("0.00000003"))
            .ok_or(AiPricingError::InvalidValue)?
            .multiply_integer(1_000_000)?;
        let total = first.add(first)?;
        assert_eq!(first.fixed_string()?, "0.03000000");
        assert_eq!(total.fixed_string()?, "0.06000000");
        assert_eq!(total.floor_i64()?, 0);
        assert_eq!(total.ceil_credit_units()?, 1);
        assert_eq!(
            ExactDecimal::parse(&json!("1e-7"))
                .ok_or(AiPricingError::InvalidValue)?
                .multiply_integer(1_000_000)?
                .fixed_string()?,
            "0.1000000"
        );
        assert_eq!(
            ExactDecimal::parse(&json!("1e2"))
                .ok_or(AiPricingError::InvalidValue)?
                .fixed_string()?,
            "100"
        );
        assert_eq!(
            ExactDecimal::parse(&json!("-1.2"))
                .ok_or(AiPricingError::InvalidValue)?
                .floor_i64()?,
            -2
        );
        assert_eq!(ExactDecimal::parse(&json!("invalid")), None);
        Ok(())
    }

    #[test]
    fn rejects_structurally_invalid_pricing_inputs() {
        assert_eq!(
            calculate_billing_for_segments(&json!({})),
            Err(AiPricingError::InvalidSegments)
        );
        assert_eq!(
            calculate_billing_for_segments(&json!(["not an object"])),
            Err(AiPricingError::InvalidSegments)
        );
        assert_eq!(
            calculate_billing_for_segments(&json!([{"kind": []}])),
            Err(AiPricingError::InvalidValue)
        );
    }

    #[test]
    fn prices_provider_published_cache_tool_and_transcription_segments()
    -> Result<(), AiPricingError> {
        let output = calculate_billing_for_segments(&json!([
            {
                "kind": "chat",
                "model": "unknown/model",
                "usage": {"cost": "0.00000003"},
                "metadata": {"provider": "openrouter"}
            },
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite",
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {
                        "cached_tokens": 200,
                        "audio_tokens": 300,
                        "cache_write_tokens": 100
                    }
                }
            },
            {
                "kind": "web_search",
                "source": "firecrawl",
                "metadata": {"web_search_requests": 1, "firecrawl_credits_used": 2}
            },
            {
                "kind": "transcribe",
                "model": "groq/whisper-large-v3",
                "audio_seconds": 1
            },
            {"kind": "summary", "source": "cache"}
        ]))?;
        assert_eq!(output["raw_usd_micros_exact"], "2382.36330000");
        assert_eq!(output["charged_credit_units"], 48);
        assert_eq!(output["pricing_complete"], true);
        assert_eq!(output["model_breakdown"][0]["usd_micros"], 0);
        assert_eq!(output["model_breakdown"][1]["usd_micros"], 413);
        assert_eq!(output["tool_breakdown"][0]["usd_micros"], 1660);
        Ok(())
    }
}
