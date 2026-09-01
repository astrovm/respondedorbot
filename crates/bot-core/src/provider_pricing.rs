//! Provider model names and published prices used by billing and reservations.

pub const PRICING_VERSION: &str = "2026-09-01";
pub const CREDIT_UNIT_USD_MICROS: i128 = 50;

pub const DEEPSEEK_MODEL: &str = "deepseek/deepseek-v4-flash-0731";
pub const GEMINI_FLASH_LITE_MODEL: &str = "google/gemini-3.1-flash-lite";
pub const GROQ_CHAT_MODEL: &str = "openai/gpt-oss-120b";
pub const GROQ_TRANSCRIPTION_MODEL: &str = "whisper-large-v3";

pub const FIRECRAWL_SEARCH_MAX_CREDITS: i128 = 2;
pub const FIRECRAWL_STANDARD_USD_MICROS_PER_CREDIT: i128 = 830;
pub const GROQ_TRANSCRIPTION_MIN_SECONDS: f64 = 10.0;
pub const GROQ_TRANSCRIPTION_USD_MICROS_PER_HOUR: f64 = 111_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPricing {
    pub input_per_million: i128,
    pub cached_input_per_million: Option<i128>,
    pub cache_write_per_million: Option<i128>,
    pub audio_input_per_million: Option<i128>,
    pub output_per_million: i128,
}

const DEEPSEEK_PRICING: TokenPricing = TokenPricing {
    input_per_million: 50_000,
    cached_input_per_million: Some(13_000),
    cache_write_per_million: None,
    audio_input_per_million: None,
    output_per_million: 160_000,
};

const GEMINI_FLASH_LITE_PRICING: TokenPricing = TokenPricing {
    input_per_million: 250_000,
    cached_input_per_million: Some(25_000),
    cache_write_per_million: Some(83_333),
    audio_input_per_million: Some(500_000),
    output_per_million: 1_500_000,
};

const GROQ_CHAT_PRICING: TokenPricing = TokenPricing {
    input_per_million: 150_000,
    cached_input_per_million: Some(75_000),
    cache_write_per_million: None,
    audio_input_per_million: None,
    output_per_million: 600_000,
};

fn base_model(model: &str) -> &str {
    model.split(':').next().unwrap_or(model)
}

#[must_use]
pub fn published_token_pricing(provider: &str, model: &str) -> Option<TokenPricing> {
    let model = base_model(model);
    if provider == "groq" && model == GROQ_CHAT_MODEL {
        return Some(GROQ_CHAT_PRICING);
    }
    match model {
        DEEPSEEK_MODEL => Some(DEEPSEEK_PRICING),
        GEMINI_FLASH_LITE_MODEL => Some(GEMINI_FLASH_LITE_PRICING),
        _ => None,
    }
}

#[must_use]
pub fn reservation_token_pricing(model: &str) -> Option<TokenPricing> {
    match base_model(model) {
        DEEPSEEK_MODEL => Some(DEEPSEEK_PRICING),
        GEMINI_FLASH_LITE_MODEL => Some(GEMINI_FLASH_LITE_PRICING),
        _ => None,
    }
}

/// Return the maximum OpenRouter prompt and completion prices in USD per
/// million tokens accepted by this deployment.
#[must_use]
pub fn openrouter_price_ceiling(model: &str) -> Option<(f64, f64)> {
    let pricing = reservation_token_pricing(model)?;
    Some((
        pricing.input_per_million as f64 / 1_000_000.0,
        pricing.output_per_million as f64 / 1_000_000.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        DEEPSEEK_MODEL, GEMINI_FLASH_LITE_MODEL, openrouter_price_ceiling,
        reservation_token_pricing,
    };

    #[test]
    fn known_models_have_one_reservation_price_and_openrouter_ceiling() {
        for model in [DEEPSEEK_MODEL, GEMINI_FLASH_LITE_MODEL] {
            assert!(reservation_token_pricing(model).is_some());
            assert!(openrouter_price_ceiling(model).is_some());
        }
        assert!(reservation_token_pricing("unknown/model").is_none());
        assert!(openrouter_price_ceiling("unknown/model").is_none());
    }
}
