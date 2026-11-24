//! Provider model identifiers and non-catalog service pricing constants.

pub const PRICING_VERSION: &str = "2026-10-17";
/// What Telegram pays out per Star, in millionths of a US dollar.
pub const STAR_PAYOUT_USD_MICROS: i128 = 13_000;
/// What one credit unit sells for at the Stars payout: 25 Stars buy 5,000
/// units, so US$0.325 / 5,000.
pub const CREDIT_UNIT_PRICE_USD_MICROS: i128 = 65;
/// AI replies cost the provider's price plus this much, in percent.
pub const AI_MARKUP_PERCENT: i128 = 30;
/// Provider cost one credit unit pays for, so the money from a pack covers
/// the AI it buys plus [`AI_MARKUP_PERCENT`].
pub const CREDIT_UNIT_USD_MICROS: i128 =
    CREDIT_UNIT_PRICE_USD_MICROS * 100 / (100 + AI_MARKUP_PERCENT);

pub const DEEPSEEK_MODEL: &str = "deepseek/deepseek-v4.1-flash-1017";
pub const DEEPSEEK_FLASH_MODEL: &str = "deepseek/deepseek-v4-flash";
pub const OPENROUTER_TRANSCRIPTION_MODEL: &str = "microsoft/mai-transcribe-2";

/// An image model and the most one picture from it can cost the provider,
/// which is what an image holds from the payer's balance until the reply
/// settles at the cost OpenRouter reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageModel {
    pub id: &'static str,
    /// The `modalities` the model accepts. An image-only model refuses a
    /// request that also asks for text.
    pub modalities: &'static [&'static str],
    pub max_usd_micros: i128,
}

/// Seedream 5.5: US$0.035 for the 2560x1440 picture it makes by default
/// (measured on 2026-11-23), plus US$0.008 for a starting picture.
pub const FAST_IMAGE_MODEL: ImageModel = ImageModel {
    id: "bytedance-seed/seedream-5-5",
    modalities: &["image"],
    max_usd_micros: 50_000,
};

/// GPT Image 2.6 Sunburst, first on the Artificial Analysis text-to-image
/// arena in November 2026. It bills US$30 per million image tokens, and a
/// large high-quality picture is several thousand tokens.
pub const BEST_IMAGE_MODEL: ImageModel = ImageModel {
    id: "openai/gpt-image-2.6-sunburst",
    modalities: &["image"],
    max_usd_micros: 400_000,
};

pub const FIRECRAWL_SEARCH_MAX_CREDITS: i128 = 2;
pub const FIRECRAWL_STANDARD_USD_MICROS_PER_CREDIT: i128 = 830;
pub const YOUTUBE_TRANSCRIPT_USD_MICROS_PER_SUCCESS: i128 = 3_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPricing {
    pub input_per_million: i128,
    pub cached_input_per_million: Option<i128>,
    pub cache_write_per_million: Option<i128>,
    pub audio_input_per_million: Option<i128>,
    pub output_per_million: i128,
}
