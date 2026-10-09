//! Provider model identifiers and non-catalog service pricing constants.

pub const PRICING_VERSION: &str = "2026-09-10";
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

pub const DEEPSEEK_MODEL: &str = "deepseek/deepseek-v4.1-flash";
pub const DEEPSEEK_FLASH_MODEL: &str = "deepseek/deepseek-v4-flash";
pub const OPENROUTER_TRANSCRIPTION_MODEL: &str = "microsoft/mai-transcribe-2";

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
