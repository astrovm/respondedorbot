//! Provider model identifiers and non-catalog service pricing constants.

pub const PRICING_VERSION: &str = "2026-09-10";
pub const CREDIT_UNIT_USD_MICROS: i128 = 50;

pub const DEEPSEEK_MODEL: &str = "deepseek/deepseek-v4.1-flash";
pub const GEMINI_FLASH_LITE_MODEL: &str = "google/gemini-3.1-flash-lite";
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
