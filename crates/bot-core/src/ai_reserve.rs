//! Conservative AI reservation estimates used before provider I/O starts.

use thiserror::Error;

pub const CHAT_OUTPUT_TOKEN_LIMIT: i64 = 1_024;
pub const REASONING_CHAT_OUTPUT_TOKEN_LIMIT: i64 = 8_192;
pub const VISION_OUTPUT_TOKEN_LIMIT: i64 = 512;
const CREDIT_UNIT_USD_MICROS: i128 = 50;
const FIRECRAWL_SEARCH_MAX_CREDITS: i128 = 2;
const FIRECRAWL_USD_MICROS_PER_CREDIT: i128 = 830;
const DEEPSEEK_MODEL: &str = "deepseek/deepseek-v4-flash-0731";
const GEMINI_MODEL: &str = "google/gemini-3.1-flash-lite-preview";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenEstimateValue {
    Empty,
    Text(String),
    Mapping(Vec<TokenEstimateValue>),
    Sequence(Vec<TokenEstimateValue>),
    Scalar(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstimatedMessage {
    pub role: TokenEstimateValue,
    pub content: TokenEstimateValue,
    pub name: TokenEstimateValue,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ReserveEstimateError {
    #[error("AI reserve estimate exceeds the supported integer range")]
    Overflow,
    #[error("AI reserve estimate requires finite audio duration")]
    NonFiniteAudioDuration,
    #[error("AI model does not define token pricing")]
    MissingTokenPricing,
}

#[derive(Debug, Clone, Copy)]
struct TokenPricing {
    input_per_million: i128,
    output_per_million: i128,
}

#[must_use]
pub fn chat_output_token_limit(model: &str) -> i64 {
    if model.split(':').next() == Some(DEEPSEEK_MODEL) {
        REASONING_CHAT_OUTPUT_TOKEN_LIMIT
    } else {
        CHAT_OUTPUT_TOKEN_LIMIT
    }
}

#[must_use]
pub fn estimate_text_tokens(text: Option<&str>) -> i64 {
    let Some(text) = text.filter(|value| !value.is_empty()) else {
        return 0;
    };
    let characters = text.chars().count();
    i64::try_from(characters.div_ceil(4)).unwrap_or(i64::MAX)
}

#[must_use]
pub fn estimate_nested_tokens(value: &TokenEstimateValue) -> i64 {
    match value {
        TokenEstimateValue::Empty => 0,
        TokenEstimateValue::Text(value) | TokenEstimateValue::Scalar(value) => {
            estimate_text_tokens(Some(value))
        }
        TokenEstimateValue::Mapping(values) | TokenEstimateValue::Sequence(values) => {
            values.iter().fold(0_i64, |total, value| {
                total.saturating_add(estimate_nested_tokens(value))
            })
        }
    }
}

#[must_use]
pub fn estimate_message_tokens(messages: &[EstimatedMessage]) -> i64 {
    messages.iter().fold(0_i64, |total, message| {
        total
            .saturating_add(estimate_nested_tokens(&message.role))
            .saturating_add(estimate_nested_tokens(&message.content))
            .saturating_add(estimate_nested_tokens(&message.name))
    })
}

pub fn estimate_chat_reserve_credit_units(
    system_message: Option<&EstimatedMessage>,
    messages: &[EstimatedMessage],
    max_output_tokens: Option<i64>,
    extra_input_tokens: i64,
    model: &str,
) -> Result<i64, ReserveEstimateError> {
    let pricing = chat_pricing(model)?;
    let mut input_tokens = i128::from(estimate_message_tokens(messages))
        .checked_add(i128::from(extra_input_tokens))
        .ok_or(ReserveEstimateError::Overflow)?;
    if let Some(system_message) = system_message {
        input_tokens = input_tokens
            .checked_add(i128::from(estimate_message_tokens(std::slice::from_ref(
                system_message,
            ))))
            .ok_or(ReserveEstimateError::Overflow)?;
    }
    let output_tokens =
        i128::from(max_output_tokens.unwrap_or_else(|| chat_output_token_limit(model)));
    let usd_micros = input_tokens
        .checked_mul(pricing.input_per_million)
        .and_then(|input| {
            output_tokens
                .checked_mul(pricing.output_per_million)
                .and_then(|output| input.checked_add(output))
        })
        .ok_or(ReserveEstimateError::Overflow)?
        / 1_000_000;
    credit_units_from_usd_micros(usd_micros)
}

pub fn estimate_vision_reserve_credit_units(
    prompt_text: &str,
    image_byte_length: usize,
    extra_input_tokens: i64,
    max_output_tokens: i64,
    model: &str,
) -> Result<i64, ReserveEstimateError> {
    let pricing = vision_pricing(model)?;
    let encoded_length = image_byte_length
        .checked_add(2)
        .and_then(|value| value.checked_div(3))
        .and_then(|value| value.checked_mul(4))
        .ok_or(ReserveEstimateError::Overflow)?;
    let image_url_characters = if image_byte_length == 0 {
        0
    } else {
        23_usize
            .checked_add(encoded_length)
            .ok_or(ReserveEstimateError::Overflow)?
    };
    let structural_tokens = estimate_text_tokens(Some("user"))
        + estimate_text_tokens(Some("input_text"))
        + estimate_text_tokens(Some("input_image"));
    let input_tokens = i128::from(structural_tokens)
        .checked_add(i128::from(estimate_text_tokens(Some(prompt_text))))
        .and_then(|value| value.checked_add(i128::try_from(image_url_characters.div_ceil(4)).ok()?))
        .and_then(|value| value.checked_add(i128::from(extra_input_tokens)))
        .ok_or(ReserveEstimateError::Overflow)?;
    let usd_micros = input_tokens
        .checked_mul(pricing.input_per_million)
        .and_then(|input| {
            i128::from(max_output_tokens)
                .checked_mul(pricing.output_per_million)
                .and_then(|output| input.checked_add(output))
        })
        .ok_or(ReserveEstimateError::Overflow)?
        / 1_000_000;
    Ok(credit_units_from_usd_micros(usd_micros)?.max(1))
}

pub fn estimate_transcription_reserve_credit_units(
    audio_seconds: f64,
) -> Result<i64, ReserveEstimateError> {
    if !audio_seconds.is_finite() {
        return Err(ReserveEstimateError::NonFiniteAudioDuration);
    }
    let seconds = audio_seconds.max(0.0);
    if seconds <= 0.0 {
        return Ok(1);
    }
    let usd_micros = (seconds.max(10.0) * 111_000.0 / 3_600.0).ceil();
    if usd_micros > i128::MAX as f64 {
        return Err(ReserveEstimateError::Overflow);
    }
    Ok(credit_units_from_usd_micros(usd_micros as i128)?.max(1))
}

pub fn estimate_firecrawl_reserve_credit_units() -> Result<i64, ReserveEstimateError> {
    Ok(credit_units_from_usd_micros(
        FIRECRAWL_SEARCH_MAX_CREDITS * FIRECRAWL_USD_MICROS_PER_CREDIT,
    )?
    .max(1))
}

pub fn credit_units_from_usd_micros(usd_micros: i128) -> Result<i64, ReserveEstimateError> {
    if usd_micros <= 0 {
        return Ok(0);
    }
    let units = usd_micros
        .checked_add(CREDIT_UNIT_USD_MICROS - 1)
        .ok_or(ReserveEstimateError::Overflow)?
        / CREDIT_UNIT_USD_MICROS;
    i64::try_from(units).map_err(|_| ReserveEstimateError::Overflow)
}

fn chat_pricing(model: &str) -> Result<TokenPricing, ReserveEstimateError> {
    match model {
        GEMINI_MODEL => Ok(TokenPricing {
            input_per_million: 250_000,
            output_per_million: 1_500_000,
        }),
        "whisper-large-v3" | "groq/whisper-large-v3" => {
            Err(ReserveEstimateError::MissingTokenPricing)
        }
        _ => Ok(TokenPricing {
            input_per_million: 30_000,
            output_per_million: 100_000,
        }),
    }
}

fn vision_pricing(model: &str) -> Result<TokenPricing, ReserveEstimateError> {
    if model == DEEPSEEK_MODEL {
        Ok(TokenPricing {
            input_per_million: 30_000,
            output_per_million: 100_000,
        })
    } else if matches!(model, "whisper-large-v3" | "groq/whisper-large-v3") {
        Err(ReserveEstimateError::MissingTokenPricing)
    } else {
        Ok(TokenPricing {
            input_per_million: 250_000,
            output_per_million: 1_500_000,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EstimatedMessage, TokenEstimateValue, chat_output_token_limit,
        credit_units_from_usd_micros, estimate_chat_reserve_credit_units,
        estimate_firecrawl_reserve_credit_units, estimate_message_tokens, estimate_nested_tokens,
        estimate_text_tokens, estimate_transcription_reserve_credit_units,
        estimate_vision_reserve_credit_units,
    };

    fn text(value: &str) -> TokenEstimateValue {
        TokenEstimateValue::Text(value.to_owned())
    }

    #[test]
    fn preserves_unicode_text_nested_and_message_estimates() {
        assert_eq!(estimate_text_tokens(None), 0);
        assert_eq!(estimate_text_tokens(Some("")), 0);
        assert_eq!(estimate_text_tokens(Some("abcde")), 2);
        assert_eq!(estimate_text_tokens(Some("😀😀😀😀😀")), 2);
        assert_eq!(
            estimate_nested_tokens(&TokenEstimateValue::Mapping(vec![
                text("hello"),
                TokenEstimateValue::Sequence(vec![
                    text("world"),
                    TokenEstimateValue::Scalar("True".to_owned()),
                ]),
            ])),
            5
        );
        let messages = vec![
            EstimatedMessage {
                role: text("user"),
                content: text("hello"),
                name: text("bob"),
            },
            EstimatedMessage {
                role: text("assistant"),
                content: TokenEstimateValue::Sequence(vec![TokenEstimateValue::Mapping(vec![
                    text("text"),
                    text("world"),
                ])]),
                name: TokenEstimateValue::Empty,
            },
        ];
        assert_eq!(estimate_message_tokens(&messages), 10);
        assert_eq!(
            estimate_chat_reserve_credit_units(
                Some(&EstimatedMessage {
                    role: text("system"),
                    content: text("rules"),
                    name: TokenEstimateValue::Empty,
                }),
                &messages,
                None,
                0,
                "deepseek/deepseek-v4-flash-0731",
            ),
            Ok(17)
        );
    }

    #[test]
    fn preserves_model_limits_credit_rounding_and_provider_reserves() {
        assert_eq!(
            chat_output_token_limit("deepseek/deepseek-v4-flash-0731:free"),
            8_192
        );
        assert_eq!(chat_output_token_limit("other"), 1_024);
        for (micros, expected) in [(-1, 0), (0, 0), (1, 1), (50, 1), (51, 2), (5_000, 100)] {
            assert_eq!(credit_units_from_usd_micros(micros), Ok(expected));
        }
        assert_eq!(estimate_firecrawl_reserve_credit_units(), Ok(34));
        assert_eq!(estimate_transcription_reserve_credit_units(0.0), Ok(1));
        assert_eq!(estimate_transcription_reserve_credit_units(1.0), Ok(7));
        assert_eq!(
            estimate_transcription_reserve_credit_units(3_600.0),
            Ok(2_220)
        );
        assert_eq!(
            estimate_vision_reserve_credit_units(
                "describe",
                100,
                0,
                512,
                "google/gemini-3.1-flash-lite-preview",
            ),
            Ok(16)
        );
    }
}
