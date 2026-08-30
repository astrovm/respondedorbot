//! Temporary Python bridge for incrementally adopting `bot-core`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use bot_core::base_conversion::{BaseConversion, convert_base as convert_base_core};
use bot_core::command_parsing::parse_command as parse_command_core;
use bot_core::credit_units::{
    CreditUnits, format_credit_units as format_credit_units_core,
    parse_credit_units as parse_credit_units_core,
    rescale_credit_units as rescale_credit_units_core,
    whole_credits_to_units as whole_credits_to_units_core,
};
use bot_core::market_context::{
    CryptoQuote as MarketCryptoQuote, DollarQuote as MarketDollarQuote, MarketSnapshot,
    format_market_context as format_market_context_core,
};
use bot_core::price_queries::{
    AmountConversion, PriceQuery, ProviderScope, parse_price_query as parse_price_query_core,
};
use bot_core::routing::{
    MediaRoutingInput, ResponseRoutingEvaluation, ResponseRoutingInput,
    evaluate_response_routing as evaluate_response_routing_core,
    should_auto_process_media as should_auto_process_media_core,
};
use bot_core::task_triggers::{
    IntegerInput, TaskTrigger, TaskTriggerInput, TriggerConfigInput, TriggerError,
    parse_task_trigger as parse_task_trigger_core,
};

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum IntegerInputDto {
    Missing,
    Invalid,
    BelowRange,
    AboveRange,
    Value { value: i64 },
}

impl From<IntegerInputDto> for IntegerInput {
    fn from(value: IntegerInputDto) -> Self {
        match value {
            IntegerInputDto::Missing => Self::Missing,
            IntegerInputDto::Invalid => Self::Invalid,
            IntegerInputDto::BelowRange => Self::BelowRange,
            IntegerInputDto::AboveRange => Self::AboveRange,
            IntegerInputDto::Value { value } => Self::Value(value),
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TriggerConfigInputDto {
    Missing,
    Unsupported,
    Cron {
        hour: IntegerInputDto,
        minute: IntegerInputDto,
        weekdays: Option<String>,
        day: IntegerInputDto,
    },
    IntervalDays {
        days: IntegerInputDto,
    },
}

impl From<TriggerConfigInputDto> for TriggerConfigInput {
    fn from(value: TriggerConfigInputDto) -> Self {
        match value {
            TriggerConfigInputDto::Missing => Self::Missing,
            TriggerConfigInputDto::Unsupported => Self::Unsupported,
            TriggerConfigInputDto::Cron {
                hour,
                minute,
                weekdays,
                day,
            } => Self::Cron {
                hour: hour.into(),
                minute: minute.into(),
                weekdays,
                day: day.into(),
            },
            TriggerConfigInputDto::IntervalDays { days } => {
                Self::IntervalDays { days: days.into() }
            }
        }
    }
}

#[derive(Deserialize)]
struct TaskTriggerInputDto {
    delay_seconds: IntegerInputDto,
    interval_seconds: IntegerInputDto,
    config: TriggerConfigInputDto,
}

impl From<TaskTriggerInputDto> for TaskTriggerInput {
    fn from(value: TaskTriggerInputDto) -> Self {
        Self {
            delay_seconds: value.delay_seconds.into(),
            interval_seconds: value.interval_seconds.into(),
            config: value.config.into(),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TaskTriggerDto {
    Delay {
        seconds: i64,
    },
    IntervalSeconds {
        seconds: i64,
    },
    IntervalDays {
        days: i64,
    },
    Cron {
        hour: i64,
        minute: i64,
        weekdays: Vec<String>,
        day: Option<i64>,
    },
}

impl From<TaskTrigger> for TaskTriggerDto {
    fn from(value: TaskTrigger) -> Self {
        match value {
            TaskTrigger::Delay { seconds } => Self::Delay { seconds },
            TaskTrigger::IntervalSeconds { seconds } => Self::IntervalSeconds { seconds },
            TaskTrigger::IntervalDays { days } => Self::IntervalDays { days },
            TaskTrigger::Cron {
                hour,
                minute,
                weekdays,
                day,
            } => Self::Cron {
                hour,
                minute,
                weekdays,
                day,
            },
        }
    }
}

#[derive(Serialize)]
struct TriggerErrorDto {
    code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<String>,
}

impl From<TriggerError> for TriggerErrorDto {
    fn from(error: TriggerError) -> Self {
        let (code, value) = match error {
            TriggerError::Required => ("required", None),
            TriggerError::UnsupportedType => ("type", None),
            TriggerError::DelayPositive => ("delay_positive", None),
            TriggerError::DelayMaximum => ("delay_max", None),
            TriggerError::IntervalMinimum => ("interval_min", None),
            TriggerError::IntervalMaximum => ("interval_max", None),
            TriggerError::DaysRequired => ("days_required", None),
            TriggerError::DaysPositive => ("days_positive", None),
            TriggerError::DaysMaximum => ("days_max", None),
            TriggerError::HourRequired => ("hour_required", None),
            TriggerError::HourRange => ("hour_range", None),
            TriggerError::MinuteRequired => ("minute_required", None),
            TriggerError::MinuteRange => ("minute_range", None),
            TriggerError::Weekday { value } => ("weekday", Some(value)),
            TriggerError::WeekdayEmpty => ("weekday_empty", None),
            TriggerError::DayRange => ("day_range", None),
        };
        Self { code, value }
    }
}

#[derive(Serialize)]
struct TaskTriggerResultDto {
    trigger: Option<TaskTriggerDto>,
    error: Option<TriggerErrorDto>,
}

#[derive(Deserialize)]
struct ResponseRoutingInputDto {
    known_command: bool,
    command_starts_with_slash: bool,
    message_text: String,
    is_private: bool,
    is_mention: bool,
    is_reply: bool,
    reply_text: String,
    ignore_link_fix_followups: bool,
    is_non_ai_command_followup: bool,
    ai_command_followups: bool,
    random_replies_enabled: bool,
    trigger_words: Option<Vec<String>>,
    random_sample: Option<f64>,
}

impl From<ResponseRoutingInputDto> for ResponseRoutingInput {
    fn from(value: ResponseRoutingInputDto) -> Self {
        Self {
            known_command: value.known_command,
            command_starts_with_slash: value.command_starts_with_slash,
            message_text: value.message_text,
            is_private: value.is_private,
            is_mention: value.is_mention,
            is_reply: value.is_reply,
            reply_text: value.reply_text,
            ignore_link_fix_followups: value.ignore_link_fix_followups,
            is_non_ai_command_followup: value.is_non_ai_command_followup,
            ai_command_followups: value.ai_command_followups,
            random_replies_enabled: value.random_replies_enabled,
            trigger_words: value.trigger_words,
            random_sample: value.random_sample,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum ProviderScopeDto {
    Crypto,
    Stock,
}

impl From<ProviderScope> for ProviderScopeDto {
    fn from(value: ProviderScope) -> Self {
        match value {
            ProviderScope::Crypto => Self::Crypto,
            ProviderScope::Stock => Self::Stock,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PriceQueryDto {
    UnsupportedTimeframe {
        timeframe: String,
    },
    AmountConversion {
        amount: f64,
        source_symbol: String,
        target_symbol: String,
        target_parameter: String,
    },
    Assets {
        query: String,
        timeframe: Option<String>,
        target_symbol: String,
        target_parameter: String,
        conversion_requested: bool,
        provider_scope: Option<ProviderScopeDto>,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BaseConversionDto {
    Success {
        number: String,
        source: u32,
        result: String,
        target: u32,
    },
    Usage,
    AlphanumericRequired,
    SourceRange {
        input: String,
    },
    TargetRange {
        input: String,
    },
    NumbersRequired,
}

impl From<BaseConversion> for BaseConversionDto {
    fn from(value: BaseConversion) -> Self {
        match value {
            BaseConversion::Success {
                number,
                source,
                result,
                target,
            } => Self::Success {
                number,
                source,
                result,
                target,
            },
            BaseConversion::Usage => Self::Usage,
            BaseConversion::AlphanumericRequired => Self::AlphanumericRequired,
            BaseConversion::SourceRange { input } => Self::SourceRange { input },
            BaseConversion::TargetRange { input } => Self::TargetRange { input },
            BaseConversion::NumbersRequired => Self::NumbersRequired,
        }
    }
}

impl From<PriceQuery> for PriceQueryDto {
    fn from(value: PriceQuery) -> Self {
        match value {
            PriceQuery::UnsupportedTimeframe { timeframe } => {
                Self::UnsupportedTimeframe { timeframe }
            }
            PriceQuery::AmountConversion(AmountConversion {
                amount,
                source_symbol,
                target_symbol,
                target_parameter,
            }) => Self::AmountConversion {
                amount,
                source_symbol,
                target_symbol,
                target_parameter,
            },
            PriceQuery::Assets {
                query,
                timeframe,
                target_symbol,
                target_parameter,
                conversion_requested,
                provider_scope,
            } => Self::Assets {
                query,
                timeframe,
                target_symbol,
                target_parameter,
                conversion_requested,
                provider_scope: provider_scope.map(Into::into),
            },
        }
    }
}

fn dynamic_number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(number) => number.as_f64(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::String(value) if !value.trim().is_empty() => value.trim().parse().ok(),
        Value::Null | Value::String(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

fn dynamic_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn dynamic_text(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn first_truthy<'a>(row: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .filter_map(|key| row.get(*key))
        .find(|value| dynamic_truthy(value))
}

fn nested_object<'a>(row: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Map<String, Value>> {
    let mut current = row;
    for key in keys {
        current = current.get(*key)?.as_object()?;
    }
    Some(current)
}

fn normalize_market_crypto(value: Option<&Value>) -> Vec<MarketCryptoQuote> {
    let Some(rows) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    rows.iter()
        .take(3)
        .filter_map(|value| {
            let row = value.as_object()?;
            let symbol = first_truthy(row, &["symbol", "name"])
                .map(dynamic_text)
                .unwrap_or_default()
                .trim()
                .to_uppercase();
            let usd = nested_object(row, &["quote", "USD"]);
            let has_usd_quote = usd.is_some_and(|value| !value.is_empty());
            let price = if has_usd_quote {
                dynamic_number(usd.and_then(|value| value.get("price")))
            } else {
                dynamic_number(row.get("price"))
            }?;
            if symbol.is_empty() {
                return None;
            }
            let change_24h = if has_usd_quote {
                dynamic_number(
                    usd.and_then(|value| value.get("changes"))
                        .and_then(Value::as_object)
                        .and_then(|changes| changes.get("24h")),
                )
            } else {
                dynamic_number(row.get("change_24h"))
            };
            let dominance = has_usd_quote
                .then(|| usd.and_then(|value| dynamic_number(value.get("dominance"))))
                .flatten();
            Some(MarketCryptoQuote {
                symbol,
                price,
                change_24h,
                dominance,
            })
        })
        .collect()
}

fn market_dollar_quote(
    label: &str,
    row: Option<&Map<String, Value>>,
    price_keys: &[&str],
) -> Option<MarketDollarQuote> {
    let row = row?;
    let price = price_keys
        .iter()
        .find_map(|key| dynamic_number(row.get(*key)))?;
    Some(MarketDollarQuote {
        label: label.to_owned(),
        price,
        bid: dynamic_number(row.get("bid")),
    })
}

fn normalize_market_dollars(value: Option<&Value>) -> Vec<MarketDollarQuote> {
    match value {
        Some(Value::Array(rows)) => rows
            .iter()
            .filter_map(|value| {
                let row = value.as_object()?;
                let label = first_truthy(row, &["name", "label"])
                    .map(dynamic_text)
                    .unwrap_or_default()
                    .trim()
                    .to_lowercase();
                let price = dynamic_number(row.get("price"))?;
                (!label.is_empty()).then_some(MarketDollarQuote {
                    label,
                    price,
                    bid: None,
                })
            })
            .collect(),
        Some(Value::Object(row)) => {
            let mep = nested_object(row, &["mep", "al30", "ci"]);
            let crypto = nested_object(row, &["cripto", "usdt"]);
            [
                market_dollar_quote(
                    "oficial",
                    row.get("oficial").and_then(Value::as_object),
                    &["price"],
                ),
                market_dollar_quote(
                    "blue",
                    row.get("blue").and_then(Value::as_object),
                    &["ask", "price"],
                ),
                market_dollar_quote("mep al30 ci", mep, &["price"]),
                market_dollar_quote(
                    "tarjeta",
                    row.get("tarjeta").and_then(Value::as_object),
                    &["price"],
                ),
                market_dollar_quote("usdt", crypto, &["ask"]),
            ]
            .into_iter()
            .flatten()
            .collect()
        }
        Some(Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) | None => {
            Vec::new()
        }
    }
}

fn normalize_market_snapshot(value: &Value) -> Option<MarketSnapshot> {
    let market = value.as_object()?;
    Some(MarketSnapshot {
        crypto: normalize_market_crypto(market.get("crypto")),
        dollars: normalize_market_dollars(market.get("dollar")),
    })
}

/// Return the compatibility protocol version shared with Python.
#[pyfunction]
fn migration_protocol_version() -> u16 {
    bot_core::migration_protocol_version()
}

/// Convert whole credits to stored hundredth-credit units.
#[pyfunction]
fn whole_credits_to_units(credits: i64) -> PyResult<i64> {
    whole_credits_to_units_core(credits)
        .map(CreditUnits::value)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Rescale legacy units to stored hundredth-credit units.
#[pyfunction]
fn rescale_credit_units(units: i64, source_scale: i64) -> PyResult<i64> {
    rescale_credit_units_core(units, Some(source_scale))
        .map(CreditUnits::value)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Parse a human credit amount into stored hundredth-credit units.
#[pyfunction]
fn parse_credit_units(value: &str) -> Option<i64> {
    parse_credit_units_core(value).map(CreditUnits::value)
}

/// Format stored hundredth-credit units with two decimal places.
#[pyfunction]
fn format_credit_units(units: i64) -> String {
    format_credit_units_core(CreditUnits::new(units))
}

/// Normalize one Telegram command token and its remaining text.
#[pyfunction]
fn parse_command(message_text: &str, bot_name: &str) -> (String, String) {
    let parsed = parse_command_core(message_text, bot_name);
    (parsed.command, parsed.message_text)
}

/// Convert an arbitrary-precision number between bases from 2 through 36.
#[pyfunction]
fn convert_base(message_text: &str) -> PyResult<String> {
    let result = convert_base_core(message_text)
        .map_err(|_| PyValueError::new_err("Unicode input requires the legacy converter"))?;
    serde_json::to_string(&BaseConversionDto::from(result))
        .map_err(|error| PyValueError::new_err(format!("cannot encode base conversion: {error}")))
}

/// Validate and normalize a task-trigger input encoded by the Python adapter.
#[pyfunction]
fn parse_task_trigger(input_json: &str) -> PyResult<String> {
    let input: TaskTriggerInputDto = serde_json::from_str(input_json)
        .map_err(|error| PyValueError::new_err(format!("invalid task trigger input: {error}")))?;
    let result = match parse_task_trigger_core(input.into()) {
        Ok(trigger) => TaskTriggerResultDto {
            trigger: Some(trigger.into()),
            error: None,
        },
        Err(error) => TaskTriggerResultDto {
            trigger: None,
            error: Some(error.into()),
        },
    };
    serde_json::to_string(&result).map_err(|error| {
        PyValueError::new_err(format!("cannot encode task trigger result: {error}"))
    })
}

/// Parse a unified market-price query into a typed request.
#[pyfunction]
fn parse_price_query(message_text: &str, valid_timeframes_json: &str) -> PyResult<String> {
    let valid_timeframes: Vec<String> = serde_json::from_str(valid_timeframes_json)
        .map_err(|error| PyValueError::new_err(format!("invalid timeframes: {error}")))?;
    serde_json::to_string(&PriceQueryDto::from(parse_price_query_core(
        message_text,
        &valid_timeframes,
    )))
    .map_err(|error| PyValueError::new_err(format!("cannot encode price query: {error}")))
}

/// Normalize cached market data and format the compact AI prompt context.
#[pyfunction]
fn format_market_info(market_json: &str) -> PyResult<String> {
    let value: Value = serde_json::from_str(market_json)
        .map_err(|error| PyValueError::new_err(format!("invalid market snapshot: {error}")))?;
    let snapshot = normalize_market_snapshot(&value)
        .ok_or_else(|| PyValueError::new_err("market snapshot must be an object"))?;
    Ok(format_market_context_core(&snapshot))
}

/// Decide whether one normalized message should auto-process attached media.
#[pyfunction]
fn should_auto_process_media(
    chat_type: &str,
    known_command: bool,
    message_text: &str,
    bot_username: Option<&str>,
    reply_username: Option<&str>,
) -> bool {
    should_auto_process_media_core(&MediaRoutingInput {
        chat_type: chat_type.to_owned(),
        known_command,
        message_text: message_text.to_owned(),
        bot_username: bot_username.map(str::to_owned),
        reply_username: reply_username.map(str::to_owned),
    })
}

/// Evaluate general response routing and request missing external inputs explicitly.
#[pyfunction]
fn evaluate_response_routing(input_json: &str) -> PyResult<&'static str> {
    let input: ResponseRoutingInputDto = serde_json::from_str(input_json)
        .map_err(|error| PyValueError::new_err(format!("invalid routing input: {error}")))?;
    Ok(match evaluate_response_routing_core(&input.into()) {
        ResponseRoutingEvaluation::Ignore => "ignore",
        ResponseRoutingEvaluation::Respond => "respond",
        ResponseRoutingEvaluation::NeedsTriggerWords => "needs_trigger_words",
        ResponseRoutingEvaluation::NeedsRandomSample => "needs_random_sample",
    })
}

/// Register the temporary `respondedorbot_rs` Python module.
#[pymodule]
fn respondedorbot_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(migration_protocol_version, module)?)?;
    module.add_function(wrap_pyfunction!(whole_credits_to_units, module)?)?;
    module.add_function(wrap_pyfunction!(rescale_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(parse_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(format_credit_units, module)?)?;
    module.add_function(wrap_pyfunction!(parse_command, module)?)?;
    module.add_function(wrap_pyfunction!(convert_base, module)?)?;
    module.add_function(wrap_pyfunction!(parse_task_trigger, module)?)?;
    module.add_function(wrap_pyfunction!(parse_price_query, module)?)?;
    module.add_function(wrap_pyfunction!(format_market_info, module)?)?;
    module.add_function(wrap_pyfunction!(should_auto_process_media, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_response_routing, module)?)?;
    Ok(())
}
